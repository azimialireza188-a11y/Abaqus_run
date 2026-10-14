# -*- coding: utf-8 -*-
"""Sequential family builds, G/L reference solves, imperfection and Step5 queue.

Run with normal Python, NOT ``abaqus cae``. The entry point launches Abaqus itself.
The original model is build-only; all imperfection modes come from solved REF_L/G.
Each family finishes its GMNIA queue before the next family starts. Existing run
directories are refused unless --resume verifies matching inputs and saved stages.
Optional steel plates use this repository's exact geometry/bolt layout. Fixed settings: reference-modes=20,
LG_mp=-0.34t,0,L/1000 and postpeak ratio=0.70. Member fy defaults to 240 MPa.
Compression is the default. --load-case bending --bending-axis-deg ANGLE uses
the existing bending load definition, twist-free REF_G and a lateral eigenmode;
its run folders and moment-rotation comparisons carry a separate BEND angle tag.
Step5 includes its default PERFECT baseline and the single LG_mp imperfect case.
Completed families update STEP5_family_force_displacement.svg (and PNG when
matplotlib is available): PERFECT solid, LG_mp dashed, one color per family.
"""
import argparse
import copy
import colorsys
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import abaqus_load_case as loads
from abaqus_pipeline_contract import STEP5_END_CONDITION_ID, STEP5_END_TAG


INPUT_FILES = ('builtup_member.csv', 'builtup_segments.csv',
               'builtup_seams.csv', 'builtup_bolts_y.csv')
REPO_FILES = ('abaqus_complete_model_m20.py', 'gdlc_classifier/gdlc_mode_decomposition.py',
              'abaqus_step4_imperfections.py', 'abaqus_step5_gmnia.py',
              'abaqus_step5_queue_nogui.py', 'step5_progress_tail.ps1',
              'runtime_resources.py', 'abaqus_pipeline_contract.py')
REPO_DEPENDENCIES = ('abaqus_step5_queue.py', 'abaqus_step5_force_displacement_core.py',
    'abaqus_step5_progress.py', 'abaqus_load_case.py', 'gdlc_classifier/abaqus_mfsm_export.py',
    'gdlc_classifier/gdlc_stiffness.py')
EXPECTED_JOBS = {'STEP5_PERFECT_FY240', 'STEP5_LG_mp_FY240'}
# '_REF1': supplied working reference loading/supports. Older Step5/P-M
# inputs/results retain their names and are never reused or overwritten.
STEP5_DIRECTORY = 'Step5_GMNIA_FY240_LG_'+STEP5_END_TAG
STEP4_RECEIPT = 'family_pipeline_step4.json'
STEP5_SETTINGS = dict(fy=240., initial_arc=.01, min_arc=1e-8, max_arc=.05,
                     max_increments=5000, field_frequency=1, postpeak_stop_ratio=.70)


def step5_settings(args):
    return dict(STEP5_SETTINGS,fy=getattr(args,'steel_column_plate_fy',240.))


def column_fy_tag(args):
    # Match the existing Step5 CAE/job naming convention exactly.
    return ('%g'%step5_settings(args)['fy']).replace('.','p').replace('+','')


def expected_jobs(args):
    return {'STEP5_'+case+'_FY'+column_fy_tag(args) for case in ('PERFECT','LG_mp')}


def default_step5_directory(args):
    return 'Step5_GMNIA_FY'+column_fy_tag(args)+'_LG_'+STEP5_END_TAG


def pm_directory(args):
    return 'Step5_PM_FY'+column_fy_tag(args)+'_LG_'+STEP5_END_TAG


def check_end_condition(info, name):
    """Reuse only the current reference end policy; never mix rigid-end-plate results."""
    if (info.get('end_condition') or {}).get('id') != STEP5_END_CONDITION_ID:
        raise ValueError('--resume GMNIA input %s was built with another end condition (expected %s); '
                         'existing results are preserved: use a new --step5-directory or output folder'
                         % (name, STEP5_END_CONDITION_ID))


def positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a positive integer')
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def cpu_request(value):
    return 'auto' if value.lower() == 'auto' else str(positive_int(value))


def skip_count_request(value):
    """Keep scalar callers unchanged; reject empty, negative and repeated sweep values."""
    values=str(value).split(',')
    if any(not re.fullmatch(r'\+?\d+',v.strip()) for v in values):
        raise argparse.ArgumentTypeError('use a nonnegative integer or comma-separated integers')
    counts=tuple(int(v.strip()) for v in values)
    if len(set(counts))!=len(counts):
        raise argparse.ArgumentTypeError('duplicate skip counts are not allowed')
    return counts[0] if len(counts)==1 else counts


def skip_configurations(args):
    value=args.steel_plate_skip_count
    if isinstance(value,int):return [args]
    configs=[]
    for count in value:
        config=copy.copy(args)
        config.steel_plate_skip_count=count
        config.skip_sweep=True
        configs.append(config)
    return configs


def batch_output_directory(args):
    directory=args.output_root/args.family_root.name
    return directory/('skip_%d'%args.steel_plate_skip_count) if getattr(args,'skip_sweep',False) else directory


def family_run_name(args,family):
    name=family.name+loads.load_case_tag(getattr(args,'load_case','compression'),
                                       getattr(args,'bending_axis_deg',0.))
    if getattr(args,'steel_plate_length',0.):
        name+='_PL'+('%g'%args.steel_plate_length).replace('.','p')
    if getattr(args,'steel_plate_length_2',0.):
        name+='_PL2'+('%g'%args.steel_plate_length_2).replace('.','p')
        name+='_PG'+('%g'%args.steel_plate_gap_mm).replace('.','p')
    if getattr(args,'steel_plate_skip_count',0):name+='_PS%d'%args.steel_plate_skip_count
    return name


def check_load_case(args,case,source):
    kind=getattr(args,'load_case','compression')
    if case.get('type','compression') != kind:
        raise ValueError('Load case differs from requested pipeline: '+source)
    if kind=='bending' and (case.get('neutral_axis_angle_deg') is None or
            loads.normalize_angle(case['neutral_axis_angle_deg']) != args.bending_axis_deg):
        raise ValueError('Bending axis differs from requested pipeline: '+source)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('family-root', 'output-root'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent,
        help='Code directory on THIS computer (default: directory containing this script)')
    parser.add_argument('--abaqus-cmd', default='abaqus',
        help='Abaqus launcher name in PATH or full path to its .bat/.cmd/executable')
    parser.add_argument('--mesh-mm', type=float, default=20.)
    parser.add_argument('--steel-plate-length',type=float,default=0.)
    parser.add_argument('--steel-plate-length-2',type=float,default=0.)
    parser.add_argument('--steel-plate-gap-mm',type=float,default=0.)
    parser.add_argument('--steel-plate-skip-count',type=skip_count_request,default=0,
        help='Nonnegative integer or independent sweep, e.g. 0,1,2,3')
    parser.add_argument('--steel-plate-fy',type=float,default=360.,
        help='Added connector/cross plate yield stress (MPa); default 360, independent of column steel')
    parser.add_argument('--steel-column-plate-fy',type=float,default=240.,
        help='Column wall yield stress (MPa); default 240, passed to Step5/P-M as --fy')
    parser.add_argument('--pm-interaction',action='store_true',help='Also build/solve existing P-M rays and collect diagrams per family/configuration')
    parser.add_argument('--pm-axes',default='required',help='P-M neutral axes: required, all, or comma-separated degrees')
    parser.add_argument('--pm-angles',default='0,15,30,45,60,75,90',help='P-M proportional ray angles in [0,90] degrees')
    parser.add_argument('--n-modes', type=positive_int, default=250)
    parser.add_argument('--n-vectors', type=positive_int, default=500)
    parser.add_argument('--max-iterations', type=positive_int, default=1250)
    parser.add_argument('--cpus', type=cpu_request, default='auto')
    parser.add_argument('--buckle-output', choices=('standard', 'detailed'), default='detailed')
    parser.add_argument('--nodal-precision', choices=('single', 'full'), default='full')
    parser.add_argument('--longitudinal-lines', type=int, default=4)
    parser.add_argument('--longitudinal-line-min-spacing-mm', type=float, default=5.)
    parser.add_argument('--local-mode', type=positive_int, default=1)
    parser.add_argument('--load-case', choices=('compression','bending'), default='compression')
    parser.add_argument('--bending-axis-deg', type=float, default=0.,
        help='Neutral-axis angle from global X; 0 means compression on the +Y side')
    parser.add_argument('--global-modes', default=None,
        help='REF_G modes: default 1, retaining the first eigenmode including torsion; an explicit pair must be degenerate')
    parser.add_argument('--riks-extrapolation', choices=('linear','none'), default='linear',
        help='Step5 Riks predictor (linear preserves the existing default; none disables extrapolation)')
    parser.add_argument('--step5-directory', default=None,
        help='Step5 folder name under imperfection; select a fresh name for a non-destructive numerical retry')
    parser.add_argument('--resume', action='store_true',
        help='Reuse verified completed stages/families; solve only unused job destinations; never overwrite analyses')
    parser.add_argument('--start-family',
        help='Start at this existing family folder (e.g. family_5); earlier families are left untouched')
    args = parser.parse_args(argv)
    if not math.isfinite(args.steel_column_plate_fy) or args.steel_column_plate_fy<=0:
        parser.error('--steel-column-plate-fy must be finite and positive')
    if args.step5_directory is None:args.step5_directory=default_step5_directory(args)
    for key in ('steel_plate_length','steel_plate_length_2','steel_plate_gap_mm'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<0:
            parser.error('--'+key.replace('_','-')+' must be finite and nonnegative')
    skip_values=(args.steel_plate_skip_count,) if isinstance(args.steel_plate_skip_count,int) else args.steel_plate_skip_count
    if (args.steel_plate_length_2 or any(skip_values)) and not args.steel_plate_length:
        parser.error('Second plates/skip count require positive --steel-plate-length')
    if args.steel_plate_gap_mm and not args.steel_plate_length_2:
        parser.error('--steel-plate-gap-mm requires positive --steel-plate-length-2')
    if args.steel_plate_fy is not None and (not math.isfinite(args.steel_plate_fy) or args.steel_plate_fy<=0):
        parser.error('--steel-plate-fy must be finite and positive')
    if args.steel_plate_length and args.steel_plate_fy is None:
        parser.error('Plate models require --steel-plate-fy')
    if args.pm_interaction:
        import abaqus_pm_interaction as pm
        try:
            angles=sorted(set(pm.parse_float_list(args.pm_angles,'--pm-angles')))
            if any(not 0.<=v<=90. for v in angles):raise ValueError('--pm-angles must be in [0,90]')
            args.pm_angles=','.join('%g'%v for v in angles)
            if args.pm_axes not in ('required','all'):
                axes=[loads.normalize_angle(v) for v in pm.parse_float_list(args.pm_axes,'--pm-axes')]
                if len(set(axes))!=len(axes):raise ValueError('duplicate P-M axes')
                args.pm_axes=','.join('%g'%v for v in sorted(axes))
        except ValueError as error:parser.error(str(error))
    if not math.isfinite(args.bending_axis_deg):
        parser.error('--bending-axis-deg must be finite')
    if args.load_case!='bending' and args.bending_axis_deg!=0.:
        parser.error('--bending-axis-deg requires --load-case bending')
    args.bending_axis_deg=loads.normalize_angle(args.bending_axis_deg)
    if args.global_modes is None:
        args.global_modes='1'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.step5_directory):
        parser.error('--step5-directory must be a single alphanumeric folder name (underscores/hyphens allowed)')
    if args.longitudinal_lines != 0 and not 2 <= args.longitudinal_lines <= 100:
        parser.error('--longitudinal-lines must be 0 or an integer from 2 to 100')
    if not math.isfinite(args.mesh_mm) or args.mesh_mm <= 0:
        parser.error('--mesh-mm must be positive and finite')
    if (not math.isfinite(args.longitudinal_line_min_spacing_mm) or
            args.longitudinal_line_min_spacing_mm < 0):
        parser.error('--longitudinal-line-min-spacing-mm must be nonnegative and finite')
    if args.n_vectors < args.n_modes:
        parser.error('--n-vectors must be at least --n-modes')
    try:
        modes = [positive_int(item.strip()) for item in args.global_modes.split(',')]
        if len(set(modes)) != len(modes):
            raise ValueError('duplicate mode numbers')
    except (argparse.ArgumentTypeError, ValueError) as error:
        parser.error('--global-modes: '+str(error))
    args.global_modes = ','.join(str(mode) for mode in modes)
    if args.local_mode > 20 or max(modes) > 20:
        parser.error('Selected reference mode numbers must be <= 20 (--reference-modes is fixed at 20)')
    for name in ('family_root', 'output_root', 'repo'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Required file missing: '+str(path))
    return path


def read_json(path):
    return json.loads(require_file(path).read_text(encoding='utf-8-sig'))


def plate_metadata_matches(left, right):
    """Compare all plate provenance, allowing only floating-point roundoff.

    CAE/JSON stage transfers can change the last bits of interval coordinates.
    Keep keys, sequence order, names and discrete values exact; the tolerance
    is far below the geometry checks and never replaces the file hash gates.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            plate_metadata_matches(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            plate_metadata_matches(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, rel_tol=1e-14, abs_tol=1e-12)
    return left == right


def family_key(path):
    match = re.fullmatch(r'family_(\d+)', path.name, re.IGNORECASE)
    return (0, int(match.group(1)), path.name) if match else (1, 0, path.name)


EIGENVALUE_GROUP_SPAN = 1e-3   # relative span of one eigenvalue group, as in Step 4 (degenerate_partners)
_EIGEN_ROW = re.compile(r'^\s*(\d+)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$')


def reference_eigenvalues(reference):
    """{mode: eigenvalue} of a completed reference solve (MODE NO / EIGENVALUE table of its
    .dat), or {} when the reference is not (yet) completed or the table is missing."""
    try:
        job = read_json(reference/'gdlc_reference.json').get('job')
    except (OSError, ValueError):
        return {}
    if not job or not completed_reference(reference, job, 'G'):
        return {}
    dat = reference/(job+'.dat')
    if not dat.is_file():
        return {}
    values, inside = {}, False
    for line in dat.read_text(errors='replace').splitlines():
        upper = line.upper()
        if 'MODE NO' in upper and 'EIGENVALUE' in upper:
            if values:
                break
            inside = True
            continue
        if inside:
            match = _EIGEN_ROW.match(line)
            if match:
                values[int(match.group(1))] = float(match.group(2))
            elif values and line.strip():
                break
    return values


def eigenvalue_group(eigenvalues, mode, span=EIGENVALUE_GROUP_SPAN):
    """Modes of the span-based eigenvalue group containing `mode` (Step 4's grouping rule)."""
    groups = []
    for value, member in sorted((v, m) for m, v in eigenvalues.items() if math.isfinite(v)):
        if not groups or abs(value-groups[-1][0][0]) > span*max(abs(groups[-1][0][0]), 1e-300):
            groups.append([])
        groups[-1].append((value, member))
    return next((sorted(m for unused, m in g) for g in groups if any(m == mode for unused, m in g)), [mode])


def family_args(args, run):
    """Arguments with the global imperfection modes this family actually uses.

    The global imperfection is ONE eigenvalue group of REF_G. An explicit list that is not
    one complete group of the solved REF_G (e.g. --global-modes 1,2 for bending, where the
    lateral-torsional mode 1 is single: 22745 vs 37010 MPa) is resolved to the complete group
    of its first mode, with a warning, instead of failing in Step 4. A single requested mode is
    passed on unchanged (Step 4 completes a degenerate pair itself for the bow direction).
    Before REF_G is solved the request is returned unchanged; the same resolution is applied by
    Step 4 provenance, receipts and --resume checks, so a resumed family compares like with like."""
    requested = [int(v) for v in str(args.global_modes).split(',') if v.strip()]
    if len(requested) < 2:
        return args
    eigenvalues = reference_eigenvalues(Path(str(run)+'_REF_G'))
    if not eigenvalues or any(m not in eigenvalues for m in requested):
        return args
    group = eigenvalue_group(eigenvalues, requested[0])
    if set(group) == set(requested):
        return args
    resolved = copy.copy(args)
    resolved.global_modes = ','.join(str(m) for m in group)
    print('WARNING %s: --global-modes %s is not one eigenvalue group of REF_G (%s); using the complete group '
          'of mode %d: %s' % (run.name, args.global_modes,
          ', '.join('%d: %.6g' % (m, eigenvalues[m]) for m in requested), requested[0], resolved.global_modes),
          flush=True)
    return resolved


def global_mode_angle(args):
    """A single compression mode may be torsional, so do not prescribe a bow."""
    if getattr(args,'load_case','compression') == 'bending':
        return args.bending_axis_deg
    return 0. if ',' in args.global_modes else None


def check_global_reference(info, reference, load_case='compression'):
    expected = 'restrained' if load_case == 'compression' else 'free'
    if info.get('twist') != expected:
        requirement = ('restrain section twist for compression' if expected == 'restrained'
                       else 'leave section twist free for bending/combined')
        raise ValueError('REF_G must %s: %s. Changed references require a new output destination; '
                         'existing results are preserved.' % (requirement, reference))


def check_resume_build(args, family, run):
    """Read-only provenance/lock gate; completed downstream stages are reusable."""
    args = family_args(args, run)
    state = read_json(run/'pipeline_status.json')
    settings = state.get('settings', {})
    if state.get('status') != 'BUILT':
        raise ValueError('--resume requires a completed BUILT original: '+str(run))
    if Path(settings.get('builtup_dir', '')).resolve() != family.resolve():
        raise ValueError('--resume source directory differs: '+str(run))
    for name in ('mesh_mm', 'n_modes', 'n_vectors', 'max_iterations', 'buckle_output',
                 'nodal_precision', 'longitudinal_lines', 'longitudinal_line_min_spacing_mm'):
        if settings.get(name) != getattr(args, name):
            raise ValueError('--resume setting differs (%s): %s' % (name, run))
    check_load_case(args,dict(type=settings.get('load_case','compression'),
                             neutral_axis_angle_deg=settings.get('bending_axis_deg')),str(run))
    check_load_case(args,state.get('build',{}).get('load_case') or {},str(run))
    for key in ('steel_plate_length','steel_plate_length_2','steel_plate_gap_mm','steel_plate_skip_count'):
        if settings.get(key,0)!=getattr(args,key,0):
            raise ValueError('--resume plate setting differs (%s): %s'%(key,run))
    job = state.get('build', {}).get('job_name')
    if not job:
        raise ValueError('--resume original job name missing: '+str(run))
    inp = require_file(run/(job+'.inp'))
    require_file(run/(job+'.cae'))
    check_source_inputs(family,run,state,inp)
    for kind in ('L', 'G'):
        reference = Path(str(run)+'_REF_'+kind)
        if not reference.exists():
            continue
        info = read_json(reference/'gdlc_reference.json')
        if (info.get('kind') != kind or info.get('source_run') != run.name or
                info.get('source_inp') != inp.name or info.get('n_modes') != 20):
            raise ValueError('--resume reference provenance differs: '+str(reference))
        if getattr(args,'load_case','compression')=='bending':
            check_load_case(args,info.get('load_case') or {},str(reference))
        if kind=='G':
            check_global_reference(info,reference,getattr(args,'load_case','compression'))
        if inp.stat().st_mtime_ns > (reference/'gdlc_reference.json').stat().st_mtime_ns:
            raise ValueError('--resume original INP changed after reference creation: '+str(reference))
        require_file(reference/(info['job']+'.inp'))
        if any(reference.glob('*.lck')):
            raise ValueError('--resume reference is locked/running: '+str(reference))
    if any(run.glob('*.lck')):
        raise ValueError('--resume original is locked/running: '+str(run))
    progress = dict(step4=False, step5=False, completed=False)
    cae=run/'imperfection'/'Step4_LG.cae'
    output=cae.parent/getattr(args,'step5_directory',STEP5_DIRECTORY)
    if cae.parent.is_dir() and any(cae.parent.rglob('*.lck')):
        raise ValueError('--resume imperfection job/database is locked/running: '+str(run))
    if output.exists():
        require_file(cae)
        infos=check_step5_inputs(args,run)
        from abaqus_step5_queue import completed_curve,backup_failed_artifacts
        entries={name:completed_curve(output,name,info,allow_incomplete=True) for name,info in infos.items()}
        for name,entry in entries.items():
            if entry is None:
                backup_failed_artifacts(str(output),name) # fail before submitting anything
        clean=all(entry is not None and not entry[0].get('error') and not entry[0].get('comparison_error')
                  and not entry[0].get('completion_reassessed')
                  for entry in entries.values())
        progress.update(step4=True,step5=True,completed=clean and (output/'STEP5_peak_summary.csv').is_file(),entries=entries)
    elif cae.exists():
        previous=cae.parent/default_step5_directory(args)
        if output != previous and previous.is_dir():
            # Verify the existing delivered input/provenance, not its failed
            # capacity. Only the fresh attempt may use a different predictor.
            infos=check_step5_inputs(args,run,output=previous,match_predictor=False)
            from abaqus_step5_queue import completed_curve
            entries={name:completed_curve(previous,name,info) for name,info in infos.items()}
            if all(entry is not None and not entry[0].get('error') and
                   not entry[0].get('comparison_error') for entry in entries.values()):
                progress.update(step4=True,completed=True,entries=entries)
                return progress
            progress['step4']=True
            return progress
        if not (cae.parent/STEP4_RECEIPT).is_file():
            raise ValueError('Cannot verify a legacy Step4-only CAE without a pipeline checkpoint or Step5 '
                             'sidecars. Existing files are preserved; use a new output folder: '+str(cae))
        receipt=read_json(cae.parent/STEP4_RECEIPT)
        if receipt != step4_receipt(args,family,run):
            raise ValueError('--resume Step4 settings/source files differ: '+str(cae))
        progress['step4']=True
    return progress


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_source_inputs(family,run,state,inp):
    """Use content, not file timestamps, to accept copied/retouched legacy CSVs.

    Old build summaries omit seam coordinates. Check those against the actual
    INP bolt endpoints as well, so changed seams cannot pass a geometry-only check.
    """
    receipt=run/'imperfection'/STEP4_RECEIPT
    if receipt.is_file():
        saved=read_json(receipt).get('files',{})
        if all(saved.get(str(family/n))==file_sha256(family/n) for n in INPUT_FILES):return
    recorded=state.get('source_inputs') or {}
    if not recorded.get('section_segments'):
        # Very old/minimal archives have no semantic snapshot to compare.
        if any((family/n).stat().st_mtime_ns>inp.stat().st_mtime_ns for n in INPUT_FILES):
            raise ValueError('--resume source CSVs changed and no complete source snapshot exists: '+str(family))
        return
    import abaqus_complete_model_m20 as builder
    current=builder.read_model_inputs(str(family))
    settings=state.get('settings') or {}
    plate_settings=dict(height=settings.get('steel_plate_length',0.),height_2=settings.get('steel_plate_length_2',0.),
                        edge_gap=settings.get('steel_plate_gap_mm',0.),skip_count=settings.get('steel_plate_skip_count',0))
    summary=builder.input_summary(current,plate_settings=plate_settings)
    if any(recorded.get(k)!=v for k,v in summary.items() if k!='source_directory'):
        raise ValueError('--resume source geometry/material/bolt settings differ: '+str(family))
    from gdlc_classifier import gdlc_mode_decomposition as classifier
    model=classifier.read_inp(inp)
    mesh=classifier.load_mesh(None,None,inp)
    xyz={(str(i),int(n)):tuple(map(float,p)) for i,n,p in zip(mesh.instance,mesh.label,mesh.xyz)}
    actual=[(a,b) for kind,a,b in model.pairs if kind=='BEAM']
    expected=[]
    for row in summary['bolt_layout']:
        z=row['z_mm'];xa,ya=row['a_xy'];xb,yb=row['b_xy']
        a=('P%d'%row['pa'],(xa,ya,z));b=('P%d'%row['pb'],(xb,yb,z))
        if row['plate_instance']:
            center=(xa,0.,z) if row['plate_direction']=='X' else (0.,ya,z)
            control=(row['plate_instance'],center)
            expected.extend(((control,a),(control,b)))
        else:expected.append((a,b))
    if len(actual)!=len(expected):
        raise ValueError('--resume source seam/bolt count differs: '+str(family))
    unused=set(range(len(actual)))
    for a,b in expected:
        match=next((j for j in unused for u,v in (actual[j],actual[j][::-1])
                    if a[0]==u[0] and b[0]==v[0] and math.dist(a[1],xyz[u])<=1e-3
                    and math.dist(b[1],xyz[v])<=1e-3),None)
        if match is None:raise ValueError('--resume source seam coordinates differ from INP: '+str(family))
        unused.remove(match)


def completed_reference(reference, job, kind):
    """Successful reference execution only; modal quality is deliberately not assessed."""
    from abaqus_step5_queue import analysis_finished
    inp, odb = (reference/(job+suffix) for suffix in ('.inp', '.odb'))
    if not all(path.is_file() and path.stat().st_size for path in (inp, odb)):
        return False
    if (reference/(job+'.lck')).exists() or odb.stat().st_mtime_ns < inp.stat().st_mtime_ns:
        return False
    return analysis_finished(str(reference), job)


def reference_ready(run, kind):
    """--resume: a reference whose solve completed is reused as it is. Any other reference
    (missing, unsolved, interrupted or failed) is regenerated by the current builder; an
    unfinished attempt is preserved there under superseded_attempts/, never deleted."""
    reference = Path(str(run)+'_REF_'+kind)
    try:
        job = read_json(reference/'gdlc_reference.json').get('job')
    except (OSError, ValueError):
        return False
    return bool(job) and completed_reference(reference, job, kind)


def step4_receipt(args,family,run,include_cae=True):
    """Small checkpoint for a future interruption between Step4 and Step5."""
    state=read_json(run/'pipeline_status.json');job=state['build']['job_name']
    paths=[family/n for n in INPUT_FILES]+[run/(job+'.inp'),run/(job+'.cae'),run/'pipeline_status.json',
                                        run/'imperfection'/'Step4_LG.cae']
    for kind in ('L','G'):
        ref=Path(str(run)+'_REF_'+kind);info=read_json(ref/'gdlc_reference.json')
        paths += [ref/'gdlc_reference.json',ref/(info['job']+'.inp'),ref/(info['job']+'.odb')]
    if not include_cae:
        paths=[p for p in paths if p.suffix.lower()!='.cae']
    args=family_args(args,run)
    receipt=dict(local_mode=args.local_mode,global_modes=args.global_modes,
                 files={str(p):file_sha256(require_file(p)) for p in paths})
    if getattr(args,'load_case','compression')=='bending':
        receipt.update(load_case='bending',bending_axis_deg=args.bending_axis_deg)
    return receipt


def check_step5_inputs(args,run,output=None,match_predictor=True):
    """Validate delivered inputs and embedded Step4 provenance without opening a CAE."""
    args=family_args(args,run)
    output=output or run/'imperfection'/getattr(args,'step5_directory',STEP5_DIRECTORY)
    require_file(output/('Step5_GMNIA_FY'+column_fy_tag(args)+'.cae'))
    require_file(output/'run_step5_queue_70_pct.bat')
    if {p.stem for p in output.glob('*.inp')} != expected_jobs(args):
        raise ValueError('Expected exactly PERFECT and LG_mp Step5 inputs: '+str(output))
    state=read_json(run/'pipeline_status.json');job=state['build']['job_name']
    check_load_case(args,state['build'].get('load_case') or {},str(run))
    # Delivered STEP5 INPs are immutable solver inputs. The original CAE is no
    # longer consumed here and can have changed only its saved viewport state;
    # bind to the original FE input/status, not that unused binary database.
    hashes={key:file_sha256(run/name) for key,name in (
        ('source_inp_sha256',job+'.inp'),('pipeline_status_sha256','pipeline_status.json'))}
    thickness=float(state['source_inputs']['thickness_mm']);length=float(state['build']['length_mm'])
    infos={}
    ref_hashes={}
    for name in sorted(expected_jobs(args)):
        info=read_json(output/(name+'_step5.json'));settings=info.get('settings',{})
        if info.get('stage')!=5 or info.get('job_name')!=name:
            raise ValueError('Invalid Step5 metadata: '+name)
        if info.get('input_sha256')!=file_sha256(output/(name+'.inp')):
            raise ValueError('Step5 input hash differs: '+name)
        check_end_condition(info,name)
        if any(settings.get(k)!=v for k,v in step5_settings(args).items()) or settings.get('stress_strain'):
            raise ValueError('--resume Step5 settings differ: '+name)
        plates=(state.get('source_inputs') or {}).get('steel_plates')
        tables=info.get('plastic_tables',{})
        column_tables=[table for material,table in tables.items()
                       if not plates or material!=plates['material']]
        if not column_tables or any(table!=[[step5_settings(args)['fy'],0.]] for table in column_tables):
            raise ValueError('--resume column material differs: '+name)
        if plates and (settings.get('steel_plate_fy')!=args.steel_plate_fy or
                info.get('plastic_tables',{}).get(plates['material'])!=[[args.steel_plate_fy,0.]]):
            raise ValueError('--resume plate material differs: '+name)
        predictor=settings.get('riks_extrapolation','linear')
        if predictor not in ('linear','none') or (match_predictor and predictor != args.riks_extrapolation):
            raise ValueError('--resume Step5 Riks extrapolation differs: '+name)
        if Path(settings.get('source_cae','')).resolve()!=run/'imperfection'/'Step4_LG.cae':
            raise ValueError('--resume Step5 source/load case differs: '+name)
        check_load_case(args,info.get('load_case') or {},name)
        reference=info.get('reference_pipeline') or {}
        if not plate_metadata_matches(reference.get('steel_plates'),plates):
            raise ValueError('--resume plate geometry/connections differ: '+name)
        if any(reference.get(k)!=v for k,v in hashes.items()):
            raise ValueError('--resume original model provenance differs: '+name)
        source=info.get('source_step4') or {};spec=source.get('specification') or {}
        s4=spec.get('settings') or {}
        gm=s4.get('global_modes')
        if gm is None:gm=[int(v) for v in str(s4.get('global_mode','')).split(',') if v]
        angle=global_mode_angle(args)
        if getattr(args,'load_case','compression')=='bending' and s4.get('skip_mode_quality_checks',False):
            raise ValueError('Bending Step4 mode quality checks were bypassed: '+name)
        if (s4.get('local_mode')!=args.local_mode or list(gm)!=[int(v) for v in args.global_modes.split(',')]
                or s4.get('global_angle_deg')!=angle):
            raise ValueError('--resume selected Step4 modes/settings differ: '+name)
        for component,kind in (('local','L'),('global','G')):
            ref=Path(str(run)+'_REF_'+kind);refinfo=read_json(ref/'gdlc_reference.json')
            if kind=='G':check_global_reference(refinfo,ref,getattr(args,'load_case','compression'))
            odb=ref/(refinfo['job']+'.odb')
            recorded=(spec.get('component_sources') or {}).get(component) or {}
            if odb not in ref_hashes:ref_hashes[odb]=file_sha256(require_file(odb))
            if Path(recorded.get('odb','')).resolve()!=odb or recorded.get('odb_sha256')!=ref_hashes[odb]:
                raise ValueError('--resume reference mode source differs: '+name+' '+component)
        case=source.get('case') or {}
        expected=(0.,0.,0.) if 'PERFECT' in name else (-.34*thickness,0.,length/1000.)
        if any(not math.isclose(float(case.get(k,float('nan'))),v,rel_tol=1e-10,abs_tol=1e-10)
               for k,v in zip(('local_component_mm','distortional_component_mm','global_component_mm'),expected)):
            raise ValueError('--resume imperfection amplitudes differ: '+name)
        infos[name]=info
    receipt=run/'imperfection'/STEP4_RECEIPT
    if receipt.is_file():
        saved=read_json(receipt)
        # STEP5 consumes its verified INPs, not either saved CAE. Keep the full
        # binary CAE check above only when resuming before STEP5 was delivered.
        saved=dict(saved,files={p:h for p,h in saved['files'].items()
                                if Path(p).suffix.lower()!='.cae'})
        if saved!=step4_receipt(args,Path(state['settings']['builtup_dir']),run,include_cae=False):
            raise ValueError('--resume Step4 checkpoint differs: '+str(run))
    return infos


def update_family_plot(batch_root,plans,finished,selection_suffix='',load_case='compression',bending_axis_deg=0.,configuration_label=None,column_fy=240.):
    """Read each finished family's curves once; redraw only the shared small figure."""
    from abaqus_step5_queue import plot_svg,plot_png,fd
    import importlib.util
    curves,styles=[],[]
    for index,(family,_,_) in enumerate(plans):
        if family.name not in finished:continue
        rgb=colorsys.hsv_to_rgb((index*.618033988749895)%1.,.72,.75)
        color='#' + ''.join('%02x'%round(v*255) for v in rgb)
        tag=('%g'%column_fy).replace('.','p').replace('+','')
        for name,label,dash in (('STEP5_PERFECT_FY'+tag,'PERFECT',''),('STEP5_LG_mp_FY'+tag,'LG_mp','8 5')):
            result, points = finished[family.name][name]
            if not result.get('capacity_verified',result.get('criterion_reached')):
                label += ' (capacity unverified: '+result.get('outcome','UNKNOWN')+')'
            curves.append((family.name+' '+label,points))
            styles.append(dict(color=color,dash=dash))
    if not curves:return
    title='Family comparison - PERFECT (solid), LG_mp (dashed)'
    if configuration_label:title+=' | '+configuration_label
    axes=fd.RESPONSE_AXES[load_case]
    tag=loads.load_case_tag(load_case,bending_axis_deg)
    fy_suffix=('_FY'+('%g'%column_fy).replace('.','p').replace('+','')) if column_fy!=240. else ''
    stem=batch_root/('STEP5_family'+axes['stem']+tag+fy_suffix+selection_suffix)
    # Replace only after rendering succeeds, so an interrupted render keeps the last figure.
    for ext,writer in (('.svg',plot_svg),('.png',plot_png)):
        if ext=='.png' and importlib.util.find_spec('matplotlib') is None:continue
        temporary=stem.with_name(stem.name+'.pending'+ext)
        try:
            writer(str(temporary),curves,[],title,axes,styles=styles)
            os.replace(temporary,stem.with_suffix(ext))
        finally:
            if temporary.exists():temporary.unlink()
    print('COMPARISON UPDATED: '+str(stem.with_suffix('.svg')),flush=True)


def preflight(args):
    """Read-only: validate every family and destination before creating any output."""
    dependencies=REPO_FILES+REPO_DEPENDENCIES+(('abaqus_pm_interaction.py',) if getattr(args,'pm_interaction',False) else ())
    for filename in dependencies:
        if not (args.repo/filename).is_file():
            raise ValueError('--repo must point to the complete code folder on this computer: %s. '
                             'Required file missing: %s. Correct --repo, or omit it to use %s.' %
                             (args.repo, args.repo/filename, Path(__file__).resolve().parent))
    if not args.family_root.is_dir():
        raise ValueError('Family root directory not found: '+str(args.family_root))
    if args.output_root.exists() and not args.output_root.is_dir():
        raise ValueError('Output root is not a directory: '+str(args.output_root))
    families = sorted((path for path in args.family_root.iterdir() if path.is_dir()), key=family_key)
    if not families:
        raise ValueError('No family folders found: '+str(args.family_root))
    start=getattr(args,'start_family',None)
    if start is not None:
        selected=next((i for i,family in enumerate(families) if family.name==start),None)
        if selected is None:
            raise ValueError('Start family not found: '+str(start))
        # Select BEFORE reading source/provenance/results. A deliberately
        # excluded failed family cannot block or be resubmitted by this run.
        print('START AT '+start+'; earlier families excluded and preserved.',flush=True)
        families=families[selected:]
    batch_root = batch_output_directory(args)
    if batch_root.exists() and not batch_root.is_dir():
        raise ValueError('Batch output is not a directory: '+str(batch_root))
    plans = []
    destinations=set()
    for family in families:
        for filename in INPUT_FILES:
            require_file(family/filename)
        run = batch_root/family_run_name(args,family)
        targets=(run,Path(str(run)+'_REF_L'),Path(str(run)+'_REF_G'))
        if any(target in destinations for target in targets):
            raise ValueError('Family/reference output names collide: '+family.name)
        destinations.update(targets)
        if run.exists() and getattr(args, 'resume', False):
            progress=check_resume_build(args, family, run)
            if getattr(args,'pm_interaction',False):progress['pm_state']=check_pm_resume(args,family,run)
            plans.append((family, run, progress))
            continue
        for target in targets:
            if target.exists():
                raise ValueError('Output already exists; choose a new --output-root: '+str(target))
        plans.append((family, run, {}))
    return plans


def run_command(command, cwd, label, repo=None):
    command = [str(item) for item in command]
    print('\n['+label+'] '+subprocess.list2cmdline(command), flush=True)
    if os.name == 'nt' and Path(command[0]).suffix.lower() in ('.bat', '.cmd'):
        # /S needs outer quotes in addition to quotes around the batch path.
        # Quote every argument, including bare shell metacharacters such as '&'.
        if any('"' in item or '\n' in item or '\r' in item or '%' in item for item in command):
            raise ValueError('Batch command arguments cannot contain quotes, newlines or %')
        line = '"'+' '.join('"'+item+'"' for item in command)+'"'
        command = [os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/s', '/v:off', '/c', line]
        # Pass the command line directly: list2cmdline would backslash-escape the
        # nested cmd.exe quotes, which cmd does not interpret like a C runtime.
        command = subprocess.list2cmdline(command[:5])+' '+line
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    # CAE noGUI does not reliably add the script directory to sys.path. Keep
    # launch cwd in outputs, and expose the requested repo to every child,
    # including the Abaqus process launched later from the generated queue BAT.
    repo = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parent
    module_paths = [str(repo), str(repo/'gdlc_classifier')]
    existing = environment.get('PYTHONPATH')
    if existing:
        module_paths.append(existing)
    environment['PYTHONPATH'] = os.pathsep.join(module_paths)
    subprocess.run(command, cwd=str(cwd), check=True, env=environment)


PM_DIRECTORY='Step5_PM_FY240_LG_'+STEP5_END_TAG
PM_RECEIPT='family_pipeline_pm.json'


def pm_request(args,family,run):
    return dict(step4=step4_receipt(args,family,run,include_cae=False),
                axes=args.pm_axes,angles=args.pm_angles,settings=step5_settings(args),
                riks_extrapolation=args.riks_extrapolation,
                steel_plate_fy=args.steel_plate_fy if args.steel_plate_length else None)


def pm_input_record(output,request):
    import abaqus_pm_interaction as pm
    manifest=read_json(output/pm.MANIFEST_FILE)
    jobs=[row['job'] for row in manifest.get('jobs',[])]
    if (manifest.get('mode')!='gmnia' or not jobs or len(set(jobs))!=len(jobs)
            or set(jobs)!={p.stem for p in output.glob('PM_*.inp')}
            or any(not re.fullmatch(r'PM_[A-Za-z0-9_]+',name) for name in jobs)):
        raise ValueError('Invalid P-M input manifest: '+str(output))
    hashes={}
    for name in jobs:
        inp=require_file(output/(name+'.inp'))
        sidecar=require_file(output/(name+'_step5.json'));info=read_json(sidecar)
        if (info.get('stage')!=5 or info.get('job_name')!=name or
                info.get('load_case',{}).get('type')!='combined' or
                info.get('input_sha256')!=file_sha256(inp)):
            raise ValueError('Invalid P-M input/metadata binding: '+name)
        check_end_condition(info,name)
        hashes[inp.name]=file_sha256(inp);hashes[sidecar.name]=file_sha256(sidecar)
    for name in (pm.MANIFEST_FILE,pm.SECTION_FILE):
        hashes[name]=file_sha256(require_file(output/name))
    return dict(request=request,files=hashes)


def pm_plot_current(output,record):
    points=output/'STEP5_PM_points.csv'
    return (points.is_file() and (output/'diagram'/'PM_interaction_points.csv').is_file()
            and record.get('plot_source_sha256')==file_sha256(points)
            and bool(record.get('plot_files'))
            and all((output/'diagram'/name).is_file() for name in record['plot_files']))


def check_pm_resume(args,family,run):
    """Bind P-M rays to this configuration before reusing any results."""
    output=run/'imperfection'/pm_directory(args)
    if not output.exists() or output.is_dir() and not any(output.iterdir()):return None
    if not args.resume:raise ValueError('P-M output already exists: '+str(output))
    receipt=read_json(output/PM_RECEIPT)
    record=pm_input_record(output,pm_request(args,family,run))
    if any(receipt.get(key)!=value for key,value in record.items()):
        raise ValueError('--resume P-M settings, provenance or inputs differ: '+str(output))
    from abaqus_step5_queue import completed_curve
    entries={name:completed_curve(output,name,read_json(path),allow_incomplete=True)
             for path in sorted(output.glob('PM_*_step5.json'))
             for name in [path.name[:-len('_step5.json')]]}
    clean=all(entry is not None and not any(entry[0].get(k) for k in
              ('error','comparison_error','completion_reassessed')) for entry in entries.values())
    status=output/'STEP5_queue_status.json'
    queue_complete=bool(entries) and clean and (output/'STEP5_PM_points.csv').is_file() and status.is_file()
    if queue_complete:
        from abaqus_step5_queue import processed_result
        results=read_json(status)
        queue_complete=(isinstance(results,list) and len(results)==len(entries) and
                        {row.get('model') for row in results}==set(entries) and
                        all(processed_result(row) and not row.get('error') and not row.get('comparison_error')
                            for row in results))
    return dict(record=receipt,queue_completed=queue_complete,
                completed=queue_complete and pm_plot_current(output,receipt))


def run_family_pm(args,family,run,abaqus,runner,prefix,state=None):
    """Orchestrate existing P-M generation, queue and collection; no new FE method."""
    import abaqus_pm_interaction as pm
    from abaqus_step5_queue import processed_result
    output=run/'imperfection'/pm_directory(args)
    state=state if state is not None else check_pm_resume(args,family,run)
    if state and state['completed']:
        print('REUSE COMPLETED P-M: '+str(output),flush=True)
        return
    def invoke(command,stage,cwd=run):
        values=[str(value) for value in command]
        if runner is run_command:run_command(values,cwd,prefix+' | '+stage,repo=args.repo)
        else:runner(values,cwd,prefix+' | '+stage)
    if state is None:
        request=pm_request(args,family,run)
        invoke([abaqus,'cae','noGUI='+str(args.repo/'abaqus_pm_interaction.py'),'--',
                '--mode','gmnia','--source-cae',run/'imperfection'/'Step4_LG.cae',
                '--output-dir',output,'--axes',args.pm_axes,'--pm-angles',args.pm_angles,
                '--models','PERFECT','STEP4_LG_mp','--fy',step5_settings(args)['fy'],'--cpus',args.cpus,
                *(['--steel-plate-fy',args.steel_plate_fy] if args.steel_plate_length else []),
                '--initial-arc','.01','--min-arc','1e-8','--max-arc','.05',
                '--max-increments','5000','--field-frequency','1','--postpeak-stop-ratio','.70',
                '--riks-extrapolation',args.riks_extrapolation],'P-M BUILD')
        record=pm_input_record(output,request)
        (output/PM_RECEIPT).write_text(json.dumps(record,indent=2),encoding='utf-8')
    else:record=state['record']
    command=[abaqus,'cae','noGUI='+str(args.repo/'abaqus_step5_queue_nogui.py'),'--',
             '--run-dir',output,'--cpus',args.cpus,'--gpus','auto','--ratio','.70',
             '--stop-method','monitor','--abaqus-cmd',abaqus]
    if args.resume:command.append('--resume')
    if not state or not state.get('queue_completed'):
        invoke(command,'P-M SOLVE QUEUE',cwd=output)
    results=read_json(output/'STEP5_queue_status.json')
    expected={name[:-len('.inp')] for name in record['files'] if name.endswith('.inp')}
    if (not isinstance(results,list) or len(results)!=len(expected) or
            {row.get('model') for row in results}!=expected or
            any(row.get('error') or row.get('comparison_error') or not processed_result(row) for row in results)):
        raise RuntimeError('P-M queue/results could not be verified: '+str(output))
    points=require_file(output/'STEP5_PM_points.csv')
    if not pm_plot_current(output,record):
        diagram=output/'diagram'
        invoke([sys.executable,args.repo/'abaqus_pm_interaction.py','--mode','collect',
                '--roots',output,'--output-dir',diagram,'--fy',step5_settings(args)['fy']],'P-M DIAGRAM')
        require_file(diagram/'PM_interaction_points.csv')
        plots=sorted(path.name for path in diagram.glob('PM_interaction*.svg'))
        if not plots:raise RuntimeError('P-M diagrams missing; matplotlib is required: '+str(diagram))
        record.update(plot_source_sha256=file_sha256(points),plot_files=plots)
        (output/PM_RECEIPT).write_text(json.dumps(record,indent=2),encoding='utf-8')


def run_pipeline(args, runner=run_command, abaqus=None, _plans=None):
    # Resolve local helper imports from the explicitly selected installation too.
    if sys.path[0] != str(args.repo):
        sys.path.insert(0, str(args.repo))
    configurations=skip_configurations(args)
    if len(configurations)>1:
        # Read-only validation of every destination before launching the first configuration.
        checked=[(config,preflight(config)) for config in configurations]
        for config,plans in checked:
            print('\n=== STEEL PLATE SKIP %d ==='%config.steel_plate_skip_count,flush=True)
            run_pipeline(config,runner=runner,abaqus=abaqus,_plans=plans)
        return
    plans = preflight(args) if _plans is None else _plans
    requested_launcher = getattr(args, 'abaqus_cmd', 'abaqus')
    abaqus = abaqus or shutil.which(requested_launcher)
    if not abaqus and any(not progress.get('completed') or getattr(args,'pm_interaction',False) and not (progress.get('pm_state') or {}).get('queue_completed') for _,_,progress in plans):
        raise ValueError('Abaqus launcher not found: %s. Set --abaqus-cmd to its full path, '
                         'or use a terminal where abaqus works.' % requested_launcher)
    if abaqus:
        # Every child runs in an output directory; a relative launcher path would break there.
        abaqus = str(Path(abaqus).resolve())
    builder, classifier, step4, step5 = (args.repo/name for name in REPO_FILES[:4])
    batch_root = batch_output_directory(args)
    batch_root.mkdir(parents=True, exist_ok=True)
    print('OUTPUT: '+str(batch_root), flush=True)
    print('COLUMN MATERIAL: fy = %g MPa'%step5_settings(args)['fy'],flush=True)
    if args.steel_plate_length:print('PLATE MATERIAL: F\'y = %g MPa'%args.steel_plate_fy,flush=True)
    finished={family.name:progress['entries'] for family,_,progress in plans if progress.get('completed')}
    selection_suffix=('_from_'+plans[0][0].name) if getattr(args,'start_family',None) else ''
    plot_options=dict(selection_suffix=selection_suffix,load_case=getattr(args,'load_case','compression'),
                      column_fy=step5_settings(args)['fy'],
                      bending_axis_deg=getattr(args,'bending_axis_deg',0.),
                      configuration_label=('plate skip %d'%args.steel_plate_skip_count) if args.steel_plate_length else None)
    if finished:update_family_plot(batch_root,plans,finished,**plot_options)
    for index, (family, run, progress) in enumerate(plans, 1):
        prefix = '%d/%d %s' % (index, len(plans), family.name)
        print('\n=== FAMILY '+prefix+' ===', flush=True)
        if progress.get('completed'):
            print('REUSE COMPLETED FAMILY: '+family.name+' (no primary analysis or ODB extraction)',flush=True)
            if getattr(args,'pm_interaction',False):
                run_family_pm(args,family,run,abaqus,runner,prefix,state=progress.get('pm_state'))
            continue
        def invoke(command, stage, cwd=batch_root):
            values = [str(item) for item in command]
            if runner is run_command:
                run_command(values, cwd, prefix+' | '+stage, repo=args.repo)
            else:
                runner(values, cwd, prefix+' | '+stage)
        if args.resume and run.exists():
            print('REUSE ORIGINAL: '+str(run), flush=True)
        else:
            invoke([abaqus, 'cae', 'noGUI='+str(builder), '--',
            '--builtup-dir', family, '--output-root', batch_root,
            '--mesh-mm', args.mesh_mm, '--n-modes', args.n_modes, '--n-vectors', args.n_vectors,
            '--max-iterations', args.max_iterations, '--cpus', args.cpus,
            '--buckle-output', args.buckle_output, '--nodal-precision', args.nodal_precision,
            '--longitudinal-lines', args.longitudinal_lines,
            '--longitudinal-line-min-spacing-mm', args.longitudinal_line_min_spacing_mm,
            '--load-case', getattr(args,'load_case','compression'),
            '--bending-axis-deg', getattr(args,'bending_axis_deg',0.),
            '--steel-plate-length', args.steel_plate_length,
            '--steel-plate-length-2', args.steel_plate_length_2,
            '--steel-plate-gap-mm', args.steel_plate_gap_mm,
            '--steel-plate-skip-count', args.steel_plate_skip_count,
                '--build-only'], 'BUILD ORIGINAL')
        state = read_json(run/'pipeline_status.json')
        if state.get('status') != 'BUILT':
            raise RuntimeError('Expected BUILT pipeline status: '+str(run))
        check_load_case(args,state.get('build',{}).get('load_case') or {},str(run))
        job = state.get('build', {}).get('job_name')
        if not job:
            raise RuntimeError('Missing original job name: '+str(run))
        for suffix in ('.cae', '.inp'):
            require_file(run/(job+suffix))
        pending = [] if progress.get('step4') else [
            kind for kind in ('L', 'G') if not (args.resume and reference_ready(run, kind))]
        if pending:
            invoke([sys.executable, classifier, run, '--make-reference-models',
                '--reference-kinds', ','.join(pending),
                '--reference-modes', '20', '--abaqus-cmd', abaqus], 'MAKE REFERENCES')
        for kind in ('L', 'G'):
            if progress.get('step4'):break # sources already verified in downstream provenance
            reference = Path(str(run)+'_REF_'+kind)
            ref_info = read_json(reference/'gdlc_reference.json')
            if kind=='G':check_global_reference(ref_info,reference,getattr(args,'load_case','compression'))
            ref_job = ref_info.get('job')
            if not ref_job:
                raise RuntimeError('Missing reference job name: '+str(reference))
            if args.resume and completed_reference(reference, ref_job, kind):
                print('REUSE COMPLETED REF_'+kind+': '+str(reference), flush=True)
            else:
                invoke([sys.executable, classifier, reference, '--solve-reference-models',
                '--solve-only', '--cpus', args.cpus,
                '--abaqus-cmd', abaqus], 'SOLVE REF_'+kind,
                cwd=run)
            require_file(reference/(ref_job+'.odb'))
            if not completed_reference(reference,ref_job,kind):
                raise RuntimeError('Reference execution did not complete successfully or essential outputs are missing/stale: '+str(reference))
        step4_cae = run/'imperfection'/'Step4_LG.cae'
        if not progress.get('step4'):
            family_modes = family_args(args, run)     # one complete REF_G eigenvalue group
            invoke([abaqus, 'cae', 'noGUI='+str(step4), '--', '--run-dir', run,
            '--local-source', str(run)+'_REF_L', '--local-mode', args.local_mode,
            '--global-source', str(run)+'_REF_G', '--global-mode', family_modes.global_modes,
            *(['--global-angle-deg',global_mode_angle(family_modes)]
              if global_mode_angle(family_modes) is not None else []),
            '--cases', 'none', '--combo', 'LG_mp=-0.34t,0,L/1000',
                '--output-cae', step4_cae], 'STEP4')
            require_file(step4_cae)
            (step4_cae.parent/STEP4_RECEIPT).write_text(json.dumps(step4_receipt(args,family,run),indent=2),encoding='utf-8')
        require_file(step4_cae)
        output = run/'imperfection'/getattr(args,'step5_directory',STEP5_DIRECTORY)
        if not progress.get('step5'):
            invoke([abaqus, 'cae', 'noGUI='+str(step5), '--', '--source-cae', step4_cae,
            '--output-dir', output, '--fy', step5_settings(args)['fy'], '--cpus', args.cpus,
            *(['--steel-plate-fy',args.steel_plate_fy] if args.steel_plate_length else []),
            '--initial-arc', '0.01', '--min-arc', '1e-8', '--max-arc', '0.05',
            '--max-increments', '5000', '--field-frequency', '1',
                '--riks-extrapolation', args.riks_extrapolation,
                '--postpeak-stop-ratio', '0.70'], 'STEP5 BUILD')
        infos=check_step5_inputs(args,run)
        inputs=set(infos)
        stop_methods={info['settings'].get('stop_method','monitor') for info in infos.values()}
        if len(stop_methods)!=1:
            raise RuntimeError('Step5 inputs have inconsistent stop methods: '+str(output))
        # Invoke the existing queue driver directly so current resource settings and
        # resume reach it, even if the saved BAT was generated on another computer.
        command=[abaqus,'cae','noGUI='+str(args.repo/'abaqus_step5_queue_nogui.py'),'--',
                 '--run-dir',output,'--cpus',args.cpus,'--gpus','auto','--ratio','0.70',
                 '--stop-method',stop_methods.pop(),'--abaqus-cmd',abaqus]
        if args.resume:command.append('--resume')
        invoke(command, 'STEP5 SOLVE QUEUE', cwd=output)
        results = read_json(output/'STEP5_queue_status.json')
        if (not isinstance(results, list) or len(results) != len(inputs) or
                {row.get('model') for row in results} != inputs):
            raise RuntimeError('Incomplete Step5 queue: '+str(output))
        from abaqus_step5_queue import completed_curve, processed_result
        for result in results:
            if result.get('error') or result.get('comparison_error') or not processed_result(result):
                raise RuntimeError('Step5 queue/plots failed: '+str(output)+' | '+str(result))
        require_file(output/'STEP5_peak_summary.csv')
        entries={name:completed_curve(output,name,info,allow_incomplete=True) for name,info in infos.items()}
        if not all(entries.values()):raise RuntimeError('Finished family curves could not be verified: '+str(output))
        finished[family.name]=entries
        update_family_plot(batch_root,plans,finished,**plot_options)
        if getattr(args,'pm_interaction',False):
            run_family_pm(args,family,run,abaqus,runner,prefix,state=progress.get('pm_state'))
        print('FINISHED: '+family.name, flush=True)
    label='SELECTED FAMILIES FINISHED' if getattr(args,'start_family',None) else 'ALL FAMILIES FINISHED'
    print('\n'+label+': '+str(batch_root), flush=True)


def main(argv=None):
    args = parse_arguments(argv)
    try:
        run_pipeline(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print('\nPIPELINE FAILED: '+str(error), file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print('\nPIPELINE INTERRUPTED; inspect running jobs before restarting.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    sys.exit(main())
