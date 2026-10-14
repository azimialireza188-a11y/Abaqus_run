# -*- coding: utf-8 -*-
"""Sequential family builds, G/L reference solves, imperfection and Step5 queue.

Run with normal Python, NOT ``abaqus cae``. The entry point launches Abaqus itself.
The original model is build-only; all imperfection modes come from solved REF_L/G.
Each family finishes its GMNIA queue before the next family starts. Existing run
directories are refused unless --resume verifies matching inputs and saved stages.
This workflow uses no added steel plates. Fixed settings: reference-modes=20,
LG_mp=-0.34t,0,L/1000, global-angle=0, member fy=240 MPa and postpeak ratio=0.70.
Step5 includes its default PERFECT baseline and the single LG_mp imperfect case.
Completed families update STEP5_family_force_displacement.svg (and PNG when
matplotlib is available): PERFECT solid, LG_mp dashed, one color per family.
"""
import argparse
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
STEP5_DIRECTORY = 'Step5_GMNIA_FY240_LG'
STEP4_RECEIPT = 'family_pipeline_step4.json'
STEP5_SETTINGS = dict(fy=240., initial_arc=.01, min_arc=1e-8, max_arc=.05,
                     max_increments=5000, field_frequency=1, postpeak_stop_ratio=.70)


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


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('family-root', 'output-root'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent,
        help='Code directory on THIS computer (default: directory containing this script)')
    parser.add_argument('--abaqus-cmd', default='abaqus',
        help='Abaqus launcher name in PATH or full path to its .bat/.cmd/executable')
    parser.add_argument('--mesh-mm', type=float, default=20.)
    parser.add_argument('--n-modes', type=positive_int, default=250)
    parser.add_argument('--n-vectors', type=positive_int, default=500)
    parser.add_argument('--max-iterations', type=positive_int, default=1250)
    parser.add_argument('--cpus', type=cpu_request, default='auto')
    parser.add_argument('--buckle-output', choices=('standard', 'detailed'), default='detailed')
    parser.add_argument('--nodal-precision', choices=('single', 'full'), default='full')
    parser.add_argument('--longitudinal-lines', type=int, default=4)
    parser.add_argument('--longitudinal-line-min-spacing-mm', type=float, default=5.)
    parser.add_argument('--local-mode', type=positive_int, default=1)
    parser.add_argument('--global-modes', default='1,2')
    parser.add_argument('--riks-extrapolation', choices=('linear','none'), default='linear',
        help='Step5 Riks predictor (linear preserves the existing default; none disables extrapolation)')
    parser.add_argument('--resume', action='store_true',
        help='Reuse verified completed stages/families; solve only unused job destinations; never overwrite analyses')
    args = parser.parse_args(argv)
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


def family_key(path):
    match = re.fullmatch(r'family_(\d+)', path.name, re.IGNORECASE)
    return (0, int(match.group(1)), path.name) if match else (1, 0, path.name)


def check_resume_build(args, family, run):
    """Read-only provenance/lock gate; completed downstream stages are reusable."""
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
    if settings.get('load_case', 'compression') != 'compression' or (state.get('source_inputs') or {}).get('steel_plates'):
        raise ValueError('--resume requires this pipeline\'s compression model without plates: '+str(run))
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
        if inp.stat().st_mtime_ns > (reference/'gdlc_reference.json').stat().st_mtime_ns:
            raise ValueError('--resume original INP changed after reference creation: '+str(reference))
        require_file(reference/(info['job']+'.inp'))
        if any(reference.glob('*.lck')):
            raise ValueError('--resume reference is locked/running: '+str(reference))
    if any(run.glob('*.lck')):
        raise ValueError('--resume original is locked/running: '+str(run))
    progress = dict(step4=False, step5=False, completed=False)
    cae=run/'imperfection'/'Step4_LG.cae'
    output=cae.parent/STEP5_DIRECTORY
    if cae.parent.is_dir() and any(cae.parent.rglob('*.lck')):
        raise ValueError('--resume imperfection job/database is locked/running: '+str(run))
    if output.exists():
        require_file(cae)
        infos=check_step5_inputs(args,run)
        from abaqus_step5_queue import completed_curve,backup_failed_artifacts
        entries={name:completed_curve(output,name,info) for name,info in infos.items()}
        for name,entry in entries.items():
            if entry is None:
                backup_failed_artifacts(str(output),name) # fail before submitting anything
        clean=all(entry is not None and not entry[0].get('error') and not entry[0].get('comparison_error')
                  and not entry[0].get('completion_reassessed')
                  for entry in entries.values())
        progress.update(step4=True,step5=True,completed=clean and (output/'STEP5_peak_summary.csv').is_file(),entries=entries)
    elif cae.exists():
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
    summary=builder.input_summary(current)
    if any(recorded.get(k)!=v for k,v in summary.items() if k!='source_directory'):
        raise ValueError('--resume source geometry/material/bolt settings differ: '+str(family))
    from gdlc_classifier import gdlc_mode_decomposition as classifier
    model=classifier.read_inp(inp)
    mesh=classifier.load_mesh(None,None,inp)
    xyz={(str(i),int(n)):tuple(map(float,p)) for i,n,p in zip(mesh.instance,mesh.label,mesh.xyz)}
    actual=[(a,b) for kind,a,b in model.pairs if kind=='BEAM']
    expected=[(('P%d'%pa,(xa,ya,z)),('P%d'%pb,(xb,yb,z)))
              for _,pa,pb,xa,ya,xb,yb in current[4] for z in current[5]]
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
    return dict(local_mode=args.local_mode,global_modes=args.global_modes,
                files={str(p):file_sha256(require_file(p)) for p in paths})


def check_step5_inputs(args,run):
    """Validate delivered inputs and embedded Step4 provenance without opening a CAE."""
    output=run/'imperfection'/STEP5_DIRECTORY
    require_file(output/'Step5_GMNIA_FY240.cae')
    require_file(output/'run_step5_queue_70_pct.bat')
    if {p.stem for p in output.glob('*.inp')} != EXPECTED_JOBS:
        raise ValueError('Expected exactly PERFECT and LG_mp Step5 inputs: '+str(output))
    state=read_json(run/'pipeline_status.json');job=state['build']['job_name']
    # Delivered STEP5 INPs are immutable solver inputs. The original CAE is no
    # longer consumed here and can have changed only its saved viewport state;
    # bind to the original FE input/status, not that unused binary database.
    hashes={key:file_sha256(run/name) for key,name in (
        ('source_inp_sha256',job+'.inp'),('pipeline_status_sha256','pipeline_status.json'))}
    thickness=float(state['source_inputs']['thickness_mm']);length=float(state['build']['length_mm'])
    infos={}
    ref_hashes={}
    for name in sorted(EXPECTED_JOBS):
        info=read_json(output/(name+'_step5.json'));settings=info.get('settings',{})
        if info.get('stage')!=5 or info.get('job_name')!=name:
            raise ValueError('Invalid Step5 metadata: '+name)
        if info.get('input_sha256')!=file_sha256(output/(name+'.inp')):
            raise ValueError('Step5 input hash differs: '+name)
        if any(settings.get(k)!=v for k,v in STEP5_SETTINGS.items()) or settings.get('stress_strain'):
            raise ValueError('--resume Step5 settings differ: '+name)
        if settings.get('riks_extrapolation','linear') != args.riks_extrapolation:
            raise ValueError('--resume Step5 Riks extrapolation differs: '+name)
        if (Path(settings.get('source_cae','')).resolve()!=run/'imperfection'/'Step4_LG.cae' or
                (info.get('load_case') or {}).get('type','compression')!='compression'):
            raise ValueError('--resume Step5 source/load case differs: '+name)
        reference=info.get('reference_pipeline') or {}
        if any(reference.get(k)!=v for k,v in hashes.items()):
            raise ValueError('--resume original model provenance differs: '+name)
        source=info.get('source_step4') or {};spec=source.get('specification') or {}
        s4=spec.get('settings') or {}
        gm=s4.get('global_modes')
        if gm is None:gm=[int(v) for v in str(s4.get('global_mode','')).split(',') if v]
        if (s4.get('local_mode')!=args.local_mode or list(gm)!=[int(v) for v in args.global_modes.split(',')]
                or s4.get('global_angle_deg')!=0.):
            raise ValueError('--resume selected Step4 modes/settings differ: '+name)
        for component,kind in (('local','L'),('global','G')):
            ref=Path(str(run)+'_REF_'+kind);refinfo=read_json(ref/'gdlc_reference.json')
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
        if saved!=step4_receipt(args,args.family_root/run.name,run,include_cae=False):
            raise ValueError('--resume Step4 checkpoint differs: '+str(run))
    return infos


def update_family_plot(batch_root,plans,finished):
    """Read each finished family's curves once; redraw only the shared small figure."""
    from abaqus_step5_queue import plot_svg,plot_png,fd
    import importlib.util
    curves,styles=[],[]
    for index,(family,_,_) in enumerate(plans):
        if family.name not in finished:continue
        rgb=colorsys.hsv_to_rgb((index*.618033988749895)%1.,.72,.75)
        color='#' + ''.join('%02x'%round(v*255) for v in rgb)
        for name,label,dash in (('STEP5_PERFECT_FY240','PERFECT',''),('STEP5_LG_mp_FY240','LG_mp','8 5')):
            result, points = finished[family.name][name]
            if not result.get('capacity_verified',result.get('criterion_reached')):
                label += ' (capacity unverified)'
            curves.append((family.name+' '+label,points))
            styles.append(dict(color=color,dash=dash))
    if not curves:return
    title='Family comparison - PERFECT (solid), LG_mp (dashed)'
    axes=fd.RESPONSE_AXES['compression']
    stem=batch_root/'STEP5_family_force_displacement'
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
    for filename in REPO_FILES+REPO_DEPENDENCIES:
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
    batch_root = args.output_root/args.family_root.name
    if batch_root.exists() and not batch_root.is_dir():
        raise ValueError('Batch output is not a directory: '+str(batch_root))
    plans = []
    destinations=set()
    for family in families:
        for filename in INPUT_FILES:
            require_file(family/filename)
        run = batch_root/family.name
        targets=(run,Path(str(run)+'_REF_L'),Path(str(run)+'_REF_G'))
        if any(target in destinations for target in targets):
            raise ValueError('Family/reference output names collide: '+family.name)
        destinations.update(targets)
        if run.exists() and getattr(args, 'resume', False):
            progress=check_resume_build(args, family, run)
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


def run_pipeline(args, runner=run_command, abaqus=None):
    # Resolve local helper imports from the explicitly selected installation too.
    if sys.path[0] != str(args.repo):
        sys.path.insert(0, str(args.repo))
    plans = preflight(args)
    requested_launcher = getattr(args, 'abaqus_cmd', 'abaqus')
    abaqus = abaqus or shutil.which(requested_launcher)
    if not abaqus and any(not progress.get('completed') for _,_,progress in plans):
        raise ValueError('Abaqus launcher not found: %s. Set --abaqus-cmd to its full path, '
                         'or use a terminal where abaqus works.' % requested_launcher)
    if abaqus:
        # Every child runs in an output directory; a relative launcher path would break there.
        abaqus = str(Path(abaqus).resolve())
    builder, classifier, step4, step5 = (args.repo/name for name in REPO_FILES[:4])
    batch_root = args.output_root/args.family_root.name
    batch_root.mkdir(parents=True, exist_ok=True)
    print('OUTPUT: '+str(batch_root), flush=True)
    finished={family.name:progress['entries'] for family,_,progress in plans if progress.get('completed')}
    if finished:update_family_plot(batch_root,plans,finished)
    for index, (family, run, progress) in enumerate(plans, 1):
        prefix = '%d/%d %s' % (index, len(plans), family.name)
        print('\n=== FAMILY '+prefix+' ===', flush=True)
        if progress.get('completed'):
            print('REUSE COMPLETED FAMILY: '+family.name+' (no analysis or ODB extraction)',flush=True)
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
                '--build-only'], 'BUILD ORIGINAL')
        state = read_json(run/'pipeline_status.json')
        if state.get('status') != 'BUILT':
            raise RuntimeError('Expected BUILT pipeline status: '+str(run))
        job = state.get('build', {}).get('job_name')
        if not job:
            raise RuntimeError('Missing original job name: '+str(run))
        for suffix in ('.cae', '.inp'):
            require_file(run/(job+suffix))
        if not progress.get('step4') and not (args.resume and all((Path(str(run)+'_REF_'+kind)/'gdlc_reference.json').is_file()
                                   for kind in ('L', 'G'))):
            invoke([sys.executable, classifier, run, '--make-reference-models',
                '--reference-modes', '20', '--abaqus-cmd', abaqus], 'MAKE REFERENCES')
        for kind in ('L', 'G'):
            if progress.get('step4'):break # sources already verified in downstream provenance
            reference = Path(str(run)+'_REF_'+kind)
            ref_info = read_json(reference/'gdlc_reference.json')
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
            invoke([abaqus, 'cae', 'noGUI='+str(step4), '--', '--run-dir', run,
            '--local-source', str(run)+'_REF_L', '--local-mode', args.local_mode,
            '--global-source', str(run)+'_REF_G', '--global-mode', args.global_modes,
            '--global-angle-deg', '0', '--cases', 'none', '--combo', 'LG_mp=-0.34t,0,L/1000',
                '--skip-mode-quality-checks',
                '--output-cae', step4_cae], 'STEP4')
            require_file(step4_cae)
            (step4_cae.parent/STEP4_RECEIPT).write_text(json.dumps(step4_receipt(args,family,run),indent=2),encoding='utf-8')
        require_file(step4_cae)
        output = run/'imperfection'/STEP5_DIRECTORY
        if not progress.get('step5'):
            invoke([abaqus, 'cae', 'noGUI='+str(step5), '--', '--source-cae', step4_cae,
            '--output-dir', output, '--fy', '240', '--cpus', args.cpus,
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
        from abaqus_step5_queue import completed_curve, finished_result
        for result in results:
            if result.get('error') or result.get('comparison_error') or not finished_result(result):
                raise RuntimeError('Step5 queue/plots failed: '+str(output)+' | '+str(result))
        require_file(output/'STEP5_peak_summary.csv')
        entries={name:completed_curve(output,name,info) for name,info in infos.items()}
        if not all(entries.values()):raise RuntimeError('Finished family curves could not be verified: '+str(output))
        finished[family.name]=entries
        update_family_plot(batch_root,plans,finished)
        print('FINISHED: '+family.name, flush=True)
    print('\nALL FAMILIES FINISHED: '+str(batch_root), flush=True)


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
