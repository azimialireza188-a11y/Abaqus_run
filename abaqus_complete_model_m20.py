# -*- coding: ascii -*-
"""Build CAE + INP, run buckling, then postprocess the resulting ODB.

Run: abaqus cae noGUI=abaqus_complete_model_m20.py -- --builtup-dir "CSV folder"
Add --mesh-mm 20 --n-modes 20 --n-vectors 60 --max-iterations 300 as needed.
Use --output-root "D:\\CFS-Column" to create an output folder named after the input.
Use --build-only to save CAE + INP without submitting an analysis.
Recover an existing run without solving again:
abaqus python abaqus_complete_model_m20.py --resume-post "existing run folder"
Postprocessing uses abaqus_modal_wavelengths.py and then abaqus_modal_report.py
beside this script. Keep all three files together when moving the workflow.
The automatic buckling pipeline stores only the nodal mode-shape output needed
for wavelength/family classification; shell S/E/SF/SE energy fields are not
requested because they do not participate in the current L/D/G classifier.
Command-line options override the defaults below for the entire pipeline.
MESH_MM is the target in both directions; bolt rows and ends stay exact.
BUILTUP_DIR is the source of geometry, material, thickness, member length,
seam coordinates and all bolt positions. Bolts are ideal BEAM MPC links,
as in CUFSM; bolt solids, holes and pretension are not represented.
General contact covers all exterior shell surfaces, including self-contact:
hard normal contact, frictionless tangential behavior, separation allowed.
Buckle freezes contact at the base state; it cannot enforce new contact in
scaled eigenmode plots. Finite-deformation contact requires nonlinear analysis.
Use --longitudinal-lines 0 for the original simplification (default), 2..99
for up to that many internal lines per curved region, or 100 for all source
lines. For 2..99, --longitudinal-line-min-spacing-mm can prevent optional
added boundaries from becoming too close along the section path. Essential
boundaries and bolt lines always remain. Lines are selected from the original
section; no coordinates or radii are changed.
Portable modal export can be generated after the same solver run, or later
from an existing ODB, without rebuilding the model. New runs retain both U
and UR; old U-only ODBs are exported without inventing rotations.
Set --check-inputs to validate the CSV data without starting Abaqus/CAE.
"""
import csv
import json
import math
import os
import sys
import argparse
import datetime
import tempfile
import threading
import time

# CAE noGUI executes scripts without defining __file__.
SCRIPT_DIR = os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename)))


BUILTUP_DIR = r'C:\Users\810200014.HAMI.000\Documents\CUFSM-Single\cfs_abaqus\A4784_t2_qm1_0_0_0_R8_lipR20t_lipLen60_M200_L3600_gap10_nb19_end25-25_row15'
SMALL_EDGE_MM = 5.0
MERGE_ANGLE_DEG = 10.0

# ================= USER SETTINGS: EDIT THESE VALUES =================
MESH_MM = 20.0        # target element size (mm), both section and length; e.g. 10.0
N_MODES = 20          # number of requested buckling modes; e.g. 5, 10, 30 or 50
N_VECTORS = 60       # vectors used per SUBSPACE iteration
MAX_ITERATIONS = 300  # iteration limit for SUBSPACE, separate from mode count
LONGITUDINAL_LINES = 0  # 0: original; 2..99: per curve; 100: all source lines
LONGITUDINAL_LINE_MIN_SPACING_MM = 0.0  # 0: disabled; applies to optional lines for 2..99
# Changing settings requires rebuilding the CAE/INP with this script.
# A finer mesh increases model size; exact bolt locations may require shorter edges.
# ====================================================================


def validate_settings():
    if (isinstance(LONGITUDINAL_LINES, bool) or
            not isinstance(LONGITUDINAL_LINES, int) or
            LONGITUDINAL_LINES not in [0] + list(range(2, 101))):
        raise ValueError('LONGITUDINAL_LINES must be 0 or an integer from 2 to 100')
    if (isinstance(LONGITUDINAL_LINE_MIN_SPACING_MM, bool) or
            not isinstance(LONGITUDINAL_LINE_MIN_SPACING_MM, (int, float)) or
            not math.isfinite(LONGITUDINAL_LINE_MIN_SPACING_MM) or
            LONGITUDINAL_LINE_MIN_SPACING_MM < 0):
        raise ValueError('LONGITUDINAL_LINE_MIN_SPACING_MM must be a finite nonnegative number')
    if (isinstance(MESH_MM, bool) or not isinstance(MESH_MM, (int, float)) or
            not MESH_MM > 0 or math.isinf(MESH_MM)):
        raise ValueError('MESH_MM must be a positive finite number in mm')
    for name, value in (('N_MODES', N_MODES), ('N_VECTORS', N_VECTORS),
                        ('MAX_ITERATIONS', MAX_ITERATIONS)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('%s must be a positive integer' % name)
    if N_VECTORS < N_MODES:
        raise ValueError('N_VECTORS must be at least N_MODES')


def parse_arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' in argv:
        argv = argv[argv.index('--')+1:]
    elif '-cae' in argv:
        # Abaqus 2024 removes the -- separator but retains its kernel options.
        filtered = []
        index = 0
        while index < len(argv):
            option = argv[index]
            if option == '-cae':
                index += 1
            elif option in ('-noGUI', '-tmpdir', '-lmlog'):
                index += 2
            else:
                filtered.append(option)
                index += 1
        argv = filtered
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builtup-dir', default=BUILTUP_DIR, help='Source CUFSM CSV directory')
    parser.add_argument('--mesh-mm', type=float, default=MESH_MM)
    parser.add_argument('--longitudinal-lines', type=int, default=LONGITUDINAL_LINES,
        help='0: original simplification; 2..99: internal lines per curved region; '
             '100: retain all source lines (essential and bolt lines always remain)')
    parser.add_argument('--longitudinal-line-min-spacing-mm', type=float,
        default=LONGITUDINAL_LINE_MIN_SPACING_MM,
        help='Minimum section-path spacing for optional lines added by --longitudinal-lines 2..99; '
             '0 disables the filter. Mandatory/bolt boundaries are never removed; 100 still retains all source lines.')
    parser.add_argument('--n-modes', type=int, default=N_MODES)
    parser.add_argument('--n-vectors', type=int, default=None)
    parser.add_argument('--max-iterations', type=int, default=MAX_ITERATIONS)
    parser.add_argument('--cpus', type=int, default=None, help='Default: all available CPUs')
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--buckle-output', choices=('standard', 'detailed'), default='standard',
                        help='Legacy compatibility option. Automatic buckling now always stores classification-only nodal mode shapes; detailed no longer adds S/E/SF/SE.')
    parser.add_argument('--nodal-precision', choices=('full', 'single'), default='full',
                        help='Nodal ODB storage precision; does not change eigensolver accuracy')
    parser.add_argument('--portable-export', choices=('off', 'parquet', 'npz', 'both'),
                        default='parquet',
                        help='Portable raw ODB export after solve; parquet is compact ZSTD columnar output')
    parser.add_argument('--portable-modes-per-shard', type=int, default=8,
                        help='Modes stored per portable mode-shape shard')
    parser.add_argument('--portable-export-workers', default='auto',
                        help='auto/all/max uses all logical CPUs for portable Parquet compression')
    parser.add_argument('--parquet-python', default=None,
                        help='Normal Python executable with pyarrow; auto-detected when omitted')
    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument('--output-dir', help='Exact new or empty output directory')
    output_options.add_argument('--output-root',
        help='Parent output directory; creates an input-named child, with _run02 etc. if it exists')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--skip-post', action='store_true', help='Build and solve, without wavelength processing')
    parser.add_argument('--modal-audit', action='store_true', help='After reports, create the direct-shape audit and graphical explorer')
    parser.add_argument('--check-inputs', action='store_true', help='Validate CSVs and settings only')
    parser.add_argument('--resume-post', metavar='RUN_DIR',
                        help='Verify completed run and postprocess its ODB; never build or submit')
    args = parser.parse_args(argv)
    import runtime_resources
    try:
        policy = runtime_resources.resolve_policy(runtime_resources.detect_resources(), args.cpus, args.gpus)
    except ValueError as error:
        parser.error(str(error))
    args.cpus, args.gpus = policy.cpus, policy.gpus
    if args.longitudinal_lines not in [0] + list(range(2, 101)):
        parser.error('--longitudinal-lines must be 0 or an integer from 2 to 100')
    if (not math.isfinite(args.longitudinal_line_min_spacing_mm) or
            args.longitudinal_line_min_spacing_mm < 0):
        parser.error('--longitudinal-line-min-spacing-mm must be a finite nonnegative number')
    if args.modal_audit and args.skip_post:
        parser.error('--modal-audit requires postprocessing; remove --skip-post')
    if args.resume_post and (args.build_only or args.skip_post or args.check_inputs or args.output_dir or args.output_root):
        parser.error('--resume-post cannot be combined with build/check/skip/output options')
    if not math.isfinite(args.mesh_mm) or args.mesh_mm <= 0:
        parser.error('--mesh-mm must be positive and finite')
    if min(args.n_modes, args.max_iterations, args.cpus) < 1:
        parser.error('--n-modes, --max-iterations and --cpus must be positive')
    if args.portable_modes_per_shard < 1:
        parser.error('--portable-modes-per-shard must be positive')
    if args.n_vectors is None:
        args.n_vectors = max(N_VECTORS, min(2*args.n_modes, args.n_modes+8))
    if args.n_vectors < args.n_modes:
        parser.error('--n-vectors must be at least --n-modes')
    args.builtup_dir = os.path.abspath(os.path.expanduser(args.builtup_dir))
    if args.output_dir:
        args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.output_root:
        args.output_root = os.path.abspath(os.path.expanduser(args.output_root))
    return args


def prepare_output_directory(args):
    """Reserve one run directory before building; all later stages use this cwd."""
    if args.output_root:
        input_name = os.path.basename(os.path.normpath(args.builtup_dir))
        if not input_name:
            raise ValueError('--builtup-dir must name a folder, not a drive root')
        os.makedirs(args.output_root, exist_ok=True)
        number = 1
        while True:
            suffix = '' if number == 1 else '_run%02d' % number
            output_dir = os.path.join(args.output_root, input_name+suffix)
            try:
                os.mkdir(output_dir)
                return output_dir
            except FileExistsError:
                number += 1
    if args.output_dir:
        if os.path.exists(args.output_dir) and (not os.path.isdir(args.output_dir) or os.listdir(args.output_dir)):
            raise ValueError('--output-dir must be new or empty to avoid mixing results')
        os.makedirs(args.output_dir, exist_ok=True)
        return args.output_dir
    root = os.path.join(SCRIPT_DIR, 'runs')
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=root)


def postprocess_arguments(report):
    # This model uses 1 MPa reference stress, no preload, and global Z as its axis.
    return [report['odb'], '--modes', '0', '--axis', 'z', '--step', 'Buckle',
            '--sigma-ref', str(report['reference_stress_MPa'])]


def require_completed(job, odb_path=None):
    """Use file evidence only when the CAE job status is unavailable.

    Some noGUI sessions return None after waitForCompletion. Never override an
    explicit running/aborted status, or accept a locked/incomplete output.
    """
    status = str(getattr(job, 'status', None)).upper()
    if status == 'COMPLETED':
        return 'CAE job.status=COMPLETED'
    if status not in ('NONE', 'UNKNOWN', '') or not odb_path:
        raise RuntimeError('Analysis status: %s; postprocessing cancelled. Check .msg/.dat/.sta.' % status)
    base = os.path.splitext(os.path.abspath(odb_path))[0]
    if (not os.path.isfile(odb_path) or os.path.getsize(odb_path) == 0 or
            os.path.exists(base+'.lck')):
        raise RuntimeError('Analysis status: %s; ODB missing, empty or locked: %s' % (status, odb_path))
    texts = {}
    for suffix in ('.sta', '.log'):
        try:
            with open(base+suffix, 'r', errors='replace') as stream:
                texts[suffix] = stream.read().upper()
        except OSError:
            texts[suffix] = ''
    combined = '\n'.join(texts.values())
    if any(word in combined for word in ('ABORTED', 'TERMINATED', 'EXITED WITH ERRORS',
                                         'HAS NOT BEEN COMPLETED', 'HAS NOT COMPLETED')):
        raise RuntimeError('Analysis output reports failure; postprocessing cancelled: '+base)
    evidence = []
    if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in texts['.sta']:
        evidence.append('.sta')
    if ('ABAQUS JOB %s COMPLETED' % os.path.basename(base).upper()) in texts['.log']:
        evidence.append('.log')
    if not evidence:
        raise RuntimeError('Analysis status: %s; no successful completion marker for %s' % (status, base))
    return 'Successful completion in '+', '.join(evidence)+'; ODB present and unlocked'


def load_postprocessor():
    import importlib.util
    path = os.path.join(SCRIPT_DIR, 'abaqus_modal_wavelengths.py')
    spec = importlib.util.spec_from_file_location('pipeline_modal_wavelengths', path)
    processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor)
    return processor


def load_enhanced_processor():
    import importlib.util
    path = os.path.join(SCRIPT_DIR, 'abaqus_modal_report.py')
    spec = importlib.util.spec_from_file_location('pipeline_enhanced_report', path)
    processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor)
    return processor


def enhanced_arguments(report):
    return [os.path.splitext(report['odb'])[0]+'_modal_wavelengths_report.json']


def load_portable_exporter():
    import importlib.util
    path = os.path.join(SCRIPT_DIR, 'abaqus_modal_export.py')
    spec = importlib.util.spec_from_file_location('pipeline_modal_export', path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    return exporter


def portable_export_backend(output_format, parquet_python=None):
    if output_format not in ('parquet', 'both'):
        return None
    return load_portable_exporter().resolve_parquet_backend(parquet_python)


def run_portable_export(odb_path, run_dir, output_format='parquet',
                        modes_per_shard=8, parquet_python=None,
                        parquet_backend=None, workers='auto'):
    exporter = load_portable_exporter()
    base = os.path.join(run_dir, 'portable_modal_export')
    output = base
    number = 2
    while os.path.exists(output):
        output = base + '_%02d' % number
        number += 1
    report = exporter.export_odb_path(
        odb_path, output, output_format=output_format,
        modes_per_shard=modes_per_shard, parquet_python=parquet_python,
        parquet_backend=parquet_backend, compression='zstd', compression_level=9,
        workers=workers)
    return dict(output_dir=report['output_dir'], manifest=os.path.join(report['output_dir'], 'modal_export.json'),
                format=report['format'], mode_count=report['mode_count'],
                rotations_available=report['rotations_available'], artifacts=report['artifacts'])


def run_modal_audit(run_dir):
    import importlib.util
    path = os.path.join(SCRIPT_DIR, 'abaqus_dsm_modal_audit.py')
    spec = importlib.util.spec_from_file_location('pipeline_modal_audit', path)
    audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    output = os.path.join(run_dir, 'modal_dsm_audit')
    number = 2
    while os.path.exists(output):
        output = os.path.join(run_dir, 'modal_dsm_audit_%02d' % number); number += 1
    summary = audit.process(audit.parse_arguments(['--run-dir', run_dir, '--output-dir', output]))
    return dict(output_dir=output, html=os.path.join(output, 'modal_explorer.html'),
                eigenspace_validation=os.path.join(output, 'eigenspace_validation.html'),
                mesh_shape_archive=summary.get('mesh_shape_archive'))


def resume_postprocessing(run_dir, modal_audit=False, portable_export='parquet',
                          portable_modes_per_shard=8, parquet_python=None,
                          parquet_backend=None, portable_export_workers='auto'):
    """Recover existing results without requiring source CSVs or a CAE session."""
    from types import SimpleNamespace
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    status_path = os.path.join(run_dir, 'pipeline_status.json')
    with open(status_path) as stream:
        state = json.load(stream)
    report = state['build']
    # Keep the ODB tied to this run even if the output directory was moved.
    report['odb'] = os.path.join(run_dir, os.path.basename(report['odb']))
    evidence = require_completed(SimpleNamespace(status=None), report['odb'])
    processor = load_postprocessor()
    enhanced = load_enhanced_processor()
    previous_dir = os.getcwd()
    os.chdir(run_dir)
    def save(status):
        state['status'] = status
        state['updated_at'] = datetime.datetime.now().isoformat()
        with open(status_path, 'w') as stream:
            json.dump(state, stream, indent=2)
    try:
        if 'error' in state:
            state['recovered_from_error'] = state.pop('error')
        state['solver_api_status'] = state.get('solver_api_status', state.get('solver_status'))
        state['solver_status'] = 'COMPLETED'
        state['completion_evidence'] = evidence
        progress('RESUME: '+evidence+'. No model rebuild or solver submission.')
        if portable_export != 'off':
            save('PORTABLE_EXPORTING')
            progress('PORTABLE EXPORT: raw ODB mode shapes -> '+portable_export+'.')
            state['portable_export'] = run_portable_export(
                report['odb'], run_dir, portable_export, portable_modes_per_shard,
                parquet_python=parquet_python, parquet_backend=parquet_backend,
                workers=portable_export_workers)
        save('POSTPROCESSING')
        progress('3/4 POSTPROCESS: all modes from '+report['odb'])
        state['postprocessing'] = processor.main(postprocess_arguments(report))
        if state['postprocessing']['available_modes'] < state['settings']['n_modes']:
            progress('WARNING: fewer modes available than requested; see modal report.')
        save('ENHANCED_POSTPROCESSING')
        progress('4/4 ENHANCED: mode families, spectra, envelopes and interactive report.')
        state['enhanced_report'] = enhanced.main(enhanced_arguments(report))
        if modal_audit:
            progress('AUDIT: direct shapes, sensitivity and graphical explorer.')
            state['modal_audit'] = run_modal_audit(run_dir)
        save('COMPLETED')
        progress('COMPLETE: recovered postprocessing; %d modes processed.' %
                 state['postprocessing']['processed_modes'])
        return state
    except Exception as error:
        state['error'] = str(error)
        save('FAILED')
        progress('FAILED during resumed postprocessing: '+str(error))
        raise
    finally:
        os.chdir(previous_dir)


def progress(message):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), message)
    print(line)
    sys.stdout.flush()
    with open('pipeline.log', 'a') as stream:
        stream.write(line+'\n')


def monitor_analysis(stop, job_name):
    """File-only monitoring thread; all Abaqus API calls stay on the main thread."""
    offsets = {}
    started = time.time()
    while not stop.wait(10):
        for suffix in ('.sta', '.log'):
            path = job_name+suffix
            try:
                with open(path, 'r') as stream:
                    stream.seek(offsets.get(path, 0))
                    lines = stream.read().splitlines()
                    offsets[path] = stream.tell()
                for line in lines[-8:]:
                    if line.strip():
                        progress('%s: %s' % (suffix, line.strip()))
            except (IOError, OSError):
                pass
        progress('Solver wait: %.0f s elapsed (no reliable percentage available).' % (time.time()-started))


def xykey(point):
    return (round(point[0], 6), round(point[1], 6))


def mandatory_longitudinal_keep(segments, bolt_points=(), sharp_angle_deg=MERGE_ANGLE_DEG):
    """Return section lines that the spacing filter is never allowed to remove.

    The two chain ends, exact bolt-row partitions and genuine sharp source
    corners are hard constraints. Smooth/low-turn lines learned from virtual
    topology remain useful candidates, but a positive spacing request may
    suppress them if they would create an excessively narrow strip.
    """
    points = [segments[0][:2]] + [seg[2:] for seg in segments]
    keep = {xykey(points[0]), xykey(points[-1])}
    keep.update(xykey(point) for point in bolt_points)
    threshold = math.radians(float(sharp_angle_deg))
    for i in range(1, len(points)-1):
        ux, uy = points[i][0]-points[i-1][0], points[i][1]-points[i-1][1]
        vx, vy = points[i+1][0]-points[i][0], points[i+1][1]-points[i][1]
        turn = math.atan2(ux*vy-uy*vx, ux*vx+uy*vy)
        if abs(turn) >= threshold:
            keep.add(xykey(points[i]))
    return keep


def select_longitudinal_lines(segments, legacy_keep, count, min_spacing_mm=0.0,
                              mandatory_keep=None):
    """Keep source vertices, prioritizing curvature while enforcing strip width.

    With min_spacing_mm == 0 this is exactly the historical selector.
    With a positive spacing, the filter applies to every retained SOFT section
    line: both optional refinement lines and low-turn lines inherited from the
    virtual-topology prepass. Only mandatory_keep may violate the spacing; in
    the model builder those are chain ends, exact bolt lines and genuine sharp
    corners.

    Distance is section arclength, not Euclidean chord distance. This is the
    relevant width of the strip created between neighboring longitudinal
    partitions. Count 100 deliberately keeps every source line unchanged.
    """
    legacy_keep = set(legacy_keep)
    if count == 0:
        return legacy_keep
    if not math.isfinite(min_spacing_mm) or min_spacing_mm < 0:
        raise ValueError('min_spacing_mm must be finite and nonnegative')

    points = [segments[0][:2]] + [seg[2:] for seg in segments]
    keys = [xykey(point) for point in points]
    if count == 100:
        return legacy_keep | set(keys)

    arclength = [0.0]
    for a, b in zip(points, points[1:]):
        arclength.append(arclength[-1] + math.hypot(b[0]-a[0], b[1]-a[1]))

    def chain_position(point):
        px, py = point
        best_error = float('inf')
        best_position = None
        for j, (a, b) in enumerate(zip(points, points[1:])):
            vx, vy = b[0]-a[0], b[1]-a[1]
            length2 = vx*vx + vy*vy
            if length2 <= 0:
                continue
            t = ((px-a[0])*vx + (py-a[1])*vy)/length2
            t = min(1.0, max(0.0, t))
            qx, qy = a[0]+t*vx, a[1]+t*vy
            error = math.hypot(px-qx, py-qy)
            if error < best_error:
                best_error = error
                best_position = arclength[j] + t*math.sqrt(length2)
        tolerance = max(1e-5, 1e-8*max(arclength[-1], 1.0))
        return best_position if best_error <= tolerance else None

    turns = [0.0] * len(points)
    regions = []
    for i in range(1, len(points)-1):
        ux, uy = points[i][0]-points[i-1][0], points[i][1]-points[i-1][1]
        vx, vy = points[i+1][0]-points[i][0], points[i+1][1]-points[i][1]
        turns[i] = math.atan2(ux*vy-uy*vx, ux*vx+uy*vy)
        if abs(turns[i]) <= math.radians(0.001):
            continue
        if (not regions or regions[-1][-1] != i-1 or
                turns[regions[-1][-1]] * turns[i] < 0):
            regions.append([])
        regions[-1].append(i)

    adjusted_regions = []
    region_end_keys = set()
    for region in regions:
        region = list(region)
        if region[0] == 1:
            region = [0] + region
        if region[-1] == len(points)-2:
            region = region + [len(points)-1]
        adjusted_regions.append(region)
        region_end_keys.update((keys[region[0]], keys[region[-1]]))

    # Exact backward compatibility when the new spacing option is disabled.
    if min_spacing_mm <= 0:
        keep = set(legacy_keep)
        keep.update((keys[0], keys[-1]))
        keep.update(region_end_keys)
        for region in adjusted_regions:
            first, last = region[0], region[-1]
            interior = region[1:-1]
            selected = [i for i in interior if keys[i] in keep]
            position = {first: 0.0}
            for i in region[1:]:
                position[i] = position[i-1] + (abs(turns[i-1])+abs(turns[i]))/2.0
            candidates = [i for i in interior if keys[i] not in keep]
            while candidates and len(selected) < count:
                anchors = [first, last] + selected
                chosen = max(candidates, key=lambda i: (
                    min(abs(position[i]-position[j]) for j in anchors), -i))
                keep.add(keys[chosen])
                selected.append(chosen)
                candidates.remove(chosen)
        return keep

    # If a caller does not distinguish hard/soft lines, preserve the older API
    # conservatively by treating the supplied legacy set as mandatory.
    hard = set(legacy_keep if mandatory_keep is None else mandatory_keep)
    hard.update((keys[0], keys[-1]))
    keep = set(hard)

    key_index = {key: i for i, key in enumerate(keys)}
    spacing_positions = []
    unmapped_hard = []
    for point in hard:
        position = chain_position(point)
        if position is None:
            unmapped_hard.append(point)
        else:
            spacing_positions.append(position)
    keep.update(unmapped_hard)

    tolerance = 1e-9*max(arclength[-1], 1.0)

    def point_spacing_ok(point):
        position = chain_position(point)
        if position is None:
            return True
        return all(abs(position-other)+tolerance >= min_spacing_mm
                   for other in spacing_positions)

    def accept_point(point):
        keep.add(point)
        position = chain_position(point)
        if position is not None:
            spacing_positions.append(position)

    # Reconsider low-turn lines inherited from virtual topology and region
    # endpoints. Keep the most curvature-sensitive line when several compete
    # for the same < min_spacing_mm neighborhood.
    soft_seed = (legacy_keep | region_end_keys) - hard
    ranked = []
    for point in soft_seed:
        index = key_index.get(point)
        importance = abs(turns[index]) if index is not None else 0.0
        position = chain_position(point)
        ranked.append((-importance, position if position is not None else float('inf'), point))
    for unused_importance, unused_position, point in sorted(ranked):
        if point_spacing_ok(point):
            accept_point(point)

    def spacing_ok_index(index):
        return point_spacing_ok(keys[index])

    for region in adjusted_regions:
        first, last = region[0], region[-1]
        interior = region[1:-1]
        selected = [i for i in interior if keys[i] in keep]
        position = {first: 0.0}
        for i in region[1:]:
            position[i] = position[i-1] + (abs(turns[i-1])+abs(turns[i]))/2.0
        candidates = [i for i in interior if keys[i] not in keep]
        while candidates and len(selected) < count:
            eligible = [i for i in candidates if spacing_ok_index(i)]
            if not eligible:
                break
            anchors = [first, last] + selected
            chosen = max(eligible, key=lambda i: (
                min(abs(position[i]-position[j]) for j in anchors), -i))
            accept_point(keys[chosen])
            selected.append(chosen)
            candidates.remove(chosen)
    return keep


def read_geometry(folder):
    """Read original section coordinates without loading the analysis script."""
    pieces = {}
    thickness = None
    with open(os.path.join(folder, 'builtup_segments.csv'), 'r') as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) < 6:
                raise ValueError('Invalid segment row %d' % line_number)
            values = [float(value) for value in row[:6]]
            if any(math.isnan(value) or math.isinf(value) for value in values):
                raise ValueError('Non-finite segment row %d' % line_number)
            piece_id, x1, y1, x2, y2, t = values
            if piece_id != int(piece_id) or t <= 0:
                raise ValueError('Invalid piece ID or thickness at row %d' % line_number)
            if math.hypot(x2 - x1, y2 - y1) <= 1e-9:
                raise ValueError('Zero-length segment at row %d' % line_number)
            if thickness is None:
                thickness = t
            elif abs(t - thickness) > 1e-9:
                raise ValueError('This script requires a uniform shell thickness')
            segments = pieces.setdefault(int(piece_id), [])
            if segments and math.hypot(segments[-1][2] - x1, segments[-1][3] - y1) > 1e-6:
                raise ValueError('Disconnected segment chain in piece %d' % piece_id)
            segments.append((x1, y1, x2, y2))
    if len(pieces) != 4:
        raise ValueError('Expected four pieces, found %d' % len(pieces))
    with open(os.path.join(folder, 'builtup_member.csv'), 'r') as stream:
        member = next(csv.DictReader(stream))
    return pieces, thickness, float(member['E_MPa']), float(member['nu'])


def read_model_inputs(folder):
    """Validate the exact exported bolt layout; do not reconstruct or round its pitch."""
    pieces, thickness, young, poisson = read_geometry(folder)
    with open(os.path.join(folder, 'builtup_seams.csv')) as f:
        seams = [[float(v) for v in row] for row in csv.reader(f) if row]
    with open(os.path.join(folder, 'builtup_bolts_y.csv')) as f:
        bolts = [float(row[0]) for row in csv.reader(f) if row]
    with open(os.path.join(folder, 'builtup_member.csv')) as f:
        member = next(csv.DictReader(f))
    length = float(member['L_mm'])
    tol = 1e-3
    values = [length, thickness, young, poisson] + bolts + [v for row in seams for v in row]
    if any(math.isnan(v) or math.isinf(v) for v in values) or length <= 0:
        raise ValueError('Model inputs must be finite, and member length must be positive')
    if abs(thickness - float(member['t_mm'])) > 1e-9:
        raise ValueError('Thickness differs between segment and member CSV files')
    if not bolts or len(bolts) != int(member['n_bolts']):
        raise ValueError('Bolt count differs between bolt-position and member CSV files')
    if any(z <= 0 or z >= length for z in bolts) or any(b <= a for a, b in zip(bolts, bolts[1:])):
        raise ValueError('Bolt positions must increase strictly and lie inside the member')
    if abs(bolts[0] - float(member['end_start_mm'])) > tol:
        raise ValueError('First bolt does not match the start end distance')
    if abs(length - bolts[-1] - float(member['end_end_mm'])) > tol:
        raise ValueError('Last bolt does not match the finish end distance')
    pitches = [b-a for a, b in zip(bolts, bolts[1:])]
    if any(abs(p-float(member['pitch_mm'])) > tol for p in pitches):
        raise ValueError('Actual bolt spacings differ from the exported pitch_mm')
    if len(seams) != 4 or sorted(pieces) != [1, 2, 3, 4]:
        raise ValueError('This builder supports the four-piece C4 section')
    if len(set(row[0] for row in seams)) != len(seams):
        raise ValueError('Duplicate seam identifiers')
    for row in seams:
        if len(row) != 7 or any(v != int(v) for v in row[:3]):
            raise ValueError('Invalid seam CSV row')
        sid, pa, pb, xa, ya, xb, yb = row
        if int(pa) not in pieces or int(pb) not in pieces or pa == pb:
            raise ValueError('Invalid pieces in seam %g' % sid)
        # For this exported section, mating lip mid-surfaces are gap + t apart.
        if abs(math.hypot(xb-xa, yb-ya) - float(member['gap_mm']) - thickness) > tol:
            raise ValueError('Seam %g does not match the exported gap and thickness' % sid)
    return pieces, thickness, young, poisson, seams, bolts, member


def input_summary(data):
    pieces, thickness, young, poisson, seams, bolts, member = data
    section_segments = {'P%d' % int(k): [[float(v) for v in seg] for seg in pieces[k]]
                        for k in sorted(pieces)}
    return dict(source_directory=os.path.abspath(BUILTUP_DIR),
        section_segments=section_segments,
        section_geometry_definition='original builtup_segments.csv in global section coordinates',
        length_mm=float(member['L_mm']), thickness_mm=thickness,
        E_MPa=young, nu=poisson, clear_gap_mm=float(member['gap_mm']),
        bolt_row_mm=float(member['bolt_row_mm']), bolts_per_seam=len(bolts),
        start_end_mm=bolts[0], finish_end_mm=float(member['L_mm'])-bolts[-1],
        pitch_mm=[b-a for a, b in zip(bolts, bolts[1:])], bolt_positions_mm=bolts,
        expected_links=len(seams)*len(bolts), connection_model='BEAM_MPC',
        bolt_solids_holes_contact_pretension=False,
        shell_contact='GENERAL_STANDARD_HARD_FRICTIONLESS')


def add_general_contact(model):
    """Define shell contact in Initial, inherited by subsequent steps.

    Keep section thickness/offset defaults. In linear Buckle, the contact
    status remains fixed at the base state, even if plotted modes overlap.
    """
    from abaqusConstants import ON, HARD, DEFAULT, FRICTIONLESS, GLOBAL, SELF
    import interaction
    prop = model.ContactProperty('Hard_Frictionless')
    prop.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON,
                        constraintEnforcementMethod=DEFAULT)
    prop.TangentialBehavior(formulation=FRICTIONLESS)
    contact = model.ContactStd(name='GeneralContact', createStepName='Initial')
    contact.includedPairs.setValuesInStep(stepName='Initial', useAllstar=ON)
    contact.contactPropertyAssignments.appendInStep(stepName='Initial',
        assignments=((GLOBAL, SELF, 'Hard_Frictionless'),))


def build(inputs=None, cpus=None, buckle_output='standard', nodal_precision='full', gpus='auto'):
    import runtime_resources
    policy = runtime_resources.resolve_policy(runtime_resources.detect_resources(), cpus, gpus)
    cpus, gpus = policy.cpus, policy.gpus
    validate_settings()
    if inputs is None:
        inputs = read_model_inputs(BUILTUP_DIR)
    pieces, thickness, young, poisson, seams, bolts, member = inputs
    LENGTH_MM = float(member['L_mm'])
    MODEL_NAME = ('BU_BOLT_L%g_M%g' % (LENGTH_MM, MESH_MM)).replace('.', 'p')
    print('Source inputs: '+json.dumps(input_summary(inputs), sort_keys=True))
    import caeModules  # initialize all CAE repositories for noGUI execution
    from abaqus import mdb, session
    from abaqusConstants import (THREE_D, DEFORMABLE_BODY, ON, OFF, CARTESIAN,
        MIDDLE_SURFACE, FROM_SECTION, XYPLANE, XZPLANE, YZPLANE, QUAD,
        STRUCTURED, FIXED, S4R, STANDARD, SUBSPACE, SET, UNIFORM, GENERAL,
        BEAM_MPC, DOF_MODE_MPC, PERCENTAGE, FULL, SINGLE)
    import mesh
    import regionToolset
    import interaction  # registers Model.MultipointConstraint in noGUI sessions
    import job as job_module

    # Verify exact C4 symmetry before sharing one native meshed part.
    for k in sorted(pieces):
        assert len(pieces[k]) == len(pieces[1])
        for a, b in zip(pieces[1], pieces[k]):
            a = list(a)
            for unused in range(k - 1):
                a = [-a[1], a[0], -a[3], a[2]]
            assert max(abs(x-y) for x, y in zip(a, b)) < 1e-6
    if MODEL_NAME in mdb.models:
        raise RuntimeError('Use a fresh CAE session: model already exists')
    model = mdb.Model(name=MODEL_NAME)
    assert hasattr(model, 'MultipointConstraint') and hasattr(mdb, 'Job')
    assert hasattr(model, 'fieldOutputRequests') and hasattr(model, 'historyOutputRequests')
    job = mdb.Job(name=MODEL_NAME, model=MODEL_NAME, numCpus=cpus, numDomains=cpus,
                  memory=100, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=False,
                  nodalOutputPrecision=FULL if nodal_precision == 'full' else SINGLE)
    model.Material(name='Steel')
    model.materials['Steel'].Elastic(table=((young, poisson),))
    model.HomogeneousShellSection(name='Shell_t', material='Steel', thickness=thickness,
                                 numIntPts=5)
    sketch = model.ConstrainedSketch(name='Section', sheetSize=2000.0)
    for x1, y1, x2, y2 in pieces[1]:
        sketch.Line(point1=(x1, y1), point2=(x2, y2))
    part = model.Part(name='ColumnPiece', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseShellExtrude(sketch=sketch, depth=LENGTH_MM)

    # Learn the significant section boundaries from the existing geometry method.
    original_features = set(part.features.keys())
    part.createVirtualTopology(
        mergeShortEdges=True, shortEdgeThreshold=SMALL_EDGE_MM,
        mergeSmallFaces=True, smallFaceAreaThreshold=SMALL_EDGE_MM * LENGTH_MM,
        ignoreRedundantEntities=True, cornerAngleTolerance=MERGE_ANGLE_DEG)
    keep = set()
    for edge in part.edges:
        vertices = edge.getVertices()
        if len(vertices) == 2:
            a, b = [part.vertices[i].pointOn[0] for i in vertices]
            if xykey(a) == xykey(b):
                keep.add(xykey(a))
    virtual_names = set(part.features.keys()) - original_features
    assert len(virtual_names) == 1, 'Expected one virtual-topology feature'
    part.features[virtual_names.pop()].suppress()
    part.regenerate()

    # Split the flat lips at the precise bolt-row coordinates, before virtual topology.
    bolt_xy = []
    for seam, pa, pb, xa, ya, xb, yb in seams:
        if int(pa) == 1:
            bolt_xy.append((xa, ya))
        if int(pb) == 1:
            bolt_xy.append((xb, yb))
    for x, y in bolt_xy:
        face = part.faces.findAt(((x, y, LENGTH_MM / 2),))
        plane = part.DatumPlaneByPrincipalPlane(
            principalPlane=XZPLANE if abs(x) < abs(y) else YZPLANE,
            offset=y if abs(x) < abs(y) else x)
        part.PartitionFaceByDatumPlane(datumPlane=part.datums[plane.id], faces=face)
        keep.add(xykey((x, y)))

    mandatory_keep = mandatory_longitudinal_keep(
        pieces[1], bolt_points=bolt_xy, sharp_angle_deg=MERGE_ANGLE_DEG)
    keep = select_longitudinal_lines(
        pieces[1], keep, LONGITUDINAL_LINES,
        LONGITUDINAL_LINE_MIN_SPACING_MM, mandatory_keep=mandatory_keep)
    if LONGITUDINAL_LINES == 100:
        # Extrusion merges collinear sketch segments into a single face, but
        # can leave their end vertices. Restore the missing lengthwise cuts
        # so maximum detail also has four-sided, structured-meshable faces.
        native_lines = set()
        for edge in part.edges:
            vertices = edge.getVertices()
            if len(vertices) == 2:
                a, b = [part.vertices[i].pointOn[0] for i in vertices]
                if xykey(a) == xykey(b):
                    native_lines.add(xykey(a))
        for x, y, x2, y2 in pieces[1][1:]:
            if xykey((x, y)) in native_lines:
                continue
            face = part.faces.findAt(((x, y, LENGTH_MM / 2),))
            use_x = abs(x2-x) >= abs(y2-y)
            plane = part.DatumPlaneByPrincipalPlane(
                principalPlane=YZPLANE if use_x else XZPLANE,
                offset=x if use_x else y)
            part.PartitionFaceByDatumPlane(datumPlane=part.datums[plane.id], faces=face)
            native_lines.add(xykey((x, y)))

    cuts = sorted(set(round(z, 7) for z in bolts + [LENGTH_MM / 2]))
    for z in cuts:
        plane = part.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=z)
        part.PartitionFaceByDatumPlane(datumPlane=part.datums[plane.id], faces=part.faces[:])
    print('Partitions ready; simplifying only redundant section boundaries.')
    sys.stdout.flush()

    # Preserve transverse bolt rows and the critical longitudinal boundaries.
    # Ignore both the redundant edges and their end vertices so small original
    # CUFSM segments cannot force tiny mesh seeds along the section.
    ignored_edges, ignored_vertices = [], set()
    for edge in part.edges:
        vertices = edge.getVertices()
        if len(vertices) != 2 or len(edge.getFaces()) != 2:
            continue
        a, b = [part.vertices[i].pointOn[0] for i in vertices]
        if xykey(a) == xykey(b) and xykey(a) not in keep:
            ignored_edges.append(edge)
            ignored_vertices.update(vertices)
    if ignored_edges or ignored_vertices:
        part.ignoreEntity(entities=tuple(ignored_edges) +
                          tuple(part.vertices[i] for i in sorted(ignored_vertices)))
    # Native extrusion can leave extra collinear end vertices with no internal
    # longitudinal edge. Remove these too, leaving four-sided virtual regions.
    redundant = tuple(v for v in part.vertices
                      if len(v.getEdges()) == 2 and xykey(v.pointOn[0]) not in keep)
    if redundant:
        part.ignoreEntity(entities=redundant)
    part.SectionAssignment(region=regionToolset.Region(faces=part.faces[:]),
        sectionName='Shell_t', offset=0.0, offsetType=MIDDLE_SURFACE,
        thicknessAssignment=FROM_SECTION)
    part.setMeshControls(regions=part.faces[:], elemShape=QUAD, technique=STRUCTURED)
    part.setElementType(regions=(part.faces[:],), elemTypes=(
        mesh.ElemType(elemCode=S4R, elemLibrary=STANDARD),))
    part.seedPart(size=MESH_MM, deviationFactor=0.1, minSizeFactor=0.1)
    for edge in part.edges:
        divisions = max(1, int(math.ceil(edge.getSize(printResults=False) / MESH_MM - 1e-6)))
        part.seedEdgeByNumber(edges=(edge,), number=divisions, constraint=FIXED)
    print('Before meshing: %d faces; edge counts %s' %
          (len(part.faces), sorted(set(len(f.getEdges()) for f in part.faces))))
    sys.stdout.flush()
    part.generateMesh()
    print('After meshing: %d nodes, %d elements; unmeshed=%s' %
          (len(part.nodes), len(part.elements), part.getUnmeshedRegions()))
    sys.stdout.flush()
    assert len(part.elements) > 0 and part.getUnmeshedRegions() is None
    assert all(str(e.type) == 'S4R' for e in part.elements), str(set(str(e.type) for e in part.elements))
    print('Mesh: %d faces, %d nodes, %d S4R per piece.' %
          (len(part.faces), len(part.nodes), len(part.elements)))
    sys.stdout.flush()

    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    instances = {}
    for k in sorted(pieces):
        name = 'P%d' % k
        instances[k] = assembly.Instance(name=name, part=part, dependent=ON)
        if k != 1:
            assembly.rotate(instanceList=(name,), axisPoint=(0., 0., 0.),
                            axisDirection=(0., 0., 1.), angle=90.0 * (k-1))
    assembly.regenerate()
    add_general_contact(model)
    model.BuckleStep(name='Buckle', previous='Initial', numEigen=N_MODES,
                    eigensolver=SUBSPACE, maxEigen=None, vectors=N_VECTORS,
                    maxIterations=MAX_ITERATIONS)
    for k, inst in sorted(instances.items()):
        for suffix, z, sign in (('0', 0., 1.), ('L', LENGTH_MM, -1.)):
            edges = inst.edges.getByBoundingBox(xMin=-1e6, yMin=-1e6, zMin=z-1e-4,
                                                xMax=1e6, yMax=1e6, zMax=z+1e-4)
            assert len(edges) > 0
            name = 'END_%s_P%d' % (suffix, k)
            assembly.Set(name=name, edges=edges)
            model.DisplacementBC(name='SS_'+name, createStepName='Initial',
                region=assembly.sets[name], u1=SET, u2=SET, ur3=SET)
            assembly.Surface(name=name+'_S', side1Edges=edges)
            model.ShellEdgeLoad(name='COMP_'+name, createStepName='Buckle',
                region=assembly.surfaces[name+'_S'], magnitude=thickness,
                distributionType=UNIFORM, traction=GENERAL,
                directionVector=((0., 0., 0.), (0., 0., sign)), follower=OFF, resultant=OFF)

    def exact_node(inst, xyz):
        nodes = inst.nodes.getByBoundingSphere(center=xyz, radius=1e-3)
        if len(nodes) != 1:
            raise RuntimeError('Expected one node at %s in %s, found %d' % (xyz, inst.name, len(nodes)))
        return nodes

    start = pieces[1][0]
    axial_nodes = exact_node(instances[1], (start[0], start[1], LENGTH_MM / 2))
    assembly.Set(name='AXIAL_ANCHOR', nodes=axial_nodes)
    model.DisplacementBC(name='AXIAL_FIX', createStepName='Initial',
                         region=assembly.sets['AXIAL_ANCHOR'], u3=SET)
    for seam, pa, pb, xa, ya, xb, yb in seams:
        for j, z in enumerate(bolts, 1):
            name = 'BOLT_S%d_%02d' % (int(seam), j)
            assembly.Set(name=name+'_A', nodes=exact_node(instances[int(pa)], (xa, ya, z)))
            assembly.Set(name=name+'_B', nodes=exact_node(instances[int(pb)], (xb, yb, z)))
            model.MultipointConstraint(name=name, controlPoint=assembly.sets[name+'_A'],
                surface=assembly.sets[name+'_B'], mpcType=BEAM_MPC,
                userMode=DOF_MODE_MPC, userType=0, csys=None)
    assert len(model.constraints) == len(seams)*len(bolts) and len(model.loads) == 8
    assert len(model.boundaryConditions) == 9
    for name in list(model.fieldOutputRequests.keys()):
        del model.fieldOutputRequests[name]
    # Classification/wavelength processing needs the nodal eigenmode shape only.
    # Do not request S/E/SF/SE at every shell section point: those fields are
    # not used by the current Local/Distortional/Global classifier and can
    # dominate ODB size and postprocessing time for hundreds of modes.
    model.FieldOutputRequest(name='ModeShapes', createStepName='Buckle', variables=('U', 'UR'))
    for name in list(model.historyOutputRequests.keys()):
        del model.historyOutputRequests[name]
    job.writeInput(consistencyChecking=ON)
    if session.viewports:
        viewport = session.viewports[session.currentViewportName]
        viewport.setValues(displayedObject=assembly)
        viewport.view.fitView()
    mdb.saveAs(pathName=os.path.join(os.getcwd(), MODEL_NAME+'.cae'))
    report = dict(length_mm=LENGTH_MM, target_mesh_mm=MESH_MM, modes=N_MODES,
        longitudinal_lines=LONGITUDINAL_LINES,
        longitudinal_line_min_spacing_mm=LONGITUDINAL_LINE_MIN_SPACING_MM,
        mandatory_longitudinal_lines=len(mandatory_keep),
        retained_section_lines=len(keep),
        vectors=N_VECTORS, max_iterations=MAX_ITERATIONS, cpus=cpus,
        job_name=MODEL_NAME, odb=os.path.abspath(MODEL_NAME+'.odb'),
        nodes=4*len(part.nodes), elements=4*len(part.elements), element_type='S4R',
        bolts_per_seam=len(bolts), rigid_links=len(model.constraints),
        boundary_conditions=len(model.boundaryConditions), loads=len(model.loads),
        reference_stress_MPa=1.0, submitted=False)
    report['contact'] = dict(type='General contact (Standard)', step='Initial',
        domain='All exterior surfaces, including self-contact', normal='HARD',
        tangential='FRICTIONLESS', allow_separation=True,
        buckle_limitation='Contact status fixed at the base state')
    report['source_inputs'] = input_summary(inputs)
    report['modal_output'] = dict(
        profile='classification_only', requested_legacy_profile=buckle_output,
        nodal_precision=nodal_precision, fields=['U', 'UR'],
        mode_shape_components='Global U and UR retained; U drives current screening while UR preserves portable/mFSM-ready nodal rotations',
        deliberately_omitted_fields=['S', 'E', 'SF', 'SE'],
        convention='Normalized perturbation mode shapes only; shell stress/strain/force energy diagnostics are intentionally not stored in the automatic buckling stage')
    with open(MODEL_NAME+'_build.json', 'w') as f:
        json.dump(report, f, indent=2)
    print('BUILD COMPLETE: CAE and INP saved. No analysis submitted yet. '+str(report))
    sys.stdout.flush()
    return job, report


def main(argv=None):
    global BUILTUP_DIR, MESH_MM, N_MODES, N_VECTORS, MAX_ITERATIONS
    global LONGITUDINAL_LINES, LONGITUDINAL_LINE_MIN_SPACING_MM
    args = parse_arguments(argv)
    parquet_backend = None
    if (args.portable_export in ('parquet', 'both') and
            not args.build_only and not args.check_inputs):
        parquet_backend = portable_export_backend(args.portable_export, args.parquet_python)
    if args.resume_post:
        return resume_postprocessing(
            args.resume_post, modal_audit=args.modal_audit,
            portable_export=args.portable_export,
            portable_modes_per_shard=args.portable_modes_per_shard,
            parquet_python=args.parquet_python, parquet_backend=parquet_backend,
            portable_export_workers=args.portable_export_workers)
    BUILTUP_DIR, MESH_MM = args.builtup_dir, args.mesh_mm
    LONGITUDINAL_LINES = args.longitudinal_lines
    LONGITUDINAL_LINE_MIN_SPACING_MM = args.longitudinal_line_min_spacing_mm
    N_MODES, N_VECTORS, MAX_ITERATIONS = args.n_modes, args.n_vectors, args.max_iterations
    validate_settings()
    inputs = read_model_inputs(BUILTUP_DIR)
    settings = vars(args).copy()
    settings['effective_buckle_output'] = 'classification_U_plus_portable_UR'
    settings['automatic_shell_energy'] = False
    if parquet_backend is not None:
        settings['portable_parquet_backend'] = parquet_backend
    if args.check_inputs:
        print(json.dumps(dict(settings=settings, source_inputs=input_summary(inputs)), indent=2))
        return
    # Resolve the local postprocessor before submitting an expensive analysis.
    processor = None
    enhanced = None
    if not args.build_only and not args.skip_post:
        processor = load_postprocessor()
        enhanced = load_enhanced_processor()
    output_dir = prepare_output_directory(args)
    previous_dir = os.getcwd()
    os.chdir(output_dir)
    settings['output_dir'] = output_dir
    state = dict(status='BUILDING', settings=settings, source_inputs=input_summary(inputs), submitted=False)
    def save_state(status):
        state['status'] = status
        state['updated_at'] = datetime.datetime.now().isoformat()
        with open('pipeline_status.json', 'w') as stream:
            json.dump(state, stream, indent=2)
    try:
        save_state('BUILDING')
        # Eigenvalue buckling supports solver parallelism, not element-loop parallelism.
        with open('abaqus_v6.env', 'w') as stream:
            stream.write('standard_parallel = SOLVER\n')
            stream.write('gpus = %d\n' % args.gpus)
            stream.write('memory = \"100%%\"\ncpus = %d\n' % args.cpus)
        progress('Output directory: '+output_dir)
        progress('Settings: '+json.dumps(settings, sort_keys=True))
        progress('1/4 BUILD: geometry, mesh, contact, CAE and INP.')
        if args.buckle_output == 'detailed':
            progress('NOTE: --buckle-output detailed is retained only for command compatibility; S/E/SF/SE are no longer requested in this buckling stage.')
        job, report = build(inputs=inputs, cpus=args.cpus,
                            buckle_output=args.buckle_output, nodal_precision=args.nodal_precision, gpus=args.gpus)
        state['build'] = report
        save_state('BUILT')
        if args.build_only:
            progress('BUILD_ONLY complete. CAE and INP saved; solver not submitted.')
            return state
        from abaqusConstants import ON
        progress('2/4 SOLVE: submitting '+job.name)
        job.submit(consistencyChecking=ON)
        report['submitted'] = True
        state['submitted'] = True
        with open(job.name+'_build.json', 'w') as stream:
            json.dump(report, stream, indent=2)
        save_state('SOLVING')
        stop = threading.Event()
        monitor = threading.Thread(target=monitor_analysis, args=(stop, job.name))
        monitor.daemon = True
        monitor.start()
        try:
            job.waitForCompletion()
        finally:
            stop.set()
            monitor.join(timeout=2)
        state['solver_api_status'] = str(job.status)
        state['solver_status'] = str(job.status)
        state['completion_evidence'] = require_completed(job, report['odb'])
        state['solver_status'] = 'COMPLETED'
        if not os.path.isfile(report['odb']):
            raise RuntimeError('Solver completed but expected ODB is missing: '+report['odb'])
        save_state('SOLVED')
        progress('Solver completed successfully: '+state['completion_evidence'])
        if args.portable_export != 'off':
            save_state('PORTABLE_EXPORTING')
            progress('PORTABLE EXPORT: raw ODB mode shapes -> '+args.portable_export+'.')
            state['portable_export'] = run_portable_export(
                report['odb'], output_dir, args.portable_export,
                args.portable_modes_per_shard, parquet_python=args.parquet_python,
                parquet_backend=parquet_backend, workers=args.portable_export_workers)
            save_state('SOLVED')
        if processor is not None:
            save_state('POSTPROCESSING')
            progress('3/4 POSTPROCESS: all modes from '+report['odb'])
            state['postprocessing'] = processor.main(postprocess_arguments(report))
            if state['postprocessing']['available_modes'] < N_MODES:
                progress('WARNING: fewer modes available than requested; see modal report.')
            save_state('ENHANCED_POSTPROCESSING')
            progress('4/4 ENHANCED: mode families, spectra, envelopes and interactive report.')
            state['enhanced_report'] = enhanced.main(enhanced_arguments(report))
            if args.modal_audit:
                progress('AUDIT: direct shapes, sensitivity and graphical explorer.')
                state['modal_audit'] = run_modal_audit(output_dir)
        else:
            progress('3/4 and 4/4 POSTPROCESS skipped by --skip-post.')
        save_state('COMPLETED')
        progress('COMPLETE: '+output_dir)
        return state
    except Exception as error:
        state['error'] = str(error)
        save_state('FAILED')
        progress('FAILED: '+str(error))
        raise
    finally:
        os.chdir(previous_dir)


if __name__ == '__main__':
    main()

