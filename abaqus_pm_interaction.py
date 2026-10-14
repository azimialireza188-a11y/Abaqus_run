# -*- coding: utf-8 -*-
"""P-M interaction of the bolted built-up section: Abaqus inputs for elastic buckling and GMNIA.

Every P-M point is a PROPORTIONAL end load along a ray of angle psi in the normalized
(P/P_y, M/M_y) plane about a neutral-axis direction theta (abaqus_load_case.py):
    sigma(x, y) = sigma_ref * s * (cos(psi) + sin(psi) * eta / c),  s = 1/(cos(psi)+sin(psi)),
psi = 0 pure compression, psi = 90 pure bending, compression on the +n side of the neutral
axis, c = extreme CUFSM centre-line fibre. The extreme compression fibre carries sigma_ref,
so LPF = 1 with sigma_ref = fy is the first-order first-yield point P/P_y + M/M_y = 1.
Elastic buckling (mode buckling) applies the stress as the consistent nodal forces of the edge
traction on the meshed end edges (CUFSM-consistent reference, free warping), with the boundary
conditions, mesh, bolts and contact of the reference model. GMNIA (mode gmnia) uses the same
consistent nodal end forces and inherited supports as the supplied working Step5 reference,
with free end warping (abaqus_pipeline_contract.STEP5_END_CONDITION). The loaded-end stress
distribution is prescribed; reported capacity belongs to this loading/boundary model.

Modes (one entry point):
  section   Section references only (no Abaqus): P_y, M_y, M_p and the full-plastic P-M curve
            of the centre-line section for every requested axis -> pm_section_reference.*
            python abaqus_pm_interaction.py --mode section --run-dir RUN --output-dir OUT
  buckling  Elastic P-M interaction: one *BUCKLE input per ray, derived from the reference
            Step-3 CAE (Lanczos, eigenvalues >= 0). Eigenvalue = critical extreme-fibre stress
            of the ray, P_cr = lambda*P_ref, M_cr = lambda*M_ref. Each ray gets its own folder
            (<job>.inp + <job>_build.json, so GDLC can classify the solved modes) and one BAT
            solves all of them with every physical core.
            abaqus cae noGUI=abaqus_pm_interaction.py -- --mode buckling --run-dir RUN --output-dir OUT
                [--cpus N]   (Abaqus CPUs of the buckling jobs; default auto = all physical cores)
  gmnia     Inelastic P-M interaction: Riks GMNIA inputs for every ray and every selected Step-4
            imperfection case (same imperfect specimen for all rays), built like Step 5 (plasticity,
            assembled-BEAM bolts, outputs, 70 % post-peak stop) and run by the Step-5 queue.
            abaqus cae noGUI=abaqus_pm_interaction.py -- --mode gmnia --source-cae STEP4.cae
                --output-dir OUT --fy 240 [all abaqus_step5_gmnia.py options]
  collect   P-M diagram from finished results (no Abaqus): STEP5_PM_points.csv of P-M, Step-5
            compression and Step-5 bending folders, elastic eigenvalues of buckling folders and
            the section references -> PM_interaction_points.csv + one plot per axis.
            python abaqus_pm_interaction.py --mode collect --roots DIR [DIR ...] --output-dir OUT

Rays: --axes required|all|<deg,...> (required = one axis per symmetry class, e.g. 0 and 45 for
the four-piece square), --pm-angles psi list (default 0,15,30,45,60,75,90). psi = 0 does not
depend on the axis and is generated once. One CAE model is prepared per imperfection case;
the rays are written from its INP by replacing only the magnitudes of the placeholder loads
(CF3 of every end node for both GMNIA and buckling), so
every ray of a case has an identical mesh, imperfection, material, bolts and step.
Units: N, mm, MPa. No analysis is submitted.
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import abaqus_load_case as load_cases
import abaqus_pipeline_contract as contract
import runtime_resources

DEFAULT_PM_ANGLES = (0., 15., 30., 45., 60., 75., 90.)
MAX_ABAQUS_NAME = 38
SECTION_FILE = 'pm_section_reference.json'
MANIFEST_FILE = 'pm_manifest.json'


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
GMNIA_MODELING_SCOPE = dict(
    load_path=('proportional P and M from zero, applied as consistent fixed-direction nodal CF3 '
               'of the prescribed end stress field; no axial preload'),
    moment_measure='applied first-order end moment',
    reference_supports=contract.STEP5_END_CONDITION['supports'],
    connections='rigid assembled BEAM connectors; no slip, bearing compliance or bolt failure',
    imperfections='one unchanged stress-free Step4 geometry for all rays of each case',
    axis_scope='required/all use perfect-section symmetry only; specimen equivalence is not established',
    capacity_scope='prescribed specimen and load path, not a worst-imperfection or arbitrary moment-gradient envelope',
    validation='generation does not establish mesh/arc-length convergence, experimental accuracy or connection adequacy')

PM_SELECTION_FIELDS = ['selection_kind','capacity_verified','provisional','frame','lpf',
                       'selection_response','selection_policy','selection_diagnostics']

def _strip_cae_arguments(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' in argv:
        return argv[argv.index('--')+1:]
    if '-cae' in argv:
        clean, i = [], 0
        while i < len(argv):
            if argv[i] == '-cae':
                i += 1
            elif argv[i] in ('-noGUI', '-tmpdir', '-lmlog'):
                i += 2
            else:
                clean.append(argv[i]); i += 1
        return clean
    return argv


def parse_float_list(text, name):
    try:
        values = [float(v) for v in str(text).replace(' ', '').split(',') if v != '']
    except ValueError:
        raise ValueError('%s must be a comma-separated list of numbers' % name)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('%s must contain finite numbers' % name)
    return values


def parse_arguments(argv=None):
    argv = _strip_cae_arguments(argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument('--mode', choices=('section', 'buckling', 'gmnia', 'collect'), required=True)
    p.add_argument('--run-dir', help='Reference Step-3 run (section, buckling; gmnia: only if the run moved)')
    p.add_argument('--axes', default='required',
                   help="Neutral-axis angles: 'required' (perfect-section representatives), 'all' (+ optional "
                        "intermediate axes) or degrees. GMNIA accepts equivalent section directions, e.g. "
                        "0,90,180,270, since imperfections may break section symmetry")
    p.add_argument('--pm-angles', default=','.join('%g' % a for a in DEFAULT_PM_ANGLES),
                   help='Ray angles psi in [0, 90] deg in the (P/Py, M/My) plane; 0 = compression, 90 = bending')
    p.add_argument('--n-modes', type=int, default=20, help='buckling: eigenvalues per ray (Lanczos, >= 0)')
    p.add_argument('--roots', nargs='+', help='collect: folders searched recursively for results')
    p.add_argument('--normalized', action='store_true',
                   help='collect: use P/Py and M/My instead of dimensional P (kN) and M (kN m)')
    p.add_argument('--fy', type=float, help='section/collect: yield stress for absolute capacities (MPa)')
    p.add_argument('--output-dir')
    p.add_argument('--cpus', type=runtime_resources.cpu_request, default='auto',
                   help="Abaqus CPUs (buckling, gmnia): 'auto' (all physical cores) or a positive integer")
    p.add_argument('--gpus', type=runtime_resources.gpu_request, default='auto',
                   help="Abaqus GPUs (buckling, gmnia): 'auto' (all visible) or an integer >= 0")
    args, rest = p.parse_known_args(argv)
    if args.normalized and args.mode!='collect':
        p.error('--normalized applies only to --mode collect')
    try:
        args.pm_angles = sorted(set(parse_float_list(args.pm_angles, '--pm-angles')))
    except ValueError as error:
        p.error(str(error))
    if any(not 0. <= a <= 90. for a in args.pm_angles):
        p.error('--pm-angles must lie in [0, 90] degrees')
    if args.axes not in ('required', 'all'):
        try:
            args.axes = sorted(set(load_cases.normalize_angle(a) for a in parse_float_list(args.axes, '--axes')))
        except ValueError as error:
            p.error(str(error))
    if args.fy is not None and (not math.isfinite(args.fy) or args.fy <= 0):
        p.error('--fy must be positive and finite')
    if args.n_modes < 1:
        p.error('--n-modes must be positive')
    if args.mode == 'gmnia':
        # every Step-5 option (fy, Riks controls, stop method, material, models, cpus) is
        # parsed and validated by the Step-5 parser itself
        import abaqus_step5_gmnia as step5
        step5_argv = list(rest) + (['--fy', repr(args.fy)] if args.fy is not None else [])
        step5_argv += ['--cpus', str(args.cpus), '--gpus', str(args.gpus)]
        if args.output_dir:
            step5_argv += ['--output-dir', args.output_dir]
        args.step5 = step5.parse_arguments(step5_argv)
        args.output_dir = args.step5.output_dir
        args.fy = args.step5.fy
    else:
        if rest:
            p.error('Unrecognized arguments for --mode %s: %s' % (args.mode, ' '.join(rest)))
        if not args.output_dir:
            p.error('--output-dir is required')
        args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.mode in ('section', 'buckling') and not args.run_dir:
        p.error('--run-dir is required for --mode %s' % args.mode)
    if args.mode == 'collect' and not args.roots:
        p.error('--roots is required for --mode collect')
    if args.run_dir:
        args.run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    if args.mode == 'buckling':
        policy = runtime_resources.resolve_solver_policy(runtime_resources.detect_resources(), args.cpus, args.gpus)
        args.cpus, args.gpus = policy.cpus, policy.gpus
    return args


# --------------------------------------------------------------------------
# Rays and section references (pure Python)
# --------------------------------------------------------------------------
def resolve_axes(spec, segments, symmetry=None, check_symmetry=True):
    symmetry = symmetry if symmetry is not None else load_cases.section_symmetry(segments)
    axes_info = load_cases.required_bending_axes(symmetry)
    if spec == 'required':
        axes = list(axes_info['required_axes_deg'])
    elif spec == 'all':
        axes = sorted(set(axes_info['required_axes_deg']) | set(axes_info['optional_intermediate_axes_deg']))
    else:
        axes = [load_cases.normalize_angle(a) for a in spec]
    if len(set(axes)) != len(axes):
        raise ValueError('Duplicate bending directions after angle normalization')
    classes = {}
    for theta in axes:
        canonical = load_cases.canonical_axis(theta, symmetry)
        if check_symmetry and canonical in classes:
            raise ValueError('Axes %g and %g are equivalent by section symmetry; keep one'
                             % (classes[canonical], theta))
        classes[canonical] = theta
    return axes, dict(symmetry=symmetry, axes_for_PM=axes_info)


def ray_list(axes, psis):
    """[(name, theta, psi)]; psi = 0 (pure compression) is independent of the axis: generated once."""
    rays = []
    if 0. in psis:
        rays.append(('PM_P00', axes[0], 0.))
    for theta in axes:
        for psi in psis:
            if psi > 0.:
                rays.append(('PM_A%s_P%s' % (load_cases.angle_token(theta), load_cases.pm_token(psi)), theta, psi))
    return rays


def section_reference(segments, thickness, axes, fy=None, points=181):
    """P_y, M_y, M_p and the full-plastic P-M curve per MPa of yield stress (centre-line section)."""
    props = load_cases.section_properties(segments, thickness)
    result = dict(definition=('centre-line thin-walled section; per MPa of yield stress; P_y = A*fy, '
                              'M_y = fy*I/c (extreme centre-line fibre, CUFSM convention), M_p = fy*Z; '
                              'plastic curve: rectangular stress blocks about a shifted plastic neutral '
                              'axis, no local buckling, no residual stress'),
                  area_mm2=props['area_mm2'], fy_MPa=fy, axes={})
    for theta in axes:
        bending = load_cases.bending_reference(segments, thickness, theta)
        curve = load_cases.plastic_interaction(segments, thickness, theta, points)
        result['axes'][load_cases.angle_token(theta)] = dict(
            neutral_axis_angle_deg=load_cases.normalize_angle(theta),
            c_extreme_mm=bending['c_extreme_mm'],
            squash_load_N_per_MPa=props['area_mm2'],
            first_yield_moment_Nmm_per_MPa=bending['reference_moment_Nmm_per_MPa'],
            plastic_moment_Nmm_per_MPa=bending['plastic_modulus_mm3'],
            plastic_curve_N_Nmm_per_MPa=[[p, m] for unused_d, p, m in curve])
    return result


def write_section_reference(directory, reference):
    with open(os.path.join(directory, SECTION_FILE), 'w', encoding='utf-8') as stream:
        json.dump(reference, stream, indent=2)
    path = os.path.join(directory, 'pm_section_reference.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['axis_deg', 'P_over_Py', 'M_over_My', 'M_over_Mp', 'P_N_per_MPa', 'M_Nmm_per_MPa'])
        for data in reference['axes'].values():
            for p_unit, m_unit in data['plastic_curve_N_Nmm_per_MPa']:
                writer.writerow([data['neutral_axis_angle_deg'], p_unit/data['squash_load_N_per_MPa'],
                                 m_unit/data['first_yield_moment_Nmm_per_MPa'],
                                 m_unit/data['plastic_moment_Nmm_per_MPa'], p_unit, m_unit])
    return [SECTION_FILE, 'pm_section_reference.csv']


# --------------------------------------------------------------------------
# Nodal forces of one ray on the template end nodes (pure Python)
# --------------------------------------------------------------------------
def ray_forces(template, thickness, definition):
    """Unit (1 MPa) consistent nodal forces of the ray, by load name, with the resultants.

    template = abaqus_step5_gmnia.pm_template_nodes(...): end nodes (load/set name, instance,
    label, side, sign, x, y) and end edges [instance, side, a, b]."""
    coords, by_key = {}, {}
    for node in template['nodes']:
        key = (node['instance'], node['side'], node['label'])
        coords[key] = (node['x'], node['y'])
        by_key[key] = node
    groups = {}
    for instance, side, a, b in template['edges']:
        groups.setdefault((instance, side), []).append((a, b))
    unit = {}
    for (instance, side), edges in groups.items():
        local = dict((label, coords[(instance, side, label)]) for edge in edges for label in edge)
        for label, value in load_cases.consistent_edge_forces(local, edges, thickness, definition).items():
            unit[(instance, side, label)] = value
    missing = set(by_key) - set(unit)
    if missing:
        raise ValueError('End nodes without an end edge: %s' % sorted(missing)[:5])
    nx, ny = definition['compression_normal']
    tx, ty = definition['neutral_axis_direction']
    xc, yc = definition['centroid_mm']
    totals = [dict(force=0., abs_force=0., moment=0., secondary=0.) for unused in (0, 1)]
    forces = {}
    for key, value in unit.items():
        node = by_key[key]
        x, y = node['x'], node['y']
        side = totals[node['side']]
        side['force'] += value
        side['abs_force'] += abs(value)
        side['moment'] += value*(nx*(x-xc)+ny*(y-yc))
        side['secondary'] += value*(tx*(x-xc)+ty*(y-yc))
        forces[node['load']] = node['sign']*value
    for side in totals:
        if abs(side['secondary']) > 1e-6*max(side['abs_force']*definition['c_extreme_mm'], 1e-300):
            raise ValueError('P-M end forces carry a moment about the other axis (centroid/mesh mismatch)')
    if not (math.isclose(totals[0]['force'], totals[1]['force'], rel_tol=1e-9, abs_tol=1e-9*totals[0]['abs_force'])
            and math.isclose(totals[0]['moment'], totals[1]['moment'], rel_tol=1e-9,
                             abs_tol=1e-9*totals[0]['abs_force']*definition['c_extreme_mm'])):
        raise ValueError('P-M reference resultants differ between the two ends')
    axial, moment = totals[0]['force'], totals[0]['moment']
    if abs(axial) <= 1e-12*totals[0]['abs_force']:
        axial = 0.
    if abs(moment) <= 1e-12*totals[0]['abs_force']*definition['c_extreme_mm']:
        moment = 0.
    return forces, dict(axial_force_N_per_MPa=axial, moment_Nmm_per_MPa=moment,
                        secondary_moment_relative=max(abs(s['secondary'])/max(abs(s['moment']), s['abs_force'] *
                                                      definition['c_extreme_mm']*1e-12) for s in totals))


def rotation_weights(template, thickness, segments, theta):
    """Work-conjugate end-rotation coefficients k_i = F_i/M_ref of the pure-bending field."""
    bending = load_cases.bending_reference(segments, thickness, theta)
    forces, totals = ray_forces(template, thickness, bending)
    m_unit = totals['moment_Nmm_per_MPa']
    if not m_unit > 0:
        raise ValueError('Degenerate bending reference on the mesh')
    weights = [[], []]
    for node in template['nodes']:
        value = forces[node['load']]
        if value != 0.0:
            weights[node['side']].append([node['instance'], node['label'], value/m_unit])
    return weights, m_unit


# --------------------------------------------------------------------------
# INP variants: replace only the CF3 magnitudes of the placeholder end loads
# --------------------------------------------------------------------------
_LOAD_NAME = re.compile(r'^\*\*\s*Name:\s*(\S+)\s+Type:\s*(?:Concentrated force|Moment)', re.I)


def set_cload_values(text, values, sets):
    """Replace the magnitude of every '*Cload' data line of the named placeholder loads.

    values: {load name: CF3} or {load name: {dof: value}} (end-plate moments: {4: CM1, 5: CM2}),
    sets: {load name: (node-set name, instance, node label)}; instance/label None for a
    reference-point set. Every listed load must occur exactly once, with exactly one data line
    per listed DOF on its own region (written by CAE as the set name or as instance.label);
    nothing else changes."""
    lines = text.split('\n')
    pending, done = None, set()
    i = 0
    while i < len(lines):
        match = _LOAD_NAME.match(lines[i].strip())
        if match and match.group(1) in values:
            pending = match.group(1)
        elif pending is not None and lines[i].strip().upper().startswith('*CLOAD'):
            j = i+1
            data = []
            while j < len(lines) and not lines[j].lstrip().startswith('*'):
                if lines[j].strip():
                    data.append(j)
                j += 1
            wanted = values[pending] if isinstance(values[pending], dict) else {3: values[pending]}
            wanted = dict((int(dof), float(value)) for dof, value in wanted.items())
            if len(data) != len(wanted):
                raise ValueError('Placeholder load %s must have %d *Cload data line(s)' % (pending, len(wanted)))
            set_name, instance, label = sets[pending]
            regions = [set_name.upper()] + ([('%s.%d' % (instance, label)).upper()] if instance is not None else [])
            seen = set()
            for index in data:
                fields = [f.strip() for f in lines[index].split(',')]
                region = fields[0].upper() if fields else ''
                dof = int(fields[1]) if len(fields) == 3 and fields[1].isdigit() else None
                if dof not in wanted or dof in seen or region not in regions:
                    raise ValueError('Unexpected *Cload data for %s: %r' % (pending, lines[index]))
                seen.add(dof)
                lines[index] = '%s, %d, %.17g' % (fields[0], dof, wanted[dof])
            if pending in done:
                raise ValueError('Placeholder load written twice: '+pending)
            done.add(pending)
            pending = None
            i = j
            continue
        elif pending is not None and lines[i].lstrip().startswith('*') and not lines[i].lstrip().startswith('**'):
            raise ValueError('Load %s is not followed by *Cload' % pending)
        i += 1
    missing = set(values) - done
    if missing:
        raise ValueError('%d placeholder loads not found in the INP, e.g. %s' % (len(missing), sorted(missing)[:3]))
    return '\n'.join(lines)


def check_name(name):
    if len(name) > MAX_ABAQUS_NAME or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
        raise ValueError('Abaqus model/job name %r is invalid or longer than %d characters; '
                         'use shorter Step-4 case names or angles with fewer decimals' % (name, MAX_ABAQUS_NAME))
    return name


# --------------------------------------------------------------------------
# Reference run (section geometry) helpers
# --------------------------------------------------------------------------
def reference_section(run_dir, expected_job=None):
    status_path = os.path.join(run_dir, contract.STATUS_FILE)
    if not os.path.isfile(status_path):
        raise ValueError('Reference pipeline status not found: '+status_path+
                         ' (pass --run-dir with the Step-3 run folder)')
    with open(status_path, encoding='utf-8') as stream:
        state = json.load(stream)
    validated = contract.validate_reference_state(state)
    if expected_job and validated['job_name'] != expected_job:
        raise ValueError('Run %s holds job %s, not the Step-4 source job %s'
                         % (run_dir, validated['job_name'], expected_job))
    source = validated['source_inputs']
    return state, validated, source['section_segments'], float(source['thickness_mm'])


# --------------------------------------------------------------------------
# Mode: section
# --------------------------------------------------------------------------
def run_section(args):
    unused_state, unused_validated, segments, thickness = reference_section(args.run_dir)
    recorded_symmetry = unused_validated['build'].get('load_case', {}).get('section_symmetry')
    if unused_validated['source_inputs'].get('steel_plates') and not recorded_symmetry:
        raise ValueError('Steel plate reference lacks full-model symmetry; rebuild the reference')
    axes, symmetry = resolve_axes(args.axes, segments, recorded_symmetry)
    reference = section_reference(segments, thickness, axes, args.fy)
    reference.update(symmetry)
    os.makedirs(args.output_dir, exist_ok=True)
    files = write_section_reference(args.output_dir, reference)
    print('SECTION P-M REFERENCE: axes %s -> %s' % (axes, ', '.join(files)))
    return reference


# --------------------------------------------------------------------------
# Mode: gmnia (Abaqus/CAE)
# --------------------------------------------------------------------------
def run_gmnia(args):
    import abaqus_step5_gmnia as step5
    from abaqus_step5_progress import report
    s5 = args.step5
    report('P-M GMNIA', 'Loading Abaqus modules')
    database, selected, perfect_source = step5.open_step4_database(s5)
    try:
        # Reference provenance: the untouched model (kept only when PERFECT is selected) or a Step-4 case.
        if step5.PERFECT_TOKEN in selected:
            manifest = step5.reference_step4_manifest(database.models[perfect_source])
        else:
            manifest = contract.validate_step4_payload(contract.parse_prefixed_json(
                database.models[selected[0]].description, step5.STEP4_PREFIX))[0]
        snapshot = manifest['reference_pipeline']
        run_dir = args.run_dir or snapshot.get('run_dir')
        unused_state, unused_validated, segments, thickness = reference_section(run_dir, snapshot.get('job_name'))
        if not math.isclose(thickness, float(snapshot['thickness_mm']), rel_tol=1e-9):
            raise ValueError('Reference run thickness differs from the Step-4 provenance')
        recorded_symmetry = snapshot.get('load_case', {}).get('section_symmetry')
        if snapshot.get('steel_plates') and not recorded_symmetry:
            raise ValueError('Steel plate reference lacks full-model symmetry; rebuild the reference')
        axes, symmetry = resolve_axes(args.axes, segments, recorded_symmetry, check_symmetry=False)
        rays = ray_list(axes, args.pm_angles)
        report('P-M RAYS', '%d rays x %d case(s): axes %s, psi %s' %
               (len(rays), len(selected), axes, args.pm_angles))
        scratch = tempfile.mkdtemp(prefix='cfs_pm_')
        previous = os.getcwd()
        outputs, job_names, entries = [], [], []
        final_user_subroutine = os.path.join(s5.output_dir, step5.STOP_SUBROUTINE_NAME)
        try:
            os.chdir(scratch)
            outputs += step5.write_solver_environment(s5)
            definitions = dict((ray[0], load_cases.combined_reference(segments, thickness, ray[1], ray[2]))
                               for ray in rays)
            rotation_cache = {}
            for i, source_name in enumerate(selected, 1):
                case = 'PERFECT' if source_name == step5.PERFECT_TOKEN else source_name[len('STEP4_'):]
                template_name = check_name('PMT_'+case)
                for ray_name, unused_theta, unused_psi in rays:
                    check_name(ray_name+'_'+case)
                report('TEMPLATE %d/%d' % (i, len(selected)), template_name+' from '+source_name)
                model = step5.copy_source_model(database, source_name, template_name, perfect_source)
                info = step5.prepare_model(model, s5, pm_template=True)
                step5.write_job_input(database, template_name, info, s5, final_user_subroutine)
                with open(template_name+'.inp') as stream:
                    template_text = stream.read()
                template = info.pop('pm_template')
                sets = {node['load']: (node['set'], node['instance'], node['label'])
                        for node in template['nodes']}
                section_mesh = dict(area_mm2=info['end_area_mm2'])
                for ray_name, theta, psi in rays:
                    name = ray_name+'_'+case
                    definition = definitions[ray_name]
                    unit, totals = ray_forces(template, thickness, definition)
                    key = (case, load_cases.angle_token(theta))
                    if key not in rotation_cache:
                        rotation_cache[key] = rotation_weights(template, thickness, segments, theta)
                    weights, m_bend = rotation_cache[key]
                    forces = {load: value*s5.reference_stress for load, value in unit.items()}
                    text = set_cload_values(template_text, forces, sets)
                    with open(name+'.inp', 'w') as stream:
                        stream.write(text)
                    step5.verify_input(name+'.inp', len(info['bolts']), bending=True)
                    p_ref = totals['axial_force_N_per_MPa']*s5.reference_stress
                    m_ref = totals['moment_Nmm_per_MPa']*s5.reference_stress
                    variant = dict(info)
                    variant.pop('reference_force_N_per_end', None)
                    variant.update(
                        source_case=case,
                        load_case=dict((k, v) for k, v in definition.items()),
                        bending_axis_deg=load_cases.normalize_angle(theta) if psi > 0 else None,
                        pm_angle_deg=psi,
                        pm_modeling_scope=dict(GMNIA_MODELING_SCOPE),
                        reference_axial_force_N=p_ref, reference_moment_Nmm=m_ref,
                        end_rotation_weights=weights,
                        pm_formula=('P = LPF*reference_axial_force_N, M = LPF*reference_moment_Nmm (proportional, '
                                    'no preload); LPF = 1: extreme-fibre stress = reference_stress'),
                        rotation_formula=('theta = sum k_i*U3_i over both ends, k_i = F_i/M_ref of the pure-bending '
                                          'field about the same axis (+ = compression fibre shortens)'),
                        section_for_PM=dict(
                            area_mm2=section_mesh['area_mm2'], squash_load_N=section_mesh['area_mm2']*s5.fy,
                            first_yield_moment_Nmm=m_bend*s5.fy,
                            plastic_moment_Nmm=s5.fy*definition['plastic_modulus_mm3'],
                            c_extreme_mm=definition['c_extreme_mm'],
                            definition=('P_y = A_mesh*fy, M_y = fy*M_ref,bending(mesh) about this axis, M_p = fy*Z '
                                        '(centre-line), c = extreme centre-line fibre')),
                        pm_check=dict(totals, expected_axial_force_N_per_MPa=definition['axial_weight']*section_mesh['area_mm2'],
                                      expected_moment_Nmm_per_MPa=definition['bending_weight']*m_bend))
                    if not math.isclose(totals['axial_force_N_per_MPa'], definition['axial_weight']*section_mesh['area_mm2'],
                                        rel_tol=1e-9, abs_tol=1e-9*section_mesh['area_mm2']):
                        raise ValueError('Axial resultant of %s differs from s*cos(psi)*A' % name)
                    if not math.isclose(totals['moment_Nmm_per_MPa'], definition['bending_weight']*m_bend,
                                        rel_tol=1e-7, abs_tol=1e-7*m_bend):
                        raise ValueError('Moment resultant of %s differs from s*sin(psi)*M_ref (mesh centroid offset?)' % name)
                    step5.write_metadata(name+step5.METADATA_SUFFIX, variant)
                    outputs += [name+'.inp', name+step5.METADATA_SUFFIX]
                    job_names.append(name)
                    entries.append(dict(job=name, case=case, source=source_name, bending_axis_deg=theta,
                                        pm_angle_deg=psi,
                        pm_modeling_scope=dict(GMNIA_MODELING_SCOPE), P_ref_N=p_ref, M_ref_Nmm=m_ref))
                    print('  %s: P_ref=%.6g kN, M_ref=%.6g kN m at LPF=1' % (name, p_ref/1e3, m_ref/1e6))
                    sys.stdout.flush()
                os.remove(template_name+'.inp')
                del database.jobs[template_name]
                del database.models[template_name]
            database.close()
            ratio_tag = ('%g' % (100.*s5.postpeak_stop_ratio)).replace('.', 'p')
            queue_name = 'run_pm_queue_'+ratio_tag+'_pct.bat'
            step5.write_queue_batch(queue_name, job_names, s5.cpus_requested, ratio=s5.postpeak_stop_ratio,
                                    stop_method=s5.stop_method, gpus=s5.gpus_requested)
            outputs.append(queue_name)
            reference = section_reference(segments, thickness, axes, s5.fy)
            reference.update(symmetry)
            outputs += write_section_reference(scratch, reference)
            with open(MANIFEST_FILE, 'w', encoding='utf-8') as stream:
                json.dump(dict(mode='gmnia', source_cae=s5.source_cae, reference_run=run_dir, fy_MPa=s5.fy,
                               reference_stress_MPa=s5.reference_stress, axes_deg=axes, pm_angles_deg=args.pm_angles,
                               cases=selected, jobs=entries, pm_modeling_scope=dict(GMNIA_MODELING_SCOPE), ray_definition=load_cases.__doc__.split('COMBINED')[1]),
                          stream, indent=2)
            outputs.append(MANIFEST_FILE)
            step5.deliver(scratch, s5.output_dir, outputs)
            print('SAVED: %s; %d P-M GMNIA inputs (+ metadata sidecars, queue BAT, section reference). '
                  'NO ANALYSIS SUBMITTED.' % (s5.output_dir, len(job_names)))
        except Exception:
            print('P-M BUILD FAILED. Diagnostic intermediate files: '+scratch)
            raise
        else:
            os.chdir(previous)
            step5.remove_scratch(scratch, 'cfs_pm_')
        finally:
            os.chdir(previous)
    except BaseException:
        database.close()
        raise


# --------------------------------------------------------------------------
# Mode: buckling (Abaqus/CAE)
# --------------------------------------------------------------------------
def write_buckling_batch(path, folders, cpus, gpus):
    lines = ['@echo off', 'setlocal', 'cd /d "%~dp0"', 'set "FAILED=0"']
    for folder in folders:
        lines += ['echo [P-M BUCKLING] %s' % folder, 'pushd "%%~dp0%s"' % folder,
                  'call abaqus job=%s input=%s.inp cpus=%d%s memory="100%%" ask_delete=OFF interactive'
                  % (folder, folder, cpus, ' gpus=%d' % gpus if gpus else ''),
                  'if errorlevel 1 set "FAILED=1"', 'popd']
    lines += ['if "%FAILED%"=="1" echo [ATTENTION] At least one buckling job failed; see its .msg/.dat.',
              'endlocal & exit /b %FAILED%', '']
    with open(path, 'w', newline='') as stream:
        stream.write('\r\n'.join(lines))


def run_buckling(args):
    import caeModules
    from abaqusConstants import ON, LANCZOS, PERCENTAGE, FULL, SINGLE
    import abaqus_step5_gmnia as step5
    state, validated, segments, thickness = reference_section(args.run_dir)
    build = validated['build']
    job = validated['job_name']
    cae = os.path.join(args.run_dir, job+'.cae')
    if not os.path.isfile(cae):
        raise ValueError('Reference CAE not found: '+cae)
    if os.path.exists(args.output_dir) and (not os.path.isdir(args.output_dir) or os.listdir(args.output_dir)):
        raise ValueError('Output directory must be new or empty: '+args.output_dir)
    recorded_symmetry = build.get('load_case', {}).get('section_symmetry')
    source_inputs = validated.get('source_inputs') or build.get('source_inputs') or {}
    if source_inputs.get('steel_plates') and not recorded_symmetry:
        raise ValueError('Steel plate reference lacks full-model symmetry; rebuild the reference')
    axes, symmetry = resolve_axes(args.axes, segments, recorded_symmetry)
    rays = ray_list(axes, args.pm_angles)
    database = contract.open_working_mdb(cae)
    try:
        if job not in database.models.keys():
            raise ValueError('Reference model %s missing from %s' % (job, cae))
        for name in list(database.jobs.keys()):
            del database.jobs[name]
        source = database.models[job]
        if set(source.steps.keys()) != {'Initial', 'Buckle'}:
            raise ValueError('Reference model must hold only the Initial and Buckle steps')
        template_name = 'PMB_TEMPLATE'
        model = database.Model(name=template_name, objectToCopy=source)
        for key in list(model.loads.keys()):
            del model.loads[key]
        assembly = model.rootAssembly
        names = sorted(source_inputs.get('section_segments') or contract.REFERENCE_INSTANCES)
        coordinates = [n.coordinates for name in names for n in assembly.instances[name].nodes]
        zmin, zmax = min(p[2] for p in coordinates), max(p[2] for p in coordinates)
        end_meshes = [(name, dict((n.label, n.coordinates) for n in assembly.instances[name].nodes),
                       [tuple(n.label for n in e.getNodes()) for e in assembly.instances[name].elements])
                      for name in names]
        template = step5.pm_template_nodes(end_meshes, zmin, zmax)
        # A fresh Lanczos Buckle step (eigenvalues >= 0: the reversed load is tension plus the
        # opposite moment, another load case). BCs and contact live in Initial and are kept.
        del model.steps['Buckle']
        model.BuckleStep(name='Buckle', previous='Initial', numEigen=args.n_modes,
                         eigensolver=LANCZOS, minEigen=0.0, maxEigen=None)
        for repository in (model.fieldOutputRequests, model.historyOutputRequests):
            for key in list(repository.keys()):
                del repository[key]
        model.FieldOutputRequest(name='ModeShapes', createStepName='Buckle', variables=('U', 'UR'))
        step5.create_pm_template_loads(model, template, 'Buckle')
        if len(model.boundaryConditions) != int(build['boundary_conditions']):
            raise ValueError('Buckling template lost a boundary condition of the reference model')
        precision = str((state.get('settings') or {}).get('nodal_precision', 'full')).lower()
        scratch = tempfile.mkdtemp(prefix='cfs_pmb_')
        previous = os.getcwd()
        folders, entries = [], []
        try:
            os.chdir(scratch)
            database.Job(name=template_name, model=template_name, numCpus=args.cpus, numDomains=args.cpus,
                         memory=100, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=False,
                         nodalOutputPrecision=FULL if precision == 'full' else SINGLE).writeInput(consistencyChecking=ON)
            with open(template_name+'.inp') as stream:
                template_text = stream.read()
            database.close()
            sets = dict((node['load'], (node['set'], node['instance'], node['label'])) for node in template['nodes'])
            for ray_name, theta, psi in rays:
                folder = check_name(ray_name.replace('PM_', 'PMB_', 1))
                definition = load_cases.combined_reference(segments, thickness, theta, psi)
                forces, totals = ray_forces(template, thickness, definition)
                text = set_cload_values(template_text, forces, sets)
                if not re.search(r'^\*BUCKLE[^\n]*LANCZOS', text, re.M | re.I):
                    raise RuntimeError('Template INP lost the Lanczos *Buckle step')
                os.makedirs(folder)
                with open(os.path.join(folder, folder+'.inp'), 'w') as stream:
                    stream.write(text)
                with open(os.path.join(folder, 'abaqus_v6.env'), 'w') as stream:
                    stream.write('standard_parallel = SOLVER\ngpus = %d\nmemory = "100%%"\ncpus = %d\n'
                                 % (args.gpus, args.cpus))
                report = dict(build)
                report.update(job_name=folder, odb=os.path.join(args.output_dir, folder, folder+'.odb'),
                              loads=len(forces), modes=args.n_modes, submitted=False,
                              eigensolver=dict(requested='lanczos', used='lanczos', min_eigenvalue=0.0,
                                               subspace_vectors_and_iterations_used=False),
                              pm_source_run=args.run_dir, pm_source_job=job,
                              load_case=dict(definition, n_loads=len(forces),
                                             reference_axial_force_mesh_N_per_MPa=totals['axial_force_N_per_MPa'],
                                             reference_moment_mesh_Nmm_per_MPa=totals['moment_Nmm_per_MPa'],
                                             critical_values='P_cr = eigenvalue * reference_axial_force_mesh_N_per_MPa, '
                                                             'M_cr = eigenvalue * reference_moment_mesh_Nmm_per_MPa'))
                with open(os.path.join(folder, folder+'_build.json'), 'w') as stream:
                    json.dump(report, stream, indent=2)
                folders.append(folder)
                entries.append(dict(job=folder, bending_axis_deg=theta, pm_angle_deg=psi,
                                    P_ref_N_per_MPa=totals['axial_force_N_per_MPa'],
                                    M_ref_Nmm_per_MPa=totals['moment_Nmm_per_MPa']))
                print('  %s: P_ref=%.6g N, M_ref=%.6g N mm per MPa extreme-fibre stress'
                      % (folder, totals['axial_force_N_per_MPa'], totals['moment_Nmm_per_MPa']))
            os.remove(template_name+'.inp')
            write_buckling_batch('run_pm_buckling.bat', folders, args.cpus, args.gpus)
            reference = section_reference(segments, thickness, axes, args.fy)
            reference.update(symmetry)
            files = write_section_reference(scratch, reference)
            with open(MANIFEST_FILE, 'w', encoding='utf-8') as stream:
                json.dump(dict(mode='buckling', reference_run=args.run_dir, reference_job=job, axes_deg=axes,
                               pm_angles_deg=args.pm_angles, n_modes=args.n_modes, jobs=entries), stream, indent=2)
            step5.deliver(scratch, args.output_dir, folders+['run_pm_buckling.bat', MANIFEST_FILE]+files)
            print('SAVED: %s; %d P-M buckling inputs in their own folders + run_pm_buckling.bat. '
                  'NO ANALYSIS SUBMITTED.' % (args.output_dir, len(folders)))
        except Exception:
            print('P-M BUCKLING BUILD FAILED. Diagnostic intermediate files: '+scratch)
            raise
        else:
            os.chdir(previous)
            step5.remove_scratch(scratch, 'cfs_pmb_')
        finally:
            os.chdir(previous)
    except BaseException:
        database.close()
        raise


# --------------------------------------------------------------------------
# Mode: collect (pure Python)
# --------------------------------------------------------------------------
_EIGEN_ROW = re.compile(r'^\s*(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$')


def first_eigenvalue(dat_path):
    """Lowest positive eigenvalue from the 'MODE NO EIGENVALUE' table of an Abaqus .dat file."""
    with open(dat_path, errors='replace') as stream:
        lines = stream.read().splitlines()
    values, inside = [], False
    for line in lines:
        if 'MODE NO' in line.upper() and 'EIGENVALUE' in line.upper():
            inside = True
            continue
        if inside:
            match = _EIGEN_ROW.match(line)
            if match:
                values.append(float(match.group(2)))
            elif values and line.strip():
                break
    positive = [v for v in values if v > 0]
    return min(positive) if positive else None


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def interaction_dataset(row):
    directory=os.path.abspath(row.get('dataset') or row.get('folder') or '.')
    if (not row.get('dataset') and row.get('source')=='ELASTIC_BUCKLING'
            and os.path.basename(directory)==row.get('model')):
        directory=os.path.dirname(directory)  # all PMB ray folders share one specimen
    return directory


def collect(args):
    points, sections = [], {}
    local_sections,ambiguous_axes,seen_directories={},{},set()
    for root in args.roots:
        root = os.path.abspath(root)
        for directory, unused_dirs, files in os.walk(root):
            identity=os.path.normcase(os.path.realpath(directory))
            if identity in seen_directories:
                unused_dirs[:]=[]
                continue
            seen_directories.add(identity)
            if SECTION_FILE in files:
                with open(os.path.join(directory, SECTION_FILE), encoding='utf-8') as stream:
                    reference=json.load(stream)
                    local_sections[directory]={key:dict(data,fy_MPa=reference.get('fy_MPa'))
                                               for key,data in reference['axes'].items()}
                    for key,data in local_sections[directory].items():
                        if key in ambiguous_axes:continue
                        if key in sections and sections[key]!=data:
                            sections.pop(key);ambiguous_axes[key]=True
                        else:sections[key]=data
            if 'STEP5_PM_points.csv' in files or 'STEP5_queue_status.json' in files:
                saved=[]
                if 'STEP5_PM_points.csv' in files:
                    with open(os.path.join(directory, 'STEP5_PM_points.csv'), encoding='utf-8-sig') as stream:
                        saved=list(csv.DictReader(stream))
                for row in refresh_saved_pm_points(directory,saved):
                    points.append(dict(row,source='GMNIA',folder=directory))
            for name in files:
                if name.endswith('_build.json') and name.startswith('PMB_'):
                    job = name[:-len('_build.json')]
                    dat = os.path.join(directory, job+'.dat')
                    with open(os.path.join(directory, name), encoding='utf-8') as stream:
                        load_case = json.load(stream).get('load_case') or {}
                    value = first_eigenvalue(dat) if os.path.isfile(dat) else None
                    if value is None:
                        continue
                    p_cr = value*load_case['reference_axial_force_mesh_N_per_MPa']
                    m_cr = value*load_case['reference_moment_mesh_Nmm_per_MPa']
                    points.append(dict(model=job, source='ELASTIC_BUCKLING', folder=directory,
                                       load_case='combined',
                                       bending_axis_deg=(load_case.get('neutral_axis_angle_deg')
                                                         if load_case.get('pm_angle_deg') else ''),
                                       pm_angle_deg=load_case.get('pm_angle_deg'), eigenvalue=value,
                                       P_kN=p_cr/1e3, M_kNm=m_cr/1e6))
    if not points:
        raise ValueError('No STEP5_PM_points.csv or solved PMB_* buckling folders found under %s' % args.roots)
    for row in points:
        row['dataset']=interaction_dataset(row)
        # pure compression / pure bending points lie on the axes of the diagram
        if _float(row.get('M_kNm')) == 0.0 and row.get('M_over_My') in ('', None):
            row['M_over_My'] = 0.0
        if _float(row.get('P_kN')) == 0.0 and row.get('P_over_Py') in ('', None):
            row['P_over_Py'] = 0.0
        # normalized coordinates of elastic points need fy and the section reference
        if row['source'] == 'ELASTIC_BUCKLING' and args.fy:
            axis = row['bending_axis_deg']
            directory=row['folder']
            while directory not in local_sections and os.path.dirname(directory)!=directory:
                directory=os.path.dirname(directory)
            specimen=local_sections.get(directory,{})
            data = specimen.get(load_cases.angle_token(float(axis))) if axis not in ('', None) else (
                next(iter(specimen.values())) if specimen else None)
            if data:
                row['Py_kN'] = data['squash_load_N_per_MPa']*args.fy/1e3
                row['My_kNm'] = data['first_yield_moment_Nmm_per_MPa']*args.fy/1e6
                row['Mp_kNm'] = data['plastic_moment_Nmm_per_MPa']*args.fy/1e6
                row['P_over_Py'] = row['P_kN']/row['Py_kN']
                row['M_over_My'] = row['M_kNm']/row['My_kNm']
    os.makedirs(args.output_dir, exist_ok=True)
    fields = ['source', 'model', 'load_case', 'bending_axis_deg', 'pm_angle_deg', 'source_case', 'P_kN', 'M_kNm',
              'Py_kN', 'My_kNm', 'Mp_kNm', 'P_over_Py', 'M_over_My', 'M_over_Mp', 'eigenvalue', 'a_L_mm',
              'a_D_mm', 'a_G_mm', 'criterion_reached', 'outcome', 'folder', 'dataset']+PM_SELECTION_FIELDS
    path = os.path.join(args.output_dir, 'PM_interaction_points.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(points)
    plots = plot_interaction(args.output_dir, points, sections,fy=args.fy,
                             normalized=getattr(args,'normalized',False))
    print('P-M DIAGRAM: %d points -> %s; %d plot(s)' % (len(points), path, len(plots)))
    return dict(points=len(points), csv=path, plots=plots)


def plot_interaction(directory, points, sections,fy=None,normalized=False):
    """Capacity envelope samples: M horizontal, P vertical, one panel per available axis.

    Section plastic strength is a separate reference, never a label for GMNIA.
    Straight connecting segments are guides through solved samples, not extra
    capacities or a fitted/smoothed envelope. Normalized plots remain optional.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available: plots skipped (CSV written)')
        return []
    axes = sorted(set(float(r['bending_axis_deg']) for r in points
                      if r.get('bending_axis_deg') not in ('', None) and _float(r.get('M_kNm')) is not None)) or [0.]
    written = []
    datasets={interaction_dataset(row) for row in points}
    try:common_dataset=os.path.commonpath(sorted(datasets))
    except ValueError:common_dataset=None  # collections can span Windows drives
    def draw(ax,theta):
        if normalized:
            ax.plot([0,1],[1,0],color='0.65',ls=':',lw=1,label='First-yield reference')
        data = sections.get(load_cases.angle_token(theta))
        if data:
            curve = data['plastic_curve_N_Nmm_per_MPa']
            material_fy=_float(fy if fy is not None else data.get('fy_MPa'))
            if normalized or material_fy is not None and material_fy>0:
                mx=1./data['first_yield_moment_Nmm_per_MPa'] if normalized else material_fy/1e6
                py=1./data['squash_load_N_per_MPa'] if normalized else material_fy/1e3
                ax.plot([m*mx for unused,m in curve],[p*py for p,unused in curve],
                        color='black',ls='--',lw=1.2,label='Plastic section reference')
        groups = {}
        for row in points:
            axis = row.get('bending_axis_deg')
            on_axis = axis in ('', None) or _float(row.get('M_kNm')) in (0.0, None) or \
                math.isclose(float(axis), theta, abs_tol=1e-6)
            x, y = (_float(row.get('M_over_My' if normalized else 'M_kNm')),
                    _float(row.get('P_over_Py' if normalized else 'P_kN')))
            if not on_axis or x is None or y is None:
                continue
            kind=row.get('selection_kind') or ('VERIFIED_PEAK' if row['source']!='GMNIA' or
                 row.get('outcome') in ('CRITERION_REACHED','CONTROLLED_TERMINATION') else 'UNVERIFIED')
            if kind=='VERIFIED_PEAK' and str(row.get('capacity_verified','True')).lower() not in ('true','1'):
                kind='UNVERIFIED'
            label = row['source'] if row['source'] != 'GMNIA' else 'GMNIA '+(row.get('source_case') or row.get('load_case', ''))
            # Identical case labels in different folders may describe different
            # families/specimens. Never combine their capacities into one curve.
            dataset=interaction_dataset(row)
            psi=_float(row.get('pm_angle_deg'))
            interior=(row.get('load_case')=='combined' and psi is not None and 0.<psi<90.)
            groups.setdefault((label, kind, dataset), []).append((x, y, interior))
        for (label, kind, dataset), xy in sorted(groups.items()):
            xy.sort(key=lambda v: math.atan2(v[1], v[0]))
            # Pure compression and bending endpoints alone do not establish
            # the member's combined-load interaction between those endpoints.
            ok=kind=='VERIFIED_PEAK'
            connect=ok and len(xy)>1 and (not dataset or any(v[2] for v in xy))
            if dataset:
                identity=os.path.relpath(dataset,common_dataset) if common_dataset else dataset
                if identity=='.':identity=os.path.basename(dataset)
                label+=' ['+identity+']'
            marker,suffix={'VERIFIED_PEAK':('o',''),
                           'SIGNIFICANT_STIFFNESS_CHANGE':('^',' (provisional: stiffness change)'),
                           'SIGNIFICANT_LOAD_DROP':('v',' (provisional: load-drop onset)'),
                           'INCOMPLETE_ASCENDING_ENDPOINT':('x',' (incomplete/provisional: ascending endpoint)')}.get(
                               kind,('x',' (capacity unverified)'))
            ax.plot([v[0] for v in xy], [v[1] for v in xy], marker=marker,
                    color='black' if dataset else '0.5',markerfacecolor='white',markersize=4,
                    ls='-' if connect else 'none', label=label+suffix)
        ax.set(xlabel='M / My' if normalized else 'Moment, M (kN m)',
               ylabel='P / Py' if normalized else 'Axial compression, P (kN)',
               xlim=(0,None),ylim=(0,None),title='Neutral axis %g deg' % theta)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.legend(fontsize=7,frameon=False)
        ax.margins(x=.06,y=.06)
    stem='PM_interaction_normalized' if normalized else 'PM_interaction'
    def save(fig,name):
        for ext in ('png','svg'):
            path=os.path.join(directory,name+'.'+ext)
            fig.savefig(path,dpi=200);written.append(path)
    for theta in axes:
        fig, ax = plt.subplots(figsize=(6, 5))
        draw(ax,theta)
        fig.tight_layout()
        save(fig,stem+'_A'+load_cases.angle_token(theta))
        if len(axes)==1:save(fig,stem)
        plt.close(fig)
    if len(axes)>1:
        columns=min(2,len(axes));rows=(len(axes)+columns-1)//columns
        fig,panels=plt.subplots(rows,columns,figsize=(6*columns,5*rows),squeeze=False)
        for ax,theta in zip(panels.flat,axes):draw(ax,theta)
        for ax in list(panels.flat)[len(axes):]:ax.set_visible(False)
        fig.tight_layout();save(fig,stem);plt.close(fig)
    return written


def refresh_saved_pm_points(directory,points):
    """Reassess saved synchronized curves read-only; legacy CSV-only inputs still work.

    This lets collect include formerly omitted failed ascending rays without an
    ODB extraction, solver submission, or edits to their original reports.
    """
    status=os.path.join(directory,'STEP5_queue_status.json')
    if not os.path.isfile(status):
        return points
    import abaqus_step5_queue as queue
    with open(status,encoding='utf-8-sig') as stream:records=json.load(stream)
    if not isinstance(records,list) or any(not isinstance(r,dict) or not r.get('model') for r in records):
        raise ValueError('Invalid queue status: '+status)
    if len({r['model'] for r in records})!=len(records):
        raise ValueError('Duplicate model in queue status: '+status)
    refreshed={}
    for record in records:
        name=record['model'];path=os.path.join(directory,name+queue.fd.METADATA_SUFFIX)
        if not os.path.isfile(path):
            if name.startswith('PM_'):raise ValueError('Missing P-M metadata: '+path)
            continue
        with open(path,encoding='utf-8-sig') as stream:info=json.load(stream)
        if not isinstance(info,dict) or info.get('stage')!=5:
            raise ValueError('Invalid STEP5 metadata: '+path)
        if queue.fd.response_kind(info)!='combined' and not name.startswith('PM_'):
            continue
        ratio=record.get('postpeak_stop_ratio',(info.get('settings') or {}).get('postpeak_stop_ratio',.70))
        entry=queue.completed_curve(directory,name,info,float(ratio),allow_incomplete=True,record=record)
        if entry is None:
            raise ValueError('P-M curve is not safely reusable; check queue status: '+name)
        result,unused_curve=entry
        point=result.get('pm_point')
        refreshed[name]=(dict(point,model=name,criterion_reached=result.get('criterion_reached'),
                              outcome=result.get('outcome'),a_L_mm=result.get('a_L_mm'),
                              a_D_mm=result.get('a_D_mm'),a_G_mm=result.get('a_G_mm')) if point else None)
    return [r for r in points if r.get('model') not in refreshed]+[r for r in refreshed.values() if r]


def select_response_point(rows, displacement_key, verified, terminal=True):
    """Select an observed increment, never interpolate a P-M point or certify a knee.

    Policy v1 uses dimensionless, conservative screening: upper 20% of observed
    load, >=5% load excursion, >=75% persistent tangent loss. These are general
    response gates, NOT material failure criteria. Noise raises the excursion
    gate (6 sigma); a knee also needs strong BIC evidence over both a line and a
    smooth quadratic. Ambiguous/undersampled changes retain the original peak.
    Equal-path sampling avoids dependence on Abaqus' adaptive increment sizes.
    """
    import numpy as np
    if not terminal or len(rows)<2:
        return None
    x=np.array([float(r[displacement_key]) for r in rows])
    p=np.array([float(r['lpf']) for r in rows])
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        raise ValueError('Nonfinite P-M selection response')
    peak=int(np.argmax(p)); scale=float(p[peak])
    if scale<=0:
        return None
    # Solver order is essential, including snap-back: never sort by displacement.
    path=np.r_[0.,np.cumsum(np.abs(np.diff(x)))]
    if path[-1]<=0:
        return None
    path/=path[-1]; p=p/scale
    diagnostics=dict(high_load_fraction=.8,minimum_excursion_fraction=.05,
                     maximum_tangent_ratio=.25,noise_sigma_multiplier=6.,bic_evidence=10.)
    def selected(index,kind,**evidence):
        return dict(row=dict(rows[index]),selection_kind=kind,
                    capacity_verified=kind=='VERIFIED_PEAK',provisional=kind!='VERIFIED_PEAK',
                    selection_policy='response-v1',selection_diagnostics=dict(diagnostics,**evidence))
    events=[]
    if len(rows)>=12:
        grid=np.linspace(0.,1.,201)
        # Keep the last converged increment at a duplicate response coordinate.
        keep=np.r_[np.diff(path)>0,True]
        raw=np.interp(grid,path[keep],p[keep])
        pad=np.pad(raw,2,mode='edge')
        smooth=np.median(np.array([pad[k:k+201] for k in range(5)]),axis=0)
        noise=max(1e-8,float(np.median(np.abs(np.diff(raw,2))))/1.652)
        excursion=max(.05,6.*noise)
        diagnostics.update(noise_fraction=noise,excursion_fraction=excursion)
        # A recovered drop must persist over several actual converged increments;
        # a one-frame spike/dip or ordinary post-peak descent cannot qualify.
        for k in range(2,199):
            if smooth[k]<.8 or not (smooth[k]>smooth[k-1] and smooth[k]>=smooth[k+1]):
                continue
            valley=k
            for j in range(k+1,201):
                if smooth[j]<smooth[valley]:valley=j
                if smooth[k]-smooth[valley]>=excursion and smooth[j]-smooth[valley]>=excursion:
                    break
            else:
                continue
            drop=float(smooth[k]-smooth[valley])
            if drop<excursion or valley<=k or float(max(smooth[valley:])-smooth[valley])<excursion:
                continue
            down=np.flatnonzero((path>grid[k])&(path<=grid[valley])&(p<smooth[k]-6.*noise))
            up=np.flatnonzero((path>grid[valley])&(p>smooth[valley]+6.*noise))
            if len(down)<3 or len(up)<3 or grid[valley]-grid[k]<.025:
                continue
            nearby=np.flatnonzero((path>=grid[max(0,k-2)])&(path<=grid[min(200,k+2)]))
            if len(nearby):
                i=int(nearby[np.argmax(p[nearby])])
                events.append((i,'SIGNIFICANT_LOAD_DROP',dict(load_drop_fraction=drop)))
        # Local continuous broken-line fit, compared with ordinary stiffness
        # variation (quadratic). Both sides need real observations and span.
        knees=[]
        for k in range(20,161):
            if smooth[k]<.8:
                continue
            mask=(path>=grid[k-20])&(path<=grid[k+40])
            left=mask&(path<grid[k]);right=mask&(path>grid[k])
            if np.count_nonzero(left)<4 or np.count_nonzero(right)<6:
                continue
            # A displacement reversal can produce a false tangent knee.
            direction=np.diff(x[mask])
            if np.any(direction<0) and np.any(direction>0):
                continue
            z=grid[k-20:k+41]-grid[k]; y=smooth[k-20:k+41]
            hinge=np.column_stack((np.ones(len(z)),z,np.maximum(z,0.)))
            fit=np.linalg.lstsq(hinge,y,rcond=None)[0]
            before=float(fit[1]);after=float(fit[1]+fit[2])
            if before<=0 or after<0 or after>.25*before or .1*(before-after)<excursion:
                continue
            tail=np.polyfit(z[-21:],y[-21:],1)[0]
            if tail<0 or tail>.25*before:
                continue
            n=min(len(z),int(np.count_nonzero(mask)))
            def bic(matrix):
                residual=y-matrix.dot(np.linalg.lstsq(matrix,y,rcond=None)[0])
                variance=max(float(np.mean(residual**2)),noise**2,1e-16)
                return n*math.log(variance)+matrix.shape[1]*math.log(n)
            evidence=min(bic(hinge[:,:2]),bic(np.column_stack((np.ones(len(z)),z,z*z))))-bic(hinge)
            if evidence>=10.:
                knees.append((k,evidence,before,after))
        # One best-supported onset per contiguous candidate cluster, not the
        # first broad window that happens to overlap the change.
        clusters=[]
        for knee in knees:
            if not clusters or knee[0]>clusters[-1][-1][0]+1:clusters.append([])
            clusters[-1].append(knee)
        for cluster in clusters:
            k,evidence,before,after=max(cluster,key=lambda v:v[1])
            i=max(0,int(np.searchsorted(path,grid[k],side='right'))-1)
            events.append((i,'SIGNIFICANT_STIFFNESS_CHANGE',
                           dict(bic_improvement=evidence,tangent_ratio=after/before)))
    if events:
        i,kind,evidence=min(events,key=lambda e:e[0])
        return selected(i,kind,**evidence)
    if verified:
        return selected(peak,'VERIFIED_PEAK')
    # An unfinished ascending endpoint is evidence of attained load only.
    # Do not relabel a descending branch or a plateau as still ascending.
    tolerance=6.*diagnostics.get('noise_fraction',1e-8)
    tail=path>=.9
    ascending=(p[-1]>=max(p)-tolerance and p[-1]>p[-2]+1e-12)
    if np.count_nonzero(tail)>=3 and np.ptp(path[tail])>0:
        ascending=(p[-1]>=max(p)-tolerance and p[-1]>p[tail][0]+tolerance and
                   np.polyfit(path[tail],p[tail],1)[0]>0)
    return selected(len(rows)-1,'INCOMPLETE_ASCENDING_ENDPOINT') if ascending else None


def main(argv=None):
    args = parse_arguments(argv)
    if args.mode == 'section':
        return run_section(args)
    if args.mode == 'collect':
        return collect(args)
    if args.mode == 'buckling':
        return run_buckling(args)
    return run_gmnia(args)


# Abaqus/CAE noGUI may execute this file in a namespace other than __main__; run unless imported.
if __name__ != 'abaqus_pm_interaction':
    main()
