# -*- coding: utf-8 -*-
"""Shared cross-stage contract for the Abaqus_run reference pipeline.

This module deliberately uses only the Python standard library so it can be
imported by normal Python, Abaqus Python and Abaqus/CAE noGUI.
The authoritative upstream model is the one recorded in pipeline_status.json
by abaqus_complete_model_m20.py. Later stages must consume that exact model
rather than rediscovering an arbitrary CAE/ODB by glob order.
"""
import hashlib
import json
import math
import os
import shutil
import tempfile


class _WorkingModelDatabase:
    """Delegate CAE operations while owning the temporary source opened by the kernel."""
    def __init__(self, database, directory):
        self._database, self._directory = database, directory
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._database, name)

    def _cleanup(self):
        if self._directory is not None:
            shutil.rmtree(self._directory)
            self._directory = None

    def saveAs(self, **kwargs):
        result = self._database.saveAs(**kwargs)
        self._cleanup()  # saveAs has released the old source and opened the output CAE.
        return result

    def close(self):
        if self._closed:
            self._cleanup()
            return
        try:
            self._database.close()
        finally:
            self._closed = True
            self._cleanup()


def open_working_mdb(source_cae, scratch_dir=None):
    """Open a disposable CAE working file; openMdb itself uses update access.

    Reading the original with shutil keeps its bytes/timestamp unchanged. This is
    temporary kernel workspace, removed after saveAs/close, never a delivered backup.
    Default workspace is alongside the scripts, not in an existing result directory.
    """
    from abaqus import openMdb
    source_cae = os.path.abspath(source_cae)
    locks = (os.path.splitext(source_cae)[0]+'.lck', source_cae+'.lck')
    if any(os.path.exists(path) for path in locks):
        raise ValueError('Source CAE is locked; close its active kernel before conversion: '+source_cae)
    before = os.stat(source_cae)
    scratch_dir = scratch_dir or os.path.dirname(os.path.abspath(__file__))
    directory = tempfile.mkdtemp(prefix='cfs_cae_work_', dir=scratch_dir)
    try:
        working = os.path.join(directory, os.path.basename(source_cae))
        shutil.copyfile(source_cae, working)
        after = os.stat(source_cae)
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or any(os.path.exists(path) for path in locks)):
            raise ValueError('Source CAE changed or became locked while staging: '+source_cae)
        return _WorkingModelDatabase(openMdb(pathName=working), directory)
    except BaseException:
        shutil.rmtree(directory)
        raise

STATUS_FILE = 'pipeline_status.json'
REFERENCE_ELEMENT_TYPE = 'S4R'
REFERENCE_CONNECTION_MODEL = 'BEAM_MPC'
REFERENCE_CONTACT = 'GENERAL_STANDARD_HARD_FRICTIONLESS'
REFERENCE_STRESS_MPA = 1.0
REFERENCE_INSTANCES = ('P1', 'P2', 'P3', 'P4')
REFERENCE_BOUNDARY_CONDITIONS = 9
REFERENCE_LOADS = 8          # compression: one uniform edge load per piece end
LOAD_CASES = ('compression', 'bending')
# Shared by Step5 and P-M: the supplied working reference's end loading/supports.
# A distinct policy tag prevents reuse of the failed rigid-end-plate inputs/results.
STEP5_END_CONDITION_ID = 'reference_nodal_ends_v1'
STEP5_END_TAG = 'REF1'
STEP5_END_CONDITION = dict(
    id=STEP5_END_CONDITION_ID,
    model='reference end nodes without added reference points or rigid end-plate couplings',
    supports=('inherited end-node U1 = U2 = UR3 = 0; U3, UR1, UR2 free; free end warping; '
              'mid-length AXIAL_ANCHOR (U3) unchanged'),
    loads=('compression: fixed-direction resultant shell-edge loads per original edge length; '
           'bending and P-M: consistent nodal CF3 of the reference stress field, scaled by LPF'),
    response='area-weighted end U3 difference and work-conjugate nodal F/M_ref-weighted end U3')


def plate_instance_names(reference):
    """Additional localized shells, kept separate from P1..P4 section members."""
    plate = reference.get('steel_plates')
    if not plate:
        return []
    names = plate.get('instances') or []
    expected = ['STEEL_PLATE_%03d' % j for j in range(1, len(names)+1)]
    if not names or names != expected:
        raise ValueError('Invalid steel plate instance provenance')
    if plate.get('element_type') != 'S4R':
        raise ValueError('Steel plates must use S4R shells')
    for key in ('height_mm', 'thickness_mm'):
        if _require_number(plate, key) <= 0:
            raise ValueError('Invalid steel plate '+key)
    groups = plate.get('groups')
    if groups:
        if [g.get('direction') for g in groups] != ['X', 'Y']:
            raise ValueError('Invalid perpendicular steel plate directions')
        connected = plate.get('assembly_type') == 'CONNECTED_CROSS'
        grouped_names = (groups[0].get('instances', []) if connected else
                         [name for g in groups for name in g.get('instances', [])])
        if grouped_names != names:
            raise ValueError('Invalid grouped steel plate instance provenance')
        if connected and (plate.get('edge_gap_mm') != 0. or
                groups[1].get('instances') != names or
                groups[1].get('bolt_positions_mm') != groups[0].get('bolt_positions_mm') or
                groups[1].get('screw_positions_mm') != groups[0].get('screw_positions_mm') or
                groups[1].get('height_mm') != groups[0].get('height_mm') or
                plate.get('height_mm') != plate.get('height_2_mm') or
                plate.get('height_mm') != groups[0].get('height_mm')):
            raise ValueError('Invalid connected cross plate provenance')
        for group in groups:
            positions = group.get('bolt_positions_mm', [])
            intervals = group.get('z_intervals_mm', [])
            height = _require_number(group, 'height_mm')
            if height <= 0 or not positions or len(positions) != len(group['instances']) or len(intervals) != len(positions):
                raise ValueError('Invalid grouped steel plate positions')
            for z, interval in zip(positions, intervals):
                if (not isinstance(z, (int, float)) or not math.isfinite(z) or
                        len(interval) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in interval) or
                        not math.isclose(interval[0], z-height/2., abs_tol=1e-7) or
                        not math.isclose(interval[1], z+height/2., abs_tol=1e-7)):
                    raise ValueError('Invalid grouped steel plate intervals')
    return list(names)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def _require_number(mapping, name):
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError('Invalid/missing numeric pipeline field: ' + name)
    return float(value)


def reference_load_case(build):
    """Load case recorded by the builder; builds without the record are compression."""
    load_case = build.get('load_case') if isinstance(build, dict) else None
    if load_case is None:
        return dict(type='compression')
    if not isinstance(load_case, dict) or load_case.get('type') not in LOAD_CASES:
        raise ValueError('Unknown reference load case: %r' % (load_case,))
    return load_case


def _validate_load_case(build):
    load_case = reference_load_case(build)
    if load_case['type'] == 'compression':
        if int(build.get('loads', -1)) != REFERENCE_LOADS:
            raise ValueError('Reference load count differs from the production model')
        return load_case
    n_loads = int(load_case.get('n_loads', -1))
    if n_loads <= 0 or int(build.get('loads', -2)) != n_loads:
        raise ValueError('Bending reference: nodal-force count differs from the recorded load case')
    for name in ('reference_moment_mesh_Nmm_per_MPa', 'reference_moment_Nmm_per_MPa', 'c_extreme_mm',
                 'neutral_axis_angle_deg'):
        value = _require_number(load_case, name)
        if name != 'neutral_axis_angle_deg' and value <= 0:
            raise ValueError('Bending reference field must be positive: ' + name)
    for name in ('compression_normal', 'neutral_axis_direction', 'centroid_mm'):
        vector = load_case.get(name)
        if not isinstance(vector, list) or len(vector) != 2:
            raise ValueError('Bending reference vector missing: ' + name)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
            raise ValueError('Bending reference vector must be finite: ' + name)
    theta = math.radians(load_case['neutral_axis_angle_deg'])
    for name, expected in (('neutral_axis_direction', (math.cos(theta), math.sin(theta))),
                           ('compression_normal', (-math.sin(theta), math.cos(theta)))):
        if any(not math.isclose(a, b, rel_tol=0., abs_tol=1e-8)
               for a, b in zip(load_case[name], expected)):
            raise ValueError('Bending reference vector disagrees with neutral-axis angle: ' + name)
    return load_case


def validate_reference_state(state):
    """Validate the scientific/model contract produced by the reference builder."""
    if not isinstance(state, dict):
        raise ValueError('pipeline_status.json must contain a JSON object')
    build = state.get('build')
    if not isinstance(build, dict):
        raise ValueError('pipeline_status.json has no completed build record')
    source = state.get('source_inputs') or build.get('source_inputs')
    if not isinstance(source, dict):
        raise ValueError('pipeline_status.json has no source_inputs record')

    job_name = str(build.get('job_name') or '').strip()
    if not job_name:
        raise ValueError('Reference build has no job_name')
    if build.get('element_type') != REFERENCE_ELEMENT_TYPE:
        raise ValueError('Reference element type is not S4R')
    if source.get('connection_model') != REFERENCE_CONNECTION_MODEL:
        raise ValueError('Reference connection model is not BEAM_MPC')
    if source.get('shell_contact') != REFERENCE_CONTACT:
        raise ValueError('Reference contact model differs from the production pipeline')
    instances = tuple(sorted((source.get('section_segments') or {}).keys()))
    if instances != REFERENCE_INSTANCES:
        raise ValueError('Reference source does not contain exactly P1..P4')
    plate_names = plate_instance_names(source)
    if plate_names:
        plate = source['steel_plates']
        first = (plate.get('groups') or [plate])[0]
        if len(first.get('screw_positions_mm', first['instances'])) != int(source.get('bolts_per_seam', -1)):
            raise ValueError('Steel plate screw-row count differs from source bolt row count')
        if not math.isclose(float(source['steel_plates']['thickness_mm']),
                            float(source['clear_gap_mm']), rel_tol=0., abs_tol=1e-8):
            raise ValueError('Steel plate thickness differs from the clear gap')
    expected_links = int(source.get('expected_links', -1))
    rigid_links = int(build.get('rigid_links', -2))
    if expected_links <= 0 or rigid_links != expected_links:
        raise ValueError('Reference rigid-link count differs from source_inputs')
    bolt_layout = source.get('bolt_layout')
    if bolt_layout is not None:
        if not bolt_layout or len(bolt_layout) != source.get('physical_bolts'):
            raise ValueError('Reference physical bolt count differs from bolt layout')
        if any(row.get('plate_instance') is not None and row['plate_instance'] not in plate_names for row in bolt_layout):
            raise ValueError('Reference bolt references an unknown steel plate')
        if sum(2 if row.get('plate_instance') else 1 for row in bolt_layout) != expected_links:
            raise ValueError('Reference bolt layout differs from rigid-link count')
        seam_rows = {str(row['seam_id']): [] for row in bolt_layout}
        for row in bolt_layout:
            seam_rows[str(row['seam_id'])].append(row['z_mm'])
        if (seam_rows != source.get('bolt_positions_by_seam_mm') or
                {key: len(rows) for key, rows in seam_rows.items()} != source.get('physical_bolts_per_seam')):
            raise ValueError('Reference per-seam bolt provenance differs from bolt layout')
    if int(build.get('boundary_conditions', -1)) != REFERENCE_BOUNDARY_CONDITIONS:
        raise ValueError('Reference boundary-condition count differs from the production model')
    _validate_load_case(build)
    sigma_ref = _require_number(build, 'reference_stress_MPa')
    if not math.isclose(sigma_ref, REFERENCE_STRESS_MPA, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Reference eigenvalue model must use 1 MPa reference stress')
    length = _require_number(build, 'length_mm')
    if length <= 0:
        raise ValueError('Reference member length must be positive')
    if int(source.get('bolts_per_seam', 0)) <= 0:
        raise ValueError('Reference bolt layout is missing')
    return dict(build=build, source_inputs=source, job_name=job_name)


def load_reference_run(run_dir, require_completed=True, allow_built=False):
    """Load one exact run of abaqus_complete_model_m20.py.

    allow_built: also accept a run that was built but not solved (status BUILT, from
    --build-only): its CAE/INP/build record are complete, only the .odb is missing
    (paths['odb'] is still the recorded name, 'solved' is False). A caller must then take
    every mode shape from another solved model (e.g. the GDLC REF_G / REF_L folders)."""
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    status_path = os.path.join(run_dir, STATUS_FILE)
    if not os.path.isfile(status_path):
        raise ValueError('Reference pipeline status not found: ' + status_path)
    state = _load_json(status_path)
    validated = validate_reference_state(state)
    status = str(state.get('status', '')).upper()
    built_only = bool(allow_built) and status == 'BUILT'
    if require_completed and status != 'COMPLETED' and not built_only:
        raise ValueError('Reference run is not COMPLETED: %s' % state.get('status'))

    job_name = validated['job_name']
    paths = dict(
        cae=os.path.join(run_dir, job_name + '.cae'),
        inp=os.path.join(run_dir, job_name + '.inp'),
        odb=os.path.join(run_dir, job_name + '.odb'),
        build_json=os.path.join(run_dir, job_name + '_build.json'),
        pipeline_status=status_path,
    )
    recorded_odb = validated['build'].get('odb')
    if recorded_odb and os.path.basename(str(recorded_odb)).lower() != os.path.basename(paths['odb']).lower():
        raise ValueError('Recorded ODB basename is inconsistent with job_name')
    for key in ('cae',) if built_only else ('cae', 'odb'):
        if not os.path.isfile(paths[key]):
            raise ValueError('Reference %s file not found: %s' % (key.upper(), paths[key]))
    if not built_only and os.path.exists(os.path.splitext(paths['odb'])[0] + '.lck'):
        raise ValueError('Reference ODB is locked; wait for the solver/writer to finish')

    return dict(
        run_dir=run_dir,
        status=state,
        settings=state.get('settings') or {},
        build=validated['build'],
        source_inputs=validated['source_inputs'],
        job_name=job_name,
        instances=list(REFERENCE_INSTANCES),
        paths=paths,
        solved=not built_only,
    )


def resolve_reference_artifact(contract, explicit, kind):
    """Resolve CAE/ODB overrides without allowing a different upstream job."""
    if kind not in ('cae', 'odb', 'inp'):
        raise ValueError('Unsupported reference artifact kind: ' + str(kind))
    expected = contract['paths'][kind]
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if os.path.normcase(path) != os.path.normcase(os.path.abspath(expected)):
            raise ValueError(
                'Explicit %s must be the exact artifact recorded by --run-dir: %s' %
                (kind.upper(), expected))
    else:
        path = expected
    if not os.path.isfile(path):
        raise ValueError('Reference %s file not found: %s' % (kind.upper(), path))
    return path


def reference_snapshot(contract, include_hashes=True):
    """Small serializable provenance block embedded in downstream CAE models."""
    build = contract['build']
    source = contract['source_inputs']
    snapshot = dict(
        producer='abaqus_complete_model_m20.py',
        run_dir=contract['run_dir'],
        job_name=contract['job_name'],
        model_name=contract['job_name'],
        instances=list(contract['instances']),
        element_type=build['element_type'],
        length_mm=float(build['length_mm']),
        target_mesh_mm=float(build['target_mesh_mm']),
        nodes=int(build['nodes']),
        elements=int(build['elements']),
        rigid_links=int(build['rigid_links']),
        boundary_conditions=int(build['boundary_conditions']),
        loads=int(build['loads']),
        reference_stress_MPa=float(build['reference_stress_MPa']),
        longitudinal_lines=int(build.get('longitudinal_lines', 0)),
        longitudinal_line_min_spacing_mm=float(
            build.get('longitudinal_line_min_spacing_mm', 0.0)),
        nodal_precision=str(contract.get('settings', {}).get('nodal_precision', 'full')),
        buckle_output=str(contract.get('settings', {}).get('buckle_output', 'standard')),
        connection_model=source['connection_model'],
        shell_contact=source['shell_contact'],
        bolts_per_seam=int(source['bolts_per_seam']),
        expected_links=int(source['expected_links']),
        thickness_mm=float(source['thickness_mm']),
        source_directory=source.get('source_directory'),
        steel_plates=source.get('steel_plates'),
        load_case=load_case_snapshot(build),
    )
    for key in ('physical_bolts', 'physical_bolts_per_seam', 'bolt_positions_by_seam_mm', 'bolt_layout'):
        if key in source:
            snapshot[key] = source[key]
    if include_hashes:
        snapshot['pipeline_status_sha256'] = file_sha256(contract['paths']['pipeline_status'])
        snapshot['source_cae_sha256'] = file_sha256(contract['paths']['cae'])
        # a BUILT (unsolved) reference has no .odb: recorded as None, never hashed
        snapshot['source_odb_sha256'] = (file_sha256(contract['paths']['odb'])
                                         if contract.get('solved', True) else None)
        snapshot['reference_solved'] = bool(contract.get('solved', True))
        if os.path.isfile(contract['paths']['inp']):
            snapshot['source_inp_sha256'] = file_sha256(contract['paths']['inp'])
    return snapshot


LOAD_CASE_SNAPSHOT_KEYS = (
    'type', 'tag', 'reference_stress_MPa', 'area_mm2', 'neutral_axis_angle_deg', 'neutral_axis_direction',
    'compression_normal', 'centroid_mm', 'c_extreme_mm', 'I_nn_mm4', 'reference_moment_Nmm_per_MPa',
    'reference_moment_mesh_Nmm_per_MPa', 'section_modulus_mm3', 'section_modulus_outer_fibre_mm3',
    'plastic_modulus_mm3', 'n_loads', 'symmetry_class_axis_deg', 'stress_definition', 'eigenvalue_meaning',
    'section_symmetry', 'bending_axes_for_PM')


def load_case_snapshot(build, expected_type=None):
    """Compact load-case record carried by Step 4/5 provenance (compression when absent)."""
    load_case = reference_load_case(build)
    if expected_type is not None and load_case['type'] != expected_type:
        raise ValueError('--load-case %s differs from the recorded reference load case %s; '
                         'Step4 preserves the source loading and cannot convert it' %
                         (expected_type, load_case['type']))
    return dict((key, load_case[key]) for key in LOAD_CASE_SNAPSHOT_KEYS if key in load_case)


def snapshot_load_case(reference):
    """Load case of a reference snapshot embedded in a Step-4/5 payload (old payloads: compression)."""
    load_case = (reference or {}).get('load_case') or dict(type='compression')
    if load_case.get('type') not in LOAD_CASES:
        raise ValueError('Unknown load case in provenance: %r' % (load_case.get('type'),))
    return load_case


def parse_prefixed_json(description, prefix):
    text = str(description or '')
    if not text.startswith(prefix):
        raise ValueError('Model description does not contain required provenance prefix: ' + prefix)
    payload = text[len(prefix):].strip()
    try:
        data = json.loads(payload)
    except Exception as error:
        raise ValueError('Invalid JSON provenance in model description: %s' % error)
    if not isinstance(data, dict):
        raise ValueError('Model provenance must be a JSON object')
    return data


def validate_step4_payload(payload):
    if payload.get('specification', {}).get('stage') != 4:
        raise ValueError('STEP4 model does not contain a stage-4 specification')
    specification = payload['specification']
    reference = specification.get('reference_pipeline')
    if not isinstance(reference, dict):
        raise ValueError('STEP4 model lacks reference-pipeline provenance')
    if reference.get('connection_model') != REFERENCE_CONNECTION_MODEL:
        raise ValueError('STEP4 source connection model is not BEAM_MPC')
    if reference.get('element_type') != REFERENCE_ELEMENT_TYPE:
        raise ValueError('STEP4 source element type is not S4R')
    if tuple(reference.get('instances') or ()) != REFERENCE_INSTANCES:
        raise ValueError('STEP4 source instances are not exactly P1..P4')
    plate_instance_names(reference)
    if int(reference.get('rigid_links', -1)) != int(reference.get('expected_links', -2)):
        raise ValueError('STEP4 reference bolt count is inconsistent')
    if not isinstance(payload.get('current_case'), dict):
        raise ValueError('STEP4 model lacks current_case provenance')
    return specification, reference
