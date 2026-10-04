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

STATUS_FILE = 'pipeline_status.json'
REFERENCE_ELEMENT_TYPE = 'S4R'
REFERENCE_CONNECTION_MODEL = 'BEAM_MPC'
REFERENCE_CONTACT = 'GENERAL_STANDARD_HARD_FRICTIONLESS'
REFERENCE_STRESS_MPA = 1.0
REFERENCE_INSTANCES = ('P1', 'P2', 'P3', 'P4')
REFERENCE_BOUNDARY_CONDITIONS = 9
REFERENCE_LOADS = 8


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
    expected_links = int(source.get('expected_links', -1))
    rigid_links = int(build.get('rigid_links', -2))
    if expected_links <= 0 or rigid_links != expected_links:
        raise ValueError('Reference rigid-link count differs from source_inputs')
    if int(build.get('boundary_conditions', -1)) != REFERENCE_BOUNDARY_CONDITIONS:
        raise ValueError('Reference boundary-condition count differs from the production model')
    if int(build.get('loads', -1)) != REFERENCE_LOADS:
        raise ValueError('Reference load count differs from the production model')
    sigma_ref = _require_number(build, 'reference_stress_MPa')
    if not math.isclose(sigma_ref, REFERENCE_STRESS_MPA, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Reference eigenvalue model must use 1 MPa reference stress')
    length = _require_number(build, 'length_mm')
    if length <= 0:
        raise ValueError('Reference member length must be positive')
    if int(source.get('bolts_per_seam', 0)) <= 0:
        raise ValueError('Reference bolt layout is missing')
    return dict(build=build, source_inputs=source, job_name=job_name)


def load_reference_run(run_dir, require_completed=True):
    """Load one exact run of abaqus_complete_model_m20.py."""
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    status_path = os.path.join(run_dir, STATUS_FILE)
    if not os.path.isfile(status_path):
        raise ValueError('Reference pipeline status not found: ' + status_path)
    state = _load_json(status_path)
    validated = validate_reference_state(state)
    if require_completed and str(state.get('status', '')).upper() != 'COMPLETED':
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
    for key in ('cae', 'odb'):
        if not os.path.isfile(paths[key]):
            raise ValueError('Reference %s file not found: %s' % (key.upper(), paths[key]))
    if os.path.exists(os.path.splitext(paths['odb'])[0] + '.lck'):
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
    )


def resolve_reference_artifact(contract, explicit, kind):
    """Resolve CAE/ODB overrides without allowing a different upstream job."""
    if kind not in ('cae', 'odb', 'inp'):
        raise ValueError('Unsupported reference artifact kind: ' + str(kind))
    expected = contract['paths'][kind]
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if os.path.basename(path).lower() != os.path.basename(expected).lower():
            raise ValueError(
                'Explicit %s must belong to reference job %s; expected basename %s' %
                (kind.upper(), contract['job_name'], os.path.basename(expected)))
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
    )
    if include_hashes:
        snapshot['pipeline_status_sha256'] = file_sha256(contract['paths']['pipeline_status'])
        snapshot['source_cae_sha256'] = file_sha256(contract['paths']['cae'])
        snapshot['source_odb_sha256'] = file_sha256(contract['paths']['odb'])
        if os.path.isfile(contract['paths']['inp']):
            snapshot['source_inp_sha256'] = file_sha256(contract['paths']['inp'])
    return snapshot


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
    if int(reference.get('rigid_links', -1)) != int(reference.get('expected_links', -2)):
        raise ValueError('STEP4 reference bolt count is inconsistent')
    if not isinstance(payload.get('current_case'), dict):
        raise ValueError('STEP4 model lacks current_case provenance')
    return specification, reference
