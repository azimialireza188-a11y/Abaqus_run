# -*- coding: utf-8 -*-
"""Portable Abaqus buckling-mode exporter.

Reads an existing ODB without rebuilding or submitting a job. New runs can
call the same API from abaqus_complete_model_m20.py. Parquet output is written
by modal_parquet_writer.py through a Python interpreter that has pyarrow.
Old U-only ODBs are supported without inventing rotations; their manifest
explicitly reports rotations_available=false.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename)))
_EIGEN = re.compile(r'eigen\s*value\s*[:=]\s*([-+]?(\d*)\.?(\d*)(?:[eEdD]([-+]?\d+))?)', re.I)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _snapshot(path, check_lock=False):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError('Source file not found: ' + path)
    if check_lock and os.path.exists(os.path.splitext(path)[0] + '.lck'):
        raise ValueError('ODB writer lock exists; close/finish the analysis first: ' +
                         os.path.splitext(path)[0] + '.lck')
    stat = os.stat(path)
    return dict(path=path, size=int(stat.st_size),
                mtime_ns=int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))))


def _unchanged(path, before, check_lock=False):
    after = _snapshot(path, check_lock=check_lock)
    if (after['size'], after['mtime_ns']) != (before['size'], before['mtime_ns']):
        raise ValueError('Source changed during export: ' + os.path.abspath(path))


def _description_resolution(description):
    match = _EIGEN.search(str(description))
    if not match or not (match.group(2) or match.group(3)):
        return None
    return .5 * 10. ** (int(match.group(4) or 0) - len(match.group(3)))


def frame_eigen(frame):
    mode = int(getattr(frame, 'mode', 0))
    if mode <= 0:
        return None
    description = str(getattr(frame, 'description', ''))
    match = _EIGEN.search(description)
    if not match or not (match.group(2) or match.group(3)):
        raise ValueError('Missing eigenvalue in mode %d description' % mode)
    printed = float(match.group(1).replace('D', 'E').replace('d', 'e'))
    if not math.isfinite(printed):
        raise ValueError('Nonfinite eigenvalue for mode %d' % mode)
    try:
        exact = float(frame.frameValue)
    except (AttributeError, TypeError, ValueError):
        return mode, printed
    resolution = _description_resolution(description)
    if (math.isfinite(exact) and resolution is not None and
            abs(exact - printed) <= 1.01 * resolution):
        return mode, exact
    return mode, printed


def _vector(value):
    precision = str(getattr(value, 'precision', 'UNKNOWN'))
    double = precision == 'DOUBLE_PRECISION'
    local = getattr(value, 'localCoordSystemDouble' if double else 'localCoordSystem', None)
    if local is not None:
        try:
            if len(local):
                raise ValueError('Local-coordinate nodal output is unsupported; global U/UR is required')
        except TypeError:
            pass
    data = getattr(value, 'dataDouble' if double else 'data')
    vector = np.asarray(data, dtype=np.float64)
    if vector.size < 3 or not np.all(np.isfinite(vector[:3])):
        raise ValueError('Invalid nodal field vector')
    return vector[:3], precision


def _mesh(odb):
    node_instances, node_labels, coordinates = [], [], []
    index = {}
    element_instances, element_labels, connectivity = [], [], []
    element_types = set()
    for instance_name in sorted(odb.rootAssembly.instances.keys()):
        instance = odb.rootAssembly.instances[instance_name]
        for node in sorted(instance.nodes, key=lambda item: int(item.label)):
            key = (str(instance_name), int(node.label))
            if key in index:
                raise ValueError('Duplicate ODB node key: %r' % (key,))
            index[key] = len(node_labels)
            node_instances.append(str(instance_name))
            node_labels.append(int(node.label))
            xyz = np.asarray(node.coordinates, dtype=np.float64)
            if xyz.size != 3 or not np.all(np.isfinite(xyz)):
                raise ValueError('Invalid node coordinates: %r' % (key,))
            coordinates.append(xyz)
        for element in sorted(instance.elements, key=lambda item: int(item.label)):
            etype = str(element.type).upper()
            element_types.add(etype)
            if etype != 'S4R':
                continue
            conn = tuple(int(v) for v in element.connectivity)
            if len(conn) != 4:
                raise ValueError('S4R element does not have four nodes')
            element_instances.append(str(instance_name))
            element_labels.append(int(element.label))
            connectivity.append(conn)
    if not node_labels:
        raise ValueError('ODB has no assembly nodes')
    return dict(
        node_instances=np.asarray(node_instances, dtype='U128'),
        node_labels=np.asarray(node_labels, dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.float64),
        element_instances=np.asarray(element_instances, dtype='U128'),
        element_labels=np.asarray(element_labels, dtype=np.int64),
        connectivity=np.asarray(connectivity, dtype=np.int64).reshape((-1, 4)),
        element_types=sorted(element_types),
        index=index)


def _select_step(odb, requested=None):
    names = list(odb.steps.keys())
    if requested:
        matches = [name for name in names if str(name).lower() == str(requested).lower()]
        if len(matches) != 1:
            raise ValueError('Requested ODB step not found uniquely: ' + str(requested))
        return matches[0]
    buckle = [name for name in names if str(name).lower() == 'buckle']
    if len(buckle) == 1:
        return buckle[0]
    candidates = []
    for name in names:
        if any(int(getattr(frame, 'mode', 0)) > 0 for frame in odb.steps[name].frames):
            candidates.append(name)
    if len(candidates) != 1:
        raise ValueError('Could not identify one buckling step; use --step')
    return candidates[0]


def _field_array(frame, field_name, index, node_count):
    """Compatibility path through individual FieldValue objects."""
    if field_name not in frame.fieldOutputs:
        return None, []
    result = np.full((node_count, 3), np.nan, dtype=np.float64)
    precisions = set()
    for value in frame.fieldOutputs[field_name].values:
        instance = getattr(value, 'instance', None)
        if instance is None:
            continue
        key = (str(instance.name), int(value.nodeLabel))
        row = index.get(key)
        if row is None:
            continue
        if np.isfinite(result[row, 0]):
            raise ValueError('Duplicate %s nodal field value: %r' % (field_name, key))
        vector, precision = _vector(value)
        result[row, :] = vector
        precisions.add(precision)
    if not np.all(np.isfinite(result)):
        missing = int(np.sum(~np.isfinite(result[:, 0])))
        raise ValueError('Incomplete %s nodal field: %d nodes missing' % (field_name, missing))
    return result, sorted(precisions)


def _bulk_block_data(block):
    precision = str(getattr(block, 'precision', 'UNKNOWN'))
    local = getattr(block, 'localCoordSystemDouble', None)
    if local is None:
        local = getattr(block, 'localCoordSystem', None)
    if local is not None:
        try:
            if np.asarray(local).size:
                raise ValueError('Local-coordinate bulk nodal output is unsupported')
        except (TypeError, ValueError):
            raise
        except Exception:
            pass
    if precision == 'DOUBLE_PRECISION' and hasattr(block, 'dataDouble'):
        data = np.asarray(block.dataDouble, dtype=np.float64)
    else:
        data = np.asarray(block.data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape((1, -1))
    if data.ndim != 2 or data.shape[1] < 3 or not np.all(np.isfinite(data[:, :3])):
        raise ValueError('Invalid bulk nodal field data')
    return data[:, :3], precision


def _build_bulk_plan(field, index, node_count):
    """Map Abaqus bulkDataBlocks to global mesh rows once, then reuse per mode."""
    try:
        blocks = list(field.bulkDataBlocks)
    except Exception:
        return None
    if not blocks:
        return None
    seen = np.zeros(node_count, dtype=bool)
    plan = []
    for block in blocks:
        instance = getattr(block, 'instance', None)
        labels_raw = getattr(block, 'nodeLabels', None)
        if instance is None or labels_raw is None:
            return None
        labels = np.asarray(labels_raw, dtype=np.int64).reshape(-1)
        if not len(labels):
            continue
        name = str(instance.name)
        rows = np.fromiter(
            (index.get((name, int(label)), -1) for label in labels),
            dtype=np.int64, count=len(labels))
        if np.any(rows < 0) or len(np.unique(rows)) != len(rows) or np.any(seen[rows]):
            return None
        # Validate shape/coordinate-system once while constructing the plan.
        data, unused_precision = _bulk_block_data(block)
        if len(data) != len(labels):
            return None
        seen[rows] = True
        plan.append((name, labels.copy(), rows))
    if not plan or not np.all(seen):
        return None
    return plan


def _field_array_bulk(field, plan, node_count):
    blocks = list(field.bulkDataBlocks)
    if len(blocks) != len(plan):
        raise ValueError('bulkDataBlocks layout changed between modes')
    result = np.empty((node_count, 3), dtype=np.float64)
    precisions = set()
    for block, (expected_name, expected_labels, rows) in zip(blocks, plan):
        instance = getattr(block, 'instance', None)
        labels = np.asarray(getattr(block, 'nodeLabels', ()), dtype=np.int64).reshape(-1)
        if (instance is None or str(instance.name) != expected_name or
                not np.array_equal(labels, expected_labels)):
            raise ValueError('bulkDataBlocks node ordering changed between modes')
        data, precision = _bulk_block_data(block)
        if len(data) != len(rows):
            raise ValueError('bulkDataBlocks data length changed between modes')
        result[rows, :] = data
        precisions.add(precision)
    if not np.all(np.isfinite(result)):
        raise ValueError('Incomplete bulk nodal field')
    return result, sorted(precisions)


def _stage_npz(stage, mesh, dofs_per_node, shards):
    n = len(mesh['node_labels'])
    dofs = np.tile(np.arange(1, dofs_per_node + 1, dtype=np.int16), n)
    node_index = np.repeat(np.arange(n, dtype=np.int64), dofs_per_node)
    instances = np.repeat(mesh['node_instances'], dofs_per_node)
    labels = np.repeat(mesh['node_labels'], dofs_per_node)
    coordinates = np.repeat(mesh['coordinates'], dofs_per_node, axis=0)
    np.savez_compressed(stage / 'raw_dof_map.npz',
        node_index=node_index, instances=instances, labels=labels, dofs=dofs,
        coordinates=coordinates)
    np.savez_compressed(stage / 'elements.npz',
        instances=mesh['element_instances'], labels=mesh['element_labels'],
        connectivity=mesh['connectivity'])
    for shard in shards:
        data = np.load(stage / shard['stage_file'], allow_pickle=False)
        u = data['u']
        if dofs_per_node == 6:
            combined = np.concatenate((u, data['ur']), axis=2)
        else:
            combined = u
        vectors = combined.transpose(0, 2, 1).reshape(
            combined.shape[0] * dofs_per_node, combined.shape[1])
        np.savez_compressed(stage / shard['npz_file'], vectors=vectors,
            modes=data['modes'], eigenvalues=data['eigenvalues'])


def _external_python_env():
    """Launch normal Python without Abaqus paths or nested native oversubscription."""
    env = os.environ.copy()
    for name in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP'):
        env.pop(name, None)
    # Parallelism is across independent Parquet shard processes. Keep each
    # process single-threaded so --workers auto maps cleanly to logical CPUs.
    for name in ('ARROW_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    return env


def _resolve_worker_count(value):
    logical = max(1, int(os.cpu_count() or 1))
    if value is None or str(value).lower() in ('auto', 'all', 'max'):
        return logical, logical
    try:
        requested = int(value)
    except (TypeError, ValueError):
        raise ValueError('workers must be auto/all/max or a positive integer')
    if requested < 1:
        raise ValueError('workers must be positive')
    return requested, logical


def _candidate_commands(explicit=None):
    result = []
    if explicit:
        result.append([os.path.abspath(os.path.expanduser(explicit))])
    env = os.environ.get('PARQUET_PYTHON')
    if env:
        result.append([os.path.abspath(os.path.expanduser(env))])
    if os.name == 'nt':
        result.append(['py', '-3'])
    result.extend([['python'], ['python3']])
    unique = []
    seen = set()
    for cmd in result:
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            unique.append(cmd)
    return unique


def resolve_parquet_backend(explicit=None):
    try:
        import pyarrow
        return dict(kind='inprocess', command=None, pyarrow_version=str(pyarrow.__version__))
    except Exception:
        pass
    probe = 'import pyarrow,sys;sys.stdout.write(str(pyarrow.__version__))'
    failures = []
    for cmd in _candidate_commands(explicit):
        try:
            run = subprocess.run(cmd + ['-c', probe], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, timeout=20,
                                 env=_external_python_env())
        except Exception as error:
            failures.append('%s: %s' % (' '.join(cmd), error))
            continue
        if run.returncode == 0:
            return dict(kind='external', command=cmd,
                        pyarrow_version=run.stdout.strip() or 'unknown')
        failures.append('%s: %s' % (' '.join(cmd), run.stderr.strip()[-240:]))
    raise RuntimeError(
        'Parquet export requires a normal Python with pyarrow. Abaqus PYTHONHOME/PYTHONPATH '
        'are removed automatically for the external interpreter. Install pyarrow in normal Python '
        '(for example: py -3 -m pip install pyarrow) or pass --parquet-python PATH. '
        'Probes: ' + ' | '.join(failures))


def _write_parquet(stage, backend, compression='zstd', compression_level=9,
                   workers='auto'):
    script = os.path.join(SCRIPT_DIR, 'modal_parquet_writer.py')
    if not os.path.isfile(script):
        raise RuntimeError('Missing modal_parquet_writer.py beside exporter')
    if backend is None:
        backend = resolve_parquet_backend()
    if backend['kind'] == 'inprocess':
        import importlib.util
        spec = importlib.util.spec_from_file_location('portable_parquet_writer', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.write_stage(str(stage), compression=compression,
                                  compression_level=compression_level,
                                  workers=workers)
    command = list(backend['command']) + [script, '--stage-dir', str(stage),
        '--compression', compression, '--compression-level', str(compression_level),
        '--workers', str(workers)]
    run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=_external_python_env())
    if run.returncode != 0:
        raise RuntimeError('Parquet writer failed: ' + run.stderr[-4000:])
    report_path = stage / 'parquet_writer_report.json'
    with open(report_path) as stream:
        return json.load(stream)


def _artifact_hashes(root, names, workers='auto'):
    requested, logical = _resolve_worker_count(workers)
    effective = min(requested, max(1, len(names)))
    if effective == 1:
        return {name: _sha256(root / name) for name in names}
    def one(name):
        return name, _sha256(root / name)
    with ThreadPoolExecutor(max_workers=effective) as pool:
        return dict(pool.map(one, names))


def export_odb_object(odb, output_dir, source_info, output_format='parquet',
                      modes_per_shard=8, step=None, parquet_backend=None,
                      compression='zstd', compression_level=9, workers='auto'):
    if output_format not in ('parquet', 'npz', 'both'):
        raise ValueError('output_format must be parquet, npz or both')
    if type(modes_per_shard) is not int or modes_per_shard < 1:
        raise ValueError('modes_per_shard must be a positive integer')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('Use a new portable export directory: ' + str(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.portable-modal-stage-', dir=str(output_dir.parent)))
    started = time.time()
    try:
        mesh = _mesh(odb)
        step_name = _select_step(odb, requested=step)
        frames = [frame for frame in odb.steps[step_name].frames
                  if int(getattr(frame, 'mode', 0)) > 0]
        if not frames:
            raise ValueError('No positive mode frames in selected buckling step')
        mode_ids = [int(frame.mode) for frame in frames]
        if len(set(mode_ids)) != len(mode_ids):
            raise ValueError('Duplicate buckling mode identifiers in selected step')
        ur_flags = [('UR' in frame.fieldOutputs) for frame in frames]
        if any(ur_flags) and not all(ur_flags):
            raise ValueError('Inconsistent UR availability across buckling modes')
        rotations_available = bool(all(ur_flags))
        if any('U' not in frame.fieldOutputs for frame in frames):
            raise ValueError('U field is required in every buckling mode')
        dofs_per_node = 6 if rotations_available else 3

        # Staging is intentionally uncompressed: serial compression inside Abaqus
        # would block ODB extraction. Final Parquet ZSTD runs in parallel later.
        np.savez(stage / '.nodes_stage.npz',
            node_instances=mesh['node_instances'], node_labels=mesh['node_labels'],
            coordinates=mesh['coordinates'])
        np.savez(stage / '.elements_stage.npz',
            element_instances=mesh['element_instances'],
            element_labels=mesh['element_labels'],
            connectivity=mesh['connectivity'])

        modes = []
        shards = []
        all_single = True
        bulk_plans = {'U': None, 'UR': None}
        bulk_fields_used = set()
        bulk_fallback_fields = set()

        def read_field(frame, field_name):
            field = frame.fieldOutputs[field_name]
            plan = bulk_plans[field_name]
            if plan is None:
                try:
                    plan = _build_bulk_plan(field, mesh['index'], len(mesh['node_labels']))
                except Exception:
                    # Abaqus Python bulk access can reject double precision.
                    # FieldValue uses the source's precision-specific accessor
                    # and independently checks the global-coordinate contract.
                    plan = None
                    bulk_fallback_fields.add(field_name)
                plan = plan if plan is not None else False
                bulk_plans[field_name] = plan
            if plan is not False:
                try:
                    values = _field_array_bulk(field, plan, len(mesh['node_labels']))
                    bulk_fields_used.add(field_name)
                    return values
                except Exception:
                    # Correctness wins over speed: if the ODB block ordering is not
                    # stable, fall back permanently to FieldValue mapping.
                    bulk_plans[field_name] = False
                    bulk_fallback_fields.add(field_name)
            return _field_array(frame, field_name, mesh['index'], len(mesh['node_labels']))

        print('[portable-export] %d modes, %d nodes, %d shard(s).' % (
              len(frames), len(mesh['node_labels']),
              int(math.ceil(float(len(frames)) / modes_per_shard))))
        sys.stdout.flush()
        for offset in range(0, len(frames), modes_per_shard):
            selected = frames[offset:offset + modes_per_shard]
            node_count = len(mesh['node_labels'])
            u = np.empty((node_count, len(selected), 3), dtype=np.float64)
            ur = (np.empty((node_count, len(selected), 3), dtype=np.float64)
                  if rotations_available else None)
            shard_modes = []
            shard_eigenvalues = []
            for j, frame in enumerate(selected):
                mode, eigenvalue = frame_eigen(frame)
                ua, up = read_field(frame, 'U')
                u[:, j, :] = ua
                urp = []
                if rotations_available:
                    ura, urp = read_field(frame, 'UR')
                    ur[:, j, :] = ura
                precisions = sorted(set(up + urp))
                all_single = (all_single and bool(precisions) and
                              all(value == 'SINGLE_PRECISION' for value in precisions))
                modes.append(dict(mode=mode, eigenvalue=float(eigenvalue),
                    description=str(getattr(frame, 'description', '')),
                    fields=['U'] + (['UR'] if rotations_available else []),
                    source_precision=precisions))
                shard_modes.append(mode)
                shard_eigenvalues.append(eigenvalue)
            number = len(shards) + 1
            stage_file = '.shape_stage_%04d.npz' % number
            npz_file = 'modes_%04d.npz' % number
            payload = dict(u=u, modes=np.asarray(shard_modes, dtype=np.int32),
                           eigenvalues=np.asarray(shard_eigenvalues, dtype=np.float64))
            if rotations_available:
                payload['ur'] = ur
            np.savez(stage / stage_file, **payload)
            shards.append(dict(stage_file=stage_file, npz_file=npz_file,
                               parquet_file='mode_shapes_%04d.parquet' % number,
                               modes=shard_modes))
            print('[portable-export] extracted modes %d-%d / %d' % (
                  offset + 1, offset + len(selected), len(frames)))
            sys.stdout.flush()

        report = dict(
            kind='PORTABLE_RAW_MODAL_DATA_ONLY',
            format=output_format,
            source_odb=os.path.abspath(source_info.get('odb_path', '')) if source_info.get('odb_path') else None,
            source_odb_sha256=source_info.get('odb_sha256'),
            source_inp=source_info.get('inp_path'),
            source_inp_sha256=source_info.get('inp_sha256'),
            step=str(step_name),
            mode_count=len(modes),
            node_count=len(mesh['node_labels']),
            s4r_element_count=len(mesh['element_labels']),
            odb_element_types=mesh['element_types'],
            rotations_available=rotations_available,
            dofs_per_node=dofs_per_node,
            raw_dof_count=int(len(mesh['node_labels']) * dofs_per_node),
            modes_per_shard=modes_per_shard,
            modes=modes,
            shards=[dict(modes=s['modes'], parquet_file=s['parquet_file'],
                         npz_file=s['npz_file']) for s in shards],
            parquet_float='float32' if all_single else 'float64',
            scientifically_eligible=False,
            default_classifier_activation=False,
            limitations=[
                'Raw portable eigenmode data only; this is not an mFSM operator pack or validated L/D/G decomposition.',
                'UR is never synthesized. Older U-only ODBs remain U-only in the export.',
                'Parquet FP32 is used only when the source nodal output is not double precision; otherwise FP64 is retained.'
            ],
            odb_extraction_backend=('bulkDataBlocks' if bulk_fields_used and not bulk_fallback_fields
                                    else 'mixed' if bulk_fields_used else 'FieldValue'),
            bulk_fields_used=sorted(bulk_fields_used),
            bulk_fallback_fields=sorted(bulk_fallback_fields),
            workers_requested=str(workers),
            logical_cpu_count=max(1, int(os.cpu_count() or 1)),
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
        )
        with open(stage / 'portable_stage.json', 'w') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)

        artifacts = []
        if output_format in ('parquet', 'both'):
            print('[portable-export] writing %d Parquet shard(s) in parallel; workers=%s.' % (
                  len(shards), str(workers)))
            sys.stdout.flush()
            parquet_report = _write_parquet(stage, parquet_backend,
                compression=compression, compression_level=compression_level,
                workers=workers)
            artifacts.extend(parquet_report['artifacts'])
            report['parquet'] = parquet_report
        if output_format in ('npz', 'both'):
            _stage_npz(stage, mesh, dofs_per_node, shards)
            artifacts.extend(['raw_dof_map.npz', 'elements.npz'])
            artifacts.extend(s['npz_file'] for s in shards)

        for hidden in list(stage.glob('.shape_stage_*.npz')) + [
                stage / '.nodes_stage.npz', stage / '.elements_stage.npz']:
            if hidden.exists():
                hidden.unlink()
        stage_json = stage / 'portable_stage.json'
        if stage_json.exists():
            stage_json.unlink()
        writer_report = stage / 'parquet_writer_report.json'
        if writer_report.exists():
            writer_report.unlink()

        report['artifacts'] = artifacts
        report['artifact_sha256'] = _artifact_hashes(stage, artifacts, workers=workers)
        report['elapsed_seconds'] = time.time() - started
        with open(stage / 'modal_export.json', 'w') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(str(stage), str(output_dir))
        report['output_dir'] = str(output_dir)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def export_odb_path(odb_path, output_dir, output_format='parquet',
                    modes_per_shard=8, step=None, parquet_python=None,
                    parquet_backend=None, compression='zstd', compression_level=9,
                    workers='auto'):
    odb_path = os.path.abspath(os.path.expanduser(odb_path))
    before = _snapshot(odb_path, check_lock=True)
    if output_format in ('parquet', 'both') and parquet_backend is None:
        parquet_backend = resolve_parquet_backend(parquet_python)
    source_info = dict(odb_path=odb_path, odb_sha256=_sha256(odb_path))
    inp_path = os.path.splitext(odb_path)[0] + '.inp'
    if os.path.isfile(inp_path):
        source_info.update(inp_path=os.path.abspath(inp_path), inp_sha256=_sha256(inp_path))
    else:
        source_info.update(inp_path=None, inp_sha256=None)
    from odbAccess import openOdb
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        report = export_odb_object(odb, output_dir, source_info,
            output_format=output_format, modes_per_shard=modes_per_shard,
            step=step, parquet_backend=parquet_backend,
            compression=compression, compression_level=compression_level,
            workers=workers)
    finally:
        odb.close()
    _unchanged(odb_path, before, check_lock=True)
    report['source_stability_verified'] = True
    manifest = os.path.join(report['output_dir'], 'modal_export.json')
    with open(manifest, 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odb', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--format', choices=('parquet', 'npz', 'both'), default='parquet')
    parser.add_argument('--modes-per-shard', type=int, default=8)
    parser.add_argument('--step', default=None)
    parser.add_argument('--parquet-python', default=None,
                        help='Normal Python executable with pyarrow; auto-detected when omitted')
    parser.add_argument('--compression', choices=('zstd', 'snappy', 'gzip'), default='zstd')
    parser.add_argument('--compression-level', type=int, default=9)
    parser.add_argument('--workers', default='auto',
                        help='auto/all/max uses all logical CPUs for Parquet compression/hashing')
    args = parser.parse_args(argv)
    report = export_odb_path(args.odb, args.output_dir, output_format=args.format,
        modes_per_shard=args.modes_per_shard, step=args.step,
        parquet_python=args.parquet_python, compression=args.compression,
        compression_level=args.compression_level, workers=args.workers)
    print(json.dumps(dict(output_dir=report['output_dir'],
        mode_count=report['mode_count'], rotations_available=report['rotations_available'],
        odb_extraction_backend=report.get('odb_extraction_backend'),
        parquet_workers=(report.get('parquet') or {}).get('workers_effective'),
        artifacts=report['artifacts']), indent=2))


if __name__ == '__main__':
    main()
