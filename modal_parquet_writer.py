# -*- coding: utf-8 -*-
"""Convert staged portable modal arrays to compact Parquet files in parallel."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import numpy as np


def _arrow():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


def _metadata(table, values):
    current = dict(table.schema.metadata or {})
    for key, value in values.items():
        current[str(key).encode('utf-8')] = str(value).encode('utf-8')
    return table.replace_schema_metadata(current)


def _write(table, path, pq, compression, compression_level, dictionary=None):
    kwargs = dict(compression=compression, use_dictionary=dictionary or False,
                  write_statistics=True)
    if compression in ('zstd', 'gzip'):
        kwargs['compression_level'] = compression_level
    pq.write_table(table, path, **kwargs)


def resolve_workers(value, shard_count):
    logical = max(1, int(os.cpu_count() or 1))
    if value is None or str(value).lower() in ('auto', 'all', 'max'):
        requested = logical
    else:
        try:
            requested = int(value)
        except (TypeError, ValueError):
            raise ValueError('workers must be auto/all/max or a positive integer')
        if requested < 1:
            raise ValueError('workers must be positive')
    return min(requested, max(1, int(shard_count))), logical, requested


def _write_common(root, manifest, compression, compression_level):
    pa, pq = _arrow()
    nodes = np.load(root / '.nodes_stage.npz', allow_pickle=False)
    elements = np.load(root / '.elements_stage.npz', allow_pickle=False)
    node_count = len(nodes['node_labels'])

    node_table = pa.table({
        'node_index': np.arange(node_count, dtype=np.int64),
        'instance': nodes['node_instances'].astype(str).tolist(),
        'label': nodes['node_labels'].astype(np.int64),
        'x': nodes['coordinates'][:, 0].astype(np.float64),
        'y': nodes['coordinates'][:, 1].astype(np.float64),
        'z': nodes['coordinates'][:, 2].astype(np.float64),
    })
    node_table = _metadata(node_table, {
        'source_odb_sha256': manifest.get('source_odb_sha256') or '',
        'coordinate_system': 'GLOBAL_UNDEFORMED',
    })
    _write(node_table, root / 'mesh_nodes.parquet', pq, compression,
           compression_level, dictionary=['instance'])

    dofs = int(manifest['dofs_per_node'])
    dof_table = pa.table({
        'raw_index': np.arange(node_count * dofs, dtype=np.int64),
        'node_index': np.repeat(np.arange(node_count, dtype=np.int64), dofs),
        'dof': np.tile(np.arange(1, dofs + 1, dtype=np.int16), node_count),
    })
    _write(dof_table, root / 'raw_dof_map.parquet', pq, compression,
           compression_level)

    element_table = pa.table({
        'instance': elements['element_instances'].astype(str).tolist(),
        'label': elements['element_labels'].astype(np.int64),
        'n1': elements['connectivity'][:, 0].astype(np.int64),
        'n2': elements['connectivity'][:, 1].astype(np.int64),
        'n3': elements['connectivity'][:, 2].astype(np.int64),
        'n4': elements['connectivity'][:, 3].astype(np.int64),
    })
    _write(element_table, root / 'elements.parquet', pq, compression,
           compression_level, dictionary=['instance'])

    mode_table = pa.table({
        'mode': np.asarray([row['mode'] for row in manifest['modes']], dtype=np.int32),
        'eigenvalue': np.asarray([row['eigenvalue'] for row in manifest['modes']], dtype=np.float64),
        'description': [row.get('description', '') for row in manifest['modes']],
        'fields': [','.join(row.get('fields', [])) for row in manifest['modes']],
        'source_precision': [','.join(row.get('source_precision', [])) for row in manifest['modes']],
    })
    _write(mode_table, root / 'modes.parquet', pq, compression,
           compression_level)
    return node_count


def _write_shard_job(args):
    (root_text, shard_index, node_count, parquet_float, rotations_available,
     source_odb_sha256, compression, compression_level) = args
    # Each process owns one shard and one pyarrow writer. The launcher constrains
    # native libraries to one thread per process, so N workers ~= N logical CPUs.
    pa, pq = _arrow()
    root = Path(root_text)
    stage_name = '.shape_stage_%04d.npz' % shard_index
    data = np.load(root / stage_name, allow_pickle=False)
    float_type = np.float64 if parquet_float == 'float64' else np.float32
    columns = {'node_index': np.arange(node_count, dtype=np.int64)}
    mode_ids = data['modes'].astype(int).tolist()
    for j, mode in enumerate(mode_ids):
        for component, name in enumerate(('u1', 'u2', 'u3')):
            columns['m%04d_%s' % (mode, name)] = data['u'][:, j, component].astype(
                float_type, copy=False)
        if rotations_available:
            for component, name in enumerate(('ur1', 'ur2', 'ur3')):
                columns['m%04d_%s' % (mode, name)] = data['ur'][:, j, component].astype(
                    float_type, copy=False)
    table = pa.table(columns)
    table = _metadata(table, {
        'modes': ','.join(str(v) for v in mode_ids),
        'source_odb_sha256': source_odb_sha256 or '',
        'vector_coordinate_system': 'GLOBAL',
        'rotations_available': rotations_available,
    })
    filename = 'mode_shapes_%04d.parquet' % shard_index
    _write(table, root / filename, pq, compression, compression_level)
    return filename, mode_ids


def write_stage(stage_dir, compression='zstd', compression_level=9, workers='auto'):
    pa, unused_pq = _arrow()
    root = Path(stage_dir)
    with open(root / 'portable_stage.json') as stream:
        manifest = json.load(stream)
    shard_count = len(manifest['shards'])
    effective_workers, logical_cpus, requested_workers = resolve_workers(
        workers, shard_count)
    node_count = _write_common(root, manifest, compression, compression_level)

    jobs = [
        (str(root), shard_index, node_count, manifest['parquet_float'],
         bool(manifest['rotations_available']), manifest.get('source_odb_sha256'),
         compression, compression_level)
        for shard_index in range(1, shard_count + 1)
    ]
    completed = {}
    if effective_workers == 1:
        for job in jobs:
            filename, modes = _write_shard_job(job)
            completed[int(filename.split('_')[-1].split('.')[0])] = filename
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = [pool.submit(_write_shard_job, job) for job in jobs]
            for future in as_completed(futures):
                filename, modes = future.result()
                completed[int(filename.split('_')[-1].split('.')[0])] = filename

    shard_artifacts = [completed[i] for i in range(1, shard_count + 1)]
    artifacts = ['mesh_nodes.parquet', 'raw_dof_map.parquet',
                 'elements.parquet', 'modes.parquet'] + shard_artifacts
    report = dict(
        format='parquet', compression=compression,
        compression_level=compression_level,
        pyarrow_version=pa.__version__, artifacts=artifacts,
        workers_requested=requested_workers,
        workers_effective=effective_workers,
        logical_cpu_count=logical_cpus,
        layout='one row per node; each shard stores mode U/UR components as columns')
    with open(root / 'parquet_writer_report.json', 'w') as stream:
        json.dump(report, stream, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir', required=True)
    parser.add_argument('--compression', choices=('zstd', 'snappy', 'gzip'), default='zstd')
    parser.add_argument('--compression-level', type=int, default=9)
    parser.add_argument('--workers', default='auto',
                        help='auto/all/max uses all logical CPUs; otherwise positive integer')
    args = parser.parse_args(argv)
    report = write_stage(args.stage_dir, args.compression, args.compression_level,
                         workers=args.workers)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
