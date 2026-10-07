"""Portable ODB export tests; no Abaqus installation or solver required."""
import os
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import abaqus_modal_export as exporter


class Node:
    def __init__(self, label, coordinates):
        self.label = label
        self.coordinates = coordinates


class Element:
    type = 'S4R'
    def __init__(self, label, connectivity):
        self.label = label
        self.connectivity = connectivity


class Instance:
    def __init__(self, name):
        self.name = name
        self.nodes = [
            Node(1, (0., 0., 0.)), Node(2, (1., 0., 0.)),
            Node(3, (1., 1., 0.)), Node(4, (0., 1., 0.))]
        self.elements = [Element(1, (1, 2, 3, 4))]


class Value:
    def __init__(self, instance, label, data, double=False):
        self.instance = instance
        self.nodeLabel = label
        self.precision = 'DOUBLE_PRECISION' if double else 'SINGLE_PRECISION'
        self.localCoordSystem = None
        self.localCoordSystemDouble = None
        self.data = data
        self.dataDouble = data


class Field:
    def __init__(self, values):
        self.values = values


class Frame:
    def __init__(self, mode, instance, with_ur, double=False):
        self.mode = mode
        self.frameValue = 10.0 * mode
        self.description = 'Mode %d: EigenValue = %.7g' % (mode, self.frameValue)
        u = [Value(instance, node.label, (mode + node.label, 2., 3.), double)
             for node in instance.nodes]
        self.fieldOutputs = {'U': Field(u)}
        if with_ur:
            ur = [Value(instance, node.label, (.1, .2, .3), double)
                  for node in instance.nodes]
            self.fieldOutputs['UR'] = Field(ur)


class FakeOdb:
    def __init__(self, with_ur, double=False):
        instance = Instance('P1')
        self.rootAssembly = types.SimpleNamespace(instances={'P1': instance})
        frames = [types.SimpleNamespace(mode=0, fieldOutputs={})]
        frames.extend(Frame(mode, instance, with_ur, double) for mode in (1, 2))
        self.steps = {'Buckle': types.SimpleNamespace(frames=frames)}


class ModalExportTests(unittest.TestCase):
    def test_documented_double_bulk_access_falls_back_without_losing_precision(self):
        odb = FakeOdb(True, double=True)
        class DoubleBlock:
            instance = odb.rootAssembly.instances['P1']
            nodeLabels = (1, 2, 3, 4)
            localCoordSystem = None
            @property
            def data(self):
                raise RuntimeError('Underlying data are double precision')
        for frame in odb.steps['Buckle'].frames[1:]:
            for field in frame.fieldOutputs.values():
                field.bulkDataBlocks = [DoubleBlock()]
            frame.fieldOutputs['U'].values[0].dataDouble = (1.000000000123, 2., 3.)
        with tempfile.TemporaryDirectory() as root:
            result = exporter.export_odb_object(odb, os.path.join(root, 'portable'), {},
                                                output_format='npz')
            with np.load(os.path.join(root, 'portable', 'modes_0001.npz')) as data:
                self.assertEqual(data['vectors'][0, 0], 1.000000000123)
            self.assertEqual(result['parquet_float'], 'float64')

    def test_duplicate_mode_ids_are_rejected_before_serialization(self):
        odb = FakeOdb(False)
        odb.steps['Buckle'].frames[2].mode = 1
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'Duplicate.*mode'):
                exporter.export_odb_object(odb, os.path.join(root, 'portable'), {},
                                           output_format='npz')

    def test_duplicate_nodal_values_cannot_silently_overwrite(self):
        odb = FakeOdb(False)
        frame = odb.steps['Buckle'].frames[1]
        frame.fieldOutputs['U'].values.append(frame.fieldOutputs['U'].values[0])
        mesh = exporter._mesh(odb)
        with self.assertRaisesRegex(ValueError, 'Duplicate.*nodal'):
            exporter._field_array(frame, 'U', mesh['index'], 4)

    def exercise(self, with_ur, double=False):
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, 'portable')
            report = exporter.export_odb_object(
                FakeOdb(with_ur, double), output,
                {'odb_path': 'fake.odb', 'odb_sha256': 'abc'},
                output_format='npz', modes_per_shard=1)
            self.assertEqual(report['rotations_available'], with_ur)
            self.assertEqual(report['dofs_per_node'], 6 if with_ur else 3)
            self.assertEqual(report['parquet_float'], 'float64' if double else 'float32')
            self.assertTrue(os.path.isfile(os.path.join(output, 'modal_export.json')))
            dof_map = np.load(os.path.join(output, 'raw_dof_map.npz'), allow_pickle=False)
            self.assertEqual(len(dof_map['dofs']), 4 * (6 if with_ur else 3))
            first = np.load(os.path.join(output, 'modes_0001.npz'), allow_pickle=False)
            self.assertEqual(first['vectors'].shape, (4 * (6 if with_ur else 3), 1))
            self.assertEqual(first['modes'].tolist(), [1])
            first.close()
            dof_map.close()
            return report

    def test_external_python_environment_drops_abaqus_python_paths(self):
        with mock.patch.dict(os.environ, {
                'PYTHONHOME': r'C:\SIMULIA\fake',
                'PYTHONPATH': r'C:\SIMULIA\fake\lib',
                'PYTHONSTARTUP': r'C:\SIMULIA\fake\startup.py',
                'KEEP_ME': 'yes'}, clear=False):
            env = exporter._external_python_env()
        self.assertNotIn('PYTHONHOME', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('PYTHONSTARTUP', env)
        self.assertEqual(env.get('KEEP_ME'), 'yes')
        for name in ('ARROW_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(env.get(name), '1')

    def test_bulk_plan_maps_complete_node_blocks_once(self):
        instance = Instance('P1')
        class Block:
            precision = 'DOUBLE_PRECISION'
            localCoordSystem = None
            localCoordSystemDouble = None
            def __init__(self):
                self.instance = instance
                self.nodeLabels = np.asarray([1, 2, 3, 4], dtype=np.int64)
                self.dataDouble = np.asarray([
                    [1., 2., 3.], [4., 5., 6.],
                    [7., 8., 9.], [10., 11., 12.]], dtype=np.float64)
                self.data = self.dataDouble.astype(np.float32)
        field = types.SimpleNamespace(bulkDataBlocks=[Block()])
        index = {('P1', label): label - 1 for label in (1, 2, 3, 4)}
        plan = exporter._build_bulk_plan(field, index, 4)
        self.assertIsNotNone(plan)
        values, precision = exporter._field_array_bulk(field, plan, 4)
        self.assertEqual(values.shape, (4, 3))
        self.assertEqual(precision, ['DOUBLE_PRECISION'])
        self.assertTrue(np.allclose(values[3], [10., 11., 12.]))

    def test_auto_workers_resolve_to_logical_cpu_count(self):
        workers, logical = exporter._resolve_worker_count('auto')
        self.assertGreaterEqual(logical, 1)
        self.assertEqual(workers, logical)

    def test_new_u_ur_odb_exports_six_dofs_without_zero_fill(self):
        report = self.exercise(True, double=True)
        self.assertEqual(report['modes'][0]['fields'], ['U', 'UR'])

    def test_old_u_only_odb_remains_supported_without_invented_rotations(self):
        report = self.exercise(False)
        self.assertEqual(report['modes'][0]['fields'], ['U'])
        self.assertIn('UR is never synthesized', ' '.join(report['limitations']))

    def test_existing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'new portable export directory'):
                exporter.export_odb_object(
                    FakeOdb(True), root,
                    {'odb_path': 'fake.odb', 'odb_sha256': 'abc'},
                    output_format='npz')

    def test_mode_value_uses_full_frame_value_only_when_consistent(self):
        frame = types.SimpleNamespace(mode=7, frameValue=12.345678901,
            description='Mode 7: EigenValue = 12.346')
        self.assertEqual(exporter.frame_eigen(frame), (7, 12.345678901))
        inconsistent = types.SimpleNamespace(mode=7, frameValue=7.0,
            description='Mode 7: EigenValue = 12.346')
        self.assertEqual(exporter.frame_eigen(inconsistent), (7, 12.346))


if __name__ == '__main__':
    unittest.main()
