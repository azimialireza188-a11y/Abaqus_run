import ast
import unittest
import numpy as np
import abaqus_step4_imperfections as step4


class ImperfectionTests(unittest.TestCase):
    def test_mesh_normals_when_cae_shadows_python_sum(self):
        import runpy
        from types import SimpleNamespace as NS

        def cae_sum(*args, **kwargs):
            raise TypeError("arg1; found 'generator', expecting a recognized type")

        namespace = runpy.run_path(step4.__file__, init_globals={'sum': cae_sum},
                                  run_name='cae_namespace_test')
        nodes = [NS(label=i+1, coordinates=xyz) for i, xyz in enumerate(
            [(0., 0., 0.), (1., 0., 0.), (1., 0., 10.), (0., 0., 10.)])]
        element = NS(label=1, type='S4R', connectivity=(1, 2, 3, 4), getNodes=lambda: nodes)
        instance = NS(name='P1', nodes=nodes, elements=[element])
        assembly = NS(instances={'P1': instance})
        values = [NS(instance=instance, nodeLabel=node.label, precision='SINGLE_PRECISION',
                     localCoordSystem=None, data=(0., 0., 0.)) for node in nodes]
        frames = [NS(description='Mode %d' % mode, fieldOutputs={'U': NS(values=values)})
                  for mode in (1, 2)]
        odb = NS(rootAssembly=assembly, steps={'Buckle': NS(frames=frames)})
        args = NS(axis='z', step='Buckle', local_mode=1, dist_mode=2)
        result = namespace['read_modes_and_mesh'](NS(rootAssembly=assembly), odb, ['P1'], args)
        np.testing.assert_allclose(result[2], [[0., -1., 0.]]*4)

    def test_step4_is_bound_to_reference_pipeline_contract(self):
        with open(step4.__file__, encoding='utf-8') as stream:
            source = stream.read()
        tree = ast.parse(source)
        self.assertIn('pipeline_contract.load_reference_run', source)
        self.assertIn("STEP4_PIPELINE_COMPATIBLE ", source)
        self.assertNotIn("glob.glob(os.path.join(run_dir, '*'+suffix))", source)
        self.assertIn("reference['job_name']", source)
        self.assertIn("reference['source_inputs']['expected_links']", source)

    def test_seven_default_cases_and_optional_signs(self):
        cases = step4.case_definitions(2., .34, .7, 1.2, 3600.)
        self.assertEqual(len(step4.DEFAULT_CASES), 7)
        self.assertAlmostEqual(cases['L_low'][0], .68)
        self.assertAlmostEqual(cases['L_high'][0], 1.4)
        self.assertEqual(cases['LD_pm'][:2], (.68, -1.2))
        self.assertEqual(cases['LD_mp'][:2], (-.68, 1.2))
        self.assertAlmostEqual(cases['G1000'][2], 3.6)
        self.assertAlmostEqual(cases['G3000'][2], 1.2)

    def test_normal_amplitude_is_normalized_and_sign_is_reproducible(self):
        u = np.array([[1., -2., 7.], [0., -4., 3.]])
        normals = np.array([[0., 1., 0.], [0., 1., 0.]])
        a, meta = step4.normalize_mode(u, normals, 2, [0, 1], 'normal')
        b, unused = step4.normalize_mode(-3*u, normals, 2, [0, 1], 'normal')
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(np.max(np.abs(np.sum(a*normals, axis=1))), 1.)
        np.testing.assert_allclose(a[:, 2], 0.)

    def test_selected_gauge_controls_amplitude(self):
        u = np.array([[2., 0., 0.], [4., 0., 0.]])
        normalized, unused = step4.normalize_mode(u, np.zeros_like(u), 2, [0], 'transverse')
        self.assertEqual(normalized[0, 0], 1.)
        self.assertEqual(normalized[1, 0], 2.)

    def test_zero_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            step4.normalize_mode(np.zeros((3, 3)), np.zeros((3, 3)), 2, [0, 1, 2], 'normal')

    def test_global_bow_has_zero_ends_and_unit_midspan(self):
        xyz = np.array([[0., 0., z] for z in [0., 900., 1800., 3600.]])
        bow = step4.global_bow(xyz, 2, 90.)
        np.testing.assert_allclose(bow[[0, -1]], 0., atol=1e-14)
        np.testing.assert_allclose(bow[2], [0., 1., 0.], atol=1e-14)

    def test_mode_selection_and_high_amplitude_are_explicit(self):
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step4.parse_arguments(['--run-dir', '.', '--output-cae', 'new.cae'])

    def test_suggestion_mode_does_not_require_cae_build_parameters(self):
        args = step4.parse_arguments(['--run-dir', '.', '--suggest'])
        self.assertTrue(args.suggest)


if __name__ == '__main__':
    unittest.main()
