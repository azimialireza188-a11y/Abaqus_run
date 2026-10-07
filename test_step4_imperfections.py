import ast
import unittest
import numpy as np
import abaqus_step4_imperfections as step4


class ImperfectionTests(unittest.TestCase):
    def test_mode_source_requires_job_completion_not_compilation_completion(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as directory:
            odb = os.path.join(directory, 'buckling.odb')
            log = os.path.join(directory, 'buckling.log')
            with open(log, 'w') as stream:
                stream.write('Compilation completed successfully\nBegin Abaqus/Standard Analysis\n')
            self.assertFalse(step4._analysis_completed(odb))
            with open(log, 'a') as stream:
                stream.write('Abaqus JOB buckling COMPLETED\n')
            self.assertTrue(step4._analysis_completed(odb))
            with open(log, 'w') as stream:
                stream.write('Abaqus JOB different COMPLETED\n')
            self.assertFalse(step4._analysis_completed(odb))

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

    def test_abaqus_repositories_are_iterated_via_keys(self):
        with open(step4.__file__, encoding='utf-8') as stream:
            source = stream.read()
        self.assertIn("source.constraints.keys()", source)
        self.assertNotIn("for name in source.constraints if", source)

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


    def test_amplitude_tokens_and_combos(self):
        self.assertEqual(step4.parse_amplitude('0.34t')[:2], ('t', .34))
        self.assertEqual(step4.parse_amplitude('-3.6')[:2], ('mm', -3.6))
        self.assertEqual(step4.parse_amplitude('3.6mm')[:2], ('mm', 3.6))
        self.assertEqual(step4.parse_amplitude('-L/1000')[:2], ('L', -.001))
        for bad in ('L/0', 'abc', '1,2', 't'):
            with self.assertRaises(ValueError):
                step4.parse_amplitude(bad)
        combos = step4.parse_combos(['LDG_pp=0.34t,3.6,L/1000', 'LG=-0.34t,0,L/3000'])
        self.assertEqual(list(combos), ['LDG_pp', 'LG'])
        self.assertAlmostEqual(step4.amplitude_mm(combos['LDG_pp'][0], 2., 3600.), .68)
        self.assertAlmostEqual(step4.amplitude_mm(combos['LDG_pp'][2], 2., 3600.), 3.6)
        for bad in (['l_low=1,0,0'], ['X=0,0,0'], ['X=1,2'], ['1X=1,0,0'], ['A=1,0,0', 'a=0,1,0']):
            with self.assertRaises(ValueError):
                step4.parse_combos(bad)

    def test_combo_only_needs_its_components(self):
        import contextlib, io
        base = ['--run-dir', '.', '--output-cae', 'new.cae']
        args = step4.parse_arguments(base+['--combo', 'LG=0.34t,0,L/1000', '--local-mode', '3'])
        self.assertEqual(args.selected_cases, ['LG'])
        self.assertEqual(args.needed_components, ['global', 'local'])
        args = step4.parse_arguments(base+['--cases', 'D', '--dist-mode', '1', '--dist-mm', '3.6',
                                            '--combo', 'LDG=0.34t,-1.8t,L/1000', '--local-mode', '2'])
        self.assertEqual(args.selected_cases, ['D', 'LDG'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step4.parse_arguments(base+['--combo', 'LG=0.34t,0,L/1000'])          # no --local-mode
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step4.parse_arguments(base+['--cases', 'G1000', '--global-source', 'x', '--global-mode', '1',
                                        '--global-angle-deg', '30'])           # single mode cannot rotate
        args = step4.parse_arguments(base+['--cases', 'G1000', '--global-source', 'x', '--global-mode', '1,2',
                                           '--global-angle-deg', '30'])
        self.assertEqual(args.global_modes, (1, 2))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step4.parse_arguments(base+['--cases', 'L_low,G1000', '--local-mode', '1',
                                        '--global-source', '.', '--global-mode', '1,2'])  # same mode, same source

    def test_degenerate_pair_is_aligned_with_requested_direction(self):
        z = np.linspace(0., 3600., 37)
        wave = np.sin(np.pi*z/3600.)
        c, s = np.cos(np.radians(37.)), np.sin(np.radians(37.))
        phi1 = np.column_stack([c*wave, s*wave, .1*wave])        # arbitrary basis of the pair
        phi2 = np.column_stack([-s*wave, c*wave, 0*wave])
        for angle in (0., 90., 30.):
            shape, meta = step4.global_eigen_shape([phi1, phi2], 2, angle)
            self.assertAlmostEqual(meta['bow_angle_deg'], angle, places=8)
            self.assertAlmostEqual(np.max(np.linalg.norm(shape, axis=1)), 1.)
            np.testing.assert_allclose(shape[:, 2], 0.)
            self.assertAlmostEqual(meta['translation_ratio'], 1.)
        single, meta = step4.global_eigen_shape([phi1], 2)
        self.assertAlmostEqual(meta['bow_angle_deg'], 37., places=8)
        with self.assertRaises(ValueError):
            step4.global_eigen_shape([phi1, 2*phi1], 2, 0.)

    def test_eigenvalue_parsing_and_degeneracy(self):
        from types import SimpleNamespace as NS
        self.assertEqual(step4.frame_eigenvalue(NS(description='Mode         1: EigenValue =   2369.1')), 2369.1)
        self.assertEqual(step4.frame_eigenvalue(NS(description='Mode 1', frameValue=5.)), 5.)
        self.assertEqual(step4.degenerate_partners({1: 2369.1, 2: 2369.1, 3: 2500.}, 1), [2])
        self.assertEqual(step4.degenerate_partners({1: 100., 2: 101.}, 1), [])

    def test_gdlc_shares_require_matching_eigenvalue(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'mode_decomposition_GDLC'))
            with open(os.path.join(d, 'mode_decomposition_GDLC', 'mode_participation.csv'), 'w') as f:
                f.write('mode,eigenvalue,rel_G_pct,rel_D_pct,rel_L_pct,rel_C_pct,rel_O_pct,reliable_dominant,confidence\n'
                        '1,268.7800,0,99.99,0.002,0.007,0,D,high\n')
            item = step4.gdlc_mode_shares(d, 1, 268.78)
            self.assertEqual(item['status'], 'ok')
            self.assertEqual(item['dominant'], 'D')
            self.assertAlmostEqual(item['shares_pct']['D'], 99.99)
            self.assertEqual(step4.gdlc_mode_shares(d, 1, 300.)['status'], 'stale_eigenvalue_mismatch')
            self.assertEqual(step4.gdlc_mode_shares(d, 2, 1.)['status'], 'mode_not_classified')
            self.assertIsNone(step4.gdlc_mode_shares(os.path.join(d, 'none'), 1, 1.))

    def test_clearance_screen_flags_new_interpiece_overclosure(self):
        keys = [('P1', 1), ('P1', 2), ('P2', 1), ('P2', 2)]
        xyz = np.array([[0., 0., 0.], [0., 0., 20.], [2., 0., 0.], [2., 0., 20.]])   # midsurfaces t=2 apart
        pairs = step4.interpiece_candidates(keys, xyz, 2.+2.*.5+1e-6)
        self.assertEqual(sorted(zip(*[p.tolist() for p in pairs])), [(0, 2), (1, 3)])
        closing = np.zeros_like(xyz); closing[1, 0] = .5                            # P1 node 2 moves toward P2
        screen = step4.clearance_screen(xyz, closing, pairs, 2.)
        self.assertEqual(screen['new_overclosure_pairs'], 1)
        self.assertAlmostEqual(screen['worst_overclosure_mm'], .5)
        opening = -closing
        self.assertEqual(step4.clearance_screen(xyz, opening, pairs, 2.)['new_overclosure_pairs'], 0)

    def test_bending_default_bow_is_lateral(self):
        self.assertEqual(step4.default_global_angle(dict(type='compression')), 0.)
        self.assertEqual(step4.default_global_angle(None), 0.)
        self.assertEqual(step4.default_global_angle(dict(type='bending', neutral_axis_angle_deg=0.)), 0.)
        self.assertEqual(step4.default_global_angle(dict(type='bending', neutral_axis_angle_deg=225.)), 45.)


if __name__ == '__main__':
    unittest.main()
