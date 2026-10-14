import ast
import unittest
import numpy as np
import abaqus_step4_imperfections as step4


class ImperfectionTests(unittest.TestCase):
    def test_all_reference_load_cases_reject_impure_or_plate_carried_modes(self):
        for load_case in ('compression','bending','combined'):
            for component,expected in (('local','L'),('global','G')):
                classification=dict(status='ok',reference_check=dict(kind=expected),
                    shares_pct=dict(G=0.,D=0.,L=0.,C=0.,O=0.),
                    plate_deformation_pct=.2,cluster_plate_deformation_max_pct=.3)
                classification['shares_pct'][expected]=99.
                step4.validate_bending_reference_mode(dict(type=load_case),component,1,classification)
                with self.subTest(load_case=load_case,component=component):
                    classification['shares_pct'][expected]=90.
                    with self.assertRaises(ValueError):
                        step4.validate_bending_reference_mode(dict(type=load_case),component,1,classification)
                    classification['shares_pct'][expected]=99.
                    classification['cluster_plate_deformation_max_pct']=12.
                    with self.assertRaises(ValueError):
                        step4.validate_bending_reference_mode(dict(type=load_case),component,1,classification)

    def test_plate_global_offsets_use_member_normalization_and_same_coefficients(self):
        member = np.array([0, 1])
        field = np.array([[1., 0., 0.], [2., 0., 0.], [10., 0., 7.]])
        shape, meta = step4.member_global_eigen_shape([field], 2, 0., member)
        np.testing.assert_allclose(shape, [[.5, 0., 0.], [1., 0., 0.], [5., 0., 0.]])
        self.assertEqual(meta['gauge_node_count'], 2)

    def test_invalid_mode_is_visible_on_stdout_even_when_cae_hides_stderr(self):
        import contextlib, io
        for option, value in (('--local-mode', 'set /p DIST_MODE='),
                              ('--dist-mode', ''), ('--global-mode', 'cd /d D:\\Abaqus_run'),
                              ('--global-mode', '%GLOBAL_MODES%')):
            with self.subTest(option=option, value=value):
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        step4.parse_arguments(['--run-dir', '.', option, value])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('STEP4 INPUT ERROR', stdout.getvalue())
                self.assertIn(option, stdout.getvalue())
                self.assertIn(repr(value), stdout.getvalue())
                self.assertIn('mode numbers', stdout.getvalue())

    def test_cae_argument_failure_is_not_reported_as_systemexit_success(self):
        import contextlib, io
        from unittest.mock import patch
        with patch.object(step4.sys, 'argv', ['ABQcaeK.exe', '-cae']), \
             patch.object(step4, 'build') as build, \
             contextlib.redirect_stdout(io.StringIO()) as stdout, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'STEP4 FAILED.*invalid arguments'):
                step4.main(['--run-dir', '.', '--local-mode', 'set'])
        build.assert_not_called()
        self.assertIn('No imperfection CAE was created', stdout.getvalue())

    def test_cae_exception_includes_actual_bad_mode_when_console_streams_are_hidden(self):
        import contextlib, io
        from unittest.mock import patch
        for option, value in (('--local-mode', 'set /p DIST_MODE=Selected'),
                              ('--global-mode', 'cd /d D:\\Abaqus_run')):
            with self.subTest(option=option), \
                 patch.object(step4.sys, 'argv', ['ABQcaeK.exe', '-cae']), \
                 patch.object(step4, 'build') as build, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError) as raised:
                    step4.main(['--run-dir', '.', option, value])
                self.assertIn(option, str(raised.exception))
                self.assertIn(repr(value), str(raised.exception))
                self.assertIn('No imperfection CAE was created', str(raised.exception))
                build.assert_not_called()

    def test_argparse_systemexit_retains_original_diagnostic(self):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                step4.parse_arguments(['--run-dir', '.', '--local-mode', 'set'])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--local-mode: invalid int value: 'set'", raised.exception.step4_argument_error)

    def test_cae_detection_also_handles_preloaded_abaqus_without_cae_flag(self):
        import contextlib, io
        from unittest.mock import patch
        with patch.object(step4.sys, 'argv', ['ABQcaeK.exe']), \
             patch.dict(step4.sys.modules, {'abaqus': object()}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'STEP4 FAILED'):
                step4.main(['--run-dir', '.', '--global-mode', 'cd'])

    def test_regular_python_argument_failure_retains_exit_code_two(self):
        import contextlib, io
        from unittest.mock import patch
        with patch.object(step4.sys, 'argv', ['abaqus_step4_imperfections.py']), \
             patch.dict(step4.sys.modules, {}, clear=False), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            step4.sys.modules.pop('abaqus', None)
            with self.assertRaises(SystemExit) as raised:
                step4.main(['--run-dir', '.', '--local-mode', 'set'])
        self.assertEqual(raised.exception.code, 2)

    def test_cae_help_is_still_a_successful_exit(self):
        import contextlib, io
        from unittest.mock import patch
        with patch.object(step4.sys, 'argv', ['ABQcaeK.exe', '-cae']), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                step4.main(['--help'])
        self.assertEqual(raised.exception.code, 0)

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

    def test_built_only_reference_supplies_no_modes_but_explicit_reference_folders_do(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as root:
            run = os.path.join(root, 'RUN')
            ref = os.path.join(root, 'RUN_REF_L')
            os.makedirs(run)
            os.makedirs(ref)
            for ext, text in (('.odb', 'odb'), ('.inp', '*Heading'),
                              ('.sta', ' THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n')):
                with open(os.path.join(ref, 'JOB_REF_L'+ext), 'w') as stream:
                    stream.write(text)
            for spec in (None, run):                       # default source = the unsolved run itself
                with self.assertRaisesRegex(ValueError, 'only BUILT'):
                    step4.resolve_mode_source(spec, None, run)
            path, info = step4.resolve_mode_source(ref, None, run)
            self.assertEqual(path, os.path.join(ref, 'JOB_REF_L.odb'))
            self.assertEqual(info['kind'], 'buckling_model')

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
        for bad in (['X=0,0,0'], ['X=1,2'], ['1X=1,0,0'], ['A=1,0,0', 'a=0,1,0']):
            with self.assertRaises(ValueError):
                step4.parse_combos(bad)

    def test_combo_accepts_every_family_mix_in_any_number_and_order(self):
        import contextlib, io
        base = ['--run-dir', '.', '--output-cae', 'new.cae', '--local-mode', '1', '--dist-mode', '3',
                '--global-source', 'x', '--global-mode', '1,2', '--cases', 'none']
        combos = ['LDG_mp=-0.34t,1.8t,L/1000', 'LDG_mm=-0.34t,-1.8t,L/1000', 'L_mp=-0.34t,0,0',
                  'LG_mp=-0.34t,0,L/1000', 'DG_p=0,1.8t,L/1000', 'LD_mp=-0.34t,1.8t,0', 'D=0,1.8t,0',
                  'G=0,0,-L/1000']
        families = dict(LDG_mp='LDG', LDG_mm='LDG', L_mp='L', LG_mp='LG', DG_p='DG', LD_mp='LD', D='D', G='G')
        for order in (combos, combos[::-1], combos[2:3]):
            argv = list(base)
            for text in order:
                argv += ['--combo', text]
            args = step4.parse_arguments(argv)
            self.assertEqual(args.selected_cases, [c.split('=')[0] for c in order])
            for name, specs in args.combos.items():
                mm = [step4.amplitude_mm(s, 2., 3600.) for s in specs]
                self.assertEqual(''.join(f for f, a in zip('LDG', mm) if a != 0.), families[name])
        mm = [step4.amplitude_mm(s, 2., 3600.) for s in step4.parse_combos(combos[:1])['LDG_mp']]
        np.testing.assert_allclose(mm, [-.68, 3.6, 3.6])
        args = step4.parse_arguments(base+['--combo', 'L_mp=-0.34t,0,0'])
        self.assertEqual(args.needed_components, ['local'])               # absent families are not read
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step4.parse_arguments(base[:-1]+['L_low', '--combo', 'l_low=0.34t,0,0'])  # clashes with a selected case

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
        # a single mode with an angle is accepted: build pairs it with an equal-eigenvalue partner
        args = step4.parse_arguments(base+['--cases', 'G1000', '--global-source', 'x', '--global-mode', '2',
                                           '--global-angle-deg', '0'])
        self.assertEqual(args.global_modes, (2,))
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

    def test_single_global_mode_with_angle_uses_its_degenerate_partner(self):
        values = {1: 296.44, 2: 1161.3, 3: 1161.3, 4: 1494.2}
        self.assertEqual(step4.global_modes_for_angle((2,), values, True)[0], (2, 3))
        self.assertEqual(step4.global_modes_for_angle((2,), values, False), ((2,), None))
        self.assertEqual(step4.global_modes_for_angle((1,), values, True), ((1,), None))
        self.assertEqual(step4.global_modes_for_angle((2, 3), values, True), ((2, 3), None))
        self.assertEqual(set(step4.global_modes_for_angle((2,), {1: 5., 2: 5., 3: 5.}, True)[0]), {1, 2, 3})
        with self.assertRaises(ValueError):
            step4.global_modes_for_angle((1, 2), {1: 5., 2: 5., 3: 5.}, True)
        with self.assertRaises(ValueError):
            step4.global_modes_for_angle((1, 2), {1: 100., 2: 100.9}, True)

    def test_global_subspace_orientation_is_basis_invariant(self):
        wave = np.sin(np.linspace(0., np.pi, 15))
        a = np.column_stack([wave, 0*wave, 0*wave])
        b = np.column_stack([0*wave, wave, 0*wave])
        torsion = np.column_stack([wave*np.linspace(-1, 1, 15), 0*wave, 0*wave])
        expected, _ = step4.global_eigen_shape([a, b, torsion], 2, 30.)
        actual, meta = step4.global_eigen_shape([7*a-2*b, 3*a+4*b+torsion, torsion], 2, 30.)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertAlmostEqual(meta['bow_angle_deg'], 30.)
        with self.assertRaises(ValueError):
            step4.global_eigen_shape([b], 2, 0.)

    def test_single_global_mode_angle_only_fixes_the_sign(self):
        z = np.linspace(0., 3600., 37)
        wave = np.sin(np.pi*z/3600.)
        phi = np.column_stack([-wave, 0*wave, 0*wave])            # bows towards -X
        shape, meta = step4.global_eigen_shape([phi], 2, 0.)
        self.assertAlmostEqual(meta['bow_angle_deg'], 0., places=8)
        self.assertEqual(meta['eigenvector_sign_multiplier'], -1.)
        shape, meta = step4.global_eigen_shape([phi], 2, 180.)
        self.assertAlmostEqual(abs(meta['bow_angle_deg']), 180., places=8)

    def test_cpus_option_space_form_and_validation(self):
        import contextlib, io
        base = ['--run-dir', '.', '--output-cae', 'new.cae', '--cases', 'G1000']
        self.assertEqual(step4.parse_arguments(base+['--cpus', '4']).cpus, 4)
        args = step4.parse_arguments(base+['--cpus', '4', '--suggest'])
        self.assertEqual(args.suggest_cpus, '4')
        self.assertEqual(step4.parse_arguments(base+['--cpus', '4', '--suggest-cpus', '6']).suggest_cpus, '6')
        self.assertEqual(step4.parse_arguments(base).cpus, 'auto')
        for bad in ('0', '-2', 'x'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                step4.parse_arguments(base+['--cpus', bad])
        self.assertEqual(step4._early_option(['a', '--cpus', '3'], '--cpus'), '3')
        self.assertEqual(step4._early_option(['--cpus=3'], '--cpus'), '3')
        self.assertIsNone(step4._early_option(['--cpus'], '--cpus'))

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

    def test_individual_shares_and_odb_provenance_are_not_cluster_shares(self):
        import os, tempfile, json
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'mode_decomposition_GDLC')
            os.mkdir(out)
            with open(os.path.join(out, 'mode_participation.csv'), 'w') as f:
                f.write('mode,eigenvalue,G_pct,D_pct,L_pct,rel_G_pct,rel_D_pct,rel_L_pct,dominant,reliable_dominant,reliable_source\n'
                        '1,100,0,20,80,0,99,1,L,D,cluster\n')
            with open(os.path.join(out, 'basis_report.json'), 'w') as f:
                json.dump(dict(source_odb_sha256='current'), f)
            item = step4.gdlc_mode_shares(d, 1, 100., 'current')
            self.assertEqual(item['shares_pct']['L'], 80.)
            self.assertEqual(item['group_shares_pct']['D'], 99.)
            self.assertEqual(item['dominant'], 'L')
            self.assertEqual(step4.gdlc_mode_shares(d, 1, 100., 'old')['status'], 'stale_odb_mismatch')
            os.unlink(os.path.join(out, 'basis_report.json'))
            self.assertEqual(step4.gdlc_mode_shares(d, 1, 100., 'current')['status'], 'unverified_odb_source')

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
