import ast
import unittest
import numpy as np
import abaqus_step4_imperfections as step4


class ImperfectionTests(unittest.TestCase):
    def test_explicit_unassessed_reference_vector_does_not_require_pure_bow(self):
        args = step4.parse_arguments(['--run-dir', '.', '--cases', 'none',
            '--combo', 'G=0,0,L/1000', '--global-source', 'ref', '--global-mode', '1',
            '--skip-mode-quality-checks', '--output-cae', 'new.cae'])
        self.assertTrue(args.skip_mode_quality_checks)
        field = np.array([[0.,1.,0.], [0.,-1.,0.]])
        with self.assertRaisesRegex(ValueError, 'no mean bow'):
            step4.global_eigen_shape([field], 2, 0.)
        shape, meta = step4.global_eigen_shape([field], 2, 0., check_direction=False)
        self.assertAlmostEqual(float(np.max(np.linalg.norm(shape, axis=1))), 1.)
        self.assertEqual(meta['translation_ratio'], 0.)
        with self.assertRaisesRegex(ValueError, 'no transverse displacement'):
            step4.global_eigen_shape([field*0], 2, 0., check_direction=False)
        other = np.array([[1.,0.,0.], [-1.,0.,0.]])
        with self.assertRaisesRegex(ValueError, 'two independent bow directions'):
            step4.global_eigen_shape([field,other], 2, 0.)
        with unittest.mock.patch.object(step4, 'sum', side_effect=TypeError('CAE shadow'), create=True):
            mixed, meta = step4.global_eigen_shape([field,other], 2, 0., check_direction=False)
        self.assertTrue(np.isfinite(mixed).all())
        self.assertEqual(meta['translation_ratio'], 0.)

    def test_unexpanded_cmd_paths_are_rejected_before_build(self):
        import contextlib, io
        base = ['--run-dir', '.', '--cases', 'none', '--combo', 'G=0,0,L/1000',
                '--output-cae', 'new.cae']
        for option in ('--run-dir', '--local-source', '--global-source', '--output-cae'):
            argv = list(base)
            if option in argv:
                argv[argv.index(option)+1] = '%RUN_DIR%_REF_L'
            else:
                argv += [option, '%RUN_DIR%_REF_L']
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                step4.parse_arguments(argv)
            self.assertIn('%RUN_DIR%', caught.exception.step4_argument_error)
            self.assertIn('same CMD', caught.exception.step4_argument_error)

    def test_bending_local_reference_uses_95_percent_for_selected_vector(self):
        item = dict(status='ok', reference_check=dict(kind='L', passed=False),
                    shares_pct=dict(L=93.2, D=5.3, G=.5, C=1., O=0.))
        with self.assertRaisesRegex(ValueError, 'REF_L.*95'):
            step4.validate_bending_reference_mode(dict(type='bending'), 'local', 1, item)
        # Failure of a different mode in the reference must not reject a pure selected mode.
        item['shares_pct'] = dict(L=96., D=3., G=.5, C=.5, O=0.)
        step4.validate_bending_reference_mode(dict(type='bending'), 'local', 3, item)
        item['shares_pct']['L'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'REF_L'):
            step4.validate_bending_reference_mode(dict(type='bending'), 'local', 3, item)
        item['shares_pct']['L'] = 93.2
        with self.assertRaisesRegex(ValueError, 'REF_L'):
            step4.validate_bending_reference_mode(dict(type='compression'), 'local', 1, item)
        step4.validate_bending_reference_mode(dict(type='bending'), 'dist', 1, item)
        item.pop('reference_check')  # explicitly selected mixed original modes stay supported
        step4.validate_bending_reference_mode(dict(type='bending'), 'local', 1, item)

    def test_bending_reference_checks_when_cae_shadows_python_sum(self):
        import runpy

        def cae_sum(*args, **kwargs):
            raise TypeError("arg1; found 'generator', expecting a recognized type")

        namespace = runpy.run_path(step4.__file__, init_globals={'sum': cae_sum},
                                  run_name='cae_namespace_test')
        validate = namespace['validate_bending_reference_mode']
        for component, kind, shares in (
                ('local', 'L', dict(L=99.9783, G=0., D=0., C=0., O=.0217)),
                ('global', 'G', dict(G=100., L=0., D=0., C=0.))):
            with self.subTest(component=component):
                item = dict(status='ok', reference_check=dict(kind=kind), shares_pct=shares)
                validate(dict(type='bending'), component, 1, item)
        item = dict(status='ok', reference_check=dict(kind='L'),
                    shares_pct=dict(L=95., G=2., D=3., C=1.))
        with self.assertRaisesRegex(ValueError, 'REF_L'):
            validate(dict(type='bending'), 'local', 1, item)

    def test_load_case_option_is_an_optional_assertion_not_a_load_override(self):
        base=['--run-dir','.', '--cases','none','--combo','G=0,0,L/1000','--output-cae','new.cae']
        self.assertIsNone(step4.parse_arguments(base).load_case)
        self.assertEqual(step4.parse_arguments(base+['--load-case','bending']).load_case,'bending')
        self.assertEqual(step4.parse_arguments(base+['--load-case','compression']).load_case,'compression')

    def test_bending_global_reference_requires_95_percent_individual_G(self):
        item = dict(status='ok', reference_check=dict(kind='G', passed=False),
                    shares_pct=dict(G=94., D=3., L=3., C=0., O=0.))
        with self.assertRaisesRegex(ValueError, 'REF_G.*95'):
            step4.validate_bending_reference_mode(dict(type='bending'), 'global', 1, item)
        item['shares_pct']['G'] = 96.
        step4.validate_bending_reference_mode(dict(type='bending'), 'global', 2, item)
        item['shares_pct']['G'] = 94.
        step4.validate_bending_reference_mode(dict(type='compression'), 'global', 1, item)

    def test_bending_global_group_keeps_moment_direction_and_rejects_torsion_dominated_bow(self):
        ring = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        p = np.tile(ring, (7, 1))
        wave = np.repeat(np.sin(np.linspace(0., np.pi, 7)), len(ring))
        def fields(bow):
            torsion = np.c_[-p[:, 1], p[:, 0], np.zeros(len(p))]*wave[:, None]
            x, y = torsion.copy(), torsion.copy()
            x[:, 0] += bow*wave
            y[:, 1] += bow*wave
            return [x, y]
        lc = dict(type='bending', neutral_axis_angle_deg=0.)
        _, meta = step4.bending_global_eigen_shape(fields(3.), 2, lc)
        self.assertAlmostEqual(meta['bow_angle_deg'], 0., places=6)
        self.assertIn('bending_lateral_check', meta)
        with self.assertRaisesRegex(ValueError, 'moment|neutral axis'):
            step4.bending_global_eigen_shape(fields(3.), 2, lc, 90.)
        with self.assertRaisesRegex(ValueError, 'no net lateral bow'):
            step4.bending_global_eigen_shape(fields(.01), 2, lc)

    def test_bending_mode_reader_rejects_reversed_or_zero_load_eigenmodes(self):
        from types import SimpleNamespace as NS
        nodes=[NS(label=i+1,coordinates=p) for i,p in enumerate(
            [(0.,0.,0.),(1.,0.,0.),(1.,0.,10.),(0.,0.,10.)])]
        elem=NS(label=1,type='S4R',connectivity=(1,2,3,4),getNodes=lambda:nodes)
        inst=NS(name='P1',nodes=nodes,elements=[elem]);assembly=NS(instances={'P1':inst})
        values=[NS(instance=inst,nodeLabel=n.label,precision='SINGLE_PRECISION',
                   localCoordSystem=None,data=(0.,0.,0.)) for n in nodes]
        frame=NS(description='Mode 1: EigenValue = 1',fieldOutputs={'U':NS(values=values)})
        odb=NS(rootAssembly=assembly,steps={'Buckle':NS(frames=[frame])})
        args=NS(axis='z',step='Buckle')
        for value in (0.,-2.,float('nan')):
            frame.description='Mode 1: EigenValue = '+str(value)
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'positive'):
                step4.read_modes_and_mesh(NS(rootAssembly=assembly),odb,['P1'],args,modes=[1],require_positive=True)
        frame.description='Mode 1: EigenValue = 5'
        self.assertEqual(step4.read_modes_and_mesh(NS(rootAssembly=assembly),odb,['P1'],args,
                         modes=[1],require_positive=True)[-1][1],5.)
        frame.description='Mode 1: EigenValue = -2'
        self.assertEqual(step4.read_modes_and_mesh(NS(rootAssembly=assembly),odb,['P1'],args,modes=[1])[-1][1],-2.)

    def test_bending_sources_must_match_load_case_family_connections_and_reference_kind(self):
        import copy, json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            root=Path(root);run=root/'family_9_BEND000';run.mkdir()
            original=run/'BASE_BEND000.inp';original.write_text('*Heading\noriginal FE input\n')
            source=dict(section_segments={'P1':[[0.,0.,1.,0.]]},bolt_positions_mm=[25.,75.],
                        expected_links=8,thickness_mm=2.)
            build=dict(job_name='BASE_BEND000',load_case=dict(type='bending',neutral_axis_angle_deg=0.),source_inputs=source,
                       length_mm=100.,nodes=12,elements=8,rigid_links=8,boundary_conditions=9,loads=16)
            reference=dict(run_dir=str(run),build=build,source_inputs=source,paths=dict(inp=str(original)))
            ref=root/'family_9_BEND000_REF_G';ref.mkdir();job='BASE_BEND000_REF_G'
            for suffix,text in (('.inp','*Heading\nreference FE input\n'),('.odb','fake'),
                                ('.sta','THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n')):
                (ref/(job+suffix)).write_text(text)
            marker=dict(kind='G',source_run=run.name,source_inp=original.name,job=job,twist='free',
                        source_inp_sha256=step4.pipeline_contract.file_sha256(original),
                        input_sha256=step4.pipeline_contract.file_sha256(ref/(job+'.inp')))
            def save(b=build,m=marker):
                (ref/(job+'_build.json')).write_text(json.dumps(b))
                (ref/'gdlc_reference.json').write_text(json.dumps(m))
            save()
            for spec in (str(ref),str(ref/(job+'.odb'))):
                path,info=step4.resolve_mode_source(spec,None,str(run),reference=reference,component='global')
                self.assertEqual(Path(path).name,job+'.odb')
                self.assertEqual(info['kind'],'gdlc_reference_G')
            for field,value,message in (('type','compression','load case'),
                                        ('neutral_axis_angle_deg',45.,'load case')):
                bad=copy.deepcopy(build);bad['load_case'][field]=value;save(b=bad)
                with self.assertRaisesRegex(ValueError,message):
                    step4.resolve_mode_source(str(ref),None,str(run),reference=reference,component='global')
            bad=copy.deepcopy(build);bad['source_inputs']['bolt_positions_mm']=[25.,65.];save(b=bad)
            with self.assertRaisesRegex(ValueError,'configuration'):
                step4.resolve_mode_source(str(ref),None,str(run),reference=reference,component='global')
            for field,value in (('kind','L'),('source_run','family_8_BEND000'),
                                ('source_inp_sha256','stale'),('input_sha256','stale'),('twist','restrained')):
                bad=dict(marker);bad[field]=value;save(m=bad)
                with self.subTest(field=field),self.assertRaises(ValueError):
                    step4.resolve_mode_source(str(ref/(job+'.odb')),None,str(run),reference=reference,component='global')
            # Legacy reference metadata has no hashes; its copied build must still match.
            legacy={k:v for k,v in marker.items() if not k.endswith('sha256')};save(m=legacy)
            step4.resolve_mode_source(str(ref),None,str(run),reference=reference,component='global')
            (ref/'gdlc_reference.json').unlink()
            for spec in (str(ref),str(ref/(job+'.odb'))):
                with self.assertRaisesRegex(ValueError,'provenance'):
                    step4.resolve_mode_source(spec,None,str(run),reference=reference,component='global')
            save(m=legacy);inp=ref/(job+'.inp');text=inp.read_text();inp.unlink()
            with self.assertRaisesRegex(ValueError,'INP'):
                step4.resolve_mode_source(str(ref/(job+'.odb')),None,str(run),reference=reference,component='global')
            inp.write_text(text)
            # The compression path keeps the existing same-mesh source flexibility.
            compression=dict(reference,build=dict(load_case=dict(type='compression')))
            save(b=dict(load_case=dict(type='compression')),m=dict(marker,source_run='old_run'))
            step4.resolve_mode_source(str(ref),None,str(run),reference=compression,component='global')
            (ref/(job+'_build.json')).unlink();(ref/'gdlc_reference.json').unlink()
            with self.assertRaisesRegex(ValueError,'provenance'):
                step4.resolve_mode_source(str(ref),None,str(run),reference=reference,component='global')

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
            import json
            with open(os.path.join(d, 'mode_decomposition_GDLC', 'basis_report.json'), 'w') as f:
                json.dump(dict(reference_check=dict(kind='L', passed=False, min_L_pct=93.2)), f)
            self.assertFalse(step4.gdlc_mode_shares(d, 1, 268.78)['reference_check']['passed'])

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

    def test_bending_global_mode_must_be_one_half_wave_lateral_torsional_along_neutral_axis(self):
        # square ring of nodes along z; lateral-torsional field: translation along the lateral
        # direction plus a twist about a centre on the tension side (y = -300)
        ring = np.array([[x, y] for x in (-100., 0., 100.) for y in (-100., 0., 100.) if (x, y) != (0., 0.)])
        z = np.linspace(0., 3600., 31)
        xyz = np.array([[x, y, zz] for zz in z for x, y in ring])
        def ltb(halfwaves, twist=1e-3, centre=(0., -300.)):
            s = np.sin(halfwaves*np.pi*xyz[:, 2]/3600.)
            u = np.zeros_like(xyz)
            u[:, 0] = -s*twist*(xyz[:, 1]-centre[1])
            u[:, 1] = s*twist*(xyz[:, 0]-centre[0])
            return u
        for sign in (1., -1.):                                      # sign of the eigenvector is arbitrary
            shape, meta = step4.orient_bending_global(*step4.global_eigen_shape([sign*ltb(1)], 2, None),
                                                      axis=2, angle_degrees=0.)
            self.assertGreater(shape[:, 0].mean(), 0.)               # 'p' = towards +lateral direction
            self.assertAlmostEqual(meta['bow_angle_deg'], 0., places=6)
            self.assertGreater(meta['translation_ratio'], .5)
            self.assertAlmostEqual(float(np.max(np.linalg.norm(shape, axis=1))), 1.)
        for bad, why in ((ltb(2), 'no net lateral bow'),                       # even half-waves
                         (ltb(1, centre=(0., 0.)), 'no net lateral bow'),     # pure torsion
                         (ltb(1, centre=(300., 0.)), 'away from the lateral')):  # bow along y
            with self.assertRaisesRegex(ValueError, why):
                step4.orient_bending_global(*step4.global_eigen_shape([bad], 2, None), axis=2, angle_degrees=0.)


class LocalReferenceQualityTests(unittest.TestCase):
    @staticmethod
    def classified(shares=None, status='ok'):
        return dict(status=status, reference_check=dict(kind='L', passed=False),
                    shares_pct=shares if shares is not None else dict(G=0., D=0., L=98.3, C=0., O=1.7))

    def test_local_reference_criterion_applies_to_every_load_case(self):
        for load_type in ('compression', 'bending', 'combined'):
            with self.subTest(load_type=load_type):
                step4.validate_bending_reference_mode(dict(type=load_type), 'local', 2, self.classified())
                impure = self.classified(dict(G=1., D=3., L=94., C=1., O=1.))
                with self.assertRaisesRegex(ValueError, 'REF_L.*95'):
                    step4.validate_bending_reference_mode(dict(type=load_type), 'local', 1, impure)

    def test_local_reference_accepts_threshold_and_rounded_total(self):
        for shares in (dict(G=2., D=3., L=95., C=0., O=0.),
                       dict(G=0., D=0., L=95., C=0., O=5.),
                       dict(G=0., D=0., L=98.3078, C=0., O=1.6921999)):
            with self.subTest(shares=shares):
                step4.validate_bending_reference_mode(dict(type='compression'), 'local', 1,
                                                      self.classified(shares))

    def test_local_reference_rejects_incomplete_or_malformed_shares(self):
        malformed = [None, {}, dict(G=0., D=0., L=100., C=0.),
                     dict(G=-1., D=0., L=100., C=0., O=1.),
                     dict(G=0., D=0., L=101., C=0., O=0.),
                     dict(G=0., D=0., L=96., C=0., O=0.),
                     dict(G=0., D=0., L=95., C=0., O=5.001)]
        for key in 'GDLCO':
            for bad in (float('nan'), float('inf'), None, 'bad'):
                shares = dict(G=0., D=0., L=98.3, C=0., O=1.7)
                shares[key] = bad
                malformed.append(shares)
        for shares in malformed:
            with self.subTest(shares=shares), self.assertRaisesRegex(ValueError, 'REF_L'):
                item = self.classified()
                item['shares_pct'] = shares
                step4.validate_bending_reference_mode(dict(type='compression'), 'local', 1, item)

    def test_unavailable_classification_and_original_mixed_sources_keep_warning_path(self):
        original = dict(status='ok', shares_pct=dict(L=20., D=80.))
        missing_marker = self.classified()
        missing_marker.pop('reference_check')
        for item in (None, original, missing_marker,
                     self.classified({}, status='stale_odb_mismatch'),
                     self.classified({}, status='unverified_odb_source')):
            with self.subTest(item=item):
                step4.validate_bending_reference_mode(dict(type='compression'), 'local', 1, item)

    def test_nonlocal_and_nonbending_global_behavior_is_preserved(self):
        step4.validate_bending_reference_mode(dict(type='bending'), 'dist', 1, self.classified({}))
        global_item = dict(status='ok', reference_check=dict(kind='G'), shares_pct=dict(G=20.))
        for load_type in ('compression', 'combined'):
            with self.subTest(load_type=load_type):
                step4.validate_bending_reference_mode(dict(type=load_type), 'global', 1, global_item)


if __name__ == '__main__':
    unittest.main()
