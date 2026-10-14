"""CLI/data-flow checks; never launch the Abaqus solver."""
import contextlib
import csv
import io
import math
import os
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
import abaqus_complete_model_m20 as builder


class ModelInputTests(unittest.TestCase):
    @contextlib.contextmanager
    def input_folder(self, changes=None, seam_change=None, segment_change=None):
        member = dict(E_MPa=200000., nu=.3, L_mm=100., t_mm=.2,
                      gap_mm=math.sqrt(10.)-.2, bolt_row_mm=1., n_bolts=2,
                      end_start_mm=25., end_end_mm=25., pitch_mm=50.)
        member.update(changes or {})
        segments = [[1, 2., 1., 2., 3., .2], [2, -1., 2., -3., 2., .2],
                    [3, -2., -1., -2., -3., .2], [4, 1., -2., 3., -2., .2]]
        seams = [[1, 1, 2, 2., 3., -1., 2.], [2, 2, 3, -3., 2., -2., -1.],
                 [3, 3, 4, -2., -3., 1., -2.], [4, 4, 1, 3., -2., 2., 1.]]
        if seam_change:
            seam_change(seams)
        if segment_change:
            segment_change(segments)
        with tempfile.TemporaryDirectory() as folder:
            for name, rows in (('builtup_segments.csv', segments),
                               ('builtup_seams.csv', seams),
                               ('builtup_bolts_y.csv', [[25.], [75.]])):
                with open(os.path.join(folder, name), 'w', newline='') as stream:
                    csv.writer(stream).writerows(rows)
            with open(os.path.join(folder, 'builtup_member.csv'), 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(member))
                writer.writeheader()
                writer.writerow(member)
            yield folder

    def test_valid_export_retains_exact_bolt_positions(self):
        with self.input_folder() as folder:
            self.assertEqual(builder.read_model_inputs(folder)[5], [25., 75.])

    def test_unstable_isotropic_material_is_rejected_before_cae(self):
        for key, value in (('E_MPa', 0.), ('E_MPa', -1.), ('nu', -.999999-1.), ('nu', .5)):
            with self.subTest(key=key, value=value), self.input_folder({key: value}) as folder:
                with self.assertRaises(ValueError):
                    builder.read_model_inputs(folder)

    def test_nonfinite_member_metadata_is_rejected(self):
        for key in ('t_mm', 'gap_mm', 'end_start_mm', 'end_end_mm', 'pitch_mm', 'bolt_row_mm'):
            with self.subTest(key=key), self.input_folder({key: 'nan'}) as folder:
                with self.assertRaises(ValueError):
                    builder.read_model_inputs(folder)

    def test_check_inputs_rejects_non_c4_geometry(self):
        def shift_third_piece(rows):
            rows[2][1] += .1
            rows[2][3] += .1
        with self.input_folder(segment_change=shift_third_piece) as folder:
            with self.assertRaises(ValueError):
                builder.read_model_inputs(folder)

    def test_seam_must_attach_to_its_piece(self):
        def shift_seam(rows):
            rows[0][3] += .1
            rows[0][5] += .1  # preserves gap; both points leave their walls
        with self.input_folder(seam_change=shift_seam) as folder:
            with self.assertRaises(ValueError):
                builder.read_model_inputs(folder)

    def test_duplicate_seam_endpoints_cannot_leave_disconnected_pieces(self):
        def duplicate_seams(rows):
            for row in rows[1:]:
                row[1:] = rows[0][1:]
        with self.input_folder(seam_change=duplicate_seams) as folder:
            with self.assertRaises(ValueError):
                builder.read_model_inputs(folder)


class PipelineTests(unittest.TestCase):
    def test_automatic_pipeline_has_no_shell_energy_stage(self):
        self.assertFalse(hasattr(builder, 'complete_shell_report'))
        with open(builder.__file__) as stream:
            source = stream.read()
        self.assertNotIn('ModalShellDiagnostics', source)
        self.assertNotIn("variables=('S', 'E', 'SF', 'SE')", source)
        self.assertNotIn('abaqus_modal_shell_energy', source)
        self.assertIn("variables=('U', 'UR')", source)

    def test_legacy_detailed_flag_is_accepted_but_classification_pipeline_stays_lean(self):
        args = builder.parse_arguments(['--buckle-output', 'detailed', '--modal-audit'])
        self.assertEqual(args.buckle_output, 'detailed')
        self.assertEqual(args.nodal_precision, 'full')
        self.assertTrue(args.modal_audit)
        self.assertEqual(args.portable_export, 'parquet')
        self.assertEqual(args.portable_modes_per_shard, 8)
        self.assertEqual(args.portable_export_workers, 'auto')
        self.assertEqual(builder.parse_arguments(['--nodal-precision', 'single']).nodal_precision, 'single')

    def test_full_run_command_accepts_longitudinal_lines_with_modal_options(self):
        args = builder.parse_arguments(['--mesh-mm', '5', '--n-modes', '250',
            '--n-vectors', '500', '--max-iterations', '1250', '--cpus', '8',
            '--buckle-output', 'detailed', '--nodal-precision', 'full',
            '--longitudinal-lines', '2', '--modal-audit'])
        self.assertEqual(args.longitudinal_lines, 2)
        self.assertEqual((args.buckle_output, args.nodal_precision), ('detailed', 'full'))
        self.assertTrue(args.modal_audit)

    def test_output_root_uses_input_folder_name_and_preserves_previous_run(self):
        with tempfile.TemporaryDirectory() as root:
            args = builder.parse_arguments(['--builtup-dir', os.path.join(root, 'input', 'Section M80'),
                                            '--output-root', os.path.join(root, 'results')])
            first = builder.prepare_output_directory(args)
            self.assertEqual(first, os.path.join(root, 'results', 'Section M80'))
            marker = os.path.join(first, 'existing.odb')
            with open(marker, 'w') as stream:
                stream.write('keep')
            second = builder.prepare_output_directory(args)
            self.assertEqual(second, first+'_run02')
            with open(marker) as stream:
                self.assertEqual(stream.read(), 'keep')

    def test_output_options_are_mutually_exclusive(self):
        for argv in (['--output-root', 'results', '--output-dir', 'exact'],
                     ['--resume-post', 'old run', '--output-root', 'results']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.parse_arguments(argv)

    def test_exact_output_directory_remains_supported(self):
        with tempfile.TemporaryDirectory() as root:
            exact = os.path.join(root, 'exact location')
            args = builder.parse_arguments(['--output-dir', exact])
            self.assertEqual(builder.prepare_output_directory(args), exact)

    def test_unknown_status_accepts_successful_output_files(self):
        with tempfile.TemporaryDirectory() as root:
            odb = os.path.join(root, 'Job.odb')
            for suffix, content in (('.odb', 'placeholder'),
                                    ('.sta', 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY')):
                with open(os.path.splitext(odb)[0]+suffix, 'w') as stream:
                    stream.write(content)
            self.assertIn('.sta', builder.require_completed(types.SimpleNamespace(status=None), odb))

    def test_unknown_status_without_success_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):
                builder.require_completed(types.SimpleNamespace(status=None), os.path.join(root, 'Job.odb'))

    def test_missing_odb_reports_solver_cpu_error(self):
        with tempfile.TemporaryDirectory() as root:
            odb = os.path.join(root, 'Job.odb')
            with open(os.path.join(root, 'Job.msg'), 'w') as stream:
                stream.write('*** Error: Abaqus Error: The number of cpus (8) exceeds the number of cpus available (4).\n')
            with self.assertRaisesRegex(RuntimeError, r'cpus \(8\).*available \(4\)'):
                builder.require_completed(types.SimpleNamespace(status=None), odb)

    def test_cli_resolves_oversized_cpu_request_before_build(self):
        import runtime_resources as resources
        inventory = resources.ResourceInventory(8, 4, 16*1024**3, 10*1024**3, [])
        with mock.patch.object(resources, 'detect_resources', return_value=inventory):
            with self.assertWarns(RuntimeWarning):
                args = builder.parse_arguments(['--cpus', '8', '--build-only'])
        self.assertEqual(args.cpus, 4)
        self.assertEqual(args.gpus, 0)

    def test_completed_status_cannot_accept_missing_or_locked_odb(self):
        with tempfile.TemporaryDirectory() as root:
            odb = os.path.join(root, 'Job.odb')
            job = types.SimpleNamespace(status='COMPLETED')
            with self.assertRaises(RuntimeError):
                builder.require_completed(job, odb)
            with open(odb, 'w') as stream:
                stream.write('placeholder')
            with open(os.path.join(root, 'Job.lck'), 'w') as stream:
                stream.write('locked')
            with self.assertRaises(RuntimeError):
                builder.require_completed(job, odb)

    def test_active_or_aborted_or_locked_job_cannot_use_file_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            odb = os.path.join(root, 'Job.odb')
            for suffix, content in (('.odb', 'placeholder'),
                                    ('.sta', 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY')):
                with open(os.path.splitext(odb)[0]+suffix, 'w') as stream:
                    stream.write(content)
            for status in ('RUNNING', 'ABORTED', 'TERMINATED'):
                with self.subTest(status=status), self.assertRaises(RuntimeError):
                    builder.require_completed(types.SimpleNamespace(status=status), odb)
            with open(os.path.splitext(odb)[0]+'.lck', 'w') as stream:
                stream.write('locked')
            with self.assertRaises(RuntimeError):
                builder.require_completed(types.SimpleNamespace(status=None), odb)

    def test_failure_log_overrides_success_marker(self):
        with tempfile.TemporaryDirectory() as root:
            odb = os.path.join(root, 'Job.odb')
            for suffix, content in (('.odb', 'placeholder'),
                    ('.sta', 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY'),
                    ('.log', 'Abaqus JOB Job ABORTED')):
                with open(os.path.splitext(odb)[0]+suffix, 'w') as stream:
                    stream.write(content)
            with self.assertRaises(RuntimeError):
                builder.require_completed(types.SimpleNamespace(status=None), odb)

    def test_cae_execfile_without_file_variable(self):
        namespace = {'__name__': 'cae_script_probe'}
        with open(builder.__file__) as stream:
            source = stream.read()
        exec(compile(source, builder.__file__, 'exec'), namespace)
        self.assertEqual(namespace['SCRIPT_DIR'], os.path.dirname(os.path.abspath(builder.__file__)))

    def test_explicit_settings_and_path_with_spaces(self):
        args = builder.parse_arguments(['--builtup-dir', 'input folder',
            '--mesh-mm', '12.5', '--n-modes', '100', '--n-vectors', '200',
            '--max-iterations', '300', '--build-only'])
        self.assertEqual(args.builtup_dir, os.path.abspath('input folder'))
        self.assertEqual((args.mesh_mm, args.n_modes, args.n_vectors,
                          args.max_iterations), (12.5, 100, 200, 300))
        self.assertTrue(args.build_only)

    def test_cae_separator(self):
        args = builder.parse_arguments(['cae', 'noGUI=script.py', '--',
                                        '--n-modes', '56', '--n-vectors', '100'])
        self.assertEqual(args.n_modes, 56)
        self.assertEqual(args.n_vectors, 100)

    def test_actual_cae_kernel_arguments(self):
        args = builder.parse_arguments(['-cae', '-noGUI', 'script.py', '-tmpdir',
            'temp folder', '-lmlog', 'ON', '--n-modes', '100', '--n-vectors', '200'])
        self.assertEqual((args.n_modes, args.n_vectors), (100, 200))

    def test_larger_mode_request_has_valid_automatic_vectors(self):
        args = builder.parse_arguments(['--n-modes', '100'])
        self.assertGreaterEqual(args.n_vectors, 100)

    def test_invalid_numeric_settings_rejected(self):
        for options in (['--mesh-mm', 'nan'], ['--mesh-mm', '0'],
                        ['--n-modes', '0'], ['--n-modes', '100', '--n-vectors', '30'],
                        ['--max-iterations', '0'], ['--cpus', '0']):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    builder.parse_arguments(options)

    def test_input_summary_persists_physical_section_segments(self):
        pieces = {
            1: [(0., 0., 10., 0.)],
            2: [(0., 0., 0., 10.)],
            3: [(0., 0., -10., 0.)],
            4: [(0., 0., 0., -10.)],
        }
        member = dict(L_mm='3600', gap_mm='10', bolt_row_mm='15')
        data = (pieces, 3.0, 200000.0, 0.3,
                [[1., 1., 2., 0., 0., 10., 0.],
                 [2., 2., 3., 0., 0., 10., 0.],
                 [3., 3., 4., 0., 0., 10., 0.],
                 [4., 4., 1., 0., 0., 10., 0.]],
                [25., 225.], member)
        with mock.patch.object(builder, 'BUILTUP_DIR', os.path.abspath('source')):
            summary = builder.input_summary(data)
        self.assertEqual(sorted(summary['section_segments']), ['P1', 'P2', 'P3', 'P4'])
        self.assertEqual(summary['section_segments']['P1'][0], [0., 0., 10., 0.])

    def test_postprocessing_uses_this_run_and_all_modes(self):
        report = dict(odb=os.path.abspath('new run/model.odb'), reference_stress_MPa=1.0)
        argv = builder.postprocess_arguments(report)
        self.assertEqual(argv[0], report['odb'])
        self.assertEqual(argv[argv.index('--modes')+1], '0')
        self.assertEqual(float(argv[argv.index('--sigma-ref')+1]), 1.0)

    def test_aborted_job_is_not_accepted(self):
        class Job:
            status = 'ABORTED'
        with self.assertRaisesRegex(RuntimeError, 'ABORTED'):
            builder.require_completed(Job())

    def exercise_pipeline(self, final_status):
        # Replace only the external CAE/solver and numerical-postprocessor boundaries.
        # Real CLI, shared settings, file paths, sequencing and failure gate are exercised.
        with tempfile.TemporaryDirectory() as root:
            output = os.path.join(root, 'result with spaces')
            with open(os.path.join(root, 'abaqus_modal_wavelengths.py'), 'w') as stream:
                stream.write("import os\ndef main(argv):\n"
                    "    assert os.path.isfile(argv[0])\n"
                    "    assert argv[argv.index('--modes')+1] == '0'\n"
                    "    with open('post_called.txt', 'w') as stream:\n"
                    "        stream.write(argv[0])\n"
                    "    return {'available_modes': 100, 'processed_modes': 100}\n")
            with open(os.path.join(root, 'abaqus_modal_report.py'), 'w') as stream:
                stream.write("import os\ndef main(argv):\n"
                    "    assert os.path.isfile('post_called.txt')\n"
                    "    assert argv[0] == os.path.abspath('CurrentRun_modal_wavelengths_report.json')\n"
                    "    with open('enhanced_called.txt', 'w') as stream:\n"
                    "        stream.write(argv[0])\n"
                    "    return {'processed_modes': 100}\n")
            class FakeJob:
                name = 'CurrentRun'
                status = 'CREATED'
                def submit(self, **kwargs):
                    self.status = 'SUBMITTED'
                def waitForCompletion(self):
                    self.status = final_status
                    if final_status in ('COMPLETED', None):
                        with open(self.name+'.odb', 'w') as stream:
                            stream.write('test boundary placeholder')
                        with open(self.name+'.sta', 'w') as stream:
                            stream.write('THE ANALYSIS HAS COMPLETED SUCCESSFULLY')
            def fake_build(inputs, cpus, buckle_output='standard', nodal_precision='full', gpus=0):
                self.assertEqual(builder.LONGITUDINAL_LINES, 2)
                self.assertEqual((buckle_output, nodal_precision), ('detailed', 'full'))
                self.assertEqual((builder.MESH_MM, builder.N_MODES, builder.N_VECTORS,
                                  builder.MAX_ITERATIONS), (12.5, 100, 200, 400))
                self.assertEqual(builder.BUILTUP_DIR, os.path.abspath(root))
                return FakeJob(), dict(odb=os.path.abspath('CurrentRun.odb'), reference_stress_MPa=1.)
            argv = ['--builtup-dir', root, '--output-dir', output, '--mesh-mm', '12.5',
                    '--n-modes', '100', '--n-vectors', '200', '--max-iterations', '400',
                    '--longitudinal-lines', '2', '--buckle-output', 'detailed',
                    '--nodal-precision', 'full', '--portable-export', 'npz',
                    '--portable-modes-per-shard', '4', '--portable-export-workers', '6',
                    '--modal-audit']
            defaults = {k: getattr(builder, k) for k in
                ('BUILTUP_DIR', 'MESH_MM', 'N_MODES', 'N_VECTORS', 'MAX_ITERATIONS',
                 'LONGITUDINAL_LINES')}
            def fake_audit(run_dir):
                self.assertEqual(run_dir, output)
                self.assertTrue(os.path.isfile(os.path.join(run_dir, 'enhanced_called.txt')))
                return {'status': 'audit_boundary_complete'}
            def fake_portable(odb_path, run_dir, output_format='parquet',
                              modes_per_shard=8, parquet_python=None,
                              parquet_backend=None, workers='auto'):
                self.assertEqual(odb_path, os.path.join(output, 'CurrentRun.odb'))
                self.assertEqual(run_dir, output)
                self.assertEqual((output_format, modes_per_shard), ('npz', 4))
                self.assertIsNone(parquet_backend)
                self.assertEqual(workers, '6')
                return {'status': 'portable_boundary_complete'}
            with mock.patch.dict(builder.__dict__, defaults), \
                 mock.patch.object(builder, 'SCRIPT_DIR', root), \
                 mock.patch.object(builder, 'read_model_inputs', return_value=()), \
                 mock.patch.object(builder, 'input_summary', return_value={}), \
                 mock.patch.object(builder, 'build', side_effect=fake_build), \
                 mock.patch.object(builder, 'run_modal_audit', side_effect=fake_audit), \
                 mock.patch.object(builder, 'run_portable_export', side_effect=fake_portable), \
                 mock.patch.dict(sys.modules, {'abaqusConstants': types.SimpleNamespace(ON=True)}), \
                 contextlib.redirect_stdout(io.StringIO()):
                if final_status in ('COMPLETED', None):
                    builder.main(argv)
                else:
                    with self.assertRaisesRegex(RuntimeError, final_status):
                        builder.main(argv)
            with open(os.path.join(output, 'pipeline_status.json')) as stream:
                state = json.load(stream)
            if final_status in ('COMPLETED', None):
                self.assertEqual(state['status'], 'COMPLETED')
                self.assertEqual(state['enhanced_report']['processed_modes'], 100)
                self.assertEqual(state['settings']['longitudinal_lines'], 2)
                self.assertEqual(state['modal_audit']['status'], 'audit_boundary_complete')
                self.assertEqual(state['portable_export']['status'], 'portable_boundary_complete')
                self.assertTrue(os.path.isfile(os.path.join(output, 'enhanced_called.txt')))
                with open(os.path.join(output, 'post_called.txt')) as stream:
                    self.assertEqual(stream.read(), os.path.join(output, 'CurrentRun.odb'))
            else:
                self.assertEqual(state['status'], 'FAILED')
                self.assertFalse(os.path.exists(os.path.join(output, 'post_called.txt')))
                self.assertFalse(os.path.exists(os.path.join(output, 'enhanced_called.txt')))

    def test_pipeline_passes_new_odb_to_postprocessor_after_completion(self):
        self.exercise_pipeline('COMPLETED')

    def test_pipeline_never_postprocesses_aborted_analysis(self):
        self.exercise_pipeline('ABORTED')

    def test_pipeline_recovers_unknown_cae_status_from_files(self):
        self.exercise_pipeline(None)

    def test_resume_routes_without_build_or_source_csv_read(self):
        with mock.patch.object(builder, 'resume_postprocessing', return_value='recovered') as resume, \
             mock.patch.object(builder, 'read_model_inputs', side_effect=AssertionError('Unexpected CSV read')), \
             mock.patch.object(builder, 'build', side_effect=AssertionError('Unexpected rebuild')):
            self.assertEqual(builder.main(['--resume-post', 'previous run',
                                            '--portable-export', 'off']), 'recovered')
            resume.assert_called_once_with(
                'previous run', modal_audit=False, portable_export='off',
                portable_modes_per_shard=8, parquet_python=None, parquet_backend=None,
                portable_export_workers='auto')

    def test_bending_options_output_suffix_and_solver(self):
        with tempfile.TemporaryDirectory() as root:
            argv = ['--builtup-dir', os.path.join(root, 'input', 'Section M80'), '--output-root',
                    os.path.join(root, 'results')]
            args = builder.parse_arguments(argv)
            self.assertEqual((args.load_case, args.eigensolver), ('compression', 'auto'))
            self.assertEqual(builder.prepare_output_directory(args), os.path.join(root, 'results', 'Section M80'))
            self.assertEqual(builder.effective_eigensolver('compression', 'auto'), 'subspace')
            args = builder.parse_arguments(argv+['--load-case', 'bending', '--bending-axis-deg', '-315'])
            self.assertEqual(args.bending_axis_deg, 45.)
            self.assertEqual(builder.prepare_output_directory(args),
                             os.path.join(root, 'results', 'Section M80_BEND045'))
            self.assertEqual(builder.effective_eigensolver('bending', 'auto'), 'lanczos')
            self.assertEqual(builder.effective_eigensolver('bending', 'subspace'), 'subspace')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            builder.parse_arguments(['--bending-axis-deg', '45'])      # needs --load-case bending

    def test_load_case_definition_records_convention_and_pm_axes(self):
        square = {1: [(-100., 100., 100., 100.)], 2: [(-100., -100., -100., 100.)],
                  3: [(100., -100., -100., -100.)], 4: [(100., 100., 100., -100.)]}
        data = (square, 2.0, 200000.0, 0.3, [], [], {})
        compression = builder.load_case_definition(data, 'compression', 0.)
        self.assertEqual((compression['type'], compression['tag']), ('compression', ''))
        self.assertAlmostEqual(compression['area_mm2'], 1600.)
        bending = builder.load_case_definition(data, 'bending', 45.)
        self.assertEqual(bending['tag'], '_BEND045')
        self.assertEqual(bending['bending_axes_for_PM']['required_axes_deg'], [0., 45.])
        self.assertAlmostEqual(bending['reference_stress_MPa'], 1.0)
        self.assertEqual(bending['symmetry_class_axis_deg'], 45.)
        with open(builder.__file__) as stream:
            source = stream.read()
        self.assertIn("eigensolver=LANCZOS, minEigen=0.0 if LOAD_CASE == 'bending' else None", source)
        self.assertIn("expected_loads = 8 if LOAD_CASE == 'compression' else load_definition['n_loads']", source)


class FamilyPipelineTests(unittest.TestCase):
    def test_portable_repo_default_and_explicit_launcher(self):
        from pathlib import Path
        import abaqus_family_pipeline as pipeline
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            previous = os.getcwd()
            try:
                os.chdir(d)
                args = pipeline.parse_arguments(['--family-root', 'families', '--output-root', 'results',
                    '--abaqus-cmd', 'custom abaqus.bat'])
            finally:
                os.chdir(previous)
            self.assertEqual(args.repo, Path(pipeline.__file__).resolve().parent)
            self.assertEqual(args.family_root, Path(d).resolve()/'families')
            self.assertEqual(args.output_root, Path(d).resolve()/'results')
            self.assertEqual(args.abaqus_cmd, 'custom abaqus.bat')

    def test_wrong_explicit_repo_is_rejected_without_silent_fallback(self):
        import abaqus_family_pipeline as pipeline
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            args = pipeline.parse_arguments(['--family-root', d, '--output-root', d, '--repo', d])
            with self.assertRaisesRegex(ValueError, '--repo'):
                pipeline.preflight(args)

    def test_selected_launcher_reaches_reference_commands_and_queue(self):
        with self.completed_family() as (pipeline,args,source,run,output):
            args.abaqus_cmd = str(run.parent/'Solver tools'/'chosen abaqus.bat')
            args.resume = False
            for kind in ('L','G'):
                report = run.with_name(run.name+'_REF_'+kind)/'mode_decomposition_GDLC'
                report.mkdir()
                (report/'mode_participation.csv').write_text('mode\n1\n')
            calls = []
            def stop_at_step4(command, cwd, label):
                calls.append((command, label))
                if label.endswith('| STEP4'):
                    raise RuntimeError('stop before model modification')
            with mock.patch.object(pipeline, 'preflight', return_value=[(source,run,{})]), \
                    mock.patch.object(pipeline.shutil, 'which', return_value=args.abaqus_cmd), \
                    mock.patch.object(pipeline, 'completed_reference', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'stop before model modification'):
                    pipeline.run_pipeline(args, runner=stop_at_step4)
            reference_calls = [cmd for cmd,label in calls if 'REFERENCES' in label or '| SOLVE REF_' in label]
            self.assertEqual(len(reference_calls), 3)
            for cmd in reference_calls:
                self.assertEqual(cmd[cmd.index('--abaqus-cmd')+1], args.abaqus_cmd)
                if '--solve-reference-models' in cmd:
                    self.assertIn('--solve-only', cmd)
                    self.assertNotIn('--stiffness', cmd)
                    self.assertNotIn('--export', cmd)
            self.assertIn('--skip-mode-quality-checks', calls[-1][0])
            calls.clear()
            args.resume = True
            def stop_at_queue(command, cwd, label):
                calls.append(command)
                raise RuntimeError('stop before solver')
            with mock.patch.object(pipeline, 'preflight', return_value=[(source,run,dict(step4=True,step5=True))]), \
                    mock.patch.object(pipeline.shutil, 'which', return_value=args.abaqus_cmd):
                with self.assertRaisesRegex(RuntimeError, 'stop before solver'):
                    pipeline.run_pipeline(args, runner=stop_at_queue)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index('--abaqus-cmd')+1], args.abaqus_cmd)

    @contextlib.contextmanager
    def completed_family(self):
        from pathlib import Path
        import abaqus_family_pipeline as pipeline
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            root=Path(d)
            source=root/'inputs'/'section'/'family_1';source.mkdir(parents=True)
            for name in pipeline.INPUT_FILES:(source/name).write_text('input')
            args=pipeline.parse_arguments(['--family-root',str(source.parent),'--output-root',str(root/'out'),
                '--repo',os.path.dirname(__file__),'--resume'])
            run=args.output_root/source.parent.name/source.name;run.mkdir(parents=True)
            state=dict(status='BUILT',settings=dict(vars(args),builtup_dir=str(source)),
                       source_inputs=dict(thickness_mm=2.),build=dict(job_name='BASE',length_mm=3600.))
            state['settings']={k:str(v) if isinstance(v,Path) else v for k,v in state['settings'].items()}
            (run/'pipeline_status.json').write_text(json.dumps(state))
            for ext in ('.inp','.cae'):(run/('BASE'+ext)).write_text('base')
            for kind in ('L','G'):
                ref=Path(str(run)+'_REF_'+kind);ref.mkdir()
                (ref/'gdlc_reference.json').write_text(json.dumps(dict(kind=kind,source_run=run.name,
                    source_inp='BASE.inp',n_modes=20,job='REF_'+kind)))
                (ref/('REF_'+kind+'.inp')).write_text('reference')
                (ref/('REF_'+kind+'.odb')).write_text('solved reference')
            cae=run/'imperfection'/'Step4_LG.cae';cae.parent.mkdir();cae.write_text('step4')
            output=cae.parent/'Step5_GMNIA_FY240_LG';output.mkdir()
            (output/'Step5_GMNIA_FY240.cae').write_text('step5')
            (output/'run_step5_queue_70_pct.bat').write_text('queue')
            reference={key:pipeline.file_sha256(run/name) for key,name in
                [('source_inp_sha256','BASE.inp'),('source_cae_sha256','BASE.cae'),
                 ('pipeline_status_sha256','pipeline_status.json')]}
            spec=dict(settings=dict(local_mode=1,global_modes=[1,2],global_angle_deg=0.,
                      local_source=str(run)+'_REF_L',global_source=str(run)+'_REF_G'),
                      component_sources={k:dict(odb=str(Path(str(run)+'_REF_'+kind)/('REF_'+kind+'.odb')),
                      odb_sha256=pipeline.file_sha256(Path(str(run)+'_REF_'+kind)/('REF_'+kind+'.odb')))
                      for k,kind in [('local','L'),('global','G')]})
            results=[]
            for name in sorted(pipeline.EXPECTED_JOBS):
                (output/(name+'.inp')).write_text('input '+name)
                perfect='PERFECT' in name
                info=dict(stage=5,job_name=name,input_sha256=pipeline.file_sha256(output/(name+'.inp')),
                    reference_pipeline=reference,reference_force_N_per_end=1000.,end_area_mm2=1000./240.,load_case=dict(type='compression'),
                    settings=dict(source_cae=str(cae),fy=240.,initial_arc=.01,min_arc=1e-8,max_arc=.05,
                        max_increments=5000,field_frequency=1,postpeak_stop_ratio=.7,stop_method='monitor'),
                    source_step4=dict(specification=spec,case=dict(local_component_mm=0. if perfect else -.68,
                        distortional_component_mm=0.,global_component_mm=0. if perfect else 3.6)))
                (output/(name+'_step5.json')).write_text(json.dumps(info))
                (output/(name+'.odb')).write_text('not opened')
                stem=output/(name+'_force_displacement')
                stem.with_suffix('.csv').write_text('frame,lpf,shortening_mm,force_kN\n0,0,0,0\n1,1,1,1\n2,.6,.9,.6\n')
                stem.with_suffix('.json').write_text(json.dumps(dict(model=name,step='GMNIA',frames=3,partial=False)))
                results.append(dict(model=name,outcome='CRITERION_REACHED',criterion_reached=True,
                                    solver_exit_code=0,partial=False,force_kN=1.))
            (output/'STEP5_queue_status.json').write_text(json.dumps(results))
            (output/'STEP5_peak_summary.csv').write_text('model\n'+'\n'.join(sorted(pipeline.EXPECTED_JOBS))+'\n')
            yield pipeline,args,source,run,output

    def test_completed_family_resume_updates_plot_without_any_solver_or_rebuild(self):
        with self.completed_family() as (pipeline,args,source,run,output):
            before={p:p.read_bytes() for p in output.iterdir() if p.is_file()}
            with mock.patch.object(pipeline.shutil,'which',return_value=None):
                pipeline.run_pipeline(args,runner=mock.Mock(side_effect=AssertionError('must not launch')))
            target=run.parent/'STEP5_family_force_displacement.svg'
            self.assertTrue(target.is_file())
            self.assertIn('family_1 LG_mp',target.read_text())
            self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_successful_unmet_postpeak_family_continues_and_resumes_without_solving(self):
        import abaqus_step5_queue as q
        with self.completed_family() as (pipeline,args,source,run,output):
            status = pipeline.read_json(output/'STEP5_queue_status.json')
            for row in status:
                name = row['model']
                row.update(outcome='INCOMPLETE_ANALYSIS', analysis_completed=False, criterion_reached=False)
                (output/(name+'.sta')).write_text('RIKS increment table')
                (output/(name+'.log')).write_text('Abaqus JOB '+name+' COMPLETED\n')
                stem = output/(name+'_force_displacement')
                stem.with_suffix('.csv').write_text(
                    'frame,lpf,shortening_mm,force_kN\n0,0,0,0\n1,1,1,1\n2,0.92,4,0.92\n')
            (output/'STEP5_queue_status.json').write_text(json.dumps(status))
            self.assertFalse(pipeline.preflight(args)[0][2]['completed']) # queue repairs stale status, no analysis
            with mock.patch.object(q, 'run_job', side_effect=AssertionError('must not solve')), \
                    mock.patch.object(q.fd, 'extract', side_effect=AssertionError('must not extract')), \
                    mock.patch.object(q, 'plot_png'), mock.patch.object(q.resources, 'detect_resources'), \
                    mock.patch.object(q.resources, 'resolve_solver_policy', return_value=types.SimpleNamespace(cpus=4,gpus=0)):
                q.run(self.queue_args(output))
            saved = pipeline.read_json(output/'STEP5_queue_status.json')
            self.assertTrue(all(q.finished_result(row) and not row['capacity_verified'] for row in saved))
            self.assertTrue(pipeline.preflight(args)[0][2]['completed'])
            with mock.patch.object(pipeline.shutil, 'which', return_value=None):
                pipeline.run_pipeline(args, runner=mock.Mock(side_effect=AssertionError('must not launch')))
            self.assertIn('capacity unverified', (run.parent/'STEP5_family_force_displacement.svg').read_text())

    def test_resume_rejects_changed_selected_modes_before_running_anything(self):
        with self.completed_family() as (pipeline,args,source,run,output):
            args.global_modes='2,3'
            with self.assertRaisesRegex(ValueError,'(?i)(mode|setting)'):
                pipeline.preflight(args)

    def test_resume_accepts_legacy_riks_default_but_rejects_changed_predictor(self):
        with self.completed_family() as (pipeline,args,source,run,output):
            self.assertTrue(pipeline.preflight(args)[0][2]['completed'])
            args.riks_extrapolation='none'
            with self.assertRaisesRegex(ValueError,'Riks extrapolation'):
                pipeline.preflight(args)
            for name in pipeline.EXPECTED_JOBS:
                path=output/(name+'_step5.json')
                info=pipeline.read_json(path)
                info['settings']['riks_extrapolation']='none'
                path.write_text(json.dumps(info))
            self.assertTrue(pipeline.preflight(args)[0][2]['completed'])


    def queue_args(self,output):
        return types.SimpleNamespace(run_dir=str(output),cae=None,jobs=None,extract_only=False,
            repair_only=False,resume=True,cpus='auto',gpus='auto',ratio=.7,stop_method='monitor',
            poll_seconds=1.,progress_seconds=15.)

    def test_partial_queue_resume_submits_only_unused_job_and_preserves_finished_row(self):
        import abaqus_step5_queue as q
        with self.completed_family() as (pipeline,args,source,run,output):
            name='STEP5_LG_mp_FY240'
            saved={p.name:p.read_bytes() for p in output.iterdir() if p.name.startswith(name) and
                   (p.suffix=='.odb' or '_force_displacement.' in p.name)}
            for filename in saved:(output/filename).unlink()
            status=pipeline.read_json(output/'STEP5_queue_status.json')
            status=[r for r in status if r['model']!=name]
            (output/'STEP5_queue_status.json').write_text(json.dumps(status))
            self.assertFalse(pipeline.preflight(args)[0][2]['completed'])
            def fake_solve(job,unused_args,policy):
                self.assertEqual(job,name)
                for filename,data in saved.items():(output/filename).write_bytes(data)
                return dict(solver_exit_code=0)
            summary=dict(csv=str(output/(name+'_force_displacement.csv')),model=name,step='GMNIA',frames=3,
                partial=False,peak=dict(force_kN=1.,force_N=1000.,lpf=1.,shortening_mm=1.))
            with mock.patch.object(q,'run_job',side_effect=fake_solve) as solve, \
                    mock.patch.object(q.fd,'extract',return_value=summary) as extract, \
                    mock.patch.object(q,'plot_png'),mock.patch.object(q.resources,'detect_resources'), \
                    mock.patch.object(q.resources,'resolve_solver_policy',return_value=types.SimpleNamespace(cpus=4,gpus=0)):
                q.run(self.queue_args(output))
            self.assertEqual(solve.call_count,1)
            self.assertEqual(extract.call_count,1)
            results=pipeline.read_json(output/'STEP5_queue_status.json')
            self.assertEqual({r['model'] for r in results},pipeline.EXPECTED_JOBS)
            self.assertTrue(all(r['outcome']=='CRITERION_REACHED' for r in results))

    def test_plot_failure_resume_repairs_from_csv_without_solver_or_extraction(self):
        import abaqus_step5_queue as q
        with self.completed_family() as (pipeline,args,source,run,output):
            status=pipeline.read_json(output/'STEP5_queue_status.json')
            status[0]['comparison_error']='temporary PNG file access failure'
            (output/'STEP5_queue_status.json').write_text(json.dumps(status))
            self.assertFalse(pipeline.preflight(args)[0][2]['completed'])
            with mock.patch.object(q,'run_job',side_effect=AssertionError('must not solve')), \
                    mock.patch.object(q.fd,'extract',side_effect=AssertionError('must not extract')), \
                    mock.patch.object(q,'plot_png'),mock.patch.object(q.resources,'detect_resources'), \
                    mock.patch.object(q.resources,'resolve_solver_policy',return_value=types.SimpleNamespace(cpus=4,gpus=0)):
                q.run(self.queue_args(output))
            self.assertTrue(all(not r.get('comparison_error') for r in pipeline.read_json(output/'STEP5_queue_status.json')))

    def test_batch_plot_grows_once_per_family_and_keeps_color_across_resume(self):
        from pathlib import Path
        import xml.etree.ElementTree as ET
        import abaqus_step5_queue as q
        with self.completed_family() as (pipeline,args,source,run,output):
            plans=[(source,run,{}),(source.with_name('family_2'),run.with_name('family_2'),{})]
            entries=pipeline.preflight(args)[0][2]['entries']
            finished={source.name:entries}
            with mock.patch.object(q,'plot_png'):
                # Mocked render must still publish a file for the atomic replacement.
                with mock.patch.object(q,'plot_png',side_effect=lambda path,*a,**k:Path(path).write_bytes(b'PNG')):
                    pipeline.update_family_plot(run.parent,plans,finished)
                    target=run.parent/'STEP5_family_force_displacement.svg'
                    first=ET.parse(target).getroot().findall('.//{http://www.w3.org/2000/svg}polyline')
                    finished['family_2']=entries
                    pipeline.update_family_plot(run.parent,plans,finished)
                    pipeline.update_family_plot(run.parent,plans,finished)
                    final=ET.parse(target).getroot().findall('.//{http://www.w3.org/2000/svg}polyline')
            self.assertEqual((len(first),len(final)),(2,4))
            self.assertEqual(first[0].get('stroke'),final[0].get('stroke'))
            self.assertNotEqual(final[0].get('stroke'),final[2].get('stroke'))

    def test_completed_inputs_do_not_depend_on_original_cae_viewport_bytes(self):
        with self.completed_family() as (pipeline,args,source,run,output):
            (run/'imperfection'/pipeline.STEP4_RECEIPT).write_text(
                json.dumps(pipeline.step4_receipt(args,source,run)))
            (run/'BASE.cae').write_text('changed unused original CAE')
            self.assertTrue(pipeline.preflight(args)[0][2]['completed'])
            (run/'BASE.inp').write_text('changed FE input')
            with self.assertRaisesRegex(ValueError,'(?i)(provenance|changed)'):
                pipeline.preflight(args)


    def test_interrupted_after_step4_resumes_at_step5_with_current_resource_arguments(self):
        from pathlib import Path
        import abaqus_step5_queue as q
        with self.completed_family() as (pipeline,args,source,run,output):
            templates={p.name:p.read_bytes() for p in output.iterdir()}
            for p in output.iterdir():p.unlink()
            output.rmdir()
            (run/'imperfection'/'Step4_LG.cae').unlink()
            for kind in ('L','G'):
                ref=Path(str(run)+'_REF_'+kind)
                (ref/('REF_'+kind+'.sta')).write_text('THE ANALYSIS HAS COMPLETED SUCCESSFULLY')
                report=ref/'mode_decomposition_GDLC';report.mkdir()
                (report/'mode_participation.csv').write_text('mode\n1\n')
                (report/'basis_report.json').write_text(json.dumps(dict(n_modes=20,
                    reference_check=dict(kind=kind,passed=True),
                    source_inp_sha256=pipeline.file_sha256(ref/('REF_'+kind+'.inp')),
                    source_odb_sha256=pipeline.file_sha256(ref/('REF_'+kind+'.odb')))))
            def interrupted(command,cwd,label):
                if label.endswith('| STEP4'):
                    (run/'imperfection'/'Step4_LG.cae').write_text('step4')
                else:raise RuntimeError('simulated interruption before STEP5 build')
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                pipeline.run_pipeline(args,runner=interrupted,abaqus='fake_abaqus')
            self.assertTrue((run/'imperfection'/pipeline.STEP4_RECEIPT).is_file())
            calls=[]
            def resumed(command,cwd,label):
                calls.append((command,label))
                output.mkdir(exist_ok=True)
                if label.endswith('| STEP5 BUILD'):
                    for filename,data in templates.items():
                        if filename.endswith(('.inp','_step5.json','.cae','.bat')):
                            (output/filename).write_bytes(data)
                elif label.endswith('| STEP5 SOLVE QUEUE'):
                    for filename,data in sorted(templates.items(),key=lambda p:not p[0].endswith('.odb')):
                        if not (output/filename).exists():(output/filename).write_bytes(data)
                else:self.fail('Unexpected rebuild: '+label)
            with mock.patch.object(q,'plot_png',side_effect=lambda path,*a,**k:Path(path).write_bytes(b'PNG')):
                pipeline.run_pipeline(args,runner=resumed,abaqus='fake_abaqus')
            self.assertEqual(len(calls),2)
            queue_command=calls[-1][0]
            self.assertIn('--resume',queue_command)
            self.assertEqual(queue_command[queue_command.index('--cpus')+1],'auto')
            self.assertEqual(queue_command[queue_command.index('--gpus')+1],'auto')

    def test_input_content_gate_accepts_retouched_sources_but_checks_actual_seams(self):
        from pathlib import Path
        import copy
        from gdlc_classifier import gdlc_mode_decomposition as classifier
        with ModelInputTests().input_folder() as folder:
            data=builder.read_model_inputs(folder)
        with self.completed_family() as (pipeline,args,source,run,output):
            state=dict(source_inputs=builder.input_summary(data))
            pairs=[];instances=[];labels=[];xyz=[]
            for _,pa,pb,xa,ya,xb,yb in data[4]:
                for z in data[5]:
                    a=('P%d'%pa,len(labels)+1);b=('P%d'%pb,len(labels)+2)
                    pairs.append(('BEAM',a,b))
                    instances.extend([a[0],b[0]]);labels.extend([a[1],b[1]])
                    xyz.extend([(xa,ya,z),(xb,yb,z)])
            for name in pipeline.INPUT_FILES:os.utime(source/name,(2000000000,2000000000))
            with mock.patch.object(builder,'read_model_inputs',return_value=data), \
                    mock.patch.object(classifier,'read_inp',return_value=types.SimpleNamespace(pairs=pairs)), \
                    mock.patch.object(classifier,'load_mesh',return_value=types.SimpleNamespace(instance=instances,label=labels,xyz=xyz)):
                pipeline.check_source_inputs(source,run,state,run/'BASE.inp')
                changed=copy.deepcopy(data);changed[4][0][3]+=.01
                with mock.patch.object(builder,'read_model_inputs',return_value=changed):
                    with self.assertRaisesRegex(ValueError,'(?i)seam coordinates'):
                        pipeline.check_source_inputs(source,run,state,run/'BASE.inp')


    def test_family_discovery_uses_numeric_order_and_parent_output_directory(self):
        with self.completed_family() as (pipeline,args,source,run,output):
            for name in ('family_10','family_2'):
                folder=source.with_name(name);folder.mkdir()
                for filename in pipeline.INPUT_FILES:(folder/filename).write_text('input')
            plans=pipeline.preflight(args)
            self.assertEqual([family.name for family,_,_ in plans],['family_1','family_2','family_10'])
            self.assertTrue(all(target==args.output_root/args.family_root.name/family.name
                                for family,target,_ in plans))
            collision=source.with_name('family_1_REF_L');collision.mkdir()
            for filename in pipeline.INPUT_FILES:(collision/filename).write_text('input')
            with self.assertRaisesRegex(ValueError,'(?i)collide'):
                pipeline.preflight(args)


if __name__ == '__main__':
    unittest.main()

