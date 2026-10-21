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


class NoPlateCommandParityTests(unittest.TestCase):
    def test_bending_single_case_uses_separate_destinations_and_selected_models(self):
        import abaqus_family_pipeline as pipeline
        args=pipeline.parse_arguments(['--family-root','inputs','--output-root','out',
            '--load-case','bending','--bending-axis-deg','360','--model-cases','imperfect',
            '--steel-column-plate-fy','240','--pm-interaction','--pm-axes','90,0','--pm-angles','90,0,45'])
        self.assertEqual(args.global_modes,'1')
        self.assertEqual(args.step5_directory,'Step5_GMNIA_FY240_LG_IMPERFECT_ONLY')
        self.assertEqual(pipeline.pm_directory(args),'Step5_PM_FY240_LG_IMPERFECT_ONLY')
        self.assertEqual(pipeline.selected_models(args),['STEP4_LG_mp'])
        self.assertEqual(args.bending_axis_deg,0.)
        self.assertEqual(args.pm_angles,'0,45,90')
        self.assertEqual(args.pm_axes,'0,90')
        self.assertFalse(hasattr(args,'steel_plate_length'))

    def test_compression_default_preserves_legacy_modes_and_output_folder(self):
        import abaqus_family_pipeline as pipeline
        args=pipeline.parse_arguments(['--family-root','inputs','--output-root','out'])
        self.assertEqual(args.global_modes,'1,2')
        self.assertEqual(args.step5_directory,'Step5_GMNIA_FY240_LG')
        self.assertEqual(pipeline.pm_directory(args),'Step5_PM_FY240_LG')

    def test_reference_group_resolution_requires_completed_eigenvalues(self):
        from pathlib import Path
        import abaqus_family_pipeline as pipeline
        args=pipeline.parse_arguments(['--family-root','inputs','--output-root','out','--global-modes','1,2'])
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            run=Path(d)/'family_1';ref=Path(str(run)+'_REF_G');ref.mkdir()
            (ref/'gdlc_reference.json').write_text(json.dumps(dict(job='REF_G')))
            (ref/'REF_G.inp').write_text('reference')
            (ref/'REF_G.dat').write_text('MODE NO EIGENVALUE\n1 100\n2 200\n3 200.01\n')
            with mock.patch.object(pipeline,'completed_reference',return_value=False):
                self.assertEqual(pipeline.family_args(args,run).global_modes,'1,2')
            with mock.patch.object(pipeline,'completed_reference',return_value=True):
                self.assertEqual(pipeline.family_args(args,run).global_modes,'1')
                args.global_modes='2,1'
                self.assertEqual(pipeline.family_args(args,run).global_modes,'2,3')
                os.utime(ref/'REF_G.dat',(1,1))
                self.assertEqual(pipeline.family_args(args,run).global_modes,'2,1')

    def test_no_plate_gate_checks_saved_metadata_before_source_shortcuts(self):
        with FamilyPipelineTests().completed_family() as (pipeline,args,source,run,output):
            state=pipeline.read_json(run/'pipeline_status.json')
            state['source_inputs']['steel_plates']={'material':'STEEL_PLATES'}
            (run/'pipeline_status.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'plate'):
                pipeline.check_resume_build(args,source,run)

    def test_pm_records_bind_cases_and_step4_source_before_reuse(self):
        import abaqus_pm_interaction as pm
        with FamilyPipelineTests().completed_family() as (p,args,source,run,primary):
            args.pm_axes='0';args.pm_angles='0,45,90'
            output=run/'imperfection'/'Step5_PM_FY240_LG';output.mkdir()
            jobs=[]
            for ray,theta,psi in pm.ray_list([0.],[0.,45.,90.]):
                for model,case in [('PERFECT','PERFECT'),('STEP4_LG_mp','LG_mp')]:
                    name=ray+'_'+case;inp=output/(name+'.inp');inp.write_text('ray '+name)
                    info=p.read_json(primary/('STEP5_'+case+'_FY240_step5.json'))
                    info.update(job_name=name,input_sha256=p.file_sha256(inp),source_case=case,
                                load_case=dict(type='combined'),pm_angle_deg=psi)
                    (output/(name+'_step5.json')).write_text(json.dumps(info))
                    jobs.append(dict(job=name,source=model,case=case,bending_axis_deg=theta,pm_angle_deg=psi))
            manifest=dict(mode='gmnia',source_cae=str(run/'imperfection'/'Step4_LG.cae'),
                reference_run=str(run),fy_MPa=240.,axes_deg=[0.],pm_angles_deg=[0.,45.,90.],
                cases=['PERFECT','STEP4_LG_mp'],jobs=jobs)
            (output/pm.MANIFEST_FILE).write_text(json.dumps(manifest))
            (output/pm.SECTION_FILE).write_text('{}')
            request=p.pm_request(args,source,run)
            p.pm_input_record(output,request)
            manifest['cases']=['PERFECT'];(output/pm.MANIFEST_FILE).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'selected models'):
                p.pm_input_record(output,request)
            manifest['cases']=['PERFECT','STEP4_LG_mp'];(output/pm.MANIFEST_FILE).write_text(json.dumps(manifest))
            sidecar=output/(jobs[0]['job']+'_step5.json');info=p.read_json(sidecar)
            info['reference_pipeline']['source_inp_sha256']='unrelated source'
            sidecar.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError,'provenance'):
                p.pm_input_record(output,request)

    def test_selected_step5_command_uses_existing_no_plate_source(self):
        with FamilyPipelineTests().completed_family() as (p,args,source,run,output):
            args.model_cases='perfect';args.step5_directory='fresh';calls=[]
            def capture(command,cwd,label):
                calls.append(command)
                raise RuntimeError('captured command')
            with mock.patch.object(p,'preflight',return_value=[(source,run,dict(step4=True))]):
                with self.assertRaisesRegex(RuntimeError,'captured command'):
                    p.run_pipeline(args,runner=capture,abaqus='fake_abaqus')
            command=calls[0];index=command.index('--models')
            self.assertEqual(command[index+1:index+3],['PERFECT','--fy'])
            self.assertEqual(command[command.index('--output-dir')+1],str(run/'imperfection'/'fresh'))
            self.assertNotIn('--steel-plate-fy',command)

    def test_bending_builder_preserves_angle_and_has_no_added_plate_flags(self):
        import abaqus_family_pipeline as p
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            root=Path(d);source=root/'inputs'/'family_1';source.mkdir(parents=True)
            for name in p.INPUT_FILES:(source/name).write_text('input')
            args=p.parse_arguments(['--family-root',str(source.parent),'--output-root',str(root/'out'),
                '--load-case','bending','--bending-axis-deg','45'])
            calls=[]
            def capture(command,cwd,label):
                calls.append(command);raise RuntimeError('captured builder')
            with self.assertRaisesRegex(RuntimeError,'captured builder'):
                p.run_pipeline(args,runner=capture,abaqus='fake_abaqus')
            command=calls[0]
            self.assertEqual(command[command.index('--load-case')+1],'bending')
            self.assertEqual(command[command.index('--bending-axis-deg')+1],'45.0')
            self.assertTrue(all(not flag.startswith('--steel-plate-') for flag in command))
            self.assertTrue(p.preflight(args)[0][1].name.endswith('_BEND045'))

    def test_start_family_excludes_earlier_incomplete_source_folders(self):
        from pathlib import Path
        import abaqus_family_pipeline as p
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            root=Path(d);families=root/'inputs';families.mkdir()
            (families/'family_1').mkdir()
            for name in ('family_2','family_10'):
                family=families/name;family.mkdir()
                for filename in p.INPUT_FILES:(family/filename).write_text('input')
            args=p.parse_arguments(['--family-root',str(families),'--output-root',str(root/'out'),
                '--start-family','family_2'])
            self.assertEqual([family.name for family,run,state in p.preflight(args)],['family_2','family_10'])
            self.assertFalse(args.output_root.exists())

    def test_discovery_ignores_unrelated_folders_but_accepts_generic_exports(self):
        from pathlib import Path
        import abaqus_family_pipeline as p
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            root=Path(d);families=root/'inputs';families.mkdir()
            (families/'New folder').mkdir()
            (families/'reports').mkdir()
            (families/'reports'/'summary.csv').write_text('unrelated')
            for name in ('family_10','family_2','custom_section'):
                family=families/name;family.mkdir()
                for filename in p.INPUT_FILES:(family/filename).write_text('input')
            args=p.parse_arguments(['--family-root',str(families),'--output-root',str(root/'out')])
            self.assertEqual([family.name for family,run,state in p.preflight(args)],
                             ['family_2','family_10','custom_section'])
            self.assertFalse(args.output_root.exists())

    def test_discovery_keeps_incomplete_named_families_and_partial_exports_visible(self):
        from pathlib import Path
        import abaqus_family_pipeline as p
        for name,partial in [('family_3',False),('custom_section',True)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
                root=Path(d);family=root/'inputs'/name;family.mkdir(parents=True)
                if partial:(family/'builtup_member.csv').write_text('partial export')
                args=p.parse_arguments(['--family-root',str(family.parent),'--output-root',str(root/'out')])
                with self.assertRaisesRegex(ValueError,'Required file missing:.*builtup_'):
                    p.preflight(args)
                self.assertFalse(args.output_root.exists())

    def test_completed_pm_queue_refreshes_only_diagrams(self):
        import abaqus_pm_interaction as pm
        with FamilyPipelineTests().completed_family() as (p,args,source,run,primary):
            args.pm_axes='0';args.pm_angles='0,45,90'
            output=run/'imperfection'/'Step5_PM_FY240_LG';output.mkdir()
            (output/'STEP5_PM_points.csv').write_text('saved responses')
            (output/'STEP5_queue_status.json').write_text(json.dumps([
                dict(model='PM_P00_PERFECT',criterion_reached=True,outcome='CRITERION_REACHED')]))
            record=dict(request={},files={'PM_P00_PERFECT.inp':'already verified'})
            calls=[]
            def collect(command,cwd,label):
                calls.append(command)
                diagram=output/'diagram';diagram.mkdir()
                (diagram/'PM_interaction_points.csv').write_text('saved plotted responses')
                (diagram/'PM_interaction.svg').write_text('<svg/>')
            state=dict(record=record,queue_completed=True,completed=False)
            p.run_family_pm(args,source,run,None,collect,'fixture',state=state)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][calls[0].index('--mode')+1],'collect')
            self.assertTrue(p.pm_plot_current(output,p.read_json(output/p.PM_RECEIPT)))
            self.assertEqual((output/'STEP5_PM_points.csv').read_text(),'saved responses')


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
            # Plot references now use actual geometry even in orchestration tests.
            (source/'builtup_segments.csv').write_text(
                '1,-10,10,10,10,2\n2,-10,-10,-10,10,2\n3,10,-10,-10,-10,2\n4,10,10,10,-10,2\n')
            (source/'builtup_member.csv').write_text(
                'L_mm,t_mm,E_MPa,nu,fy_MPa\n3600,2,200000,0.3,240\n')
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




class PmOnlyPipelineTests(unittest.TestCase):
    """Small orchestration checks; no native Abaqus or existing result files."""
    def setUp(self):
        from pathlib import Path
        import shutil
        import uuid
        import abaqus_family_pipeline as pipeline
        self.p=pipeline
        self.root=Path(__file__).resolve().parent/('pm_only_test_'+uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree,self.root)
        self.family=self.root/'source'/'family_1';self.family.mkdir(parents=True)
        self.args=pipeline.parse_arguments(['--family-root',str(self.family.parent),
            '--output-root',str(self.root/'out'),'--pm-only','--parallel-jobs','auto','--resume'])
        self.run=pipeline.batch_output_directory(self.args)/pipeline.family_run_name(self.args,self.family)
        self.run.mkdir(parents=True)
        self.cae=self.run/'imperfection'/'Step4_LG.cae'
        self.cae.parent.mkdir();self.cae.write_text('step4')

    def test_default_stays_standalone_and_sequential(self):
        args=self.p.parse_arguments(['--family-root','source','--output-root','out'])
        self.assertFalse(args.pm_only)
        self.assertFalse(args.pm_interaction)
        self.assertEqual(args.parallel_jobs,'1')

    def test_pm_only_implies_interaction_and_accepts_shared_job_budget(self):
        self.assertTrue(self.args.pm_interaction)
        self.assertEqual(self.args.parallel_jobs,'auto')
        for value in ('0','-1','1.5'):
            with self.subTest(value=value),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.p.parse_arguments(['--family-root','source','--output-root','out','--parallel-jobs',value])

    def test_verified_step4_skips_all_standalone_processing(self):
        progress=dict(step4=True,completed=True)  # no standalone entries required
        with mock.patch.object(self.p,'preflight',return_value=[(self.family,self.run,progress)]), \
                mock.patch.object(self.p,'run_family_pm') as pm, \
                mock.patch.object(self.p,'check_step5_inputs',side_effect=AssertionError('standalone input check')), \
                mock.patch.object(self.p,'refresh_step5_reference_plots',side_effect=AssertionError('standalone plots')), \
                mock.patch.object(self.p,'update_family_plot',side_effect=AssertionError('standalone comparison')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.p.run_pipeline(self.args,runner=mock.Mock(side_effect=AssertionError('native command')),abaqus='fake_abaqus')
        pm.assert_called_once()

    def resume_fixture(self):
        state=dict(status='BUILT',settings=dict(vars(self.args),builtup_dir=str(self.family)),build=dict(job_name='BASE'))
        from pathlib import Path
        state['settings']={k:str(v) if isinstance(v,Path) else v for k,v in state['settings'].items()}
        (self.run/'pipeline_status.json').write_text(json.dumps(state))
        (self.run/'BASE.inp').write_text('base');(self.run/'BASE.cae').write_text('base')
        (self.cae.parent/self.p.STEP4_RECEIPT).write_text(json.dumps(dict(verified='step4')))

    def test_resume_pm_only_does_not_require_or_inspect_standalone(self):
        self.resume_fixture()
        # Existing unrelated/partial standalone output must not gate a verified Step4.
        (self.cae.parent/self.args.step5_directory).mkdir()
        with mock.patch.object(self.p,'family_args',side_effect=lambda a,r:a), \
                mock.patch.object(self.p,'check_source_inputs'), \
                mock.patch.object(self.p,'step4_receipt',return_value=dict(verified='step4')), \
                mock.patch.object(self.p,'check_step5_inputs',side_effect=AssertionError('standalone inputs')):
            progress=self.p.check_resume_build(self.args,self.family,self.run)
        self.assertTrue(progress['step4'])
        self.assertFalse(progress['step5'])
        self.assertFalse(progress['completed'])

    def test_resume_pm_only_checks_step4_checkpoint(self):
        self.resume_fixture()
        with mock.patch.object(self.p,'family_args',side_effect=lambda a,r:a), \
                mock.patch.object(self.p,'check_source_inputs'), \
                mock.patch.object(self.p,'step4_receipt',return_value=dict(verified='changed')):
            with self.assertRaisesRegex(ValueError,'Step4 settings/source files differ'):
                self.p.check_resume_build(self.args,self.family,self.run)

    def test_pm_only_keeps_step4_preparation_then_skips_standalone_build(self):
        from pathlib import Path
        self.resume_fixture()
        for kind in ('L','G'):
            ref=Path(str(self.run)+'_REF_'+kind);ref.mkdir()
            (ref/'gdlc_reference.json').write_text(json.dumps(dict(kind=kind,job='REF_'+kind)))
            (ref/('REF_'+kind+'.odb')).write_text('reference')
        calls=[]
        def runner(command,cwd,label):
            calls.append(label)
            if 'STEP5' in label:raise AssertionError('standalone Step5 must be skipped')
        with mock.patch.object(self.p,'preflight',return_value=[(self.family,self.run,{})]), \
                mock.patch.object(self.p,'family_args',side_effect=lambda a,r:a), \
                mock.patch.object(self.p,'completed_reference',return_value=True), \
                mock.patch.object(self.p,'step4_receipt',return_value=dict(verified='step4')), \
                mock.patch.object(self.p,'run_family_pm') as pm, \
                mock.patch.object(self.p,'check_step5_inputs',side_effect=AssertionError('standalone inputs')), \
                mock.patch.object(self.p,'update_family_plot',side_effect=AssertionError('standalone plot')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.p.run_pipeline(self.args,runner=runner,abaqus='fake_abaqus')
        self.assertTrue(any(label.endswith('| STEP4') for label in calls))
        self.assertFalse(any('STEP5' in label for label in calls))
        pm.assert_called_once()

    def test_pm_queue_receives_parallel_jobs(self):
        import abaqus_step5_queue as queue
        output=self.run/'imperfection'/self.p.pm_directory(self.args);output.mkdir()
        (output/'STEP5_PM_points.csv').write_text('point')
        state=dict(completed=False,queue_completed=False,record=dict(files={'PM_TEST.inp':'hash'}))
        calls=[]
        with mock.patch.object(self.p,'read_json',return_value=[dict(model='PM_TEST')]), \
                mock.patch.object(self.p,'pm_plot_current',return_value=True), \
                mock.patch.object(queue,'processed_result',return_value=True), \
                contextlib.redirect_stdout(io.StringIO()):
            self.p.run_family_pm(self.args,self.family,self.run,'fake_abaqus',
                lambda cmd,cwd,label:calls.append((cmd,label)),'family',state=state)
        self.assertEqual(len(calls),1)
        command,label=calls[0]
        self.assertTrue(label.endswith('P-M SOLVE QUEUE'))
        self.assertEqual(command[command.index('--parallel-jobs')+1],'auto')


if __name__ == '__main__':
    unittest.main()

