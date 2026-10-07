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


if __name__ == '__main__':
    unittest.main()

