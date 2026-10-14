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
sys.path.insert(0, os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename))))
import abaqus_complete_model_m20 as builder


class ModelInputTests(unittest.TestCase):
    def test_plate_skip_pattern_keeps_all_screws_and_both_edge_plates(self):
        data = self.plate_fixture()
        baseline = builder.steel_plate_layout(data, 40., 40., 0.)
        full_bolts = builder.bolt_layout(data, baseline)
        physical = lambda rows: [{k: v for k, v in row.items() if k not in ('plate_instance', 'plate_direction')} for row in rows]
        for skip, x in ((0, data[5]), (1, [25., 500., 975.]),
                        (2, [25., 975.]), (100, [25., 975.])):
            with self.subTest(skip=skip):
                layout = builder.steel_plate_layout(data, 40., 40., 0., skip_count=skip)
                self.assertEqual([g['bolt_positions_mm'] for g in layout['groups']], [x, x])
                rows = builder.bolt_layout(data, layout)
                self.assertEqual(physical(rows), physical(full_bolts))
                self.assertEqual(sum(2 if r['plate_instance'] else 1 for r in rows), len(rows)+4*len(layout['instances']))
                for g, old in zip(layout['groups'], baseline['groups']):
                    self.assertEqual(g['z_intervals_mm'][0], old['z_intervals_mm'][0])
                    self.assertEqual(g['z_intervals_mm'][-1], old['z_intervals_mm'][-1])
                self.assertEqual(layout['instances'], ['STEEL_PLATE_%03d' % j for j in range(1, len(x)+1)])
                self.assertTrue(all(g['instances']==layout['instances'] for g in layout['groups']))
                if skip == 0:
                    self.assertEqual(layout, baseline)

    def test_skip_cannot_change_gap_based_rows_or_restore_colliding_candidates(self):
        data = self.plate_fixture(500., (25., 175., 325., 475.))
        full = builder.steel_plate_layout(data, 40., 80., 5.)
        sparse = builder.steel_plate_layout(data, 40., 80., 5., skip_count=1)
        for old, group in zip(full['groups'], sparse['groups']):
            self.assertEqual(group['screw_positions_mm'], old['bolt_positions_mm'])
        self.assertEqual(sparse['groups'][1]['omitted_candidates'], full['groups'][1]['omitted_candidates'])
        self.assertEqual([r['z_mm'] for r in builder.bolt_layout(data, full)],
                         [r['z_mm'] for r in builder.bolt_layout(data, sparse)])

    def test_skip_single_direction_and_meeting_at_member_center(self):
        data = self.plate_fixture(130., (25., 105.))
        layout = builder.steel_plate_layout(data, 40., 40., 0., skip_count=1)
        self.assertEqual(layout['groups'][1]['bolt_positions_mm'], [25., 105.])
        data = self.plate_fixture(1000., (25., 215., 405., 595., 785., 975.))
        layout = builder.steel_plate_layout(data, 40., skip_count=1)
        self.assertEqual(layout['bolt_positions_mm'], [25., 405., 595., 975.])
        self.assertEqual(len(builder.bolt_layout(data, layout)), 24)

    def test_skip_cli_validation_and_output_provenance(self):
        flags = ['--steel-plate-length', '40', '--steel-plate-length-2', '40']
        self.assertEqual(builder.parse_arguments(flags).steel_plate_skip_count, 0)
        for value in ('-1', '1.5'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.parse_arguments(flags+['--steel-plate-skip-count', value])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            builder.parse_arguments(['--steel-plate-skip-count', '1'])
        with tempfile.TemporaryDirectory() as root:
            args = builder.parse_arguments(flags+['--steel-plate-skip-count', '1',
                '--builtup-dir', os.path.join(root, 'input', 'case'), '--output-root', root])
            self.assertEqual(os.path.basename(builder.prepare_output_directory(args)), 'case_PL40_PL240_PG0_PS1')

    def test_sparse_plate_contract_uses_full_screw_rows(self):
        import abaqus_pipeline_contract as contract
        from test_pipeline_contract import valid_state
        data = self.plate_fixture()
        data[-1]['bolt_row_mm'] = 15.
        with mock.patch.object(builder, 'STEEL_PLATE_LENGTH', 40.), \
             mock.patch.object(builder, 'STEEL_PLATE_LENGTH_2', 40.), \
             mock.patch.object(builder, 'STEEL_PLATE_SKIP_COUNT', 1, create=True):
            source = builder.input_summary(data)
        state = valid_state('unused')
        state['source_inputs'] = source
        state['build'].update(length_mm=1000., rigid_links=source['expected_links'])
        contract.validate_reference_state(state)
        self.assertEqual(source['physical_bolts'], 20)
        self.assertEqual(source['expected_links'], 32)
        self.assertEqual(len(source['steel_plates']['instances']), 3)

    def plate_fixture(self, length=1000., bolts=(25., 262.5, 500., 737.5, 975.)):
        pieces = {1: [(6., 150., 200., 6.)], 2: [(-200., 6., -6., 150.)],
                  3: [(-6., -150., -200., -6.)], 4: [(200., -6., 6., -150.)]}
        seams = [[1, 1, 2, 6., 160., -6., 160.], [2, 2, 3, -160., 6., -160., -6.],
                 [3, 3, 4, -6., -160., 6., -160.], [4, 4, 1, 160., -6., 160., 6.]]
        return (pieces, 2., 200000., .3, seams, list(bolts), {'L_mm': length, 'gap_mm': 10.})

    def test_cross_uses_common_height_and_exact_source_screws_for_varied_counts(self):
        for bolts in ((40.,), (40., 360.), (40., 120., 200., 280., 360.),
                      (40., 104., 168., 232., 296., 360.)):
            data=self.plate_fixture(400.,bolts)
            for skip in (0,1,2,3,100):
                layout=builder.steel_plate_layout(data,40.,60.,0.,skip_count=skip)
                self.assertEqual(layout['assembly_type'],'CONNECTED_CROSS')
                self.assertTrue(all(g['height_mm']==60. for g in layout['groups']))
                expected=[z for j,z in enumerate(bolts) if min(j,len(bolts)-1-j)%(skip+1)==0]
                self.assertEqual(layout['bolt_positions_mm'],expected)
                for sid in range(1,5):
                    rows=[r for r in builder.bolt_layout(data,layout) if r['seam_id']==sid]
                    self.assertEqual([r['z_mm'] for r in rows],list(bolts))
                    self.assertEqual([r['z_mm'] for r in rows if r['plate_instance']],expected)

    def test_cross_checks_overlap_after_skip_but_always_retains_end_plates(self):
        data=self.plate_fixture(200.,(25.,55.,85.,115.,145.,175.))
        with self.assertRaisesRegex(ValueError,'overlap|touch'):
            builder.steel_plate_layout(data,40.,40.,0.)
        layout=builder.steel_plate_layout(data,40.,40.,0.,skip_count=2)
        self.assertEqual(layout['bolt_positions_mm'],[25.,175.])
        with self.assertRaisesRegex(ValueError,'member end'):
            builder.steel_plate_layout(data,40.,60.,0.,skip_count=100)

    def test_plate_models_close_rounding_gaps_of_the_zero_clearance_fit_only(self):
        calls = []

        class Repository:
            def __init__(self, name):
                self.name = name

            def setValuesInStep(self, **kwargs):
                calls.append((self.name, kwargs))

            def appendInStep(self, **kwargs):
                calls.append((self.name, kwargs))

        class Property:
            def NormalBehavior(self, **kwargs):
                pass

            def TangentialBehavior(self, **kwargs):
                pass

        class Model:
            def ContactProperty(self, name):
                return Property()

            def ContactStd(self, name, createStepName):
                return types.SimpleNamespace(includedPairs=Repository('pairs'),
                    contactPropertyAssignments=Repository('properties'),
                    initializationAssignments=Repository('initialization'))

            def StdInitialization(self, **kwargs):
                calls.append(('StdInitialization', kwargs))

        constants = types.SimpleNamespace(ON='ON', HARD='HARD', DEFAULT='DEFAULT', FRICTIONLESS='FRICTIONLESS',
                                          GLOBAL='GLOBAL', SELF='SELF', ADJUST='ADJUST')
        with mock.patch.dict(sys.modules, {'abaqusConstants': constants, 'interaction': types.ModuleType('i')}):
            builder.add_general_contact(Model())
            self.assertEqual([c[0] for c in calls], ['pairs', 'properties'])        # bare model unchanged
            del calls[:]
            builder.add_general_contact(Model(), builder.PLATE_FIT_TOLERANCE_T*2.)
        self.assertEqual(calls[2], ('StdInitialization', dict(name='SnugFit', overclosureType='ADJUST',
                                                             openingTolerance=2e-4)))
        self.assertEqual(calls[3], ('initialization', dict(stepName='Initial',
                                                          assignments=(('GLOBAL', 'SELF', 'SnugFit'),))))

    def test_perpendicular_plates_with_positive_gap_keep_legacy_staggering(self):
        data = self.plate_fixture()
        layout = builder.steel_plate_layout(data, 40., 40., 5.)
        first, second = layout['groups']
        self.assertEqual(first['bolt_positions_mm'], data[5])
        self.assertEqual(second['direction'], 'Y')
        self.assertEqual(second['bolt_positions_mm'], [70., 307.5, 692.5, 930.])
        self.assertEqual(second['z_intervals_mm'][0], [50., 90.])
        self.assertEqual(second['z_intervals_mm'][-1], [910., 950.])
        rows = builder.bolt_layout(data, layout)
        self.assertEqual([r['z_mm'] for r in rows if r['seam_id'] == 1],
                         [25., 70., 307.5, 692.5, 930., 975.])
        self.assertEqual(len(rows), 22)
        self.assertEqual(sum(2 if r['plate_instance'] else 1 for r in rows), 40)
        self.assertEqual([r['z_mm'] for r in rows if r['seam_id'] == 2], data[5])

    def test_perpendicular_edge_clearance_and_unequal_heights(self):
        layout = builder.steel_plate_layout(self.plate_fixture(), 40., 60., 5.)
        first, second = layout['groups']
        self.assertEqual(second['bolt_positions_mm'][0], 80.)
        self.assertEqual(second['z_intervals_mm'][0], [50., 110.])
        for a, b in first['z_intervals_mm']:
            for c, d in second['z_intervals_mm']:
                self.assertGreaterEqual(max(c-b, a-d), 5.-1e-7)

    def test_central_duplicate_is_one_complete_plate(self):
        layout = builder.steel_plate_layout(self.plate_fixture(140., (25., 115.)), 40., 40., 5.)
        self.assertEqual(layout['groups'][1]['bolt_positions_mm'], [70.])
        self.assertEqual(len(set(layout['instances'])), 3)

    def test_colliding_middle_pair_is_removed_symmetrically(self):
        layout = builder.steel_plate_layout(self.plate_fixture(500., (25., 175., 325., 475.)), 40., 80., 5.)
        second = layout['groups'][1]
        self.assertEqual(second['bolt_positions_mm'], [90., 410.])
        self.assertEqual(len(second['omitted_candidates']), 2)
        with self.assertRaisesRegex(ValueError, 'No.*perpendicular'):
            builder.steel_plate_layout(self.plate_fixture(190., (25., 165.)), 40., 80., 5.)

    def test_second_plate_cli_and_dependency(self):
        args = builder.parse_arguments(['--steel-plate-length', '40', '--steel-plate-length-2', '60'])
        self.assertEqual(args.steel_plate_length_2, 60.)
        self.assertEqual(args.steel_plate_gap_mm, 0.)
        for flags in (['--steel-plate-length-2', '40'], ['--steel-plate-length-2', '-1'],
                      ['--steel-plate-gap-mm', 'nan']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                builder.parse_arguments(flags)

    def test_dual_plate_reference_contract_and_bolt_provenance(self):
        import abaqus_pipeline_contract as contract
        from test_pipeline_contract import valid_state
        data = self.plate_fixture()
        data[-1]['bolt_row_mm'] = 15.
        with mock.patch.object(builder, 'STEEL_PLATE_LENGTH', 40.), \
             mock.patch.object(builder, 'STEEL_PLATE_LENGTH_2', 40.):
            source = builder.input_summary(data)
        state = valid_state('unused')
        state['source_inputs'] = source
        state['build'].update(length_mm=1000., rigid_links=source['expected_links'])
        validated = contract.validate_reference_state(state)
        snap = contract.reference_snapshot(dict(validated, run_dir='unused',
            instances=list(contract.REFERENCE_INSTANCES), settings={}), include_hashes=False)
        self.assertEqual(len(contract.plate_instance_names(snap)), 5)
        self.assertEqual(snap['physical_bolts_per_seam'], {'1': 5, '2': 5, '3': 5, '4': 5})
        self.assertEqual(snap['bolt_layout'], source['bolt_layout'])
        source['bolt_layout'][0]['plate_instance'] = 'STEEL_PLATE_999'
        with self.assertRaisesRegex(ValueError, 'bolt'):
            contract.validate_reference_state(state)

    def test_plate_layout_fills_clear_gap_and_centres_each_bolt_row(self):
        pieces = {1: [(6., 150., 200., 6.)], 2: [(-200., 6., -6., 150.)],
                  3: [(-6., -150., -200., -6.)], 4: [(200., -6., 6., -150.)]}
        seams = [[2, 2, 3, -160., 6., -160., -6.], [4, 4, 1, 160., -6., 160., 6.]]
        data = (pieces, 2., 200000., .3, seams, [25., 75.], {'L_mm': 100., 'gap_mm': 10.})
        layout = builder.steel_plate_layout(data, 40.)
        self.assertEqual(layout['thickness_mm'], 10.)
        self.assertEqual(layout['x_limits_mm'], [-200., 200.])
        self.assertEqual(layout['z_intervals_mm'], [[5., 45.], [55., 95.]])
        self.assertEqual(layout['bolt_seam_ids'], [2, 4])
        self.assertEqual(len(layout['instances']), 2)
        self.assertIsNone(builder.steel_plate_layout(data, 0.))
        for height in (51., float('nan'), -1.):
            with self.assertRaises(ValueError):
                builder.steel_plate_layout(data, height)

    def test_plate_cli_accepts_height_and_rejects_negative(self):
        self.assertEqual(builder.parse_arguments(['--steel-plate-length', '40']).steel_plate_length, 40.)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            builder.parse_arguments(['--steel-plate-length', '-1'])

    def test_mesh_counts_ignore_cae_injected_sum(self):
        assembly = types.SimpleNamespace(instances={'P1': types.SimpleNamespace(nodes=[1, 2], elements=[1]),
            'STEEL_PLATE_001': types.SimpleNamespace(nodes=[1, 2, 3, 4], elements=[1])})
        with mock.patch.object(builder, 'sum', side_effect=TypeError('CAE sum'), create=True):
            self.assertEqual(builder.assembly_mesh_counts(assembly), (6, 2))

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
        for gap in (0.,5.):
            with mock.patch.object(builder,'STEEL_PLATE_LENGTH',40.), \
                    mock.patch.object(builder,'STEEL_PLATE_LENGTH_2',40.), \
                    mock.patch.object(builder,'STEEL_PLATE_GAP_MM',gap):
                plated=builder.load_case_definition(data,'bending',0.)
            self.assertEqual(90 in plated['section_symmetry']['rotations_deg'],gap==0.)
        with open(builder.__file__) as stream:
            source = stream.read()
        self.assertIn("eigensolver=LANCZOS, minEigen=0.0 if LOAD_CASE == 'bending' else None", source)
        self.assertIn("expected_loads = 8 if LOAD_CASE == 'compression' else load_definition['n_loads']", source)


class WorkflowAlignmentTests(unittest.TestCase):
    def test_first_global_selection_and_requested_twist_policy(self):
        import abaqus_family_pipeline as p
        import abaqus_step4_imperfections as step4
        import numpy as np
        args=p.parse_arguments(['--family-root','input','--output-root','output'])
        self.assertEqual(args.global_modes,'1')
        self.assertIsNone(p.global_mode_angle(args))
        p.check_global_reference({'twist':'restrained'},'reference')
        p.check_global_reference({'twist':'free'},'reference','bending')
        for info in ({'twist':'free'},{}):
            with self.assertRaisesRegex(ValueError,'restrain section twist'):
                p.check_global_reference(info,'reference')
        # Pure torsion has no mean bow. Retain its full transverse vector,
        # rather than replacing it with a higher flexural eigenmode.
        field=np.array([[0.,1.,0.],[0.,-1.,0.],[1.,0.,0.],[-1.,0.,0.]])
        shape,meta=step4.global_eigen_shape([field],2,p.global_mode_angle(args))
        self.assertEqual(meta['modes_combined'],1)
        np.testing.assert_allclose(abs(shape),abs(field))
        self.assertAlmostEqual(meta['translation_ratio'],0.)

    def test_independent_column_and_connector_yields_and_default_names(self):
        import abaqus_family_pipeline as p
        base=['--family-root','source','--output-root','target']
        defaults=p.parse_arguments(base)
        self.assertEqual((defaults.steel_column_plate_fy,defaults.steel_plate_fy),(240.,360.))
        self.assertEqual(defaults.step5_directory,p.STEP5_DIRECTORY)
        args=p.parse_arguments(base+['--steel-column-plate-fy','350','--steel-plate-fy','240'])
        self.assertEqual((args.steel_column_plate_fy,args.steel_plate_fy),(350.,240.))
        self.assertEqual(args.step5_directory,'Step5_GMNIA_FY350_LG_REF1')
        self.assertEqual(p.expected_jobs(args),{'STEP5_PERFECT_FY350','STEP5_LG_mp_FY350'})
        self.assertEqual(p.step5_settings(args)['fy'],350.)
        for value in ('0','-1','nan','inf'):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                p.parse_arguments(base+['--steel-column-plate-fy',value])

    def test_column_yield_reaches_compression_and_bending_step5(self):
        for kind in ('compression','bending'):
            with self.completed_family(load_case=kind) as (p,args,source,run,output):
                args.steel_column_plate_fy=350.;args.steel_plate_fy=240.;args.steel_plate_length=40.
                calls=[]
                def capture(command,cwd,label):
                    calls.append(command);raise RuntimeError('captured material command')
                with mock.patch.object(p,'preflight',return_value=[(source,run,{'step4':True})]):
                    with self.assertRaisesRegex(RuntimeError,'captured material'):
                        p.run_pipeline(args,runner=capture,abaqus='fake')
                command=calls[0]
                self.assertEqual(float(command[command.index('--fy')+1]),350.)
                self.assertEqual(float(command[command.index('--steel-plate-fy')+1]),240.)

    def test_resume_never_mixes_superseded_end_condition_results(self):
        with self.completed_family() as (p,args,source,run,output):
            self.assertTrue(output.name.endswith('_REF1'))
            self.assertEqual(set(p.check_step5_inputs(args,run)),p.EXPECTED_JOBS)
            name=sorted(p.EXPECTED_JOBS)[0];path=output/(name+'_step5.json')
            info=p.read_json(path);info.pop('end_condition')
            path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError,'another end condition'):
                p.check_step5_inputs(args,run)

    def test_resume_rejects_changed_column_yield_or_inconsistent_column_table(self):
        with self.completed_family() as (p,args,source,run,output):
            args.steel_column_plate_fy=350.
            with self.assertRaises((ValueError,OSError)):
                p.check_step5_inputs(args,run)
            args.steel_column_plate_fy=240.
            name=sorted(p.EXPECTED_JOBS)[0];path=output/(name+'_step5.json')
            info=p.read_json(path);info['plastic_tables']={'Steel':[[350.,0.]]}
            path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError,'column material differs'):
                p.check_step5_inputs(args,run)

    def test_resume_checks_connector_yield_independently_of_column(self):
        with self.completed_family() as (p,args,source,run,output):
            plates={'material':'SteelPlate'}
            status=p.read_json(run/'pipeline_status.json')
            status['source_inputs']['steel_plates']=plates
            (run/'pipeline_status.json').write_text(json.dumps(status))
            args.steel_plate_fy=355.
            for name in p.EXPECTED_JOBS:
                path=output/(name+'_step5.json');info=p.read_json(path)
                info['settings']['steel_plate_fy']=355.
                info['plastic_tables']={'Steel':[[240.,0.]],'SteelPlate':[[355.,0.]]}
                info['reference_pipeline'].update(steel_plates=plates,
                    pipeline_status_sha256=p.file_sha256(run/'pipeline_status.json'))
                path.write_text(json.dumps(info))
            self.assertEqual(set(p.check_step5_inputs(args,run)),p.EXPECTED_JOBS)
            args.steel_plate_fy=240.
            with self.assertRaisesRegex(ValueError,'plate material differs'):
                p.check_step5_inputs(args,run)

    def test_step5_plate_comparison_accepts_roundoff_but_rejects_model_changes(self):
        import copy
        for skip in (0,1,2,3):
            with self.subTest(skip=skip), self.completed_family() as (p,args,source,run,output):
                plates=dict(material='SteelPlate',skip_count=skip,
                    instances=['STEEL_PLATE_%03d'%i for i in range(1,5-skip)],
                    groups=[dict(direction='X',z_intervals_mm=[[991.111111,1031.1111110000002]])])
                status=p.read_json(run/'pipeline_status.json')
                status['source_inputs']['steel_plates']=plates
                (run/'pipeline_status.json').write_text(json.dumps(status))
                args.steel_plate_fy=355.
                for name in p.EXPECTED_JOBS:
                    path=output/(name+'_step5.json');info=p.read_json(path)
                    recorded=copy.deepcopy(plates)
                    recorded['groups'][0]['z_intervals_mm'][0][1]=1031.111111
                    info['settings']['steel_plate_fy']=355.
                    info['plastic_tables']={'Steel':[[240.,0.]],'SteelPlate':[[355.,0.]]}
                    info['reference_pipeline'].update(steel_plates=recorded,
                        pipeline_status_sha256=p.file_sha256(run/'pipeline_status.json'))
                    path.write_text(json.dumps(info))
                self.assertEqual(set(p.check_step5_inputs(args,run)),p.EXPECTED_JOBS)
                path=output/(sorted(p.EXPECTED_JOBS)[0]+'_step5.json')
                good=p.read_json(path)
                for change in ('coordinate','instance','count','missing','direction'):
                    with self.subTest(change=change):
                        info=copy.deepcopy(good);recorded=info['reference_pipeline']['steel_plates']
                        if change=='coordinate':recorded['groups'][0]['z_intervals_mm'][0][1]+=1e-6
                        elif change=='instance':recorded['instances'][0]='DIFFERENT'
                        elif change=='count':recorded['skip_count']+=1
                        elif change=='missing':del recorded['groups']
                        else:recorded['groups'][0]['direction']='Y'
                        path.write_text(json.dumps(info))
                        with self.assertRaisesRegex(ValueError,'plate geometry/connections differ'):
                            p.check_step5_inputs(args,run)
                        path.write_text(json.dumps(good))

    def test_nondefault_column_yield_completed_resume_and_comparison(self):
        for kind in ('compression','bending'):
            with self.completed_family(load_case=kind,column_fy=350.) as (p,args,source,run,output):
                with mock.patch.object(p.shutil,'which',return_value=None):
                    p.run_pipeline(args,runner=mock.Mock(side_effect=AssertionError('no analyses')))
                response='_moment_rotation_BEND000' if kind=='bending' else '_force_displacement'
                self.assertTrue((run.parent/('STEP5_family'+response+'_FY350.svg')).is_file())

    def test_plate_family_bending_arguments_and_names(self):
        import abaqus_family_pipeline as pipeline
        from pathlib import Path
        args=pipeline.parse_arguments(['--family-root','source','--output-root','target',
            '--load-case','bending','--bending-axis-deg','45',
            '--steel-plate-length','40','--steel-plate-length-2','40',
            '--steel-plate-gap-mm','0','--steel-plate-skip-count','1','--steel-plate-fy','360'])
        self.assertEqual(pipeline.family_run_name(args,Path('family_9')),'family_9_BEND045_PL40_PL240_PG0_PS1')
        self.assertEqual(args.global_modes,'1')
        self.assertEqual(args.steel_plate_fy,360.)
        plain=pipeline.parse_arguments(['--family-root','source','--output-root','target'])
        self.assertEqual(pipeline.family_run_name(plain,Path('family_9')),'family_9')
        self.assertEqual(plain.global_modes,'1')
        self.assertIsNone(pipeline.global_mode_angle(plain))

    def test_step4_and_step5_preserve_load_case_and_plate_material_options(self):
        import abaqus_step4_imperfections as step4
        import abaqus_step5_gmnia as step5
        s4=step4.parse_arguments(['--run-dir','source','--load-case','bending',
            '--local-mode','1','--global-source','source_REF_G','--global-mode','1','--cases','none','--combo','LG=-0.34t,0,L/1000',
            '--output-cae','Step4.cae'])
        self.assertEqual(s4.load_case,'bending')
        s5=step5.parse_arguments(['--source-cae','Step4.cae','--output-dir','out','--fy','240',
            '--steel-plate-fy','360','--load-case','bending','--riks-extrapolation','none'])
        self.assertEqual((s5.load_case,s5.steel_plate_fy,s5.riks_extrapolation),('bending',360.,'none'))

    def test_imperfect_pm_specimen_can_use_equivalent_perfect_axes(self):
        import abaqus_pm_interaction as pm
        data=ModelInputTests().plate_fixture()
        symmetry=dict(rotations_deg=[0,90,180,270],mirror_lines_deg=[0.,45.,90.,135.])
        axes,unused=pm.resolve_axes([0.,90.],data[0],symmetry=symmetry,check_symmetry=False)
        self.assertEqual(axes,[0.,90.])
        with self.assertRaises(ValueError):
            pm.resolve_axes([0.,90.],data[0],symmetry=symmetry)

    def test_queue_does_not_label_stopped_response_as_verified_capacity(self):
        import abaqus_step5_queue as queue
        result=dict(outcome='INCOMPLETE_ANALYSIS',analysis_state='INCOMPLETE',curve_available=True,
                    analysis_completed=False,solver_exit_code=0,criterion_reached=False,partial=False)
        queue.capacity_status(result)
        self.assertFalse(result['capacity_verified'])
        self.assertFalse(queue.finished_result(result))
        self.assertTrue(queue.processed_result(result))
        self.assertFalse(queue.processed_result(dict(result,analysis_state='UNKNOWN')))


    def test_bending_resume_moment_plot_and_axis_protection(self):
        import abaqus_step5_queue as q
        from pathlib import Path
        with self.completed_family('bending',45.) as (pipeline,args,source,run,output):
            self.assertTrue(pipeline.preflight(args)[0][2]['completed'])
            with mock.patch.object(q,'plot_png',side_effect=lambda path,*a,**k:Path(path).write_bytes(b'PNG')):
                pipeline.run_pipeline(args,runner=mock.Mock(side_effect=AssertionError('must not solve')),
                                      abaqus='fake_abaqus')
            chart=run.parent/'STEP5_family_moment_rotation_BEND045.svg'
            self.assertTrue(chart.is_file())
            self.assertIn('End rotation',chart.read_text())
            self.assertIn('Bending moment',chart.read_text())
            self.assertFalse((run.parent/'STEP5_family_force_displacement.svg').exists())
            args.bending_axis_deg=0.
            with self.assertRaisesRegex(ValueError,'Bending axis'):
                pipeline.check_resume_build(args,source,run)
            args.bending_axis_deg=45.
            ref=Path(str(run)+'_REF_G')/'gdlc_reference.json'
            info=pipeline.read_json(ref);info['twist']='restrained'
            ref.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError,'twist free'):
                pipeline.check_resume_build(args,source,run)


    def test_bending_builder_and_step4_commands_use_same_loading_and_lateral_direction(self):
        from pathlib import Path
        with self.completed_family() as (pipeline,args,source,run,output):
            args.load_case='bending';args.bending_axis_deg=45.;args.global_modes='1';args.resume=False
            calls=[]
            def stop(command,cwd,label):
                calls.append((command,label))
                raise RuntimeError('captured build')
            with mock.patch.object(pipeline,'preflight',return_value=[(source,run,{})]):
                with self.assertRaisesRegex(RuntimeError,'captured build'):
                    pipeline.run_pipeline(args,runner=stop,abaqus='fake_abaqus')
            command=calls[0][0]
            self.assertEqual(command[command.index('--load-case')+1],'bending')
            self.assertEqual(float(command[command.index('--bending-axis-deg')+1]),45.)
            # Existing fixture supplies immutable files; native FE operations are mocked.
            state=pipeline.read_json(run/'pipeline_status.json')
            state['settings'].update(load_case='bending',bending_axis_deg=45.)
            state['build']['load_case']=dict(type='bending',neutral_axis_angle_deg=45.)
            (run/'pipeline_status.json').write_text(json.dumps(state))
            ref=Path(str(run)+'_REF_G')/'gdlc_reference.json'
            info=pipeline.read_json(ref);info['twist']='free';ref.write_text(json.dumps(info))
            calls.clear()
            def stop_step4(command,cwd,label):
                calls.append((command,label))
                if label.endswith('STEP4'):raise RuntimeError('captured step4')
            with mock.patch.object(pipeline,'preflight',return_value=[(source,run,{})]), \
                    mock.patch.object(pipeline,'completed_reference',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'captured step4'):
                    pipeline.run_pipeline(args,runner=stop_step4,abaqus='fake_abaqus')
            command=calls[-1][0]
            self.assertEqual(command[command.index('--global-mode')+1],'1')
            self.assertEqual(float(command[command.index('--global-angle-deg')+1]),45.)
            self.assertNotIn('--skip-mode-quality-checks',command)


    @contextlib.contextmanager
    def completed_family(self,load_case='compression',angle=0.,column_fy=240.):
        from pathlib import Path
        import abaqus_family_pipeline as pipeline
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            root=Path(d)
            source=root/'inputs'/'section'/'family_1';source.mkdir(parents=True)
            for name in pipeline.INPUT_FILES:(source/name).write_text('input')
            args=pipeline.parse_arguments(['--family-root',str(source.parent),'--output-root',str(root/'out'),
                '--repo',os.path.dirname(__file__),'--resume','--load-case',load_case,'--bending-axis-deg',str(angle),
                '--steel-column-plate-fy',str(column_fy)])
            run=args.output_root/source.parent.name/pipeline.family_run_name(args,source);run.mkdir(parents=True)
            state=dict(status='BUILT',settings=dict(vars(args),builtup_dir=str(source)),
                       source_inputs=dict(thickness_mm=2.),build=dict(job_name='BASE',length_mm=3600.))
            state['settings']={k:str(v) if isinstance(v,Path) else v for k,v in state['settings'].items()}
            lc=dict(type=load_case,neutral_axis_angle_deg=angle)
            if load_case=='bending':state['build']['load_case']=lc
            (run/'pipeline_status.json').write_text(json.dumps(state))
            for ext in ('.inp','.cae'):(run/('BASE'+ext)).write_text('base')
            for kind in ('L','G'):
                ref=Path(str(run)+'_REF_'+kind);ref.mkdir()
                (ref/'gdlc_reference.json').write_text(json.dumps(dict(kind=kind,source_run=run.name,
                    source_inp='BASE.inp',n_modes=20,job='REF_'+kind,load_case=lc,
                    twist='restrained' if load_case=='compression' else 'free')))
                (ref/('REF_'+kind+'.inp')).write_text('reference')
                (ref/('REF_'+kind+'.odb')).write_text('solved reference')
            cae=run/'imperfection'/'Step4_LG.cae';cae.parent.mkdir();cae.write_text('step4')
            output=cae.parent/args.step5_directory;output.mkdir()
            (output/('Step5_GMNIA_FY'+pipeline.column_fy_tag(args)+'.cae')).write_text('step5')
            (output/'run_step5_queue_70_pct.bat').write_text('queue')
            reference={key:pipeline.file_sha256(run/name) for key,name in
                [('source_inp_sha256','BASE.inp'),('source_cae_sha256','BASE.cae'),
                 ('pipeline_status_sha256','pipeline_status.json')]}
            spec=dict(settings=dict(local_mode=1,global_modes=[int(v) for v in args.global_modes.split(',')],global_angle_deg=pipeline.global_mode_angle(args),
                      local_source=str(run)+'_REF_L',global_source=str(run)+'_REF_G'),
                      component_sources={k:dict(odb=str(Path(str(run)+'_REF_'+kind)/('REF_'+kind+'.odb')),
                      odb_sha256=pipeline.file_sha256(Path(str(run)+'_REF_'+kind)/('REF_'+kind+'.odb')))
                      for k,kind in [('local','L'),('global','G')]})
            results=[]
            for name in sorted(pipeline.expected_jobs(args)):
                (output/(name+'.inp')).write_text('input '+name)
                perfect='PERFECT' in name
                info=dict(stage=5,job_name=name,input_sha256=pipeline.file_sha256(output/(name+'.inp')),
                    end_condition=dict(id=pipeline.STEP5_END_CONDITION_ID),
                    plastic_tables={'Steel':[[column_fy,0.]]},
                    reference_pipeline=reference,reference_force_N_per_end=1000.,end_area_mm2=1000./240.,load_case=lc,
                    settings=dict(source_cae=str(cae),fy=column_fy,initial_arc=.01,min_arc=1e-8,max_arc=.05,
                        max_increments=5000,field_frequency=1,postpeak_stop_ratio=.7,stop_method='monitor'),
                    source_step4=dict(specification=spec,case=dict(local_component_mm=0. if perfect else -.68,
                        distortional_component_mm=0.,global_component_mm=0. if perfect else 3.6)))
                if load_case=='bending':
                    info.pop('reference_force_N_per_end')
                    info.update(reference_moment_Nmm=1e6,first_yield_moment_Nmm=1e6,bending_axis_deg=angle)
                (output/(name+'_step5.json')).write_text(json.dumps(info))
                (output/(name+'.odb')).write_text('not opened')
                response='_moment_rotation' if load_case=='bending' else '_force_displacement'
                columns='rotation_mrad,moment_kNm' if load_case=='bending' else 'shortening_mm,force_kN'
                stem=output/(name+response)
                stem.with_suffix('.csv').write_text('frame,lpf,'+columns+'\n0,0,0,0\n1,1,1,1\n2,.6,.9,.6\n')
                stem.with_suffix('.json').write_text(json.dumps(dict(model=name,step='GMNIA',frames=3,partial=False)))
                results.append(dict(model=name,outcome='CRITERION_REACHED',criterion_reached=True,
                                    solver_exit_code=0,partial=False,**{('moment_kNm' if load_case=='bending' else 'force_kN'):1.}))
            (output/'STEP5_queue_status.json').write_text(json.dumps(results))
            (output/'STEP5_peak_summary.csv').write_text('model\n'+'\n'.join(sorted(pipeline.expected_jobs(args)))+'\n')
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


    def test_plate_parameters_reach_builder_and_nonlinear_material(self):
        import abaqus_step5_queue as queue
        with self.completed_family() as (pipeline,args,source,run,output):
            args.steel_plate_length=40.;args.steel_plate_length_2=60.
            args.steel_plate_gap_mm=0.;args.steel_plate_skip_count=2;args.steel_plate_fy=360.
            args.resume=False;calls=[]
            def capture(command,cwd,label):
                calls.append(command)
                raise RuntimeError('captured')
            with mock.patch.object(pipeline,'preflight',return_value=[(source,run,{})]):
                with self.assertRaisesRegex(RuntimeError,'captured'):
                    pipeline.run_pipeline(args,runner=capture,abaqus='fake_abaqus')
            command=calls[-1]
            for name,value in [('--steel-plate-length','40.0'),('--steel-plate-length-2','60.0'),
                               ('--steel-plate-gap-mm','0.0'),('--steel-plate-skip-count','2')]:
                self.assertEqual(command[command.index(name)+1],value)
            args.resume=True;calls.clear()
            with mock.patch.object(pipeline,'preflight',return_value=[(source,run,{'step4':True})]):
                with self.assertRaisesRegex(RuntimeError,'captured'):
                    pipeline.run_pipeline(args,runner=capture,abaqus='fake_abaqus')
            command=calls[-1]
            self.assertEqual(command[command.index('--steel-plate-fy')+1],'360.0')
            self.assertEqual(command[command.index('--source-cae')+1],str(run/'imperfection'/'Step4_LG.cae'))

    def test_resume_source_checks_full_direct_and_plate_bolt_layout(self):
        import abaqus_complete_model_m20 as builder
        from gdlc_classifier import gdlc_mode_decomposition as classifier
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np
        pipeline=__import__('abaqus_family_pipeline')
        data=ModelInputTests().plate_fixture()
        data[-1]['bolt_row_mm']=15.
        for height2,gap,skip in [(0.,0.,0),(40.,0.,1),(40.,10.,2)]:
            settings=dict(steel_plate_length=40.,steel_plate_length_2=height2,
                          steel_plate_gap_mm=gap,steel_plate_skip_count=skip)
            summary=builder.input_summary(data,plate_settings=dict(height=40.,height_2=height2,edge_gap=gap,skip_count=skip))
            nodes={};pairs=[]
            def node(instance,xyz):
                key=(instance,tuple(xyz))
                if key not in nodes:nodes[key]=(instance,sum(k[0]==instance for k in nodes)+1)
                return nodes[key]
            for row in summary['bolt_layout']:
                xa,ya=row['a_xy'];xb,yb=row['b_xy'];z=row['z_mm']
                a=node('P%d'%row['pa'],(xa,ya,z));b=node('P%d'%row['pb'],(xb,yb,z))
                if row['plate_instance']:
                    center=(xa,0.,z) if row['plate_direction']=='X' else (0.,ya,z)
                    ctl=node(row['plate_instance'],center);pairs.extend([('BEAM',ctl,a),('BEAM',ctl,b)])
                else:pairs.append(('BEAM',a,b))
            mesh=SimpleNamespace(instance=np.array([v[0] for v in nodes.values()]),
                                 label=np.array([v[1] for v in nodes.values()]),xyz=np.array([k[1] for k in nodes]))
            with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
                run=Path(d);state=dict(settings=settings,source_inputs=summary)
                with mock.patch.object(builder,'read_model_inputs',return_value=data), \
                     mock.patch.object(classifier,'read_inp',return_value=SimpleNamespace(pairs=pairs)), \
                     mock.patch.object(classifier,'load_mesh',return_value=mesh):
                    pipeline.check_source_inputs(run/'source',run,state,run/'job.inp')
                    mesh.xyz[0,0]+=1.
                    with self.assertRaisesRegex(ValueError,'seam coordinates'):
                        pipeline.check_source_inputs(run/'source',run,state,run/'job.inp')


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


    def queue_args(self,output):
        return types.SimpleNamespace(run_dir=str(output),cae=None,jobs=None,extract_only=False,
            repair_only=False,resume=True,cpus='auto',gpus='auto',ratio=.7,stop_method='monitor',
            poll_seconds=1.,progress_seconds=15.)


class SkipSweepTests(unittest.TestCase):
    def args(self,*extra):
        import abaqus_family_pipeline as pipeline
        return pipeline.parse_arguments(['--family-root','source','--output-root','target',
            '--steel-plate-length','40','--steel-plate-length-2','40',*extra])

    def test_list_and_scalar_are_backward_compatible_and_have_distinct_roots(self):
        import abaqus_family_pipeline as pipeline
        scalar=self.args('--steel-plate-skip-count','1')
        self.assertEqual(scalar.steel_plate_skip_count,1)
        self.assertEqual(pipeline.batch_output_directory(scalar),scalar.output_root/scalar.family_root.name)
        args=self.args('--steel-plate-skip-count','0,1,2,3','--resume')
        configs=pipeline.skip_configurations(args)
        self.assertEqual([a.steel_plate_skip_count for a in configs],[0,1,2,3])
        self.assertEqual(len({pipeline.batch_output_directory(a) for a in configs}),4)
        self.assertTrue(all(a.resume and a.steel_plate_fy==360. for a in configs))
        self.assertEqual([pipeline.batch_output_directory(a).name for a in configs],['skip_0','skip_1','skip_2','skip_3'])
        self.assertEqual(args.steel_plate_skip_count,(0,1,2,3))

    def test_invalid_skip_lists_are_rejected(self):
        for value in ('','1,','-1','0,-2','0,1,1','1.5','a','0,,2'):
            with self.subTest(value=value),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                self.args('--steel-plate-skip-count',value)

    def test_all_configurations_preflight_before_any_execution(self):
        import abaqus_family_pipeline as pipeline
        args=self.args('--steel-plate-skip-count','0,1,2,3')
        checked=[];runner=mock.Mock(side_effect=AssertionError('must not execute'))
        def preflight(config):
            checked.append(config.steel_plate_skip_count)
            if config.steel_plate_skip_count==3:raise ValueError('invalid final configuration')
            return []
        with mock.patch.object(pipeline,'preflight',side_effect=preflight):
            with self.assertRaisesRegex(ValueError,'invalid final'):
                pipeline.run_pipeline(args,runner=runner,abaqus='fake')
        self.assertEqual(checked,[0,1,2,3]);runner.assert_not_called()

    def test_pm_options_and_bending_reach_each_configuration(self):
        import abaqus_family_pipeline as pipeline
        args=self.args('--steel-plate-skip-count','0,1','--load-case','bending',
            '--bending-axis-deg','45','--pm-interaction','--pm-axes','0,90','--pm-angles','0,45,90')
        for config in pipeline.skip_configurations(args):
            self.assertTrue(config.pm_interaction)
            self.assertEqual((config.load_case,config.bending_axis_deg,config.global_modes),('bending',45.,'1'))
            self.assertEqual((config.pm_axes,config.pm_angles),('0,90','0,45,90'))


    def test_completed_sweep_reuses_all_four_configurations_without_launching(self):
        import abaqus_family_pipeline as pipeline
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            args=self.args('--steel-plate-skip-count','0,1,2,3','--resume')
            args.output_root=Path(d)
            runner=mock.Mock(side_effect=AssertionError('no solver or extraction'))
            checked=[]
            def preflight(config):
                checked.append(config.steel_plate_skip_count)
                return [(Path('family_1'),Path('unused'),dict(completed=True,entries={}))]
            with mock.patch.object(pipeline,'preflight',side_effect=preflight), \
                 mock.patch.object(pipeline,'update_family_plot'),mock.patch.object(pipeline.shutil,'which',return_value=None):
                pipeline.run_pipeline(args,runner=runner)
            self.assertEqual(checked,[0,1,2,3]);runner.assert_not_called()
            self.assertEqual({p.name for p in (args.output_root/args.family_root.name).iterdir()},
                             {'skip_0','skip_1','skip_2','skip_3'})

    def test_each_sweep_builder_gets_one_scalar_skip_and_the_correct_output_root(self):
        import abaqus_family_pipeline as pipeline
        from pathlib import Path
        fixture=WorkflowAlignmentTests()
        for count in range(4):
            with fixture.completed_family() as (_,args,source,run,output):
                args.steel_plate_length=40.;args.steel_plate_length_2=40.
                args.steel_plate_skip_count=count;args.skip_sweep=True;args.resume=False
                calls=[]
                def capture(command,cwd,label):
                    calls.append(command);raise RuntimeError('captured scalar builder')
                with mock.patch.object(pipeline,'preflight',return_value=[(source,run,{})]):
                    with self.assertRaisesRegex(RuntimeError,'captured scalar builder'):
                        pipeline.run_pipeline(args,runner=capture,abaqus='fake')
                command=calls[0]
                self.assertEqual(command[command.index('--steel-plate-skip-count')+1],str(count))
                self.assertEqual(Path(command[command.index('--output-root')+1]).name,'skip_%d'%count)

    def test_pm_build_queue_diagram_and_resume_use_only_matching_configuration(self):
        import abaqus_family_pipeline as pipeline
        import abaqus_pm_interaction as pm
        import abaqus_step5_queue as queue
        from pathlib import Path
        fixture=WorkflowAlignmentTests()
        with fixture.completed_family() as (_,args,source,run,unused):
            args.pm_interaction=True;args.pm_axes='0';args.pm_angles='0,45,90'
            args.steel_column_plate_fy=350.;args.steel_plate_fy=240.;args.steel_plate_length=40.
            output=run/'imperfection'/'Step5_PM_FY350_LG_REF1';calls=[]
            names=['PM_P00_PERFECT','PM_A000_P45_PERFECT','PM_A000_P90_PERFECT']
            def runner(command,cwd,label):
                calls.append(command)
                if '--mode' in command and command[command.index('--mode')+1]=='gmnia':
                    parsed=pm.parse_arguments(command[command.index('--')+1:])
                    self.assertEqual(parsed.step5.source_cae,str(run/'imperfection'/'Step4_LG.cae'))
                    self.assertEqual((parsed.step5.fy,parsed.step5.steel_plate_fy),(350.,240.))
                    output.mkdir()
                    for name in names:
                        inp=output/(name+'.inp');inp.write_text(name)
                        (output/(name+'_step5.json')).write_text(json.dumps(dict(stage=5,job_name=name,
                            input_sha256=pipeline.file_sha256(inp),load_case=dict(type='combined'),
                            end_condition=dict(id=pipeline.STEP5_END_CONDITION_ID))))
                    (output/pm.MANIFEST_FILE).write_text(json.dumps(dict(mode='gmnia',jobs=[dict(job=n) for n in names])))
                    (output/pm.SECTION_FILE).write_text('{}')
                elif '--mode' in command:
                    self.assertEqual(float(command[command.index('--fy')+1]),350.)
                    diagram=output/'diagram';diagram.mkdir(exist_ok=True)
                    (diagram/'PM_interaction_points.csv').write_text('diagram data')
                    (diagram/'PM_interaction.svg').write_text('<svg/>')
                else:
                    self.assertIn('--resume',command)
                    (output/'STEP5_queue_status.json').write_text(json.dumps([
                        dict(model=name,outcome='CRITERION_REACHED',criterion_reached=True,solver_exit_code=0) for name in names]))
                    (output/'STEP5_PM_points.csv').write_text('PM data')
            pipeline.run_family_pm(args,source,run,'fake',runner,'test')
            self.assertEqual(len(calls),3)
            entry=(dict(criterion_reached=True,capacity_verified=True),[dict(lpf=1.)])
            with mock.patch.object(queue,'completed_curve',return_value=entry):
                state=pipeline.check_pm_resume(args,source,run)
                self.assertTrue(state['completed'])
                pipeline.run_family_pm(args,source,run,None,mock.Mock(side_effect=AssertionError('no repeat')), 'test',state=state)
                (output/'diagram'/'PM_interaction.svg').unlink()
                state=pipeline.check_pm_resume(args,source,run)
                self.assertTrue(state['queue_completed']);self.assertFalse(state['completed'])
                calls.clear()
                pipeline.run_family_pm(args,source,run,None,runner,'test',state=state)
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0][calls[0].index('--mode')+1],'collect')
                args.pm_angles='0,30,90'
                with self.assertRaisesRegex(ValueError,'P-M settings'):
                    pipeline.check_pm_resume(args,source,run)
                args.pm_angles='0,45,90'
                (output/(names[0]+'.inp')).write_text('changed model')
                with self.assertRaisesRegex(ValueError,'input/metadata'):
                    pipeline.check_pm_resume(args,source,run)


class FamilyPipelineTests(unittest.TestCase):
    def test_family_pipeline_longitudinal_lines_match_builder_limits(self):
        p = self.pipeline_module()
        base = ['--family-root', '.', '--output-root', 'out', '--repo', '.']
        for value in ('0', '2', '100'):
            self.assertEqual(p.parse_arguments(base+['--longitudinal-lines', value]).longitudinal_lines, int(value))
        for value in ('1', '101', '-1'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p.parse_arguments(base+['--longitudinal-lines', value])

    def pipeline_module(self):
        import importlib.util
        path = os.path.join(os.path.dirname(__file__), 'abaqus_family_pipeline.py')
        self.assertTrue(os.path.isfile(path), 'Parameterized family pipeline is missing')
        spec = importlib.util.spec_from_file_location('abaqus_family_pipeline', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, root, names=('family_10', 'family_2', 'family_1')):
        from pathlib import Path
        family_root, repo, output = (Path(root)/name for name in ('families with spaces', 'repo', 'results'))
        repo.mkdir()
        for filename in self.pipeline_module().REPO_FILES+self.pipeline_module().REPO_DEPENDENCIES:
            path = repo/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('test external boundary', encoding='utf-8')
        for name in names:
            folder = family_root/name
            folder.mkdir(parents=True)
            for filename in ('builtup_member.csv', 'builtup_segments.csv', 'builtup_seams.csv', 'builtup_bolts_y.csv'):
                (folder/filename).write_text('test external boundary', encoding='utf-8')
        return ['--family-root', str(family_root), '--output-root', str(output), '--repo', str(repo)], output

    def test_family_pipeline_checks_all_inputs_before_creating_outputs(self):
        p = self.pipeline_module()
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as root:
            argv, output = self.fixture(root)
            (Path(argv[1])/'family_10'/'builtup_seams.csv').unlink()
            with self.assertRaisesRegex(ValueError, 'builtup_seams.csv'):
                p.preflight(p.parse_arguments(argv))
            self.assertFalse(output.exists())

    def test_family_pipeline_all_stages_sequential_and_parameters_forwarded(self):
        p = self.pipeline_module()
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as root:
            argv, output = self.fixture(root)
            args = p.parse_arguments(argv+['--mesh-mm', '12.5', '--n-modes', '80', '--n-vectors', '160',
                '--max-iterations', '400', '--cpus', 'auto', '--longitudinal-lines', '3',
                '--longitudinal-line-min-spacing-mm', '4', '--local-mode', '3', '--global-modes', '2,3'])
            calls = []
            fail_queue = [False]
            def fake_runner(command, cwd, label):
                self.assertNotEqual(Path(cwd), args.repo, 'Do not launch CAE in the code repository')
                self.assertNotEqual(Path(cwd), args.repo/'gdlc_classifier')
                calls.append(list(command))
                def arg(key): return command[command.index(key)+1]
                def save(path, value):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value), encoding='utf-8')
                if '--builtup-dir' in command:
                    run = Path(arg('--output-root'))/Path(arg('--builtup-dir')).name
                    save(run/'pipeline_status.json', {'status': 'BUILT', 'build': {'job_name': 'J'}})
                    (run/'J.cae').touch(); (run/'J.inp').touch()
                elif '--make-reference-models' in command:
                    for kind in ('L', 'G'):
                        save(Path(command[2]+'_REF_'+kind)/'gdlc_reference.json',
                             {'job': 'R', 'twist': 'restrained'} if kind == 'G' else {'job': 'R'})
                elif '--solve-reference-models' in command:
                    folder = Path(command[2]); (folder/'R.odb').touch()
                    path = folder/'mode_decomposition_GDLC'/'mode_participation.csv'
                    path.parent.mkdir(); path.touch()
                elif '--output-cae' in command:
                    path = Path(arg('--output-cae')); path.parent.mkdir(); path.touch()
                elif '--source-cae' in command:
                    folder = Path(arg('--output-dir')); folder.mkdir()
                    (folder/'Step5_GMNIA_FY240.cae').touch()
                    (folder/'run_step5_queue_70_pct.bat').touch()
                    for name in ('STEP5_PERFECT_FY240', 'STEP5_LG_mp_FY240'):
                        (folder/(name+'.inp')).touch()
                        save(folder/(name+'_step5.json'), {'job_name': name})
                else:
                    folder = Path(cwd)
                    save(folder/'STEP5_queue_status.json', [
                        {'model': 'STEP5_PERFECT_FY240', 'outcome': 'CRITERION_REACHED', 'criterion_reached':True, 'solver_exit_code':0},
                        {'model': 'STEP5_LG_mp_FY240', 'outcome': 'SOLVER_ERROR' if fail_queue[0] else 'CONTROLLED_TERMINATION',
                         'criterion_reached':not fail_queue[0], 'solver_exit_code':1,
                         'termination_requested':not fail_queue[0]}])
                    (folder/'STEP5_peak_summary.csv').touch()
            with contextlib.ExitStack() as patches:
                patches.enter_context(mock.patch.object(p,'step4_receipt',return_value={}))
                patches.enter_context(mock.patch.object(p,'completed_reference',return_value=True))
                patches.enter_context(mock.patch.object(p,'check_step5_inputs',return_value={name:dict(settings={}) for name in p.EXPECTED_JOBS}))
                patches.enter_context(mock.patch.object(__import__('abaqus_step5_queue'),'completed_curve',return_value=({},[{}])))
                patches.enter_context(mock.patch.object(p,'update_family_plot'))
                patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
                p.run_pipeline(args, runner=fake_runner, abaqus='abaqus.bat')
            self.assertEqual(len(calls), 21)
            builds = [c for c in calls if '--builtup-dir' in c]
            self.assertEqual([Path(c[c.index('--builtup-dir')+1]).name for c in builds],
                             ['family_1', 'family_2', 'family_10'])
            makes = [c for c in calls if '--make-reference-models' in c]
            self.assertEqual(len(makes), 3)
            self.assertTrue(all(c[c.index('--reference-kinds')+1] == 'L,G' for c in makes))
            for command in builds:
                self.assertEqual(command[command.index('--mesh-mm')+1], '12.5')
                self.assertEqual(command[command.index('--n-modes')+1], '80')
                self.assertIn('--build-only', command)
            step4 = next(c for c in calls if '--output-cae' in c)
            self.assertEqual(step4[step4.index('--local-mode')+1], '3')
            self.assertEqual(step4[step4.index('--global-mode')+1], '2,3')
            self.assertEqual(step4[step4.index('--combo')+1], 'LG_mp=-0.34t,0,L/1000')
            self.assertTrue((output/'families with spaces'/'family_10'/'imperfection'/'Step5_GMNIA_FY240_LG_REF1'/'STEP5_peak_summary.csv').is_file())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                p.preflight(args)
            args.output_root = Path(root)/'failed results'
            calls.clear()
            fail_queue[0] = True
            with contextlib.ExitStack() as patches:
                patches.enter_context(mock.patch.object(p,'step4_receipt',return_value={}))
                patches.enter_context(mock.patch.object(p,'completed_reference',return_value=True))
                patches.enter_context(mock.patch.object(p,'check_step5_inputs',return_value={name:dict(settings={}) for name in p.EXPECTED_JOBS}))
                patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with self.assertRaisesRegex(RuntimeError, 'queue/plots failed'):
                    p.run_pipeline(args, runner=fake_runner, abaqus='abaqus.bat')
            self.assertEqual(len(calls), 7, 'A failed queue must not start the next family')

    def test_global_modes_resolve_to_one_complete_ref_g_eigenvalue_group(self):
        p = self.pipeline_module()
        from pathlib import Path
        from types import SimpleNamespace as NS
        table = ('\n MODE NO      EIGENVALUE\n\n\n       1       %s    \n       2       %s    \n'
                 '       3       %s    \n\n THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n')
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as root:
            run = Path(root)/'family_1_BEND000'
            ref = Path(str(run)+'_REF_G'); ref.mkdir(parents=True)
            (ref/'gdlc_reference.json').write_text(json.dumps({'job': 'J'}))
            args = NS(global_modes='1,2', load_case='bending', bending_axis_deg=0.)
            # REF_G not solved yet: the request is passed on unchanged
            self.assertIs(p.family_args(args, run), args)
            with mock.patch.object(p, 'completed_reference', return_value=True), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                (ref/'J.dat').write_text(table % ('22745.', '37010.', '47043.'))
                resolved = p.family_args(args, run)
                self.assertEqual(resolved.global_modes, '1')          # single lateral-torsional mode
                self.assertEqual(args.global_modes, '1,2')            # caller's arguments untouched
                self.assertIn('WARNING', out.getvalue())
                self.assertEqual(p.global_mode_angle(resolved), 0.)
                (ref/'J.dat').write_text(table % ('3638.9', '3638.9', '7934.1'))
                compression = NS(global_modes='1,2', load_case='compression', bending_axis_deg=0.)
                self.assertIs(p.family_args(compression, run), compression)   # complete degenerate pair
                wrong = NS(global_modes='1,3', load_case='compression', bending_axis_deg=0.)
                self.assertEqual(p.family_args(wrong, run).global_modes, '1,2')
                single = NS(global_modes='1', load_case='compression', bending_axis_deg=0.)
                self.assertIs(p.family_args(single, run), single)

    def test_resume_regenerates_only_unfinished_references(self):
        p = self.pipeline_module()
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as root:
            run = Path(root)/'family_1'
            for kind in ('L', 'G'):
                ref = Path(str(run)+'_REF_'+kind); ref.mkdir(parents=True)
                (ref/'gdlc_reference.json').write_text(json.dumps({'job': 'J_'+kind}))
            with mock.patch.object(p, 'completed_reference', side_effect=lambda ref, job, kind: kind == 'G'):
                self.assertEqual([k for k in ('L', 'G') if not p.reference_ready(run, k)], ['L'])
            self.assertFalse(p.reference_ready(Path(root)/'missing', 'L'))

    def test_family_pipeline_failed_build_cannot_start_reference_solve(self):
        p = self.pipeline_module()
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as root:
            argv, output = self.fixture(root, ('family_1',))
            calls = []
            def failed_build(command, cwd, label):
                calls.append(command)
                folder = output/'families with spaces'/'family_1'
                folder.mkdir(parents=True)
                (folder/'pipeline_status.json').write_text('{"status":"FAILED"}', encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'BUILT'):
                p.run_pipeline(p.parse_arguments(argv), runner=failed_build, abaqus='abaqus.bat')
            self.assertEqual(len(calls), 1)

    @unittest.skipUnless(os.name == 'nt', 'Windows batch launcher')
    def test_family_pipeline_batch_paths_and_arguments_with_spaces(self):
        p = self.pipeline_module()
        from pathlib import Path
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__),prefix='family check & spaces ') as root:
            batch = Path(root)/'check args.bat'
            batch.write_text('@echo off\nif not "%~1"=="hello world" exit /b 17\n'
                             'if not "%~2"=="a&b" exit /b 18\nexit /b 0\n')
            with contextlib.redirect_stdout(io.StringIO()):
                p.run_command([str(batch), 'hello world', 'a&b'], root, 'batch test')


@unittest.skipUnless(os.environ.get('CFS_CROSS_PLATE_GEOMETRY_DIR'), 'optional plate-only CAE geometry check')
class CrossPlateGeometryTests(unittest.TestCase):
    def test_native_cross_shares_joint_nodes_and_exact_bolt_locations(self):
        import caeModules
        from abaqus import mdb
        from abaqusConstants import CARTESIAN
        counts=[]
        for j,(height,width) in enumerate(((40.,200.),(30.,240.))):
            data=ModelInputTests().plate_fixture(400.,(40.,120.,200.,280.,360.))
            data[0][1][0]=(6.,200.,width,6.)
            data[0][2][0]=(-width,6.,-6.,200.)
            data[0][3][0]=(-6.,-200.,-width,-6.)
            data[0][4][0]=(width,-6.,6.,-200.)
            layout=builder.steel_plate_layout(data,40.,height,0.,skip_count=1)
            model=mdb.Model(name='CrossGeometry%d'%j)
            assembly=model.rootAssembly
            assembly.DatumCsysByDefault(CARTESIAN)
            instances=builder.build_steel_plates(model,assembly,layout)
            self.assertEqual(sorted(instances),layout['instances'])
            self.assertEqual(len(assembly.instances),len(layout['instances']))
            for name,z in zip(layout['instances'],layout['bolt_positions_mm']):
                inst=instances[name]
                builder.validate_cross_plate_mesh(model.parts[inst.partName],height)
                for x,y in ((-160.,0.),(160.,0.),(0.,-160.),(0.,160.)):
                    self.assertEqual(len(inst.nodes.getByBoundingSphere(center=(x,y,z),radius=1e-6)),1)
                self.assertAlmostEqual(min(n.coordinates[2] for n in inst.nodes),z-height/2.)
                self.assertAlmostEqual(max(n.coordinates[2] for n in inst.nodes),z+height/2.)
            self.assertEqual(len(model.constraints),0) # continuity comes from shared nodes
            counts.append(dict(height_mm=height,width_mm=2*width,assemblies=len(instances),
                nodes_per_assembly=len(next(iter(instances.values())).nodes)))
        with open(os.path.join(os.environ['CFS_CROSS_PLATE_GEOMETRY_DIR'],'cross_geometry_check.json'),'w') as stream:
            json.dump(dict(verified=counts,solver_submitted=False),stream,indent=2)


@unittest.skipUnless(os.environ.get('CFS_STEEL_PLATE_SMOKE_DIR'), 'optional build-only CAE integration check')
class PlateCaeIntegrationTests(unittest.TestCase):
    def test_step5_build_delivers_cae_inputs_metadata_and_queue_with_plate_fy360(self):
        """Exercise the public builder, including opening Step4 and publishing the BAT; never solve."""
        import numpy as np
        import caeModules
        from abaqus import openMdb
        import abaqus_pipeline_contract as contract
        import abaqus_step4_imperfections as step4
        import abaqus_step5_gmnia as step5
        root = os.environ['CFS_STEEL_PLATE_SMOKE_DIR']
        with open(os.path.join(root, 'pipeline_status.json')) as stream:
            state = json.load(stream)
        validated = contract.validate_reference_state(state)
        job = validated['job_name']
        reference = dict(validated, run_dir=root, instances=list(contract.REFERENCE_INSTANCES),
            settings=state['settings'], paths={})
        snapshot = contract.reference_snapshot(reference, include_hashes=False)
        with tempfile.TemporaryDirectory(prefix='plate_step5_delivery_') as folder:
            source_cae = os.path.join(folder, 'Step4_smoke.cae')
            output = os.path.join(folder, 'Step5_smoke')
            database = openMdb(pathName=os.path.join(root, job+'.cae'))
            try:
                source = database.models[job]
                manifest = dict(stage=4, reference_pipeline=snapshot,
                    source_repository_counts={key: len(getattr(source, key)) for key in
                        ('boundaryConditions', 'constraints', 'interactions', 'loads', 'materials', 'sections')})
                source.setValues(description=step5.STEP4_MANIFEST_MARKER+json.dumps(manifest))
                model = database.Model(name='STEP4_PLATE_SMOKE', objectToCopy=source)
                names = snapshot['instances']+contract.plate_instance_names(snapshot)
                keys = [(name, n.label) for name in names for n in model.rootAssembly.instances[name].nodes]
                xyz = np.asarray([n.coordinates for name in names for n in model.rootAssembly.instances[name].nodes])
                topology = {name: {e.label: tuple(n.label for n in e.getNodes())
                    for e in model.rootAssembly.instances[name].elements} for name in names}
                step4.apply_offsets(model, names, keys, xyz, step4.global_bow(xyz, 2, 0.)*.1, topology)
                model.setValues(description=step5.STEP4_PREFIX+json.dumps(
                    dict(specification=manifest, current_case={'global_component_mm': .1, 'local_component_mm': 0.})))
                database.saveAs(pathName=source_cae)
            finally:
                database.close()
            args = step5.parse_arguments(['--source-cae', source_cae, '--output-dir', output,
                '--fy', '240', '--steel-plate-fy', '360', '--initial-arc', '.01', '--min-arc', '1e-8',
                '--max-arc', '.05', '--max-increments', '5000', '--field-frequency', '1',
                '--postpeak-stop-ratio', '.70'])
            step5.build(args)
            expected = ('STEP5_PERFECT_FY240', 'STEP5_PLATE_SMOKE_FY240')
            for name in expected:
                self.assertTrue(os.path.isfile(os.path.join(output, name+'.inp')))
                with open(os.path.join(output, name+step5.METADATA_SUFFIX)) as stream:
                    info = json.load(stream)
                self.assertEqual(info['plastic_tables']['SteelPlate'], [[360., 0.]])
                self.assertEqual(info['plastic_tables']['Steel'], [[240., 0.]])
                self.assertEqual(len(info['bolts']), snapshot['expected_links'])
                self.assertEqual(info['input_sha256'], contract.file_sha256(os.path.join(output, name+'.inp')))
            final_cae = os.path.join(output, 'Step5_GMNIA_FY240.cae')
            self.assertTrue(os.path.isfile(final_cae))
            with open(os.path.join(output, 'run_step5_queue_70_pct.bat')) as stream:
                queue = stream.read()
            for name in expected:
                self.assertIn(name, queue)
            self.assertFalse(any(name.endswith('.odb') for name in os.listdir(output)))
            database = openMdb(pathName=final_cae)
            try:
                self.assertEqual(sorted(database.models.keys()), sorted(expected))
                for model in database.models.values():
                    np.testing.assert_allclose(model.materials['SteelPlate'].plastic.table, ((360., 0.),))
                    self.assertEqual(len(model.rootAssembly.instances), 4+len(contract.plate_instance_names(snapshot)))
            finally:
                database.close()

    def test_plate_imperfect_mesh_and_step5_input_without_solver(self):
        import numpy as np
        import caeModules
        from abaqus import openMdb
        import abaqus_pipeline_contract as contract
        import abaqus_step4_imperfections as step4
        import abaqus_step5_gmnia as step5
        root = os.environ['CFS_STEEL_PLATE_SMOKE_DIR']
        with open(os.path.join(root, 'pipeline_status.json')) as stream:
            state = json.load(stream)
        validated = contract.validate_reference_state(state)
        job = validated['job_name']
        reference = dict(validated, run_dir=root, instances=list(contract.REFERENCE_INSTANCES),
            settings=state['settings'], paths={})
        snapshot = contract.reference_snapshot(reference, include_hashes=False)
        expected_rotations=([0,90,180,270] if snapshot['steel_plates'].get('assembly_type')=='CONNECTED_CROSS'
                            else [0,180])
        self.assertEqual(snapshot['load_case']['section_symmetry']['rotations_deg'], expected_rotations)
        database = openMdb(pathName=os.path.join(root, job+'.cae'))
        try:
            model = database.models[job]
            names = snapshot['instances']+contract.plate_instance_names(snapshot)
            for group in snapshot['steel_plates'].get('groups') or [snapshot['steel_plates']]:
                direction = group['direction']
                normal_axis = 1 if direction == 'X' else 0
                transverse_axis = 0 if direction == 'X' else 1
                limits = group[('x' if direction == 'X' else 'y')+'_limits_mm']
                for name, bounds in zip(group['instances'], group['z_intervals_mm']):
                    points = np.array([n.coordinates for n in model.rootAssembly.instances[name].nodes])
                    self.assertLess(np.max(np.abs(points[:, normal_axis])), 1e-7)
                    np.testing.assert_allclose([points[:, 2].min(), points[:, 2].max()], bounds, atol=1e-7)
                    np.testing.assert_allclose([points[:, transverse_axis].min(), points[:, transverse_axis].max()], limits, atol=1e-7)
            for bolt in snapshot.get('bolt_layout', []):
                base = 'BOLT_S%d_%02d' % (bolt['seam_id'], bolt['row'])
                a = model.rootAssembly.sets[base+'_A'].nodes
                b = model.rootAssembly.sets[base+'_B'].nodes
                self.assertEqual((len(a), len(b)), (1, 1))
                self.assertAlmostEqual(a[0].coordinates[2], bolt['z_mm'])
                self.assertAlmostEqual(b[0].coordinates[2], bolt['z_mm'])
                self.assertEqual(base+'_PLATE' in model.constraints, bool(bolt['plate_instance']))
            keys = [(name, n.label) for name in names for n in model.rootAssembly.instances[name].nodes]
            xyz = np.asarray([n.coordinates for name in names for n in model.rootAssembly.instances[name].nodes])
            topology = {name: {e.label: tuple(n.label for n in e.getNodes())
                for e in model.rootAssembly.instances[name].elements} for name in names}
            mesh_folder = os.path.join(root, 'mesh_smoke')
            os.makedirs(mesh_folder, exist_ok=True)
            np.savez(os.path.join(mesh_folder, 'raw_dof_map.npz'),
                node_instances=np.array([key[0] for key in keys]),
                node_labels=np.array([key[1] for key in keys]), node_coordinates=xyz)
            offsets = step4.global_bow(xyz, 2, 0.)*.1
            step4.apply_offsets(model, names, keys, xyz, offsets, topology)
            manifest = dict(stage=4, reference_pipeline=snapshot,
                source_repository_counts={key: len(getattr(model, key)) for key in
                    ('boundaryConditions', 'constraints', 'interactions', 'loads', 'materials', 'sections')})
            model.setValues(description=step5.STEP4_PREFIX+json.dumps(
                dict(specification=manifest, current_case={'local_component_mm': 0.})))
            args = step5.parse_arguments(['--source-cae', os.path.join(root, job+'.cae'),
                '--output-dir', os.path.join(root, 'step5_smoke'), '--fy', '240', '--steel-plate-fy', '355', '--gpus', '0'])
            info = step5.prepare_model(model, args)
            self.assertEqual(info['plastic_tables']['SteelPlate'], [[355., 0.]])
            self.assertEqual(info['plastic_tables']['Steel'], [[240., 0.]])
            self.assertEqual(len(info['bolts']), snapshot['expected_links'])
            with tempfile.TemporaryDirectory(prefix='plate_step5_input_') as folder:
                previous = os.getcwd()
                try:
                    os.chdir(folder)
                    step5.write_job_input(database, job, info, args, '')
                    self.assertTrue(os.path.isfile(job+'.inp'))
                    self.assertFalse(os.path.isfile(job+'.odb'))
                finally:
                    os.chdir(previous)
        finally:
            database.close()


if __name__ == '__main__':
    if os.environ.get('CFS_CROSS_PLATE_GEOMETRY_DIR'):
        outcome=unittest.TextTestRunner(verbosity=2).run(
            unittest.TestLoader().loadTestsFromTestCase(CrossPlateGeometryTests))
        if not outcome.wasSuccessful():
            raise RuntimeError('Cross plate geometry check failed; no analyses submitted')
    elif os.environ.get('CFS_STEEL_PLATE_SMOKE_DIR'):
        smoke_root = os.environ['CFS_STEEL_PLATE_SMOKE_DIR']
        with open(os.path.join(smoke_root, 'cae_integration_test.log'), 'w') as stream:
            with contextlib.redirect_stdout(stream):
                outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(
                    unittest.TestLoader().loadTestsFromTestCase(PlateCaeIntegrationTests))
        if not outcome.wasSuccessful() or outcome.skipped:
            raise RuntimeError('Build-only plate integration failed; see cae_integration_test.log')
        with open(os.path.join(smoke_root, 'cae_integration_verified.json'), 'w') as stream:
            json.dump({'passed': outcome.testsRun, 'solver_submitted': False}, stream)
    else:
        unittest.main()

