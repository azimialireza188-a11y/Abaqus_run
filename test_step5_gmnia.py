import ast
import contextlib
import io
import unittest
import abaqus_step5_gmnia as step5


class Step5Tests(unittest.TestCase):
    def test_fy_is_required_and_arc_limits_are_consistent(self):
        base = ['--source-cae', 'source.cae', '--output-dir', 'out']
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step5.parse_arguments(base)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step5.parse_arguments(base+['--fy', '350', '--initial-arc', '.1', '--max-arc', '.01'])
        args = step5.parse_arguments(base+['--fy', '350'])
        self.assertEqual(args.reference_stress, 350.)

    def test_default_postpeak_stop_ratio_is_70_percent(self):
        base = ['--source-cae', 'source.cae', '--output-dir', 'out', '--fy', '240']
        args = step5.parse_arguments(base)
        self.assertAlmostEqual(args.postpeak_stop_ratio, 0.70)
        self.assertEqual(args.max_increments, 5000)
        self.assertIsNone(args.max_end_displacement_mm)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step5.parse_arguments(base+['--postpeak-stop-ratio', '1.0'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step5.parse_arguments(base+['--postpeak-stop-ratio', '0'])

    def test_urdfil_tracks_peak_and_stops_at_ratio(self):
        source = step5.render_postpeak_urdfil(0.70)
        self.assertIn('SUBROUTINE URDFIL', source)
        self.assertIn('CALL POSFIL(KSTEP,KINC,ARRAY,JRCD)', source)
        self.assertIn('JRRAY(1,2).EQ.2000', source)
        self.assertIn('CURRLPF=ARRAY(11)', source)
        self.assertIn('PEAKLPF=CURRLPF', source)
        self.assertIn('CURRLPF.LE.RATIO*PEAKLPF', source)
        self.assertIn('LSTOP=1', source)
        self.assertIn('RATIO=0.7D0', source)

    def test_perfect_reference_is_a_default_step5_case(self):
        with open(step5.__file__, encoding='utf-8') as stream:
            source = stream.read()
        self.assertIn("PERFECT_TOKEN = 'PERFECT'", source)
        self.assertIn("name = 'STEP5_PERFECT_FY'+tag", source)
        self.assertIn('perfect_reference=True', source)
        self.assertIn('[PERFECT_TOKEN] + imperfect', source)

    def test_inp_trigger_is_injected_before_end_step(self):
        import os
        import tempfile
        handle, path = tempfile.mkstemp(suffix='.inp')
        os.close(handle)
        try:
            with open(path, 'w') as stream:
                stream.write('*HEADING\n*STEP\n*STATIC, RIKS\n0.1, 1.0\n*END STEP\n')
            step5.inject_urdfil_trigger(path)
            with open(path) as stream:
                text = stream.read()
            self.assertIn('*NODE FILE, NSET=STEP5_STOP, FREQUENCY=1', text)
            self.assertLess(text.index('*NODE FILE'), text.index('*END STEP'))
        finally:
            os.remove(path)

    def test_step5_requires_pipeline_compatible_step4_provenance(self):
        with open(step5.__file__, encoding='utf-8') as stream:
            source = stream.read()
        self.assertIn("STEP4_PREFIX = 'STEP4_PIPELINE_COMPATIBLE '", source)
        self.assertIn('pipeline_contract.validate_step4_payload', source)
        self.assertIn("reference['expected_links']", source)
        self.assertIn("source='BEAM_MPC from abaqus_complete_model_m20.py'", source)
        self.assertIn("added_compliance='none intentionally introduced'", source)

    def test_step5_job_inherits_reference_nodal_precision_contract(self):
        with open(step5.__file__, encoding='utf-8') as stream:
            source = stream.read()
        self.assertIn("nodal_precision = str(info['reference_pipeline'].get('nodal_precision'", source)
        self.assertIn('memory=100, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=False', source)
        self.assertIn('nodalOutputPrecision=FULL if nodal_precision', source)

    def test_end_weights_and_reference_force_use_one_end(self):
        # A shell 10 mm wide, 100 mm long, 2 mm thick.
        xyz = {1: (0., 0., 0.), 2: (10., 0., 0.),
               3: (10., 0., 100.), 4: (0., 0., 100.)}
        weights = step5.end_weights(xyz, [(1, 2, 3, 4)], 2., 0., 100.)
        self.assertEqual(weights[0], {1: 10., 2: 10.})
        self.assertEqual(weights[1], {3: 10., 4: 10.})
        self.assertEqual(sum(weights[0].values())*350., 7000.)

    def test_no_solver_submission_path_exists(self):
        with open(step5.__file__, encoding='utf-8') as stream:
            tree = ast.parse(stream.read())
        forbidden = {'submit', 'waitForCompletion', 'system', 'Popen'}
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(called & forbidden)


if __name__ == '__main__':
    unittest.main()
