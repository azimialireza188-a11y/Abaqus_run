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
