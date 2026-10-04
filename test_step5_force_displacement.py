import math
import unittest
import abaqus_step5_force_displacement as fd


class ForceDisplacementTests(unittest.TestCase):
    def test_weight_map_rejects_duplicate_and_nonpositive(self):
        self.assertEqual(fd.weight_map([['P1', 1, 2.0]]), {('P1', 1): 2.0})
        with self.assertRaises(ValueError):
            fd.weight_map([['P1', 1, 2.0], ['p1', 1, 3.0]])
        with self.assertRaises(ValueError):
            fd.weight_map([['P1', 1, 0.0]])

    def test_lpf_for_frame_matches_history_abscissa_not_frame_value_as_lpf(self):
        history = [(0.0, 0.0), (0.1, 0.75), (0.2, 1.10)]
        self.assertAlmostEqual(fd.lpf_for_frame(0.1, history), 0.75)
        self.assertAlmostEqual(fd.lpf_for_frame(0.2, history), 1.10)

    def test_peak_summary_uses_maximum_force(self):
        rows = [
            dict(frame=0, lpf=0.0, force_N=0.0, force_kN=0.0, shortening_mm=0.0,
                 nominal_stress_MPa=0.0),
            dict(frame=1, lpf=1.2, force_N=12000.0, force_kN=12.0, shortening_mm=2.0,
                 nominal_stress_MPa=288.0),
            dict(frame=2, lpf=0.9, force_N=9000.0, force_kN=9.0, shortening_mm=4.0,
                 nominal_stress_MPa=216.0)]
        peak = fd.peak_summary(rows)
        self.assertEqual(peak['frame'], 1)
        self.assertAlmostEqual(peak['force_kN'], 12.0)
        self.assertAlmostEqual(peak['shortening_mm'], 2.0)

    def test_ticks_are_finite_and_ordered(self):
        ticks = fd._nice_ticks(0.0, 13.0)
        self.assertTrue(ticks)
        self.assertTrue(all(math.isfinite(x) for x in ticks))
        self.assertEqual(ticks, sorted(ticks))


if __name__ == '__main__':
    unittest.main()
