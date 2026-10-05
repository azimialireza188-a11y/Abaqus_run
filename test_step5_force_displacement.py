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

    def test_weighted_u3_survives_abaqus_sum_shadowing(self):
        import runpy
        from types import SimpleNamespace as NS

        def abaqus_sum(*args, **kwargs):
            raise TypeError("arg1; expecting a recognized Abaqus type")

        namespace = runpy.run_path(
            fd.__file__, init_globals={'sum': abaqus_sum},
            run_name='force_displacement_sum_shadow_test')
        instance = NS(name='P1')
        values = [
            NS(instance=instance, nodeLabel=1, precision='SINGLE_PRECISION', data=(0.0, 0.0, 2.0)),
            NS(instance=instance, nodeLabel=2, precision='SINGLE_PRECISION', data=(0.0, 0.0, 4.0)),
        ]
        field = NS(getSubset=lambda region: NS(values=values))
        weights = {('P1', 1): 1.0, ('P1', 2): 3.0}
        self.assertAlmostEqual(namespace['weighted_u3'](field, object(), weights), 3.5)

    def test_lpf_history_alignment_uses_sequence_when_abscissae_are_offset(self):
        from types import SimpleNamespace as NS
        frames = [NS(frameValue=0.0), NS(frameValue=0.10), NS(frameValue=0.20)]
        history = [(0.0, 0.0), (0.05, 0.75), (0.15, 1.10)]
        values, meta = fd.align_lpf_history(frames, history)
        self.assertEqual(values, [0.0, 0.75, 1.10])
        self.assertEqual(meta['mode'], 'same_count_by_sequence')
        self.assertAlmostEqual(meta['max_abs_frameValue_minus_historyX'], 0.05)

    def test_lpf_history_alignment_accepts_missing_initial_sample(self):
        from types import SimpleNamespace as NS
        frames = [NS(frameValue=0.0), NS(frameValue=0.10), NS(frameValue=0.20)]
        history = [(0.05, 0.75), (0.15, 1.10)]
        values, meta = fd.align_lpf_history(frames, history)
        self.assertEqual(values, [0.0, 0.75, 1.10])
        self.assertEqual(meta['mode'], 'synthesized_initial_zero_then_sequence')

    def test_lpf_history_alignment_rejects_ambiguous_count(self):
        from types import SimpleNamespace as NS
        frames = [NS(frameValue=0.0), NS(frameValue=0.10), NS(frameValue=0.20)]
        with self.assertRaises(ValueError):
            fd.align_lpf_history(frames, [(0.0, 0.0)])

    def test_partial_live_odb_uses_only_synchronized_prefix(self):
        from types import SimpleNamespace as NS

        class FakeField:
            def __init__(self, values):
                self.values = values
            def getSubset(self, region=None):
                return self

        instance = NS(name='P1')
        value = NS(instance=instance, nodeLabel=1, precision='SINGLE_PRECISION', data=(0.0, 0.0, 0.0))
        frames = []
        for i in range(6):
            frames.append(NS(frameValue=float(i), fieldOutputs={'U': FakeField([value])}))
        history = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.2), (3.0, 0.3), (4.0, 0.4)]

        step = NS(frames=frames, historyRegions={})
        odb = NS(
            steps={'GMNIA': step},
            rootAssembly=NS(nodeSets={'STEP5_BOTTOM': object(), 'STEP5_TOP': object()}))

        original_lpf_history = fd.lpf_history
        original_weighted_u3 = fd.weighted_u3
        try:
            fd.lpf_history = lambda _step: history
            fd.weighted_u3 = lambda field, region, weights: 0.0
            info = {
                'end_area_weights': [[['P1', 1, 1.0]], [['P1', 1, 1.0]]],
                'reference_force_N_per_end': 1000.0,
                'settings': {'reference_stress': 240.0},
            }
            rows, meta = fd.curve_rows(odb, 'GMNIA', info, allow_partial=True)
            self.assertTrue(meta['partial'])
            self.assertEqual(meta['dropped_field_frames'], 1)
            self.assertEqual(len(rows), 5)
        finally:
            fd.lpf_history = original_lpf_history
            fd.weighted_u3 = original_weighted_u3

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

    def test_sparse_metadata_is_rejected_before_reading_odb(self):
        with self.assertRaisesRegex(ValueError, 'field_frequency'):
            fd.curve_rows(None,'GMNIA',{'settings':{'field_frequency':2}})

    def test_extra_nonzero_history_cannot_be_assigned_to_initial_frame(self):
        from types import SimpleNamespace as NS
        with self.assertRaisesRegex(ValueError, 'initial'):
            fd.align_lpf_history([NS(frameValue=0),NS(frameValue=2)],[(0,0),(1,1.2),(2,.7)])

    def test_ticks_are_finite_and_ordered(self):
        ticks = fd._nice_ticks(0.0, 13.0)
        self.assertTrue(ticks)
        self.assertTrue(all(math.isfinite(x) for x in ticks))
        self.assertEqual(ticks, sorted(ticks))


if __name__ == '__main__':
    unittest.main()
