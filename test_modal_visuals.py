import unittest
import numpy as np
from types import SimpleNamespace as NS
import abaqus_modal_visuals as v
import abaqus_modal_report as report


class DirectShapeTests(unittest.TestCase):
    def test_local_coordinate_displacements_are_rejected_not_misclassified(self):
        import abaqus_modal_wavelengths as base
        value = NS(instance=NS(name='P1'), nodeLabel=1, precision='SINGLE_PRECISION',
                   localCoordSystem=np.eye(3), data=(1., 0., 0.))
        frame = NS(fieldOutputs={'U': NS(values=[value])})
        with self.assertRaisesRegex(ValueError, 'global nodal'):
            base.read_displacements(frame, {('P1', 1): 0}, 1)

    def test_double_precision_bulk_data_survives_missing_single_accessor(self):
        import abaqus_modal_wavelengths as base
        block = NS(instance=NS(name='P1'), nodeLabels=(1,), localCoordSystem=None,
                   dataDouble=((1.000000000123, 2., 3.),))
        frame = NS(fieldOutputs={'U': NS(bulkDataBlocks=[block], values=[])})
        result = base.read_displacements(frame, {('P1', 1): 0}, 1, {'P1': np.array([-1, 0])})
        self.assertEqual(result[0, 0], 1.000000000123)

    def test_irregular_station_weights_and_permuted_tracks(self):
        tracks = [dict(z=np.array([0., 1., 4.]), indices=[0, 1, 2]),
                  dict(z=np.array([4., 0., 1.]), indices=[5, 3, 4])]
        grid = v.common_grid(tracks, 1e-6)
        np.testing.assert_allclose(grid['weights'], [.125, .5, .375])
        np.testing.assert_array_equal(grid['indices'], [[0, 3], [1, 4], [2, 5]])
        tracks[1]['z'][2] = 2.
        self.assertIsNone(v.common_grid(tracks, 1e-6))

    def test_raw_shape_keeps_high_harmonic_that_a_sine_fit_would_drop(self):
        xy = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        p = report.SectionProjector(xy, [(0, 1), (1, 2), (2, 3)], ['A']*4, np.ones(4))
        z = np.linspace(0, 1, 101)
        u = np.zeros((101, 4, 2)); u[:, :, 0] = np.sin(70*np.pi*z)[:, None]
        weights = np.ones(101)/100; weights[[0, -1]] *= .5
        vector = v.weighted_raw(u, weights, p.sqrtw)
        shares = v.section_percentages(vector, p)
        np.testing.assert_allclose(shares, [0, 0, 100], atol=1e-10)

    def test_independent_piece_motion_is_separate_from_whole_section_motion(self):
        xy = np.array([[0., 0.], [1., 0.], [0., 2.], [1., 2.]])
        p = report.SectionProjector(xy, [(0, 1), (2, 3)], ['A', 'A', 'B', 'B'], np.ones(4))
        q = v.relative_piece_basis(p, ['A', 'A', 'B', 'B'])
        u = np.array([[0., 1.], [0., 1.], [0., -1.], [0., -1.]])
        shares = v.rigid_shares(u.ravel(), p, q)
        total = (shares['whole_section_rigid_percent']+shares['relative_piece_rigid_percent']+
                 shares['within_piece_deformation_percent'])
        self.assertAlmostEqual(total, 100.)
        self.assertGreater(shares['relative_piece_rigid_percent'], 99.)
        self.assertAlmostEqual(shares['within_piece_deformation_percent'], 0., places=10)

    def test_sensitivity_is_percentage_points_not_probability(self):
        result = v.sensitivity([[10, 80, 10], [25, 65, 10], [15, 75, 10]], .9)
        self.assertEqual(result['max_range_pp'], 15.)
        self.assertEqual(result['labels'], ['Mixed']*3)

    def test_preview_uses_actual_peak_section_not_a_harmonic_reconstruction(self):
        tracks = [dict(z=np.array([0., 1., 2.]), indices=[0, 1, 2]),
                  dict(z=np.array([0., 1., 2.]), indices=[3, 4, 5])]
        u = np.zeros((6, 3)); u[1, 0] = 2.; u[4, 1] = -1.
        payload = v.mode_preview(u, tracks, [0, 1], 0., 2.)
        self.assertEqual(payload['peak_z_mm'], 1.)
        np.testing.assert_allclose(payload['sections'][payload['peak_index']], [[2, 0], [0, -1]])


if __name__ == '__main__':
    unittest.main()
