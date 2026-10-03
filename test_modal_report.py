import unittest
import numpy as np
import abaqus_modal_report as report


class ModalReportTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.column_stack((np.linspace(0, 10, 9), np.zeros(9)))
        self.edges = [(i, i+1) for i in range(8)]
        self.fit = report.SectionProjector(self.xy, self.edges, ['P']*9, np.ones(9))

    def test_rigid_translation_and_rotation_are_global(self):
        u = np.column_stack((np.full(9, 3.), 2.+.3*self.xy[:, 0]))
        shares = self.fit.shares(u[None, :, :])
        self.assertAlmostEqual(shares[0], 1., places=10)

    def test_plate_bending_residual_is_local_like(self):
        u = np.zeros((9, 2))
        u[:, 1] = np.sin(np.linspace(0, np.pi, 9))
        y = u.ravel()
        y -= self.fit.qglobal.dot(self.fit.qglobal.T.dot(y))
        self.assertAlmostEqual(self.fit.shares(y.reshape(1, 9, 2))[2], 1., places=10)

    def test_section_stretch_is_only_a_distortional_proxy(self):
        # This deliberate limitation must remain documented: no cFSM strain constraints.
        u = np.column_stack((self.xy[:, 0]-5., np.zeros(9)))
        self.assertAlmostEqual(self.fit.shares(u[None, :, :])[1], 1., places=10)

    def test_shares_are_scale_sign_invariant_and_sum_to_one(self):
        rng = np.random.RandomState(4)
        u = rng.randn(3, 9, 2)
        a, b = self.fit.shares(u), self.fit.shares(-17*u)
        np.testing.assert_allclose(a, b, atol=1e-12)
        self.assertAlmostEqual(sum(a), 1., places=12)

    def test_family_is_unresolved_when_geometry_proxy_unsupported(self):
        self.assertEqual(report.family_label([.1, .1, .8], False), 'Unresolved')
        self.assertEqual(report.family_label([.9, .05, .05], False), 'Global-like')

    def test_curved_panel_gets_explicit_proxy_warning(self):
        xy = self.xy.copy()
        xy[:, 1] = .04*(xy[:, 0]-5.)**2
        fit = report.SectionProjector(xy, self.edges, ['P']*9, np.ones(9))
        self.assertTrue(fit.supported)
        self.assertTrue(fit.metadata['curved_panel_proxy'])
        np.testing.assert_allclose(fit.qglobal.T.dot(fit.qdist), 0., atol=1e-10)

    def test_mixed_family_is_not_longitudinal_mixing(self):
        self.assertEqual(report.family_label([.4, .3, .3], True), 'Mixed')

    def test_envelope_uses_positive_dominant_samples_only(self):
        rows = [dict(mode=1, dominant_halfwaves=2, eigenvalue=10., quality_ok=True),
                dict(mode=2, dominant_halfwaves=2, eigenvalue=8., quality_ok=False),
                dict(mode=3, dominant_halfwaves=4, eigenvalue=-1., quality_ok=True)]
        result = report.sample_envelope(rows, 100., 'eigenvalue')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['minimum_mode'], 2)
        self.assertEqual(result[0]['filtered_minimum_mode'], 1)
        self.assertEqual(result[0]['half_wavelength_mm'], 50.)

    def test_envelope_does_not_bridge_missing_harmonics(self):
        rows = [dict(halfwaves=n, half_wavelength_mm=100./n, minimum_value=float(n))
                for n in (1, 2, 5)]
        x, y = report.envelope_line(rows, 'minimum_value')
        self.assertTrue(np.isnan(y[2]))


if __name__ == '__main__':
    unittest.main()
