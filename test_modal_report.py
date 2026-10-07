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

    def test_plate_bending_with_fixed_fold_lines_is_local_like(self):
        # End anchors do not translate.  A nonzero panel mean must no longer be
        # misread as whole-section rigid translation.
        u = np.zeros((9, 2))
        u[:, 1] = np.sin(np.linspace(0, np.pi, 9))
        shares = self.fit.shares(u[None, :, :])
        self.assertGreater(shares[2], .999999)
        self.assertLess(shares[0], 1e-10)
        self.assertLess(shares[1], 1e-10)

    def test_section_stretch_is_other_not_distortional(self):
        # Chord extension is a non-DSM Other/ST-like motion, not D.
        u = np.column_stack((self.xy[:, 0]-5., np.zeros(9)))
        d = self.fit.component_diagnostics(u[None, :, :])
        self.assertGreater(d['other_percent'], 99.999)
        self.assertLess(d['distortional_percent'], 1e-8)
        self.assertEqual(report.family_label(self.fit.shares(u[None, :, :]), self.fit.supported,
                                             other_percent=d['other_percent']), 'Other-like')

    def test_shares_are_scale_sign_invariant_and_sum_to_one(self):
        rng = np.random.RandomState(4)
        u = rng.randn(3, 9, 2)
        a, b = self.fit.shares(u), self.fit.shares(-17*u)
        np.testing.assert_allclose(a, b, atol=1e-12)
        self.assertAlmostEqual(sum(a), 1., places=12)

    def test_arbitrary_eigenvector_scale_does_not_underflow_or_overflow_shares(self):
        u = np.zeros((9, 2))
        u[:, 1] = np.sin(np.linspace(0, np.pi, 9))
        for scale in (1e-180, -1e180):
            with self.subTest(scale=scale):
                diagnostics = self.fit.component_diagnostics(scale*u)
                self.assertAlmostEqual(diagnostics['local_percent'], 100.)
                self.assertTrue(all(np.isfinite(value) for value in diagnostics.values()))


    def test_family_is_unresolved_when_geometry_proxy_unsupported(self):
        self.assertEqual(report.family_label([.1, .1, .8], False), 'Unresolved')
        self.assertEqual(report.family_label([.9, .05, .05], False), 'Global-like')

    def test_curved_mesh_without_physical_walls_is_conservative(self):
        xy = self.xy.copy()
        xy[:, 1] = .04*(xy[:, 0]-5.)**2
        fit = report.SectionProjector(xy, self.edges, ['P']*9, np.ones(9))
        self.assertEqual(fit.metadata['wall_source'], 'mesh_straight_runs')
        self.assertTrue(fit.metadata['curved_panel_proxy'] or not fit.supported)

    def test_physical_wall_geometry_overrides_radius_mesh_bias(self):
        # Two long physical walls connected by a short radius-like transition.
        xy = np.array([[0.,0.],[2.5,0.],[5.,0.],[7.5,0.],[10.,0.],
                       [10.7,.3],[11.,1.],[11.,3.],[11.,5.],[11.,7.],[11.,9.]])
        edges=[(i,i+1) for i in range(len(xy)-1)]
        segs={'P': [[0.,0.,10.,0.],
                    [10.,0.,10.7,.3],[10.7,.3,11.,1.],
                    [11.,1.,11.,9.]]}
        fit=report.SectionProjector(xy,edges,['P']*len(xy),np.ones(len(xy)),
                                    physical_segments=segs)
        self.assertEqual(fit.metadata['wall_source'],'builtup_segments.csv')
        self.assertGreaterEqual(fit.metadata['physical_wall_count'],2)
        # Local bending on the first wall with moving end folds must remain Local.
        u=np.zeros((len(xy),2))
        t=np.linspace(0.,1.,5)
        u[:5,1]=.2*t + np.sin(np.pi*t)
        d=fit.component_diagnostics(u)
        self.assertGreater(d['local_percent'],50.)
        self.assertGreater(d['wall_curvature_index'],0.)

    def test_linear_moving_chord_is_not_local(self):
        u=np.zeros((9,2))
        u[:,1]=np.linspace(-2.,3.,9)
        d=self.fit.component_diagnostics(u)
        self.assertLess(d['local_percent'],1e-8)

    def test_mixed_family_is_not_longitudinal_mixing(self):
        self.assertEqual(report.family_label([.4, .3, .3], True), 'Mixed')

    def test_piece_rigid_motion_is_assembly_not_distortional(self):
        xy = np.array([[0., 0.], [.5, 0.], [1., 0.],
                       [0., 2.], [.5, 2.], [1., 2.]])
        edges = [(0, 1), (1, 2), (3, 4), (4, 5)]
        fit = report.SectionProjector(xy, edges, ['A']*3+['B']*3, np.ones(6))
        # Opposite normal translations of the two pieces cannot be represented
        # by one whole-section rigid motion.
        u = np.array([[0., 1.]]*3+[[0., -1.]]*3)
        d = fit.component_diagnostics(u)
        self.assertGreater(d['assembly_percent'], 99.999)
        self.assertEqual(report.family_label(fit.shares(u), fit.supported,
                                             assembly_percent=d['assembly_percent']), 'Assembly-like')

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
