import unittest
import numpy as np
import abaqus_modal_validation as v
import tempfile
import os
import json
from unittest.mock import patch


class ValidationTests(unittest.TestCase):
    def test_cluster_mean_cannot_prove_purity(self):
        # Two arbitrary 50/50 vectors span pure L and pure D directions.
        x = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
        parts = [x[:1], x[1:], np.zeros((1, 2))]
        r = v.component_bounds(parts)
        np.testing.assert_allclose(r['mean_percent'], [50, 50, 0], atol=1e-12)
        np.testing.assert_allclose(r['min_percent'], [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(r['max_percent'], [100, 100, 0], atol=1e-12)
        self.assertEqual(r['stable_family'], 'Mixed')

    def test_bounds_do_not_depend_on_eigenvector_rotation(self):
        rng = np.random.default_rng(9)
        q, _ = np.linalg.qr(rng.normal(size=(15, 3)))
        a = v.component_bounds([q[:5], q[5:10], q[10:]])
        change = np.array([[3., 1., 0.], [0., 2., 1.], [1., 0., 4.]])
        b = v.component_bounds([q[:5]@change, q[5:10]@change, q[10:]@change])
        np.testing.assert_allclose(a['min_percent'], b['min_percent'], atol=1e-10)
        np.testing.assert_allclose(a['max_percent'], b['max_percent'], atol=1e-10)

    def test_zero_observed_direction_is_not_certified(self):
        with self.assertRaises(ValueError):
            v.component_bounds([np.ones((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))])

    def test_exact_union_quadrature_for_bilinear_fields(self):
        s = np.array([0., 1.]); z = np.array([0., 2.])
        u = np.zeros((1, 2, 2, 3))
        u[0, :, :, 0] = np.array([[0., 1.], [2., 3.]])  # u=s+z
        qs, ws = v.gauss_union(s, np.array([0., .3, 1.]))
        qz, wz = v.gauss_union(z, np.array([0., .8, 2.]))
        y = v.sample_piece(u, s, z, qs, qz)
        self.assertAlmostEqual(float(np.sum(y[0, :, :, 0]**2*wz[:, None]*ws)), 16./3., places=12)

    def test_principal_angles_are_rotation_and_scale_invariant(self):
        rng = np.random.default_rng(4)
        q, _ = np.linalg.qr(rng.normal(size=(50, 3)))
        r = v.subspace_match(q, q@np.array([[2., 1., 0.], [0., 3., -1.], [1., 0., 2.]]))
        self.assertTrue(r['same_dimension'])
        self.assertAlmostEqual(r['minimum_cosine_squared'], 1., places=12)

    def test_missing_direction_is_detected(self):
        q = np.eye(5)[:, :3]
        r = v.subspace_match(q, q[:, :2])
        self.assertFalse(r['same_dimension'])
        self.assertAlmostEqual(r['coverage_a'], 2./3.)

    def test_sampling_does_not_extrapolate(self):
        with self.assertRaises(ValueError):
            v.sample_piece(np.zeros((1, 2, 2, 3)), [0., 1.], [0., 1.], [-.1], [.5])

    def test_ambiguous_section_path_is_rejected(self):
        with self.assertRaises(ValueError):
            v.ordered_path(np.array([[0, 0], [1, 0], [2, 0], [1, 1]]), [(0, 1), (1, 2), (1, 3)])

    def test_path_orientation_is_geometric_not_node_number(self):
        xy = np.array([[2., 0.], [0., 0.], [1., 0.]])
        self.assertEqual(v.ordered_path(xy, [(0, 2), (1, 2)]), [1, 2, 0])

    def test_proxy_cluster_keeps_piece_rigid_motion_out_of_distortional(self):
        import abaqus_modal_report as report
        xy = np.array([[0., 0.], [.5, 0.], [1., 0.],
                       [0., 2.], [.5, 2.], [1., 2.]])
        fit = report.SectionProjector(xy, [(0, 1), (1, 2), (3, 4), (4, 5)],
                                      ['A']*3+['B']*3, np.ones(6))
        u = np.array([[0., 1.]]*3+[[0., -1.]]*3).ravel()
        r = v.proxy_cluster(u[:, None], fit, dominance=.9, max_assembly_percent=25.)
        self.assertEqual(r['stable_family'], 'Assembly')
        self.assertGreater(r['assembly_min_percent'], 99.)

    def test_proxy_cluster_keeps_extension_out_of_distortional(self):
        import abaqus_modal_report as report
        xy = np.column_stack((np.linspace(0., 10., 9), np.zeros(9)))
        fit = report.SectionProjector(xy, [(i, i+1) for i in range(8)], ['P']*9, np.ones(9))
        u = np.column_stack((xy[:, 0]-5., np.zeros(9))).ravel()
        weighted = u*fit.sqrtw
        r = v.proxy_cluster(weighted[:, None], fit, dominance=.9,
                            max_assembly_percent=25., max_other_percent=25.)
        self.assertEqual(r['stable_family'], 'Other')
        self.assertGreater(r['other_min_percent'], 99.)

    def test_force_split_recovers_analytic_energy_families(self):
        k = np.diag([2., 3., 4., 5.])
        j = np.eye(4)[:, :2]
        p = v.ForceProjector(k, j, np.array([[1., 0.]]))
        r = p.project(np.ones(4))
        np.testing.assert_allclose(r['components']['G'].ravel(), [1, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(r['components']['D'].ravel(), [0, 1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(r['components']['L'].ravel(), [0, 0, 1, 1], atol=1e-12)
        np.testing.assert_allclose(r['energy_percent'], np.array([9, 3, 2])*100/14, atol=1e-12)
        self.assertLess(r['cross_relative'], 1e-12)

    def test_force_split_invariant_to_wall_load_units(self):
        k = np.diag([2., 3., 4.]); j = np.eye(3)[:, :2]; e = np.array([[1., 2.]])
        d = np.diag([3., 7.])
        a = v.ForceProjector(k, j, e).project(np.ones(3))
        b = v.ForceProjector(k, j@d, e@d).project(np.ones(3))
        np.testing.assert_allclose(a['energy_percent'], b['energy_percent'], atol=1e-10)

    def test_force_split_rejects_unrestrained_stiffness(self):
        with self.assertRaises(ValueError):
            v.ForceProjector(np.diag([1., 0., 1.]), np.eye(3)[:, :2], np.array([[1., 0.]]))

    def test_real_mesh_interpolation_and_rotated_cluster_matching(self):
        def archive(path, n, rotate=False):
            s = np.linspace(0., 1., n); z = np.linspace(0., 2., n+1)
            ss, zz = np.meshgrid(s, z)
            u = np.zeros((3, len(z), len(s), 3))
            u[0, :, :, 0] = 1+ss+zz
            u[1, :, :, 1] = ss-zz
            u[2, :, :, 2] = 1+ss*zz
            if rotate:
                first = u[0].copy(); second = u[1].copy()
                u[0] = 3*(first+second); u[1] = 2*(first-second)
            meta = dict(model_signature='synthetic', physics_keywords_signature='synthetic', names=['P1'],
                        axis=2, transverse=[0, 1], sigma_ref_MPa=1., mesh_nodes=n*(n+1))
            np.savez(path, metadata=json.dumps(meta), modes=[1, 2, 3], eigenvalues=[100, 100, 200],
                     z=z, p0_s=s, p0_xy=np.column_stack([s, np.zeros(n)]), p0_u=u)
        with tempfile.TemporaryDirectory() as folder:
            a, b = os.path.join(folder, 'a.npz'), os.path.join(folder, 'b.npz')
            archive(a, 2); archive(b, 4, True)
            with patch.object(v, 'write_mesh_plot'), patch.object(v, 'write_mesh_html'):
                result = v.compare_shapes(a, b, os.path.join(folder, 'report'))
            first = result['matches'][0]
            self.assertEqual(first['modes_a'], [1, 2])
            self.assertTrue(first['numerical_checks_pass'])
            self.assertAlmostEqual(first['minimum_cosine_squared'], 1., places=12)
            self.assertIn('UPPER_SPECTRUM_BOUNDARY_NOT_CLOSED', result['matches'][1]['reasons'])

    def test_changed_boundary_keywords_change_signature(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'model.inp')
            with open(path, 'w') as f: f.write('*Boundary\nEND, 1, 1\n')
            a = v.physics_keyword_signature(path)
            with open(path, 'w') as f: f.write('*Boundary\nEND, 1, 2\n')
            self.assertNotEqual(a, v.physics_keyword_signature(path))

    def test_curved_wall_reference_handles_different_chord_lengths(self):
        reference = np.array([[0., 0.], [.5, .2], [1., 0.]])
        coarse = reference[[0, 2]]
        sc, sr = v.reference_coordinates(coarse, reference)
        sf, unused = v.reference_coordinates(reference, reference)
        self.assertAlmostEqual(sc[-1], sf[-1])
        self.assertGreater(sc[-1], np.linalg.norm(coarse[1]-coarse[0]))
        with self.assertRaises(ValueError):
            v.reference_coordinates(np.array([[0., 0.], [.5, .8], [1., 0.]]), reference)

    def test_neighbor_gap_is_checked_even_for_bounded_width_clusters(self):
        import abaqus_dsm_modal_audit as audit
        groups = audit.close_clusters([dict(eigenvalue=x) for x in (100., 100.09, 100.18, 200.)], .001)
        result = v.spectral_isolation(groups, .001)
        self.assertTrue(result[0]['unresolved_neighbor'])
        self.assertTrue(result[1]['unresolved_neighbor'])
        self.assertFalse(any(r['isolated'] for r in result))

    def test_gram_whitening_is_invariant_to_modal_scaling(self):
        g = np.diag([1e-12, 1.])
        w = v.gram_whitener(g)
        np.testing.assert_allclose(w.T@g@w, np.eye(2), atol=1e-12)

    def test_separable_integration_equals_direct_gauss_for_nonuniform_grids(self):
        rng = np.random.default_rng(51)
        sa, sb = np.array([0., .3, 1.]), np.array([0., .2, .7, 1.])
        za, zb = np.array([0., .8, 2.]), np.array([0., .1, .9, 2.])
        ua, ub = rng.normal(size=(3, 3, 3, 3)), rng.normal(size=(4, 4, 4, 3))
        qs, ws = v.gauss_union(sa, sb); qz, wz = v.gauss_union(za, zb)
        w = np.sqrt(wz[:, None]*ws)[None, :, :, None]
        a = (v.sample_piece(ua, sa, za, qs, qz)*w).reshape(3, -1)
        b = (v.sample_piece(ub, sb, zb, qs, qz)*w).reshape(4, -1)
        np.testing.assert_allclose(v.surface_gram(ua, ub, sa, sb, za, zb), a@b.T, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
