import unittest
import numpy as np
import abaqus_dsm_modal_audit as audit


class ModalAuditTests(unittest.TestCase):
    def test_known_mixture_and_mode_scale_invariance(self):
        basis = dict(L=np.eye(3)[:, :1], D=np.eye(3)[:, 1:2], G=np.eye(3)[:, 2:])
        projector = audit.MechanicalProjector(basis, np.ones(3))
        u = np.sqrt([.6, .3, .1])
        a = projector.project(u)
        b = projector.project(-7*u)
        np.testing.assert_allclose(a['percentages'], [60., 30., 10.], atol=1e-10)
        np.testing.assert_allclose(b['percentages'], a['percentages'], atol=1e-10)
        self.assertLess(a['relative_residual'], 1e-12)

    def test_shared_family_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.MechanicalProjector(dict(L=np.eye(3)[:, :1], D=np.eye(3)[:, :1],
                                          G=np.eye(3)[:, 2:]), np.ones(3))

    def test_missing_component_remains_residual(self):
        q = np.eye(4)
        p = audit.MechanicalProjector(dict(L=q[:, :1], D=q[:, 1:2], G=q[:, 2:3]), np.ones(4))
        result = p.project(np.ones(4))
        self.assertAlmostEqual(result['relative_residual'], .5)
        self.assertEqual(audit.classify(result, .9, .05), 'Unresolved')

    def test_degenerate_subspace_is_rotation_invariant(self):
        p = audit.MechanicalProjector(dict(L=np.eye(3)[:, :1], D=np.eye(3)[:, 1:2],
                                          G=np.eye(3)[:, 2:]), np.ones(3))
        modes = np.eye(3)[:, :2]
        rotation = np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
        a = p.project(modes, subspace=True)
        b = p.project((modes @ rotation)*[3., 5.], subspace=True)
        np.testing.assert_allclose(a['percentages'], b['percentages'], atol=1e-10)
        np.testing.assert_allclose(a['percentages'], [50., 50., 0.], atol=1e-10)

    def test_numerical_closeness_is_not_a_review(self):
        candidate = dict(mode=1, stress_MPa=100., family='L')
        result = audit.assess_family(candidate, None, None, None, 'abc', True, .05)
        self.assertIsNone(result['accepted_stress_MPa'])
        self.assertIn('mesh_comparison_missing', result['reasons'])

    def test_percentage_does_not_scale_eigenvalue(self):
        row = dict(mode=2, eigenvalue=300., stress_MPa=300., family='Mixed',
                   mechanical_eligible=True, percentages=[60., 40., 0.])
        self.assertIsNone(audit.candidate_for('L', [row]))

    def test_energy_cross_terms_close_and_are_not_hidden(self):
        k = np.array([[2., 1., 0.], [1., 2., 0.], [0., 0., 1.]])
        parts = dict(L=np.array([1., 0., 0.]), D=np.array([0., 1., 0.]),
                     G=np.zeros(3), R=np.zeros(3))
        result = audit.energy_partition(parts, k)
        self.assertAlmostEqual(result['diagonal_percent']['L'], 100./3.)
        self.assertAlmostEqual(result['cross_percent']['L:D'], 100./3.)
        self.assertAlmostEqual(sum(result['diagonal_percent'].values())+
                               sum(result['cross_percent'].values()), 100.)

    def test_indefinite_energy_is_rejected_on_observed_subspace(self):
        with self.assertRaises(ValueError):
            audit.energy_partition(dict(L=np.array([1., 0.]), D=np.array([0., 1.]),
                                        G=np.zeros(2), R=np.zeros(2)), np.diag([2., -1.]))

    def test_mapped_basis_reads_rotations_and_compatible_elastic_energy(self):
        import json, os, tempfile
        from types import SimpleNamespace as NS
        from scipy.sparse import csr_matrix
        odb = NS(rootAssembly=NS(instances={'P1': NS(nodes=[NS(label=1, coordinates=(0., 0., 0.))])}))
        metadata = dict(source_odb_sha256='odb_hash', model_signature='model_hash', method='synthetic test only',
            family_definition_id='known_axes', validation_notes='analytic orthogonal unit vectors',
            metric_definition='unit test weights', coordinate_space_review='three explicitly selected test DOFs',
            stiffness_kind='elastic', stiffness_source='analytic diagonal test stiffness')
        k = csr_matrix(np.diag([2., 3., 4.]))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'basis.npz')
            np.savez(path, metadata=json.dumps(metadata), instances=np.array(['P1']*3), labels=[1]*3,
                dofs=[1, 2, 6], coordinates=np.zeros((3, 3)), weights=np.ones(3),
                L=np.eye(3)[:, :1], D=np.eye(3)[:, 1:2], G=np.eye(3)[:, 2:],
                K_data=k.data, K_indices=k.indices, K_indptr=k.indptr)
            projector, keys, stiffness, unused = audit.load_basis(path, odb, 'odb_hash', 'model_hash')
            def field(data):
                return NS(values=[NS(instance=NS(name='P1'), nodeLabel=1, precision='SINGLE_PRECISION',
                                     localCoordSystem=None, data=data)])
            frame = NS(fieldOutputs={'U': field((1., 2., 3.)), 'UR': field((0., 0., 4.))})
            u = audit.read_mapped_mode(frame, keys)
            np.testing.assert_allclose(u, [1., 2., 4.])
            result = projector.project(u)
            energy = audit.energy_partition(dict(result['components'], R=result['residual']), stiffness)
            self.assertAlmostEqual(energy['total_normalized_mode_energy'], 39.)
            self.assertAlmostEqual(energy['diagonal_percent']['G'], 100.*32./39.)

    def test_force_based_pack_is_loaded_with_mapped_elastic_energy(self):
        import json, os, tempfile
        from types import SimpleNamespace as NS
        from scipy.sparse import csr_matrix
        odb = NS(rootAssembly=NS(instances={'P1': NS(nodes=[NS(label=1, coordinates=(0., 0., 0.))])}))
        metadata = dict(source_odb_sha256='odb', model_signature='model', method='analytic test',
            family_definition_id='test', validation_notes='synthetic only', metric_definition='unit',
            coordinate_space_review='known restrained test DOFs', stiffness_kind='elastic', stiffness_source='analytic',
            format='force_based_KJE', wall_definition='test axes', equilibrium_definition='first wall',
            constraint_mapping_review='identity in test', contact_state_review='none in test')
        k = csr_matrix(np.diag([2., 3., 4.]))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'force.npz')
            np.savez(path, metadata=json.dumps(metadata), instances=['P1']*3, labels=[1]*3, dofs=[1, 2, 6],
                coordinates=np.zeros((3, 3)), weights=np.ones(3), J=np.eye(3)[:, :2], E=[[1., 0.]],
                K_data=k.data, K_indices=k.indices, K_indptr=k.indptr)
            projector, keys, stiffness, unused = audit.load_basis(path, odb, 'odb', 'model')
            r = projector.project([1., 2., 4.])
            energy = audit.energy_partition(dict(r['components'], R=r['residual']), stiffness)
            np.testing.assert_allclose([energy['diagonal_percent'][f] for f in 'LDG'], [3200/39, 600/39, 100/39])


if __name__ == '__main__':
    unittest.main()
