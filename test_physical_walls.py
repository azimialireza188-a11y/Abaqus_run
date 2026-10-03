import unittest
import numpy as np
import abaqus_modal_report as report


def wavy_wall(n=81):
    x = np.linspace(0., 180., n)
    return np.column_stack((x, 6.*np.sin(2.*np.pi*x/180.)))


def projector(xy):
    segments = np.column_stack((xy[:-1], xy[1:])).tolist()
    return report.SectionProjector(xy, [(i, i+1) for i in range(len(xy)-1)],
        ['P']*len(xy), np.ones(len(xy)), physical_segments={'P': segments})


class PhysicalWallsTests(unittest.TestCase):
    def test_smooth_waviness_is_one_complete_wall(self):
        p = projector(wavy_wall())
        self.assertEqual(p.metadata['physical_wall_count'], 1)
        self.assertAlmostEqual(p.metadata['physical_wall_coverage_fraction'], 1.)

    def test_compact_bend_separates_two_walls_at_all_scales(self):
        a = np.linspace(-np.pi/2., 0., 25)
        xy = np.vstack([np.column_stack((np.linspace(0., 100., 31), np.zeros(31))),
            np.column_stack((100.+5.*np.cos(a[1:]), 5.+5.*np.sin(a[1:]))),
            np.column_stack((np.full(30, 105.), np.linspace(5., 105., 31)[1:]))])
        for scale in (.001, 1., 1000.):
            p = projector(xy*scale)
            self.assertEqual(p.metadata['physical_wall_count'], 2)
            self.assertEqual(len(p.metadata['bend_zones']), 1)
            self.assertGreater(p.metadata['physical_wall_coverage_fraction'], .9)

    def test_sharp_fold_is_not_merged_with_walls(self):
        xy = np.vstack([np.column_stack((np.linspace(0., 100., 31), np.zeros(31))),
            np.column_stack((np.full(30, 100.), np.linspace(0., 100., 31)[1:]))])
        self.assertEqual(projector(xy).metadata['physical_wall_count'], 2)

    def test_curved_wall_rigid_motion_has_zero_local_and_curvature(self):
        xy = wavy_wall()
        p = projector(xy)
        u = np.array([2., -3.]) + .17*np.column_stack((-xy[:, 1], xy[:, 0]))
        np.testing.assert_allclose(p.plocal@u.ravel(), 0., atol=1e-11)
        self.assertLess(p.component_diagnostics(u)['wall_curvature_index'], 1e-15)

    def test_fixed_end_normal_bending_of_curved_wall_is_local(self):
        xy = wavy_wall()
        tangent = np.gradient(xy, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        u = np.sin(np.linspace(0., np.pi, len(xy)))[:, None]*normal
        p = projector(xy)
        self.assertGreater(p.component_diagnostics(u)['local_percent'], 99.999999)
        np.testing.assert_allclose((p.plocal+p.pdist+p.pglobal+p.passembly+p.pother)@u.ravel(),
                                   u.ravel(), atol=1e-11)

    def test_local_bending_is_unchanged_by_superposed_rigid_motion(self):
        xy = wavy_wall(); p = projector(xy)
        u = np.column_stack((np.zeros(len(xy)), np.sin(np.linspace(0., np.pi, len(xy)))))
        rigid = .3*np.column_stack((-xy[:, 1], xy[:, 0])) + np.array([3., 7.])
        np.testing.assert_allclose(p.plocal@(u+rigid).ravel(), p.plocal@u.ravel(), atol=1e-11)

    def test_channel_flange_rotation_is_distortional_without_local_bending(self):
        left = np.column_stack((np.zeros(9), np.linspace(40., 0., 9)))
        web = np.column_stack((np.linspace(0., 100., 21)[1:], np.zeros(20)))
        right = np.column_stack((np.full(8, 100.), np.linspace(0., 40., 9)[1:]))
        xy = np.vstack((left, web, right))
        u = np.zeros_like(xy)
        u[:9, 0] = -left[:, 1]/40.
        u[-8:, 0] = right[:, 1]/40.
        diagnostics = projector(xy).component_diagnostics(u)
        self.assertGreater(diagnostics['distortional_percent'], 99.999999)
        self.assertLess(diagnostics['local_percent'], 1e-15)


if __name__ == '__main__':
    unittest.main()
