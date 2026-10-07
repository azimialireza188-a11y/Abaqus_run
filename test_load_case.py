"""Analytical section/load regressions; no Abaqus execution."""
import math
import unittest

import abaqus_load_case as loads


class LoadCaseTests(unittest.TestCase):
    def test_symmetry_requires_walls_to_match_not_just_endpoints(self):
        # Endpoints lie on two concentric squares, but the four connecting
        # walls form a chiral pinwheel. Reflection reverses its handedness.
        walls = [(1., 0., 0., 2.), (0., 1., -2., 0.),
                 (-1., 0., 0., -2.), (0., -1., 2., 0.)]
        symmetry = loads.section_symmetry(walls)
        self.assertEqual(symmetry['rotations_deg'], [0, 90, 180, 270])
        self.assertEqual(symmetry['mirror_lines_deg'], [])
        self.assertEqual(loads.required_bending_axes(symmetry)['fundamental_range_width_deg'], 90.)

    def test_section_properties_are_invariant_under_translation(self):
        square = [(-1., -1., 1., -1.), (1., -1., 1., 1.),
                  (1., 1., -1., 1.), (-1., 1., -1., -1.)]
        translated = [tuple(x + 1.e9 for x in edge) for edge in square]
        props = loads.section_properties(translated, 1.)
        self.assertAlmostEqual(props['area_mm2'], 8.)
        self.assertEqual(props['centroid_mm'], [1.e9, 1.e9])
        self.assertAlmostEqual(props['Ixx_mm4'], 16./3.)
        self.assertAlmostEqual(props['Iyy_mm4'], 16./3.)
        self.assertAlmostEqual(props['Ixy_mm4'], 0.)

    def test_invalid_section_does_not_produce_nan_reference(self):
        for section, thickness in (([(0., 0., 0., 0.)], 1.),
                                   ([(0., 0., 1., 0.)], math.inf),
                                   ([(0., 0., math.nan, 1.)], 1.)):
            with self.subTest(section=section, thickness=thickness):
                with self.assertRaises(ValueError):
                    loads.section_properties(section, thickness)

    def test_distinct_normalized_angles_have_distinct_run_names(self):
        self.assertNotEqual(loads.angle_token(45.1234561), loads.angle_token(45.1234564))
        self.assertEqual(loads.angle_token(45.9999999), '045p9999999')


if __name__ == '__main__':
    unittest.main()
