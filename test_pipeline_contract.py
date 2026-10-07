import json
import os
import tempfile
import unittest

import abaqus_pipeline_contract as contract


def valid_state(run_dir):
    source = dict(
        section_segments={name: [[0., 0., 1., 0.]] for name in ('P1', 'P2', 'P3', 'P4')},
        connection_model='BEAM_MPC',
        shell_contact='GENERAL_STANDARD_HARD_FRICTIONLESS',
        expected_links=76,
        bolts_per_seam=19,
        thickness_mm=2.0,
        source_directory=os.path.join(run_dir, 'source'))
    build = dict(
        job_name='BU_BOLT_L3600_M20',
        odb=os.path.join(run_dir, 'BU_BOLT_L3600_M20.odb'),
        element_type='S4R',
        rigid_links=76,
        boundary_conditions=9,
        loads=8,
        reference_stress_MPa=1.0,
        length_mm=3600.0,
        target_mesh_mm=20.0,
        nodes=1000,
        elements=900,
        longitudinal_lines=6,
        longitudinal_line_min_spacing_mm=5.0)
    return dict(
        status='COMPLETED',
        settings=dict(nodal_precision='full', buckle_output='detailed'),
        source_inputs=source,
        build=build)


class PipelineContractTests(unittest.TestCase):
    def make_run(self):
        root = tempfile.mkdtemp()
        state = valid_state(root)
        with open(os.path.join(root, 'pipeline_status.json'), 'w') as stream:
            json.dump(state, stream)
        for suffix in ('.cae', '.odb', '.inp'):
            with open(os.path.join(root, state['build']['job_name']+suffix), 'wb') as stream:
                stream.write((suffix+' test').encode('ascii'))
        return root, state

    def test_load_reference_run_uses_recorded_job_name_not_glob_order(self):
        root, state = self.make_run()
        with open(os.path.join(root, 'AAA_wrong.cae'), 'wb') as stream:
            stream.write(b'wrong')
        loaded = contract.load_reference_run(root)
        self.assertEqual(loaded['job_name'], state['build']['job_name'])
        self.assertEqual(os.path.basename(loaded['paths']['cae']),
                         state['build']['job_name']+'.cae')
        self.assertEqual(loaded['instances'], ['P1', 'P2', 'P3', 'P4'])

    def test_reference_contract_rejects_changed_connection_model(self):
        root, state = self.make_run()
        state['source_inputs']['connection_model'] = 'CONNECTOR'
        with self.assertRaisesRegex(ValueError, 'BEAM_MPC'):
            contract.validate_reference_state(state)

    def test_reference_contract_rejects_inconsistent_link_count(self):
        root, state = self.make_run()
        state['build']['rigid_links'] = 75
        with self.assertRaisesRegex(ValueError, 'rigid-link'):
            contract.validate_reference_state(state)

    def test_reference_contract_rejects_noncompleted_run(self):
        root, state = self.make_run()
        state['status'] = 'FAILED'
        with open(os.path.join(root, 'pipeline_status.json'), 'w') as stream:
            json.dump(state, stream)
        with self.assertRaisesRegex(ValueError, 'not COMPLETED'):
            contract.load_reference_run(root)

    def test_explicit_artifact_must_belong_to_same_reference_job(self):
        root, state = self.make_run()
        loaded = contract.load_reference_run(root)
        wrong = os.path.join(root, 'different.odb')
        with open(wrong, 'wb') as stream:
            stream.write(b'wrong')
        with self.assertRaisesRegex(ValueError, 'exact artifact'):
            contract.resolve_reference_artifact(loaded, wrong, 'odb')

    def test_step4_payload_requires_reference_provenance(self):
        payload = dict(specification=dict(stage=4), current_case={})
        with self.assertRaisesRegex(ValueError, 'reference-pipeline'):
            contract.validate_step4_payload(payload)

    def test_reference_snapshot_carries_mesh_and_longitudinal_settings(self):
        root, state = self.make_run()
        loaded = contract.load_reference_run(root)
        snap = contract.reference_snapshot(loaded, include_hashes=False)
        self.assertEqual(snap['target_mesh_mm'], 20.0)
        self.assertEqual(snap['longitudinal_lines'], 6)
        self.assertEqual(snap['longitudinal_line_min_spacing_mm'], 5.0)
        self.assertEqual(snap['nodal_precision'], 'full')
        self.assertEqual(snap['expected_links'], 76)

    def bending_state(self, root):
        import abaqus_load_case as lc
        state = valid_state(root)
        square = {'P1': [[-100., 100., 100., 100.]], 'P2': [[100., 100., 100., -100.]],
                  'P3': [[100., -100., -100., -100.]], 'P4': [[-100., -100., -100., 100.]]}
        definition = lc.bending_reference(square, 2.0, 45.)
        definition.update(n_loads=40, reference_moment_mesh_Nmm_per_MPa=definition['reference_moment_Nmm_per_MPa'])
        state['build'].update(loads=40, load_case=definition)
        return state

    def test_compression_is_the_default_and_keeps_eight_loads(self):
        root, state = self.make_run()
        self.assertEqual(contract.reference_load_case(state['build']), dict(type='compression'))
        snap = contract.reference_snapshot(contract.load_reference_run(root), include_hashes=False)
        self.assertEqual(snap['load_case'], dict(type='compression'))
        self.assertEqual(contract.snapshot_load_case({}), dict(type='compression'))
        state['build']['loads'] = 40
        with self.assertRaisesRegex(ValueError, 'load count'):
            contract.validate_reference_state(state)

    def test_bending_reference_contract(self):
        state = self.bending_state(tempfile.mkdtemp())
        contract.validate_reference_state(state)
        snap = contract.load_case_snapshot(state['build'])
        self.assertEqual(snap['type'], 'bending')
        self.assertEqual(snap['neutral_axis_angle_deg'], 45.)
        self.assertNotIn('section_symmetry', snap)
        state['build']['loads'] = 41
        with self.assertRaisesRegex(ValueError, 'nodal-force count'):
            contract.validate_reference_state(state)
        state = self.bending_state(tempfile.mkdtemp())
        state['build']['load_case']['type'] = 'torsion'
        with self.assertRaises(ValueError):
            contract.validate_reference_state(state)

    def test_bending_vectors_must_be_finite_and_match_recorded_angle(self):
        for key, vector in (("centroid_mm", [float("nan"), 0.]),
                            ("compression_normal", [1., 1.]),
                            ("neutral_axis_direction", [0., 1.])):
            with self.subTest(key=key):
                state = self.bending_state("unused")
                state['build']['load_case'][key] = vector
                with self.assertRaises(ValueError):
                    contract.validate_reference_state(state)

    def test_bending_convention_and_symmetry_axes(self):
        import math
        import abaqus_load_case as lc
        square = [(-100., -100., 100., -100.), (100., -100., 100., 100.),
                  (100., 100., -100., 100.), (-100., 100., -100., -100.)]
        ref0 = lc.bending_reference(square, 2.0, 0.)
        inertia = 2.0*(2*200**3/12.+2*200*100**2)
        self.assertAlmostEqual(ref0['I_nn_mm4'], inertia)
        self.assertAlmostEqual(ref0['c_extreme_mm'], 100.)
        self.assertAlmostEqual(ref0['reference_moment_Nmm_per_MPa'], inertia/100.)
        self.assertAlmostEqual(ref0['plastic_modulus_mm3'], 2.0*(2*200*100+2*2*100**2/2.))
        self.assertAlmostEqual(lc.unit_stress(ref0, 0., 100.), 1.0)      # compression at +Y for theta = 0
        self.assertAlmostEqual(lc.unit_stress(ref0, 0., -100.), -1.0)
        ref45 = lc.bending_reference(square, 2.0, 45.)
        self.assertAlmostEqual(ref45['c_extreme_mm'], 100.*math.sqrt(2.))
        self.assertAlmostEqual(lc.unit_stress(ref45, -100., 100.), 1.0)  # corner on the +n side
        symmetry = lc.section_symmetry(square)
        self.assertEqual(symmetry['rotations_deg'], [0, 90, 180, 270])
        self.assertEqual(symmetry['mirror_lines_deg'], [0., 45., 90., 135.])
        axes = lc.required_bending_axes(symmetry)
        self.assertEqual(axes['required_axes_deg'], [0., 45.])
        self.assertEqual(axes['fundamental_range_width_deg'], 45.)
        for theta, canonical in ((90., 0.), (135., 45.), (180., 0.), (200., 20.), (315., 45.)):
            self.assertAlmostEqual(lc.canonical_axis(theta, symmetry), canonical)
        # a C4-only pinwheel has no mirror line: the range to sample is 90 degrees
        pinwheel = [(0., 0., 100., 0.), (100., 0., 100., 30.)]
        pin = [tuple(v for v in seg) for seg in pinwheel]
        segs = []
        for k in range(4):
            a = math.radians(90*k)
            rot = lambda x, y: (math.cos(a)*x-math.sin(a)*y, math.sin(a)*x+math.cos(a)*y)
            segs += [rot(s[0], s[1])+rot(s[2], s[3]) for s in pin]
        self.assertEqual(lc.section_symmetry(segs)['mirror_lines_deg'], [])
        self.assertEqual(lc.required_bending_axes(lc.section_symmetry(segs))['fundamental_range_width_deg'], 90.)
        self.assertEqual([lc.load_case_tag('bending', a) for a in (0, 45, 22.5, -45)],
                         ['_BEND000', '_BEND045', '_BEND022p5', '_BEND315'])
        self.assertEqual(lc.load_case_tag('compression'), '')

    def test_consistent_edge_forces_reproduce_the_linear_stress_resultants(self):
        import abaqus_load_case as lc
        square = [(-100., -100., 100., -100.), (100., -100., 100., 100.),
                  (100., 100., -100., 100.), (-100., 100., -100., -100.)]
        ref = lc.bending_reference(square, 2.0, 30.)
        # one wall discretized in 7 linear edges: force resultant and moment are exact
        pts = dict((i, (-100.+200.*i/7, 100.)) for i in range(8))
        edges = [(i, i+1) for i in range(7)]
        q = lc.consistent_edge_forces(pts, edges, 2.0, ref)
        nx, ny = ref['compression_normal']
        eta = lambda p: nx*p[0]+ny*p[1]
        exact_force = 2.0*sum((eta(pts[a])+eta(pts[b]))/2*200./7 for a, b in edges)/ref['c_extreme_mm']
        exact_moment = 2.0*sum((eta(pts[a])**2+eta(pts[a])*eta(pts[b])+eta(pts[b])**2)/3*200./7
                               for a, b in edges)/ref['c_extreme_mm']
        self.assertAlmostEqual(sum(q.values()), exact_force)
        self.assertAlmostEqual(sum(q[i]*eta(pts[i]) for i in q), exact_moment)


if __name__ == '__main__':
    unittest.main()
