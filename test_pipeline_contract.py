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
        with self.assertRaisesRegex(ValueError, 'expected basename'):
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


if __name__ == '__main__':
    unittest.main()
