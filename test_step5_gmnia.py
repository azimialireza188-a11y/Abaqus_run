import ast
import contextlib
import io
import unittest
import abaqus_step5_gmnia as step5


def _fake_bending_mesh(nz=4, length=100.):
    """Four C4 copies of a lipped corner piece, extruded along Z (fake CAE objects)."""
    import math
    from types import SimpleNamespace as NS
    chain = [(20., 100.), (20., 110.), (60., 110.), (110., 110.), (110., 60.), (110., 20.), (100., 20.)]
    instances = {}
    for k in range(4):
        a = math.radians(90*k)
        ring = [(math.cos(a)*x-math.sin(a)*y, math.sin(a)*x+math.cos(a)*y) for x, y in chain]
        name = 'P%d' % (k+1)
        nodes, elements = {}, []
        n = len(ring)
        for iz in range(nz+1):
            for i, (x, y) in enumerate(ring):
                label = 1+iz*n+i
                nodes[label] = NS(label=label, coordinates=(x, y, length*iz/nz), instanceName=name, _el=[])
        for iz in range(nz):
            for i in range(n-1):
                ids = [1+iz*n+i, 1+iz*n+i+1, 1+(iz+1)*n+i+1, 1+(iz+1)*n+i]
                e = NS(label=len(elements)+1, nodes=[nodes[j] for j in ids])
                e.getNodes = (lambda e=e: e.nodes)
                elements.append(e)
                for j in ids:
                    nodes[j]._el.append(e)
        for node in nodes.values():
            node.getElements = (lambda node=node: node._el)
        seq = list(nodes.values())

        class Nodes(list):
            def sequenceFromLabels(self, labels, nodes=nodes):
                return [nodes[l] for l in labels]
        instances[name] = NS(name=name, nodes=Nodes(seq), elements=elements)
    segments = {}
    for k in range(4):
        a = math.radians(90*k)
        ring = [(math.cos(a)*x-math.sin(a)*y, math.sin(a)*x+math.cos(a)*y) for x, y in chain]
        segments['P%d' % (k+1)] = [[p[0], p[1], q[0], q[1]] for p, q in zip(ring, ring[1:])]
    return instances, segments, chain


class _FakeModel(object):
    def __init__(self, instances):
        from types import SimpleNamespace as NS
        self.loads = {}
        model = self

        class Assembly(object):
            def __init__(self):
                self.sets = {}
                self.instances = instances

            def Set(self, name, nodes=None, edges=None):
                self.sets[name] = NS(nodes=list(nodes or []), name=name)
                return self.sets[name]
        self.rootAssembly = Assembly()

    def ConcentratedForce(self, name, createStepName, region, cf3, distributionType=None, follower=None):
        from types import SimpleNamespace as NS
        self.loads[name] = NS(region=(region.name,), cf3=cf3, step=createStepName)


class Step5Tests(unittest.TestCase):
    def test_compression_load_preserves_the_reference_force_in_nlgeom(self):
        """The LPF force formula requires dead loading per original edge length."""
        import json
        from types import SimpleNamespace as NS
        from unittest import mock
        instances, unused, unused2 = _fake_bending_mesh()
        model = _FakeModel(instances)
        assembly = model.rootAssembly
        assembly.surfaces = {}
        for name, inst in instances.items():
            inst.partName = name
            for element in inst.elements:
                element.type = 'S4R'
            for suffix in ('0', 'L'):
                end = 'END_%s_%s' % (suffix, name)
                assembly.sets[end] = NS(edges=[NS(getSize=lambda **kw: 200.)])
                assembly.surfaces[end+'_S'] = NS(name=end+'_S')
                model.loads['COMP_'+end] = NS(region=(end+'_S',))
        model.parts = {name: NS(sectionAssignments=[NS(sectionName='shell')]) for name in instances}
        model.sections = {'shell': NS(thickness=2., numIntPts=5, material='steel')}
        model.materials = {'steel': NS(elastic=NS(table=((200000., .3),)), Plastic=mock.Mock())}
        model.boundaryConditions = {str(i): object() for i in range(9)}
        model.constraints = {'BOLT_1': object()}
        model.interactions = {'GeneralContact': object()}
        model.interactionProperties = {'Hard_Frictionless': object()}
        model.steps = {'Initial': object(), 'Buckle': object()}
        model.fieldOutputRequests = {}; model.historyOutputRequests = {}
        reference = dict(instances=list(instances), element_type='S4R', boundary_conditions=9,
                         loads=8, rigid_links=1, expected_links=1, connection_model='BEAM_MPC')
        counts = {key: len(getattr(model, key)) for key in
                  ('boundaryConditions', 'constraints', 'interactions', 'loads', 'materials', 'sections')}
        model.description = step5.STEP4_PREFIX+json.dumps(dict(current_case={}, specification=dict(
            stage=4, reference_pipeline=reference, source_repository_counts=counts)))
        assembly.SetFromNodeLabels = lambda name, **kw: assembly.sets.update({name: object()})
        model.StaticRiksStep = mock.Mock()
        model.ShellEdgeLoad = mock.Mock(side_effect=StopIteration('load boundary reached'))
        args = step5.parse_arguments(['--source-cae', 'source.cae', '--output-dir', 'out', '--fy', '240'])
        constants = NS(ON=True, OFF=False, UNIFORM='UNIFORM', GENERAL='GENERAL')
        with mock.patch.dict('sys.modules', {'abaqusConstants': constants}), \
                mock.patch.object(step5, 'replace_bolts', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(StopIteration, 'load boundary'):
                step5.prepare_model(model, args)
        applied = model.ShellEdgeLoad.call_args.kwargs
        self.assertTrue(applied['resultant'])
        self.assertFalse(applied['follower'])
        self.assertEqual(applied['magnitude'], 480.)

    def test_nogui_without_file_resolves_queue_path_before_chdir(self):
        import os
        import tempfile
        with open(step5.__file__, encoding='utf-8') as stream:
            source = stream.read()
        namespace = {'__name__': 'nogui_test'}
        exec(compile(source, step5.__file__, 'exec'), namespace)
        self.assertEqual(namespace['SCRIPT_DIR'], os.path.dirname(os.path.abspath(step5.__file__)))
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                namespace['write_queue_batch']('queue.bat', ['STEP5_D_FY240'])
                with open('queue.bat') as f:
                    text = f.read()
                self.assertIn(os.path.join(namespace['SCRIPT_DIR'], 'abaqus_step5_queue_nogui.py'), text)
            finally:
                os.chdir(previous)

    def test_fy_is_required_and_arc_limits_are_consistent(self):
        base = ['--source-cae', 'source.cae', '--output-dir', 'out']
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step5.parse_arguments(base)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step5.parse_arguments(base+['--fy', '350', '--initial-arc', '.1', '--max-arc', '.01'])
        args = step5.parse_arguments(base+['--fy', '350'])
        self.assertEqual(args.reference_stress, 350.)

    def test_default_postpeak_stop_ratio_is_70_percent(self):
        base = ['--source-cae', 'source.cae', '--output-dir', 'out', '--fy', '240']
        args = step5.parse_arguments(base)
        self.assertAlmostEqual(args.postpeak_stop_ratio, 0.70)
        self.assertEqual(args.max_increments, 5000)
        self.assertIsNone(args.max_end_displacement_mm)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step5.parse_arguments(base+['--postpeak-stop-ratio', '1.0'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            step5.parse_arguments(base+['--postpeak-stop-ratio', '0'])

    def test_urdfil_tracks_peak_and_stops_at_ratio(self):
        source = step5.render_postpeak_urdfil(0.70)
        self.assertIn('SUBROUTINE URDFIL', source)
        self.assertIn('CALL POSFIL(KSTEP,KINC,ARRAY,JRCD)', source)
        self.assertIn('JRRAY(1,2).EQ.2000', source)
        self.assertIn('CURRLPF=ARRAY(11)', source)
        self.assertIn('PEAKLPF=CURRLPF', source)
        self.assertIn('CURRLPF.LE.RATIO*PEAKLPF', source)
        self.assertIn('LSTOP=1', source)
        self.assertIn('RATIO=0.7D0', source)

    def test_perfect_reference_is_a_default_step5_case(self):
        with open(step5.__file__, encoding='utf-8') as stream:
            source = stream.read()
        self.assertIn("PERFECT_TOKEN = 'PERFECT'", source)
        self.assertIn("name = 'STEP5_PERFECT_FY'+tag", source)
        self.assertIn('perfect_reference=True', source)
        self.assertIn('[PERFECT_TOKEN] + imperfect', source)

    def test_inp_trigger_is_injected_before_end_step(self):
        import os
        import tempfile
        handle, path = tempfile.mkstemp(suffix='.inp')
        os.close(handle)
        try:
            with open(path, 'w') as stream:
                stream.write('*HEADING\n*STEP\n*STATIC, RIKS\n0.1, 1.0\n*END STEP\n')
            step5.inject_urdfil_trigger(path)
            with open(path) as stream:
                text = stream.read()
            self.assertIn('*NODE FILE, NSET=STEP5_STOP, FREQUENCY=1', text)
            self.assertLess(text.index('*NODE FILE'), text.index('*END STEP'))
        finally:
            os.remove(path)

    def test_step5_requires_pipeline_compatible_step4_provenance(self):
        with open(step5.__file__, encoding='utf-8') as stream:
            source = stream.read()
        self.assertIn("STEP4_PREFIX = 'STEP4_PIPELINE_COMPATIBLE '", source)
        self.assertIn('pipeline_contract.validate_step4_payload', source)
        self.assertIn("reference['expected_links']", source)
        self.assertIn("source='BEAM_MPC from abaqus_complete_model_m20.py'", source)
        self.assertIn("added_compliance='none intentionally introduced'", source)

    def test_step5_job_inherits_reference_nodal_precision_contract(self):
        with open(step5.__file__, encoding='utf-8') as stream:
            source = stream.read()
        self.assertIn("nodal_precision = str(info['reference_pipeline'].get('nodal_precision'", source)
        self.assertIn('memory=100, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=False', source)
        self.assertIn('nodalOutputPrecision=FULL if nodal_precision', source)

    def test_end_weights_and_reference_force_use_one_end(self):
        # A shell 10 mm wide, 100 mm long, 2 mm thick.
        xyz = {1: (0., 0., 0.), 2: (10., 0., 0.),
               3: (10., 0., 100.), 4: (0., 0., 100.)}
        weights = step5.end_weights(xyz, [(1, 2, 3, 4)], 2., 0., 100.)
        self.assertEqual(weights[0], {1: 10., 2: 10.})
        self.assertEqual(weights[1], {3: 10., 4: 10.})
        self.assertEqual(sum(weights[0].values())*350., 7000.)

    def test_no_solver_submission_path_exists(self):
        with open(step5.__file__, encoding='utf-8') as stream:
            tree = ast.parse(stream.read())
        forbidden = {'submit', 'waitForCompletion', 'system', 'Popen'}
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(called & forbidden)


    def test_default_material_is_elastic_perfectly_plastic(self):
        self.assertEqual(step5.plastic_table(None, 200000., 240.), ((240., 0.),))

    def test_engineering_curve_is_converted_to_true_plastic_table(self):
        import math, os, tempfile
        rows = [(240./200000., 240.), (0.02, 300.), (0.10, 360.)]
        table = step5.plastic_table(rows, 200000., 240.)
        self.assertEqual(table[0][1], 0.)
        self.assertAlmostEqual(table[1][0], 300.*1.02)
        self.assertAlmostEqual(table[1][1], math.log(1.02)-306./200000.)
        self.assertTrue(all(b[1] > a[1] and b[0] >= a[0] for a, b in zip(table, table[1:])))
        with self.assertRaises(ValueError):
            step5.plastic_table([(0.0012, 250.), (0.02, 300.)], 200000., 240.)   # first stress != fy
        with self.assertRaises(ValueError):
            step5.plastic_table([(0.0012, 240.), (0.02, 200.)], 200000., 240.)   # softening
        handle, path = tempfile.mkstemp(suffix='.csv'); os.close(handle)
        try:
            with open(path, 'w') as f:
                f.write('strain,stress_MPa\n0.0012,240\n0.02,300\n')
            self.assertEqual(step5.read_stress_strain(path), [(0.0012, 240.), (0.02, 300.)])
            args = step5.parse_arguments(['--source-cae', 's.cae', '--output-dir', 'o', '--fy', '240',
                                          '--stress-strain', path])
            self.assertEqual(len(args.stress_strain_rows), 2)
        finally:
            os.remove(path)

    def test_bending_loads_are_consistent_across_step3_and_step5(self):
        """Step-3 consistent nodal forces of the linear stress and the Step-5 recomputation agree;
        the work-conjugate rotation weights recover a rigid end rotation exactly."""
        import sys, types, math
        import abaqus_complete_model_m20 as builder
        import abaqus_load_case as lc
        fake_constants = types.ModuleType('abaqusConstants')
        fake_constants.UNIFORM, fake_constants.OFF = 'UNIFORM', False
        instances, segments, chain = _fake_bending_mesh()
        for theta in (0., 45., 22.5):
            definition = lc.bending_reference(segments, 2.0, theta)
            model = _FakeModel(instances)
            asm = model.rootAssembly
            ends = []
            for k, (name, inst) in enumerate(sorted(instances.items()), 1):
                for suffix, z, sign in (('0', 0., 1.), ('L', 100., -1.)):
                    set_name = 'END_%s_P%d' % (suffix, k)
                    asm.Set(name=set_name, nodes=[n for n in inst.nodes if abs(n.coordinates[2]-z) < 1e-9])
                    ends.append((k, inst, suffix, sign, set_name))
            old = sys.modules.get('abaqusConstants')
            sys.modules['abaqusConstants'] = fake_constants
            try:
                result = builder.apply_bending_end_forces(model, asm, ends, 2.0, definition)
            finally:
                if old is None:
                    del sys.modules['abaqusConstants']
                else:
                    sys.modules['abaqusConstants'] = old
            self.assertEqual(result['n_loads'], len(model.loads))
            # the chord mesh is the exact centre-line geometry here: applied moment = I/c
            self.assertAlmostEqual(result['mesh_to_centreline_moment_ratio'], 1.0, places=12)
            self.assertLess(result['net_axial_force_relative'], 1e-12)
            self.assertLess(result['secondary_moment_relative'], 1e-9)
            definition.update(result)
            snap = dict((k, definition[k]) for k in definition if k in
                        ('type', 'compression_normal', 'centroid_mm', 'c_extreme_mm', 'neutral_axis_angle_deg',
                         'reference_moment_mesh_Nmm_per_MPa', 'plastic_modulus_mm3'))
            pieces = [(name, dict((n.label, n.coordinates) for n in inst.nodes),
                       [tuple(n.label for n in e.getNodes()) for e in inst.elements])
                      for name, inst in sorted(instances.items())]
            loads, expected, info = step5.bending_end_loads(model, pieces, 2.0, 0., 100.,
                                                            snap, 240., len(model.loads), 240.)
            self.assertEqual(expected, set(model.loads))
            for name, node_set, force in loads:
                self.assertAlmostEqual(force, 240.*model.loads[name].cf3, places=9)
            self.assertAlmostEqual(info['reference_moment_Nmm'], 240.*definition['I_nn_mm4']/definition['c_extreme_mm'])
            self.assertAlmostEqual(info['first_yield_moment_Nmm'], info['reference_moment_Nmm'])
            # rigid rotations beta_b (bottom) and beta_t (top) of plane end sections + axial shift
            nx, ny = definition['compression_normal']
            coords = dict(((n.instanceName, n.label), n.coordinates) for inst in instances.values() for n in inst.nodes)
            for side, beta, shift in ((0, 2e-3, 0.7), (1, -3e-3, -0.2)):
                rot = sum(k*(shift+beta*(nx*coords[(i, l)][0]+ny*coords[(i, l)][1]))
                          for i, l, k in info['end_rotation_weights'][side])
                self.assertAlmostEqual(rot, beta if side == 0 else -beta, places=12)

    def test_bending_end_loads_reject_moved_end_nodes(self):
        import abaqus_load_case as lc
        instances, segments, chain = _fake_bending_mesh()
        definition = lc.bending_reference(segments, 2.0, 0.)
        pieces = [(name, dict((n.label, n.coordinates) for n in inst.nodes),
                   [tuple(n.label for n in e.getNodes()) for e in inst.elements])
                  for name, inst in sorted(instances.items())]
        definition['reference_moment_mesh_Nmm_per_MPa'] = 1.01*definition['reference_moment_Nmm_per_MPa']
        with self.assertRaises(ValueError):
            step5.bending_end_loads(_FakeModel(instances), pieces, 2.0, 0., 100.,
                                    definition, 240., 1, 240.)


if __name__ == '__main__':
    unittest.main()
