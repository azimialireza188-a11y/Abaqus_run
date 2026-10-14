import ast
import contextlib
import io
import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename))))
import abaqus_step5_gmnia as step5


class SteelPlateTests(unittest.TestCase):
    def test_plate_yield_is_required_and_independent_of_member_curve(self):
        from types import SimpleNamespace as NS
        tables = []
        material = NS(elastic=NS(table=((200000., .3),)), Plastic=lambda **kw: tables.append(kw['table']))
        model = NS(materials={'SteelPlate': material})
        reference = {'steel_plates': {'material': 'SteelPlate'}}
        with self.assertRaisesRegex(ValueError, 'steel-plate-fy'):
            step5.apply_plate_plasticity(model, reference, NS(steel_plate_fy=None))
        result = step5.apply_plate_plasticity(model, reference, NS(steel_plate_fy=355., fy=240.))
        self.assertEqual(tables, [((355., 0.),)])
        self.assertEqual(result, {'SteelPlate': [[355., 0.]]})
        self.assertEqual(step5.apply_plate_plasticity(model, {}, NS()), {})


@contextlib.contextmanager
def _incremental_builder(directory, on_prepare=None, on_queue=None, stop_method='monitor'):
    """Exercise file delivery with tiny stand-ins; no CAE import or solver."""
    from pathlib import Path
    from types import SimpleNamespace as NS
    from unittest import mock
    import tempfile
    real_mkdtemp = tempfile.mkdtemp
    output = Path(directory) / 'output'
    sources = ['STEP4_first', 'STEP4_second']
    database = NS(models={name: NS(name=name) for name in sources}, closed=False)
    def save_as(pathName):
        Path(pathName).write_bytes(b'fake final CAE')
    def close():
        database.closed = True
    database.saveAs, database.close = save_as, close
    args = NS(fy=240., output_dir=str(output), cpus=2, gpus=0, cpus_requested='auto',
              gpus_requested='auto', postpeak_stop_ratio=.7, stop_method=stop_method)
    def copy_model(db, source, name, perfect):
        model = NS(name=name)
        db.models[name] = model
        return model
    def prepare(model, unused):
        if on_prepare:
            on_prepare(model, output)
        return dict(bolts=[], reference_force_N_per_end=100., settings=dict(stop_method=stop_method))
    def write_input(db, name, info, unused, subroutine):
        Path(name+'.inp').write_text('*Heading\n'+name+'\n', encoding='ascii')
    real_queue = step5.write_queue_batch
    def queue(*a, **kw):
        if on_queue:
            on_queue(database, output)
        return real_queue(*a, **kw)
    with mock.patch.multiple(step5,
            open_step4_database=lambda unused: (database, sources, 'perfect'),
            copy_source_model=copy_model, prepare_model=prepare,
            write_job_input=write_input, write_queue_batch=queue), \
            mock.patch.object(step5.tempfile, 'mkdtemp',
                side_effect=lambda **kw: real_mkdtemp(dir=directory, **kw)), \
            contextlib.redirect_stdout(io.StringIO()):
        yield args, database, output


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

                self.referencePoints = {}

            def Set(self, name, nodes=None, edges=None, referencePoints=None):
                self.sets[name] = NS(nodes=list(nodes or []), name=name,
                                     referencePoints=list(referencePoints or []))
                return self.sets[name]

            def ReferencePoint(self, point):
                feature = NS(id=len(self.referencePoints)+1)
                self.referencePoints[feature.id] = NS(point=tuple(point))
                return feature
        self.rootAssembly = Assembly()

    def ConcentratedForce(self, name, createStepName, region, cf3, distributionType=None, follower=None):
        from types import SimpleNamespace as NS
        self.loads[name] = NS(region=(region.name,), cf3=cf3, step=createStepName, follower=follower)

    def Moment(self, name, createStepName, region, cm1, cm2, distributionType=None, follower=None):
        from types import SimpleNamespace as NS
        self.loads[name] = NS(region=(region.name,), cm1=cm1, cm2=cm2, step=createStepName, follower=follower)


class Step5Tests(unittest.TestCase):
    def test_each_verified_case_is_delivered_before_next_case_is_prepared(self):
        import json, tempfile
        for method in ('monitor', 'urdfil'):
            with self.subTest(stop_method=method), tempfile.TemporaryDirectory() as directory:
                def before_prepare(model, output):
                    self.assertTrue((output/'abaqus_v6.env').is_file())
                    self.assertTrue((output/step5.STOP_SUBROUTINE_NAME).is_file())
                    self.assertFalse((output/'run_step5_queue_70_pct.bat').exists())
                    if model.name == 'STEP5_second_FY240':
                        name = 'STEP5_first_FY240'
                        self.assertTrue((output/(name+'.inp')).is_file())
                        info = json.loads((output/(name+step5.METADATA_SUFFIX)).read_text())
                        self.assertEqual(info['input_sha256'],
                            step5.pipeline_contract.file_sha256(str(output/(name+'.inp'))))
                        if method == 'urdfil':
                            self.assertEqual(info['stop_subroutine_sha256'],
                                step5.pipeline_contract.file_sha256(str(output/step5.STOP_SUBROUTINE_NAME)))
                with _incremental_builder(directory, on_prepare=before_prepare, stop_method=method) as (args, db, output):
                    step5.build(args)
                self.assertEqual(len(list(output.iterdir())), 8)
                self.assertTrue((output/'run_step5_queue_70_pct.bat').is_file())
                self.assertTrue((output/'Step5_GMNIA_FY240.cae').is_file())

    def test_failed_later_case_preserves_delivered_cases_without_a_queue(self):
        import os, tempfile
        previous = os.getcwd()
        def fail_second(model, output):
            if model.name == 'STEP5_second_FY240':
                raise RuntimeError('second-case failure')
        with tempfile.TemporaryDirectory() as directory:
            with _incremental_builder(directory, on_prepare=fail_second) as (args, db, output):
                with self.assertRaisesRegex(RuntimeError, 'second-case failure'):
                    step5.build(args)
                self.assertTrue(db.closed)
            self.assertEqual(os.getcwd(), previous)
            self.assertTrue((output/'STEP5_first_FY240.inp').is_file())
            self.assertTrue((output/'STEP5_first_FY240_step5.json').is_file())
            self.assertFalse((output/'STEP5_second_FY240.inp').exists())
            self.assertFalse((output/'run_step5_queue_70_pct.bat').exists())
            self.assertFalse((output/'Step5_GMNIA_FY240.cae').exists())

    def test_queue_is_written_only_after_final_cae_and_all_cases_are_delivered(self):
        import tempfile
        def before_queue(db, output):
            self.assertTrue(db.closed)
            self.assertTrue((output/'Step5_GMNIA_FY240.cae').is_file())
            for case in ('first', 'second'):
                name = 'STEP5_'+case+'_FY240'
                self.assertTrue((output/(name+'.inp')).is_file())
                self.assertTrue((output/(name+step5.METADATA_SUFFIX)).is_file())
        with tempfile.TemporaryDirectory() as directory:
            with _incremental_builder(directory, on_queue=before_queue) as (args, db, output):
                step5.build(args)

    def test_compression_load_preserves_the_reference_force_in_nlgeom(self):
        """The LPF force formula requires dead loading per original edge length."""
        for pm_template,stop in ((False,None),(False,2.5),(True,None),(True,2.5)):
            import json
            from types import SimpleNamespace as NS
            from unittest import mock
            instances, unused, unused2 = _fake_bending_mesh()
            model = _FakeModel(instances)
            assembly = model.rootAssembly
            assembly.sets['STEP5_BOLTS'] = NS(name='STEP5_BOLTS')
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
            model.boundaryConditions = {'AXIAL_FIX': mock.Mock()}
            for name in instances:
                for suffix in ('0', 'L'):
                    model.boundaryConditions['SS_END_%s_%s' % (suffix, name)] = mock.Mock()
            model.constraints = {'BOLT_1': object()}
            model.interactions = {'GeneralContact': object()}
            model.interactionProperties = {'Hard_Frictionless': object()}
            model.steps = {'Initial': object(), 'Buckle': object()}
            model.fieldOutputRequests = {}; model.historyOutputRequests = {}
            reference = dict(instances=list(instances), element_type='S4R', boundary_conditions=9,
                             loads=8, rigid_links=1, expected_links=1, connection_model='BEAM_MPC',
                             shell_contact='GENERAL_STANDARD_HARD_FRICTIONLESS')
            counts = {key: len(getattr(model, key)) for key in
                      ('boundaryConditions', 'constraints', 'interactions', 'loads', 'materials', 'sections')}
            model.description = step5.STEP4_PREFIX+json.dumps(dict(current_case={}, specification=dict(
                stage=4, reference_pipeline=reference, source_repository_counts=counts)))
            assembly.SetFromNodeLabels = lambda name, **kw: assembly.sets.update({name: NS(name=name)})
            assembly.SetFromElementLabels = lambda name, **kw: assembly.sets.update({name: NS(name=name)})
            model.FieldOutputRequest = mock.Mock()
            model.HistoryOutputRequest = mock.Mock()
            model.setValues = mock.Mock()
            model.StaticRiksStep = mock.Mock()
            model.ShellEdgeLoad = mock.Mock()
            model.Coupling = mock.Mock()
            model.DisplacementBC = mock.Mock()
            args = step5.parse_arguments(['--source-cae', 'source.cae', '--output-dir', 'out', '--fy', '240'])
            args.max_end_displacement_mm = stop
            constants = NS(ON=True, OFF=False, UNIFORM='UNIFORM', GENERAL='GENERAL', KINEMATIC='KINEMATIC',
                           WHOLE_SURFACE='WHOLE_SURFACE', SET='SET', UNSET='UNSET')
            with mock.patch.dict('sys.modules', {'abaqusConstants': constants}), \
                    mock.patch.object(step5, 'replace_bolts', return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()):
                info = step5.prepare_model(model, args, pm_template=pm_template)
            # Reference procedure: keep the inherited nodal supports and free end warping.
            for name, bc in model.boundaryConditions.items():
                bc.setValues.assert_not_called()
            model.Coupling.assert_not_called()
            model.DisplacementBC.assert_not_called()
            self.assertFalse(assembly.referencePoints)
            self.assertEqual(info['end_condition']['id'], step5.pipeline_contract.STEP5_END_CONDITION_ID)
            if not pm_template:
                p = info['reference_force_N_per_end']
                self.assertAlmostEqual(p, 240.*info['end_area_mm2'])
                self.assertEqual(model.ShellEdgeLoad.call_count, 8)
                for call in model.ShellEdgeLoad.call_args_list:
                    kw = call.kwargs
                    self.assertEqual(kw['magnitude'], 480.)  # thickness 2 mm * 240 MPa
                    self.assertTrue(kw['resultant'])
                    self.assertFalse(kw['follower'])
                    sign = 1. if 'END_0_' in kw['name'] else -1.
                    self.assertEqual(kw['directionVector'], ((0., 0., 0.), (0., 0., sign)))
            else:
                model.ShellEdgeLoad.assert_not_called()
                self.assertTrue(info['pm_template']['nodes'])
                self.assertNotIn('end_plates', info['pm_template'])
                self.assertEqual(set(model.loads), {n['load'] for n in info['pm_template']['nodes']})
                self.assertTrue(all(load.cf3 == 1. and not load.follower for load in model.loads.values()))
            self.assertEqual(info['stop_positive_U3_mm'], stop)
            self.assertEqual(info['settings']['max_end_displacement_mm'], stop)
            self.assertEqual(info['postpeak_stop']['ratio'], .70)
            self.assertEqual(info['plastic_tables']['steel'], [[240.,0.]])
            options=model.StaticRiksStep.call_args.kwargs
            self.assertEqual(options['nodeOn'], stop is not None)
            self.assertEqual(options.get('maximumDisplacement'), stop)
            self.assertEqual(options['initialArcInc'], args.initial_arc)
            self.assertEqual(options['minArcInc'], args.min_arc)
            self.assertEqual(options['maxArcInc'], args.max_arc)
            self.assertEqual(options['maxNumInc'], args.max_increments)
            self.assertTrue(options['nlgeom'])
            if stop is not None:
                self.assertEqual(options['dof'],3)
                self.assertIs(options['region'],assembly.sets['STEP5_STOP'])
            model.setValues.assert_called_once()
            self.assertEqual(json.loads(model.setValues.call_args.kwargs['description'][len('STEP5_GMNIA '):])['stop_positive_U3_mm'],stop)


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

    def test_cpus_space_form_reaches_jobs_env_and_queue_batch(self):
        import os
        import tempfile
        from unittest import mock
        import runtime_resources
        base = ['--source-cae', 'source.cae', '--output-dir', 'out', '--fy', '240']
        inventory = runtime_resources.ResourceInventory(16, 8, 1, 1, [])
        with mock.patch.object(runtime_resources, 'detect_resources', return_value=inventory):
            args = step5.parse_arguments(base+['--cpus', '4', '--initial-arc', '0.01', '--min-arc', '1e-8',
                                               '--max-arc', '0.05', '--max-increments', '5000',
                                               '--field-frequency', '1', '--postpeak-stop-ratio', '0.70'])
            self.assertEqual((args.cpus, args.cpus_requested), (4, 4))
            auto = step5.parse_arguments(base)
            self.assertEqual((auto.cpus, auto.cpus_requested), (8, 'auto'))
            for bad in ('0', 'four'):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    step5.parse_arguments(base+['--cpus', bad])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'q.bat')
            step5.write_queue_batch(path, ['STEP5_PERFECT_FY240', 'STEP5_LDG_pp_FY240'], args.cpus_requested,
                                    ratio=args.postpeak_stop_ratio, gpus=args.gpus_requested)
            with open(path) as stream:
                text = stream.read()
            self.assertIn(' --cpus 4 --gpus auto --ratio 0.7 ', text)
            self.assertNotRegex(text, r'--[a-z-]+=')            # the Abaqus launcher rejects --name=value
            self.assertTrue(all(line.strip() for line in text.split('\n')[:-1]))   # no blank line inside
            with self.assertRaises(Exception):
                step5.write_queue_batch(path, ['STEP5_D_FY240'], 0)

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
        import os, tempfile
        source = step5.render_postpeak_urdfil(0.70)
        self.assertIn('SUBROUTINE URDFIL', source)
        self.assertIn('CALL POSFIL(KSTEP,KINC,ARRAY,JRCD)', source)
        self.assertIn('JRRAY(1,2).EQ.2000', source)
        self.assertIn('CURRLPF=ARRAY(11)', source)
        self.assertIn('PEAKLPF=CURRLPF', source)
        self.assertIn('CURRLPF.LE.RATIO*PEAKLPF', source)
        self.assertIn('LSTOP=1', source)
        self.assertIn('RATIO=0.7D0', source)
        self.assertIn('RATIO=1D-07', step5.render_postpeak_urdfil(1e-7))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'queue.bat')
            step5.write_queue_batch(path, ['STEP5_TEST'], ratio=.701234567, stop_method='urdfil')
            with open(path) as stream:
                import re
                value = re.search(r'--ratio (\S+)', stream.read()).group(1)
            self.assertEqual(float(value), .701234567)

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

    def test_written_input_preserves_reference_solution_defaults(self):
        import os, tempfile
        from pathlib import Path
        from types import SimpleNamespace as NS
        from unittest import mock
        from test_pm_interaction import fake_template, cae_like_inp
        template, unused = fake_template()
        base = cae_like_inp(template)
        base = base[:base.index('** STEP5 URDFIL trigger')]+'*End Step\n'
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            try:
                os.chdir(directory)
                def write_input(**kw):
                    Path('TEST.inp').write_text(base)
                database = NS(Job=lambda **kw: NS(writeInput=write_input))
                constants = NS(ON=True, PERCENTAGE='PERCENTAGE', FULL='FULL', SINGLE='SINGLE')
                info = dict(reference_pipeline={}, bolts={'BOLT_1': {}}, pm_template=template)
                args = NS(cpus=12, stop_method='monitor')
                with mock.patch.dict('sys.modules', {'abaqusConstants': constants}):
                    step5.write_job_input(database, 'TEST', info, args, '')
                text = Path('TEST.inp').read_text()
                self.assertNotRegex(text.upper(), r'^\*(?:CONTACT CONTROLS|CONTROLS|COUPLING|KINEMATIC)\b')
                self.assertNotIn('*Contact Controls', text)
                self.assertNotIn('*Controls', text)
                self.assertIn('*NODE FILE, NSET=STEP5_STOP, FREQUENCY=1', text)
                self.assertEqual(info['input_sha256'], step5.pipeline_contract.file_sha256('TEST.inp'))
                for extra in ('*Coupling, constraint name=STALE\n', '*Controls, analysis=DISCONTINUOUS\n'):
                    Path('TEST.inp').write_text(text.replace('*End Step', extra+'*End Step'))
                    with self.assertRaisesRegex(RuntimeError, 'reference'):
                        step5.verify_input('TEST.inp', 1, bending=True)
            finally:
                os.chdir(previous)

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


@unittest.skipUnless(os.environ.get('CFS_REFERENCE_STEP4_CAE'), 'optional build-only Abaqus check')
class ReferenceCaeIntegrationTests(unittest.TestCase):
    def test_perfect_and_imperfect_bending_inputs_preserve_reference_ends_and_plate_mesh(self):
        import json, tempfile
        from pathlib import Path
        from abaqus import openMdb
        source = os.environ['CFS_REFERENCE_STEP4_CAE']
        original_hash = step5.pipeline_contract.file_sha256(source)
        original_mtime = os.stat(source).st_mtime_ns
        with tempfile.TemporaryDirectory(dir=os.path.dirname(step5.__file__)) as directory:
            output = Path(directory)/'step5'
            args = step5.parse_arguments(['--source-cae', source, '--output-dir', str(output),
                '--fy', '240', '--steel-plate-fy', '240', '--models', 'PERFECT', 'STEP4_LG_mp'])
            step5.build(args)  # Build only; the builder has no solver submission path.
            names = ['STEP5_PERFECT_FY240', 'STEP5_LG_mp_FY240']
            for name in names:
                info = json.loads((output/(name+step5.METADATA_SUFFIX)).read_text())
                self.assertEqual(info['end_condition']['id'], 'reference_nodal_ends_v1')
                self.assertEqual(info['plastic_tables']['SteelPlate'], [[240., 0.]])
                self.assertEqual(info['input_sha256'], step5.pipeline_contract.file_sha256(str(output/(name+'.inp'))))
                text = (output/(name+'.inp')).read_text().upper()
                self.assertNotIn('*COUPLING', text)
                self.assertNotIn('*CONTACT CONTROLS', text)
                self.assertNotIn('*CONTROLS', text)
                self.assertNotIn('*DSLOAD', text)
                self.assertIn('*CLOAD', text)
            self.assertFalse(list(output.glob('*.odb')))
            self.assertEqual(step5.pipeline_contract.file_sha256(source), original_hash)
            self.assertEqual(os.stat(source).st_mtime_ns, original_mtime)
            database = openMdb(pathName=str(output/'Step5_GMNIA_FY240.cae'))
            try:
                for name in names:
                    model = database.models[name]
                    self.assertEqual(len(model.boundaryConditions), 9)
                    self.assertFalse(model.rootAssembly.referencePoints)
                    self.assertEqual(len(model.constraints), 0)
                    self.assertEqual(len(model.loads), len(info['end_rotation_weights'][0])+len(info['end_rotation_weights'][1]))
            finally:
                database.close()


if __name__ == '__main__':
    if os.environ.get('CFS_REFERENCE_STEP4_CAE'):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.TestLoader().loadTestsFromTestCase(ReferenceCaeIntegrationTests))
        if not result.wasSuccessful():
            raise RuntimeError('Reference Step5 build-only integration check failed')
    else:
        unittest.main()
