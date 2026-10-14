"""P-M interaction generator: ray loads, INP variants, orchestration with fake CAE objects, collection.
No Abaqus execution."""
import csv
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import abaqus_load_case as load_cases
import abaqus_pm_interaction as pm
import abaqus_step5_gmnia as step5
from test_step5_gmnia import _fake_bending_mesh


def fake_template():
    instances, segments, unused = _fake_bending_mesh()
    end_meshes = [(name, dict((n.label, n.coordinates) for n in inst.nodes),
                   [tuple(n.label for n in e.getNodes()) for e in inst.elements])
                  for name, inst in sorted(instances.items())]
    return step5.pm_template_nodes(end_meshes, 0., 100.), segments


def cae_like_inp(template):
    """Text in the layout Abaqus/CAE writes for named concentrated forces in a Riks step."""
    lines = ['*Heading', '** Job name: PMT_X Model name: PMT_X', '*Element, type=CONN3D2, elset=W',
             '*Connector Section, elset=W', 'Beam,', '*Plastic', '240., 0.',
             '*Step, name=GMNIA, nlgeom=YES', '*Static, riks', '0.01, 1., 1e-08, 0.05', '**',
             '** LOADS', '**']
    for node in template['nodes']:
        lines += ['** Name: %s   Type: Concentrated force' % node['load'], '*Cload', '%s, 3, 1.' % node['set']]
    lines += ['*Output, field', '*Element Output', 'S, LE, PEEQ', '*Element Output, elset=W', 'CTF1, CTF2',
              '** STEP5 URDFIL trigger', '*NODE FILE, NSET=STEP5_STOP, FREQUENCY=1', 'U', '*End Step', '']
    return '\n'.join(lines)


def fake_end_plates():
    """End-plate records in the layout of step5.create_end_plates (no CAE)."""
    return [dict(side=side, set='STEP5_RP_'+suffix, xyz=[0., 0., z], coupling='STEP5_END_PLATE_'+suffix,
                 force_load='STEP5_END_%s_FORCE' % suffix, moment_load='STEP5_END_%s_MOMENT' % suffix)
            for side, suffix, z in ((0, 'BOTTOM', 0.), (1, 'TOP', 100.))]


def cae_like_gmnia_inp():
    """CAE layout of the GMNIA template: placeholder CF3 and CM1/CM2 at the end-plate RPs."""
    lines = ['*Heading', '** Job name: PMT_X Model name: PMT_X', '*Element, type=CONN3D2, elset=W',
             '*Connector Section, elset=W', 'Beam,', '*Plastic', '240., 0.']
    for plate in fake_end_plates():
        lines += ['** Constraint: %s' % plate['coupling'],
                  '*Coupling, constraint name=%s, ref node=%s, surface=%s_CNS_, influence radius=1e+20'
                  % (plate['coupling'], plate['set'], plate['coupling']), '*Kinematic', '1, 3']
    lines += ['*Step, name=GMNIA, nlgeom=YES', '*Static, riks', '0.01, 1., 1e-08, 0.05', '**',
              '** LOADS', '**']
    for plate in fake_end_plates():
        lines += ['** Name: %s   Type: Concentrated force' % plate['force_load'], '*Cload',
                  '%s, 3, 1.' % plate['set'],
                  '** Name: %s   Type: Moment' % plate['moment_load'], '*Cload',
                  '%s, 4, 1.' % plate['set'], '%s, 5, 1.' % plate['set']]
    lines += ['*Output, field', '*Element Output', 'S, LE, PEEQ', '*Element Output, elset=W', 'CTF1, CTF2',
              '*Contact Controls, stiffness scale factor=USER ADAPTIVE', '0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.',
              '*Controls, analysis=DISCONTINUOUS', '*Controls, parameters=LINE SEARCH', '5',
              '** STEP5 URDFIL trigger', '*NODE FILE, NSET=STEP5_STOP, FREQUENCY=1', 'U', '*End Step', '']
    return '\n'.join(lines)


def cload_dof_values(text):
    """{load name: {dof: value}} of every named *Cload block."""
    out, name = {}, None
    for line in text.split('\n'):
        if line.startswith('** Name:'):
            name = line.split()[2]
        elif name and line[:1].isalnum():
            region, dof, value = [f.strip() for f in line.split(',')]
            out.setdefault(name, {})[int(dof)] = float(value)
        elif line.startswith('*') and not line.upper().startswith('*CLOAD'):
            name = None
    return out


def cload_values(text):
    values, name = {}, None
    for line in text.split('\n'):
        if line.startswith('** Name:'):
            name = line.split()[2]
        elif name and not line.startswith('*'):
            values[name] = float(line.split(',')[2])
            name = None
    return values


class PMInteractionTests(unittest.TestCase):
    def test_combined_reference_is_unit_at_extreme_fibre_and_reduces_to_pure_cases(self):
        square = [(-100., -100., 100., -100.), (100., -100., 100., 100.),
                  (100., 100., -100., 100.), (-100., 100., -100., -100.)]
        for theta in (0., 45.):
            for psi in (0., 30., 90.):
                ref = load_cases.combined_reference(square, 2., theta, psi)
                nx, ny = ref['compression_normal']
                c = ref['c_extreme_mm']
                self.assertAlmostEqual(load_cases.unit_stress(ref, nx*c, ny*c), 1.0)
                self.assertAlmostEqual(ref['axial_weight']+ref['bending_weight'], 1.0)
        self.assertEqual(load_cases.combined_reference(square, 2., 0., 0.)['bending_weight'], 0.)
        self.assertEqual(load_cases.combined_reference(square, 2., 0., 90.)['axial_weight'], 0.)
        with self.assertRaises(ValueError):
            load_cases.combined_reference(square, 2., 0., 95.)

    def test_plastic_curve_spans_pure_bending_to_squash_load(self):
        square = [(-100., -100., 100., -100.), (100., -100., 100., 100.),
                  (100., 100., -100., 100.), (-100., 100., -100., -100.)]
        rows = load_cases.plastic_interaction(square, 2., 0., 21)
        self.assertAlmostEqual(rows[0][1], 0., places=9)
        self.assertAlmostEqual(rows[0][2], load_cases.bending_reference(square, 2., 0.)['plastic_modulus_mm3'])
        self.assertAlmostEqual(rows[-1][1], 1600.)
        self.assertAlmostEqual(rows[-1][2], 0., places=6)
        self.assertTrue(all(b[1] >= a[1]-1e-9 for a, b in zip(rows, rows[1:])))

    def test_rays_compression_once_and_names_are_valid(self):
        rays = pm.ray_list([0., 45.], [0., 30., 90.])
        self.assertEqual([r[0] for r in rays], ['PM_P00', 'PM_A000_P30', 'PM_A000_P90', 'PM_A045_P30', 'PM_A045_P90'])
        self.assertEqual(pm.check_name('PM_A022p5_P22p5_LDG_pp'), 'PM_A022p5_P22p5_LDG_pp')
        with self.assertRaises(ValueError):
            pm.check_name('PM_A022p5_P22p5_'+'X'*30)

    def test_equivalent_axes_are_rejected(self):
        unused, segments = fake_template()
        axes, info = pm.resolve_axes('required', segments)
        self.assertEqual(axes, [0., 45.])
        with self.assertRaises(ValueError):
            pm.resolve_axes([0., 90.], segments)

    def test_x_plates_keep_orthogonal_bending_axes_distinct(self):
        unused, segments = fake_template()
        symmetry = load_cases.section_symmetry(segments)
        symmetry['rotations_deg'] = [0, 180]
        symmetry['mirror_lines_deg'] = [0., 90.]
        axes, info = pm.resolve_axes([0., 90.], segments, symmetry)
        self.assertEqual(axes, [0., 90.])

    def test_ray_forces_reproduce_resultants_and_rotation_weights(self):
        template, segments = fake_template()
        bending = load_cases.bending_reference(segments, 2., 45.)
        unused, pure = pm.ray_forces(template, 2., bending)
        area = load_cases.section_properties(segments, 2.)['area_mm2']
        for psi in (0., 20., 45., 90.):
            ref = load_cases.combined_reference(segments, 2., 45., psi)
            forces, totals = pm.ray_forces(template, 2., ref)
            if psi == 0.:
                self.assertEqual(totals['moment_Nmm_per_MPa'], 0.)
            if psi == 90.:
                self.assertEqual(totals['axial_force_N_per_MPa'], 0.)
            self.assertAlmostEqual(totals['axial_force_N_per_MPa'], ref['axial_weight']*area, places=6)
            self.assertAlmostEqual(totals['moment_Nmm_per_MPa'], ref['bending_weight']*pure['moment_Nmm_per_MPa'],
                                   places=4)
            # equal and opposite at the two ends
            ends = [sum(v for n in template['nodes'] if n['side'] == s for v in [forces[n['load']]]) for s in (0, 1)]
            self.assertAlmostEqual(ends[0], -ends[1], places=6)
        weights, m_unit = pm.rotation_weights(template, 2., segments, 45.)
        nx, ny = bending['compression_normal']
        xy = dict(((n['instance'], n['label']), (n['x'], n['y'])) for n in template['nodes'] if n['side'] == 0)
        rot = sum(k*(0.3+2e-3*(nx*xy[(i, l)][0]+ny*xy[(i, l)][1])) for i, l, k in weights[0])
        self.assertAlmostEqual(rot, 2e-3, places=12)

    def test_inp_variant_replaces_only_cload_magnitudes(self):
        template, segments = fake_template()
        text = cae_like_inp(template)
        sets = dict((n['load'], (n['set'], n['instance'], n['label'])) for n in template['nodes'])
        values = dict((n['load'], 0.5*i) for i, n in enumerate(template['nodes']))
        out = pm.set_cload_values(text, values, sets)
        self.assertEqual(cload_values(out), values)
        self.assertEqual([l for l in out.split('\n') if not l[:1].isalnum() and 'P' not in l[:3]],
                         [l for l in text.split('\n') if not l[:1].isalnum() and 'P' not in l[:3]])
        # instance.label form is accepted as the same node, anything else is refused
        alt = text.replace(template['nodes'][0]['set']+', 3', '%s.%d, 3' % (template['nodes'][0]['instance'],
                                                                           template['nodes'][0]['label']), 1)
        pm.set_cload_values(alt, values, sets)
        with self.assertRaises(ValueError):
            pm.set_cload_values(text.replace(template['nodes'][0]['set']+', 3', 'OTHER, 3', 1), values, sets)
        with self.assertRaises(ValueError):
            pm.set_cload_values(text, dict(values, PM_END_MISSING=1.), dict(sets, PM_END_MISSING=('X', 'P1', 1)))

    def test_end_plate_placeholders_take_force_and_both_moment_components(self):
        text = cae_like_gmnia_inp()
        plates = fake_end_plates()
        sets = dict((p[key], (p['set'], None, None)) for p in plates for key in ('force_load', 'moment_load'))
        values = {}
        for p in plates:
            values[p['force_load']] = {3: 10.+p['side']}
            values[p['moment_load']] = {4: 2.5-p['side'], 5: -7.}
        out = pm.set_cload_values(text, values, sets)
        self.assertEqual(cload_dof_values(out), values)
        missing_dof = dict(values)
        missing_dof[plates[0]['moment_load']] = {4: 1.}
        with self.assertRaises(ValueError):            # one data line per requested DOF
            pm.set_cload_values(text, missing_dof, sets)
        with self.assertRaises(ValueError):            # the RP set is the only accepted region
            pm.set_cload_values(text.replace('STEP5_RP_TOP, 5', 'P1.1, 5'), values, sets)

    def test_gmnia_orchestration_writes_consistent_rays(self):
        template, segments = fake_template()
        with tempfile.TemporaryDirectory() as root:
            cae = os.path.join(root, 'Step4.cae')
            open(cae, 'w').close()
            out = os.path.join(root, 'out')
            models = {'JOB': NS(description='')}
            database = NS(models=models, jobs={}, close=mock.Mock())

            def prepare(model, args, pm_template=False):
                self.assertTrue(pm_template)
                return dict(stage=5, settings=dict(fy=args.fy, reference_stress=args.reference_stress,
                                                   field_frequency=1), bolts={'BOLT_S1_01': {}},
                            reference_pipeline={'nodal_precision': 'full'}, end_area_mm2=area,
                            end_area_weights=[[['P1', 1, 1.]], [['P1', 2, 1.]]],
                            source_step4=dict(case={}), end_condition=dict(step5.pipeline_contract.STEP5_END_CONDITION),
                            pm_template=json.loads(json.dumps(template)))

            def write_job(db, name, info, args, sub):
                db.jobs[name] = object()
                with open(name+'.inp', 'w') as stream:
                    stream.write(cae_like_inp(template))

            def copy(db, source, name, perfect):
                db.models[name] = object()
                return db.models[name]

            area = load_cases.section_properties(segments, 2.)['area_mm2']
            manifest = dict(reference_pipeline=dict(run_dir=root, job_name='JOB', thickness_mm=2.))
            with mock.patch.object(step5, 'open_step4_database',
                                   return_value=(database, ['PERFECT', 'STEP4_LDG_pp'], 'JOB')), \
                    mock.patch.object(step5, 'reference_step4_manifest', return_value=manifest), \
                    mock.patch.object(pm, 'reference_section', return_value=({}, {}, segments, 2.)), \
                    mock.patch.object(step5, 'copy_source_model', side_effect=copy), \
                    mock.patch.object(step5, 'prepare_model', side_effect=prepare), \
                    mock.patch.object(step5, 'write_job_input', side_effect=write_job):
                args = pm.parse_arguments(['--mode', 'gmnia', '--source-cae', cae, '--output-dir', out,
                                           '--fy', '240', '--pm-angles', '0,45,90'])
                pm.run_gmnia(args)
            files = sorted(os.listdir(out))
            jobs = sorted(f[:-4] for f in files if f.endswith('.inp'))
            self.assertEqual(len(jobs), 2*5)       # (P00 + 2 axes x 2 angles) x 2 cases
            self.assertIn('PM_A045_P45_LDG_pp', jobs)
            self.assertIn('run_pm_queue_70_pct.bat', files)
            self.assertIn(pm.MANIFEST_FILE, files)
            self.assertIn(pm.SECTION_FILE, files)
            self.assertFalse(any(f.startswith('PMT_') for f in files))
            with open(os.path.join(out, 'run_pm_queue_70_pct.bat')) as stream:
                self.assertIn('PM_A000_P90_PERFECT', stream.read())
            for job in jobs:
                with open(os.path.join(out, job+step5.METADATA_SUFFIX)) as stream:
                    info = json.load(stream)
                with open(os.path.join(out, job+'.inp')) as stream:
                    values = cload_dof_values(stream.read())
                lc = info['load_case']
                nx, ny = lc['compression_normal']
                cx, cy = lc['centroid_mm']
                self.assertEqual(set(values), {n['load'] for n in template['nodes']})
                for side, sign in ((0, 1.), (1, -1.)):
                    nodes = [n for n in template['nodes'] if n['side'] == side]
                    p = sum(values[n['load']][3] for n in nodes)
                    m = sum(values[n['load']][3]*(nx*(n['x']-cx)+ny*(n['y']-cy)) for n in nodes)
                    self.assertAlmostEqual(p, sign*info['reference_axial_force_N'], delta=1e-6*max(1., abs(p)))
                    self.assertAlmostEqual(m, sign*info['reference_moment_Nmm'], delta=1e-6*max(1., abs(m)))
                self.assertEqual(info['end_condition']['id'], step5.pipeline_contract.STEP5_END_CONDITION_ID)
                self.assertEqual(lc['type'], 'combined')
                self.assertAlmostEqual(info['section_for_PM']['squash_load_N'], area*240.)
                # LPF = 1 is first yield: P/Py + M/My = 1
                util = info['reference_axial_force_N']/info['section_for_PM']['squash_load_N'] + \
                    info['reference_moment_Nmm']/info['section_for_PM']['first_yield_moment_Nmm']
                self.assertAlmostEqual(util, 1.0, places=9)

    def test_collect_combines_gmnia_and_elastic_points(self):
        template, segments = fake_template()
        with tempfile.TemporaryDirectory() as root:
            ref = pm.section_reference(segments, 2., [0.])
            pm.write_section_reference(root, ref)
            with open(os.path.join(root, 'STEP5_PM_points.csv'), 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['model', 'load_case', 'bending_axis_deg', 'pm_angle_deg',
                                                            'P_kN', 'M_kNm', 'P_over_Py', 'M_over_My', 'outcome'])
                writer.writeheader()
                writer.writerow(dict(model='PM_P00_PERFECT', load_case='combined', bending_axis_deg='',
                                     pm_angle_deg=0, P_kN=100, M_kNm=0, P_over_Py=.6, M_over_My=0,
                                     outcome='CRITERION_REACHED'))
                writer.writerow(dict(model='PM_A000_P45_PERFECT', load_case='combined', bending_axis_deg=0,
                                     pm_angle_deg=45, P_kN=50, M_kNm=5, P_over_Py=.3, M_over_My=.3,
                                     outcome='CRITERION_REACHED'))
            folder = os.path.join(root, 'PMB_A000_P45')
            os.makedirs(folder)
            with open(os.path.join(folder, 'PMB_A000_P45_build.json'), 'w') as stream:
                json.dump(dict(load_case=dict(neutral_axis_angle_deg=0., pm_angle_deg=45.,
                                              reference_axial_force_mesh_N_per_MPa=400.,
                                              reference_moment_mesh_Nmm_per_MPa=1e4)), stream)
            with open(os.path.join(folder, 'PMB_A000_P45.dat'), 'w') as stream:
                stream.write(' MODE NO      EIGENVALUE\n\n\n       1       387.73    \n       2       389.13\n\n'
                             '  OTHER TEXT\n')
            out = os.path.join(root, 'diagram')
            result = pm.collect(pm.parse_arguments(['--mode', 'collect', '--roots', root, '--output-dir', out,
                                                    '--fy', '240']))
            self.assertEqual(result['points'], 3)
            with open(result['csv'], encoding='utf-8-sig') as stream:
                rows = list(csv.DictReader(stream))
            elastic = [r for r in rows if r['source'] == 'ELASTIC_BUCKLING'][0]
            self.assertAlmostEqual(float(elastic['P_kN']), 387.73*400./1e3)
            self.assertAlmostEqual(float(elastic['M_kNm']), 387.73*1e4/1e6)
            self.assertTrue(os.path.isfile(os.path.join(out, 'PM_interaction_A000.png')))

    def test_collect_keeps_skip_datasets_and_section_normalizations_separate(self):
        from pathlib import Path
        template,segments=fake_template()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            root=Path(d)
            for count,thickness in [(0,2.),(1,4.)]:
                dataset=root/('skip_%d'%count);dataset.mkdir()
                pm.write_section_reference(str(dataset),pm.section_reference(segments,thickness,[0.]))
                job='PMB_A000_P45';folder=dataset/job;folder.mkdir()
                (folder/(job+'_build.json')).write_text(json.dumps(dict(load_case=dict(
                    neutral_axis_angle_deg=0.,pm_angle_deg=45.,reference_axial_force_mesh_N_per_MPa=400.,
                    reference_moment_mesh_Nmm_per_MPa=1e4))))
                (folder/(job+'.dat')).write_text('MODE NO      EIGENVALUE\n\n       1       100.0\n')
            args=pm.parse_arguments(['--mode','collect','--roots',str(root),str(root/'skip_0'),
                                     '--output-dir',str(root/'diagram'),'--fy','240'])
            with mock.patch.object(pm,'plot_interaction',return_value=[]) as plot:
                result=pm.collect(args)
            self.assertEqual(result['points'],2)
            with open(result['csv'],encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len({row['dataset'] for row in rows}),2)
            ordered=sorted(rows,key=lambda r:r['dataset'])
            self.assertAlmostEqual(float(ordered[1]['Py_kN']),2*float(ordered[0]['Py_kN']))
            self.assertAlmostEqual(float(ordered[1]['P_over_Py']),.5*float(ordered[0]['P_over_Py']))
            self.assertEqual(plot.call_args.args[2],{},'No shared plastic reference for differing sections')

    def test_dataset_identity_is_not_a_job_or_shared_leaf_folder_name(self):
        self.assertNotEqual(pm.interaction_dataset(dict(source='GMNIA',folder='skip_0/Step5_PM')),
                            pm.interaction_dataset(dict(source='GMNIA',folder='skip_1/Step5_PM')))
        self.assertEqual(pm.interaction_dataset(dict(source='ELASTIC_BUCKLING',folder='skip_0/PMB_A000_P45',
                                                     model='PMB_A000_P45')),os.path.abspath('skip_0'))

    def test_plots_do_not_connect_different_skip_configurations(self):
        import matplotlib.pyplot as plt
        from pathlib import Path
        axes=[];original=plt.subplots
        def capture(*args,**kwargs):
            fig,ax=original(*args,**kwargs);axes.append(ax);return fig,ax
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            points=[]
            for count,offset in [(0,0.),(1,10.)]:
                for i,psi in enumerate((30.,45.,60.)):
                    points.append(dict(source='GMNIA',source_case='LG_mp',load_case='combined',
                        folder=str(Path(d)/('skip_%d'%count)/'Step5_PM'),bending_axis_deg=0.,
                        pm_angle_deg=psi,M_kNm=offset+i+1.,P_kN=3.-i,
                        selection_kind='VERIFIED_PEAK',capacity_verified=True))
            with mock.patch.object(plt,'subplots',side_effect=capture):
                pm.plot_interaction(d,points,{})
            lines=[line for line in axes[0].get_lines() if line.get_label().startswith('GMNIA')]
            self.assertEqual(len(lines),2)
            self.assertTrue(all(len(line.get_xdata())==3 for line in lines))
            self.assertTrue(any(max(line.get_xdata())<10. for line in lines))
            self.assertTrue(any(min(line.get_xdata())>10. for line in lines))

    def test_first_eigenvalue_reads_abaqus_dat_table(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'x.dat')
            with open(path, 'w') as stream:
                stream.write('   E I G E N V A L U E    O U T P U T\n\n MODE NO      EIGENVALUE\n\n\n'
                             '       1      -12.5    \n       2       387.73    \n       3       392.17\n\n'
                             '                       P A R T I C I P A T I O N\n')
            self.assertEqual(pm.first_eigenvalue(path), 387.73)

    def test_buckling_orchestration_writes_one_folder_per_ray(self):
        import sys
        import types
        template, segments = fake_template()
        instances, unused_segments, unused = _fake_bending_mesh()

        class Repo(dict):
            pass

        class Model(object):
            def __init__(self):
                self.loads = Repo(COMP=1)
                self.steps = Repo(Initial=1, Buckle=1)
                self.fieldOutputRequests = Repo(old=1)
                self.historyOutputRequests = Repo(old=1)
                self.boundaryConditions = Repo(('BC%d' % i, 1) for i in range(9))
                self.rootAssembly = NS(instances=instances, sets={},
                                       SetFromNodeLabels=lambda name, nodeLabels: self.rootAssembly.sets.__setitem__(
                                           name, NS(name=name)))
                self.created = []

            def BuckleStep(self, **kw):
                self.steps['Buckle'] = kw

            def FieldOutputRequest(self, **kw):
                self.fieldOutputRequests[kw['name']] = kw

            def ConcentratedForce(self, name, **kw):
                self.loads[name] = kw

        written = {}

        class Database(object):
            def __init__(self):
                self.models = {'JOB': Model()}
                self.jobs = {}

            def Model(self, name, objectToCopy):
                self.models[name] = Model()
                return self.models[name]

            def Job(self, name, **kw):
                model = self.models[kw['model']]
                def write(consistencyChecking=None):
                    written['model'] = model
                    lines = ['*Heading', '*Step, name=Buckle', '*Buckle, eigensolver=LANCZOS', '20, 0., , , ']
                    for node in template['nodes']:
                        lines += ['** Name: %s   Type: Concentrated force' % node['load'], '*Cload',
                                  '%s, 3, 1.' % node['set']]
                    lines += ['*End Step', '']
                    with open(name+'.inp', 'w') as stream:
                        stream.write('\n'.join(lines))
                return NS(writeInput=write)

            def close(self):
                pass

        constants = types.ModuleType('abaqusConstants')
        for key in ('ON', 'LANCZOS', 'PERCENTAGE', 'FULL', 'SINGLE', 'UNIFORM', 'OFF'):
            setattr(constants, key, key)
        modules = {'caeModules': types.ModuleType('caeModules'), 'abaqusConstants': constants,
                   'abaqus': NS(openMdb=lambda pathName: Database())}
        build = dict(job_name='JOB', boundary_conditions=9, loads=8, source_inputs=dict(thickness_mm=2.))
        state = dict(settings=dict(nodal_precision='full'))
        with tempfile.TemporaryDirectory() as root:
            open(os.path.join(root, 'JOB.cae'), 'w').close()
            out = os.path.join(root, 'buckling')
            with mock.patch.dict(sys.modules, modules), \
                    mock.patch.object(pm, 'reference_section',
                                      return_value=(state, dict(build=build, job_name='JOB'), segments, 2.)):
                args = pm.parse_arguments(['--mode', 'buckling', '--run-dir', root, '--output-dir', out,
                                           '--pm-angles', '0,60', '--n-modes', '30'])
                pm.run_buckling(args)
            model = written['model']
            self.assertEqual(model.steps['Buckle']['minEigen'], 0.)
            self.assertEqual(model.steps['Buckle']['numEigen'], 30)
            self.assertEqual(sorted(model.fieldOutputRequests), ['ModeShapes'])
            self.assertNotIn('COMP', model.loads)
            folders = sorted(f for f in os.listdir(out) if os.path.isdir(os.path.join(out, f)))
            self.assertEqual(folders, ['PMB_A000_P60', 'PMB_A045_P60', 'PMB_P00'])
            for folder in folders:
                with open(os.path.join(out, folder, folder+'_build.json')) as stream:
                    build_json = json.load(stream)
                with open(os.path.join(out, folder, folder+'.inp')) as stream:
                    values = cload_values(stream.read())
                lc = build_json['load_case']
                bottom = [n for n in template['nodes'] if n['side'] == 0]
                self.assertAlmostEqual(sum(values[n['load']] for n in bottom),
                                       lc['reference_axial_force_mesh_N_per_MPa'], places=6)
                self.assertEqual(build_json['job_name'], folder)
                self.assertTrue(os.path.isfile(os.path.join(out, folder, 'abaqus_v6.env')))
            with open(os.path.join(out, 'run_pm_buckling.bat')) as stream:
                self.assertIn('job=PMB_A045_P60 input=PMB_A045_P60.inp', stream.read())


if __name__ == '__main__':
    unittest.main()
