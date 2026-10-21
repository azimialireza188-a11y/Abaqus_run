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
    def test_plain_plot_connects_verified_and_stopped_points_within_each_case(self):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        points=[]
        for case in ('PERFECT','LG_mp'):
            for angle,m,p,kind in ((0.,0.,100.,'VERIFIED_PEAK'),
                                   (45.,5.,60.,'UNVERIFIED_OBSERVED_MAXIMUM'),
                                   (90.,10.,0.,'INCOMPLETE_ASCENDING_ENDPOINT')):
                points.append(dict(source='GMNIA',source_case=case,dataset='specimen',load_case='combined',
                    bending_axis_deg=0.,pm_angle_deg=angle,M_kNm=m,P_kN=p,selection_kind=kind,
                    capacity_verified=kind=='VERIFIED_PEAK'))
        figures=[]
        with tempfile.TemporaryDirectory() as root, mock.patch.object(Figure,'savefig',autospec=True,
                side_effect=lambda fig,*args,**kw:figures.append(fig)):
            pm.plot_interaction(root,points,{})
        connections=[line for line in figures[0].axes[0].lines if len(line.get_xdata())>1]
        self.assertEqual(len(connections),2)
        self.assertTrue(all(len(line.get_xdata())==3 for line in connections))
        self.assertTrue(all(5. in line.get_xdata() for line in connections))
        self.assertEqual({line.get_linestyle() for line in connections},{'-','--'})
        self.assertFalse(plt.get_fignums())

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
                            source_step4=dict(case={}), pm_template=json.loads(json.dumps(template)))

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
                    values = cload_values(stream.read())
                bottom = [n for n in template['nodes'] if n['side'] == 0]
                p = sum(values[n['load']] for n in bottom)
                self.assertAlmostEqual(p, info['reference_axial_force_N'], places=5)
                lc = info['load_case']
                nx, ny = lc['compression_normal']
                m = sum(values[n['load']]*(nx*n['x']+ny*n['y']) for n in bottom)
                self.assertAlmostEqual(m, info['reference_moment_Nmm'], delta=1e-6*max(1., abs(m)))
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

    def _case_plot_lines(self, points, normalized=False):
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        captured=[];original=plt.subplots
        def capture(*args,**kwargs):
            fig,axes=original(*args,**kwargs)
            captured.extend(list(axes.flat) if hasattr(axes,'flat') else [axes])
            return fig,axes
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as directory, \
                mock.patch.object(plt,'subplots',side_effect=capture), \
                mock.patch.object(Figure,'savefig'):
            pm.plot_interaction(directory,points,{},normalized=normalized)
        return [[line for line in ax.lines if line.get_label().startswith('GMNIA')] for ax in captured]

    def test_provisional_case_markers_stay_distinct_on_dimensional_normalized_and_panel_plots(self):
        points=[]
        for case in ('PERFECT','LG_mp'):
            for axis in (0.,90.):
                for i,psi in enumerate((30.,60.)):
                    points.append(dict(source='GMNIA',source_case=case,load_case='combined',
                        dataset=os.path.abspath('specimen'),bending_axis_deg=axis,pm_angle_deg=psi,
                        M_kNm=1.+i,P_kN=3.-i,M_over_My=.1*(1.+i),P_over_Py=.1*(3.-i),
                        selection_kind='INCOMPLETE_ASCENDING_ENDPOINT',capacity_verified=False))
        for normalized in (False,True):
            panels=self._case_plot_lines(points,normalized)
            self.assertEqual(len(panels),4)  # two separate figures and the two-panel figure
            for lines in panels:
                self.assertEqual(len(lines),2)
                by_case={line.get_label().split()[1]:line for line in lines}
                perfect,imperfect=by_case['PERFECT'],by_case['LG_mp']
                self.assertEqual(perfect.get_marker(),'o')
                self.assertEqual(imperfect.get_marker(),'s')
                self.assertNotEqual(perfect.get_color(),imperfect.get_color())
                for line in lines:
                    self.assertEqual(line.get_linestyle(),'None')
                    self.assertEqual(line.get_markerfacecolor(),'white')
                    self.assertIn('incomplete/provisional',line.get_label())
                    self.assertEqual(len(line.get_xdata()),2)

    def test_verified_case_curves_use_case_styles_and_keep_imperfection_cases_separate(self):
        points=[]
        for case in ('PERFECT','STEP4_LG_mp','STEP4_LDG_pp'):
            for i,psi in enumerate((30.,60.)):
                points.append(dict(source='GMNIA',source_case=case,load_case='combined',
                    bending_axis_deg=0.,pm_angle_deg=psi,M_kNm=1.+i,P_kN=3.-i,
                    selection_kind='VERIFIED_PEAK',capacity_verified=True))
        lines=self._case_plot_lines(points)[0]
        self.assertEqual(len(lines),3)
        by_case={line.get_label().split()[1]:line for line in lines}
        self.assertEqual(set(by_case),{'PERFECT','LG_mp','LDG_pp'})
        perfect=by_case['PERFECT']
        self.assertEqual((perfect.get_marker(),perfect.get_linestyle()),('o','-'))
        for case in ('LG_mp','LDG_pp'):
            line=by_case[case]
            self.assertEqual((line.get_marker(),line.get_linestyle()),('s','--'))
            self.assertNotEqual(line.get_color(),perfect.get_color())
            self.assertEqual(len(line.get_xdata()),2)

    def test_case_identity_falls_back_to_legacy_model_and_job_names(self):
        fixtures=[(dict(model='PM_A000_P30_PERFECT'),'PERFECT','o'),
                  (dict(model='PM_P00_LG_mp'),'LG_mp','s'),
                  (dict(model='STEP5_LDG_pp_FY240'),'LDG_pp','s'),
                  (dict(job_name='STEP5_PERFECT_FY240'),'PERFECT','o'),
                  (dict(job='STEP4_custom_shape'),'custom_shape','s'),
                  (dict(source_case='STEP4_custom_shape',model='PM_P00_PERFECT'),'custom_shape','s'),
                  (dict(source_case='IMPERFECT'),'IMPERFECT','s')]
        for metadata,case,marker in fixtures:
            with self.subTest(metadata=metadata):
                point=dict(source='GMNIA',load_case='combined',bending_axis_deg=0.,pm_angle_deg=30.,
                           M_kNm=1.,P_kN=2.,selection_kind='VERIFIED_PEAK',capacity_verified=True,**metadata)
                line=self._case_plot_lines([point])[0][0]
                self.assertEqual(line.get_label().split()[1],case)
                self.assertEqual(line.get_marker(),marker)

    def test_case_styles_preserve_selection_groups_and_connection_permissions(self):
        import copy
        points=[]
        selections=[('VERIFIED_PEAK',True),('SIGNIFICANT_STIFFNESS_CHANGE',False),
                    ('SIGNIFICANT_LOAD_DROP',False),('INCOMPLETE_ASCENDING_ENDPOINT',False),
                    ('VERIFIED_PEAK',False)]
        for group,(kind,verified) in enumerate(selections):
            for i,psi in enumerate((30.,60.)):
                points.append(dict(source='GMNIA',source_case='PERFECT',load_case='combined',
                    bending_axis_deg=0.,pm_angle_deg=psi,M_kNm=10.*group+i+1.,P_kN=3.-i,
                    selection_kind=kind,capacity_verified=verified,provisional=not verified,
                    selection_policy='response-v1',frame=5+group,lpf=.5,
                    selection_diagnostics={'evidence':'retained'}))
        original=copy.deepcopy(points)
        lines=self._case_plot_lines(points)[0]
        self.assertEqual(points,original)
        self.assertEqual(len(lines),5)
        self.assertEqual(sorted(len(line.get_xdata()) for line in lines),[2]*5)
        connected=[line for line in lines if line.get_linestyle()!='None']
        self.assertEqual(len(connected),1)
        self.assertEqual(list(connected[0].get_xdata()),[2.,1.])
        self.assertEqual(connected[0].get_linestyle(),'-')
        for line in lines:
            self.assertEqual(line.get_marker(),'o')
            if line not in connected:
                self.assertEqual(line.get_markerfacecolor(),'white')

    def test_axial_only_results_show_all_reference_axes_and_state_missing_member_envelope(self):
        import matplotlib.pyplot as plt
        from pathlib import Path
        square=[(-10.,-10.,10.,-10.),(10.,-10.,10.,10.),
                (10.,10.,-10.,10.),(-10.,10.,-10.,-10.)]
        reference=pm.section_reference(square,1.,[0.,90.],fy=240.,points=11)
        sections={axis:dict(data,fy_MPa=240.) for axis,data in reference['axes'].items()}
        captured=[];original=plt.subplots
        def capture(*args,**kwargs):
            fig,axes=original(*args,**kwargs);captured.extend(list(axes.flat) if hasattr(axes,'flat') else [axes])
            return fig,axes
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as directory:
            points=[dict(source='GMNIA',source_case='LG_mp',load_case='combined',
                folder=directory,bending_axis_deg=None,pm_angle_deg=0.,M_kNm=0.,P_kN=10.,
                selection_kind='VERIFIED_PEAK',capacity_verified=True)]
            with mock.patch.object(plt,'subplots',side_effect=capture):
                files=pm.plot_interaction(directory,points,sections)
            self.assertTrue((Path(directory)/'PM_interaction_A090.svg').is_file())
            self.assertTrue(any(Path(path).name=='PM_interaction.png' for path in files))
            for ax in captured:
                self.assertTrue(any('not sampled' in text.get_text() for text in ax.texts))
                self.assertTrue({'A','B'}.issubset({text.get_text() for text in ax.texts}))
                gmnia=[line for line in ax.lines if line.get_label().startswith('GMNIA')]
                self.assertEqual(len(gmnia),1)
                self.assertEqual(gmnia[0].get_linestyle(),'None')
                self.assertLess(ax.get_legend().get_bbox_to_anchor().y1,ax.get_window_extent().y0)

    def test_explicit_collect_report_uses_current_model_fy_or_requires_it(self):
        import abaqus_report_reference as reader
        with tempfile.TemporaryDirectory() as root:
            model = 'STEP5_B_FY360'
            path = os.path.join(root, 'STEP5_PM_points.csv')
            with open(path, 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['model', 'load_case', 'bending_axis_deg', 'P_kN', 'M_kNm'])
                writer.writeheader()
                writer.writerow(dict(model=model, load_case='bending', bending_axis_deg=0., P_kN=0., M_kNm=50.))
            metadata_path = os.path.join(root, model+'_step5.json')
            with open(metadata_path, 'w') as stream:
                json.dump(dict(stage=5, settings=dict(fy=360.)), stream)
            args = pm.parse_arguments(['--mode', 'collect', '--roots', root, '--output-dir',
                                       os.path.join(root, 'diagram'), '--cufsm-report-dir', root])
            with mock.patch.object(reader, 'read_reference', return_value={'fy_MPa':360.}) as read, \
                    mock.patch.object(pm, 'plot_interaction', return_value=[]):
                pm.collect(args)
            read.assert_called_once_with(root, fy=360.)
            with open(metadata_path, 'w') as stream:
                json.dump(dict(stage=5), stream)
            with mock.patch.object(reader, 'read_reference') as read, self.assertRaisesRegex(ValueError, '--fy'):
                pm.collect(args)
            read.assert_not_called()
            with open(metadata_path, 'w') as stream:
                json.dump(dict(stage=5, settings=dict(fy=360.)), stream)
            second = dict(source='GMNIA', folder=root, model='STEP5_OTHER')
            with open(os.path.join(root, 'STEP5_OTHER_step5.json'), 'w') as stream:
                json.dump(dict(stage=5, settings=dict(fy=240.)), stream)
            first = dict(source='GMNIA', folder=root, model=model)
            with self.assertRaisesRegex(ValueError, '--fy'):
                pm.collected_report_fy([first, second], {})

    def test_collect_uses_explicit_or_embedded_report_without_changing_capacities(self):
        import abaqus_report_reference as reader
        reference = dict(schema='cufsm_builtup_reference_v1', fy_MPa=240., AsFy_kN=1100.)
        with tempfile.TemporaryDirectory() as root:
            metadata_path = os.path.join(root, 'STEP5_B_FY240_step5.json')
            original = json.dumps(dict(stage=5, report_reference=reference))
            with open(metadata_path, 'w') as stream:
                stream.write(original)
            path = os.path.join(root, 'STEP5_PM_points.csv')
            with open(path, 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['model', 'load_case', 'bending_axis_deg',
                    'P_kN', 'M_kNm', 'Py_kN', 'My_kNm', 'P_over_Py', 'M_over_My'])
                writer.writeheader()
                writer.writerow(dict(model='STEP5_B_FY240', load_case='bending', bending_axis_deg=0.,
                    P_kN=0., M_kNm=50., Py_kN=1000., My_kNm=100., P_over_Py=0., M_over_My=.5))
            argv = ['--mode', 'collect', '--roots', root, '--output-dir', os.path.join(root, 'diagram')]
            with mock.patch.object(pm, 'plot_interaction', return_value=[]) as plot:
                pm.collect(pm.parse_arguments(argv))
            self.assertEqual(plot.call_args.kwargs['report_reference'], reference)
            with mock.patch.object(reader, 'read_reference', return_value=reference) as read, \
                    mock.patch.object(pm, 'plot_interaction', return_value=[]) as plot:
                args = pm.parse_arguments(argv+['--cufsm-report-dir', root, '--fy', '360'])
                result = pm.collect(args)
            read.assert_called_once_with(root, fy=360.)
            self.assertEqual(plot.call_args.kwargs['report_reference'], reference)
            with open(result['csv'], encoding='utf-8-sig') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(float(row['My_kNm']), 100.)
            self.assertEqual(float(row['M_over_My']), .5)
            with open(metadata_path) as stream:
                self.assertEqual(stream.read(), original)

    def test_normalized_plot_omits_report_lines_with_clear_reason_when_scales_missing(self):
        import contextlib, io
        import matplotlib.pyplot as plt
        reference = dict(schema='cufsm_builtup_reference_v1', AsFy_kN=1100., axes={
            'x': dict(axis_deg=0., SFy_kNm=130., ZFy_kNm=190.)})
        points = [dict(source='GMNIA', bending_axis_deg=0., M_kNm=50., P_kN=200.,
                       M_over_My=.5, P_over_Py=.2)]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(plt, 'close'), \
                contextlib.redirect_stdout(output):
            pm.plot_interaction(directory, points, {}, normalized=True, report_reference=reference)
            self.assertFalse(any('CUFSM report' in line.get_label() for line in plt.gcf().axes[0].lines))
        self.assertIn('actual panel Py/My are unavailable', output.getvalue())
        plt.close('all')

    def test_normalized_report_scales_require_actual_panel_capacity(self):
        points = [dict(bending_axis_deg=0., M_kNm=50., Py_kN=1000., My_kNm=100.),
                  dict(bending_axis_deg=90., M_kNm=40., Py_kN=1000., My_kNm=80.)]
        self.assertEqual(pm.normalized_report_scales(points, 0., None, None), (100., 1000.))
        self.assertEqual(pm.normalized_report_scales(points, 90., None, None), (80., 1000.))
        self.assertIsNone(pm.normalized_report_scales([], 0., None, None))
        points.append(dict(bending_axis_deg=0., M_kNm=50., Py_kN=1100., My_kNm=100.))
        self.assertIsNone(pm.normalized_report_scales(points, 0., None, None))
        section = dict(fy_MPa=240., first_yield_moment_Nmm_per_MPa=500000., squash_load_N_per_MPa=4000.)
        self.assertEqual(pm.normalized_report_scales([], 0., section, None), (120., 960.))

    def test_pure_compression_does_not_assign_its_unlabelled_my_to_other_axes(self):
        points = [dict(bending_axis_deg=None, M_kNm=0., P_kN=100., My_kNm=100., Py_kN=200.),
                  dict(bending_axis_deg=90., M_kNm=75., P_kN=0., My_kNm=150., Py_kN=200.)]
        self.assertEqual(pm.normalized_report_scales(points, 90., None, None), (150., 200.))

    def test_dimensional_report_benchmarks_have_correct_orientations(self):
        import matplotlib.pyplot as plt
        reference = dict(schema='cufsm_builtup_reference_v1', group='builtup_4_pieces',
                         fy_MPa=240., AsFy_kN=1100., axes={
                             'x': dict(axis_deg=0., SFy_kNm=130., ZFy_kNm=190.),
                             'z': dict(axis_deg=90., SFy_kNm=140., ZFy_kNm=200.)})
        points = [dict(source='GMNIA', bending_axis_deg=0., M_kNm=50., P_kN=200.,
                       M_over_My=.5, P_over_Py=.2, My_kNm=100., Py_kN=1000.)]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(plt, 'close'):
            pm.plot_interaction(directory, points, {}, report_reference=reference)
            ax = plt.gcf().axes[0]
            lines = {line.get_label().split()[0]: line for line in ax.lines}
            self.assertEqual(list(lines['SFy'].get_xdata()), [130., 130.])
            self.assertEqual(list(lines['ZFy'].get_xdata()), [190., 190.])
            self.assertEqual(list(lines['AsFy'].get_ydata()), [1100., 1100.])
            self.assertGreater(ax.get_xlim()[1], 190.)
            self.assertGreater(ax.get_ylim()[1], 1100.)
            pm.plot_interaction(directory, points, {}, normalized=True, report_reference=reference)
            ax = plt.gcf().axes[0]
            lines = {line.get_label().split()[0]: line for line in ax.lines}
            self.assertEqual(list(lines['SFy'].get_xdata()), [1.3, 1.3])
            self.assertEqual(list(lines['ZFy'].get_xdata()), [1.9, 1.9])
            self.assertEqual(list(lines['AsFy'].get_ydata()), [1.1, 1.1])
        plt.close('all')

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



class ResponseSelectionTests(unittest.TestCase):
    def rows(self, loads):
        return [dict(frame=i, lpf=load, shortening_mm=float(i)) for i, load in enumerate(loads)]

    def test_verified_peak_and_incomplete_ascending_endpoint_remain_distinct(self):
        rows = self.rows([0., .4, 1., .6])
        selected = pm.select_response_point(rows, 'shortening_mm', True)
        self.assertEqual(selected['row']['frame'], 2)
        self.assertEqual(selected['selection_kind'], 'VERIFIED_PEAK')
        self.assertTrue(selected['capacity_verified'])
        observed=pm.select_response_point(rows, 'shortening_mm', False)
        self.assertEqual(observed['selection_kind'],'UNVERIFIED_OBSERVED_MAXIMUM')
        self.assertEqual(observed['row']['frame'],2)
        self.assertFalse(observed['capacity_verified'])
        ascending = pm.select_response_point(self.rows([0., .4, .9]), 'shortening_mm', False)
        self.assertEqual(ascending['selection_kind'], 'INCOMPLETE_ASCENDING_ENDPOINT')
        self.assertEqual(ascending['row']['frame'], 2)
        self.assertFalse(ascending['capacity_verified'])
        self.assertEqual(ascending['selection_policy'], 'response-v4')
        self.assertIsNone(pm.select_response_point(self.rows([0., .4, .9]), 'shortening_mm', False, False))

    def test_failed_descending_or_plateau_response_is_never_replaced_with_last_increment(self):
        for loads in ([0., .5, 1., .8], [0., .5, 1., 1.]):
            observed=pm.select_response_point(self.rows(loads), 'shortening_mm', False)
            self.assertEqual(observed['selection_kind'],'UNVERIFIED_OBSERVED_MAXIMUM')
            self.assertEqual(observed['row']['frame'],2)
            self.assertTrue(observed['provisional'])
            self.assertFalse(observed['capacity_verified'])

    def test_response_v1_retains_supported_knee_and_recovered_drop_onsets(self):
        import numpy as np
        knee = [min(x, .7)+.03*max(x-.7, 0.) for x in np.linspace(0., 2., 201)]
        drop = list(np.linspace(0., 1., 81))+list(np.linspace(.995, .85, 25))+list(np.linspace(.855, 1.1, 60))
        for values, kind, frame in ((knee, 'SIGNIFICANT_STIFFNESS_CHANGE', 70),
                                    (drop, 'SIGNIFICANT_LOAD_DROP', 80)):
            point = pm.select_response_point(self.rows(values), 'shortening_mm', False)
            self.assertEqual(point['selection_kind'], kind)
            self.assertEqual(point['row']['frame'], frame)
            self.assertTrue(point['provisional'])
            self.assertFalse(point['capacity_verified'])
            self.assertNotEqual(point['row']['frame'], len(values)-1)

    def test_imperfect_specimen_axes_can_include_equivalent_perfect_directions(self):
        segments = [(-1., -1., 1., -1.), (1., -1., 1., 1.),
                    (1., 1., -1., 1.), (-1., 1., -1., -1.)]
        axes, unused = pm.resolve_axes([0., 90., 180., 270.], segments, check_symmetry=False)
        self.assertEqual(axes, [0., 90., 180., 270.])
        with self.assertRaises(ValueError):
            pm.resolve_axes([0., 90.], segments)
        with self.assertRaises(ValueError):
            pm.resolve_axes([0., 360.], segments, check_symmetry=False)



class SofteningResponseSelectionTests(unittest.TestCase):
    def response(self, x, loads):
        return [dict(frame=i, lpf=float(load), shortening_mm=float(displacement))
                for i, (displacement, load) in enumerate(zip(x, loads))]

    def test_persistent_transition_into_softening_is_a_provisional_stiffness_change(self):
        import numpy as np
        x = np.linspace(0., 1.25, 201)
        loads = np.minimum(x, 1.)-.025*np.maximum(x-1., 0.)
        point = pm.select_response_point(self.response(x, loads), 'shortening_mm', False)
        self.assertIsNotNone(point)
        self.assertEqual(point['selection_kind'], 'SIGNIFICANT_STIFFNESS_CHANGE')
        self.assertEqual(point['row']['frame'], 160)
        self.assertLess(point['selection_diagnostics']['tangent_ratio'], 0.)
        self.assertTrue(point['provisional'])
        self.assertFalse(point['capacity_verified'])
        self.assertLess(point['row']['frame'], len(loads)-1)

    def test_one_increment_negative_tangent_is_not_a_stiffness_change(self):
        import numpy as np
        x = np.linspace(0., 1., 201)
        loads = x.copy()
        loads[175] -= .02
        point = pm.select_response_point(self.response(x, loads), 'shortening_mm', False)
        self.assertEqual(point['selection_kind'], 'INCOMPLETE_ASCENDING_ENDPOINT')
        self.assertEqual(point['row']['frame'], 200)

    def test_smooth_curvature_into_a_shallow_descending_branch_is_not_a_knee(self):
        import numpy as np
        x = np.linspace(0., 1.2, 201)
        loads = 2.*x-x*x
        point=pm.select_response_point(self.response(x, loads), 'shortening_mm', False)
        self.assertEqual(point['selection_kind'],'UNVERIFIED_OBSERVED_MAXIMUM')
        self.assertEqual(point['row']['frame'],int(np.argmax(loads)))
        self.assertFalse(point['capacity_verified'])
        self.assertIsNone(pm.select_response_point(self.response(x, loads), 'shortening_mm', False, False))



class IndependentResponseSelectionTests(unittest.TestCase):
    def response(self, axial_knee=.72, bending_knee=.5, ratio=.5):
        import numpy as np
        loads=np.linspace(0.,1.,201)
        def displacement(p,knee):
            return min(p,knee)+max(0.,p-knee)/ratio
        return [dict(frame=i,lpf=float(p),force_kN=100.*p,moment_kNm=10.*p,
                     shortening_mm=displacement(p,axial_knee),
                     rotation_mrad=displacement(p,bending_knee)) for i,p in enumerate(loads)]

    def test_earliest_bending_knee_selects_one_synchronized_frame_before_peak(self):
        rows=self.response()
        selected=pm.select_pm_response_point(rows,True)
        self.assertEqual(selected['selection_kind'],'SIGNIFICANT_STIFFNESS_CHANGE')
        self.assertEqual(selected['selection_response'],'rotation_mrad')
        self.assertLessEqual(abs(selected['row']['frame']-100),2)
        self.assertEqual(selected['row'],rows[selected['row']['frame']])
        self.assertFalse(selected['capacity_verified'])
        self.assertTrue(selected['provisional'])

    def test_axial_knee_can_precede_bending_and_is_scale_invariant(self):
        rows=self.response(.5,.72)
        expected=pm.select_pm_response_point(rows,True)
        for r in rows:
            r['shortening_mm']*=1e-4;r['rotation_mrad']*=1e4
        selected=pm.select_pm_response_point(rows,True)
        self.assertEqual(selected['selection_response'],'shortening_mm')
        self.assertEqual(selected['row']['frame'],expected['row']['frame'])
        self.assertLessEqual(abs(selected['row']['frame']-100),2)

    def test_unloaded_channel_is_ignored_in_both_pure_endpoints(self):
        for force,displacement,loaded in [('moment_kNm','rotation_mrad','shortening_mm'),
                                          ('force_kN','shortening_mm','rotation_mrad')]:
            rows=self.response(.55,.55)
            for r in rows:
                r[force]=0.;r[displacement]=1e-14*(r['frame']%7)
            selected=pm.select_pm_response_point(rows,True)
            self.assertEqual(selected['selection_response'],loaded)
            self.assertIn(displacement,selected['selection_diagnostics']['inactive_channels'])
            self.assertLess(selected['row']['frame'],150)

    def test_small_change_and_smooth_curvature_do_not_replace_peak(self):
        import numpy as np
        for variant in ('small','quadratic','single_dip'):
            rows=self.response(.5,.5,.85)
            if variant!='small':
                for i,r in enumerate(rows):
                    x=i/200.;p=2*x-x*x if variant=='quadratic' else x-(.02 if i==170 else 0.)
                    r.update(lpf=p,force_kN=100*p,moment_kNm=10*p,
                             shortening_mm=x,rotation_mrad=10*x)
            selected=pm.select_pm_response_point(rows,True)
            self.assertEqual(selected['selection_kind'],'VERIFIED_PEAK',variant)
            self.assertEqual(selected['row']['frame'],200)

    def test_queue_uses_the_same_frame_for_all_geometries_cases_and_terminal_states(self):
        import abaqus_step5_queue as queue
        rows=self.response()
        for variant in ('HSS','No plates','With plates'):
            for case in ('PERFECT','LG_mp'):
                failed=case=='LG_mp'
                result=dict(model='PM_A000_P30_'+case,
                    capacity_verified=not failed,analysis_state='FAILED' if failed else 'TERMINATED',
                    outcome='SOLVER_ERROR' if failed else 'CONTROLLED_TERMINATION',
                    path_review_required=True,postpeak_path_check=dict(status='RETRACE_SUSPECTED'))
                info=dict(load_case=dict(type='combined'),source_case=case,variant=variant,
                          reference_axial_force_N=100000.,reference_moment_Nmm=10000000.,
                          settings=dict(fy=240.))
                queue.select_pm_result(result,info,rows)
                point=result['pm_point'];frame=point['frame']
                self.assertLessEqual(abs(frame-100),2)
                self.assertEqual(point['P_kN'],rows[frame]['force_kN'])
                self.assertEqual(point['M_kNm'],rows[frame]['moment_kNm'])
                self.assertEqual(point['selection_response'],'rotation_mrad')
                self.assertEqual(point['selection_policy'],'response-v4-dual')
                self.assertFalse(point['capacity_verified'])
                self.assertTrue(result['path_review_required'])
                self.assertEqual(result['capacity_verified'],not failed)
                self.assertIn('postpeak_path_check',json.loads(point['selection_diagnostics']))

    def test_small_startup_reversal_does_not_hide_a_later_bending_knee(self):
        rows=self.response(.85,.5)
        rows[1]['rotation_mrad']=-1e-7
        point=pm.select_pm_response_point(rows,True)
        self.assertEqual(point['selection_response'],'rotation_mrad')
        self.assertLessEqual(abs(point['row']['frame']-100),2)

    def test_later_sign_crossing_does_not_reverse_the_initial_branch_direction(self):
        rows=self.response()
        for i,r in enumerate(rows):
            r['force_kN']=0.
            if i>160:r['rotation_mrad']=rows[160]['rotation_mrad']-.08*(i-160)
        point=pm.select_pm_response_point(rows,True)
        self.assertEqual(point['selection_kind'],'SIGNIFICANT_STIFFNESS_CHANGE')
        self.assertEqual(point['selection_response'],'rotation_mrad')
        self.assertLessEqual(abs(point['row']['frame']-100),2)

    def test_simultaneous_events_retain_a_real_csv_key_and_both_triggers(self):
        rows=self.response(.5,.5)
        point=pm.select_pm_response_point(rows,True)
        self.assertIn(point['selection_response'],rows[0])
        self.assertEqual(point['selection_diagnostics']['trigger_channels'],
                         ['shortening_mm','rotation_mrad'])

    def test_recovered_drop_is_not_lost_when_rotation_reverses(self):
        loads=[i*.025 for i in range(37)]+[.89,.87,.85,.83,.81,.82,.84,.86,.89,.92,.96,1.]
        rows=[dict(frame=i,lpf=p,force_kN=100*p,moment_kNm=10*p,
                   shortening_mm=i*.03,rotation_mrad=(p if i<42 else .81+(i-41)*.02))
              for i,p in enumerate(loads)]
        point=pm.select_pm_response_point(rows,False)
        self.assertEqual(point['selection_kind'],'SIGNIFICANT_LOAD_DROP')
        self.assertEqual(point['row']['frame'],36)
        self.assertFalse(point['capacity_verified'])

    def test_live_response_is_not_selected_and_retracing_tail_does_not_move_onset(self):
        rows=self.response()
        self.assertIsNone(pm.select_pm_response_point(rows,False,False))
        first=pm.select_pm_response_point(rows,True)
        tail=[dict(r,frame=201+i) for i,r in enumerate(reversed(rows[130:190]))]
        selected=pm.select_pm_response_point(rows+tail,True)
        self.assertEqual(selected['row']['frame'],first['row']['frame'])

if __name__ == '__main__':
    unittest.main()
