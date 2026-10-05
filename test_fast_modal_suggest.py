import unittest
import numpy as np
from types import SimpleNamespace as NS
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
from unittest.mock import patch
import abaqus_fast_modal_suggest as fast
from abaqus_modal_report import SectionProjector


def geometry():
    # Four independent L pieces with physical panel endpoints and interior nodes.
    base = np.array([[0., 0.], [10., 0.], [20., 0.], [20., 10.], [20., 20.]])
    xy = np.vstack([base+[100*(i%2), 100*(i//2)] for i in range(4)])
    edges = [(5*i+j, 5*i+j+1) for i in range(4) for j in range(4)]
    pieces = ['P%d'%i for i in range(4) for j in range(5)]
    source = {'P%d'%i: [list(xy[5*i+j])+list(xy[5*i+j+1]) for j in range(4)] for i in range(4)}
    return xy, edges, pieces, source


class FastSuggestTests(unittest.TestCase):
    def setUp(self):
        self.xy, edges, pieces, source = geometry()
        self.fit = SectionProjector(self.xy, edges, pieces, np.ones(len(pieces)),
                                    physical_segments=source, compute_bases=False)
        self.kernel = fast.ScreeningKernel(self.fit)

    def test_whole_section_translation_and_twist_are_global(self):
        for u in (np.tile([2., -3.], (20, 1)), np.c_[-self.xy[:, 1], self.xy[:, 0]]):
            result = self.kernel.evaluate(u.reshape(1, -1))
            self.assertAlmostEqual(result['global_percent'], 100.)
            self.assertEqual(fast.classify(result, True)['family'], 'GLOBAL')

    def test_local_panel_bending_not_consumed_by_distortional(self):
        u = np.zeros((20, 2)); u[1, 1] = 1.
        result = self.kernel.evaluate(u.reshape(1, -1))
        self.assertAlmostEqual(result['local_percent'], 100.)
        self.assertEqual(fast.classify(result, True)['family'], 'LOCAL')

    def test_relative_piece_motion_is_diagnostic(self):
        u = np.zeros((20, 2)); u[:5, 0] = 1.
        result = self.kernel.evaluate(u.reshape(1, -1))
        self.assertGreater(result['assembly_percent'], 25.)
        self.assertEqual(fast.classify(result, True)['status'], 'UNRESOLVED')

    def test_inextensional_panel_rotation_is_distortional(self):
        u = np.zeros((20, 2))
        for i in range(4):
            # Change the included panel angle without chord extension.
            u[5*i, 1]=2.; u[5*i+1, 1]=1.
            u[5*i+3, 0]=-1.; u[5*i+4, 0]=-2.
            # Remove each piece's rigid anchor motion; retain only distortion.
            p=self.xy[5*i:5*i+5]; c=p-p.mean(axis=0)
            rigid=np.zeros((5,2,3)); rigid[:,0,0]=1.; rigid[:,1,1]=1.
            rigid[:,0,2]=-c[:,1]; rigid[:,1,2]=c[:,0]
            a=np.linalg.lstsq(rigid[[0,2,4]].reshape(-1,3),
                             u[5*i:5*i+5][[0,2,4]].ravel(),rcond=None)[0]
            u[5*i:5*i+5]-=rigid@a
        result = self.kernel.evaluate(u.reshape(1, -1))
        self.assertAlmostEqual(result['distortional_percent'], 100.)
        self.assertEqual(fast.classify(result, True)['family'], 'DISTORTIONAL')

    def test_kernel_matches_existing_geometry_diagnostics(self):
        u = np.random.default_rng(15).normal(size=(7, 40))
        expected = self.fit.component_diagnostics(u)
        actual = self.kernel.evaluate(u)
        for key in ('global_percent', 'local_percent', 'distortional_percent',
                    'assembly_percent', 'other_percent', 'component_norm_sum_over_input'):
            self.assertAlmostEqual(actual[key], expected[key], places=9)

    def test_sign_and_amplitude_do_not_change_classification(self):
        u = np.random.default_rng(16).normal(size=(7, 40))
        a, b = self.kernel.evaluate(u), self.kernel.evaluate(-1e-12*u)
        for key in ('global_percent','local_percent','distortional_percent'):
            self.assertAlmostEqual(a[key], b[key], places=8)

    def test_zero_and_nonfinite_fields_rejected(self):
        for value in (0., float('nan')):
            with self.assertRaises(ValueError):
                self.kernel.evaluate(np.full((1, 40), value))

    def test_missing_geometry_cannot_confirm_local(self):
        split = dict(local_percent=100., distortional_percent=0., global_percent=0.,
                     assembly_percent=0., other_percent=0., reconstruction_relative_error=0.)
        self.assertEqual(fast.classify(split, False)['status'], 'UNRESOLVED')

    def test_repeated_cluster_disagreement_blocks_suggestion(self):
        rows = [dict(mode=1,eigenvalue=100.,family='LOCAL',status='CANDIDATE',flags=[]),
                dict(mode=2,eigenvalue=100.001,family='DISTORTIONAL',status='CANDIDATE',flags=[])]
        fast.flag_clusters(rows)
        self.assertTrue(all(r['status']=='UNRESOLVED' for r in rows))

    def test_longitudinal_weights_sum_to_length(self):
        w = fast.longitudinal_weights([0., 1., 3., 10.])
        self.assertAlmostEqual(w.sum(), 10.)
        with self.assertRaises(ValueError): fast.longitudinal_weights([0., 1., 1.])

    def test_workers_use_available_cpus_without_fixed_ceiling(self):
        self.assertEqual(fast.worker_count(24, 100, 1000000, 1000), 24)
        self.assertEqual(fast.worker_count(24, 3, 1000000, 1000), 3)
        self.assertEqual(fast.worker_count(24, 100, 2500, 1000), 2)

    def test_suggest_cli_preserves_existing_command(self):
        import abaqus_step4_imperfections as step4
        args=step4.parse_arguments(['--run-dir','.', '--suggest'])
        self.assertEqual(args.suggest_source, 'odb')
        self.assertEqual(args.suggest_cpus, 'auto')

    def test_curved_reference_rigid_rotation_not_local(self):
        x=np.linspace(0.,180.,31)
        xy=np.c_[x,6.*np.sin(2*np.pi*x/180.)]
        source={'P':np.c_[xy[:-1],xy[1:]].tolist()}
        p=SectionProjector(xy,[(i,i+1) for i in range(30)],['P']*31,np.ones(31),
                           physical_segments=source,compute_bases=False)
        u=np.c_[-xy[:,1],xy[:,0]]+[2.,-3.]
        result=fast.ScreeningKernel(p).evaluate(u.reshape(1,-1))
        self.assertLess(result['local_percent'],1e-15)
        self.assertAlmostEqual(result['global_percent'],100.)

    def test_fast_and_full_section_operators_agree(self):
        xy,edges,pieces,source=geometry()
        full=SectionProjector(xy,edges,pieces,np.ones(len(xy)),physical_segments=source)
        for name in ('plocal','pglobal','passembly','pdist','pother'):
            np.testing.assert_allclose(getattr(self.fit,name),getattr(full,name),atol=1e-12)

    def test_mapping_uses_local_mesh_spacing_not_float_tolerance(self):
        metadata=dict(geometry_supported=True,source_wall_layouts={'P1':dict(walls=[{}])},
                      physical_wall_count=1,wall_endpoint_mapping=[
            dict(piece='P1',error_mm=5.212650233111133,adjacent_edge_mm=20.,
                 piece_end=False)])
        result=fast.wall_mapping_status(metadata,['P1'],.0036)
        self.assertTrue(result['supported'])
        self.assertTrue(result['mesh_snapped'])
        self.assertAlmostEqual(result['checks'][0]['allowed_error_mm'],10.036)

    def test_mapping_rejects_large_or_free_end_offsets(self):
        for error,piece_end in ((11.,False),(5.2,True)):
            metadata=dict(geometry_supported=True,source_wall_layouts={'P1':dict(walls=[{}])},
                          physical_wall_count=1,wall_endpoint_mapping=[
                dict(piece='P1',error_mm=error,adjacent_edge_mm=20.,piece_end=piece_end)])
            self.assertFalse(fast.wall_mapping_status(metadata,['P1'],.0036)['supported'])

    def test_missing_wall_mapping_not_accepted(self):
        metadata=dict(geometry_supported=True,source_wall_layouts={'P1':dict(walls=[{},{}])},
                      physical_wall_count=1,wall_endpoint_mapping=[])
        self.assertFalse(fast.wall_mapping_status(metadata,['P1'],.0036)['supported'])

    def test_coarse_curved_mesh_accepts_between_node_boundary_but_not_shifted_source(self):
        a=np.linspace(-np.pi/2.,0.,25)
        source_xy=np.vstack((np.c_[np.linspace(0.,100.,31),np.zeros(31)],
            np.c_[100.+5.*np.cos(a[1:]),5.+5.*np.sin(a[1:])],
            np.c_[np.full(30,105.),np.linspace(5.,105.,31)[1:]]))
        xy=np.array([[0.,0.],[25.,0.],[50.,0.],[75.,0.],[99.,0.],
                     [102.5,.669872981],[104.330127019,2.5],[105.,5.],
                     [105.,30.],[105.,55.],[105.,80.],[105.,105.]])
        def mapping(points):
            p=SectionProjector(xy,[(i,i+1) for i in range(len(xy)-1)],['P1']*len(xy),
                               np.ones(len(xy)),physical_segments={'P1':np.c_[points[:-1],points[1:]].tolist()},
                               compute_bases=False)
            return p,fast.wall_mapping_status(p.metadata,['P1'],.0036)
        p,result=mapping(source_xy)
        self.assertGreater(p.metadata['maximum_wall_mapping_error_mm'],.036)
        self.assertTrue(result['supported']);self.assertTrue(result['mesh_snapped'])
        unused,wrong=mapping(source_xy+[1.,0.])
        self.assertFalse(wrong['supported'])

    def test_explicit_odb_source_configures_startup_threads(self):
        import runpy
        import abaqus_step4_imperfections as step4
        from runtime_resources import configure_threads
        with patch.object(sys,'argv',['script','--suggest','--suggest-source','odb']),\
             patch('runtime_resources.configure_threads',wraps=configure_threads) as configure:
            runpy.run_path(step4.__file__,run_name='startup_check')
        configure.assert_called_once_with(1)

    def test_gpu_selection_uses_parallel_cpu_batch_comparison(self):
        class FakeCupy:
            cuda=NS(Device=lambda i:contextlib.nullcontext(),
                    Stream=NS(null=NS(synchronize=lambda:None)),
                    runtime=NS(getDeviceCount=lambda:3,
                               memGetInfo=lambda:(10**9,10**9)))
            def __getattr__(self,name):
                return (lambda x:x) if name=='asnumpy' else getattr(np,name)
        cp=FakeCupy(); field=np.ones((2,40))
        with patch.dict(sys.modules,{'cupy':cp}),\
             patch.object(fast,'benchmark_batch',return_value=0.) as bench:
            backend,devices,info=fast.select_backend(self.kernel,field,3,24,100,100)
        self.assertIs(backend,np);self.assertEqual(devices,())
        self.assertEqual(bench.call_args.args[1:],(24,48))
        self.assertEqual(info['reason'],'PARALLEL_CPU_MEASURED_FASTER')
        with patch.dict(sys.modules,{'cupy':cp}),\
             patch.object(fast,'benchmark_batch',return_value=1e6):
            backend,devices,info=fast.select_backend(self.kernel,field,3,24,100,100)
        self.assertIs(backend,cp);self.assertEqual(devices,(0,1,2))
        self.assertEqual(info['reason'],'MEASURED_FASTER_BATCH_THROUGHPUT')


def fake_odb():
    xy, edges, pieces, source=geometry()
    source={'P%d'%(i+1):source['P%d'%i] for i in range(4)}
    instances={}
    for i in range(4):
        nodes=[NS(label=5*k+j+1,coordinates=tuple(xy[5*i+j])+ (float(z),))
               for k,z in enumerate(np.linspace(0.,100.,9)) for j in range(5)]
        elements=[]
        for k in range(8):
            for j in range(4):
                elements.append(NS(type='S4R',connectivity=(5*k+j+1,5*k+j+2,5*(k+1)+j+2,5*(k+1)+j+1)))
        instances['P%d'%(i+1)]=NS(name='P%d'%(i+1),nodes=nodes,elements=elements)
    frames=[]
    for mode in (1,2):
        values=[]
        for name,instance in instances.items():
            for node in instance.nodes:
                w=np.sin(np.pi*node.coordinates[2]/100.)
                u=(w,0.,0.) if mode==1 else (0.,w if (node.label-1)%5==1 else 0.,0.)
                values.append(NS(instance=instance,nodeLabel=node.label,precision='SINGLE_PRECISION',
                                 localCoordSystem=None,data=u))
        frames.append(NS(description='Mode %d: EigenValue = %d'%(mode,100*mode),
                         frameValue=float(100*mode),fieldOutputs={'U':NS(values=values)}))
    odb=NS(rootAssembly=NS(instances=instances),steps={'Buckle':NS(frames=frames)},closed=False)
    def close(): odb.closed=True
    odb.close=close
    return odb,source


class IntegrationTests(unittest.TestCase):
    def test_same_command_reads_odb_writes_results_and_uses_cache(self):
        import abaqus_step4_imperfections as step4
        from runtime_resources import ResourceInventory
        odb,source=fake_odb()
        with tempfile.TemporaryDirectory() as directory:
            path=os.path.join(directory,'column.odb')
            with open(path,'wb') as stream: stream.write(b'fake-odb')
            with open(os.path.join(directory,'column.cae'),'wb') as stream: stream.write(b'fake-cae')
            build=os.path.join(directory,'column_build.json')
            inputs=dict(section_segments=source,connection_model='BEAM_MPC',
                        shell_contact='GENERAL_STANDARD_HARD_FRICTIONLESS',
                        expected_links=4,bolts_per_seam=1,thickness_mm=2.)
            build_record=dict(job_name='column',element_type='S4R',rigid_links=4,
                              boundary_conditions=9,loads=8,reference_stress_MPa=1.,
                              length_mm=100.,target_mesh_mm=12.5,nodes=180,elements=128,
                              source_inputs=inputs)
            with open(build,'w') as stream: json.dump(build_record,stream)
            with open(os.path.join(directory,'pipeline_status.json'),'w') as stream:
                json.dump(dict(status='COMPLETED',build=build_record,source_inputs=inputs),stream)
            args=step4.parse_arguments(['--run-dir',directory,'--suggest','--suggest-gpus','0'])
            inventory=ResourceInventory(2,2,100000000,100000000,[])
            reads=[]
            def open_odb(**kwargs):
                self.assertTrue(kwargs['readOnly']); reads.append(threading.get_ident())
                return odb
            with patch.dict(sys.modules,{'odbAccess':NS(openOdb=open_odb)}),\
                 patch.object(fast,'detect_resources',return_value=inventory),\
                 patch.object(fast,'available_memory',return_value=(100000000,100000000)),\
                 contextlib.redirect_stdout(io.StringIO()):
                first=step4.main(['--run-dir',directory,'--suggest','--suggest-gpus','0'])
                self.assertTrue(odb.closed)
                self.assertEqual([r['family'] for r in first['modes']],['GLOBAL','LOCAL'])
                self.assertEqual([r['status'] for r in first['modes']],['CANDIDATE']*2)
                self.assertEqual(reads,[threading.get_ident()])
                self.assertEqual(first['reference_pipeline']['job_name'],'column')
                with self.assertRaisesRegex(ValueError,'exact artifact'):
                    step4.main(['--run-dir',directory,'--suggest','--odb',os.path.join(directory,'wrong.odb')])
                second=step4.main(['--run-dir',directory,'--suggest','--suggest-gpus','0'])
                self.assertEqual(second['signature'],first['signature'])
                self.assertEqual(len(reads),1)
                with open(build,'a') as stream: stream.write(' ')
                step4.main(['--run-dir',directory,'--suggest','--suggest-gpus','0'])
                self.assertEqual(len(reads),2)
                # Extra historical build metadata cannot replace the validated
                # pipeline's physical walls or disable their classification.
                with open(os.path.join(directory,'unrelated_build.json'),'w') as stream:
                    json.dump(dict(source_inputs={}),stream)
                third=step4.main(['--run-dir',directory,'--suggest','--suggest-gpus','0'])
                self.assertEqual(third['modes'][1]['family'],'LOCAL')
                self.assertEqual(third['modes'][1]['status'],'CANDIDATE')
                self.assertEqual(len(reads),3)
            self.assertTrue(os.path.isfile(first['report_path']))
            self.assertTrue(os.path.isfile(os.path.splitext(first['report_path'])[0]+'.csv'))

    def test_locked_odb_is_rejected(self):
        import abaqus_step4_imperfections as step4
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.odb','.lck'):
                with open(os.path.join(directory,'column'+suffix),'w') as stream: stream.write('x')
            args=step4.parse_arguments(['--run-dir',directory,'--suggest'])
            with self.assertRaisesRegex(ValueError,'locked'): fast.suggest(args)

    def test_failed_csv_write_cannot_publish_cache_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path=os.path.join(directory,'result.json')
            os.mkdir(os.path.join(directory,'result.csv'))
            with self.assertRaises(OSError):
                fast.write_report(path,dict(modes=[]))
            self.assertFalse(os.path.exists(path))


if __name__ == '__main__': unittest.main()
