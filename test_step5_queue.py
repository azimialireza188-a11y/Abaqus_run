import os
import tempfile
import unittest
import io
import abaqus_step5_gmnia as build
import importlib.util
from unittest import mock
from types import SimpleNamespace


class QueueTests(unittest.TestCase):
    def test_completion_combines_current_job_evidence_instead_of_short_circuiting_sta(self):
        from pathlib import Path
        q=self.queue()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            root=Path(d); name='STEP5_LG_mp_FY240'
            (root/(name+'.sta')).write_text('RIKS increment table, no terminal footer yet')
            (root/(name+'.log')).write_text('Abaqus JOB '+name+' COMPLETED\n')
            self.assertTrue(q.analysis_finished(d,name))
            (root/(name+'.log')).unlink()
            (root/(name+'_queue_solver.log')).write_text('10/11/2026 15:00:00 Abaqus JOB '+name+' COMPLETED\n')
            self.assertTrue(q.analysis_finished(d,name))
            (root/(name+'_queue_solver.log')).write_text('Abaqus JOB STEP5_PERFECT_FY240 COMPLETED\nSent Terminate message to Abaqus job STEP5_PERFECT_FY240\n')
            self.assertFalse(q.analysis_finished(d,name))
            (root/(name+'.sta')).write_text('THE ANALYSIS HAS NOT BEEN COMPLETED')
            (root/(name+'.log')).write_text('Abaqus JOB '+name+' COMPLETED\n')
            self.assertFalse(q.analysis_finished(d,name)) # contradictory terminal evidence is not success
            (root/(name+'.sta')).write_text('RIKS increment table')
            (root/(name+'.msg')).write_text('***ERROR: Too many attempts made for this increment.\n')
            self.assertEqual(q.completion_details(d,name)['analysis_state'],'FAILED')
            self.assertIn('TOO MANY ATTEMPTS',q.completion_details(d,name)['completion_diagnostics'][0])
            (root/(name+'.msg')).write_text('***ERROR: Process terminated by external request (SIGTERM or SIGINT received).\n')
            (root/(name+'.log')).write_text('Abaqus/Analysis exited with errors\n')
            details=q.completion_details(d,name)
            self.assertEqual(details['analysis_state'],'TERMINATED')
            self.assertEqual(q.outcome(dict(details,solver_exit_code=1,termination_requested=True),True),'CONTROLLED_TERMINATION')

    def test_completion_failure_is_not_validated_by_exit_zero_or_a_postpeak_crossing(self):
        q=self.queue()
        result=dict(solver_exit_code=0,analysis_completed=False,analysis_state='FAILED',partial=False)
        self.assertEqual(q.outcome(result,True),'SOLVER_ERROR')
        self.assertEqual(q.outcome(dict(result,analysis_state='TERMINATED'),False),'INCOMPLETE_ANALYSIS')
        self.assertEqual(q.outcome(dict(result,analysis_state='UNKNOWN'),False),'SOLVER_STATUS_UNKNOWN')
        self.assertEqual(q.outcome(dict(result,analysis_state='TERMINATED',termination_requested=True),True),'CONTROLLED_TERMINATION')

    def test_finished_solve_without_postpeak_is_reusable_but_not_capacity(self):
        import csv, json
        q = self.queue()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            name = 'STEP5_TEST'
            info = dict(reference_force_N_per_end=1000.)
            result = dict(model=name, outcome='INCOMPLETE_ANALYSIS', analysis_completed=False, criterion_reached=False,
                          solver_exit_code=0, partial=False, force_kN=1., Pu_over_Py=.9,
                          pm_point=dict(P_kN=1.))
            stem = os.path.join(d, name+'_force_displacement')
            with open(os.path.join(d, name+'.odb'), 'w') as f: f.write('synthetic ODB')
            with open(os.path.join(d, name+'.sta'), 'w') as f:
                f.write('THE ANALYSIS HAS COMPLETED SUCCESSFULLY')
            with open(stem+'.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'lpf', 'shortening_mm', 'force_kN'])
                writer.writerows([(0,0,0,0), (1,1,1,1), (2,.92,4,.92)])
            with open(stem+'.json', 'w') as f:
                json.dump(dict(model=name, step='GMNIA', frames=3, partial=False), f)
            with open(os.path.join(d, 'STEP5_queue_status.json'), 'w') as f:
                json.dump([result], f)
            reused, points = q.completed_curve(d, name, info)
            self.assertTrue(q.finished_result(reused))
            self.assertEqual(reused['outcome'],'CRITERION_NOT_REACHED')
            self.assertEqual(reused['previous_outcome'],'INCOMPLETE_ANALYSIS')
            self.assertFalse(reused['capacity_verified'])
            self.assertFalse(reused['criterion_reached'])
            self.assertNotIn('Pu_over_Py', reused)
            self.assertIsNone(reused.get('pm_point'))
            self.assertEqual(points[-1], (4., .92))
            # A successful launcher exit is insufficient when the analysis did not finish.
            with open(os.path.join(d, name+'.sta'), 'w') as f: f.write('ANALYSIS NOT COMPLETED')
            self.assertIsNone(q.completed_curve(d, name, info))
            with open(stem+'.json','w') as f:
                stat=os.stat(os.path.join(d,name+'.odb'))
                json.dump(dict(model=name,step='GMNIA',frames=3,partial=False,
                    odb_job_status='JOB_STATUS_COMPLETED_SUCCESSFULLY',source_odb_size=stat.st_size,
                    source_odb_mtime_ns=stat.st_mtime_ns),f)
            with open(os.path.join(d,name+'.sta'),'w') as f: f.write('RIKS increment table')
            self.assertIsNotNone(q.completed_curve(d,name,info)) # bound ODB diagnostic is sufficient
            with open(os.path.join(d,name+'.sta'),'w') as f: f.write('THE ANALYSIS HAS NOT BEEN COMPLETED')
            self.assertIsNone(q.completed_curve(d,name,info)) # explicit negative evidence still wins
            self.assertFalse(q.finished_result(dict(result, analysis_completed=True, partial=True)))
            self.assertFalse(q.finished_result(dict(result, analysis_completed=True, solver_exit_code=1)))

    def test_queue_finished_without_crossing_and_stopped_curve_continue_without_capacity(self):
        import csv, json
        q = self.queue()
        for finished in (True, False):
            with self.subTest(finished=finished), tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
                name = 'STEP5_TEST'
                for suffix in ('.inp','.odb'):
                    with open(os.path.join(d,name+suffix),'w') as f: f.write('synthetic')
                with open(os.path.join(d,name+'.sta'),'w') as f:
                    f.write('RIKS increment table' if finished else 'ANALYSIS ABORTED')
                path = os.path.join(d,name+'_force_displacement.csv')
                with open(path,'w',newline='') as f:
                    writer=csv.writer(f);writer.writerow(['shortening_mm','force_kN','lpf'])
                    writer.writerows([(0,0,0),(1,1,1),(4,.92,.92)])
                summary=dict(csv=path,peak=dict(force_kN=1.,shortening_mm=1.,lpf=1.),partial=False)
                info=dict(stage=5,settings=dict(field_frequency=1,fy=240),
                          reference_force_N_per_end=1000.,end_area_mm2=1000./240.)
                args=SimpleNamespace(run_dir=d,jobs=[name],cae='synthetic.cae',cpus=1,gpus=0,
                    extract_only=True,repair_only=False,ratio=.7,stop_method='monitor')
                def extracted(unused):
                    if finished:
                        with open(os.path.join(d,name+'.log'),'w') as f:
                            f.write('Abaqus JOB '+name+' COMPLETED\n')
                    return summary
                with mock.patch.object(q,'metadata',return_value={name:info}), \
                        mock.patch.object(q.resources,'detect_resources'), \
                        mock.patch.object(q.resources,'resolve_solver_policy',return_value=SimpleNamespace(cpus=1,gpus=0)), \
                        mock.patch.object(q.fd,'extract',side_effect=extracted), \
                        mock.patch.object(q,'plot_png'), mock.patch.object(q,'write_comparison'), \
                        mock.patch.object(q,'report') as report:
                    if finished:
                        q.run(args)
                        self.assertTrue(any(c.args[0]=='POSTPEAK NOT VERIFIED' for c in report.call_args_list))
                    else:
                        q.run(args)
                        self.assertTrue(any(c.args[0]=='CONTINUE AFTER STOP' for c in report.call_args_list))
                with open(os.path.join(d,'STEP5_queue_status.json')) as f: result=json.load(f)[0]
                self.assertFalse(result['capacity_verified'])
                self.assertNotIn('Pu_over_Py',result)
                self.assertEqual(result['force_kN'],1.)

    def test_completed_curve_is_reused_without_opening_odb_and_rejects_stale_csv(self):
        import json,csv
        q=self.queue()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            name='STEP5_TEST'
            info=dict(reference_force_N_per_end=1000.)
            result=dict(model=name,outcome='CRITERION_REACHED',criterion_reached=True,
                        solver_exit_code=0,partial=False,force_kN=1.)
            stem=os.path.join(d,name+'_force_displacement')
            with open(os.path.join(d,name+'.odb'),'w') as f:f.write('not a real ODB')
            with open(stem+'.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['frame','lpf','shortening_mm','force_kN'])
                writer.writeheader()
                writer.writerows([dict(frame=i,lpf=y,shortening_mm=x,force_kN=y)
                                  for i,(x,y) in enumerate([(0.,0.),(1.,1.),(.9,.6)])])
            with open(stem+'.json','w') as f:
                json.dump(dict(model=name,step='GMNIA',frames=3,partial=False),f)
            with open(os.path.join(d,'STEP5_queue_status.json'),'w') as f:json.dump([result],f)
            reused,points=q.completed_curve(d,name,info,.7)
            self.assertEqual(reused['outcome'],'CRITERION_REACHED')
            self.assertEqual(points,[(0.,0.),(1.,1.),(.9,.6)]) # preserve Riks path, not sorted by x
            os.utime(os.path.join(d,name+'.odb'),(2000000000,2000000000))
            with self.assertRaisesRegex(ValueError,'(?i)stale'):
                q.completed_curve(d,name,info,.7)

    def test_family_styles_are_paired_and_svg_legend_is_not_clipped(self):
        q=self.queue()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            path=os.path.join(d,'families.svg')
            curves=[('family_%d %s'%(i,k),[(0.,0.),(1.,2.)])
                    for i in range(1,11) for k in ('PERFECT','LG_mp')]
            styles=[dict(color='#%06x'%(i+1),dash='8 5' if j else '')
                    for i in range(10) for j in range(2)]
            q.plot_svg(path,curves,[],'Families',styles=styles)
            import xml.etree.ElementTree as ET
            root=ET.parse(path).getroot()
            polys=root.findall('.//{http://www.w3.org/2000/svg}polyline')
            self.assertEqual(len(polys),20)
            for i in range(0,20,2):
                self.assertEqual(polys[i].get('stroke'),polys[i+1].get('stroke'))
                self.assertFalse(polys[i].get('stroke-dasharray'))
                self.assertEqual(polys[i+1].get('stroke-dasharray'),'8 5')
            self.assertGreater(float(root.get('height')),900)

    def test_saved_stopped_curve_is_reused_only_with_incomplete_opt_in(self):
        import csv, json
        q = self.queue()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            name = 'STEP5_TEST'
            info = dict(reference_force_N_per_end=1000.)
            record = dict(model=name, outcome='SOLVER_ERROR', criterion_reached=False,
                          solver_exit_code=1, partial=False, force_kN=.9)
            stem = os.path.join(d, name+'_force_displacement')
            with open(os.path.join(d, name+'.odb'), 'w') as stream:
                stream.write('fake ODB; must never open')
            with open(os.path.join(d, name+'.sta'), 'w') as stream:
                stream.write('ANALYSIS ABORTED')
            with open(stem+'.csv', 'w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['frame', 'shortening_mm', 'force_kN', 'lpf'])
                writer.writerows([(0, 0., 0., 0.), (1, 1., .5, .5), (2, 2., .9, .9)])
            with open(stem+'.json', 'w') as stream:
                json.dump(dict(model=name, step='GMNIA', frames=3, partial=False), stream)
            with open(os.path.join(d, 'STEP5_queue_status.json'), 'w') as stream:
                json.dump([record], stream)
            with mock.patch.object(q.fd, 'extract', side_effect=AssertionError('no extraction')):
                self.assertIsNone(q.completed_curve(d, name, info))
                result, curve = q.completed_curve(d, name, info, allow_incomplete=True, record=record)
            self.assertEqual(curve[-1], (2., .9))
            self.assertTrue(q.processed_result(result))
            self.assertFalse(q.finished_result(result))
            self.assertFalse(result['capacity_verified'])
            with open(stem+'.csv', 'a') as stream:
                stream.write('1,3,1,1\n')
            with self.assertRaisesRegex(ValueError, 'summary/CSV mismatch'):
                q.completed_curve(d, name, info, allow_incomplete=True, record=record)

    def test_pm_selection_keeps_original_peak_and_solver_outcome(self):
        import numpy as np
        q = self.queue()
        info = dict(load_case=dict(type='combined'), settings=dict(fy=240.), source_case='PERFECT',
                    reference_axial_force_N=1000., reference_moment_Nmm=2e6,
                    section_for_PM=dict(area_mm2=10., first_yield_moment_Nmm=4e6))
        loads = [min(x, .7)+.03*max(x-.7, 0.) for x in np.linspace(0., 2., 201)]
        rows = [dict(frame=i, lpf=float(value), extreme_fibre_shortening_mm=float(i),
                     force_kN=float(value), moment_kNm=2.*float(value)) for i, value in enumerate(loads)]
        result = dict(model='PM_A000_P45_PERFECT', outcome='SOLVER_ERROR', analysis_state='FAILED',
                      capacity_verified=False, force_kN=max(loads), moment_kNm=2.*max(loads))
        observed_peak = result['force_kN']
        q.select_pm_result(result, info, rows)
        self.assertEqual(result['outcome'], 'SOLVER_ERROR')
        self.assertEqual(result['force_kN'], observed_peak)
        self.assertFalse(result['capacity_verified'])
        self.assertEqual(result['pm_selection_kind'], 'SIGNIFICANT_STIFFNESS_CHANGE')
        self.assertAlmostEqual(result['pm_point']['P_kN'], .7)
        self.assertAlmostEqual(result['pm_point']['M_kNm'], 1.4)
        self.assertTrue(result['pm_point']['provisional'])
        rows[70]['moment_kNm'] = 99.
        with self.assertRaisesRegex(ValueError, 'synchronized LPF'):
            q.select_pm_result(result, info, rows)

    def test_zero_gpu_override_replaces_stale_environment(self):
        self.assertIn('gpus=0', self.queue().job_command('STEP5_TEST', 8, 0, 'monitor'))

    def test_progress_is_mirrored_to_external_log(self):
        import abaqus_step5_progress as progress
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'progress.log')
            with mock.patch.dict(os.environ, {'STEP5_PROGRESS_LOG': path}), \
                    mock.patch.object(progress.sys, 'stderr', io.StringIO()):
                progress.report('TEST', 'hello')
            with open(path, encoding='utf-8') as stream:
                text = stream.read()
            self.assertIn('[TEST] hello', text)

    def test_running_solver_reports_progress_even_before_odb_exists(self):
        q = self.queue()
        with tempfile.TemporaryDirectory() as d:
            process = mock.Mock(returncode=0)
            process.poll.side_effect = [None, None, None, 0]
            output = io.StringIO()
            args = SimpleNamespace(run_dir=d, stop_method='monitor', ratio=.7,
                                   poll_seconds=1, progress_seconds=15)
            with mock.patch.object(q, 'launch', return_value=process), \
                    mock.patch.object(q.time, 'sleep'), \
                    mock.patch.object(q.time, 'monotonic', side_effect=[0, 0, 1, 15, 16]), \
                    mock.patch.object(q.sys, 'stderr', output):
                q.run_job('STEP5_D_FY240', args, SimpleNamespace(cpus=12, gpus=0))
            text = output.getvalue()
            self.assertEqual(text.count('[SOLVING]'), 2)
            self.assertIn('Waiting for ODB', text)
            self.assertIn('elapsed=00:00:15', text)
            self.assertIn('[SOLVER EXIT]', text)

    def test_solver_progress_reads_latest_status_without_requiring_complete_file(self):
        q = self.queue()
        with tempfile.TemporaryDirectory() as d:
            name = 'STEP5_D_FY240'
            with open(os.path.join(d, name+'.sta'), 'w') as stream:
                stream.write('HEADER\n  1  27  1  0  3  3  0.45\n\n')
            self.assertTrue(hasattr(q, 'solver_progress'), 'live status reporting is missing')
            message = q.solver_progress(d, name, 65, [0, 1, .8], False)
            self.assertIn('elapsed=00:01:05', message)
            self.assertIn('LPF=0.8', message)
            self.assertIn('P/Pmax=0.800', message)
            self.assertIn('1  27  1', message)

    def test_solver_auto_uses_all_physical_cores_without_limiting_python_workers(self):
        import runtime_resources as r
        inventory=r.ResourceInventory(24,12,64*1024**3,60*1024**3,[])
        self.assertTrue(hasattr(r,'resolve_solver_policy'))
        self.assertEqual(r.resolve_solver_policy(inventory).cpus,12)
        self.assertEqual(r.resolve_policy(inventory).cpus,24)
        self.assertEqual(r.resolve_solver_policy(inventory,cpu_override=8).cpus,8)
        self.assertEqual(r.resolve_solver_policy(inventory).memory_reserve_bytes,0)

    def test_solver_oversized_request_uses_all_available_cores(self):
        import runtime_resources as r
        inventory=r.ResourceInventory(8,4,16*1024**3,10*1024**3,[])
        with self.assertWarnsRegex(RuntimeWarning, '8.*4'):
            policy=r.resolve_solver_policy(inventory,cpu_override=8)
        self.assertEqual(policy.cpus,4)
        self.assertEqual(r.resolve_policy(inventory).cpus,8)
        self.assertEqual((policy.memory_reserve_bytes,policy.gpu_reserve_bytes),(0,0))

    def test_zero_launcher_exit_without_odb_reports_solver_log(self):
        q=self.queue()
        with tempfile.TemporaryDirectory() as d:
            process=mock.Mock(returncode=0)
            process.poll.return_value=0
            def launch(cmd, **kwargs):
                kwargs['stdout'].write('Abaqus Error: The number of cpus (24) exceeds the number of cpus available (12).\n')
                return process
            previous=os.getcwd()
            try:
                os.chdir(d)
                with mock.patch.object(q,'launch',side_effect=launch):
                    result=q.run_job('STEP5_D_FY240',SimpleNamespace(run_dir=d,stop_method='monitor',ratio=.7,poll_seconds=1),SimpleNamespace(cpus=24,gpus=0))
            finally:
                os.chdir(previous)
            self.assertIn('cpus available (12)',result.get('error',''))
            self.assertEqual(result.get('outcome'),'SOLVER_ERROR')

    def test_nogui_driver_calls_main_even_in_nonmain_namespace(self):
        from pathlib import Path
        q=self.queue()
        driver=Path(q.__file__).with_name('abaqus_step5_queue_nogui.py')
        self.assertTrue(driver.is_file(), 'explicit noGUI driver missing')
        calls=[]
        with mock.patch.object(q,'main',side_effect=lambda: calls.append('called')):
            exec(compile(driver.read_text(),str(driver),'exec'),{'__name__':'abaqus'})
        self.assertEqual(calls,['called'])

    def test_driver_executes_in_isolated_foreign_cwd_without_main_namespace(self):
        import subprocess
        import sys
        from pathlib import Path
        q=self.queue()
        driver=str(Path(q.__file__).with_name('abaqus_step5_queue_nogui.py'))
        code="import sys; from pathlib import Path; p=%r; sys.argv=['abaqus','--','--help']; exec(compile(Path(p).read_text(),p,'exec'),{'__name__':'abaqus'})" % driver
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run([sys.executable,'-I','-c',code],cwd=d,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('STEP5 NOGUI DRIVER STARTED',result.stderr + result.stdout)
        self.assertIn('--run-dir',result.stdout)

    def test_main_accepts_cae_launcher_flags_without_separator(self):
        import sys
        q=self.queue()
        with mock.patch.object(sys,'argv',['abaqus','-cae','-noGUI',q.__file__,'--run-dir','folder','--repair-only',
                '--abaqus-cmd','C:/Solver tools/abq2024.bat']), mock.patch.object(q,'run') as run:
            q.main()
        self.assertTrue(run.call_args.args[0].repair_only)
        self.assertEqual(run.call_args.args[0].run_dir,os.path.abspath('folder'))
        self.assertEqual(run.call_args.args[0].abaqus_cmd,'C:/Solver tools/abq2024.bat')

    def test_absolute_nogui_script_imports_siblings_from_another_directory(self):
        import subprocess
        import sys
        q=self.queue()
        code = "from pathlib import Path; p=%r; ns={'__name__':'nogui_test'}; exec(compile(Path(p).read_text(),p,'exec'),ns); assert ns['SCRIPT_PATH']==p" % os.path.abspath(q.__file__)
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run([sys.executable,'-I','-c',code],cwd=d,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_nogui_without_file_resolves_runner_path(self):
        q=self.queue()
        with open(q.__file__, encoding='utf-8') as f:
            source=f.read()
        namespace={'__name__':'nogui_queue_test'}
        exec(compile(source,q.__file__,'exec'),namespace)
        self.assertEqual(namespace.get('SCRIPT_PATH'), os.path.abspath(q.__file__))

    def queue(self):
        self.assertIsNotNone(importlib.util.find_spec('abaqus_step5_queue'), 'compiler-free queue is missing')
        import abaqus_step5_queue
        return abaqus_step5_queue

    def test_stop_requires_positive_peak_then_drop(self):
        q = self.queue()
        self.assertIsNone(q.crossing([0, .1, .5, 1.0, .8], .7))
        self.assertEqual(q.crossing([0, .1, 1, .7, .6], .7)['index'], 3)
        self.assertIsNone(q.crossing([0, -.1, -.2], .7))
        self.assertIsNone(q.crossing([0, .7, 1], .7))
        with self.assertRaises(ValueError):
            q.crossing([0, float('nan')], .7)

    def test_job_command_has_all_resources_and_no_user_in_monitor_mode(self):
        q = self.queue()
        cmd = q.job_command('STEP5_D_FY240', 24, 2, 'monitor')
        self.assertIn('cpus=24', cmd)
        self.assertIn('gpus=2', cmd)
        self.assertIn('memory=100%', cmd)
        self.assertIn('standard_parallel=all', cmd)
        self.assertFalse(any(x.startswith('user=') for x in cmd))
        self.assertTrue(any(x.startswith('user=') for x in q.job_command('STEP5_D_FY240',24,0,'urdfil')))

    def test_solver_failure_is_not_silently_accepted_as_controlled_stop(self):
        q = self.queue()
        self.assertEqual(q.outcome({'solver_exit_code': 1}, True), 'SOLVER_ERROR')
        self.assertEqual(q.outcome({'solver_exit_code': 1, 'termination_requested': True}, True), 'CONTROLLED_TERMINATION')
        self.assertEqual(q.outcome({'solver_exit_code': 0}, False), 'CRITERION_NOT_REACHED')
        self.assertEqual(q.outcome({'solver_exit_code': 0}, True), 'CRITERION_REACHED')

    def test_missing_solver_status_and_partial_data_do_not_validate_capacity(self):
        q = self.queue()
        # a crossing inside the synchronized data is a complete result even when the exit
        # status of an earlier session is unknown or trailing frames were dropped
        self.assertEqual(q.outcome({}, True), 'CRITERION_REACHED')
        self.assertEqual(q.outcome({}, False), 'SOLVER_STATUS_UNKNOWN')
        self.assertEqual(q.outcome({'analysis_completed': True}, True), 'CRITERION_REACHED')
        self.assertEqual(q.outcome({'solver_exit_code': 0, 'partial': True}, True), 'CRITERION_REACHED')
        self.assertEqual(q.outcome({'solver_exit_code': 0, 'partial': True}, False), 'PARTIAL_RESULT')
        self.assertEqual(q.outcome({'solver_exit_code': 3}, True), 'SOLVER_ERROR')

    def test_extract_only_preserves_a_previous_solver_failure(self):
        import csv, json
        q = self.queue()
        with tempfile.TemporaryDirectory() as directory:
            name = 'STEP5_D_FY240'
            for suffix in ('.inp', '.odb'):
                with open(os.path.join(directory, name+suffix), 'w') as stream:
                    stream.write('test boundary')
            status = os.path.join(directory, 'STEP5_queue_status.json')
            with open(status, 'w') as stream:
                json.dump([dict(model=name, solver_exit_code=1, termination_requested=False,
                                outcome='SOLVER_ERROR')], stream)
            csv_path = os.path.join(directory, name+'_force_displacement.csv')
            with open(csv_path, 'w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['shortening_mm', 'force_kN', 'lpf'])
                writer.writerows([(0, 0, 0), (1, 240, 1), (2, 120, .5)])
            summary = dict(csv=csv_path, peak=dict(force_kN=240., shortening_mm=1., lpf=1.), partial=False)
            info = dict(stage=5, settings=dict(field_frequency=1, fy=240),
                        reference_force_N_per_end=240000., end_area_mm2=1000.)
            args = SimpleNamespace(run_dir=directory, jobs=[name], cae='test.cae', cpus=1, gpus=0,
                                   extract_only=True, repair_only=False, ratio=.7, stop_method='monitor')
            with mock.patch.object(q, 'metadata', return_value={name: info}), \
                    mock.patch.object(q.resources, 'detect_resources'), \
                    mock.patch.object(q.resources, 'resolve_solver_policy', return_value=SimpleNamespace(cpus=1, gpus=0)), \
                    mock.patch.object(q.fd, 'extract', return_value=summary), \
                    mock.patch.object(q, 'plot_png'), mock.patch.object(q, 'write_comparison'), \
                    mock.patch.object(q, 'report'):
                with self.assertRaises(RuntimeError):
                    q.run(args)
            with open(status) as stream:
                result = json.load(stream)[0]
            self.assertEqual(result['solver_exit_code'], 1)
            self.assertEqual(result['outcome'], 'SOLVER_ERROR')
            self.assertTrue(result['criterion_reached'])

    def test_queue_cpus_option_and_repair_keep_the_requested_value(self):
        import contextlib
        q = self.queue()
        argv = ['abaqus', '--', '--run-dir', '.', '--cpus', '4', '--jobs', 'STEP5_D_FY240']
        seen = {}
        with mock.patch('sys.argv', argv), mock.patch.object(q, 'run', side_effect=lambda a: seen.update(vars(a))):
            q.main()
        self.assertEqual((seen['cpus'], seen['gpus']), (4, 'auto'))
        for bad in ('0', 'x'):
            with mock.patch('sys.argv', argv[:5]+[bad]), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                q.main()
        self.assertEqual(q.job_command('STEP5_D_FY240', 4, 0, 'monitor')[3], 'cpus=4')

    def test_queue_batch_invokes_runner_instead_of_fortran(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'q.bat')
            build.write_queue_batch(p, ['STEP5_D_FY240'], 24)
            with open(p) as f:
                s = f.read()
            self.assertIn('abaqus_step5_queue_nogui.py', s)
            self.assertIn('STEP5_PROGRESS_LOG', s)
            self.assertIn('step5_progress_tail.ps1', s)
            self.assertIn('start "" /b powershell.exe', s)
            self.assertNotIn('user="step5_postpeak_stop.for"', s)

    def test_comparison_contains_all_curves_and_yield_line(self):
        q = self.queue()
        curves = [('D', [(0,0),(1,100),(2,65)]), ('PERFECT', [(0,0),(2,120)])]
        with tempfile.TemporaryDirectory() as d:
            q.write_comparison(d, curves, [114.8], [])
            with open(os.path.join(d,'STEP5_all_force_displacement_combined.svg')) as f:
                s = f.read()
            self.assertIn('PERFECT', s)
            self.assertIn('AsFy', s)
            self.assertTrue(os.path.getsize(os.path.join(d,'STEP5_all_force_displacement_combined.png')) > 1000)

    def test_sta_riks_parser_skips_unconverged_attempts(self):
        q=self.queue()
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,'job.sta')
            with open(path,'w') as f:
                f.write('   1    23   1U    0     6     6             0.941      0.03711               R\n')
                f.write('   1    23   2     0     3     3             0.953      0.01244               R\n')
                f.write('   1    24   1     0     5     5             0.971      0.01795               R\n')
            records=q.read_sta_lpf(path)
            self.assertEqual([r['lpf'] for r in records],[.953,.971])
            self.assertEqual(records[0]['increment'],23)
            self.assertAlmostEqual(records[0]['half_unit'],.0005)

    def test_sta_crossing_is_conservative_against_print_rounding(self):
        q=self.queue()
        records=[dict(lpf=1.000,half_unit=.0005),
                 dict(lpf=.700,half_unit=.0005)]
        self.assertIsNone(q.crossing_sta(records,.70))
        records.append(dict(lpf=.699,half_unit=.0005))
        hit=q.crossing_sta(records,.70)
        self.assertIsNotNone(hit)
        self.assertLessEqual(hit['conservative_ratio'],.70)

    def test_monitor_stops_with_native_command_and_records_expected_nonzero_exit(self):
        q = self.queue()
        with tempfile.TemporaryDirectory() as d:
            name = 'STEP5_D_FY240'
            process = mock.Mock(returncode=1)
            process.poll.side_effect = [None, 1]
            control = mock.Mock()
            control.wait.return_value = 0
            commands = []
            def launch(cmd, **kwargs):
                commands.append((cmd, kwargs['cwd']))
                if len(commands) == 1:
                    with open(os.path.join(d,name+'.odb'),'w') as f:
                        f.write('mock native ODB boundary')
                    return process
                return control
            args = SimpleNamespace(run_dir=d, stop_method='monitor', ratio=.7, poll_seconds=1)
            previous = os.getcwd()
            try:
                os.chdir(d)
                records=[dict(lpf=0.0,half_unit=0.0005),dict(lpf=1.0,half_unit=0.0005),dict(lpf=.65,half_unit=0.0005)]
                with mock.patch.object(q,'launch',side_effect=launch), mock.patch.object(q,'read_sta_lpf',return_value=records), mock.patch.object(q.time,'sleep'):
                    result=q.run_job(name,args,SimpleNamespace(cpus=24,gpus=0))
            finally:
                os.chdir(previous)
            self.assertTrue(result['termination_requested'])
            self.assertEqual(result['observed_crossing']['index'],2)
            self.assertEqual(commands[1],(['abaqus','terminate','job='+name], d))
            self.assertEqual(q.outcome(result,True),'CONTROLLED_TERMINATION')

    def test_custom_launcher_is_used_for_submission_and_monitor_termination(self):
        q = self.queue()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            name = 'STEP5_D_FY240'
            process = mock.Mock(returncode=1)
            process.poll.side_effect = [None, 1]
            control = mock.Mock()
            control.wait.return_value = 0
            args = SimpleNamespace(run_dir=d, stop_method='monitor', ratio=.7, poll_seconds=1,
                                   abaqus_cmd='C:/Solver tools/custom abaqus.bat')
            records = [dict(lpf=x,half_unit=.0005) for x in (0.,1.,.65)]
            with mock.patch.object(q,'launch',side_effect=[process,control]) as launch, \
                    mock.patch.object(q,'read_sta_lpf',return_value=records), \
                    mock.patch.object(q.time,'sleep'):
                result = q.run_job(name,args,SimpleNamespace(cpus=4,gpus=0))
            self.assertEqual(launch.call_args_list[0].args[0][0],args.abaqus_cmd)
            self.assertEqual(launch.call_args_list[1].args[0], [args.abaqus_cmd,'terminate','job='+name])
            self.assertTrue(result['termination_requested'])

    def test_monitor_retries_unflushed_odb_then_detects_crossing(self):
        q=self.queue()
        with tempfile.TemporaryDirectory() as d:
            name='STEP5_D_FY240'
            process=mock.Mock(returncode=1)
            process.poll.side_effect=[None,None,1]
            control=mock.Mock();control.wait.return_value=0
            def launch(cmd, **kwargs):
                if cmd[1].startswith('job='):
                    with open(os.path.join(d,name+'.odb'),'w') as f:
                        f.write('mock native boundary')
                    return process
                return control
            previous=os.getcwd()
            try:
                os.chdir(d)
                records=[dict(lpf=0.0,half_unit=0.0005),dict(lpf=1.0,half_unit=0.0005),dict(lpf=.6,half_unit=0.0005)]
                with mock.patch.object(q,'launch',side_effect=launch), mock.patch.object(q,'read_sta_lpf',side_effect=[[],records]), mock.patch.object(q.time,'sleep'):
                    result=q.run_job(name,SimpleNamespace(run_dir=d,stop_method='monitor',ratio=.7,poll_seconds=1),SimpleNamespace(cpus=24,gpus=0))
            finally:
                os.chdir(previous)
            self.assertTrue(result['termination_requested'])

    def test_existing_solver_artifacts_are_refused_without_moving_anything(self):
        q=self.queue()
        with tempfile.TemporaryDirectory() as d:
            name='STEP5_D_FY240'
            for ext in ('.inp','.log','.dat','.msg'):
                with open(os.path.join(d,name+ext),'w') as f:
                    f.write(ext)
            with self.assertRaises(ValueError):
                q.backup_failed_artifacts(d,name)
            self.assertTrue(os.path.isfile(os.path.join(d,name+'.inp')))
            self.assertTrue(os.path.exists(os.path.join(d,name+'.log')))
            self.assertEqual(len(os.listdir(d)), 4)

    def test_occupied_resume_reports_actual_failure_and_saved_status_without_writes(self):
        import json
        from pathlib import Path
        q=self.queue()
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as d:
            root=Path(d); name='STEP5_LG_mp_FY240'
            (root/(name+'.msg')).write_text('***ERROR: Too many attempts made for this increment.\n')
            (root/'STEP5_queue_status.json').write_text(json.dumps([
                dict(model=name,outcome='INCOMPLETE_ANALYSIS',solver_exit_code=0,
                     criterion_reached=False,termination_requested=False)]))
            before={p.name:p.read_bytes() for p in root.iterdir()}
            with self.assertRaises(ValueError) as error:
                q.backup_failed_artifacts(d,name)
            message=str(error.exception)
            self.assertIn('analysis_state=FAILED',message)
            self.assertIn('TOO MANY ATTEMPTS',message)
            self.assertIn('saved_outcome=INCOMPLETE_ANALYSIS',message)
            self.assertIn('solver_exit_code=0',message)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            (root/(name+'.msg')).write_text('Increment table only')
            (root/'STEP5_queue_status.json').unlink()
            with self.assertRaisesRegex(ValueError,'saved_status=missing'):
                q.backup_failed_artifacts(d,name)

    def test_submission_contract_checks_input_and_stop_ratio(self):
        q = self.queue()
        with tempfile.TemporaryDirectory() as d:
            name = 'STEP5_TEST'
            path = os.path.join(d, name+'.inp')
            with open(path, 'w') as stream: stream.write('original input')
            info = dict(job_name=name, input_sha256=q.contract.file_sha256(path),
                        settings=dict(postpeak_stop_ratio=.7))
            self.assertTrue(q.validate_submission(d, name, info, 'urdfil', .7))
            with self.assertRaises(ValueError): q.validate_submission(d, name, info, 'urdfil', .8)
            q.validate_submission(d, name, info, 'monitor', .8)
            with open(path, 'w') as stream: stream.write('changed input')
            with self.assertRaises(ValueError): q.validate_submission(d, name, info, 'monitor', .7)
            with self.assertRaises(ValueError): q.validate_submission(d, name, {}, 'monitor', .7)

    def test_bending_comparison_and_pm_points(self):
        import csv
        q=self.queue()
        info={'settings':{'fy':240},'load_case':{'type':'bending'},'bending_axis_deg':45.,
              'section_for_PM':{'area_mm2':1000.,'first_yield_moment_Nmm':5e7,'plastic_moment_Nmm':7e7}}
        point=q.pm_point(info,{'moment_kNm':55.})
        self.assertEqual((point['load_case'],point['P_kN'],point['M_kNm']),('bending',0.0,55.))
        self.assertAlmostEqual(point['My_kNm'],50.); self.assertAlmostEqual(point['Mp_kNm'],70.)
        self.assertAlmostEqual(point['Py_kN'],240.)
        cpoint=q.pm_point({'settings':{'fy':240},'end_area_mm2':1000.},{'force_kN':200.})
        self.assertEqual((cpoint['load_case'],cpoint['P_kN'],cpoint['M_kNm']),('compression',200.,0.0))
        axes=q.fd.RESPONSE_AXES['bending']
        with tempfile.TemporaryDirectory() as d:
            rows=[dict(model='STEP5_D_FY240',moment_kNm=55.,My_kNm=50.,Mu_over_My=1.1,pm_point=point,
                       criterion_reached=True,outcome='CRITERION_REACHED')]
            q.write_comparison(d,[('D',[(0.,0.),(1.,55.)])],[50.],rows,axes)
            with open(os.path.join(d,'STEP5_peak_summary.csv'),encoding='utf-8-sig') as f:
                self.assertIn('Mu_over_My',f.readline())
            with open(os.path.join(d,'STEP5_PM_points.csv'),encoding='utf-8-sig') as f:
                pm=list(csv.DictReader(f))
            self.assertEqual(pm[0]['load_case'],'bending'); self.assertEqual(float(pm[0]['M_kNm']),55.)
            with open(os.path.join(d,'STEP5_all_moment_rotation_combined.svg')) as f:
                text=f.read()
            self.assertIn('My 50.000 kN m',text)
            self.assertIn('End rotation, theta (mrad)',text)
            self.assertTrue(os.path.getsize(os.path.join(d,'STEP5_all_moment_rotation_combined.png')) > 1000)

    def test_labelled_report_lines_in_comparison_svg_and_png(self):
        q = self.queue()
        import matplotlib.pyplot as plt
        curves = [('Bending', [(0., 0.), (1., 3.)])]
        refs = [{'value': 5., 'label': 'SFy report x'}, {'value': 7., 'label': 'ZFy report x'}]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(plt, 'close'):
            path = os.path.join(directory, 'comparison')
            q.plot_svg(path+'.svg', curves, refs, 'Bending', q.fd.RESPONSE_AXES['bending'])
            with open(path+'.svg', encoding='utf-8') as stream:
                svg = stream.read()
            self.assertIn('SFy report x', svg)
            self.assertIn('ZFy report x', svg)
            q.plot_png(path+'.png', curves, refs, 'Bending', q.fd.RESPONSE_AXES['bending'])
            ax = plt.gcf().axes[0]
            report_lines = [line for line in ax.lines if 'report' in line.get_label()]
            self.assertEqual([list(line.get_ydata()) for line in report_lines], [[5., 5.], [7., 7.]])
            self.assertGreater(ax.get_ylim()[1], 7.)
        plt.close('all')

    def test_cae_fallback_keeps_labelled_horizontal_report_lines(self):
        q = self.queue()
        import builtins
        original_import = builtins.__import__
        def without_matplotlib(name, *args, **kwargs):
            if name == 'matplotlib':
                raise ImportError('exercise CAE fallback')
            return original_import(name, *args, **kwargs)
        chart = SimpleNamespace(setValues=mock.Mock(), autoColor=mock.Mock())
        plot = SimpleNamespace(charts={'chart': chart}, title=SimpleNamespace(setValues=mock.Mock()))
        session = SimpleNamespace(xyPlots={}, xyDataObjects={}, XYPlot=mock.Mock(return_value=plot),
            XYData=mock.Mock(side_effect=lambda **kw: kw), Curve=mock.Mock(side_effect=lambda **kw: kw),
            viewports={'view': SimpleNamespace(setValues=mock.Mock())},
            printOptions=SimpleNamespace(setValues=mock.Mock()), pngOptions=SimpleNamespace(setValues=mock.Mock()),
            printToFile=mock.Mock())
        with mock.patch.object(builtins, '__import__', side_effect=without_matplotlib), mock.patch.dict('sys.modules', {
                'abaqus': SimpleNamespace(session=session), 'abaqusConstants': SimpleNamespace(PNG='PNG', OFF=False),
                'visualization': SimpleNamespace()}):
            q.plot_png('unused.png', [('curve', [(0., 0.), (2., 3.)])],
                       [{'value': 7., 'label': 'ZFy report built-up x'}], 'Bending', q.fd.RESPONSE_AXES['bending'])
        reference = session.XYData.call_args_list[-1].kwargs
        self.assertEqual(reference['data'], ((0., 7.), (2., 7.)))
        self.assertEqual(reference['legendLabel'], 'ZFy report built-up x')

    def test_report_lines_are_deduplicated_by_value_and_identity(self):
        q = self.queue()
        lines = [5.]
        q.merge_capacity_lines(lines, [5., {'value': 5., 'label': 'SFy report'},
                                       {'value': 7., 'label': 'ZFy report'}], q.fd.RESPONSE_AXES['bending'])
        q.merge_capacity_lines(lines, [{'value': 7., 'label': 'ZFy report'}], q.fd.RESPONSE_AXES['bending'])
        self.assertEqual(lines, [5., {'value': 5., 'label': 'SFy report'}, {'value': 7., 'label': 'ZFy report'}])

    def test_queue_report_override_preserves_capacity_ratios_and_metadata(self):
        import csv, json
        import abaqus_report_reference as reader
        q = self.queue()
        reference = dict(schema='cufsm_builtup_reference_v1', AsFy_kN=1100., axes={
            'x': dict(axis_deg=0., SFy_kNm=130., ZFy_kNm=190.),
            'z': dict(axis_deg=90., SFy_kNm=140., ZFy_kNm=200.)})
        with tempfile.TemporaryDirectory() as directory:
            name = 'STEP5_B_FY240'
            info = dict(stage=5, settings=dict(field_frequency=1, fy=240.),
                        load_case=dict(type='bending'), bending_axis_deg=0.,
                        reference_moment_Nmm=20e6, first_yield_moment_Nmm=20e6, end_area_mm2=1000.)
            metadata_path = os.path.join(directory, name+'_step5.json')
            original = json.dumps(info)
            with open(metadata_path, 'w') as stream:
                stream.write(original)
            with open(os.path.join(directory, name+'.inp'), 'w') as stream:
                stream.write('test input')
            csv_path = os.path.join(directory, name+'_moment_rotation.csv')
            with open(csv_path, 'w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['rotation_mrad', 'moment_kNm', 'lpf'])
                writer.writerows([(0., 0., 0.), (1., 50., 1.), (2., 25., .5)])
            summary = dict(csv=csv_path, peak=dict(moment_kNm=50., rotation_mrad=1., lpf=1.), partial=False)
            args = SimpleNamespace(run_dir=directory, jobs=[name], cae=None, cpus=1, gpus=0,
                                   extract_only=True, repair_only=False, ratio=.7, stop_method='monitor',
                                   cufsm_report_dir='report-folder')
            completed = dict(analysis_completed=True, analysis_state='COMPLETED', completion_diagnostics=[],
                             completion_sources=['test'])
            with mock.patch.object(reader, 'read_reference', return_value=reference) as read, \
                    mock.patch.object(q.resources, 'detect_resources'), \
                    mock.patch.object(q.resources, 'resolve_solver_policy', return_value=SimpleNamespace(cpus=1, gpus=0)), \
                    mock.patch.object(q, 'completion_details', return_value=completed), \
                    mock.patch.object(q.fd, 'extract', return_value=summary) as extract, \
                    mock.patch.object(q, 'plot_png') as plot, mock.patch.object(q, 'write_comparison') as compare, \
                    mock.patch.object(q, 'report'):
                q.run(args)
            read.assert_called_once_with('report-folder', fy=240.)
            self.assertEqual(extract.call_args.args[0].cufsm_report_dir, 'report-folder')
            self.assertEqual([line['value'] for line in plot.call_args.args[2]], [130., 190.])
            result = compare.call_args.args[3][0]
            self.assertEqual(result['My_kNm'], 20.)
            self.assertEqual(result['Mu_over_My'], 2.5)
            self.assertEqual(compare.call_args.kwargs['report_reference'], reference)
            with open(metadata_path) as stream:
                self.assertEqual(stream.read(), original)

    def test_resumed_saved_curves_receive_report_lines_without_extraction(self):
        import json
        import abaqus_report_reference as reader
        q = self.queue()
        reference = dict(schema='cufsm_builtup_reference_v1', AsFy_kN=1100., axes={})
        with tempfile.TemporaryDirectory() as directory:
            name = 'STEP5_D_FY240'
            info = dict(stage=5, settings=dict(field_frequency=1, fy=240.),
                        reference_force_N_per_end=240000., end_area_mm2=1000.)
            for filename, content in ((name+'.inp', 'test input'), (name+'_step5.json', json.dumps(info))):
                with open(os.path.join(directory, filename), 'w') as stream:
                    stream.write(content)
            saved = dict(model=name, outcome='CRITERION_REACHED', peak_interpretation='verified',
                         criterion_reached=True, analysis_completed=True, analysis_state='COMPLETED')
            args = SimpleNamespace(run_dir=directory, jobs=[name], cae=None, cpus=1, gpus=0,
                                   extract_only=False, repair_only=False, resume=True, ratio=.7,
                                   stop_method='monitor', cufsm_report_dir='report-folder')
            with mock.patch.object(reader, 'read_reference', return_value=reference), \
                    mock.patch.object(q.resources, 'detect_resources'), \
                    mock.patch.object(q.resources, 'resolve_solver_policy', return_value=SimpleNamespace(cpus=1, gpus=0)), \
                    mock.patch.object(q, 'validate_submission'), \
                    mock.patch.object(q, 'completed_curve', return_value=(saved, [(0., 0.), (1., 200.)])), \
                    mock.patch.object(q.fd, 'extract') as extract, mock.patch.object(q, 'run_job') as solve, \
                    mock.patch.object(q, 'plot_png') as png, mock.patch.object(q, 'plot_svg') as svg, \
                    mock.patch.object(q, 'write_comparison'), mock.patch.object(q, 'report'):
                q.run(args)
            extract.assert_not_called()
            solve.assert_not_called()
            self.assertEqual(png.call_args.args[2][0]['value'], 1100.)
            self.assertEqual(svg.call_args.args[2][0]['value'], 1100.)
