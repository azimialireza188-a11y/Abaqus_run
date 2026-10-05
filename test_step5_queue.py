import os
import tempfile
import unittest
import io
import abaqus_step5_gmnia as build
import importlib.util
from unittest import mock
from types import SimpleNamespace


class QueueTests(unittest.TestCase):
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
        with mock.patch.object(sys,'argv',['abaqus','-cae','-noGUI',q.__file__,'--run-dir','folder','--repair-only']), mock.patch.object(q,'run') as run:
            q.main()
        self.assertTrue(run.call_args.args[0].repair_only)
        self.assertEqual(run.call_args.args[0].run_dir,os.path.abspath('folder'))

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

    def test_failed_compile_artifacts_are_backed_up_without_moving_input(self):
        q=self.queue()
        with tempfile.TemporaryDirectory() as d:
            name='STEP5_D_FY240'
            for ext in ('.inp','.log','.dat','.msg'):
                with open(os.path.join(d,name+ext),'w') as f:
                    f.write(ext)
            paths=q.backup_failed_artifacts(d,name)
            self.assertEqual(len(paths),3)
            self.assertTrue(os.path.isfile(os.path.join(d,name+'.inp')))
            self.assertFalse(os.path.exists(os.path.join(d,name+'.log')))
            self.assertTrue(all(os.path.isfile(p) for p in paths))
