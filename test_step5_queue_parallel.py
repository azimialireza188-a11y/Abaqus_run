"""Small solver-free checks of independent Step5 scheduling."""
import json
import os
from pathlib import Path
import shutil
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import abaqus_step5_queue as q


class ParallelQueueTests(unittest.TestCase):
    def directory(self):
        path = Path(__file__).resolve().parent / ('queue_test_' + uuid.uuid4().hex)
        path.mkdir()
        self.addCleanup(shutil.rmtree, path)
        scratch_base = mock.patch.object(q, 'SCRIPT_DIR', str(path))
        scratch_base.start()
        self.addCleanup(scratch_base.stop)
        return path

    def test_slots_share_all_cores_memory_and_one_gpu_owner(self):
        policies = q.parallel_policies(SimpleNamespace(cpus=12, gpus=1), 5, 7)
        self.assertEqual([p.cpus for p in policies], [3, 3, 2, 2, 2])
        self.assertEqual([p.gpus for p in policies], [1, 0, 0, 0, 0])
        self.assertEqual(sum(p.memory_percent for p in policies), 100)
        self.assertEqual(len(q.parallel_policies(SimpleNamespace(cpus=12, gpus=0), 'auto', 7)), 2)
        self.assertEqual(len(q.parallel_policies(SimpleNamespace(cpus=2, gpus=0), 10, 7)), 2)

    def test_long_queue_uses_bounded_native_paths_for_all_job_name_lengths(self):
        directory=(r'D:\column-result\PM_compare_no_plates_with_plates_A3184_t2\with_plates\New folder'
                   r'\family_1_BEND000_PL40_PL240_PG0_PS3\imperfection\Step5_PM_FY240_LG_REF1_PERFECT_ONLY')
        for name in ('PM_A000_P15_PERFECT','PM_'+'A'*35):
            with self.subTest(name=name):
                work=q.parallel_workspace(directory,name)
                scratch=q.parallel_scratch(directory,name)
                self.assertLess(q.native_path_lengths(scratch,name)['preprocessor'],240)
                self.assertLess(q.native_path_lengths(work,name)['artifacts'],256)
                self.assertNotIn(name,os.path.basename(work))
                self.assertNotEqual(scratch,q.parallel_scratch(directory,name+'X'))

    def test_preprocessor_error_is_reported_even_when_launcher_returns_zero_and_odb_exists(self):
        path=self.directory();name='PM_A000_P15_PERFECT'
        def native(cmd,**kwargs):
            (path/(name+'.odb')).write_bytes(b'header only; no GMNIA step')
            (path/(name+'.dat')).write_text('***ERROR: UNABLE TO OPEN INPUT FILE ON UNIT 5. FILENAME\n missing_f.inp\n')
            return SimpleNamespace(poll=lambda:0,returncode=0)
        args=SimpleNamespace(run_dir=str(path),stop_method='monitor',ratio=.7,poll_seconds=1)
        with mock.patch.object(q,'launch',side_effect=native):
            result=q.run_job(name,args,SimpleNamespace(cpus=6,gpus=0))
        self.assertEqual(result['analysis_state'],'FAILED')
        self.assertIn('preprocess',result['error'].lower())
        self.assertIn('UNABLE TO OPEN INPUT FILE',result['error'])

    def test_duplicate_queue_manager_is_rejected_and_claim_releases(self):
        path = self.directory()
        with q.queue_claim(str(path)):
            with self.assertRaisesRegex(ValueError, 'queue.*active|active.*queue'):
                with q.queue_claim(str(path)):
                    self.fail('Second queue entered')
        with q.queue_claim(str(path)):
            pass

    def test_native_paths_fail_early_and_convergence_failure_remains_extractable(self):
        with self.assertRaisesRegex(ValueError, 'path limit.*output-root'):
            q.parallel_workspace(os.path.join(str(Path.cwd()), 'x'*180), 'PM_A000_P15_PERFECT')
        path = self.directory(); name = 'PM_A000_P15_PERFECT'
        (path/(name+'.sta')).write_text('THE ANALYSIS HAS NOT BEEN COMPLETED\n')
        (path/(name+'.msg')).write_text('***ERROR: TIME INCREMENT REQUIRED IS LESS THAN THE MINIMUM SPECIFIED\n')
        result = q.completion_details(str(path),name)
        self.assertEqual(result['analysis_state'],'FAILED')
        self.assertFalse(result['preprocessing_failed'])
        result.update(curve_available=True,outcome='SOLVER_ERROR')
        self.assertTrue(q.processed_result(result))

    def test_scratch_ownership_is_per_queue_and_stale_scratch_is_preserved(self):
        path = self.directory(); name = 'PM_A000_P15_PERFECT'
        (path/(name+'.inp')).write_text('*Heading\n')
        scratch = Path(q.parallel_scratch(str(path),name))
        self.assertNotEqual(str(scratch),q.parallel_scratch(str(path/'other'),name))
        scratch.mkdir(parents=True)
        (scratch/'evidence.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'preserved'):
            q.validate_parallel_source(str(path),name)
        self.assertEqual((scratch/'evidence.txt').read_text(),'preserve')

    @unittest.skipUnless(os.name=='nt','Windows native path aliases')
    def test_native_extended_path_alias_is_not_mistaken_for_scratch_redirection(self):
        path=self.directory(); name='PM_A000_P30_PERFECT'
        (path/'.abq_scratch').mkdir()
        (path/'.queue_work').mkdir()
        realpath=q.os.path.realpath
        def extended(candidate):
            resolved=realpath(candidate)
            if (os.sep+'.abq_scratch' in resolved or os.sep+'.queue_work' in resolved) and not resolved.startswith('\\\\?\\'):
                return '\\\\?\\'+resolved
            return resolved
        expected=q.parallel_scratch(str(path),name)
        with mock.patch.object(q.os.path,'realpath',side_effect=extended):
            scratch=q.parallel_scratch(str(path),name)
            workspace=q.parallel_workspace(str(path),name)
        self.assertEqual(os.path.normcase(os.path.normpath(scratch)),os.path.normcase(os.path.normpath(expected)))
        self.assertEqual(os.path.normcase(os.path.dirname(workspace)),os.path.normcase(str(path/'.queue_work')))
        import ctypes
        short_path=ctypes.WinDLL('kernel32',use_last_error=True).GetShortPathNameW
        short_path.argtypes=(ctypes.c_wchar_p,ctypes.c_wchar_p,ctypes.c_uint32)
        short_path.restype=ctypes.c_uint32
        aliases=[]
        for root in (path/'.abq_scratch',path/'.queue_work'):
            buffer=ctypes.create_unicode_buffer(32768)
            if short_path(str(root),buffer,len(buffer)):
                aliases.append((str(root),buffer.value))
        def short_alias(candidate):
            resolved=realpath(candidate)
            for long,short in aliases:
                if resolved==long or resolved.startswith(long+os.sep):
                    return short+resolved[len(long):]
            return resolved
        with mock.patch.object(q.os.path,'realpath',side_effect=short_alias):
            self.assertEqual(q.parallel_scratch(str(path),name),expected)
            self.assertEqual(q.parallel_workspace(str(path),name),workspace)

    def test_real_scratch_redirection_is_rejected_without_falling_back_or_launching(self):
        path=self.directory(); name='PM_A000_P30_PERFECT'
        realpath=q.os.path.realpath
        def redirected(candidate):
            return realpath(candidate).replace(os.sep+'.abq_scratch',os.sep+'outside_scratch')
        with mock.patch.object(q.os.path,'realpath',side_effect=redirected):
            with self.assertRaisesRegex(ValueError,'scratch.*redirect'):
                q.parallel_scratch(str(path),name)
        self.assertFalse((path/'outside_scratch').exists())

    def test_scratch_reparse_point_is_rejected_even_if_realpath_spelling_matches(self):
        import stat
        path=self.directory(); lstat=q.os.lstat
        def reparse(candidate,*args,**kwargs):
            if os.path.basename(candidate)=='.abq_scratch':
                return SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
            return lstat(candidate,*args,**kwargs)
        with mock.patch.object(q.os,'lstat',side_effect=reparse):
            with self.assertRaisesRegex(ValueError,'scratch.*redirect.*junction'):
                q.parallel_scratch(str(path),'PM_A000_P30_PERFECT')

    def test_resume_retries_an_unsubmitted_job_but_preserves_any_existing_native_result(self):
        path=self.directory();name='PM_A000_P30_PERFECT'
        record=dict(model=name,outcome='SOLVER_ERROR',error='scratch validation failed before launch')
        (path/'STEP5_queue_status.json').write_text(json.dumps([record]))
        self.assertIsNone(q.completed_curve(str(path),name,{},allow_incomplete=True))
        self.assertFalse(q.backup_failed_artifacts(str(path),name))
        odb=path/(name+'.odb');odb.write_bytes(b'preserve native result')
        with self.assertRaisesRegex(ValueError,'Missing completed result artifact'):
            q.completed_curve(str(path),name,{},allow_incomplete=True)
        with self.assertRaisesRegex(ValueError,'preserved'):
            q.backup_failed_artifacts(str(path),name)
        self.assertEqual(odb.read_bytes(),b'preserve native result')

    def test_parallel_solver_jobs_overlap_without_parallel_extraction(self):
        barrier = threading.Barrier(2)
        calls = []
        policies = q.parallel_policies(SimpleNamespace(cpus=12, gpus=1), 2, 2)
        def solve(name, args, policy):
            calls.append((name, policy.cpus, policy.gpus, threading.get_ident()))
            barrier.wait(timeout=3)
            return {'model': name, 'solver_exit_code': 0}
        with mock.patch.object(q, 'run_job', side_effect=solve):
            rows = list(q.solver_results(['PM_A000_P00_PERFECT', 'PM_A000_P15_PERFECT'],
                                         SimpleNamespace(), policies))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({c[3] for c in calls}), 2)
        self.assertEqual(sorted(c[1] for c in calls), [6, 6])
        self.assertEqual(sorted(c[2] for c in calls), [0, 1])

    def test_finished_slot_is_refilled_while_other_initial_job_is_running(self):
        first_pair = threading.Barrier(2)
        third_started = threading.Event()
        calls = {}
        policies = q.parallel_policies(SimpleNamespace(cpus=12, gpus=1), 2, 3)
        def solve(name, args, policy):
            calls[name] = (policy.cpus, policy.gpus)
            if name in ('PM_A000_P00_PERFECT', 'PM_A000_P15_PERFECT'):
                first_pair.wait(timeout=3)
            if name == 'PM_A000_P15_PERFECT':
                if not third_started.wait(timeout=1):
                    raise AssertionError('Idle slot was not refilled while another job was running')
            if name == 'PM_A000_P30_PERFECT':
                third_started.set()
            return {'model': name}
        with mock.patch.object(q, 'run_job', side_effect=solve):
            rows = list(q.solver_results(['PM_A000_P00_PERFECT', 'PM_A000_P15_PERFECT',
                                         'PM_A000_P30_PERFECT'], SimpleNamespace(), policies))
        self.assertTrue(all('error' not in result for _, result in rows), rows)
        self.assertEqual(calls['PM_A000_P30_PERFECT'], calls['PM_A000_P00_PERFECT'])

    def test_slots_refill_even_while_result_consumer_is_busy(self):
        first_pair = threading.Barrier(2)
        release_second, release_third, fourth_started = (threading.Event() for _ in range(3))
        policies = q.parallel_policies(SimpleNamespace(cpus=12, gpus=1), 2, 4)
        def solve(name, args, policy):
            if name in ('PM_A000_P00_PERFECT', 'PM_A000_P15_PERFECT'):
                first_pair.wait(timeout=3)
            if name == 'PM_A000_P15_PERFECT':
                release_second.wait(timeout=3)
            if name == 'PM_A000_P30_PERFECT':
                release_third.wait(timeout=3)
            if name == 'PM_A000_P45_PERFECT':
                fourth_started.set()
            return {'model': name}
        with mock.patch.object(q, 'run_job', side_effect=solve):
            pending = q.solver_results(['PM_A000_P00_PERFECT', 'PM_A000_P15_PERFECT',
                                       'PM_A000_P30_PERFECT', 'PM_A000_P45_PERFECT'],
                                      SimpleNamespace(), policies)
            try:
                self.assertEqual(next(pending)[0], 'PM_A000_P00_PERFECT')
                release_second.set()
                # Simulate ODB extraction without resuming the result generator.
                self.assertTrue(fourth_started.wait(timeout=1), 'Busy consumer prevented slot refill')
            finally:
                release_second.set(); release_third.set(); pending.close()

    def test_parallel_job_uses_private_workspace_and_delivers_native_artifacts(self):
        path = self.directory()
        name = 'PM_A000_P15_PERFECT'
        (path / (name + '.inp')).write_text('*Heading\n')
        (path / 'abaqus_v6.env').write_text('memory="100%"\n')
        args = SimpleNamespace(run_dir=str(path), stop_method='monitor', ratio=.7,
                               poll_seconds=1, isolated_solver=True)
        def native(cmd, **kwargs):
            cwd = Path(kwargs['cwd'])
            self.assertNotEqual(cwd, path)
            self.assertEqual((cwd / (name + '.inp')).read_text(), '*Heading\n')
            self.assertIn('memory=50%', cmd)
            self.assertTrue(any(s.startswith('scratch=') for s in cmd))
            (cwd / (name + '.odb')).write_text('mock odb')
            (cwd / (name + '.sta')).write_text('THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n')
            return SimpleNamespace(poll=lambda: 0, returncode=0)
        with mock.patch.object(q, 'launch', side_effect=native):
            result = q.run_job(name, args, SimpleNamespace(cpus=6, gpus=0, memory_percent=50))
        self.assertEqual((path / (name + '.odb')).read_text(), 'mock odb')
        self.assertTrue(result['analysis_completed'])
        self.assertFalse(Path(q.parallel_workspace(str(path), name)).exists())

    def test_parallel_input_with_external_include_is_refused_before_launch(self):
        path = self.directory()
        name = 'PM_A000_P15_PERFECT'
        (path / (name + '.inp')).write_text('*INCLUDE, INPUT=elsewhere.inp\n')
        args = SimpleNamespace(run_dir=str(path), stop_method='monitor', ratio=.7,
                               poll_seconds=1, isolated_solver=True)
        with mock.patch.object(q, 'launch') as launch:
            with self.assertRaisesRegex(ValueError, 'external|INCLUDE'):
                q.run_job(name, args, SimpleNamespace(cpus=6, gpus=0, memory_percent=50))
        launch.assert_not_called()

    def test_native_exception_preserves_workspace_instead_of_deleting_live_files(self):
        path = self.directory()
        name = 'PM_A000_P15_PERFECT'
        (path / (name + '.inp')).write_text('*Heading\n')
        args = SimpleNamespace(run_dir=str(path), isolated_solver=True)
        with mock.patch.object(q, '_run_job_native', side_effect=RuntimeError('termination failed')):
            with self.assertRaisesRegex(RuntimeError, 'termination failed'):
                q.run_job(name, args, SimpleNamespace(cpus=6, gpus=0, memory_percent=50))
        self.assertTrue((Path(q.parallel_workspace(str(path), name)) / (name + '.inp')).is_file())

    def test_parallel_delivery_waits_for_transient_native_lock(self):
        path = self.directory()
        name = 'PM_A000_P15_PERFECT'
        (path / (name + '.inp')).write_text('*Heading\n')
        work = Path(q.parallel_workspace(str(path), name))
        args = SimpleNamespace(run_dir=str(path), isolated_solver=True)
        def solve(*unused):
            (work / (name + '.odb')).write_text('mock odb')
            (work / (name + '.lck')).write_text('')
            (work / (name + '.sta')).write_text('THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n')
            return {'analysis_state': 'RUNNING'}
        with mock.patch.object(q, '_run_job_native', side_effect=solve), \
                mock.patch.object(q.time, 'sleep', side_effect=lambda _: (work / (name + '.lck')).unlink()):
            result = q.run_job(name, args, SimpleNamespace(cpus=6, gpus=0, memory_percent=50))
        self.assertEqual(result['analysis_state'], 'COMPLETED')
        self.assertTrue((path / (name + '.odb')).is_file())

    def test_persistent_native_lock_preserves_outputs_in_private_workspace(self):
        path = self.directory()
        name = 'PM_A000_P15_PERFECT'
        (path / (name + '.inp')).write_text('*Heading\n')
        work = Path(q.parallel_workspace(str(path), name))
        def solve(*unused):
            (work / (name + '.odb')).write_text('mock odb')
            (work / (name + '.lck')).write_text('')
            return {}
        with mock.patch.object(q, '_run_job_native', side_effect=solve), \
                mock.patch.object(q.time, 'sleep') as wait:
            with self.assertRaisesRegex(ValueError, 'remains locked'):
                q.run_job(name, SimpleNamespace(run_dir=str(path), isolated_solver=True),
                          SimpleNamespace(cpus=6, gpus=0, memory_percent=50))
        self.assertEqual(wait.call_count, 30)
        self.assertTrue((work / (name + '.odb')).is_file())
        self.assertFalse((path / (name + '.odb')).exists())

    def test_redirected_workspace_root_is_rejected_before_creation(self):
        path = self.directory()
        name = 'PM_A000_P15_PERFECT'
        (path / (name + '.inp')).write_text('*Heading\n')
        realpath = q.os.path.realpath
        def redirected(candidate):
            result = realpath(candidate)
            return result.replace(os.sep+'.queue_work', os.sep+'outside_workspace')
        with mock.patch.object(q.os.path, 'realpath', side_effect=redirected), \
                mock.patch.object(q, 'launch') as launch:
            with self.assertRaisesRegex(ValueError, 'workspace.*(?:redirect|outside|contain)|redirect.*workspace'):
                q.run_job(name, SimpleNamespace(run_dir=str(path), isolated_solver=True),
                          SimpleNamespace(cpus=6, gpus=0, memory_percent=50))
        launch.assert_not_called()
        self.assertFalse((path / '.queue_work').exists())

    def test_closing_scheduler_requests_termination_of_outstanding_worker(self):
        barrier = threading.Barrier(2)
        terminated = threading.Event()
        def solve(name, args, policy):
            barrier.wait(timeout=3)
            if name.endswith('P15_PERFECT'):
                if not args.queue_cancel.wait(timeout=3):
                    raise AssertionError('Outstanding worker was not cancelled')
                terminated.set()
            return {'model': name}
        policies = q.parallel_policies(SimpleNamespace(cpus=12, gpus=1), 2, 2)
        with mock.patch.object(q, 'run_job', side_effect=solve):
            pending = q.solver_results(['PM_A000_P00_PERFECT','PM_A000_P15_PERFECT'],
                                       SimpleNamespace(), policies)
            self.assertEqual(next(pending)[0], 'PM_A000_P00_PERFECT')
            pending.close()
        self.assertTrue(terminated.is_set())


class TimeLimitTests(unittest.TestCase):
    def directory(self):
        path=Path(__file__).resolve().parent/('timeout_test_'+uuid.uuid4().hex)
        path.mkdir();self.addCleanup(shutil.rmtree,path)
        return path

    def test_native_time_limit_stops_at_one_hour_for_both_stop_methods(self):
        for method in ('monitor','urdfil'):
            path=self.directory();name='PM_A000_P75_PERFECT';clock=[0.];alive=[True];commands=[]
            process=SimpleNamespace(poll=lambda:None if alive[0] else 0,returncode=0)
            def launch(command,**kwargs):
                commands.append(command)
                if 'terminate' in command:
                    alive[0]=False
                    (path/(name+'.msg')).write_text('***ERROR: Process terminated by external request')
                    return SimpleNamespace(wait=lambda timeout:0)
                return process
            def sleep(unused):
                clock[0]=3600. if clock[0]<3600. else clock[0]+1.
                if clock[0]>3665.:raise AssertionError('Virtual solver exceeded the bounded test clock')
            args=SimpleNamespace(run_dir=str(path),stop_method=method,ratio=.7,poll_seconds=1)
            with mock.patch.object(q,'launch',side_effect=launch),mock.patch.object(q.time,'monotonic',side_effect=lambda:clock[0]),mock.patch.object(q.time,'sleep',side_effect=sleep),mock.patch.object(q,'report'):
                result=q.run_job(name,args,SimpleNamespace(cpus=8,gpus=0))
            self.assertEqual(len([c for c in commands if 'terminate' in c]),1)
            self.assertTrue(result['time_limit_reached'])
            self.assertTrue(result['time_limit_termination_confirmed'])
            self.assertEqual(result['time_limit_seconds'],3600.)
            self.assertEqual(result['termination_reason'],'TIME_LIMIT')
            self.assertEqual(result['outcome'],'TIME_LIMIT')
            self.assertNotIn('error',result)

    def test_native_shutdown_wait_is_bounded(self):
        path=self.directory();clock=[0.];commands=[]
        process=SimpleNamespace(poll=lambda:None,returncode=None)
        def launch(command,**kwargs):
            commands.append(command)
            return SimpleNamespace(wait=lambda timeout:0) if 'terminate' in command else process
        def sleep(unused):
            clock[0]=3600. if clock[0]<3600. else clock[0]+10.
            if clock[0]>3670.:raise AssertionError('Shutdown exceeded virtual-clock guard')
        args=SimpleNamespace(run_dir=str(path),stop_method='urdfil',ratio=.7,poll_seconds=10)
        with mock.patch.object(q,'launch',side_effect=launch),mock.patch.object(q.time,'monotonic',side_effect=lambda:clock[0]),mock.patch.object(q.time,'sleep',side_effect=sleep),mock.patch.object(q,'report'):
            with self.assertRaisesRegex(q.SolverShutdownError,'shutdown did not finish'):
                q.run_job('PM_STUCK',args,SimpleNamespace(cpus=8,gpus=0))
        self.assertEqual(clock[0],3660.)
        self.assertEqual(len([c for c in commands if 'terminate' in c]),1)

    def test_normal_completion_does_not_request_time_termination(self):
        path=self.directory();name='PM_A000_P75_PERFECT'
        (path/(name+'.odb')).write_bytes(b'opaque result')
        # _run_job_native intentionally refuses occupied inputs, so create on submit.
        (path/(name+'.odb')).unlink()
        def launch(command,**kwargs):
            self.assertNotIn('terminate',command)
            (path/(name+'.odb')).write_bytes(b'opaque result')
            return SimpleNamespace(poll=lambda:0,returncode=0)
        with mock.patch.object(q,'launch',side_effect=launch),mock.patch.object(q,'report'):
            result=q.run_job(name,SimpleNamespace(run_dir=str(path),stop_method='monitor',ratio=.7,poll_seconds=1),SimpleNamespace(cpus=8,gpus=0))
        self.assertFalse(result.get('time_limit_reached',False))

    def timed_result(self,name):
        return dict(model=name,time_limit_reached=True,time_limit_termination_confirmed=True,
                    time_limit_seconds=3600.,elapsed_seconds=3601.,termination_reason='TIME_LIMIT',
                    termination_requested=True,termination_command_exit_code=0,solver_exit_code=0,
                    criterion_reached=False,analysis_state='TERMINATED',analysis_completed=False,
                    completion_sources=['QUEUE_TIME_LIMIT:TERMINATED'],completion_diagnostics=[])

    def test_time_limit_cannot_be_certified_even_with_a_saved_load_drop(self):
        result=self.timed_result('PM_TEST');result['criterion_reached']=True
        result['outcome']=q.outcome(result,True)
        q.capacity_status(result)
        self.assertEqual(result['outcome'],'TIME_LIMIT')
        self.assertFalse(result['capacity_verified'])
        self.assertFalse(q.finished_result(result))
        result['curve_available']=True
        self.assertTrue(q.processed_result(result))

    def test_queue_continues_and_resume_skips_timeouts_without_usable_data(self):
        path=self.directory();names=['PM_A000_P75_PERFECT','PM_A000_P90_PERFECT']
        info=dict(stage=5,load_case=dict(type='combined'),pm_angle_deg=75.,bending_axis_deg=0.,
                  source_case='PERFECT',settings=dict(field_frequency=1,fy=240.))
        for name in names:
            (path/(name+'.inp')).write_text('*Heading')
            (path/(name+'_step5.json')).write_text(json.dumps(info))
        args=SimpleNamespace(run_dir=str(path),jobs=names,cae=None,cpus=1,gpus=0,
            ratio=.7,stop_method='monitor',extract_only=False,repair_only=False,resume=False)
        with mock.patch.object(q,'validate_submission'),mock.patch.object(q.resources,'detect_resources'),mock.patch.object(q.resources,'resolve_solver_policy',return_value=SimpleNamespace(cpus=1,gpus=0)),mock.patch.object(q,'report'),mock.patch.object(q,'run_job',side_effect=lambda name,*a:self.timed_result(name)) as solve,mock.patch.object(q.fd,'extract',side_effect=ValueError('No converged response')):
            q.run(args)
            self.assertEqual(solve.call_count,2)
            records=json.loads((path/'STEP5_queue_status.json').read_text())
            self.assertTrue(all(r['outcome']=='TIME_LIMIT' and not r.get('error') for r in records))
            self.assertTrue(all(q.processed_result(r) for r in records))
            solve.reset_mock();args.resume=True;q.run(args);solve.assert_not_called()
        import abaqus_pm_interaction as pm
        result=pm.collect(pm.parse_arguments(['--mode','collect','--roots',str(path),'--output-dir',str(path/'diagram')]))
        self.assertEqual(result['points'],0)
        audit=json.loads(Path(result['audit']).read_text())
        self.assertEqual(audit['summary']['time_limits'],2)
        self.assertEqual(audit['summary']['omitted_points'],2)
        self.assertTrue((path/'diagram'/'PM_interaction.png').is_file())

    def test_unconfirmed_shutdown_is_not_a_processed_timeout(self):
        result=self.timed_result('PM_TEST');result['time_limit_termination_confirmed']=False
        result['outcome']='TIME_LIMIT';result['curve_available']=False
        self.assertFalse(q.processed_result(result))

    def test_failed_shutdown_stops_scheduler_before_next_submission(self):
        args=SimpleNamespace();policy=SimpleNamespace(cpus=1,gpus=0,memory_percent=100)
        with mock.patch.object(q,'run_job',side_effect=q.SolverShutdownError('native solver still live')) as solve:
            with self.assertRaisesRegex(q.SolverShutdownError,'still live'):
                list(q.solver_results(['PM_FIRST','PM_NEXT'],args,[policy]))
        self.assertEqual(solve.call_count,1)


if __name__ == '__main__':
    unittest.main()
