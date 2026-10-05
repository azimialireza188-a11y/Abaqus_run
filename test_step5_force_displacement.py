import math
import unittest
import abaqus_step5_force_displacement_core as fd


class ForceDisplacementTests(unittest.TestCase):
    def test_live_snapshot_writes_four_outputs_without_changing_sources(self):
        import contextlib, io, json, os, tempfile
        from types import SimpleNamespace as NS
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            model='STEP5_PERFECT_FY240'
            cae=os.path.join(d,'step5.cae'); odb_path=os.path.join(d,model+'.odb')
            lock=os.path.join(d,model+'.lck')
            for path in (cae,odb_path,lock):
                with open(path,'w') as stream: stream.write('source unchanged')
            info={'settings':{'fy':240,'reference_stress':240,'field_frequency':1},
                  'end_area_mm2':4,'reference_force_N_per_end':1000,
                  'end_area_weights':[[['P1',1,1]],[['P1',2,1]]]}
            bottom=object(); top=object(); instance=NS(name='P1')
            def field(i):
                def subset(region):
                    label=1 if region is bottom else 2
                    return NS(values=[NS(instance=instance,nodeLabel=label,
                        precision='SINGLE_PRECISION',data=(0,0,0 if label==1 else -i))])
                return NS(getSubset=subset)
            frames=[NS(frameValue=i,fieldOutputs={'U':field(i)}) for i in range(3)]
            step=NS(frames=frames,historyRegions={'Assembly':NS(historyOutputs={
                'LPF':NS(data=[(0,0),(1,1),(2,.6)])})})
            odb=NS(steps={'GMNIA':step},rootAssembly=NS(nodeSets={
                'STEP5_BOTTOM':bottom,'STEP5_TOP':top}),close=mock.Mock())
            db=NS(models={model:NS(description=fd.STEP5_PREFIX+json.dumps(info))},close=mock.Mock())
            opener=mock.Mock(return_value=odb)
            args=NS(cae=cae,odb=odb_path,model=None,step='GMNIA',output_dir=d,allow_partial=True)
            modules={'caeModules':NS(),'abaqus':NS(openMdb=lambda **kw:db),
                     'odbAccess':NS(openOdb=opener)}
            with mock.patch.dict('sys.modules',modules), contextlib.redirect_stdout(io.StringIO()):
                summary=fd.extract(args)
            opener.assert_called_once_with(path=odb_path,readOnly=True)
            self.assertTrue(summary['partial'])
            self.assertEqual(summary['peak']['force_kN'],1)
            self.assertEqual(summary['peak']['shortening_mm'],1)
            for extension in ('csv','json','svg','png'):
                path=os.path.join(d,model+'_force_displacement_partial.'+extension)
                self.assertGreater(os.path.getsize(path),100)
            for path in (cae,odb_path,lock):
                with open(path) as stream: self.assertEqual(stream.read(),'source unchanged')
            db.close.assert_called_once(); odb.close.assert_called_once()

    def test_partial_flag_allows_readonly_locked_odb(self):
        import contextlib
        import io
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cae=os.path.join(d,'x.cae');odb=os.path.join(d,'x.odb')
            for path in (cae,odb,os.path.join(d,'x.lck')):
                with open(path,'w') as f: f.write('')
            args=fd.parse_arguments(['--cae',cae,'--odb',odb,'--allow-partial'])
            self.assertTrue(args.allow_partial)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                fd.parse_arguments(['--cae',cae,'--odb',odb])

    def test_live_history_ahead_of_fields_uses_synchronized_prefix(self):
        from types import SimpleNamespace as NS
        from unittest import mock
        frames=[NS(frameValue=i,fieldOutputs={'U':object()}) for i in (0,1)]
        step=NS(frames=frames)
        odb=NS(steps={'GMNIA':step},rootAssembly=NS(nodeSets={'STEP5_BOTTOM':object(),'STEP5_TOP':object()}))
        info={'reference_force_N_per_end':1000,'settings':{'reference_stress':240},'end_area_weights':[[['P1',1,1]],[['P1',2,1]]]}
        with mock.patch.object(fd,'lpf_history',return_value=[(0,0),(1,1),(2,.8)]), mock.patch.object(fd,'weighted_u3',return_value=0):
            rows,meta=fd.curve_rows(odb,'GMNIA',info,allow_partial=True)
        self.assertEqual([r['lpf'] for r in rows],[0,1])
        self.assertTrue(meta['partial'])
        self.assertEqual(meta['dropped_history_samples'],1)

    def test_live_history_without_initial_sample_can_lead_at_equal_counts(self):
        from types import SimpleNamespace as NS
        from unittest import mock
        frames=[NS(frameValue=i,fieldOutputs={'U':object()}) for i in (0,1,2)]
        odb=NS(steps={'GMNIA':NS(frames=frames)},rootAssembly=NS(nodeSets={
            'STEP5_BOTTOM':object(),'STEP5_TOP':object()}))
        info={'reference_force_N_per_end':1000,'settings':{'reference_stress':240},
              'end_area_weights':[[['P1',1,1]],[['P1',2,1]]]}
        with mock.patch.object(fd,'lpf_history',return_value=[(1,.4),(2,.8),(3,.9)]), mock.patch.object(fd,'weighted_u3',return_value=0):
            rows,meta=fd.curve_rows(odb,'GMNIA',info,allow_partial=True)
        self.assertEqual([r['lpf'] for r in rows],[0,.4,.8])
        self.assertTrue(meta['partial'])
        self.assertEqual(meta['dropped_history_samples'],1)

    def test_extractor_entry_runs_without_main_namespace_or_file(self):
        from pathlib import Path
        from unittest import mock
        entry=Path(fd.__file__).with_name('abaqus_step5_force_displacement.py')
        calls=[]
        with mock.patch.object(fd,'main',side_effect=lambda: calls.append(1)):
            exec(compile(entry.read_text(),str(entry),'exec'),{'__name__':'abaqus'})
        self.assertEqual(calls,[1])

    def test_weight_map_rejects_duplicate_and_nonpositive(self):
        self.assertEqual(fd.weight_map([['P1', 1, 2.0]]), {('P1', 1): 2.0})
        with self.assertRaises(ValueError):
            fd.weight_map([['P1', 1, 2.0], ['p1', 1, 3.0]])
        with self.assertRaises(ValueError):
            fd.weight_map([['P1', 1, 0.0]])

    def test_weighted_u3_survives_abaqus_sum_shadowing(self):
        import runpy
        from types import SimpleNamespace as NS

        def abaqus_sum(*args, **kwargs):
            raise TypeError("arg1; expecting a recognized Abaqus type")

        namespace = runpy.run_path(
            fd.__file__, init_globals={'sum': abaqus_sum},
            run_name='force_displacement_sum_shadow_test')
        instance = NS(name='P1')
        values = [
            NS(instance=instance, nodeLabel=1, precision='SINGLE_PRECISION', data=(0.0, 0.0, 2.0)),
            NS(instance=instance, nodeLabel=2, precision='SINGLE_PRECISION', data=(0.0, 0.0, 4.0)),
        ]
        field = NS(getSubset=lambda region: NS(values=values))
        weights = {('P1', 1): 1.0, ('P1', 2): 3.0}
        self.assertAlmostEqual(namespace['weighted_u3'](field, object(), weights), 3.5)

    def test_lpf_history_alignment_uses_sequence_when_abscissae_are_offset(self):
        from types import SimpleNamespace as NS
        frames = [NS(frameValue=0.0), NS(frameValue=0.10), NS(frameValue=0.20)]
        history = [(0.0, 0.0), (0.05, 0.75), (0.15, 1.10)]
        values, meta = fd.align_lpf_history(frames, history)
        self.assertEqual(values, [0.0, 0.75, 1.10])
        self.assertEqual(meta['mode'], 'same_count_by_sequence')
        self.assertAlmostEqual(meta['max_abs_frameValue_minus_historyX'], 0.05)

    def test_lpf_history_alignment_accepts_missing_initial_sample(self):
        from types import SimpleNamespace as NS
        frames = [NS(frameValue=0.0), NS(frameValue=0.10), NS(frameValue=0.20)]
        history = [(0.05, 0.75), (0.15, 1.10)]
        values, meta = fd.align_lpf_history(frames, history)
        self.assertEqual(values, [0.0, 0.75, 1.10])
        self.assertEqual(meta['mode'], 'synthesized_initial_zero_then_sequence')

    def test_lpf_history_alignment_rejects_ambiguous_count(self):
        from types import SimpleNamespace as NS
        frames = [NS(frameValue=0.0), NS(frameValue=0.10), NS(frameValue=0.20)]
        with self.assertRaises(ValueError):
            fd.align_lpf_history(frames, [(0.0, 0.0)])

    def test_partial_live_odb_uses_only_synchronized_prefix(self):
        from types import SimpleNamespace as NS

        class FakeField:
            def __init__(self, values):
                self.values = values
            def getSubset(self, region=None):
                return self

        instance = NS(name='P1')
        value = NS(instance=instance, nodeLabel=1, precision='SINGLE_PRECISION', data=(0.0, 0.0, 0.0))
        frames = []
        for i in range(6):
            frames.append(NS(frameValue=float(i), fieldOutputs={'U': FakeField([value])}))
        history = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.2), (3.0, 0.3), (4.0, 0.4)]

        step = NS(frames=frames, historyRegions={})
        odb = NS(
            steps={'GMNIA': step},
            rootAssembly=NS(nodeSets={'STEP5_BOTTOM': object(), 'STEP5_TOP': object()}))

        original_lpf_history = fd.lpf_history
        original_weighted_u3 = fd.weighted_u3
        try:
            fd.lpf_history = lambda _step: history
            fd.weighted_u3 = lambda field, region, weights: 0.0
            info = {
                'end_area_weights': [[['P1', 1, 1.0]], [['P1', 1, 1.0]]],
                'reference_force_N_per_end': 1000.0,
                'settings': {'reference_stress': 240.0},
            }
            rows, meta = fd.curve_rows(odb, 'GMNIA', info, allow_partial=True)
            self.assertTrue(meta['partial'])
            self.assertEqual(meta['dropped_field_frames'], 1)
            self.assertEqual(len(rows), 5)
        finally:
            fd.lpf_history = original_lpf_history
            fd.weighted_u3 = original_weighted_u3

    def test_peak_summary_uses_maximum_force(self):
        rows = [
            dict(frame=0, lpf=0.0, force_N=0.0, force_kN=0.0, shortening_mm=0.0,
                 nominal_stress_MPa=0.0),
            dict(frame=1, lpf=1.2, force_N=12000.0, force_kN=12.0, shortening_mm=2.0,
                 nominal_stress_MPa=288.0),
            dict(frame=2, lpf=0.9, force_N=9000.0, force_kN=9.0, shortening_mm=4.0,
                 nominal_stress_MPa=216.0)]
        peak = fd.peak_summary(rows)
        self.assertEqual(peak['frame'], 1)
        self.assertAlmostEqual(peak['force_kN'], 12.0)
        self.assertAlmostEqual(peak['shortening_mm'], 2.0)

    def test_sparse_metadata_is_rejected_before_reading_odb(self):
        with self.assertRaisesRegex(ValueError, 'field_frequency'):
            fd.curve_rows(None,'GMNIA',{'settings':{'field_frequency':2}})

    def test_extra_nonzero_history_cannot_be_assigned_to_initial_frame(self):
        from types import SimpleNamespace as NS
        with self.assertRaisesRegex(ValueError, 'initial'):
            fd.align_lpf_history([NS(frameValue=0),NS(frameValue=2)],[(0,0),(1,1.2),(2,.7)])

    def test_ticks_are_finite_and_ordered(self):
        ticks = fd._nice_ticks(0.0, 13.0)
        self.assertTrue(ticks)
        self.assertTrue(all(math.isfinite(x) for x in ticks))
        self.assertEqual(ticks, sorted(ticks))


if __name__ == '__main__':
    unittest.main()
