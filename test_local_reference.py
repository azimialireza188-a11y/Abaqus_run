"""Small Local-reference checks; no Abaqus or production-result writes."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

import numpy as np
from gdlc_classifier import gdlc_mode_decomposition as gdlc


class LocalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parent/('local_reference_test_'+uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree,self.root)
        (self.root/'gdlc_reference.json').write_text(json.dumps(dict(kind='L',source_run='source')))

    def row(self,**changes):
        return dict(dict(mode=1,eigenvalue=575.23,G_pct=0.,D_pct=0.,L_pct=98.44,
                         C_pct=0.,O_pct=1.56,dominant='L'),**changes)

    def test_nonfinite_row_after_valid_row_cannot_pass_reference_check(self):
        for key in ('G_pct','D_pct','L_pct','C_pct','O_pct','eigenvalue'):
            with self.subTest(key=key):
                report=gdlc.reference_check(self.root,[self.row(),self.row(**{key:float('nan')})])
                self.assertFalse(report['passed'])

    def test_empty_or_incomplete_reference_is_a_failed_check(self):
        for rows in ([],[dict(eigenvalue=1.,G_pct=0.,D_pct=0.,L_pct=100.,C_pct=0.)]):
            with self.subTest(rows=rows):
                self.assertFalse(gdlc.reference_check(self.root,rows)['passed'])

    def test_local_reference_reports_other_content_and_limited_scope(self):
        report=gdlc.reference_check(self.root,[self.row()])
        self.assertTrue(report['passed'])
        self.assertAlmostEqual(report['max_O_pct'],1.56)
        self.assertEqual(report['interpretation'],'corner_restrained_local_reference')
        self.assertFalse(report['eigenspace_purity_assessed'])

    def test_external_archive_eigenspace_bound_is_honored_when_provided(self):
        rows=[self.row(subspace_L_min_pct=90.),self.row(mode=2,subspace_L_min_pct=90.)]
        report=gdlc.reference_check(self.root,rows)
        self.assertFalse(report['passed'])
        self.assertTrue(report['eigenspace_purity_assessed'])
        report=gdlc.reference_check(self.root,[rows[0],self.row(mode=2)])
        self.assertFalse(report['passed'])
        self.assertFalse(report['eigenspace_purity_assessed'])

    def test_partial_boundary_does_not_silently_remove_a_rounded_corner_row(self):
        row=np.array([1.,0.,-.5,.5])
        keys=[('P',1),('P',2)]
        with self.assertRaisesRegex(ValueError,'(?i)(boundary|constrained|coverage)'):
            gdlc.local_reference_terms(row,0,keys,(1,2),{(('P',1),1)},set())
        # If every nonzero term is already fixed, the equation is implied.
        self.assertEqual(gdlc.local_reference_terms(row,0,keys,(1,2),
            {(('P',1),1),(('P',2),1),(('P',2),2)},set()),[])

    def test_fixed_rhs_is_removed_and_mpc_dependencies_are_rejected(self):
        keys=[('P',1),('P',2)]
        terms=gdlc.local_reference_terms(np.array([1.,0.,-2.,0.]),0,keys,(1,2),
                                         {(('P',2),1)},set())
        self.assertEqual(terms,[(('P',1),1,1.)])
        with self.assertRaisesRegex(ValueError,'(?i)(MPC|dependent)'):
            gdlc.local_reference_terms(np.array([1.,0.,-2.,0.]),0,keys,(1,2),set(),{('P',2)})

    def test_fold_coverage_must_block_each_pieces_three_rigid_motions(self):
        ring=gdlc._bench_ring([(0,0),(10,0),(10,10),(0,10)],True)
        ring.folds=[dict(nodes=[i],ref=i) for i in range(4)]
        report=gdlc.local_reference_rigid_coverage(ring)
        self.assertTrue(report['passed'])
        self.assertEqual(report['ranks'],[3])
        ring.folds=ring.folds[:1]
        report=gdlc.local_reference_rigid_coverage(ring)
        self.assertFalse(report['passed'])
        self.assertEqual(report['ranks'],[2])

    def fixture(self,boundary=''):
        run=self.root/'source';run.mkdir()
        inp=run/'job.inp'
        inp.write_text('*Assembly, name=A\n*End Assembly\n*Shell Section\n1.\n'
                       '*Step, name=B\n*Buckle\n20,0\n'+boundary+'*End Step\n')
        p=np.array([(0,0),(5,0),(10,0),(10,5),(10,10),(5,10),(0,10),(0,5)],float)
        n=len(p);z=np.array([0.,10.,20.])
        xyz=np.array([[x,y,s] for s in z for x,y in p])
        elems=np.array([[r*n+i,r*n+(i+1)%n,(r+1)*n+(i+1)%n,(r+1)*n+i]
                        for r in range(2) for i in range(n)])
        mesh=gdlc.Mesh(np.arange(3*n),np.array(['P']*(3*n)),np.arange(1,3*n+1),xyz,
            elems,np.array([0.,0.,1.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]),
            xyz[:,2],xyz[:,:2],np.repeat(np.arange(3),n),z,20.)
        return run,inp,mesh

    def test_writer_audits_partial_sharp_boundary_at_dof_level(self):
        run,inp,mesh=self.fixture('*Boundary\nP.1,1,1\n')
        with mock.patch.object(gdlc,'choose_export',return_value=(run,{})), \
                mock.patch.object(gdlc,'find_inp',return_value=inp), \
                mock.patch.object(gdlc,'load_mesh',return_value=mesh):
            folder=gdlc.make_reference_models(run,gdlc.Settings(),kinds=('L',))[0]
        report=json.loads((folder/'gdlc_reference.json').read_text())
        coverage=report['constraint_coverage']
        self.assertEqual(coverage['required_rows'],24)
        self.assertEqual(coverage['implied_by_existing_boundaries'],1)
        self.assertEqual(coverage['added_rows'],23)
        self.assertTrue(coverage['complete'])

    def test_untracked_source_eliminations_are_not_silently_reused(self):
        run,inp,mesh=self.fixture()
        text=inp.read_text().replace('*End Assembly','*Equation\n2\nP.1,1,1.,P.2,1,-1.\n*End Assembly')
        inp.write_text(text)
        with mock.patch.object(gdlc,'choose_export',return_value=(run,{})), \
                mock.patch.object(gdlc,'find_inp',return_value=inp), \
                mock.patch.object(gdlc,'load_mesh',return_value=mesh):
            with self.assertRaisesRegex(ValueError,'(?i)(source|equation|dependent)'):
                gdlc.make_reference_models(run,gdlc.Settings(),kinds=('L',))
        self.assertFalse((self.root/'source_REF_L').exists())

    def test_local_only_cli_does_not_build_global_reference(self):
        (self.root/'gdlc_reference.json').unlink()  # Source model, not a reference of a reference.
        inp=self.root/'job.inp';inp.write_text('*Buckle\n20\n')
        with mock.patch.object(gdlc,'find_runs',return_value=[self.root]), \
                mock.patch.object(gdlc,'list_exports',return_value=[(self.root,{})]), \
                mock.patch.object(gdlc,'ensure_export',return_value=None), \
                mock.patch.object(gdlc,'make_reference_models') as make:
            with self.assertRaises(SystemExit) as status:
                gdlc.main([str(self.root),'--make-reference-models','--reference-kinds','L',
                           '--export','never'])
        self.assertEqual(status.exception.code,0)
        self.assertEqual(make.call_args.kwargs['kinds'],('L',))


if __name__=='__main__':
    unittest.main()
