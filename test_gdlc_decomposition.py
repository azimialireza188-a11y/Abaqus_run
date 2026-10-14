"""Small numerical and archive-contract regressions; no Abaqus solve required."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gdlc_classifier import gdlc_mode_decomposition as gdlc
from gdlc_classifier import abaqus_mfsm_export as exporter


def test_plate_motion_is_kept_in_other_energy_field_not_member_kinematic_shares():
    mesh = SimpleNamespace(e1=np.array([1., 0., 0.]), e2=np.array([0., 1., 0.]),
                           instance=np.array(['P1', 'STEEL_PLATE_001']))
    U = np.array([[[1., 2., 3.], [4., 5., 6.]]])
    components = gdlc.component_fields([], mesh, U)
    np.testing.assert_allclose(components[gdlc.CLASSES.index('O'), 1, :, 0], [4., 5.])
    np.testing.assert_allclose(components[:4, 1], 0.)


def test_plate_only_axial_stations_do_not_change_member_integration_weights():
    mesh = SimpleNamespace(ring_z=np.array([0., 25., 50., 75., 100.]))
    group = SimpleNamespace(rings=[0, 2, 4])
    np.testing.assert_allclose(gdlc.member_ring_weights([group], mesh), [25., 0., 50., 0., 25.])


def test_plate_thickness_does_not_override_wall_thickness(tmp_path):
    inp = tmp_path/'plate.inp'
    inp.write_text('*Shell Section, material=Steel\n2., 5\n'
                   '*Shell Section, material=SteelPlate\n10., 9\n')
    parsed = gdlc.read_inp(inp)
    assert parsed.thicknesses == [2.]
    assert parsed.plate_thicknesses == [10.]
    inp.write_text('*Shell Section, material=SteelPlate\nnan, 9\n')
    with pytest.raises(ValueError, match='plate thickness'):
        gdlc.read_inp(inp)


def test_split_plate_bolts_retain_physical_member_seam_diagnostics():
    master = ('STEEL_PLATE_001', 9)
    pairs = [('BEAM', master, ('P2', 10)), ('BEAM', master, ('P3', 20)),
             ('BEAM', ('P1', 2), ('P4', 3))]
    assert gdlc.member_seam_pairs(pairs) == [
        ('BEAM', ('P1', 2), ('P4', 3)), ('BEAM', ('P2', 10), ('P3', 20))]
    assert len(pairs) == 3


def test_selftest_does_not_dispatch_abaqus(tmp_path, monkeypatch):
    monkeypatch.setattr(gdlc, "find_runs", lambda paths: [tmp_path])
    monkeypatch.setattr(gdlc, "list_exports", lambda path: [(tmp_path, {})])
    for name in ("ensure_export", "ensure_stiffness", "solve_reference_model"):
        monkeypatch.setattr(gdlc, name, lambda *a: pytest.fail("self-test dispatched a solver/export"))
    seen = []
    monkeypatch.setattr(gdlc, "process_run", lambda path, st: seen.append(st.selftest_only))
    gdlc.main([str(tmp_path), "--selftest-only", "--solve-reference-models", "--cpus", "auto"])
    assert seen == [True]


def _fake_status(path):
    """odb_status stand-in: 'ok' once a fake solve marked the folder as completed."""
    if (path.parent / 'solved').exists():
        return 'ok'
    return 'failed' if path.exists() else 'missing'


def test_reference_solver_preserves_unfinished_attempt_and_never_touches_running_job(tmp_path, monkeypatch):
    (tmp_path / 'gdlc_reference.json').write_text('{}')
    (tmp_path / 'job.inp').write_text('*Step\n*Buckle, eigensolver=LANCZOS\n2, 0\n*End Step\n')
    (tmp_path / 'job.odb').write_bytes(b'old result')
    (tmp_path / 'job.msg').write_text('interrupted subspace iterations')
    (tmp_path / 'job.simdir').mkdir()
    monkeypatch.setattr(gdlc, 'odb_status', _fake_status)
    monkeypatch.setattr(gdlc, 'abaqus_resources', lambda st: [])
    runs = []
    def fake_run(cmd, cwd):
        runs.append(cmd)
        assert not (cwd / 'job.msg').exists() and not (cwd / 'job.simdir').exists()
        (cwd / 'job.odb').write_bytes(b'new result')
        (cwd / 'solved').touch()
        return 0
    monkeypatch.setattr(gdlc, '_run_cmd', fake_run)
    assert gdlc.solve_reference_model(tmp_path, gdlc.Settings())
    archived = list((tmp_path / 'superseded_attempts').iterdir())
    assert len(archived) == 1 and 'unfinished_failed' in archived[0].name
    assert (archived[0] / 'job.odb').read_bytes() == b'old result'
    assert (archived[0] / 'job.msg').is_file() and (archived[0] / 'job.simdir').is_dir()
    assert (tmp_path / 'job.inp').is_file() and not (archived[0] / 'job.inp').exists()
    assert not gdlc.solve_reference_model(tmp_path, gdlc.Settings()) and len(runs) == 1
    (tmp_path / 'solved').unlink()
    (tmp_path / 'job.lck').write_text('running')
    with pytest.raises(RuntimeError, match='running'):
        gdlc.solve_reference_model(tmp_path, gdlc.Settings())
    assert (tmp_path / 'job.odb').read_bytes() == b'new result' and len(runs) == 1


def test_lanczos_refusal_is_preserved_and_solved_again_with_subspace(tmp_path, monkeypatch):
    (tmp_path / 'gdlc_reference.json').write_text(json.dumps(dict(
        kind='L', n_modes=20, eigensolver='lanczos', min_eigenvalue=0.0, source_min_eigenvalue=None)))
    (tmp_path / 'job.inp').write_text('*Step\n*Buckle, eigensolver=LANCZOS\n20, 0\n*End Step\n')
    monkeypatch.setattr(gdlc, 'odb_status', _fake_status)
    monkeypatch.setattr(gdlc, 'abaqus_resources', lambda st: [])
    def fake_run(cmd, cwd):
        if 'LANCZOS' in (cwd / 'job.inp').read_text():
            (cwd / 'job.dat').write_text(' ***ERROR: THE LANCZOS EIGENSOLVER CANNOT BE USED\n'
                                         '           FOR THIS MODEL\n\n ***NOTE: other\n')
            (cwd / 'job.odb').write_bytes(b'aborted')
            return 1
        (cwd / 'job.odb').write_bytes(b'subspace result')
        (cwd / 'solved').touch()
        return 0
    monkeypatch.setattr(gdlc, '_run_cmd', fake_run)
    assert gdlc.lanczos_refusal(tmp_path, 'job') is None
    assert gdlc.solve_reference_model(tmp_path, gdlc.Settings())
    text = (tmp_path / 'job.inp').read_text()
    assert '*Buckle, eigensolver=SUBSPACE\n20, , 80, 1250' in text
    meta = json.loads((tmp_path / 'gdlc_reference.json').read_text())
    assert meta['lanczos_refused'] and meta['eigensolver'] == 'subspace' and meta['min_eigenvalue'] is None
    assert 'LANCZOS EIGENSOLVER CANNOT BE USED FOR THIS MODEL' in meta['lanczos_refusal']
    assert meta['input_sha256'] == gdlc.file_sha256(tmp_path / 'job.inp')
    archived = next((tmp_path / 'superseded_attempts').iterdir())
    assert 'lanczos_refused' in archived.name and (archived / 'job.dat').is_file()
    assert (tmp_path / 'job.odb').read_bytes() == b'subspace result'


def test_active_buckling_constraints_use_case_two_not_union(tmp_path):
    p = tmp_path / "constraints.inp"
    p.write_text("*Assembly, name=A\n*End Assembly\n*Boundary\nP.1,1,1\n"
                 "*Step, name=B, perturbation\n*Buckle\n2\n"
                 "*Boundary, op=NEW, load case=1\nP.2,2,2\n"
                 "*Boundary, op=NEW, load case=2\nP.3,3,3\n*End Step\n")
    info = gdlc.read_inp(p)
    assert info.boundaries == [([("P", 3)], {3})]


def test_perturbation_constraints_apply_when_no_case_two_exists(tmp_path):
    p = tmp_path / "constraints.inp"
    p.write_text("*Boundary\nP.1,1,1\n*Step, name=B\n*Buckle\n2\n"
                 "*Boundary, op=NEW, load case=1\nP.2,2,2\n*End Step\n")
    assert gdlc.read_inp(p).boundaries == [([("P", 2)], {2})]


def test_windows_process_policy_obeys_platform_limit_and_keeps_threads(monkeypatch):
    import runtime_resources as r
    monkeypatch.setattr(r.os, "name", "nt")
    assert r.process_workers(250, cpus=192) <= 61
    assert r.process_workers(2, cpus=192) == 2


def test_ram_detection_does_not_need_optional_psutil(monkeypatch):
    import runtime_resources as r
    monkeypatch.setattr(r, "available_memory", lambda: (123456, 98765))
    monkeypatch.setattr(gdlc, "psutil", None)
    assert gdlc._available_ram() == 98765


def test_report_flag_counts_match_whole_tokens_once_per_mode():
    rows = [dict(flags="ENERGY_CROSS_TERM;ENERGY_INCOMPATIBLE_COMPONENTS;CLUSTER_6"),
            dict(flags="CROSS_TERM;ENERGY_CROSS_TERM;ENERGY_CROSS_TERM"),
            dict(flags="ENERGY_BEYOND_SECTION;NEW_DIAGNOSTIC"), dict(flags="")]
    counts = gdlc.report_flag_counts(rows)
    assert counts["CROSS_TERM"] == 1
    assert counts["ENERGY_CROSS_TERM"] == 2
    assert counts["ENERGY_INCOMPATIBLE_COMPONENTS"] == 1
    assert counts["ENERGY_BEYOND_SECTION"] == 1
    assert counts["NEW_DIAGNOSTIC"] == 1
    assert counts["LOW_MARGIN"] == 0
    assert "CLUSTER_6" not in counts


def test_report_separates_kinematic_energy_and_combined_confidence():
    rows = [dict(confidence="low", kinematic_confidence="high", energy_confidence="low",
                 dominant="L", G_pct=0., D_pct=0., L_pct=100., C_pct=0., O_pct=0.,
                 cross_term_pct=0.)]
    expected = dict(combined=dict(high=0, medium=0, low=1),
                    kinematic=dict(high=1, medium=0, low=0),
                    energy=dict(high=0, medium=0, low=1))
    assert gdlc.reliability_summary(rows)["confidence_by_measure"] == expected
    stats = gdlc.dominance_stats(rows)
    assert stats["confidence_by_measure"] == expected
    assert stats["confidence_counts"] == expected["combined"]


def test_report_does_not_invent_energy_confidence_without_energy_evaluation():
    rows = [dict(confidence="high", G_pct=0., D_pct=0., L_pct=100., C_pct=0., O_pct=0.,
                 cross_term_pct=0.)]
    result = gdlc.reliability_summary(rows)["confidence_by_measure"]
    assert result["energy"] is None
    assert result["kinematic"] == result["combined"] == dict(high=1, medium=0, low=0)


def test_modal_arrays_can_spill_without_changing_values(monkeypatch):
    monkeypatch.setattr(gdlc, "_available_ram", lambda: 1)
    a = gdlc._mode_array((2, 3, 3))
    assert isinstance(a, np.memmap)
    a[:] = np.arange(18).reshape(a.shape)
    np.testing.assert_array_equal(a, np.arange(18).reshape(a.shape))


@pytest.mark.parametrize("points", [
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)],
])
def test_fold_free_rigid_directions_have_unique_decomposition(points):
    ring = gdlc._bench_ring(points, False)
    gdlc.detect_reference_points(ring, 0.1, gdlc.Settings())
    basis = gdlc.build_basis(ring)
    assert basis["T"].shape == (2 * len(points), 2 * len(points))
    assert gdlc.self_test(ring)["passed"]
    rigid = np.tile([2.0, -3.0], len(points))
    np.testing.assert_allclose(ring.proj["G"] @ rigid, rigid, atol=1e-9)
    assert gdlc._shares(ring, rigid)["G"] == pytest.approx(100)


def test_empty_eigenvalue_set_has_no_clusters():
    assert gdlc.clusters_of(np.array([]), 0.001) == []


def test_missing_raw_translation_is_rejected(tmp_path):
    # One node, only U1/U2: U3 must never be silently manufactured as zero.
    np.savez(tmp_path / "raw_dof_map.npz", instances=["P", "P"], labels=[1, 1], dofs=[1, 2])
    np.savez(tmp_path / "modes_0001.npz", vectors=[[1.0], [2.0]], modes=[1])
    meta = dict(modes=[dict(mode=1)], shards=["modes_0001.npz"], modes_per_shard=1)
    mesh = SimpleNamespace(node_index=np.array([0]), instance=np.array(["P"]), label=np.array([1]))
    with pytest.raises((ValueError, RuntimeError), match="(?i)(missing|incomplete|translation)"):
        gdlc._load_modes_npz(tmp_path, meta, mesh, gdlc.Settings())


def test_cluster_shares_do_not_depend_on_eigenvector_scaling():
    # Orthogonal pure G and pure L vectors spanning the same 2D eigenspace.
    # Arbitrary eigenvector scaling is not grounds to discard one direction.
    H = {k: np.zeros((2, 2)) for k in gdlc.CLASSES}
    H["G"][0, 0] = 1e-16
    H["L"][1, 1] = 1.0
    res = dict(H=H, Htot=np.diag([1e-16, 1.0]),
               E=np.array([[1e-16, 0, 0, 0, 0], [0, 0, 1, 0, 0]]))
    got = gdlc.reliable_shares(res, np.array([10.0, 10.0]), gdlc.Settings())
    np.testing.assert_allclose(got[:, [0, 2]], 50.0, atol=1e-10)


def _orthogonal_family_summary(families, transform=None):
    n = len(families)
    V = np.eye(n) if transform is None else np.asarray(transform)
    H = {k: V.T @ np.diag([float(f == k) for f in families]) @ V for k in gdlc.CLASSES}
    total = V.T @ V
    E = np.stack([np.diag(H[k]) for k in gdlc.CLASSES], axis=1)
    res = dict(H=H, Hs=H, Htot=total, E=E, Es=E, Etot=np.diag(total),
               prof=np.repeat(np.vstack([E.T, np.diag(total)[None]])[:, None, :], 3, axis=1),
               dz=np.ones(3))
    return gdlc.summarise(res, list(range(1, n+1)), np.ones(n)*10., 1., [],
                          SimpleNamespace(ring_z=np.array([0., 1., 2.])), gdlc.Settings())


def test_near_equal_separate_global_and_local_modes_are_not_claimed_as_coupled_modes():
    # A group containing seven pure G and one pure L has a misleading 87.5% G mean.
    rows, clusters = _orthogonal_family_summary(['G']*7+['L'])
    assert rows[-1]['label'] == 'L' and rows[0]['label'] == 'G'
    assert all(row['reliable_dominant'] == 'UNRESOLVED' for row in rows)
    assert all(row['reliable_character'] == 'SUBSPACE_UNRESOLVED' for row in rows)
    assert all('SUBSPACE_DOMINANCE_UNRESOLVED' in row['flags'] for row in rows)
    assert all(row['confidence'] == 'low' for row in rows)
    assert clusters[0]['subspace_L_min_pct'] == pytest.approx(0.)
    assert clusters[0]['subspace_L_max_pct'] == pytest.approx(100.)
    assert all(row['first_share50_mode'] is None for row in gdlc.mechanism_critical(rows))
    assert gdlc.select_top_modes(rows,'L',1) == [7]


def test_cluster_family_ranges_are_basis_invariant_and_keep_pure_local_doublets():
    families = ['G', 'L']
    angle = .37
    transform = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ np.diag([1e-6, 1e3])
    a, _ = _orthogonal_family_summary(families)
    b, _ = _orthogonal_family_summary(families, transform)
    for family in gdlc.CLASSES:
        for bound in ('min', 'max'):
            key = 'subspace_%s_%s_pct' % (family, bound)
            assert a[0][key] == pytest.approx(b[0][key], abs=1e-8)
    local, _ = _orthogonal_family_summary(['L', 'L'], transform)
    assert all(row['reliable_dominant'] == 'L' and row['confidence'] == 'high' for row in local)


def test_mixed_kinematics_is_explicit_and_c_is_not_generic_coupling():
    shares = dict(G=0., D=25., L=75., C=0., O=0.)
    mixed = gdlc._assess(shares, shares, gdlc.Settings())
    assert mixed['character'] == 'MIXED_KINEMATICS'
    assert mixed['label'] == 'L+D' and 'MIXED' in mixed['flags']
    relative = dict(G=0., D=0., L=0., C=100., O=0.)
    assert gdlc._assess(relative, relative, gdlc.Settings())['character'] == 'DOMINANT_C'


def test_cluster_average_above_fifty_is_not_per_mode_fifty_percent_evidence():
    E = np.array([[.45,.40,.15,0.,0.],[.95,.05,0.,0.,0.]])
    H = {k:np.diag(E[:,j]) for j,k in enumerate(gdlc.CLASSES)}
    res = dict(H=H,Hs=H,Htot=np.eye(2),E=E,Es=E,Etot=np.ones(2),dz=np.ones(3),
               prof=np.repeat(np.vstack([E.T,np.ones((1,2))])[:,None,:],3,axis=1))
    rows,_ = gdlc.summarise(res,[1,2],np.array([10.,10.]),1.,[],
                            SimpleNamespace(ring_z=np.array([0.,1.,2.])),gdlc.Settings())
    assert rows[0]['reliable_dominant'] == 'G'
    global_result = next(row for row in gdlc.mechanism_critical(rows) if row['mechanism']=='G')
    assert global_result['first_share50_mode'] is None


@pytest.mark.parametrize('family', ['G', 'L'])
def test_cluster_other_maximum_and_reference_purity_cannot_be_hidden_by_rotation(tmp_path, family):
    rotation = np.array([[1., 1.], [-1., 1.]])/np.sqrt(2.)
    H = {k: np.zeros((2, 2)) for k in gdlc.CLASSES}
    H[family] = rotation.T @ np.diag([1., .92]) @ rotation
    H['O'] = rotation.T @ np.diag([0., .08]) @ rotation
    E = np.stack([np.diag(H[k]) for k in gdlc.CLASSES], axis=1)
    res = dict(H=H, Hs=H, Htot=np.eye(2), E=E, Es=E, Etot=np.ones(2), dz=np.ones(3),
               prof=np.repeat(np.vstack([E.T, np.ones((1, 2))])[:, None, :], 3, axis=1))
    rows, _ = gdlc.summarise(res, [1, 2], np.array([10., 10.]), 1., [],
                            SimpleNamespace(ring_z=np.array([0., 1., 2.])), gdlc.Settings())
    # Both sampled vectors have only4% O; the same subspace contains an8% O vector.
    assert rows[0]['O_pct'] == pytest.approx(4.)
    assert rows[0]['subspace_O_max_pct'] == pytest.approx(8.)
    assert all(row['confidence'] == 'low' and 'UNRELIABLE_OTHER' in row['flags'] for row in rows)
    assert all(row['reliable_character'] == 'UNRESOLVED' for row in rows)
    (tmp_path/'gdlc_reference.json').write_text(json.dumps(dict(kind=family)))
    assert not gdlc.reference_check(tmp_path, rows)['passed']


def test_direct_and_plate_bolts_with_reversed_endpoints_share_one_diagnostic_seam():
    p = np.array([[0., 0.], [1., 0.], [0., 2.], [1., 2.]])
    mesh = SimpleNamespace(instance=np.array(['P1','P1','P2','P2']*2+['STEEL_PLATE_001']),
        label=np.array([1,2,1,2,3,4,3,4,1]), p=np.vstack([p,p,[[0.,1.]]]),
        s=np.array([0.]*4+[10.]*4+[10.]), size=10.)
    ring = gdlc.RingGeom(n=4, p=p, inst=mesh.instance[:4], edges=np.array([[0,1],[2,3]]),
                         rings=[0,1], node_rows=np.array([[0,1,2,3],[4,5,6,7]]))
    pairs = [('BEAM', ('P1',1), ('P2',1)),
             ('BEAM', ('STEEL_PLATE_001',1), ('P2',3)),
             ('BEAM', ('STEEL_PLATE_001',1), ('P1',3))]
    gdlc.attach_seams(ring, mesh, gdlc.InpInfo(pairs=pairs))
    assert len(ring.seams) == 1
    assert ring.seams[0]['n_links'] == 2 and ring.seams[0]['z'] == [0.,10.]
    assert ring.seams[0]['A'] == 0 and ring.seams[0]['B'] == 2
    assert len(pairs) == 3


def test_plate_cluster_guard_is_invariant_to_mixing_near_equal_modes(tmp_path):
    xyz = np.array([[0.,0.,0.],[1.,0.,0.],[1.,0.,1.],[0.,0.,1.],
                    [0.,.5,0.],[1.,.5,0.],[1.,.5,1.],[0.,.5,1.]])
    mesh = SimpleNamespace(instance=np.array(['P1']*4+['STEEL_PLATE_001']*4), xyz=xyz,
                           elems=np.array([[0,1,2,3],[4,5,6,7]]))
    U = np.zeros((2,8,3))
    U[0,:4,0] = np.sqrt(11.5)
    U[0,4:,1] = [1.,-1.,1.,-1.]  # 8% plate deformation, mode2 has0%
    U[1,:4,2] = np.sqrt(12.5)
    rotation = np.array([[1.,1.],[-1.,1.]])/np.sqrt(2.)
    rotated = np.einsum('ij,jnc->inc',rotation,U) # 4% each; subspace maximum still8%
    for field in (U, rotated):
        shares = gdlc.plate_motion_shares(mesh,field,clusters=[[0,1]])
        rows = [dict(cluster=1,flags='',confidence='high') for _ in range(2)]
        crow = [dict(cluster=1,flags='',confidence='high')]
        gdlc.add_plate_columns(rows,crow,shares,gdlc.Settings())
        assert all('PLATE_DEFORMATION' in row['flags'] for row in rows)
        assert crow[0]['cluster_plate_deformation_max_pct'] == pytest.approx(8.)
        (tmp_path/'gdlc_reference.json').write_text(json.dumps(dict(kind='G')))
        for row in rows:
            row.update(G_pct=100.,eigenvalue=10.)
        assert not gdlc.reference_check(tmp_path,rows)['passed']


def test_bulk_export_rejects_local_coordinates_without_orientation_width():
    block = SimpleNamespace(precision="DOUBLE_PRECISION", dataDouble=((1., 2., 3.),),
                            localCoordSystemDouble=((1., 0., 0., 0.),))
    with pytest.raises(ValueError, match="Local-coordinate"):
        exporter._block_array(block)


def test_non_shell_four_node_elements_are_rejected(tmp_path):
    inp = tmp_path / "mesh.inp"
    inp.write_text("*Part, name=P\n*Element, type=C3D4\n1, 1, 2, 3, 4\n*End Part\n"
                   "*Assembly, name=A\n*Instance, name=P-1, part=P\n*End Instance\n*End Assembly\n")
    with pytest.raises(RuntimeError, match="(?i)shell"):
        gdlc.inp_shell_elements(inp)


def test_stale_export_cannot_be_classified_as_current_run(tmp_path):
    (tmp_path / "job.inp").write_text("changed input")
    export = tmp_path / "modal_export"
    export.mkdir()
    for name in ("raw_dof_map.npz", "modes_0001.npz"):
        (export / name).touch()
    (export / "modal_export.json").write_text(json.dumps(dict(
        shards=["modes_0001.npz"], mode_count=1, source_inp_sha256="stale")))
    with pytest.raises(RuntimeError, match="(?i)(match|stale|differ)"):
        gdlc.choose_export(tmp_path, None)


def test_double_bulk_unavailable_falls_back_to_field_values():
    instance = SimpleNamespace(name="P")
    value = SimpleNamespace(instance=instance, nodeLabel=1, precision="DOUBLE_PRECISION",
                            dataDouble=(1., 2., 3.))
    # The Python bulk API may not expose double precision; FieldValue does.
    block = SimpleNamespace(instance=instance, nodeLabels=[1], precision="DOUBLE_PRECISION")
    field = SimpleNamespace(bulkDataBlocks=[block], values=[value])
    frame = SimpleNamespace(fieldOutputs={"U": field, "UR": field})
    indexers, _ = exporter._node_indexers([("P", 1)])
    vector, _, backends = exporter._read_mode(frame, indexers, 1)
    np.testing.assert_array_equal(vector, [1., 2., 3., 1., 2., 3.])
    assert backends == ["values", "values"]


def _parse_equations(text):
    lines, eqs, i = text.splitlines(), [], 0
    while i < len(lines):
        if lines[i].strip().lower().startswith("*equation"):
            n, terms, j = int(lines[i + 1]), [], i + 2
            while len(terms) < n:
                parts = [p.strip() for p in lines[j].split(",") if p.strip()]
                assert len(parts) <= 12                       # at most 4 terms per data line
                for k in range(0, len(parts), 3):
                    inst, label = parts[k].split(".")
                    terms.append(((inst, int(label)), int(parts[k + 1]), float(parts[k + 2])))
                j += 1
            eqs.append(terms)
            i = j
        else:
            i += 1
    return eqs


def _reference_run(tmp_path, monkeypatch, boundary=""):
    run = tmp_path / "run"
    run.mkdir()
    inp = run / "job.inp"
    inp.write_text("*Assembly, name=A\n*End Assembly\n*Shell Section\n0.1\n"
                   "*Step, name=B\n*Buckle\n2\n" + boundary + "*End Step\n")
    p = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    mesh = gdlc.Mesh(np.arange(12), np.array(["P"] * 12), np.arange(1, 13),
                     np.array([[x, y, z] for y in (0., 10., 20.) for x, z in p]),
                     np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
                               [4, 5, 9, 8], [5, 6, 10, 9], [6, 7, 11, 10], [7, 4, 8, 11]]),
                     np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 1.]),
                     np.repeat([0., 10., 20.], 4), np.tile(p, (3, 1)), np.repeat([0, 1, 2], 4),
                     np.array([0., 10., 20.]), 20.)
    monkeypatch.setattr(gdlc, "choose_export", lambda *a: (run, {}))
    monkeypatch.setattr(gdlc, "find_inp", lambda *a: inp)
    monkeypatch.setattr(gdlc, "load_mesh", lambda *a: mesh)
    return run, mesh


@pytest.mark.parametrize('skip', [0, 1, 2, 3])
def test_unbounded_references_use_positive_lanczos_and_keep_twist(tmp_path, monkeypatch, skip):
    run, mesh = _reference_run(tmp_path, monkeypatch)
    scale = 10. ** skip
    mesh.xyz[:, [0, 2]] *= scale
    mesh.p *= scale
    (run/'job_build.json').write_text(json.dumps(dict(load_case=dict(type='compression'),
        settings=dict(steel_plate_skip_count=skip))))
    outputs = gdlc.make_reference_models(run, gdlc.Settings(), n_modes=20)
    for output in outputs:
        text = next(output.glob('*.inp')).read_text()
        # compression source: Lanczos for the 20 lowest eigenvalues >= 0 (no negative search,
        # no subspace stall on dense local spectra)
        assert '*Buckle, eigensolver=LANCZOS\n20, 0\n' in text
        info = json.loads((output/'gdlc_reference.json').read_text())
        assert info['eigensolver'] == 'lanczos' and info['min_eigenvalue'] == 0.0
        assert info['source_min_eigenvalue'] is None and info['n_modes'] == 20
        if info['kind'] == 'G':
            assert info['twist'] == 'restrained'
            assert info['formulation'] == gdlc.REF_G_FORMULATION_NO_TWIST


def test_governing_global_candidate_is_retained_without_forcing_confidence():
    def row(mode, eigenvalue, share, confidence):
        return dict(mode=mode,eigenvalue=eigenvalue,sigma_cr_MPa=eigenvalue,
            G_pct=share,D_pct=100.-share,L_pct=0.,C_pct=0.,dominant='G',
            reliable_dominant='G',confidence=confidence)
    rows = [row(9,90.,99.,'high'),row(1,10.,60.,'medium')]
    original = json.dumps(rows)
    critical = next(r for r in gdlc.mechanism_critical(rows) if r['mechanism']=='G')
    assert critical['first_dominant_mode'] == 1
    assert 1 in [rows[i]['mode'] for i in gdlc.select_top_modes(rows,'G',1)]
    rows[1]['confidence'] = 'low'
    critical = next(r for r in gdlc.mechanism_critical(rows) if r['mechanism']=='G')
    assert critical['first_candidate_mode'] == 1
    assert critical['first_candidate_confidence'] == 'low'
    assert critical['first_dominant_mode'] == 9
    assert 1 in [rows[i]['mode'] for i in gdlc.select_top_modes(rows,'G',1)]
    rows[1]['confidence'] = 'medium'
    assert json.dumps(rows) == original


def test_flexural_only_reference_is_not_certified_as_full_global(tmp_path):
    rows = [dict(mode=1,eigenvalue=10.,G_pct=100.)]
    for twist, eligible in [('restrained',False),('free',True)]:
        (tmp_path/'gdlc_reference.json').write_text(json.dumps(dict(kind='G',twist=twist)))
        check = gdlc.reference_check(tmp_path, rows)
        assert check['passed']
        assert check['first_eigenvalue'] == 10.
        assert check['governing_global_eligible'] is eligible


def test_rigid_reference_rings_are_equations_in_the_section_plane(tmp_path, monkeypatch):
    # Member axis = y, so the in-plane DOFs are 1 and 3 and the axial DOF 2 stays free.
    # The end ring y = 0 carries an in-plane BC and is therefore left alone.
    run, mesh = _reference_run(tmp_path, monkeypatch,
                               "*Boundary, op=NEW, load case=1\nP.1, 1, 3\nP.2, 1, 3\nP.3, 1, 3\nP.4, 1, 3\n")
    out = gdlc.make_reference_models(run, gdlc.Settings(ref_g_twist="free"), kinds=("G",))[0]
    text = next(out.glob("*.inp")).read_text()
    keywords = {k for k, _, _, _ in exporter._inp_blocks(text)}
    assert not keywords & {"COUPLING", "KINEMATIC", "NODE"}       # no reference nodes / couplings
    assert "eigensolver=LANCZOS" in text
    info = json.loads((out / "gdlc_reference.json").read_text())
    assert info["formulation"] == gdlc.REF_G_FORMULATION and info["n_rings_rigid"] == 2
    eqs = _parse_equations(text)
    assert len(eqs) == 2 * (2 * 4 - 3)                            # 3 rigid DOFs left per ring
    dependents = [t[0][:2] for t in eqs]
    assert len(set(dependents)) == len(dependents)
    assert all(t[0][2] == 1.0 for t in eqs)
    assert not set(dependents) & {(n, d) for t in eqs for n, d, _ in t[1:]}
    assert all(d in (1, 3) for t in eqs for _, d, _ in t)
    assert all(n[1] > 4 for t in eqs for n, _, _ in t)            # BC ring untouched
    xy = {("P", int(l)): mesh.p[i] for i, l in enumerate(mesh.label)}
    rng = np.random.default_rng(1)
    motion = {r: rng.normal(size=3) for r in (1, 2)}               # per ring: t1, t3, rotation
    def u(node, dof):
        t1, t3, th = motion[(node[1] - 1) // 4]
        x, z = xy[node]
        return t1 - th * z if dof == 1 else t3 + th * x
    for t in eqs:
        assert abs(sum(c * u(n, d) for n, d, c in t)) < 1e-12


def test_rigid_ring_masters_prefer_bc_nodes_and_avoid_mpc_independent_nodes():
    pts = np.array([[0., 0., 0.], [3., 0., 0.], [3., 2., 0.], [0., 2., 0.], [1.5, 1., 0.]])
    ring = [(("P", i + 1), p) for i, p in enumerate(pts)]
    lines, n = gdlc.rigid_ring_equations(ring, [1, 2], mpc_independent={("P", 3)}, bc_nodes={("P", 5)})
    eqs = _parse_equations("\n".join(lines))
    assert n == len(eqs) == 2 * 5 - 3
    dependents = {t[0][0] for t in eqs}
    assert ("P", 5) not in dependents                              # BC node = master a
    masters = {n_ for t in eqs for n_, _, _ in t[1:]}
    assert ("P", 3) not in masters                                 # MPC-independent node is not a master


def test_old_coupling_lanczos_refusal_does_not_force_subspace(tmp_path, monkeypatch):
    run, _ = _reference_run(tmp_path, monkeypatch)
    inp = run / 'job.inp'
    inp.write_text(inp.read_text().replace('*Buckle\n2', '*Buckle, eigensolver=LANCZOS\n2, 0.'))
    ref = tmp_path / "run_REF_G"
    ref.mkdir()
    (ref / "gdlc_reference.json").write_text(json.dumps(dict(kind="G", lanczos_refused=True)))
    for twist, formulation in (("free", gdlc.REF_G_FORMULATION), ("restrained", gdlc.REF_G_FORMULATION_NO_TWIST)):
        (ref / "gdlc_reference.json").write_text(json.dumps(dict(kind="G", lanczos_refused=True)))
        gdlc.make_reference_models(run, gdlc.Settings(ref_g_twist=twist), kinds=("G",))
        assert "eigensolver=LANCZOS" in next(ref.glob("*.inp")).read_text()
        (ref / "gdlc_reference.json").write_text(json.dumps(
            dict(kind="G", lanczos_refused=True, formulation=formulation)))
        gdlc.make_reference_models(run, gdlc.Settings(ref_g_twist=twist), kinds=("G",))
        assert "eigensolver=LANCZOS" not in next(ref.glob("*.inp")).read_text()


def test_regenerated_reference_preserves_failed_attempt_and_refuses_completed(tmp_path, monkeypatch):
    run, _ = _reference_run(tmp_path, monkeypatch)
    out = gdlc.make_reference_models(run, gdlc.Settings(), kinds=("L",))[0]
    inp = next(out.glob("*.inp"))
    job = inp.stem
    current = inp.read_text()
    old = current.replace('*Buckle, eigensolver=LANCZOS\n20, 0\n', '*Buckle, eigensolver=SUBSPACE\n20, , 80, 1250\n')
    assert old != current
    inp.write_text(old)                                           # earlier generator, interrupted solve
    (out / (job + ".msg")).write_text("no eigenvalue converged")
    (out / (job + ".odb")).write_bytes(b"partial")
    (out / (job + ".lck")).write_text("running")
    with pytest.raises(FileExistsError, match="running"):
        gdlc.make_reference_models(run, gdlc.Settings(), kinds=("L",))
    assert inp.read_text() == old and (out / (job + ".odb")).read_bytes() == b"partial"
    (out / (job + ".lck")).unlink()
    monkeypatch.setattr(gdlc, "odb_status", lambda p: "failed" if p.exists() else "missing")
    gdlc.make_reference_models(run, gdlc.Settings(), kinds=("L",))
    assert inp.read_text() == current and not (out / (job + ".odb")).exists()
    archived = next((out / "superseded_attempts").iterdir())
    assert (archived / (job + ".odb")).read_bytes() == b"partial"
    assert (archived / (job + ".inp")).read_text() == old
    assert (archived / "gdlc_reference_of_this_attempt.json").is_file()
    (archived / (job + "_build.json")).write_text("{}")             # even a complete-looking archive
    (archived / "gdlc_reference.json").write_text("{}")
    assert archived not in gdlc.find_runs([str(tmp_path)]) and out in gdlc.find_runs([str(tmp_path)])
    (archived / (job + "_build.json")).unlink()
    (archived / "gdlc_reference.json").unlink()
    gdlc.make_reference_models(run, gdlc.Settings(), kinds=("L",))   # identical input: untouched
    assert len(list((out / "superseded_attempts").iterdir())) == 1
    (out / (job + ".odb")).write_bytes(b"completed")
    monkeypatch.setattr(gdlc, "odb_status", lambda p: "ok" if p.exists() else "missing")
    inp.write_text(old)
    with pytest.raises(FileExistsError, match="never overwritten"):
        gdlc.make_reference_models(run, gdlc.Settings(), kinds=("L",))
    assert (out / (job + ".odb")).read_bytes() == b"completed" and inp.read_text() == old


def test_reference_kinds_option_selects_the_generated_references(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(gdlc, "find_runs", lambda paths: [tmp_path])
    monkeypatch.setattr(gdlc, "list_exports", lambda path: [(tmp_path, {})])
    monkeypatch.setattr(gdlc, "ensure_export", lambda *a: None)
    monkeypatch.setattr(gdlc, "make_reference_models", lambda r, st, kinds, n_modes: seen.append((kinds, n_modes)))
    with pytest.raises(SystemExit) as stop:
        gdlc.main([str(tmp_path), "--make-reference-models", "--reference-kinds", "l", "--reference-modes", "20"])
    assert stop.value.code == 0 and seen == [(("L",), 20)]
    with pytest.raises(SystemExit) as stop:
        gdlc.main([str(tmp_path), "--make-reference-models"])
    assert stop.value.code == 0 and seen[-1] == (("G", "L"), 20)
    for bad in ("X", "L,L", ","):
        with pytest.raises(SystemExit) as stop:
            gdlc.main([str(tmp_path), "--make-reference-models", "--reference-kinds", bad])
        assert stop.value.code == 2


@pytest.mark.parametrize("load_case, expected", [(None, "restrained"), ("compression", "restrained"),
                                                 ("bending", "free"), ("combined", "free")])
def test_ref_g_auto_restores_compression_twist_restraint(tmp_path, load_case, expected):
    if load_case:
        (tmp_path / "job_build.json").write_text(json.dumps(dict(load_case=dict(type=load_case))))
    assert gdlc.ref_g_twist_mode(tmp_path, "auto")[0] == expected
    assert gdlc.ref_g_twist_mode(tmp_path, "free")[0] == "free"
    assert gdlc.ref_g_twist_mode(tmp_path, "restrained")[0] == "restrained"


def test_twist_restrained_reference_rings_only_translate(tmp_path, monkeypatch):
    # Member axis = y (in-plane DOFs 1 and 3); end ring y = 0 has in-plane BCs and is left alone.
    run, mesh = _reference_run(tmp_path, monkeypatch,
                               "*Boundary, op=NEW, load case=1\nP.1, 1, 3\nP.2, 1, 3\nP.3, 1, 3\nP.4, 1, 3\n")
    out = gdlc.make_reference_models(run, gdlc.Settings(ref_g_twist='restrained'), kinds=("G",))[0]
    info = json.loads((out / "gdlc_reference.json").read_text())
    assert info["twist"] == "restrained" and info["formulation"] == gdlc.REF_G_FORMULATION_NO_TWIST
    text = next(out.glob("*.inp")).read_text()
    assert "eigensolver=LANCZOS" in text
    eqs = _parse_equations(text)
    assert len(eqs) == 2 * 2 * (4 - 1)                            # 2 rigid DOFs (translations) left per ring
    dependents = [t[0][:2] for t in eqs]
    assert len(set(dependents)) == len(dependents)
    assert not set(dependents) & {(n, d) for t in eqs for n, d, _ in t[1:]}
    assert all(len(t) == 2 and d in (1, 3) for t in eqs for _, d, _ in t)
    assert all(n[1] > 4 for t in eqs for n, _, _ in t)            # BC ring untouched
    xy = {("P", int(l)): mesh.p[i] for i, l in enumerate(mesh.label)}
    def u(node, dof, t1, t3, th):
        x, z = xy[node]
        return t1 - th * z if dof == 1 else t3 + th * x
    rng = np.random.default_rng(2)
    for _ in range(3):
        t1, t3 = rng.normal(size=2)
        for t in eqs:                                             # any in-plane translation is admissible
            assert abs(sum(c * u(n, d, t1, t3, 0.0) for n, d, c in t)) < 1e-12
    # a rotation about the member axis (twist) violates the equations
    assert max(abs(sum(c * u(n, d, 0.0, 0.0, 1e-3) for n, d, c in t)) for t in eqs) > 1e-4


def test_local_reference_with_only_other_content_does_not_pass(tmp_path):
    (tmp_path / "gdlc_reference.json").write_text(json.dumps(dict(kind="L", source_run="source")))
    rows = [dict(G_pct=0., D_pct=0., C_pct=0., L_pct=0., O_pct=100.,
                 eigenvalue=1., dominant="L")]
    assert not gdlc.reference_check(tmp_path, rows)["passed"]


@pytest.mark.parametrize("indices", [[0], [0, 0], [0, 2]])
def test_parquet_mode_requires_each_mesh_node_once(tmp_path, indices):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(dict(node_index=indices, m0001_u1=[1.] * len(indices),
                         m0001_u2=[2.] * len(indices), m0001_u3=[3.] * len(indices)))
    pq.write_table(table, tmp_path / "shape.parquet")
    meta = dict(shards=[dict(parquet_file="shape.parquet", modes=[1])])
    mesh = SimpleNamespace(node_index=np.array([0, 1]))
    with pytest.raises((ValueError, RuntimeError), match="(?i)(node|incomplete)"):
        gdlc.load_modes(tmp_path, meta, mesh, gdlc.Settings())


def test_nonfinite_modal_field_rejected_before_projection():
    mesh = SimpleNamespace(ring_z=np.array([0., 1.]))
    with pytest.raises(ValueError, match="(?i)nonfinite"):
        gdlc.decompose([], mesh, np.array([[[np.nan, 0., 0.]]]), gdlc.Settings(gpu="off"))


def test_zero_length_ring_edge_has_clear_error():
    ring = gdlc._bench_ring([(0, 0), (0, 0), (1, 0)], False)
    with pytest.raises(ValueError, match="(?i)(zero|degenerate)"):
        gdlc.detect_reference_points(ring, 0.1, gdlc.Settings())


def test_failed_reference_check_is_not_reported_as_pure_critical_value(tmp_path):
    import csv
    source, reference = tmp_path / "source", tmp_path / "source_REF_L"
    for run in (source, reference):
        out = run / "mode_decomposition_GDLC"
        out.mkdir(parents=True)
        row = dict(mode=1, eigenvalue=10., sigma_cr_MPa=10., G_pct=0., D_pct=0.,
                   L_pct=10., C_pct=0., O_pct=90., dominant="L", label="L", confidence="low")
        gdlc.write_csv(out / "mode_participation.csv", [row])
    (reference / "gdlc_reference.json").write_text(json.dumps(dict(kind="L", source_run="source")))
    gdlc.cross_run_summary([source, reference], gdlc.Settings(plots="none"))
    with open(tmp_path / "GDLC_summary" / "runs_summary.csv", newline="") as stream:
        rows = list(csv.DictReader(stream))
    src = next(r for r in rows if r["run"] == "source")
    assert not src.get("pure_L_reference_eigenvalue")
    assert float(src["constrained_L_reference_eigenvalue"]) == 10.
    assert src["L_reference_check_passed"] == "False"


def test_input_repeated_sets_are_additive_and_symmetry_dofs_are_exact(tmp_path):
    inp = tmp_path / "sets.inp"
    inp.write_text("*Assembly, name=A\n*Nset, nset=S, instance=P1\n1\n"
                   "*Nset, nset=S, instance=P2\n2\n*End Assembly\n*Boundary\nS, ZSYMM\n")
    info = gdlc.read_inp(inp)
    assert set(info.nsets["S"]) == {("P1", 1), ("P2", 2)}
    assert info.boundaries[0][1] == {3, 4, 5}


def test_mixed_thickness_is_not_silently_classified_as_uniform(tmp_path):
    inp = tmp_path / "sections.inp"
    inp.write_text("*Shell Section, elset=A, material=M\n1.\n"
                   "*Shell Section, elset=B, material=M\n2.\n")
    with pytest.raises(ValueError, match="(?i)uniform.*thickness"):
        gdlc.read_inp(inp)


def test_clusters_are_invariant_to_export_mode_order():
    assert gdlc.clusters_of(np.array([10., 20., 10.001]), 0.001) == [[0, 2], [1]]


def test_parquet_cannot_use_different_input_for_fold_geometry(tmp_path, monkeypatch):
    inp = tmp_path / "job.inp"
    inp.write_text("different geometry")
    monkeypatch.setattr(gdlc, "choose_export", lambda *a: (tmp_path, dict(
        source_inp_sha256="old", shards=[dict(parquet_file="shape.parquet", modes=[1])])) )
    monkeypatch.setattr(gdlc, "find_inp", lambda *a: inp)
    with pytest.raises(RuntimeError, match="(?i)(sha256|differ)"):
        gdlc.process_run(tmp_path, gdlc.Settings(plots="none"))


def test_exporter_rejects_duplicate_nodal_values():
    value = SimpleNamespace(instance=SimpleNamespace(name="P"), nodeLabel=1,
                            precision="DOUBLE_PRECISION", dataDouble=(1., 2., 3.))
    indexers, _ = exporter._node_indexers([("P", 1)])
    with pytest.raises(ValueError, match="(?i)duplicate"):
        exporter._read_field_values(SimpleNamespace(values=[value, value]),
                                    indexers, np.full((1, 6), np.nan), 0)


def test_odb_completion_requires_actual_matching_job(tmp_path):
    odb = tmp_path / "job.odb"
    odb.touch()
    for text in ("Compilation completed successfully\n", "Abaqus JOB another COMPLETED\n"):
        odb.with_suffix(".log").write_text(text)
        assert gdlc.odb_status(odb) == "failed"
    odb.with_suffix(".log").write_text("Abaqus JOB job COMPLETED\n")
    assert gdlc.odb_status(odb) == "ok"


def test_planar_shell_does_not_silently_guess_member_axis(tmp_path):
    np.savez(tmp_path / "raw_dof_map.npz", node_instances=["P"] * 4, node_labels=[1, 2, 3, 4],
             node_coordinates=[[0., 0., 0.], [1., 0., 0.], [1., 0., 10.], [0., 0., 10.]])
    inp = tmp_path / "job.inp"
    inp.write_text("*Part, name=P\n*Node\n1, 0., 0., 0.\n2, 1., 0., 0.\n3, 1., 0., 10.\n4, 0., 0., 10.\n*Element, type=S4R\n1, 1, 2, 3, 4\n*End Part\n"
                   "*Assembly, name=A\n*Instance, name=P, part=P\n*End Instance\n*End Assembly\n")
    with pytest.raises(ValueError, match="(?i)(axis|prismatic)"):
        gdlc.load_mesh(tmp_path, dict(shards=["modes.npz"]), inp)


@pytest.mark.parametrize('load_case', ['compression', 'bending', 'combined'])
def test_local_reference_axis_aligned_corner_offset_is_a_boundary_not_a_one_term_equation(tmp_path, monkeypatch, load_case):
    # Rounded 90-degree fold turned by 45 degrees: the virtual corner lies straight below the
    # fold reference node, so one in-plane row reduces to u_r,2 = 0. Abaqus rejects one-term
    # *Equation rows, so that DOF must be held by a boundary condition in every load case.
    c = np.cos(np.pi / 4)
    p = np.array([(0, 20), (0, 14), (0, 8), (0, 2), (2 - 2 ** .5, 2 - 2 ** .5), (2, 0), (8, 0), (14, 0), (20, 0)],
                 float) @ np.array([[c, -c], [c, c]]).T
    n, nr = len(p), 3
    mesh = gdlc.Mesh(np.arange(n * nr), np.array(["P"] * (n * nr)), np.arange(1, n * nr + 1),
                     np.array([[x, y, z] for z in (0., 10., 20.) for x, y in p]),
                     np.array([[r * n + i, r * n + i + 1, (r + 1) * n + i + 1, (r + 1) * n + i]
                               for r in range(nr - 1) for i in range(n - 1)]),
                     np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                     np.repeat([0., 10., 20.], n), np.tile(p, (nr, 1)), np.repeat(np.arange(nr), n),
                     np.array([0., 10., 20.]), 20.)
    run = tmp_path / "run"
    run.mkdir()
    inp = run / "job.inp"
    (run / 'job_build.json').write_text(json.dumps(dict(load_case=dict(type=load_case))))
    inp.write_text("*Assembly, name=A\n*End Assembly\n*Shell Section\n1.0\n*Step, name=B\n*Buckle\n2\n"
                   "*Boundary, op=NEW, load case=1\nP.1, 3, 3\n*Boundary, op=NEW, load case=2\nP.1, 3, 3\n"
                   "*End Step\n")
    monkeypatch.setattr(gdlc, "choose_export", lambda *a: (run, {}))
    monkeypatch.setattr(gdlc, "find_inp", lambda *a: inp)
    monkeypatch.setattr(gdlc, "load_mesh", lambda *a: mesh)
    out = gdlc.make_reference_models(run, gdlc.Settings(), kinds=("L",))[0]
    text = next(out.glob("*.inp")).read_text()
    eqs = _parse_equations(text)
    assert eqs and all(len(t) >= 2 for t in eqs)
    info = json.loads((out / "gdlc_reference.json").read_text())
    assert info["n_dofs_held"] == {"2": 3}
    assert text.count("LREF_FOLDS_U2, 2, 2") == 2                 # both load cases
    held = {(("P", 5), 2), (("P", 14), 2), (("P", 23), 2)}
    assert not held & {t[0][:2] for t in eqs}
    # A coarse three-node arc measures chord rotation using the reference node itself.
    # This kinematic definition is independent of the applied load case.
    assert info['n_equations_shell_rotation'] == 0
    g = gdlc.build_ring_groups(mesh)[0]
    gdlc.detect_reference_points(g, 1., gdlc.Settings())
    _, _, Q, _ = gdlc.build_operators(g)
    rng = np.random.default_rng(17)
    u = gdlc.null_space(Q) @ rng.normal(size=(2*g.n-Q.shape[0], nr))
    field = {}
    for gr, rows in enumerate(g.node_rows):
        for j, row in enumerate(rows):
            for d in (1, 2):
                field[(('P', int(mesh.label[row])), d)] = u[2*j+d-1, gr]
            field[(('P', int(mesh.label[row])), 6)] = rng.normal()
    assert max(abs(sum(v*field[(node,d)] for node,d,v in terms)) for terms in eqs) < 1e-10
    dependent = {(terms[0][0], terms[0][1]) for terms in eqs}
    assert len(dependent) == len(eqs)
    assert not dependent & {(node,d) for terms in eqs for node,d,_ in terms[1:]}


def _built_only_plate_run(tmp_path):
    """Built, unsolved pipeline run (BUILD_ONLY: .inp + _build.json, no .odb/export): a square
    tube P1 (axis z, rings z = 0, 10, 20) and a transverse plate instance (translated and
    rotated 90 deg about x into the plane y = 0.5) whose centre node is the BEAM control of
    two wall nodes of ring z = 10, as a bolt through a steel plate is built."""
    run = tmp_path / "run"
    run.mkdir()
    sq = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    wall_nodes = [f"{1 + 4 * k + i}, {x:g}, {y:g}, {z:g}" for k, z in enumerate((0., 10., 20.))
                  for i, (x, y) in enumerate(sq)]
    wall_elems = [f"{1 + 4 * k + i}, {1 + 4 * k + i}, {1 + 4 * k + (i + 1) % 4}, "
                  f"{5 + 4 * k + (i + 1) % 4}, {5 + 4 * k + i}" for k in range(2) for i in range(4)]
    plate_nodes = [f"{1 + i + 3 * j}, {0.5 * i:g}, {j - 1.:g}, 0." for j in range(3) for i in range(3)]
    plate_elems = [f"{1 + i + 2 * j}, {1 + i + 3 * j}, {2 + i + 3 * j}, {5 + i + 3 * j}, {4 + i + 3 * j}"
                   for j in range(2) for i in range(2)]
    text = "\n".join(
        ["*Heading", "*Part, name=ColumnPiece", "*Node"] + wall_nodes +
        ["*Element, type=S4R"] + wall_elems +
        ["*Elset, elset=W, generate", "1, 8, 1", "*Shell Section, elset=W, material=Steel", "0.1, 5",
         "*End Part", "*Part, name=SteelPlatePart", "*Node"] + plate_nodes +
        ["*Element, type=S4R"] + plate_elems +
        ["*Elset, elset=PL, generate", "1, 4, 1", "*Shell Section, elset=PL, material=SteelPlate", "0.2, 9",
         "*End Part", "*Assembly, name=Assembly", "*Instance, name=P1, part=ColumnPiece", "*End Instance",
         "*Instance, name=STEEL_PLATE_001, part=SteelPlatePart", "0., 0.5, 10.",
         "0., 0.5, 10., 1., 0.5, 10., 90.", "*End Instance",
         "*Nset, nset=BA, instance=STEEL_PLATE_001", "5,", "*Nset, nset=BB, instance=P1", "5,",
         "*Nset, nset=BC, instance=P1", "7,", "*Nset, nset=END0, instance=P1", "1, 2, 3, 4",
         "*MPC", "BEAM, BB, BA", "*MPC", "BEAM, BC, BA", "*End Assembly",
         "*Material, name=Steel", "*Elastic", "200000., 0.3", "*Material, name=SteelPlate", "*Elastic",
         "200000., 0.3", "*Step, name=Buckle, nlgeom=NO, perturbation", "*Buckle", "250, , 500, 1250",
         "*Boundary, op=NEW, load case=1", "END0, 1, 3", "*Boundary, op=NEW, load case=2", "END0, 1, 3",
         "*End Step"]) + "\n"
    (run / "job.inp").write_text(text)
    (run / "job_build.json").write_text(json.dumps(dict(load_case=dict(type="compression"))))
    return run


def test_built_only_run_is_found_and_gets_reference_models_from_its_inp(tmp_path, monkeypatch):
    run = _built_only_plate_run(tmp_path)
    assert gdlc.find_runs([str(tmp_path)]) == [run]                # no .odb, no export
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "x.inp").write_text("*Step\n*Buckle\n1\n*End Step\n")
    assert gdlc.find_runs([str(tmp_path)]) == [run]                # no _build.json: not a pipeline run
    with pytest.raises(SystemExit, match="not found"):             # mistyped folder: clear message
        gdlc.main([str(tmp_path / "no_such_run"), "--make-reference-models"])
    mesh = gdlc.load_mesh(None, None, run / "job.inp")             # geometry from the .inp itself
    plate = mesh.instance == "STEEL_PLATE_001"
    np.testing.assert_allclose(mesh.xyz[plate & (mesh.label == 5)], [[0.5, 0.5, 10.]], atol=1e-12)
    np.testing.assert_allclose(mesh.axis, [0., 0., 1.])
    with pytest.raises(SystemExit) as done:
        gdlc.main([str(run), "--make-reference-models", "--ref-g-twist", "free"])
    assert done.value.code == 0
    ref_g = tmp_path / "run_REF_G"
    info = json.loads((ref_g / "gdlc_reference.json").read_text())
    assert info["n_rings_rigid"] == 2 and info["n_mpc_control_nodes_rigid"] == 1 and info["geometry"] == "inp"
    assert (tmp_path / "run_REF_L" / "gdlc_reference.json").is_file()
    eqs = _parse_equations((ref_g / "job_REF_G.inp").read_text())
    nodes = {n for t in eqs for n, _, _ in t}
    assert ("P1", 5) not in nodes and ("P1", 7) not in nodes       # eliminated by the plate bolt MPCs
    assert ("STEEL_PLATE_001", 5) in {t[0][0] for t in eqs}        # the plate bolt node is in the ring
    assert len(eqs) == (2 * 3 - 3) + (2 * 4 - 3) + 1             # plus plate-control UR3 = ring twist
    xy = {("P1", 1 + 4 * k + i): p for k in range(3) for i, p in enumerate([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])}
    xy[("STEEL_PLATE_001", 5)] = (0.5, 0.5)
    rng = np.random.default_rng(3)
    for _ in range(3):                                             # every in-plane rigid motion is admissible
        t1, t2, th = rng.normal(size=3)
        u = {(n, d): (t1 - th * xy[n][1]) if d == 1 else (t2 + th * xy[n][0]) for n in xy for d in (1, 2)}
        u[("STEEL_PLATE_001", 5), 6] = th
        assert max(abs(sum(c * u[(n, d)] for n, d, c in t)) for t in eqs) < 1e-12
    # once the run is solved and exported, regenerating keeps the recorded .inp geometry
    # (.odb coordinates would change masters/coefficients of the inputs already solved)
    text = (ref_g / "job_REF_G.inp").read_text()
    monkeypatch.setattr(gdlc, "choose_export", lambda *a: pytest.fail("geometry source switched"))
    gdlc.make_reference_models(run, gdlc.Settings(ref_g_twist="free"), kinds=("G",))
    assert (ref_g / "job_REF_G.inp").read_text() == text


@pytest.mark.parametrize('twist', ['free', 'restrained'])
@pytest.mark.parametrize('through_plate', [False, True])
def test_global_reference_beam_offsets_cannot_move_walls_out_of_rigid_ring(tmp_path, twist, through_plate):
    from scipy.linalg import null_space
    run = _built_only_plate_run(tmp_path)
    source = run / 'job.inp'
    if not through_plate:
        source.write_text(source.read_text().replace(
            '*MPC\nBEAM, BB, BA\n*MPC\nBEAM, BC, BA', '*MPC\nBEAM, BB, BC'))
    mesh = gdlc.load_mesh(None, None, source)
    inp = gdlc.read_inp(source)
    out = gdlc.make_reference_models(run, gdlc.Settings(ref_g_twist=twist), kinds=('G',))[0]
    reference = out / 'job_REF_G.inp'
    parsed = gdlc.read_inp(reference)
    assert parsed.pairs == inp.pairs                 # no screw or attachment rewritten
    keys = [(str(i), int(n)) for i, n in zip(mesh.instance, mesh.label)]
    xyz = dict(zip(keys, mesh.xyz))
    variables = [(k, d) for k in keys for d in (1, 2, 6)]
    column = {v: j for j, v in enumerate(variables)}
    equations = _parse_equations(reference.read_text())
    rows = []
    def add(terms):
        row = np.zeros(len(variables))
        for k, d, coefficient in terms:
            row[column[k, d]] += coefficient
        rows.append(row)
    for terms in equations:
        add(terms)
    for kind, ctl, dep in inp.pairs:
        assert kind == 'BEAM'
        dx, dy, _ = xyz[dep] - xyz[ctl]
        add([(dep, 1, 1.), (ctl, 1, -1.), (ctl, 6, dy)])
        add([(dep, 2, 1.), (ctl, 2, -1.), (ctl, 6, -dx)])
        add([(dep, 6, 1.), (ctl, 6, -1.)])
    for nodes, dofs in parsed.boundaries:
        for node in nodes:
            for d in dofs & {1, 2, 6}:
                add([(node, d, 1.)])
    admissible = null_space(np.array(rows))
    wall = [('P1', n) for n in (5, 6, 7, 8)]
    w = admissible[[column[node, d] for node in wall for d in (1, 2)]]
    rigid = np.array([[1., 0., -xyz[node][1]] if d == 1 else [0., 1., xyz[node][0]]
                      for node in wall for d in (1, 2)])
    if twist == 'restrained':
        rigid = rigid[:, :2]
    residual = w - rigid @ np.linalg.lstsq(rigid, w, rcond=None)[0]
    assert np.linalg.norm(residual) < 1e-10
    # Both flexural translations and (only when free) ring twist remain available.
    assert np.linalg.matrix_rank(w, tol=1e-10) == rigid.shape[1]


@pytest.mark.parametrize('inplane', [[1, 2], [1, 3], [2, 3]])
def test_beam_control_rotation_uses_global_axis_sign_and_retained_masters(inplane):
    ring = [(('P', n+1), p) for n, p in enumerate([[0., 0.], [3., 0.], [3., 2.], [0., 2.]])]
    control = ('P', 3)
    lines, count = gdlc.rigid_ring_equations(ring, inplane, {control}, twist=True,
                                            beam_rotation_nodes={control})
    eqs = _parse_equations('\n'.join(lines))
    rot_dof = 9 - sum(inplane)
    sign = np.cross(np.eye(3)[inplane[0]-1], np.eye(3)[inplane[1]-1])[rot_dof-4]
    u = {(node, d): (2.-.4*p[1] if d == inplane[0] else -1.+.4*p[0])
         for node, p in ring for d in inplane}
    u[control, rot_dof] = sign*.4
    assert count == 2*len(ring)-3+1
    assert max(abs(sum(c*u[node, d] for node, d, c in row)) for row in eqs) < 1e-12
    dependents = {row[0][:2] for row in eqs}
    assert not dependents & {term[:2] for row in eqs for term in row[1:]}


def test_plate_deformation_is_separated_from_rigid_plate_motion_and_fails_the_reference_check():
    # one wall quad and one plate quad (plane y = 0.5); plate nodes 4..7
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.],
                    [0., .5, 0.], [1., .5, 0.], [1., .5, 1.], [0., .5, 1.]])
    mesh = SimpleNamespace(instance=np.array(["P1"] * 4 + ["STEEL_PLATE_001"] * 4), xyz=xyz,
                           elems=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    rng = np.random.default_rng(4)
    U = np.zeros((3, 8, 3))
    U[:, :4] = rng.normal(size=(3, 4, 3))                         # wall motion in every mode
    a, om = rng.normal(size=3), rng.normal(size=3)
    U[0, 4:] = a + np.cross(om, xyz[4:] - xyz[4:].mean(0))         # mode 1: plate moves rigidly
    U[1, 4:, 1] = [0., 0., 1., 0.]                                 # mode 2: plate corner bends out
    motion, deform = gdlc.plate_motion_shares(mesh, U)
    assert motion[0] > 1. and deform[0] < 1e-20 and motion[2] == deform[2] == 0.
    assert deform[1] > 1.
    assert gdlc.plate_motion_shares(SimpleNamespace(instance=np.array(["P1"])), U[:, :1]) is None
    rows = [dict(cluster=k + 1, flags="", confidence="high") for k in range(3)]
    gdlc.add_plate_columns(rows, [], (motion, np.array([0., 50., 0.])), gdlc.Settings())
    assert [r["confidence"] for r in rows] == ["high", "low", "high"]
    assert rows[1]["flags"].startswith("PLATE_DEFORMATION")


def test_local_reference_check_requires_plates_not_to_carry_the_modes(tmp_path):
    (tmp_path / "gdlc_reference.json").write_text(json.dumps(dict(kind="L")))
    row = dict(eigenvalue=600., G_pct=.4, D_pct=.5, C_pct=.1, L_pct=99., reliable_dominant="L")
    assert gdlc.reference_check(tmp_path, [dict(row)])["passed"]
    ok = dict(row, plate_motion_pct=3., plate_deformation_pct=.2)
    assert gdlc.reference_check(tmp_path, [ok])["passed"]
    bad = dict(row, plate_motion_pct=60., plate_deformation_pct=40.)
    check = gdlc.reference_check(tmp_path, [ok, bad])
    assert not check["passed"] and check["max_plate_deformation_pct"] == 40.


def _coordinate_precision_export(tmp_path, corners, closed, pieces=1, nw=4, na=3):
    """Small prismatic mesh with transformed instances and FP32 archive coordinates."""
    points = gdlc._bench_rounded(corners, 5., nw=nw, na=na, closed=closed)
    n = len(points)
    lines = ['*Part, name=WALL', '*Node']
    lines += [f'{r*n+i+1}, {x:.17g}, {y:.17g}, {z:.17g}'
              for r,z in enumerate((0., 40., 80.)) for i,(x,y) in enumerate(points)]
    lines += ['*Element, type=S4R']
    edges = n if closed else n-1
    lines += [f'{r*edges+i+1}, {r*n+i+1}, {r*n+(i+1)%n+1}, '
              f'{(r+1)*n+(i+1)%n+1}, {(r+1)*n+i+1}' for r in range(2) for i in range(edges)]
    lines += ['*End Part', '*Assembly, name=A']
    for piece in range(pieces):
        lines += [f'*Instance, name=P{piece+1}, part=WALL',
                  f'{13.12345+piece*300}, -31.77777, 5.1234567',
                  '0., 0., 0., 0., 0., 1., 22.5', '*End Instance']
    lines += ['*End Assembly']
    inp = tmp_path/'precision.inp'
    inp.write_text('\n'.join(lines)+'\n')
    source = gdlc.load_mesh(None, {}, inp)
    order = np.random.default_rng(37).permutation(len(source.label))
    archive = dict(node_instances=source.instance[order], node_labels=source.label[order],
                   node_coordinates=source.xyz[order].astype(np.float32).astype(float))
    np.savez(tmp_path/'raw_dof_map.npz', **archive)
    return inp, source, order, archive


@pytest.mark.parametrize('corners,closed,pieces,nw,na,expected_D', [
    ([(80,200),(0,200),(0,0),(80,0)], False, 1, 4, 3, 0),
    ([(80,180),(80,200),(0,200),(0,0),(80,0),(80,20)], False, 2, 8, 5, 4),
    ([(0,0),(200,0),(200,200),(0,200)], True, 1, 3, 2, 1),
    ([(100,180),(100,200),(0,200),(0,0),(-50,0),(-50,30)], False, 1, 6, 4, 2),
])
def test_quantized_export_uses_verified_input_geometry_without_changing_node_order(tmp_path,corners,closed,pieces,nw,na,expected_D):
    inp,source,order,archive = _coordinate_precision_export(tmp_path,corners,closed,pieces,nw,na)
    loaded = gdlc.load_mesh(tmp_path,dict(shards=['modes_0001.npz']),inp)
    np.testing.assert_array_equal(loaded.xyz,source.xyz[order])
    np.testing.assert_array_equal(loaded.instance,archive['node_instances'])
    np.testing.assert_array_equal(loaded.label,archive['node_labels'])
    assert loaded.geometry['source'] == 'inp_verified_against_export'
    assert loaded.geometry['fp32_rounded_coordinate_count'] > 0
    ring = gdlc.build_ring_groups(loaded)[0]
    gdlc.detect_reference_points(ring,2.,gdlc.Settings())
    gdlc.build_basis(ring)
    assert gdlc.self_test(ring)['passed']
    assert ring.basis['sizes']['G'] == 3
    assert ring.basis['sizes']['C'] == 3*(pieces-1)
    assert ring.basis['sizes']['D'] == expected_D


@pytest.mark.parametrize('defect', ['coordinates', 'duplicate_node', 'missing_node'])
def test_input_geometry_is_not_substituted_for_an_incompatible_export(tmp_path,defect):
    inp,_,_,archive = _coordinate_precision_export(tmp_path,[(80,200),(0,200),(0,0),(80,0)],False)
    if defect == 'coordinates':
        archive['node_coordinates'][0,0] += 1e-6  # smaller than old geometric tolerances, but not rounding
    elif defect == 'duplicate_node':
        archive['node_labels'][0] = archive['node_labels'][1]
    else:
        for key in archive:
            archive[key] = archive[key][1:]
    np.savez(tmp_path/'raw_dof_map.npz', **archive)
    with pytest.raises(ValueError,match='(?i)(coordinates|duplicate|node.*(differ|match|missing))'):
        gdlc.load_mesh(tmp_path,dict(shards=['modes_0001.npz']),inp)


def test_basis_selftest_is_reproducible_across_calls():
    ring = gdlc._bench_ring(gdlc._bench_rounded([(80,200),(0,200),(0,0),(80,0)],5.,closed=False),False)
    gdlc.detect_reference_points(ring,2.,gdlc.Settings())
    gdlc.build_basis(ring)
    assert gdlc.self_test(ring) == gdlc.self_test(ring)
