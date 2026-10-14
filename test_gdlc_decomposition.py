"""Small numerical and archive-contract regressions; no Abaqus solve required."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gdlc_classifier import gdlc_mode_decomposition as gdlc
from gdlc_classifier import abaqus_mfsm_export as exporter


def test_solve_only_reuses_reference_without_export_classification_or_stiffness(tmp_path, monkeypatch):
    (tmp_path/'gdlc_reference.json').write_text(json.dumps(dict(job='REF_L')))
    (tmp_path/'REF_L.inp').write_text('*Step\n*Buckle\n20\n*End Step\n')
    (tmp_path/'REF_L.odb').write_bytes(b'synthetic')
    (tmp_path/'REF_L.sta').write_text('THE ANALYSIS HAS COMPLETED SUCCESSFULLY')
    monkeypatch.setattr(gdlc, 'find_runs', lambda paths: [tmp_path])
    for name in ('ensure_export', 'ensure_stiffness', 'process_run', '_run_cmd'):
        monkeypatch.setattr(gdlc, name, lambda *a: pytest.fail('unexpected solve/export/classification'))
    assert gdlc.main([str(tmp_path), '--solve-reference-models', '--solve-only']) == 0
    (tmp_path/'REF_L.lck').write_text('running')
    assert gdlc.main([str(tmp_path), '--solve-reference-models', '--solve-only']) == 1
    (tmp_path/'REF_L.lck').unlink()
    (tmp_path/'REF_L.sta').write_text('ANALYSIS ABORTED')
    assert gdlc.main([str(tmp_path), '--solve-reference-models', '--solve-only']) == 1


def test_selftest_does_not_dispatch_abaqus(tmp_path, monkeypatch):
    monkeypatch.setattr(gdlc, "find_runs", lambda paths: [tmp_path])
    monkeypatch.setattr(gdlc, "list_exports", lambda path: [(tmp_path, {})])
    for name in ("ensure_export", "ensure_stiffness", "solve_reference_model"):
        monkeypatch.setattr(gdlc, name, lambda *a: pytest.fail("self-test dispatched a solver/export"))
    seen = []
    monkeypatch.setattr(gdlc, "process_run", lambda path, st: seen.append(st.selftest_only))
    gdlc.main([str(tmp_path), "--selftest-only", "--solve-reference-models", "--cpus", "auto"])
    assert seen == [True]


def test_reference_solver_refuses_old_results(tmp_path, monkeypatch):
    (tmp_path / 'gdlc_reference.json').write_text('{}')
    (tmp_path / 'job.inp').write_text('*Step\n*Buckle\n2\n*End Step\n')
    (tmp_path / 'job.odb').write_bytes(b'old result')
    monkeypatch.setattr(gdlc, 'odb_status', lambda *a:'stale')
    monkeypatch.setattr(gdlc, '_run_cmd', lambda *a: pytest.fail('overwriting old result'))
    with pytest.raises(RuntimeError, match='preserve'):
        gdlc.solve_reference_model(tmp_path, gdlc.Settings())
    assert (tmp_path / 'job.odb').read_bytes() == b'old result'
    (tmp_path / 'job.odb').unlink()
    (tmp_path / 'job.lck').write_text('running before ODB creation')
    monkeypatch.setattr(gdlc, 'odb_status', lambda *a:'missing')
    with pytest.raises(RuntimeError, match='preserved'):
        gdlc.solve_reference_model(tmp_path, gdlc.Settings())


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


def test_selftest_rejects_nonfinite_component_even_when_global_translation_is_correct():
    ring = gdlc._bench_ring(gdlc._bench_rounded([(80,200),(0,200),(0,0),(80,0)],5.,closed=False),False)
    gdlc.detect_reference_points(ring,2.,gdlc.Settings())
    gdlc.build_basis(ring)
    ring.proj['D'][0,0] = float('nan')
    assert not gdlc.self_test(ring)['passed']


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


def test_compression_references_use_lanczos_with_nonnegative_range(tmp_path, monkeypatch):
    run, _ = _reference_run(tmp_path, monkeypatch)                 # compression source: '*Buckle' without range
    for out in gdlc.make_reference_models(run, gdlc.Settings(), kinds=("L", "G")):
        text = next(out.glob("*.inp")).read_text()
        assert "*Buckle, eigensolver=LANCZOS\n20, 0\n" in text
        info = json.loads((out / "gdlc_reference.json").read_text())
        assert info["eigensolver"] == "lanczos" and info["min_eigenvalue"] == 0.0
        assert info["source_min_eigenvalue"] is None and info["n_modes"] == 20
    lines = gdlc.set_buckle_solver(["*Buckle", "250, , 500, 1250"], 20, "subspace", None)
    assert lines[1] == "20, , 80, 1250"                            # subspace fallback unchanged


@pytest.mark.parametrize("load_case, expected", [(None, "restrained"), ("compression", "restrained"),
                                                 ("bending", "free"), ("combined", "free")])
def test_ref_g_twist_is_restrained_for_compression_and_free_for_bending(tmp_path, load_case, expected):
    if load_case:
        (tmp_path / "job_build.json").write_text(json.dumps(dict(load_case=dict(type=load_case))))
    assert gdlc.ref_g_twist_mode(tmp_path, "auto")[0] == expected
    assert gdlc.ref_g_twist_mode(tmp_path, "free")[0] == "free"
    assert gdlc.ref_g_twist_mode(tmp_path, "restrained")[0] == "restrained"


def test_twist_restrained_reference_rings_only_translate(tmp_path, monkeypatch):
    # Member axis = y (in-plane DOFs 1 and 3); end ring y = 0 has in-plane BCs and is left alone.
    run, mesh = _reference_run(tmp_path, monkeypatch,
                               "*Boundary, op=NEW, load case=1\nP.1, 1, 3\nP.2, 1, 3\nP.3, 1, 3\nP.4, 1, 3\n")
    out = gdlc.make_reference_models(run, gdlc.Settings(), kinds=("G",))[0]       # auto, no build.json
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
    inp.write_text("*Part, name=P\n*Node\n1,0,0,0\n2,1,0,0\n3,1,0,10\n4,0,0,10\n"
                   "*Element, type=S4R\n1, 1, 2, 3, 4\n*End Part\n"
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


def _built_only_run(tmp_path):
    """Built, unsolved pipeline run (BUILD_ONLY: .inp + _build.json, no .odb/export): a square
    tube (axis z, rings z = 0, 10, 20) whose instance is translated by (2, 3, 0) and rotated
    90 deg about the member axis, so the assembly coordinates differ from the part ones."""
    run = tmp_path / "run"
    run.mkdir()
    sq = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    nodes = [f"{1 + 4 * k + i}, {x:g}, {y:g}, {z:g}" for k, z in enumerate((0., 10., 20.))
             for i, (x, y) in enumerate(sq)]
    elems = [f"{1 + 4 * k + i}, {1 + 4 * k + i}, {1 + 4 * k + (i + 1) % 4}, "
             f"{5 + 4 * k + (i + 1) % 4}, {5 + 4 * k + i}" for k in range(2) for i in range(4)]
    text = "\n".join(
        ["*Heading", "*Part, name=ColumnPiece", "*Node"] + nodes + ["*Element, type=S4R"] + elems +
        ["*Elset, elset=W, generate", "1, 8, 1", "*Shell Section, elset=W, material=Steel", "0.1, 5",
         "*End Part", "*Assembly, name=Assembly", "*Instance, name=P1, part=ColumnPiece", "2., 3., 0.",
         "2., 3., 0., 2., 3., 1., 90.", "*End Instance",
         "*Nset, nset=END0, instance=P1", "1, 2, 3, 4", "*End Assembly",
         "*Material, name=Steel", "*Elastic", "200000., 0.3",
         "*Step, name=Buckle, nlgeom=NO, perturbation", "*Buckle", "250, , 500, 1250",
         "*Boundary, op=NEW, load case=1", "END0, 1, 3", "*Boundary, op=NEW, load case=2", "END0, 1, 3",
         "*End Step"]) + "\n"
    (run / "job.inp").write_text(text)
    (run / "job_build.json").write_text(json.dumps(dict(load_case=dict(type="compression"))))
    return run


def test_built_only_run_is_found_and_gets_reference_models_from_its_inp(tmp_path, monkeypatch):
    run = _built_only_run(tmp_path)
    assert gdlc.find_runs([str(tmp_path)]) == [run]                # no .odb, no export
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "x.inp").write_text("*Step\n*Buckle\n1\n*End Step\n")
    assert gdlc.find_runs([str(tmp_path)]) == [run]                # no _build.json: not a pipeline run
    with pytest.raises(SystemExit, match="not found"):             # mistyped folder: clear message
        gdlc.main([str(tmp_path / "no_such_run"), "--make-reference-models"])
    mesh = gdlc.load_mesh(None, None, run / "job.inp")             # geometry from the .inp itself
    sq = {1: (0., 0.), 2: (1., 0.), 3: (1., 1.), 4: (0., 1.)}
    xy = {("P1", 4 * k + i): (2. - y, 3. + x) for k in range(3) for i, (x, y) in sq.items()}
    for row, (inst, lab) in enumerate(zip(mesh.instance, mesh.label)):
        np.testing.assert_allclose(mesh.xyz[row, :2], xy[(str(inst), int(lab))], atol=1e-12)
    np.testing.assert_allclose(mesh.axis, [0., 0., 1.])
    with pytest.raises(SystemExit) as done:
        gdlc.main([str(run), "--make-reference-models", "--ref-g-twist", "free"])
    assert done.value.code == 0
    ref_g = tmp_path / "run_REF_G"
    info = json.loads((ref_g / "gdlc_reference.json").read_text())
    assert info["n_rings_rigid"] == 2 and info["geometry"] == "inp"
    assert (tmp_path / "run_REF_L" / "gdlc_reference.json").is_file()
    eqs = _parse_equations((ref_g / "job_REF_G.inp").read_text())
    assert len(eqs) == 2 * (2 * 4 - 3)
    rng = np.random.default_rng(3)
    for _ in range(3):                                             # every in-plane rigid motion is admissible
        t1, t2, th = rng.normal(size=3)
        u = {(n, d): (t1 - th * xy[n][1]) if d == 1 else (t2 + th * xy[n][0]) for n in xy for d in (1, 2)}
        assert max(abs(sum(c * u[(n, d)] for n, d, c in t)) for t in eqs) < 1e-12
    # once the run is solved and exported, regenerating keeps the recorded .inp geometry
    # (.odb coordinates would change masters/coefficients of the inputs already solved)
    text = (ref_g / "job_REF_G.inp").read_text()
    monkeypatch.setattr(gdlc, "choose_export", lambda *a: pytest.fail("geometry source switched"))
    gdlc.make_reference_models(run, gdlc.Settings(ref_g_twist="free"), kinds=("G",))
    assert (ref_g / "job_REF_G.inp").read_text() == text


@pytest.mark.parametrize("case,twist",[("compression","restrained"),("bending","free")])
def test_reference_models_keep_source_loads_connections_and_exact_build_provenance(tmp_path,case,twist):
    run=_built_only_run(tmp_path);inp=run/'job.inp'
    original=inp.read_text().replace('*Buckle\n250, , 500, 1250',
                                   '*Buckle, eigensolver=LANCZOS\n250, 0.' if case=='bending' else '*Buckle\n250, , 500, 1250')
    original=original.replace('*End Assembly','*MPC\nBEAM, P1.5, P1.6\n*End Assembly')
    original=original.replace('*End Step','*Cload\nP1.9, 3, -1.\nP1.10, 3, 1.\n*End Step')
    inp.write_text(original)
    build=dict(load_case=dict(type=case,neutral_axis_angle_deg=0.),source_inputs=dict(bolt_positions_mm=[10.]))
    (run/'job_build.json').write_text(json.dumps(build))
    # An unrelated build record must not decide twist or replace the copied source record.
    (run/'AAA_KE_build.json').write_text(json.dumps(dict(load_case=dict(type='compression'))))
    outputs=gdlc.make_reference_models(run,gdlc.Settings())
    for ref in outputs:
        info=json.loads((ref/'gdlc_reference.json').read_text());text=(ref/(info['job']+'.inp')).read_text()
        assert info['source_inp_sha256']==gdlc.file_sha256(inp)
        assert info['input_sha256']==gdlc.file_sha256(ref/(info['job']+'.inp'))
        assert info['load_case']['type']==case
        assert json.loads((ref/(info['job']+'_build.json')).read_text())==build
        assert '*MPC\nBEAM, P1.5, P1.6' in text and '*Cload\nP1.9, 3, -1.\nP1.10, 3, 1.' in text
        if case=='bending':assert gdlc.buckle_min_eigenvalue(text.splitlines())==0.
        if info['kind']=='G':assert info['twist']==twist
