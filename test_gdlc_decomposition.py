"""Small numerical and archive-contract regressions; no Abaqus solve required."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gdlc_classifier import gdlc_mode_decomposition as gdlc
from gdlc_classifier import abaqus_mfsm_export as exporter


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


def test_reference_coupling_leaves_axial_y_translation_free(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    inp = run / "job.inp"
    inp.write_text("*Assembly, name=A\n*End Assembly\n*Shell Section\n0.1\n"
                   "*Step, name=B\n*Buckle\n2\n*End Step\n")
    p = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    mesh = gdlc.Mesh(np.arange(8), np.array(["P"] * 8), np.arange(1, 9),
                     np.array([[x, y, z] for y in (0., 10.) for x, z in p]),
                     np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]),
                     np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array([0., 0., 1.]),
                     np.repeat([0., 10.], 4), np.tile(p, (2, 1)), np.repeat([0, 1], 4),
                     np.array([0., 10.]), 10.)
    monkeypatch.setattr(gdlc, "choose_export", lambda *a: (run, {}))
    monkeypatch.setattr(gdlc, "find_inp", lambda *a: inp)
    monkeypatch.setattr(gdlc, "load_mesh", lambda *a: mesh)
    out = gdlc.make_reference_models(run, gdlc.Settings(), kinds=("G",))[0]
    blocks = list(exporter._inp_blocks(next(out.glob("*.inp")).read_text()))
    for keyword, _, _, lines in blocks:
        if keyword == "KINEMATIC":
            dofs = set()
            for line in lines:
                lo, hi = map(int, line.split(","))
                dofs.update(range(lo, hi + 1))
            assert dofs == {1, 3}


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
    inp.write_text("*Part, name=P\n*Element, type=S4R\n1, 1, 2, 3, 4\n*End Part\n"
                   "*Assembly, name=A\n*Instance, name=P, part=P\n*End Instance\n*End Assembly\n")
    with pytest.raises(ValueError, match="(?i)(axis|prismatic)"):
        gdlc.load_mesh(tmp_path, dict(shards=["modes.npz"]), inp)
