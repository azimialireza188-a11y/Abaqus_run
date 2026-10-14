"""Elastic-stiffness export and strain-energy evaluation of the GDLC split (no Abaqus needed).

A synthetic built-up column stands in for Abaqus: two rounded lipped channels back to back
(instances P1, P2), BEAM MPC bolts between the webs, simply supported ends and one axial
anchor. Its elastic stiffness is a 3D frame of Euler-Bernoulli beams on every element edge and
diagonal (exact six rigid-body modes, 6 DOF per node); its "buckling modes" are generalised
eigenvectors of that stiffness and a string-type geometric stiffness, computed in the
constrained space, so they satisfy the MPCs and supports exactly and are K-orthogonal, like
Abaqus buckling modes. fake_abaqus_matrix_output() writes the .mtx files of the matrix job."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gdlc_classifier import gdlc_mode_decomposition as gdlc
from gdlc_classifier import gdlc_stiffness as gst



def section(nw=3, na=2):
    lip = [(80, 180), (80, 200), (0, 200), (0, 0), (80, 0), (80, 20)]
    a = np.array(gdlc._bench_rounded(lip, 5, nw=nw, na=na, closed=False))
    b = a * np.array([-1.0, 1.0]) + np.array([-10.0, 0.0])
    return [a, b]


def build(root: Path, nz=13, length=1200.0, n_modes=8, bolt_rings=(3, 6, 9)):
    pieces = section()
    z = np.linspace(0.0, length, nz)
    names = ["P1", "P2"]
    nodes, elems = {}, {}
    for name, pts in zip(names, pieces):
        n = len(pts)
        lab = lambda r, i: r * n + i + 1
        nodes[name] = {lab(r, i): (pts[i][0], pts[i][1], z[r]) for r in range(nz) for i in range(n)}
        elems[name] = [(r * (n - 1) + i + 1, (lab(r, i), lab(r, i + 1), lab(r + 1, i + 1), lab(r + 1, i)))
                       for r in range(nz - 1) for i in range(n - 1)]
    # bolts: web nodes of P1 (x = 0) and P2 (x = -10) at the same y, on some rings
    pa, pb = pieces
    web_a = [i for i in range(len(pa)) if abs(pa[i][0]) < 1e-9 and 40 < pa[i][1] < 160]
    bolts = []
    for r in bolt_rings:
        for i in web_a[:2]:
            j = int(np.argmin(np.linalg.norm(pb - (pa[i] + np.array([-10.0, 0.0])), axis=1)))
            bolts.append(((("P1", r * len(pa) + i + 1)), ("P2", r * len(pb) + j + 1)))
    lines = ["*Heading", "synthetic", "*Preprint, echo=NO"]
    for name in names:
        lines += [f"*Part, name=PART_{name}", "*Node"]
        lines += [f"{k}, {x:.9f}, {y:.9f}, {zz:.9f}" for k, (x, y, zz) in nodes[name].items()]
        lines += ["*Element, type=S4R"] + [f"{e}, {a}, {b}, {c}, {d}" for e, (a, b, c, d) in elems[name]]
        lines += ["*Elset, elset=ALL, generate", f"1, {len(elems[name])}, 1",
                  "*Shell Section, elset=ALL, material=Steel", "2., 5", "*End Part"]
    lines += ["*Assembly, name=Assembly"]
    for name in names:
        lines += [f"*Instance, name={name}, part=PART_{name}", "*End Instance"]
    ends = {name: [k for k, p in nodes[name].items() if p[2] in (z[0], z[-1])] for name in names}
    for name in names:
        lines += [f"*Nset, nset=END_{name}, instance={name}"] + [", ".join(map(str, ends[name][i:i + 16]))
                                                                for i in range(0, len(ends[name]), 16)]
    mid = nz // 2
    lines += ["*Nset, nset=ANCHOR, instance=P1", str(mid * len(pa) + 1)]
    for k, (a, b) in enumerate(bolts):
        lines += ["*MPC", f"BEAM, {b[0]}.{b[1]}, {a[0]}.{a[1]}"]
    lines += ["*End Assembly", "*Material, name=Steel", "*Elastic", "200000., 0.3"]
    lines += ["*Boundary", "ANCHOR, 3, 3"]
    for name in names:
        lines += ["*Boundary", f"END_{name}, 1, 2", f"END_{name}, 6, 6"]
    lines += ["*Step, name=Buckle, nlgeom=NO, perturbation", "*Buckle", f"{n_modes}, , 40, 300",
              "*Output, field", "*Node Output", "U, UR", "*End Step"]
    run = root / "RUN"
    run.mkdir(parents=True, exist_ok=True)
    inp = run / "JOB.inp"
    inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    keys = [(name, k) for name in names for k in sorted(nodes[name])]
    xyz = np.array([nodes[i][k] for i, k in keys])
    edges = set()
    for name in names:
        row = {k: r for r, (i, k) in enumerate(keys) if i == name}
        for _, (a, b, c, d) in elems[name]:
            q = [row[a], row[b], row[c], row[d]]
            for u, v in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)):
                edges.add(tuple(sorted((q[u], q[v]))))
    K = frame_stiffness(xyz, sorted(edges))
    return dict(run=run, inp=inp, keys=keys, xyz=xyz, K=K, bolts=bolts, nodes=nodes, z=z, n_modes=n_modes)


def frame_stiffness(xyz, edges, E=200000.0, G=76923.0, A=30.0, I=40.0, J=60.0):
    import scipy.sparse as sp
    n6 = 6 * len(xyz)
    ri, ci, vv = [], [], []
    for a, b in edges:
        d = xyz[b] - xyz[a]
        L = float(np.linalg.norm(d))
        ex = d / L
        ref = np.array([0.0, 0.0, 1.0]) if abs(ex[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        ey = np.cross(ref, ex); ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R3 = np.vstack([ex, ey, ez])
        k = np.zeros((12, 12))
        EA, EI, GJ = E * A / L, E * I, G * J / L
        for (i, j), v in {(0, 0): EA, (0, 6): -EA, (6, 6): EA, (3, 3): GJ, (3, 9): -GJ, (9, 9): GJ}.items():
            k[i, j] += v
        for (u, th, s) in ((1, 5, 1.0), (2, 4, -1.0)):           # bending in y (rot z) and z (rot y)
            c1, c2, c3 = 12 * EI / L ** 3, 6 * EI / L ** 2, EI / L
            idx = [u, th, u + 6, th + 6]
            kk = np.array([[c1, s * c2, -c1, s * c2], [s * c2, 4 * c3, -s * c2, 2 * c3],
                           [-c1, -s * c2, c1, -s * c2], [s * c2, 2 * c3, -s * c2, 4 * c3]])
            for p_ in range(4):
                for q_ in range(4):
                    k[idx[p_], idx[q_]] += kk[p_, q_]
        k = np.triu(k) + np.triu(k, 1).T
        Rt = np.kron(np.eye(4), R3)
        kg = Rt.T @ k @ Rt
        dofs = np.r_[6 * a + np.arange(6), 6 * b + np.arange(6)]
        ri.append(np.repeat(dofs, 12)); ci.append(np.tile(dofs, 12)); vv.append(kg.ravel())
    return sp.coo_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))), shape=(n6, n6)).tocsr()


def modes_and_export(m: dict):
    """Constrained generalised eigenvectors + an npz modal export (abaqus_mfsm_export layout)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    inp = gdlc.read_inp(m["inp"])
    key_row = {k: r for r, k in enumerate(m["keys"])}
    T, indep, fixed, dep, _ = gst.constraint_transform(m["xyz"], key_row, inp.pairs, inp.boundaries)
    Kq = (T.T @ m["K"] @ T).tocsc()
    # string-type geometric stiffness on longitudinal edges (translations only)
    xyz = m["xyz"]
    rows, cols, vals = [], [], []
    by_xy = {}
    for r, k in enumerate(m["keys"]):
        by_xy.setdefault((k[0], round(xyz[r][0], 6), round(xyz[r][1], 6)), []).append(r)
    for chain in by_xy.values():
        chain.sort(key=lambda r: xyz[r][2])
        for a, b in zip(chain, chain[1:]):
            L = xyz[b][2] - xyz[a][2]
            for c in range(3):
                for (i, j, v) in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
                    rows.append(6 * i + c); cols.append(6 * j + c); vals.append(v / L)
    Kg = sp.coo_matrix((vals, (rows, cols)), shape=m["K"].shape).tocsr()
    Kgq = (T.T @ Kg @ T).tocsc()
    mu, phi = sla.eigsh(Kgq, k=m["n_modes"], M=Kq, which="LA")
    order = np.argsort(-mu)
    lam = 1.0 / mu[order]
    V = np.asarray(T @ phi[:, order])                     # (6N, nm)
    exp = m["run"] / "modal_export_gdlc"
    exp.mkdir(exist_ok=True)
    N = len(m["keys"])
    inst = np.array([k[0] for k in m["keys"]])
    lab = np.array([k[1] for k in m["keys"]], dtype=np.int64)
    np.savez(exp / "raw_dof_map.npz", instances=np.repeat(inst, 6), labels=np.repeat(lab, 6),
             dofs=np.tile(np.arange(1, 7), N), node_instances=inst, node_labels=lab, node_coordinates=xyz)
    modes = list(range(1, m["n_modes"] + 1))
    np.savez(exp / "modes_0001.npz", vectors=V, modes=np.array(modes))
    meta = dict(kind="PORTABLE_RAW_MODAL_DATA_ONLY", mode_count=len(modes), modes_per_shard=len(modes),
                shards=["modes_0001.npz"], modes=[dict(mode=k, eigenvalue=float(e)) for k, e in zip(modes, lam)],
                source_inp_sha256=gdlc.file_sha256(m["inp"]), rotations_available=True)
    (exp / "modal_export.json").write_text(json.dumps(meta), encoding="utf-8")
    return V, lam, Kg


def fake_abaqus_matrix_output(folder: Path, job: str, m: dict, triangle="lower", with_load=True,
                              numbering="sequential"):
    """The .mtx files of the matrix job (MATRIX INPUT format, internal node numbers)."""
    K = m["K"].tocoo()
    n_int = np.arange(1, len(m["keys"]) + 1) if numbering == "sequential" else np.arange(len(m["keys"]), 0, -1)
    i, j, v = K.row, K.col, K.data
    keep = (i >= j) if triangle == "lower" else np.ones(len(i), bool)
    i, j, v = i[keep], j[keep], v[keep]
    with open(folder / f"{job}_STIF1.mtx", "w") as f:
        f.write("** ASSEMBLED STIFFNESS MATRIX (synthetic)\n")
        for a, b, x in zip(i, j, v):
            f.write(f"{n_int[a // 6]},{a % 6 + 1}, {n_int[b // 6]},{b % 6 + 1}, {x:.16e}\n")
    if with_load:
        codes = json.loads((folder / f"{job}_codes.json").read_text())["code_base"]
        names = sorted({k[0] for k in m["keys"]}, key=lambda s: [k[0] for k in m["keys"]].index(s))
        with open(folder / f"{job}_LOAD1.mtx", "w") as f:
            f.write("*CLOAD, REAL\n")
            for r, (inst, lab) in enumerate(m["keys"]):
                f.write(f"{n_int[r]},1, {float((names.index(inst) + 1) * codes + lab):.16e}\n")


@pytest.fixture(scope="module")
def column(tmp_path_factory):
    m = build(tmp_path_factory.mktemp("synthetic"))
    m["V"], m["lam"], m["Kg"] = modes_and_export(m)
    return m


def test_matrix_job_is_the_model_data_plus_one_raw_stiffness_step(column):
    folder, job = gst.write_matrix_job(column["run"], column["inp"])
    lines = (folder / f"{job}.inp").read_text().splitlines()
    kws = [l.split(",")[0].strip().lower() for l in lines if l.startswith("*") and not l.startswith("**")]
    assert "*boundary" not in kws and "*buckle" not in kws
    assert kws.count("*step") == 1 and "*matrix generate" in kws and "*mpc" in kws
    assert "*Matrix Generate, STIFFNESS, LOAD, MPC=NO" in lines
    assert "*Matrix Output, STIFFNESS, LOAD, FORMAT=MATRIX INPUT" in lines
    i = lines.index("*Cload")
    cl = lines[i + 1:lines.index("*End Step")]
    assert len(cl) == len(column["keys"])
    base = json.loads((folder / f"{job}_codes.json").read_text())["code_base"]
    assert len({float(l.split(",")[2]) for l in cl}) == len(cl) and base > max(k[1] for k in column["keys"])


@pytest.mark.parametrize('restart', ['*Restart, write, frequency=0',
                                    '*RESTART, WRITE, FREQUENCY=10, OVERLAY'])
@pytest.mark.parametrize('location', ['model', 'step'])
def test_matrix_export_ignores_write_only_restart_output(column, restart, location):
    lines = column['inp'].read_text().splitlines()
    expected = gst.matrix_job_lines(lines)
    i = next(i for i, line in enumerate(lines) if line.lower().startswith(
        '*step' if location == 'model' else '*end step'))
    with_restart = lines[:i] + [restart] + lines[i:]
    assert gst.matrix_job_lines(with_restart) == expected
    assert with_restart[i] == restart   # source input remains untouched


@pytest.mark.parametrize('restart', ['*Restart, read, step=1',
                                    '*RESTART, READ, WRITE, STEP=1', '*Restart',
                                    '*Restart, R, WRITE'])
def test_matrix_export_still_rejects_restart_state_reading(column, restart):
    lines = column['inp'].read_text().splitlines()
    i = next(i for i, line in enumerate(lines) if line.lower().startswith('*step'))
    with pytest.raises(NotImplementedError, match='(?i)restart'):
        gst.matrix_job_lines(lines[:i] + [restart] + lines[i:])


def test_contact_fallback_removes_only_contact_and_boundary_blocks():
    lines = ["*Part, name=A", "*Node", "1, 0., 0., 0.", "2, 1., 0., 0.", "*End Part", "*Assembly, name=X",
             "*Instance, name=I1, part=A", "*End Instance", "*End Assembly", "*Boundary", "I1.1, 1, 6",
             "*Contact", "*Contact Inclusions, ALL EXTERIOR", "*Contact Property Assignment", " ,  , P",
             "*Surface Interaction, name=P", "1.,", "*Step, name=S", "*Buckle", "5", "*End Step"]
    keep, _, _, nb = gst.matrix_job_lines(lines)
    drop, codes, base, _ = gst.matrix_job_lines(lines, drop_contact=True)
    kw = lambda ls: [l.split(",")[0].lower() for l in ls if l.startswith("*") and not l.startswith("**")]
    assert nb == 1 and "*boundary" not in kw(keep) and "*contact" in kw(keep)
    assert not any(k.startswith("*contact") for k in kw(drop)) and "*surface interaction" in kw(drop)
    assert "*buckle" not in kw(drop) and codes == {("I1", 1): base + 1, ("I1", 2): base + 2}


def test_inactive_internal_zero_matrix_entries_are_ignored(tmp_path):
    path = tmp_path / 'operator.mtx'
    path.write_text('-20,1,-20,1,0.0\n-20,2,7,1,0.0\n7,1,7,1,12.5\n')
    rn, rd, cn, cd, val = gst.read_operator(path)
    np.testing.assert_array_equal(rn, [7])
    np.testing.assert_array_equal(cn, [7])
    np.testing.assert_array_equal(val, [12.5])
    *_, info = gst.read_operator(path, return_info=True)
    assert info['inactive_internal_entries_removed'] == 2
    assert info['inactive_internal_nodes_removed'] == 1


@pytest.mark.parametrize('record', ['-20,1,-20,1,1e-30', '-20,1,7,1,1.',
                                    '7,1,-20,1,-1.', '-20,1,-20,1,nan'])
def test_active_or_nonfinite_internal_matrix_entries_are_not_discarded(tmp_path, record):
    path = tmp_path / 'operator.mtx'
    path.write_text(record+'\n7,1,7,1,12.5\n')
    with pytest.raises(ValueError, match='(?i)(internal|nonfinite)'):
        gst.read_operator(path)


def test_internal_loads_are_not_silently_lost(tmp_path):
    path = tmp_path / 'load.mtx'
    path.write_text('-20,1,0.0\n7,1,123.0\n')
    n, d, v = gst.read_load(path)
    np.testing.assert_array_equal(n, [7])
    np.testing.assert_array_equal(v, [123.])
    path.write_text('-20,1,1.0\n7,1,123.0\n')
    with pytest.raises(ValueError, match='(?i)internal'):
        gst.read_load(path)


def test_completed_matrix_output_is_reused_after_an_ingestion_failure(column, monkeypatch):
    folder, job = gst.write_matrix_job(column['run'], column['inp'])
    fake_abaqus_matrix_output(folder, job, column)
    with (folder / (job+'_STIF1.mtx')).open('a') as stream:
        stream.write('-20,1,-20,1,0.0\n')
    (folder / (job+'.sta')).write_text('THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n')
    (folder / (job+'.lck')).write_text('still running')
    assert gst.completed_matrix_job(column['run'], column['inp']) is None
    (folder / (job+'.lck')).unlink()
    assert gst.completed_matrix_job(column['run'], column['inp']) == (folder, job)
    companion = folder / (job+'.inp')
    original_input = companion.read_text()
    companion.write_text('different companion model')
    assert gst.completed_matrix_job(column['run'], column['inp']) is None
    companion.write_text(original_input)
    matrix_hash = gdlc.file_sha256(folder / (job+'_STIF1.mtx'))
    monkeypatch.setattr(gdlc, '_run_cmd', lambda *a: pytest.fail('completed matrix job solved again'))
    meta = gdlc.ensure_stiffness(column['run'], gdlc.Settings(stiffness='required', plots='none'))
    assert meta['matrix_job_reused'] is True
    assert meta['matrix']['inactive_internal_entries_removed'] == 1
    assert gdlc.file_sha256(folder / (job+'_STIF1.mtx')) == matrix_hash
    # A processed export is subsequently reused through its verified cache, not re-ingested.
    assert gst.completed_matrix_job(column['run'], column['inp']) is None


@pytest.mark.parametrize("numbering, triangle", [("sequential", "lower"), ("reversed", "both")])
def test_node_map_from_load_codes_passes_rigid_body_check(column, tmp_path, numbering, triangle):
    folder, job = gst.write_matrix_job(column["run"], column["inp"])
    fake_abaqus_matrix_output(folder, job, column, triangle=triangle, numbering=numbering)
    nodes = gst.instance_nodes(column["inp"].read_text().splitlines())
    xyz_of = {k: column["xyz"][r] for r, k in enumerate(column["keys"])}
    info = gst.ingest_matrix_job(folder, job, nodes, xyz_of)
    assert info["node_map"].startswith("load-encoded")
    assert max(info["rigid_body_residual"]) < 1e-12
    import scipy.sparse as sp
    K = sp.load_npz(folder / gst.RAW_NPZ)
    assert abs(K - column["K"]).max() <= 1e-12 * abs(column["K"]).max()


def test_wrong_node_map_is_refused(column):
    folder, job = gst.write_matrix_job(column["run"], column["inp"])
    fake_abaqus_matrix_output(folder, job, column, numbering="reversed", with_load=False)
    for f in folder.glob(f"{job}_LOAD*.mtx"):
        f.unlink()
    nodes = gst.instance_nodes(column["inp"].read_text().splitlines())
    xyz_of = {k: column["xyz"][r] for r, k in enumerate(column["keys"])}
    with pytest.raises(RuntimeError, match="rigid-body"):
        gst.ingest_matrix_job(folder, job, nodes, xyz_of)


def test_lift_is_the_minimum_energy_admissible_completion(column):
    inp = gdlc.read_inp(column["inp"])
    key_row = {k: r for r, k in enumerate(column["keys"])}
    model = gst.EnergyModel(column["K"], column["xyz"], key_row, inp.pairs, inp.boundaries, [1, 2], 2)
    rng = np.random.default_rng(3)
    W = rng.standard_normal((len(model.P), 3))
    Z, SW, res = model.lift(W)
    assert res < 1e-10
    q = np.zeros((len(model.indep), 3)); q[model.P] = W; q[model.F] = Z
    e0 = np.einsum("im,im->m", q, model.Kq @ q)
    np.testing.assert_allclose(e0, np.einsum("im,im->m", W, SW), rtol=1e-10)
    q2 = q.copy(); q2[model.F] += 1e-3 * rng.standard_normal(Z.shape)
    assert np.all(np.einsum("im,im->m", q2, model.Kq @ q2) > e0)
    # the exported modes satisfy every MPC/support; a perturbed vector does not
    assert model.constraint_error(column["V"]).max() < 1e-10
    bad = column["V"][:, :1].copy(); bad[6 * next(iter(model.master_of))] += 1.0
    assert model.constraint_error(bad)[0] > 1e-3


def test_end_to_end_energy_split(column, monkeypatch):
    m = column

    def fake_run(cmd, cwd):
        job = [c for c in cmd if c.startswith("job=")][0][4:]
        fake_abaqus_matrix_output(Path(cwd), job, m)
        return 0

    monkeypatch.setattr(gdlc, "_run_cmd", fake_run)
    st = gdlc.Settings(plots="none", stiffness="required", abaqus_cpus=1)
    extra_input = m['run'] / 'unrelated.inp'
    extra_input.write_text('unrelated input; export hashes identify the correct one')
    try:
        meta = gdlc.ensure_stiffness(m["run"], st)
        assert meta and gst.stiffness_status(m["run"], gdlc.file_sha256(m["inp"]))
        rep = gdlc.process_run(m["run"], st)
    finally:
        extra_input.unlink()
    ev = rep["energy_evaluation"]
    assert ev["available"]
    v = ev["validation"]
    assert v["mode_constraint_error_max"] < 1e-10 and v["k_orthogonality"]["max_normalised_offdiagonal"] < 1e-8
    assert v["lift_relative_residual_max"] < 1e-10
    out = m["run"] / st.out_name
    rows = list(csv.DictReader(open(out / "mode_participation.csv", encoding="utf-8")))
    for r in rows:
        assert abs(sum(float(r[f"E_{k}_signed_pct"]) for k in gdlc.CLASSES) - 100) < 1e-3   # csv: 4 decimals
        assert abs(sum(float(r[f"E_rel_{k}_pct"]) for k in gdlc.CLASSES) - 100) < 1e-3
        assert float(r["E_beyond_section_pct"]) > -1e-6               # in-plane minimum <= mode energy
        assert r["E_reliable_dominant"] in gdlc.MECH
    z = np.load(out / "mode_energy.npz")
    np.testing.assert_allclose(z["E"], np.transpose(z["E"], (0, 2, 1)), rtol=1e-10, atol=1e-12)
    assert np.all(z["E"].sum((1, 2)) <= z["U"] * (1 + 1e-9))
    # reused on the next call (same .inp): no new Abaqus job
    monkeypatch.setattr(gdlc, "_run_cmd", lambda *a: pytest.fail("matrix job run again"))
    assert gdlc.ensure_stiffness(m["run"], st)["created_at"] == meta["created_at"]


def test_energy_evaluation_can_be_switched_off(column):
    rep = gdlc.process_run(column["run"], gdlc.Settings(plots="none", stiffness="never", sensitivity=False))
    assert rep["energy_evaluation"] == dict(available=False, reason="disabled (--stiffness never)")


@pytest.mark.parametrize("problem", ["orthogonality", "residual", "balance", "negative", "nonfinite"])
def test_unacceptable_energy_is_refused(problem):
    E = np.array([np.eye(5)])
    U = np.array([6.])
    residual, orth = 1e-12, dict(max_normalised_offdiagonal=1e-10)
    if problem == "orthogonality": orth["max_normalised_offdiagonal"] = .2
    if problem == "residual": residual = .1
    if problem == "balance": U[:] = 1.
    if problem == "negative": E[0, 0, 0] = -1.
    if problem == "nonfinite": E[0, 0, 0] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        gst.validate_energy_result(E, U, residual, orth)


def test_energy_solve_uses_parallel_rhs_on_large_cpu_counts():
    from types import SimpleNamespace
    model = gst.EnergyModel.__new__(gst.EnergyModel)
    model.threads = 192
    calls = []
    model.lu = SimpleNamespace(solve=lambda b: (calls.append(b.shape), b.copy())[1])
    B = np.arange(640.).reshape(2, 320)
    np.testing.assert_array_equal(model.solve(B), B)
    assert len(calls) > 1


def test_bad_energy_diagnostics_cannot_have_high_energy_confidence():
    M = np.zeros((5, 5)); M[0, 0] = 10.; M[1, 1] = 9.; M[0, 1] = M[1, 0] = -9.
    rows = [dict(flags="", confidence="high", reliable_dominant="G")]
    en = dict(E=np.array([M]), Ecl=np.array([M]), U=np.array([100.]), bolt=np.array([.8]))
    gdlc.add_energy_columns(rows, [], en, [1], gdlc.Settings())
    assert rows[0]["kinematic_confidence"] == "high"
    assert rows[0]["energy_confidence"] == "low"
    assert rows[0]["confidence"] != "high"


def test_matrix_export_rejects_a_preloaded_or_unmapped_model(column):
    lines = column["inp"].read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if l.lower().startswith("*step"))
    for extra in (["*Step, name=Preload", "*Static", "1., 1.", "*End Step"],
                  ["*Equation", "2", "P1.1, 1, 1., P2.1, 1, -1."],
                  ["*Initial Conditions, type=STRESS", "ALL, 1., 0., 0."]):
        with pytest.raises((ValueError, NotImplementedError), match="(?i)(preload|initial|constraint|equation|step)"):
            gst.matrix_job_lines(lines[:i] + extra + lines[i:])


def test_matrix_failure_does_not_retry_a_different_contact_model(column, monkeypatch):
    calls = []
    monkeypatch.setattr(gst, "stiffness_status", lambda *a: None)
    monkeypatch.setattr(gdlc, "_run_cmd", lambda cmd, cwd: (calls.append(cmd), 1)[1])
    with pytest.raises(RuntimeError, match="(?i)matrix"):
        gdlc.ensure_stiffness(column["run"], gdlc.Settings(stiffness="required", plots="none"))
    assert len(calls) == 1


def test_projection_is_invariant_to_memory_chunking(column, monkeypatch):
    exp, meta = gdlc.choose_export(column["run"], None)
    mesh = gdlc.load_mesh(exp, meta, column["inp"])
    st = gdlc.Settings(gpu="off", plots="none", stiffness="never")
    groups = gdlc.build_ring_groups(mesh)
    for g in groups:
        gdlc.detect_reference_points(g, 2., st); gdlc.build_basis(g)
    modes, U = gdlc.load_modes(exp, meta, mesh, st)
    big = gdlc.decompose(groups, mesh, U, st)
    per_ring = max(2 * g.n * len(modes) * 8 * 16 for g in groups)
    monkeypatch.setattr(gdlc, "_available_ram", lambda: per_ring * 8)
    small = gdlc.decompose(groups, mesh, U, st)
    for key in ("Htot", "E", "Es", "prof", "fold_ratio"):
        np.testing.assert_allclose(small[key], big[key], rtol=1e-10, atol=1e-9)


def test_allocation_failure_reduces_chunk_and_preserves_results(column, monkeypatch):
    exp, meta = gdlc.choose_export(column["run"], None)
    mesh = gdlc.load_mesh(exp, meta, column["inp"])
    st = gdlc.Settings(gpu="off", plots="none", stiffness="never")
    groups = gdlc.build_ring_groups(mesh)
    for g in groups:
        gdlc.detect_reference_points(g, 2., st); gdlc.build_basis(g)
    _, U = gdlc.load_modes(exp, meta, mesh, st)
    expected = gdlc.decompose(groups, mesh, U, st)
    original = gdlc._projection_chunk
    attempts = []
    def flaky(*args, **kw):
        attempts.append(len(args[3]))
        if len(attempts) == 1: raise MemoryError("simulated allocation failure")
        return original(*args, **kw)
    monkeypatch.setattr(gdlc, "_projection_chunk", flaky)
    got = gdlc.decompose(groups, mesh, U, st)
    assert attempts[1] < attempts[0]
    np.testing.assert_allclose(got["E"], expected["E"], rtol=1e-10, atol=1e-9)


def test_too_small_vram_falls_back_before_first_allocation(column):
    from types import SimpleNamespace as NS
    exp, meta = gdlc.choose_export(column['run'], None)
    mesh = gdlc.load_mesh(exp, meta, column['inp'])
    st = gdlc.Settings(gpu='off', plots='none', stiffness='never')
    g = gdlc.build_ring_groups(mesh)[0]
    gdlc.detect_reference_points(g, 2., st); gdlc.build_basis(g)
    _, U = gdlc.load_modes(exp, meta, mesh, st)
    ids = np.arange(len(g.node_rows)); dz = np.ones(len(ids))
    expected = gdlc._projection_range(g, mesh, U, ids, dz, np, False, 1)[1]
    fake = NS(cuda=NS(runtime=NS(memGetInfo=lambda:(0,0))),
              get_default_memory_pool=lambda:NS(free_bytes=lambda:0))
    got = gdlc._projection_range(g, mesh, U, ids, dz, fake, True, 1)[1]
    np.testing.assert_allclose(got['total'], expected['total'])


def test_reference_required_reports_energy_unavailable_without_solving(column, monkeypatch):
    marker = column['run'] / 'gdlc_reference.json'
    marker.write_text(json.dumps(dict(kind='G')))
    monkeypatch.setattr(gdlc, '_run_cmd', lambda *a: pytest.fail('reference classification dispatched a solver'))
    try:
        rep = gdlc.process_run(column['run'], gdlc.Settings(plots='none', stiffness='required'))
        assert rep['energy_evaluation']['available'] is False
        assert 'reference' in rep['energy_evaluation']['reason']
    finally:
        marker.unlink()


def test_matrix_outputs_are_never_overwritten_and_cache_checks_hashes(column, monkeypatch):
    def fake_run(cmd, cwd):
        job = next(c[4:] for c in cmd if c.startswith('job='))
        fake_abaqus_matrix_output(Path(cwd), job, column)
        return 0
    monkeypatch.setattr(gdlc, '_run_cmd', fake_run)
    st = gdlc.Settings(plots='none', stiffness='required', abaqus_cpus='auto')
    gdlc.ensure_stiffness(column['run'], st)
    meta = gst.stiffness_status(column['run'], gdlc.file_sha256(column['inp']))
    first_folder = Path(meta['export_directory'])
    first = first_folder / gst.RAW_NPZ
    original = first.read_bytes()
    st.stiffness = 'always'
    gdlc.ensure_stiffness(column['run'], st)
    assert first.read_bytes() == original
    meta = gst.stiffness_status(column['run'], gdlc.file_sha256(column['inp']))
    assert Path(meta['export_directory']) != first_folder
    originals = {folder / gst.RAW_NPZ:(folder / gst.RAW_NPZ).read_bytes()
                 for folder in column['run'].glob(gst.STIFFNESS_DIR+'*') if (folder / gst.RAW_NPZ).is_file()}
    try:
        for path in originals:
            with path.open('ab') as stream: stream.write(b'tampered')
        assert gst.stiffness_status(column['run'], gdlc.file_sha256(column['inp'])) is None
    finally:
        for path, data in originals.items(): path.write_bytes(data)
