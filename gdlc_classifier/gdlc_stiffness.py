# -*- coding: utf-8 -*-
r"""
gdlc_stiffness.py - elastic stiffness of a buckling run and the strain-energy evaluation
of the GDLC components (used by gdlc_mode_decomposition.py; no separate command).

Part 1 - what Abaqus has to provide (run once per buckling run, after its solve)
------------------------------------------------------------------------------
A small companion job, written from the run's own .inp: the model data unchanged (mesh,
sections, material, MPCs, contact) except the model-data *Boundary blocks, and one step

    *Step, name=GDLC_KE
    *Matrix Generate, STIFFNESS, LOAD, MPC=NO
    *Matrix Output, STIFFNESS, LOAD, FORMAT=MATRIX INPUT
    *Cload            one load per node on DOF 1 whose value encodes (instance, label)
    *End Step

  * STIFFNESS, MPC=NO, no boundary condition: the raw assembled elastic stiffness K of the
    elements (6 DOF per shell node). It is the stiffness of the buckling problem
    K phi = lambda (-K_g) phi at the unloaded base state (*BUCKLE is a perturbation step
    from the initial state; no preload, so no initial-stress term). Boundary conditions
    and the BEAM/TIE/PIN MPCs are applied afterwards in Python from the .inp, so the DOF
    space is explicit and the matrix can be checked exactly: K must annihilate the six
    rigid-body motions of the whole model (a wrong node map fails this test by orders of
    magnitude).
  * LOAD: Abaqus writes matrices with its internal node numbers. The coded *Cload values
    come back in the load matrix next to those numbers, which gives the exact map from
    internal number to (instance, label) without relying on an undocumented numbering.
    (Fallback, also checked by the rigid-body test: internal numbers = instance order,
    labels ascending.)
  * The geometric stiffness K_g is NOT exported: Abaqus has no direct output for it, and
    differencing a preloaded tangent stiffness mixes in initial-displacement terms. The
    constrained reference stresses come from REF_G / REF_L; check their actual modal purity.

Outputs in <run>/stiffness_export/: <job>_KE.inp and Abaqus' files (<job>_KE_STIF1.mtx,
<job>_KE_LOAD1.mtx, ...), stiffness_raw.npz (scipy.sparse CSR, full symmetric, N mm),
stiffness_dofs.npz (instance, label, dof, coordinates of every row) and
stiffness_export.json (provenance: .inp sha256, node-map method, validation).

Part 2 - energy evaluation of the GDLC components (in-plane kinematic classes)
------------------------------------------------------------------------------
P = in-plane translations of every node without a support on them (the GDLC variables),
F = the other DOFs (axial translation, rotations) without supports, with the axial and
rotational rows of the bolt MPCs eliminated (axial slip and rotation continuity at the
bolts, i.e. the composite action, are kept). A GDLC component w_f (in-plane field of class
f) is completed to a full displacement field by minimum strain energy:
    q_F = -K_FF^-1 K_FP w_f,P ,  S = K_PP - K_PF K_FF^-1 K_FP (never formed)
The in-plane MPC rows are relaxed for the components: a class component need not be
compatible with the bolts in its own plane (C, the relative rigid motion of the pieces, is
not); overwriting its in-plane values at the bolt nodes would create artificial kinks and
spurious energy. Their in-plane incompatibility along the bolt axes is reported instead
(bolt_incompatibility; it cancels in the sum for a compatible mode). The modes themselves
are checked against EVERY constraint of the .inp. Energies (factor 1/2 omitted):
    E_fg = w_f^T S w_g          (5 x 5 per mode; sum_fg E_fg = w^T S w)
    U    = v^T K v              (the mode itself, with its own U3 and UR, raw K: exact)
    U - w^T S w >= 0            energy of the mode not implied by its in-plane field
                                ("beyond section": axial/rotation content above the minimum)
The split of w^T S w is exact; the diagonal shares E_ff / sum E_gg and the additive signed
shares sum_g E_fg / sum E are reported, with the energy cross term.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import warnings
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

STIFFNESS_DIR = "stiffness_export"
EXPORT_JSON = "stiffness_export.json"
RAW_NPZ = "stiffness_raw.npz"
DOFS_NPZ = "stiffness_dofs.npz"
FORMAT_VERSION = 2
STEP_NAME = "GDLC_KE"
RIGID_TOL = 1e-7            # max |K r| / max (|K| |r|) for the six rigid-body motions (at full
                            # precision; relaxed to 50 x the printed relative precision of the file)
SYMMETRY_TOL = 1e-10        # max |K - K^T| / max |K|
SUPPORTED_MPC = ("BEAM", "TIE", "PIN")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def _kw(line: str):
    parts = [p.strip() for p in line.strip()[1:].split(",")]
    prm = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            prm[k.strip().lower()] = v.strip()
        elif p:
            prm[p.lower()] = True
    return parts[0].lower(), prm


def _is_keyword(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("*") and not s.startswith("**")


# ----------------------------------------------------------------------------
# Part 1a: the matrix-generation job
# ----------------------------------------------------------------------------
def instance_nodes(lines: list[str]) -> list[tuple[str, list[int]]]:
    """[(INSTANCE, sorted node labels)] in *Instance order (part-level *Node blocks)."""
    parts: dict[str, list[int]] = {}
    order: list[tuple[str, str]] = []
    part = None
    in_node = False
    for raw in lines:
        if _is_keyword(raw):
            kw, prm = _kw(raw)
            in_node = False
            if kw == "part":
                part = str(prm.get("name")).upper()
                parts.setdefault(part, [])
            elif kw == "end part":
                part = None
            elif kw == "node" and part is not None:
                in_node = True
            elif kw == "instance":
                order.append((str(prm.get("name")).upper(), str(prm.get("part")).upper()))
            elif kw == "node" and part is None:
                raise ValueError("assembly-level nodes are not supported by the stiffness export")
            continue
        if in_node and raw.strip() and not raw.lstrip().startswith("**"):
            parts[part].append(int(raw.split(",")[0]))
    out = [(name, sorted(parts.get(p, []))) for name, p in order]
    if not out or not all(lab for _, lab in out):
        raise ValueError("no part instances with nodes found in the .inp")
    return out


def matrix_job_lines(inp_lines: list[str], drop_contact: bool = False):
    """Model data of the run (up to its first *Step) without model-data *Boundary blocks
    or write-only *Restart output controls (which do not import an analysis state),
    followed by the matrix-generation step. drop_contact also removes the contact definitions
    (*Contact...; legacy helper option, never used by the production workflow). Returns (lines, node_codes, code_base, removed_boundary_blocks)."""
    i_step = next((i for i, l in enumerate(inp_lines) if _is_keyword(l) and _kw(l)[0] == "step"), None)
    if i_step is None:
        raise ValueError("no *Step in the .inp")
    cards = [_kw(l) for l in inp_lines if _is_keyword(l)]
    keywords = [kw for kw, _ in cards]
    if keywords.count("step") != 1 or "buckle" not in keywords:
        raise ValueError("elastic export requires one initial-state buckling step; preload steps are not supported")
    unsupported = {"equation", "coupling", "kinematic", "distributing", "rigid body",
                   "transform", "tie", "initial conditions", "import", "model change"}
    found = sorted(set(keywords) & unsupported)
    if found:
        raise NotImplementedError("unmapped constraint or initial/preload state in stiffness export: " + ", ".join(found))
    restart_write_params = {'write', 'frequency', 'number interval', 'time marks', 'overlay'}
    for kw, params in cards:
        if kw == 'restart' and ('write' not in params or set(params) - restart_write_params):
            raise NotImplementedError('restart state reading is not supported by the initial-state '
                                      'elastic export; only explicit write-only *Restart output controls are allowed')
    model, removed, skip = [], 0, False
    for raw in inp_lines[:i_step]:
        if _is_keyword(raw):
            kw, _ = _kw(raw)
            if kw == "include":
                raise ValueError("*Include files are not supported by the stiffness export")
            if kw == 'restart':
                # Output-only; the companion matrix step does not need restart files.
                # READ (including READ+WRITE) was rejected above. This option has no data lines.
                skip = False
                continue
            skip = kw == "boundary" or (drop_contact and kw.startswith("contact"))
            removed += kw == "boundary"
            if skip:
                model.append("** GDLC: removed for the raw stiffness: " + raw.strip())
                continue
        elif skip:
            if not raw.lstrip().startswith("**"):
                model.append("** GDLC: removed  " + raw.strip())
            continue
        model.append(raw.rstrip("\r\n"))
    nodes = instance_nodes(inp_lines)
    max_label = max(max(lab) for _, lab in nodes)
    base = 10 ** (len(str(max_label)) + 1)
    codes = {}
    cload = []
    for k, (inst, labels) in enumerate(nodes, 1):
        for lab in labels:
            c = k * base + lab
            codes[(inst, lab)] = c
            cload.append(f"{inst}.{lab}, 1, {c}.")
    step = ["** ----------------------------------------------------------------",
            "** GDLC stiffness export: raw elastic stiffness (no boundary condition, MPCs not",
            "** applied) and a load matrix whose values encode (instance, node label).",
            f"*Step, name={STEP_NAME}",
            "*Matrix Generate, STIFFNESS, LOAD, MPC=NO",
            "*Matrix Output, STIFFNESS, LOAD, FORMAT=MATRIX INPUT",
            "*Cload"] + cload + ["*End Step"]
    return model + step, codes, base, removed


def write_matrix_job(run_dir: Path, inp_path: Path, drop_contact: bool = False) -> tuple[Path, str]:
    """Write <run>/stiffness_export/<job>_KE.inp (+ the run's abaqus_v6.env). Returns (folder, job)."""
    out = run_dir / STIFFNESS_DIR
    suffix = 2
    while out.exists() and any(out.iterdir()):
        out = run_dir / f'{STIFFNESS_DIR}_{suffix}'
        suffix += 1
    out.mkdir(exist_ok=True)
    job = inp_path.stem + "_KE"
    lines = inp_path.read_text(encoding="utf-8", errors="replace").splitlines()
    new, codes, base, removed = matrix_job_lines(lines, drop_contact)
    (out / f"{job}.inp").write_text("\n".join(new) + "\n", encoding="utf-8")
    env = run_dir / "abaqus_v6.env"
    if env.is_file():
        (out / "abaqus_v6.env").write_text(env.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    (out / f"{job}_codes.json").write_text(json.dumps(dict(code_base=base, removed_boundary_blocks=removed,
                                                           contact_removed=drop_contact,
                                                           source_inp=inp_path.name)), encoding="utf-8")
    return out, job


def completed_matrix_job(run_dir: Path, inp_path: Path) -> tuple[Path, str] | None:
    """Recover completed, unprocessed output after an interrupted/failed ingestion.

    Never overwrite processed artifacts or trust just a job/source name: the entire
    companion input and node-code definition must match the current source model.
    """
    job = inp_path.stem + '_KE'
    lines, _, base, removed = matrix_job_lines(inp_path.read_text(encoding='utf-8', errors='replace').splitlines())
    expected = '\n'.join(lines)+'\n'
    folders = sorted((p for p in run_dir.glob(STIFFNESS_DIR+'*') if p.is_dir()),
                     key=lambda p:p.stat().st_mtime_ns, reverse=True)
    for folder in folders:
        if (folder / (job+'.lck')).exists() or any((folder / f).exists() for f in (RAW_NPZ, DOFS_NPZ, EXPORT_JSON)):
            continue
        if len(list(folder.glob(job+'_STIF*.mtx'))) != 1 or len(list(folder.glob(job+'_LOAD*.mtx'))) != 1:
            continue
        try:
            if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' not in (folder / (job+'.sta')).read_text(errors='replace').upper():
                continue
            if (folder / (job+'.inp')).read_text(encoding='utf-8', errors='replace') != expected:
                continue
            codes = json.loads((folder / (job+'_codes.json')).read_text(encoding='utf-8'))
            if (codes.get('code_base') != base or codes.get('removed_boundary_blocks') != removed
                    or codes.get('contact_removed') is not False or codes.get('source_inp') != inp_path.name):
                continue
        except (OSError, ValueError):
            continue
        return folder, job
    return None


# ----------------------------------------------------------------------------
# Part 1b: reading the .mtx files
# ----------------------------------------------------------------------------
def _numeric_block(path: Path, width: int) -> np.ndarray:
    """All numeric data lines of an Abaqus MATRIX INPUT text file as an (n, width) array.
    Keyword/comment lines (starting with '*') are skipped; every data line must have exactly
    `width` comma-separated values."""
    rows, n_lines = [], 0
    pat = re.compile(rb"(?m)^[ \t]*\*.*(?:\r?\n|$)")
    tail = b""
    with open(path, "rb") as f:
        while True:
            blk = f.read(64 * 1024 * 1024)
            if not blk and not tail:
                break
            data = tail + blk
            if blk:
                cut = data.rfind(b"\n")
                if cut < 0:
                    tail = data
                    continue
                data, tail = data[:cut + 1], data[cut + 1:]
            else:
                tail = b""
            data = pat.sub(b"", data)
            n_lines += sum(1 for ln in data.split(b"\n") if ln.strip())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                vals = np.fromstring(data.replace(b",", b" "), dtype=np.float64, sep=" ")
            rows.append(vals)
            if not blk:
                break
    vals = np.concatenate(rows) if rows else np.zeros(0)
    if vals.size != n_lines * width:
        raise ValueError(f"{path.name}: expected {width} values per data line "
                         f"({n_lines} lines, {vals.size} values)")
    return vals.reshape(-1, width)


def printed_digits(path: Path, sample: int = 2000) -> int:
    """Significant digits of the matrix values as printed (last field of the first data lines)."""
    digits = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.lstrip().startswith("*") or not line.strip():
                continue
            v = line.rsplit(",", 1)[-1].strip().upper().split("E")[0].lstrip("+-")
            digits = max(digits, len(v.replace(".", "").lstrip("0")) or 1)
            sample -= 1
            if sample <= 0:
                break
    return digits or 16


def read_operator(path: Path, return_info=False):
    """MATRIX INPUT entries; discard only exactly zero entries at inactive internal nodes.

    Internal nodes are valid Abaqus labels (e.g. contact/Lagrange variables). Nonzero
    internal entries require a mapped/condensed operator and cannot be deleted.
    """
    a = _numeric_block(path, 5)
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{path.name}: nonfinite matrix fields')
    ints = a[:, :4]
    if not np.all(ints == np.round(ints)):
        raise ValueError(f"{path.name}: non-integer node/DOF fields")
    rn, rd, cn, cd = (ints[:, k].astype(np.int64) for k in range(4))
    if np.any(rn == 0) or np.any(cn == 0):
        raise ValueError(f'{path.name}: zero node labels are invalid')
    internal = (rn < 0) | (cn < 0)
    if np.any(a[internal, 4] != 0.):
        raise ValueError(f'{path.name}: nonzero internal-node stiffness/coupling requires explicit '
                         'internal-DOF treatment; these entries cannot be discarded')
    negative_nodes = np.unique(np.r_[rn[rn < 0], cn[cn < 0]])
    info = dict(inactive_internal_entries_removed=int(internal.sum()),
                inactive_internal_nodes_removed=int(len(negative_nodes)))
    if np.any(internal):
        keep = ~internal
        rn, rd, cn, cd, a = rn[keep], rd[keep], cn[keep], cd[keep], a[keep]
    if np.any((rd < 1) | (rd > 6) | (cd < 1) | (cd > 6)):
        raise ValueError(f"{path.name}: DOF outside 1..6 (acoustic/other DOFs are not supported)")
    result = (rn, rd, cn, cd, a[:, 4].copy())
    return (*result, info) if return_info else result


def read_load(path: Path):
    """(node, dof, value) of a MATRIX INPUT load file."""
    a = _numeric_block(path, 3)
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{path.name}: nonfinite load fields')
    if not np.all(a[:, :2] == np.round(a[:, :2])):
        raise ValueError(f"{path.name}: non-integer node/DOF fields")
    n, d = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)
    if np.any(n == 0):
        raise ValueError(f'{path.name}: zero node labels are invalid')
    if np.any(a[n < 0, 2] != 0.):
        raise ValueError(f'{path.name}: nonzero internal-node load cannot be discarded')
    keep = n > 0
    return n[keep], d[keep], a[keep, 2].copy()


def decode_node_map(n, d, v, base: int, n_inst: int) -> dict[int, tuple[int, int]]:
    """internal node number -> (instance index 1.., label) from the coded DOF-1 loads."""
    sel = d == 1
    code = np.round(v[sel]).astype(np.int64)
    if np.any(np.abs(v[sel] - code) > 1e-6 * np.maximum(1.0, np.abs(code))):
        raise ValueError("load codes are not integers (precision of the load matrix too low)")
    inst, lab = code // base, code % base
    if np.any((inst < 1) | (inst > n_inst)) or np.any(lab <= 0):
        raise ValueError("load codes outside the encoded range")
    nodes = n[sel]
    out = {}
    for a, i, l in zip(nodes.tolist(), inst.tolist(), lab.tolist()):
        if out.setdefault(a, (i, l)) != (i, l):
            raise ValueError(f"internal node {a} carries two different codes")
    if len(set(out.values())) != len(out):
        raise ValueError("two internal nodes decode to the same (instance, label)")
    return out


# ----------------------------------------------------------------------------
# Part 1c: assembly in a named DOF order and the exact checks
# ----------------------------------------------------------------------------
def assemble(rn, rd, cn, cd, val, row_of_node: dict[int, int], n_rows: int):
    """Full symmetric CSR in the order (node row, dof) -> 6*row + dof-1. A file with one
    triangle only is mirrored; a file with both is checked for symmetry."""
    import scipy.sparse as sp
    lut_keys = np.fromiter(row_of_node.keys(), dtype=np.int64)
    lut_vals = np.fromiter(row_of_node.values(), dtype=np.int64)
    lut = np.full(int(max(lut_keys.max(), rn.max(), cn.max())) + 1, -1, np.int64)
    lut[lut_keys] = lut_vals
    ri, ci = lut[rn], lut[cn]
    if np.any(ri < 0) or np.any(ci < 0):
        raise ValueError("matrix entries for nodes missing from the node map")
    i = 6 * ri + rd - 1
    j = 6 * ci + cd - 1
    # triangle test in the FILE's own numbering (a node map does not preserve it)
    fi, fj = 6 * rn + rd, 6 * cn + cd
    upper, lower = np.any(fi < fj), np.any(fi > fj)
    if upper and lower:
        K = sp.coo_matrix((val, (i, j)), shape=(6 * n_rows,) * 2).tocsr()
        mirrored = False
    else:
        off = i != j
        K = sp.coo_matrix((np.r_[val, val[off]], (np.r_[i, j[off]], np.r_[j, i[off]])),
                          shape=(6 * n_rows,) * 2).tocsr()
        mirrored = True
    K.sum_duplicates()
    asym = abs(K - K.T).max() if not mirrored else 0.0
    kmax = abs(K).max()
    if asym > SYMMETRY_TOL * kmax:
        raise ValueError(f"stiffness matrix is not symmetric (|K-K^T|max/|K|max = {asym/kmax:.2e})")
    return K, dict(one_triangle_mirrored=mirrored, symmetry_error=float(asym / kmax), nnz=int(K.nnz),
                   k_max=float(kmax))


def rigid_body_fields(xyz: np.ndarray) -> np.ndarray:
    """(6N, 6) rigid translations x, y, z and small rotations about x, y, z (centroid)."""
    n = len(xyz)
    c = xyz - xyz.mean(0)
    R = np.zeros((6 * n, 6))
    for k in range(3):
        R[k::6, k] = 1.0
    for k in range(3):
        th = np.eye(3)[k]
        R[np.arange(n)[:, None] * 6 + np.arange(3)[None, :], 3 + k] = np.cross(th, c)
        R[3 + k::6, 3 + k] = 1.0
    return R


def rigid_body_residual(K, xyz: np.ndarray) -> list[float]:
    """max |K r| / max (|K| |r|) for each rigid motion r: ~1e-12 for a correct matrix and node map."""
    R = rigid_body_fields(xyz)
    KR = K @ R
    aK = abs(K)
    return [float(np.abs(KR[:, k]).max() / max(float((aK @ np.abs(R[:, k])).max()), 1e-300)) for k in range(6)]


def ingest_matrix_job(folder: Path, job: str, nodes: list[tuple[str, list[int]]], xyz_of: dict) -> dict:
    """Read <job>_STIF1.mtx (+ LOAD1), build the node map, assemble and check K, write
    stiffness_raw.npz / stiffness_dofs.npz. xyz_of: (INSTANCE, label) -> coordinates."""
    import scipy.sparse as sp
    t0 = time.time()
    stif = sorted(folder.glob(f"{job}_STIF*.mtx"))
    if len(stif) != 1:
        raise FileNotFoundError(f"expected one {job}_STIF*.mtx in {folder}, found {len(stif)}")
    meta_codes = json.loads((folder / f"{job}_codes.json").read_text(encoding="utf-8"))
    rn, rd, cn, cd, val, internal_info = read_operator(stif[0], return_info=True)
    digits = printed_digits(stif[0])
    tol = max(RIGID_TOL, 50.0 * 10.0 ** (1 - digits))
    keys = [(inst, lab) for inst, labs in nodes for lab in labs]          # instance order, label asc.
    row_of_key = {k: r for r, k in enumerate(keys)}
    xyz = np.array([xyz_of[k] for k in keys], dtype=np.float64)
    candidates = []
    load = sorted(folder.glob(f"{job}_LOAD*.mtx"))
    if len(load) == 1:
        ln, ld, lv = read_load(load[0])
        try:
            dm = decode_node_map(ln, ld, lv, int(meta_codes["code_base"]), len(nodes))
            candidates.append(("load-encoded node codes",
                               {a: row_of_key[(nodes[i - 1][0], l)] for a, (i, l) in dm.items()}))
        except Exception as e:                       # fall back, reported
            candidates.append(("load-encoded node codes (unusable: %s)" % str(e)[:120], None))
    seq = {r + 1: r for r in range(len(keys))}
    candidates.append(("sequential internal numbering (instance order, labels ascending)", seq))
    tried = []
    for name, mp in candidates:
        if mp is None:
            tried.append(dict(method=name, passed=False))
            continue
        if len(mp) != len(keys) or set(np.unique(rn)) - set(mp):
            tried.append(dict(method=name, passed=False, reason="node map does not cover the matrix nodes"))
            continue
        try:
            K, info = assemble(rn, rd, cn, cd, val, mp, len(keys))
        except ValueError as e:
            tried.append(dict(method=name, passed=False, reason=str(e)))
            continue
        res = rigid_body_residual(K, xyz)
        ok = max(res) <= tol
        tried.append(dict(method=name, passed=bool(ok), rigid_body_residual=res))
        if ok:
            sp.save_npz(folder / RAW_NPZ, K, compressed=True)
            inst = np.repeat(np.array([k[0] for k in keys]), 6)
            lab = np.repeat(np.array([k[1] for k in keys], dtype=np.int64), 6)
            dof = np.tile(np.arange(1, 7, dtype=np.int64), len(keys))
            np.savez_compressed(folder / DOFS_NPZ, instance=inst, label=lab, dof=dof,
                                coordinates=np.repeat(xyz, 6, axis=0))
            return dict(node_map=name, node_map_trials=tried, matrix=dict(info, printed_digits=digits, **internal_info),
                        rigid_body_residual=res, rigid_body_tolerance=tol,
                        n_nodes=len(keys), n_dofs=6 * len(keys), stif_file=stif[0].name,
                        stif_sha256=_sha256(stif[0]), raw_sha256=_sha256(folder / RAW_NPZ),
                        dofs_sha256=_sha256(folder / DOFS_NPZ), elapsed_s=time.time() - t0)
    raise RuntimeError(f"no node map reproduces the rigid-body null space of K: {tried}")


def stiffness_status(run_dir: Path, inp_sha: str) -> dict | None:
    """The export's json when it is complete and belongs to the run's current .inp."""
    files = sorted(run_dir.glob(STIFFNESS_DIR+'*/'+EXPORT_JSON), key=lambda p:p.stat().st_mtime_ns, reverse=True)
    for f in files:
        d = f.parent
        try:
            meta = json.loads(f.read_text(encoding='utf-8'))
            if meta.get('format_version') != FORMAT_VERSION or meta.get('source_inp_sha256') != inp_sha:
                continue
            if any(not (d/name).is_file() or meta.get(key) != _sha256(d/name)
                   for name, key in ((RAW_NPZ, 'raw_sha256'), (DOFS_NPZ, 'dofs_sha256'))):
                continue
        except (OSError, ValueError):
            continue
        return dict(meta, export_directory=str(d.resolve()))
    return None


def load_stiffness(run_dir: Path, mesh_instance: np.ndarray, mesh_label: np.ndarray, folder=None):
    """K (CSR) permuted to the mesh node order (6 DOF per node, 6*row + dof-1)."""
    import scipy.sparse as sp
    d = Path(folder) if folder is not None else run_dir / STIFFNESS_DIR
    K = sp.load_npz(d / RAW_NPZ).tocsr()
    z = np.load(d / DOFS_NPZ, allow_pickle=False)
    inst, lab, dof = np.char.upper(z["instance"].astype(str)), z["label"], z["dof"]
    row = {(i, int(l)): r for r, (i, l) in enumerate(zip(mesh_instance, mesh_label))}
    try:
        node_rows = np.array([row[(i, int(l))] for i, l in zip(inst[::6], lab[::6])], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"stiffness node {e} is not in the mesh of the modal export")
    if len(node_rows) != len(mesh_label) or len(set(node_rows.tolist())) != len(node_rows):
        raise ValueError("stiffness nodes and mesh nodes differ")
    if not np.array_equal(dof, np.tile(np.arange(1, 7), len(node_rows))):
        raise ValueError("stiffness_dofs.npz is not node-major with 6 DOF per node")
    new = (6 * node_rows[:, None] + np.arange(6)[None, :]).ravel()      # old index -> new index
    perm = np.empty_like(new)
    perm[new] = np.arange(len(new))
    return K[perm][:, perm].tocsr()


# ----------------------------------------------------------------------------
# Part 2: constraints, minimum-energy lift and the component energies
# ----------------------------------------------------------------------------
def constraint_transform(xyz: np.ndarray, key_row: dict, pairs, boundaries, relax_dofs=()):
    """v = T q for the model's BEAM/TIE/PIN MPCs and boundary conditions (small rotations).
    relax_dofs: translational DOFs (1..3) of the MPC-dependent nodes that are NOT eliminated
    (they stay independent; used for the in-plane DOFs in the energy lift, see EnergyModel).
    Returns (T csr (6N x nq), indep (nq,) DOF indices, fixed bool (6N,), dependent bool (6N,),
    master_of: dependent node row -> (master row, kind))."""
    import scipy.sparse as sp
    n6 = 6 * len(xyz)
    fixed = np.zeros(n6, bool)
    for nodes, dofs in boundaries:
        for k in nodes:
            r = key_row.get((str(k[0]).upper(), int(k[1])))
            if r is None:
                raise ValueError(f"boundary node {k} not in the mesh")
            for dof_ in dofs:
                if 1 <= dof_ <= 6:
                    fixed[6 * r + dof_ - 1] = True
    dependent = np.zeros(n6, bool)
    master_of = {}
    relax = set(int(d) for d in relax_dofs)
    for kind, a, b in pairs:
        if kind not in SUPPORTED_MPC:
            raise NotImplementedError(f"connection type {kind} is not supported by the energy evaluation "
                                      f"(supported: {', '.join(SUPPORTED_MPC)})")
        ra, rb = key_row.get((str(a[0]).upper(), int(a[1]))), key_row.get((str(b[0]).upper(), int(b[1])))
        if ra is None or rb is None:
            raise ValueError(f"MPC node {a} / {b} not in the mesh")
        if rb in master_of:
            raise ValueError(f"node {b} is dependent in two MPCs")
        master_of[rb] = (ra, kind)
        for dof_ in range(1, 4 if kind == "PIN" else 7):
            if dof_ not in relax:
                dependent[6 * rb + dof_ - 1] = True
    for rb, (ra, _) in master_of.items():
        if any(r_ == ra for r_ in master_of) or dependent[6 * ra:6 * ra + 6].any():
            raise NotImplementedError("chained MPCs (a master that is itself dependent) are not supported")
        if (fixed[6 * rb:6 * rb + 6] & dependent[6 * rb:6 * rb + 6]).any():
            raise ValueError(f"MPC-dependent node row {rb} also carries a boundary condition")
    indep = np.nonzero(~fixed & ~dependent)[0]
    col = np.full(n6, -1, np.int64)
    col[indep] = np.arange(len(indep))
    ri, ci, vv = [indep], [col[indep]], [np.ones(len(indep))]

    def add(row, dof_index, coef):
        c = col[dof_index]
        if dependent[row] and c >= 0 and coef != 0.0:      # a fixed master DOF contributes zero
            ri.append(np.array([row])); ci.append(np.array([c])); vv.append(np.array([coef]))

    for rb, (ra, kind) in master_of.items():
        d = xyz[rb] - xyz[ra]
        for c3 in range(3):
            add(6 * rb + c3, 6 * ra + c3, 1.0)
        if kind == "BEAM":                          # u_b = u_a + theta_a x d
            # x: th_y d_z - th_z d_y ; y: th_z d_x - th_x d_z ; z: th_x d_y - th_y d_x
            add(6 * rb + 0, 6 * ra + 4, d[2]); add(6 * rb + 0, 6 * ra + 5, -d[1])
            add(6 * rb + 1, 6 * ra + 5, d[0]); add(6 * rb + 1, 6 * ra + 3, -d[2])
            add(6 * rb + 2, 6 * ra + 3, d[1]); add(6 * rb + 2, 6 * ra + 4, -d[0])
        if kind in ("BEAM", "TIE"):
            for c3 in range(3):
                add(6 * rb + 3 + c3, 6 * ra + 3 + c3, 1.0)
    T = sp.coo_matrix((np.concatenate(vv), (np.concatenate(ri), np.concatenate(ci))),
                      shape=(n6, len(indep))).tocsr()
    return T, indep, fixed, dependent, master_of


class EnergyModel:
    """Energy lift of in-plane fields (all CPU threads for the solves).

    Exact constraints (T_exact: every MPC row, every BC) check the exported modes. For the
    lift the in-plane translations of ALL nodes are prescribed (the GDLC variables, P) and
    the remaining DOFs F (axial translation, rotations) follow by minimum strain energy, with
    the MPC rows of the axial translation and the rotations kept (axial slip and rotation
    continuity at the bolts = the composite action) and the in-plane MPC rows relaxed: a
    class component need not be compatible with the bolts in its own plane (C, the relative
    rigid motion of the pieces, is not). Overwriting its in-plane values at the bolt nodes
    instead would create artificial kinks and spurious energy. The in-plane incompatibility
    of the split is reported separately (bolt_incompatibility)."""

    def __init__(self, K, xyz, key_row, pairs, boundaries, inplane_dofs, threads: int):
        import scipy.sparse.linalg as sla
        t0 = time.time()
        self.threads = max(1, int(threads))
        self.K = K.tocsr()
        self.n6 = K.shape[0]
        self.inplane = sorted(int(d) for d in inplane_dofs)
        self.T_exact, self.indep_exact, self.fixed, _, self.master_of = constraint_transform(
            xyz, key_row, pairs, boundaries)
        self.T, self.indep, _, self.dependent, _ = constraint_transform(xyz, key_row, pairs, boundaries,
                                                                       relax_dofs=self.inplane)
        Kq = (self.T.T @ self.K @ self.T).tocsr()
        Kq.eliminate_zeros()
        dof_of = self.indep % 6 + 1
        isP = np.isin(dof_of, self.inplane)
        diag = Kq.diagonal()
        if np.any(diag < 0):
            raise ValueError("negative elastic diagonal: minimum-energy completion is not justified")
        # Do not compare translational and rotational stiffness diagonals: they
        # have different units. Only structurally empty rows are eliminated.
        zero = (~isP) & (np.diff(Kq.indptr) == 0)
        self.P = np.nonzero(isP)[0]
        self.F = np.nonzero(~isP & ~zero)[0]
        self.n_zero_stiffness = int(zero.sum())
        self.Kq = Kq
        self.KPP = Kq[self.P][:, self.P].tocsr()
        self.KPF = Kq[self.P][:, self.F].tocsr()
        self.KFP = self.KPF.T.tocsr()
        KFF = Kq[self.F][:, self.F].tocsc()
        self.lu = sla.splu(KFF, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                           options=dict(SymmetricMode=True))
        self.KFF = KFF
        gdof = self.indep[self.P]                           # P DOF -> (node row, in-plane component)
        self.P_row = gdof // 6
        self.P_comp = np.searchsorted(np.asarray(self.inplane), gdof % 6 + 1)
        self.fill = int(self.lu.L.nnz + self.lu.U.nnz)
        # bolts: (master row, dependent row, unit in-plane direction of the bolt)
        self.bolts = []
        for rb, (ra, kind) in self.master_of.items():
            d = np.array([xyz[rb][k - 1] - xyz[ra][k - 1] for k in self.inplane])
            nd = float(np.linalg.norm(d))
            self.bolts.append((ra, rb, d / nd if nd > 0 else None))
        self.setup_s = time.time() - t0

    def solve(self, B: np.ndarray) -> np.ndarray:
        """K_FF^-1 B, columns split over all threads (SuperLU releases the GIL)."""
        B = np.asarray(B, dtype=np.float64)
        if B.ndim == 1 or B.shape[1] <= 1 or self.threads == 1:
            return self.lu.solve(B)
        idx = np.array_split(np.arange(B.shape[1]), min(self.threads, B.shape[1]))
        out = np.empty_like(B)
        try:
            from threadpoolctl import threadpool_limits
            inner = threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            inner = nullcontext()
        with inner, ThreadPoolExecutor(max_workers=len(idx)) as ex:
            for cols, X in zip(idx, ex.map(lambda c: self.lu.solve(np.asfortranarray(B[:, c])), idx)):
                out[:, cols] = X
        return out

    def constraint_error(self, v: np.ndarray) -> np.ndarray:
        """||T q - v|| / ||v|| per column with the EXACT constraints: 0 when the vectors satisfy
        every MPC and support of the .inp (checks the modes and the constraint mapping)."""
        rec = self.T_exact @ v[self.indep_exact]
        return np.linalg.norm(rec - v, axis=0) / np.maximum(np.linalg.norm(v, axis=0), 1e-300)

    def energy(self, v: np.ndarray) -> np.ndarray:
        """v^T K v per column (exact strain energy x 2 of full vectors; no constraint needed)."""
        return np.einsum("im,im->m", v, self.K @ v)

    def inplane_P(self, w: np.ndarray) -> np.ndarray:
        """In-plane fields w (N, 2, m) -> values on P (|P|, m)."""
        return w[self.P_row, self.P_comp, :]

    def lift(self, WP: np.ndarray):
        """Minimum-energy completion of prescribed P values: returns (Z = q_F, S W, residual)."""
        rhs = -(self.KFP @ WP)
        Z = self.solve(rhs)
        res = float(np.linalg.norm(self.KFF @ Z - rhs) / max(np.linalg.norm(rhs), 1e-300))
        SW = self.KPP @ WP + self.KPF @ Z
        return Z, SW, res

    def bolt_incompatibility(self, comps: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Relative motion along the bolt axis that the class components carry individually
        (the bolts forbid it; the components of a compatible mode cancel it in their sum), as
        RMS per bolt and class relative to the RMS in-plane nodal displacement of the mode:
        sqrt(mean_{f,b} ((w_f(B) - w_f(A)) . d_b)^2) / sqrt(mean_nodes |w|^2)."""
        bl = [(a, b, d) for a, b, d in self.bolts if d is not None]
        if not bl:
            return np.zeros(w.shape[-1])
        A = np.array([a for a, _, _ in bl]); B = np.array([b for _, b, _ in bl])
        D = np.array([d for _, _, d in bl])                                   # (nb, 2)
        rel = np.einsum("kbcm,bc->kbm", comps[:, B] - comps[:, A], D)          # (5, nb, m)
        num = (rel ** 2).mean((0, 1))
        den = (w ** 2).sum(1).mean(0)
        return np.sqrt(num / np.maximum(den, 1e-300))


def component_energies(model: EnergyModel, comps: np.ndarray, v_modes: np.ndarray):
    """comps: (5, N, 2, m) GDLC in-plane components of m modes (global in-plane DOF order);
    v_modes: (6N, m) the modes themselves (U and UR). Returns dict with
      E  (5m x 5m)  w_f,i^T S w_g,j for all class/mode pairs of the batch (class-major),
      U  (m,)       v^T K v of each mode,
      bolt_incompat (m,), lift_residual."""
    nc, N, _, m = comps.shape
    W = np.concatenate([model.inplane_P(comps[k]) for k in range(nc)], axis=1)        # (|P|, 5m)
    Z, SW, res = model.lift(W)
    E = W.T @ SW
    E = 0.5 * (E + E.T)
    w = comps.sum(0)
    return dict(E=E, U=model.energy(v_modes), bolt_incompat=model.bolt_incompatibility(comps, w),
                lift_residual=res)


def k_orthogonality(model: EnergyModel, v_modes: np.ndarray, groups: list[list[int]], block_size=None) -> dict:
    """Largest |phi_i^T K phi_j| / sqrt(...) over mode pairs in DIFFERENT eigenvalue groups
    (buckling modes of distinct eigenvalues are K-orthogonal): checks K, the node map, the
    constraints and the exported mode shapes together."""
    nm = v_modes.shape[1]
    block_size = nm if block_size is None else max(1, int(block_size))
    G = np.empty((nm, nm))
    for start in range(0, nm, block_size):
        stop = min(nm, start+block_size)
        G[:, start:stop] = v_modes.T @ (model.K @ v_modes[:, start:stop])
    d = np.sqrt(np.maximum(np.diag(G), 1e-300))
    C = np.abs(G) / d[:, None] / d[None, :]
    same = np.zeros_like(C, bool)
    for g in groups:
        same[np.ix_(g, g)] = True
    C[same] = 0.0
    i, j = np.unravel_index(int(np.argmax(C)), C.shape)
    return dict(max_normalised_offdiagonal=float(C.max()), pair=[int(i), int(j)],
                median_row_max=float(np.median(C.max(1))) if len(C) else 0.0)


def validate_energy_result(E, U, lift_residual, orthogonality, orth_tol=1e-2, lift_tol=1e-8):
    """Necessary numerical acceptance checks; these do not prove physical family uniqueness."""
    E, U = np.asarray(E), np.asarray(U)
    coupling = orthogonality["max_normalised_offdiagonal"]
    if not np.isfinite(coupling) or coupling > orth_tol:
        raise ValueError("elastic matrix/modes fail K-orthogonality acceptance")
    if not np.isfinite(lift_residual) or lift_residual > lift_tol:
        raise ValueError("minimum-energy solve residual exceeds acceptance tolerance")
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(U)) or np.any(U <= 0):
        raise ValueError("nonfinite or nonpositive full-mode energy")
    diag = np.diagonal(E, axis1=1, axis2=2)
    scale = np.maximum(np.max(np.abs(E), axis=(1, 2)), U)
    tol = 1e-8 * scale
    total = E.sum(axis=(1, 2))
    # Eigenvalues of each tiny (5x5) Gram matrix also catch indefinite lifts.
    if (np.any(diag < -tol[:, None]) or np.any(np.linalg.eigvalsh(E) < -tol[:, None])
            or np.any(total < -tol) or np.any(total > U + tol)):
        raise ValueError("minimum-energy balance/positive-semidefiniteness check failed")
    return dict(passed=True, orthogonality_tolerance=orth_tol, lift_residual_tolerance=lift_tol,
                energy_relative_tolerance=1e-8)
