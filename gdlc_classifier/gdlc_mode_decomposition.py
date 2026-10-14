#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
gdlc_mode_decomposition.py
==========================

Quantitative G / D / L / C / O decomposition of Abaqus linear-buckling
eigenmodes of prismatic thin-walled members (single or built-up, with discrete
connections), computed slice by slice and integrated along the member.

Usage (Windows example)
-----------------------
    python gdlc_mode_decomposition.py "D:\CFS-Column"               # every run below this folder
    python gdlc_mode_decomposition.py "D:\CFS-Column\<run folder>"   # one run
    python gdlc_mode_decomposition.py <run> --selftest-only          # basis + self-test, no modes read
    python gdlc_mode_decomposition.py <run> --stiffness never        # kinematic split only, no Abaqus matrix job
    python gdlc_mode_decomposition.py "D:\CFS-Column" --make-reference-models
        # writes <run>_REF_G (in-plane rigid sections -> pure G eigenvalue/shape) and
        # <run>_REF_L (fold virtual corners held -> local reference) Abaqus inputs next to every run

Integrated pipeline (one command): for every run folder below the given paths
  1. (--solve-reference-models) solve <run>_REF_G / _REF_L folders whose .odb is missing,
     missing with an unused solver destination (solver chosen from supported model features);
  2. export the CURRENT <job>.odb with abaqus_mfsm_export.py (called through
     'abaqus python', same folder as this script) unless an export whose recorded
     sha256 matches the .odb and .inp already exists (--export auto|always|never);
  3. (--stiffness auto, default) once per run and .inp: write <run>/stiffness_export/<job>_KE.inp
     (the run's model data + one *MATRIX GENERATE step: raw elastic stiffness, MPC=NO, no
     boundary condition), solve it with Abaqus, read and check the matrix (gdlc_stiffness.py);
  4. classify; results go to <run>/mode_decomposition_GDLC/. With the stiffness, every class
     component is also evaluated in strain energy (E_*_pct columns, mode_energy.npz).
A run folder is a folder holding a modal export (*/modal_export.json), or a buckling
pipeline folder: one <job>.inp with a *Buckle step + <job>_build.json (or
gdlc_reference.json), solved (<job>.odb) or built only. --make-reference-models needs only
the .inp (geometry from the modal export when one exists, else from the .inp itself);
classification skips an unsolved run until its .odb exists. Both export formats are read:
npz (abaqus_mfsm_export.py: raw_dof_map.npz + modes_####.npz; element connectivity taken
from the matching .inp, checked by sha256) and the legacy parquet export.
Load case: compression and bending runs (abaqus_complete_model_m20.py --load-case) are
handled alike; the decomposition is kinematic and load independent. For bending the
eigenvalue is the critical extreme-fibre stress, M_cr_kNm = eigenvalue * M_ref is added,
and modes with eigenvalue <= 0 (reversed load) are excluded.

Method (what each number means)
-------------------------------
1. Slices. Nodes are grouped into cross-section rings along the member axis
   (axis found from the shell-element normals). Each ring's in-plane nodal
   translations form a vector phi_r (2 DOF per node). Rotations are not used.

2. Reference points (main nodes), detected automatically on every ring:
   * folds   - contiguous nodes whose discrete radius of curvature
               R_i = mean adjacent segment length / |turning angle| is below
               fold_radius_factor * t, with total turning >= min_fold_angle_deg.
               The arc is grown over its transition nodes (>= 25 % of peak turning).
               Each fold arc is treated as a RIGID corner (Beregszaszi & Adany 2019,
               cFSM with rigid corner element for rounded corners); its driver is the
               translation of the virtual sharp corner (intersection of the two wall
               tangents), so the D/G driver count equals the sharp-corner count.
               Verified counts (in-plane D): lipped channel 2, plain channel 0,
               square tube 1 - identical to cFSM/GBT, with or without rounded corners.
               For curved (non-straight) walls, D also contains patterns in which the
               folds approach each other by bending of the curved wall.
   * junctions - nodes joining >= 3 wall segments (branched sections).
   * free ends - nodes with one wall segment (not drivers; they follow).
   * seams   - node pairs joined by *MPC (BEAM/TIE/PIN/LINK) or connector elements.
               They are used for diagnostics only (seam jump profiles); they do not
               enter any class definition (see C below).
   * pieces  - connected parts of the ring's wall graph (the separate members of a
               built-up section).

3. Classes (per ring, exact direct sum of the 2n-dimensional in-plane space).
   Every class is defined by the kinematics of the WALL folds (cFSM criteria); the
   connectors are not used, because they act only at discrete fastener rows and the
   pieces are free between them.
   I  = inextensional fields: no change of segment length (no transverse
        extension, cFSM/GBT criterion) and rigid fold arcs.
   L  = fields in I with every fold/junction (virtual corner) translation = 0
        (plate-type bending with folds held; cFSM "local" criterion).
   F  = fields in I with minimum transverse bending energy for given fold
        translations (= cFSM transverse-equilibrium criterion for D/G). F is
        energy-orthogonal to L and contains every rigid motion of every piece.
   G  = rigid in-plane motion of the whole section (3 vectors).
   C  = relative rigid in-plane motion of the pieces of a built-up section: every
        piece moves as a rigid body, but not together with the others
        (3 x n_pieces - 3 vectors, W-orthogonal to G; empty for a one-piece section).
        This is the cross-section distortion of the ASSEMBLY that happens through
        the connections (e.g. corner pieces rotating about bolted seams like a
        linkage); its stiffness comes from the connectors and the longitudinal
        bending of the pieces. It is a physical family, not a residual.
   D  = F without G and C (W-orthogonal to both): distortion of the pieces' own
        cross-sections (fold translations relative to the piece's rigid motion).
        D + C = distortion of the assembled section (reported as D_assembly).
   O  = complement of I (transverse extension, corner-arc deformation, mesh noise),
        excluding fold-driver motions attainable in I and any fixed-fold rigid
        motions already assigned to F. O can carry incompatible fold translations.
        Large O indicates that the idealized inextensional families omit important
        content, which can be physical rather than numerical error.
   R^{2n} = G + C + D + L + O exactly, so every slice has ONE decomposition;
   there is no fitting residual once the geometry-dependent class spaces are fixed.
   Their fold thresholds and transverse bending proxy remain modeling choices.
   (Up to v1.4.0 the seams entered L/F as rigid-link constraints at EVERY ring and C
   was the seam-jump space. Lips of bolted pieces that buckle locally between the
   fasteners have such jumps, so local modes were reported as up to 80 % C, or as
   G+D before v1.4.0.)

4. Percentages. W = tributary mid-line length per node; dz = tributary length
   per ring. For each mode: E_M = sum_r dz_r * ||phi_{M,r}||_W^2 ,
   share_M = E_M / sum_N E_N  (share of the cross-sectional deformation, integrated
   along the member; NOT a share of strain energy or of the critical load).
   G, C and D are mutually W-orthogonal and O is W-orthogonal to L; L is
   energy-orthogonal to G+C+D (the cFSM L/D criterion) but not W-orthogonal, so the
   cross term  1 - sum_N E_N / ||phi||^2  is reported (it is the overlap of the L
   component with the others; zero for a pure mode).
   A second, exactly additive measure is also reported (no normalisation):
   signed_M = sum_r dz_r <phi_M,r , phi_r>_W / ||phi||^2   (sum over M = 100 % exactly;
   a negative value means that component opposes the total deformation).
   Both measures come from the same unique decomposition; they differ only in how
   the cross terms are allotted.

4b. Strain energy (when <run>/stiffness_export exists; gdlc_stiffness.py). The kinematic
   classes above are not changed; each class component of a mode (its in-plane field) is
   completed to a full displacement field by MINIMUM STRAIN ENERGY with the real elastic
   stiffness K of the shell model (axial translation and rotations free, supports and the
   axial/rotational bolt MPC rows kept), E_fg = w_f^T S w_g. Unlike the transverse-bending
   proxy used to build D, this includes longitudinal bending, warping, shear and the
   composite action through the bolts (a rigid-section G field stores the real flexural /
   torsional energy). Reported: E_<X>_pct (E_ff / sum E_gg), E_<X>_signed_pct (sum_g E_fg /
   sum E, exactly additive), E_cross_term_pct, E_rel_* (group result, basis-invariant),
   E_beyond_section_pct (energy of the mode v^T K v - with its own U3 and UR - not implied by
   its in-plane field), bolt_incompatibility_pct (relative motion along the bolt axes that the
   components carry individually; it cancels in the sum). Checks before any number is used:
   rigid-body null space of K, every exported mode satisfies every MPC/support of the .inp,
   K-orthogonality of modes with distinct eigenvalues.

5. Reliability checks (conventions; calibrate on benchmarks):
   * O share > max_other_pct           -> flag UNRELIABLE_OTHER
   * dominant class differs between the normalised and the signed measure
                                       -> flag AMBIGUOUS_DOMINANT
   * eigenvalue groups whose total spread (max-min)/min <= cluster_rel_tol (span
     criterion, no chaining) -> also reported as a group; group shares are invariant
     to how Abaqus picked vectors inside the group.
   * |cross term| > max_cross_pct (default 100 %) -> flag CROSS_TERM, confidence at most
     medium. The cFSM D shapes (minimum transverse bending energy) overlap strongly with
     L in the displacement norm (a pure cFSM distortional mode of a lipped channel has
     about 40-55 % of its W-norm inside span(L)), so cross terms of several 10 % are
     normal for L-D interacting modes and do not by themselves make a split unreliable;
     an overlap larger than the mode itself does.
   * C share on the fastener rings (C_at_bolt_rings_pct): relative rigid motion of the
     pieces where they are bolted - near 0 when the fasteners enforce composite action.
   * RELIABLE result per mode (rel_*_pct, reliable_label, reliable_dominant): the cluster
     result when the mode belongs to a near-degenerate group, its own result otherwise.
     All statistics (dominant mechanism, mechanism_critical_stresses) use it.
   * split sensitivity: the decomposition is repeated with the fold-arc rotation taken from
     either side of every rounded arc (both admissible measures of a rigid arc's rotation;
     FE arcs are not exactly rigid). split_spread_pct = largest change of a reliable share,
     rel_<X>_min/max_pct = the band. > max_spread_pct (10) -> DEFINITION_SENSITIVE;
     dominant class changes -> DOMINANT_UNSTABLE.
   * fold_motion_ratio = RMS virtual-corner translation / RMS mid-line displacement
     (rigid translation 1, pure local 0): independent kinematic check of G+C+D content.
   * halfwave_<X>_mm = member length / number of humps of the class's slice profile: the
     classes are cross-sectional, so a G or C component at the local half-wavelength is a
     section motion that follows the local buckles, not member-level global buckling.
   * confidence per mode: low = UNRELIABLE_OTHER, AMBIGUOUS_DOMINANT, DOMINANT_UNSTABLE or
     margin < 5 % (LOW_MARGIN); medium = margin < 20 % (MIXED), CROSS_TERM,
     DEFINITION_SENSITIVE or ENERGY_DISAGREES (the energy dominant class - resolved, i.e. its
     normalised and signed energy measures agree - differs from the kinematic one); high
     otherwise (margin = gap between the first and second
     mechanism, smaller of the two measures). Thresholds are conventions. Flags and
     confidence always refer to the reliable (cluster or mode) result.
   * Built-in self-test: synthetic fields with known components must be recovered
     to machine precision before any mode is processed.
   * Benchmarks (--benchmarks, also run before every classification): cFSM D counts of
     classic sections; a two-piece built-up section (whole rigid motion -> 100 % G,
     opposite rigid motion of the pieces -> 100 % C, any field with fixed folds ->
     exactly 0 % G+C+D); share convergence of one smooth field on a 4x refined mesh.

References (method sources)
---------------------------
Adany & Schafer (2006, 2008) cFSM; Adany & Schafer (2014) generalized cFSM;
Li, Joo, Adany & Schafer (2011/2013) modal identification of shell FE models;
Cai & Moen (2015, 2016) section-wise GBT identification of FE modes;
Beregszaszi & Adany (2019) cFSM with rigid corner elements (rounded corners).
The C class (relative rigid motion of the pieces of a built-up section) is an extension
of the cFSM G/D split to multi-piece sections proposed in this project.

Resource use: available logical CPUs for useful BLAS/I/O work, live-memory chunks and temporary maps,
optional CUDA (CuPy) for the projections, process pool for plots.
"""
from __future__ import annotations

import os
import sys

# The same stdlib-only policy configures the solver and postprocessor.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runtime_resources as resources


# ----------------------------------------------------------------------------
# 0. Resource set-up (must run before numpy is imported)
# ----------------------------------------------------------------------------
def _requested_threads() -> int:
    n = resources.logical_cpu_count()
    for i, a in enumerate(sys.argv):
        if a.startswith('--threads='):
            try:
                n = max(1, int(a.split('=', 1)[1]))
            except ValueError:
                pass
        if a == "--threads" and i + 1 < len(sys.argv):
            try:
                n = max(1, int(sys.argv[i + 1]))
            except ValueError:
                pass
    return n


N_THREADS = _requested_threads()
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "VECLIB_MAXIMUM_THREADS",
           "BLIS_NUM_THREADS"):
    os.environ[_k] = str(N_THREADS)          # deliberately overrides: use every core

import argparse
import concurrent.futures as cf
import datetime as _dt
import hashlib
import json
import math
import re
import subprocess
import time
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    pa.set_cpu_count(N_THREADS)
    pa.set_io_thread_count(max(N_THREADS, 8))
except Exception:                             # pragma: no cover
    pq = None

try:
    import psutil
except Exception:                             # pragma: no cover
    psutil = None

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(N_THREADS)
except Exception:
    pass

try:                                          # package import (tests) or script folder
    from . import gdlc_stiffness as gst
except ImportError:
    import gdlc_stiffness as gst

CLASSES = ("G", "D", "L", "C", "O")
CLASS_NAMES = {"G": "G global (rigid section)", "D": "D distortional (pieces' sections)",
               "L": "L local (folds fixed)", "C": "C relative rigid motion of the pieces",
               "O": "O other (extension, arc deformation)"}
VERSION = "1.10.1"


# ----------------------------------------------------------------------------
# 1. Settings
# ----------------------------------------------------------------------------
@dataclass
class Settings:
    fold_radius_factor: float = 15.0      # fold if R_i < factor * t
    min_fold_angle_deg: float = 30.0      # min total turning of a fold arc
    cluster_rel_tol: float = 0.001        # group if (max-min)/min <= tol (span, no chaining)
    max_other_pct: float = 5.0
    max_cross_pct: float = 100.0          # |cross term| above this -> CROSS_TERM flag, confidence <= medium
    max_spread_pct: float = 10.0          # split spread over corner-kinematics variants above this -> DEFINITION_SENSITIVE
    max_energy_beyond_pct: float = 10.0  # diagnostic screening policy, not a code/standard limit
    max_bolt_incompat_pct: float = 5.0   # diagnostic screening policy, not a connection-strength limit
    sensitivity: bool = True              # compute the split spread (2 extra projections)
    conf_low_margin_pct: float = 5.0      # margin below this -> confidence "low"
    conf_high_margin_pct: float = 20.0    # margin at/above this (and no flag) -> "high"
    mixed_label_pct: float = 20.0         # classes >= this appear in the label (e.g. "L+D")
    thickness: float | None = None        # override t (mm)
    export_dir: str | None = None
    inp: str | None = None
    out_name: str = "mode_decomposition_GDLC"
    gpu: str = "auto"                     # auto | on | off
    plots: str = "all"                    # all | summary | none
    selftest_only: bool = False
    max_modes: int | None = None          # for quick checks only
    threads: int = N_THREADS
    top_modes: int = 3                    # figures: top-k modes of the dominant mechanism
    export: str = "auto"                  # auto | always | never  (ODB -> modal export)
    abaqus_cmd: str = "abaqus"
    abaqus_cpus: int | str | None = None  # cpus= of the Abaqus solves (auto/None: physical cores)
    exporter: str | None = None           # default: abaqus_mfsm_export.py next to this script
    export_compression: str = "store"     # store (fast, larger) | zlib
    stiffness: str = "auto"               # auto | always | required | never (elastic K export + energy split)
    ref_g_twist: str = "auto"             # REF_G: auto | restrained | free  (section rotation about the axis)


def log(msg: str) -> None:
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _available_ram() -> int:
    return resources.available_memory()[1]


def _mode_array(shape):
    """Use anonymous disk backing when a modal array does not fit live RAM.

    The temporary backing file is outside the run/export folders and is owned
    by the array; original simulation artifacts are never used as scratch.
    """
    nbytes = math.prod(shape) * 8
    if nbytes <= _available_ram():
        try:
            return np.full(shape, np.nan, dtype=np.float64)
        except MemoryError:
            pass
    backing = tempfile.TemporaryFile(prefix="gdlc_modes_")
    backing.truncate(nbytes)
    a = np.memmap(backing, dtype=np.float64, mode="r+", shape=shape)
    a._backing_file = backing
    a[:] = np.nan
    return a


def _process_pool(tasks):
    workers = resources.process_workers(tasks, N_THREADS)
    return cf.ProcessPoolExecutor(max_workers=workers, initializer=resources.initialise_worker,
                                  initargs=(max(1, N_THREADS // workers),))


# ----------------------------------------------------------------------------
# 2. Discovery of runs and inputs
# ----------------------------------------------------------------------------
EXPORT_DEFAULT_NAME = "modal_export_gdlc"
_SHA_CACHE: dict = {}


def file_sha256(path: Path) -> str:
    """sha256 of a file (cached per path/size/mtime); 8 MB blocks."""
    p = Path(path)
    st_ = p.stat()
    key = (str(p.resolve()), st_.st_size, st_.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(blk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def export_format(meta: dict) -> str:
    """'npz'  : abaqus_mfsm_export.py  (raw_dof_map.npz + modes_####.npz, shards = file names)
       'parquet': legacy portable exporter (mesh_nodes/elements/mode_shapes_####.parquet)."""
    sh = meta.get("shards") or []
    if sh and isinstance(sh[0], str):
        return "npz"
    return "parquet"


def _export_complete(exp_dir: Path, meta: dict) -> bool:
    shards = meta.get("shards") or []
    if not shards:
        return False
    if export_format(meta) == "npz":
        need = ["raw_dof_map.npz"] + list(shards)
    else:
        need = ["mesh_nodes.parquet", "elements.parquet"] + [sh.get("parquet_file", "") for sh in shards]
    return all((exp_dir / f).is_file() for f in need)


def list_exports(run_dir: Path) -> list[tuple[Path, dict]]:
    """Every complete modal export directly below run_dir (any folder name)."""
    out = []
    for mj in sorted(run_dir.glob("*/modal_export.json")):
        try:
            meta = json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("kind", "PORTABLE_RAW_MODAL_DATA_ONLY") != "PORTABLE_RAW_MODAL_DATA_ONLY":
            continue
        if _export_complete(mj.parent, meta):
            out.append((mj.parent, meta))
    return out


def run_sources(run_dir: Path) -> tuple[Path | None, Path | None]:
    """(odb, inp) of a run folder: the .odb whose stem has a matching .inp; None if absent
    or ambiguous."""
    odbs = sorted(run_dir.glob("*.odb"))
    pairs = [(o, o.with_suffix(".inp")) for o in odbs
             if o.with_suffix(".inp").is_file() and odb_status(o) == "ok"]
    if len(pairs) == 1:
        return pairs[0]
    inps = sorted(run_dir.glob("*.inp"))
    return (None, inps[0]) if (not pairs and len(inps) == 1) else (None, None)


def choose_export(run_dir: Path, forced: str | None) -> tuple[Path, dict]:
    """Best export of a run: it must belong to the run's current .odb/.inp when they exist
    (sha256 recorded by the exporter); among those the one with most modes, newest."""
    if forced:
        d = Path(forced)
        meta = json.loads((d / "modal_export.json").read_text(encoding="utf-8"))
        if not _export_complete(d, meta):
            raise FileNotFoundError(f"incomplete export: {d}")
        return d, meta
    odb, inp = run_sources(run_dir)
    odb_sha = file_sha256(odb) if odb else None
    inp_sha = file_sha256(inp) if inp else None
    best = None
    for d, meta in list_exports(run_dir):
        ok_odb = odb_sha is None or meta.get("source_odb_sha256") == odb_sha
        ok_inp = inp_sha is None or meta.get("source_inp_sha256") in (None, inp_sha)
        key = (ok_odb and ok_inp, int(meta.get("mode_count", 0)), (d / "modal_export.json").stat().st_mtime)
        if best is None or key > best[0]:
            best = (key, d, meta)
    if best is None:
        raise FileNotFoundError(f"No complete modal export (*/modal_export.json) in {run_dir}")
    if not best[0][0]:
        raise RuntimeError(f"export {best[1].name} does not match the current .odb/.inp "
                           "(sha256 differs); re-export with --export always")
    return best[1], best[2]


def _is_buckle_run(run_dir: Path) -> bool:
    """A run folder of the buckling pipeline: <job>.inp with a *Buckle step, and either the
    pipeline's <job>_build.json or a GDLC reference marker."""
    odb, inp = run_sources(run_dir)
    if inp is None:
        return False
    if not (any(run_dir.glob("*_build.json")) or (run_dir / "gdlc_reference.json").is_file()):
        return False
    try:
        with open(inp, encoding="utf-8", errors="replace") as f:
            return any(l.lstrip().lower().startswith("*buckle") for l in f)
    except OSError:
        return False


def find_runs(paths: list[str]) -> list[Path]:
    """Run folders below the given paths: folders that hold a complete modal export, and
    buckling-pipeline folders (<job>.inp with a *Buckle step + *_build.json or
    gdlc_reference.json), solved or not. A built but unsolved run (pipeline BUILD_ONLY: .inp,
    no .odb yet) is a run folder too: --make-reference-models needs only its .inp, and the
    classification skips it with a message until it is solved and exported. Unrelated .odb
    files (GMNIA, single-piece studies, ...) are not picked up."""
    cand = []
    for p in paths:
        p = Path(p)
        roots = [p] + [d for d in p.rglob("*") if d.is_dir()] if p.is_dir() else []
        for d in roots:
            if (d / "modal_export.json").is_file():
                continue                                  # an export folder itself
            if list_exports(d) or _is_buckle_run(d):
                cand.append(d)
    out, seen = [], set()
    for r in cand:
        k = str(r.resolve())
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def find_inp(run_dir: Path, meta: dict, forced: str | None) -> Path | None:
    """The .inp of the export: the file whose sha256 equals the one recorded by the exporter,
    else the recorded file name, else the single .inp of the folder."""
    if forced:
        return Path(forced)
    inps = sorted(run_dir.glob("*.inp"))
    want = meta.get("source_inp_sha256")
    if want:
        for f in inps:
            if file_sha256(f) == want:
                return f
    src = meta.get("source_inp")
    if src:
        cand = run_dir / Path(str(src).replace("\\", "/")).name
        if cand.is_file():
            if want:
                log(f"  WARNING: {cand.name} differs from the .inp the export was made from")
            return cand
    return inps[0] if len(inps) == 1 else None


# ----------------------------------------------------------------------------
# 2b. Integrated pipeline: Abaqus solve (reference models) and ODB export
# ----------------------------------------------------------------------------
def _run_cmd(cmd: list[str], cwd: Path) -> int:
    log("  $ " + subprocess.list2cmdline(cmd))
    if os.name == "nt":                                   # abaqus is a .bat launcher on Windows
        return subprocess.call(subprocess.list2cmdline(cmd), cwd=str(cwd), shell=True)
    return subprocess.call(cmd, cwd=str(cwd))


def abaqus_cpus(st: Settings) -> int:
    """cpus= of the Abaqus solves: --cpus, else all physical cores. Abaqus/Standard counts
    cores, not SMT threads, and refuses a larger request; ../runtime_resources.py (the
    pipeline's policy) clamps a request above the physical cores with a warning."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import runtime_resources
        return runtime_resources.resolve_solver_policy(runtime_resources.detect_resources(),
                                                       st.abaqus_cpus, 0).cpus
    except Exception as error:                       # pipeline module absent: no clamping
        if st.abaqus_cpus is None:
            log(f"  cpu policy unavailable ({error}); using --threads = {N_THREADS} Abaqus cpus")
        return int(st.abaqus_cpus or N_THREADS)


def abaqus_resources(st: Settings) -> list[str]:
    policy = resources.resolve_solver_policy(resources.detect_resources(), st.abaqus_cpus, "auto")
    return [f"cpus={policy.cpus}", f"gpus={policy.gpus}", "memory=100%", "standard_parallel=all"]


def solve_reference_model(run_dir: Path, st: Settings) -> bool:
    """Solve a GDLC reference model (folder with gdlc_reference.json and no .odb yet)."""
    if not (run_dir / "gdlc_reference.json").is_file():
        return False
    inps = sorted(run_dir.glob("*.inp"))
    if len(inps) != 1:
        log(f"  cannot solve {run_dir.name}: expected one .inp, found {len(inps)}")
        return False
    job = inps[0].stem
    status = odb_status(run_dir / f"{job}.odb")
    if status in ("ok", "locked"):
        return False
    if status != "missing":
        raise RuntimeError(f"{job}.odb is {status}; preserve this run and generate a new reference directory")
    occupied = [p for ext in ('.dat', '.msg', '.sta', '.log', '.sim', '.res', '.lck', '.com', '.prt', '.fil')
                if (p := run_dir / (job+ext)).exists()]
    if occupied:
        raise RuntimeError(f"existing reference solver artifacts are preserved: {occupied}")
    cmd = [st.abaqus_cmd, f"job={job}", f"input={inps[0].name}"] + abaqus_resources(st) + [
           "ask_delete=OFF", "interactive"]
    t0 = time.time()
    rc = _run_cmd(cmd, run_dir)
    ok = rc == 0 and odb_status(run_dir / f"{job}.odb") == "ok"
    log(f"  reference model {job}: {'solved' if ok else 'SOLVE FAILED (see the .msg/.dat files)'} "
        f"in {time.time() - t0:.0f} s")
    return ok


def ensure_export(run_dir: Path, st: Settings) -> Path | None:
    """Make sure the run has an export of its CURRENT .odb (sha256 match); otherwise call
    abaqus_mfsm_export.py through 'abaqus python'. Returns the export folder or None."""
    odb, inp = run_sources(run_dir)
    if odb is None or inp is None:
        for o in sorted(run_dir.glob("*.odb")):
            stt = odb_status(o)
            if stt != "ok":
                log(f"  {o.name}: analysis {stt} (see {o.stem}.dat / .msg) - not exported")
        return None
    sha = file_sha256(odb)
    inp_sha = file_sha256(inp)
    if st.export == "auto":
        for d, meta in list_exports(run_dir):
            if meta.get("source_odb_sha256") == sha and meta.get("source_inp_sha256") in (None, inp_sha):
                return d
    exporter = Path(st.exporter) if st.exporter else Path(__file__).resolve().parent / "abaqus_mfsm_export.py"
    if not exporter.is_file():
        raise FileNotFoundError(f"exporter not found: {exporter}")
    out = run_dir / EXPORT_DEFAULT_NAME
    k = 2
    while out.exists():                                    # the exporter never overwrites
        out = run_dir / f"{EXPORT_DEFAULT_NAME}_{k}"
        k += 1
    t0 = time.time()
    rc = _run_cmd([st.abaqus_cmd, "python", str(exporter), "--odb", str(odb), "--inp", str(inp),
                   "--output-dir", str(out), "--compression", st.export_compression], exporter.parent)
    if rc != 0 or not (out / "modal_export.json").is_file():
        raise RuntimeError(f"export failed (exit code {rc}) for {odb}")
    meta = json.loads((out / "modal_export.json").read_text(encoding="utf-8"))
    if meta.get("source_odb_sha256") != sha or not _export_complete(out, meta):
        raise RuntimeError(f"export {out} is incomplete or does not match {odb.name}")
    log(f"  exported {meta.get('mode_count')} modes in {time.time()-t0:.1f} s -> {out.name}")
    return out


def ensure_stiffness(run_dir: Path, st: Settings) -> dict | None:
    """Make sure the run has the raw elastic stiffness of its CURRENT .inp
    (<run>/stiffness_export, see gdlc_stiffness.py): otherwise write the matrix-generation
    job, run it with Abaqus, read the .mtx files and check the matrix (symmetry, rigid-body
    null space). Reference models (gdlc_reference.json) are skipped. With --stiffness auto a
    failure is logged and the classification continues without the energy evaluation;
    --stiffness required stops."""
    if st.stiffness == "never" or (run_dir / "gdlc_reference.json").is_file():
        return None
    try:
        exp_dir, emeta = choose_export(run_dir, st.export_dir)
        inp = find_inp(run_dir, emeta, st.inp)
        if inp is None:
            raise FileNotFoundError("no unique <job>.inp in the run folder")
        inp_sha = file_sha256(inp)
        if st.stiffness != "always":
            meta = gst.stiffness_status(run_dir, inp_sha)
            if meta:
                return meta
        import scipy.sparse  # noqa: F401  (needed to assemble and factorise K)
        mesh = load_mesh(exp_dir, emeta, inp)
        t0 = time.time()
        recovered = gst.completed_matrix_job(run_dir, inp) if st.stiffness != 'always' else None
        if recovered:
            folder, job = recovered
            log(f'  reusing completed matrix job: {folder.name}/{job} (input and node codes match)')
        else:
            folder, job = gst.write_matrix_job(run_dir, inp)
            rc = _run_cmd([st.abaqus_cmd, f"job={job}", f"input={job}.inp"] + abaqus_resources(st) + [
                           "ask_delete=OFF", "interactive"], folder)
            if rc != 0 or not any(folder.glob(f"{job}_STIF*.mtx")):
                raise RuntimeError(f"Abaqus matrix generation failed (exit code {rc}); see "
                                   f"{folder / (job + '.dat')} and .msg. Contact/model data were not removed.")
        nodes = gst.instance_nodes(inp.read_text(encoding="utf-8", errors="replace").splitlines())
        xyz_of = {(i, int(l)): mesh.xyz[k] for k, (i, l) in enumerate(zip(mesh.instance, mesh.label))}
        info = gst.ingest_matrix_job(folder, job, nodes, xyz_of)
        inactive = info['matrix'].get('inactive_internal_entries_removed', 0)
        if inactive:
            log(f"  ignored {inactive} exactly-zero internal matrix entries at "
                f"{info['matrix']['inactive_internal_nodes_removed']} internal nodes; no nonzero internal coupling discarded")
        meta = dict(format_version=gst.FORMAT_VERSION, program=f"gdlc_mode_decomposition.py v{VERSION}",
                    source_inp=inp.name, source_inp_sha256=inp_sha, job=job,
                    created_at=_dt.datetime.now().isoformat(timespec="seconds"),
                    solve_and_read_s=time.time() - t0, contact_removed=False, matrix_job_reused=bool(recovered),
                    content="raw assembled elastic stiffness of the elements: MPC=NO, no boundary "
                            "condition (applied from the .inp by the energy evaluation); base state = "
                            "initial state of the buckling step (no initial-stress term)",
                    **info)
        (folder / gst.EXPORT_JSON).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"  stiffness: {info['n_dofs']} DOF, nnz {info['matrix']['nnz']}, node map: {info['node_map']}, "
            f"rigid-body residual {max(info['rigid_body_residual']):.1e} -> {folder.name}")
        return meta
    except Exception as e:
        if st.stiffness == "required":
            raise
        log(f"  WARNING: no elastic stiffness for {run_dir.name} ({str(e).splitlines()[0][:200]}); "
            "the energy evaluation is skipped")
        return None


# ----------------------------------------------------------------------------
# 3. Minimal Abaqus .inp reader (thickness, assembly node sets, connections)
# ----------------------------------------------------------------------------
@dataclass
class InpInfo:
    thicknesses: list = field(default_factory=list)
    pairs: list = field(default_factory=list)          # (kind, (instA,labelA), (instB,labelB))
    notes: list = field(default_factory=list)
    nsets: dict = field(default_factory=dict)          # assembly node sets -> [(inst,label)]
    mpc_dependent: set = field(default_factory=set)    # nodes eliminated by *MPC
    boundaries: list = field(default_factory=list)     # ([(inst,label)], set(dofs))


def _kw_params(line: str) -> tuple[str, dict]:
    parts = [p.strip() for p in line[1:].split(",")]
    kw = parts[0].lower()
    prm = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            prm[k.strip().lower()] = v.strip()
        elif p:
            prm[p.strip().lower()] = True
    return kw, prm


def read_inp(path: Path) -> InpInfo:
    info = InpInfo()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nsets: dict[str, list] = {}
    in_assembly = False
    cur = None              # (kind, data)
    pending_section = False
    conn_elsets_rigid: dict[str, bool] = {}
    conn_elements: dict[str, list] = {}
    elset_members: dict[str, list] = {}
    cur_elset = None
    step_index, buckle_step = -1, None
    bc_records = []

    def ref_nodes(tok: str, default_inst: str | None = None):
        tok = tok.strip()
        if not tok:
            return []
        tok = tok.upper()
        if tok in nsets:
            return nsets[tok]
        if "." in tok:
            inst, rest = tok.split(".", 1)
            if rest.isdigit():
                return [(inst, int(rest))]
            if rest in nsets:
                return nsets[rest]
        if tok.isdigit():
            return [(default_inst, int(tok))]
        return []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            kw, prm = _kw_params(line)
            cur = None
            pending_section = False
            cur_elset = None
            if kw == "step":
                step_index += 1
            elif kw == "buckle":
                if buckle_step is not None:
                    raise ValueError("multiple buckling steps require an explicit step selection")
                buckle_step = step_index
            elif kw == "assembly":
                in_assembly = True
            elif kw == "end assembly":
                in_assembly = False
            elif kw == "shell section":
                if "composite" in prm:
                    raise ValueError("GDLC requires a homogeneous shell section, not a composite layup")
                pending_section = True
            elif kw == "nset" and in_assembly:
                name = str(prm.get("nset")).upper()
                nsets.setdefault(name, [])
                inst_ = prm.get("instance")
                cur = ("nset", name, str(inst_).upper() if inst_ else None, "generate" in prm)
            elif kw == "elset" and in_assembly:
                cur_elset = prm.get("elset")
                elset_members.setdefault(cur_elset, [])
                cur = ("elset", cur_elset, "generate" in prm)
            elif kw == "mpc":
                cur = ("mpc",)
            elif kw == "boundary":
                cur = ("bc", step_index, prm)
            elif kw == "element" and in_assembly and str(prm.get("type", "")).upper().startswith("CONN"):
                cur = ("conn", prm.get("elset"))
            elif kw == "connector section":
                es = prm.get("elset")
                cur = ("connsec", es)
            continue
        if pending_section:
            try:
                info.thicknesses.append(float(line.split(",")[0]))
            except ValueError:
                pass
            pending_section = False
            continue
        if cur is None:
            continue
        toks = [t.strip() for t in line.split(",") if t.strip()]
        if cur[0] == "nset":
            _, name, inst, gen = cur
            if gen and len(toks) >= 2:
                a, b = int(toks[0]), int(toks[1])
                s = int(toks[2]) if len(toks) > 2 else 1
                nsets[name] += [(inst, k) for k in range(a, b + 1, s)]
            else:
                for t in toks:
                    if t.isdigit():
                        nsets[name].append((inst, int(t)))
                    elif t.upper() in nsets:
                        nsets[name] += nsets[t.upper()]
        elif cur[0] == "elset":
            elset_members[cur[1]] += toks
        elif cur[0] == "mpc":
            kind = toks[0].upper()
            refs = [ref_nodes(t) for t in toks[1:]]
            if refs and refs[0]:
                info.mpc_dependent.update(refs[0])
            if kind in ("BEAM", "TIE", "PIN", "LINK") and len(refs) >= 2 and refs[0] and refs[1]:
                for b_, a_ in zip(refs[0], refs[1]):
                    info.pairs.append((kind, a_, b_))
        elif cur[0] == "bc":
            nodes = ref_nodes(toks[0])
            if nodes and len(toks) >= 2:
                t1 = toks[1].upper()
                if t1.isdigit():
                    a_ = int(t1)
                    b_ = int(toks[2]) if len(toks) > 2 and toks[2].isdigit() else a_
                    dofs = set(range(a_, b_ + 1))
                else:
                    types = {"PINNED": {1, 2, 3}, "ENCASTRE": {1, 2, 3, 4, 5, 6},
                             "XSYMM": {1, 5, 6}, "YSYMM": {2, 4, 6}, "ZSYMM": {3, 4, 5},
                             "XASYMM": {2, 3, 4}, "YASYMM": {1, 3, 5}, "ZASYMM": {1, 2, 6}}
                    if t1 not in types:
                        raise ValueError(f"Unsupported boundary type {t1}")
                    dofs = types[t1]
                bc_records.append((cur[1], cur[2], (nodes, dofs)))
        elif cur[0] == "conn":
            if len(toks) >= 3:
                na, nb = ref_nodes(toks[1]), ref_nodes(toks[2])
                if na and nb:
                    conn_elements.setdefault(cur[1] or "_", []).append((na[0], nb[0]))
        elif cur[0] == "connsec":
            conn_elsets_rigid[cur[1]] = any(k in line.upper() for k in ("BEAM", "WELD"))
            cur = None
    for es, prs in conn_elements.items():
        kind = "CONN_RIGID" if conn_elsets_rigid.get(es, False) else "CONN"
        for a_, b_ in prs:
            info.pairs.append((kind, a_, b_))
    if not info.pairs:
        info.notes.append("no MPC/connector pairs found: seam diagnostics unavailable; C depends on wall connectivity")
    if info.thicknesses and (not np.all(np.isfinite(info.thicknesses)) or min(info.thicknesses) <= 0
                            or not np.allclose(info.thicknesses, info.thicknesses[0], rtol=1e-8, atol=0)):
        raise ValueError("GDLC currently requires uniform positive finite shell thickness")
    info.nsets = nsets
    if buckle_step is None:
        info.boundaries = [bc for _, _, bc in bc_records]
    else:
        # LOAD CASE=1 defines the perturbation stress problem; LOAD CASE=2
        # defines the eigenvector constraints. OP=NEW applies to the whole
        # step/load case, not independently to each repeated *Boundary block.
        active = []
        for stage in sorted({s for s, _, _ in bc_records if s <= buckle_step}):
            has_case_two = any(s == stage and str(p.get("load case", "1")) == "2"
                               for s, p, _ in bc_records)
            selected = [(p, bc) for s, p, bc in bc_records if s == stage
                        and (stage != buckle_step or not has_case_two
                             or str(p.get("load case", "1")) == "2")]
            if any(str(p.get("op", "MOD")).upper() == "NEW" for p, _ in selected):
                active = []
            active.extend(bc for _, bc in selected)
        info.boundaries = active
    return info


# ----------------------------------------------------------------------------
# 4. Mesh -> rings -> ring graphs
# ----------------------------------------------------------------------------
@dataclass
class Mesh:
    node_index: np.ndarray       # (N,)
    instance: np.ndarray         # (N,) str
    label: np.ndarray            # (N,)
    xyz: np.ndarray              # (N,3)
    elems: np.ndarray            # (E,4) row positions into node arrays
    axis: np.ndarray             # (3,)
    e1: np.ndarray
    e2: np.ndarray
    s: np.ndarray                # axial coordinate
    p: np.ndarray                # (N,2) in-plane coordinates
    ring_of: np.ndarray          # (N,) ring id
    ring_z: np.ndarray           # (R,)
    size: float
    geometry: dict = field(default_factory=dict)


def _read_parquet(path: Path):
    if pq is None:
        raise RuntimeError("this export is in parquet format: pip install pyarrow")
    return pq.read_table(path, use_threads=True).to_pandas()


def inp_shell_elements(inp_path: Path) -> list:
    """[(INSTANCE, label, (n1, n2, n3, n4))] of every 4-node shell element of every part
    instance in the .inp (part-level *Element blocks; instance names upper-case)."""
    parts: dict = {}
    insts: dict = {}
    part = None
    cur = None
    with open(inp_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("**"):
                continue
            if line.startswith("*"):
                kw, prm = _kw_params(line)
                cur = None
                if kw == "part":
                    part = str(prm.get("name")).upper()
                    parts[part] = []
                elif kw == "end part":
                    part = None
                elif kw == "instance":
                    insts[str(prm.get("name")).upper()] = str(prm.get("part")).upper()
                elif kw == "element" and part is not None:
                    cur = str(prm.get("type", "")).upper()
                    if cur not in ("S4", "S4R", "S4R5", "S4RS", "S4RSW"):
                        raise RuntimeError(f"element type {cur}: only supported 4-node shells are allowed")
                continue
            if cur is not None:
                v = [int(x) for x in line.split(",") if x.strip()]
                if len(v) == 5:
                    parts[part].append((v[0], tuple(v[1:])))
                else:
                    raise RuntimeError(f"element type {cur} with {len(v)-1} nodes: only 4-node shells "
                                       f"(S4, S4R, ...) are supported")
    out = []
    for name, pname in insts.items():
        for lab, conn in parts.get(pname, []):
            out.append((name, lab, conn))
    if not out:
        raise RuntimeError(f"no 4-node shell elements found in {inp_path}")
    return out


def inp_assembly_nodes(inp_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(INSTANCE, label, xyz) of every part-instance node of the .inp in assembly coordinates
    (instance translation and rotation applied). Read with the exporter's own INP reader
    (abaqus_mfsm_export.read_input, next to this script), so the node order, the transforms
    and the instance names are exactly those an npz export of the solved run carries."""
    import importlib.util
    src = Path(__file__).resolve().parent / "abaqus_mfsm_export.py"
    spec = importlib.util.spec_from_file_location("abaqus_mfsm_export", src)
    if spec is None or not src.is_file():
        raise FileNotFoundError(f"INP reader not found: {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nodes = mod.read_input(inp_path)["nodes"]
    inst = np.char.upper(np.array([k[0] for k in nodes], dtype=str))
    lab = np.array([k[1] for k in nodes], dtype=np.int64)
    xyz = np.array([nodes[k] for k in nodes], dtype=np.float64).reshape(-1, 3)
    return inst, lab, xyz


def verified_input_coordinates(inst, lab, xyz, inp_path: Path):
    """Recover INP geometry only after matching every exported node and its precision.

    Converting ODB FP32 coordinates to FP64 cannot recover lost geometric relations.
    Accept an exported value only if it matches the input to FP64 arithmetic error,
    or is exactly the input's FP32 rounding. No geometric tolerance or smoothing is
    used; real coordinate differences and incomplete/duplicate node maps fail.
    """
    if not len(lab) or xyz.shape != (len(lab), 3) or len(inst) != len(lab) or not np.all(np.isfinite(xyz)):
        raise ValueError('Invalid/nonfinite exported node coordinates')
    keys = [(str(i), int(n)) for i, n in zip(inst, lab)]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate exported node instance/label pairs')
    source_inst, source_lab, source_xyz = inp_assembly_nodes(inp_path)
    source_keys = [(str(i), int(n)) for i, n in zip(source_inst, source_lab)]
    if len(set(source_keys)) != len(source_keys) or set(keys) != set(source_keys):
        raise ValueError('Export/INP node identities differ; cannot substitute input geometry')
    positions = {key: j for j, key in enumerate(source_keys)}
    expected = source_xyz[[positions[key] for key in keys]]
    if not np.all(np.isfinite(expected)):
        raise ValueError('Nonfinite INP node coordinates')
    # Bound arithmetic roundoff in assembly transformations, not mesh/model error.
    scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True),
                       np.max(np.ptp(expected, axis=0)))
    double_match = np.abs(xyz-expected) <= 32*np.finfo(float).eps*scale
    with np.errstate(over='ignore', invalid='ignore'):
        single_match = xyz == expected.astype(np.float32).astype(np.float64)
    bad = ~(double_match | single_match)
    if np.any(bad):
        row, component = np.argwhere(bad)[0]
        raise ValueError('Export/INP coordinates differ beyond FP64 roundoff or exact FP32 rounding '
                         f'at {keys[row]}, component {component+1}: '
                         f'export={xyz[row,component]:.17g}, INP={expected[row,component]:.17g}')
    return expected, dict(source='inp_verified_against_export',
                          verification='FP64_ROUNDOFF_OR_EXACT_FP32_ROUNDING',
                          fp32_rounded_coordinate_count=int(np.count_nonzero(~double_match & single_match)),
                          max_export_coordinate_difference=float(np.max(np.abs(xyz-expected))))


def load_mesh(exp_dir: Path | None, meta: dict | None = None, inp_path: Path | None = None) -> Mesh:
    """Preserve export node order; use matching INP geometry at its original precision.

    Without an INP (legacy parquet only), retain exported geometry and its validation
    requirements. With no export, read the built model's INP directly.
    """
    geometry = dict(source='inp' if exp_dir is None else 'modal_export')
    if exp_dir is not None:
        meta = meta or json.loads((exp_dir / "modal_export.json").read_text(encoding="utf-8"))
    if exp_dir is None or export_format(meta) == "npz":     # .inp elements, as for an npz export
        if inp_path is None:
            raise RuntimeError("an npz export or a mesh without export needs the run's .inp "
                               "for the element connectivity")
        if exp_dir is None:
            inst, lab, xyz = inp_assembly_nodes(inp_path)
        else:
            d = np.load(exp_dir / "raw_dof_map.npz", allow_pickle=False)
            inst = np.char.upper(d["node_instances"].astype(str))
            lab = d["node_labels"].astype(np.int64)
            xyz = d["node_coordinates"].astype(np.float64)
            xyz, geometry = verified_input_coordinates(inst, lab, xyz, inp_path)
        idx = np.arange(len(lab), dtype=np.int64)
        key = {(i, int(l)): k for k, (i, l) in enumerate(zip(inst, lab))}
        els = inp_shell_elements(inp_path)
        try:
            E = np.array([[key[(i, n)] for n in conn] for i, _, conn in els], dtype=np.int64)
        except KeyError as e:
            raise RuntimeError(f"element node {e} of {inp_path.name} is not in the export: wrong .inp?")
    else:
        nd = _read_parquet(exp_dir / "mesh_nodes.parquet")
        el = _read_parquet(exp_dir / "elements.parquet")
        nd = nd.sort_values("node_index").reset_index(drop=True)
        idx = nd["node_index"].to_numpy()
        inst = np.char.upper(nd["instance"].astype(str).to_numpy().astype(str))
        lab = nd["label"].to_numpy().astype(np.int64)
        xyz = nd[["x", "y", "z"]].to_numpy(dtype=np.float64)
        key = {(i, l): k for k, (i, l) in enumerate(zip(inst, lab))}
        cols = [c for c in el.columns if re.fullmatch(r"n\d+", c)]
        E = np.empty((len(el), len(cols)), dtype=np.int64)
        einst = np.char.upper(el["instance"].astype(str).to_numpy().astype(str))
        for j, c in enumerate(cols):
            lab_c = el[c].to_numpy()
            E[:, j] = [key[(a, int(b))] for a, b in zip(einst, lab_c)]
        if inp_path is not None:
            xyz, geometry = verified_input_coordinates(inst, lab, xyz, inp_path)
    # member axis = direction most orthogonal to every shell normal
    P = xyz[E[:, :4]] if E.shape[1] >= 4 else xyz[E]
    nrm = np.cross(P[:, 2] - P[:, 0], P[:, 3 if P.shape[1] > 3 else 1] - P[:, 1])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
    w, V = np.linalg.eigh(nrm.T @ nrm)
    if not np.all(np.isfinite(w)) or w[1] <= 1e-10 * max(w[2], 1e-300):
        raise ValueError("Shell normals do not determine a unique member axis (planar/degenerate section)")
    if abs(w[0]) > 1e-8 * w[1]:
        raise ValueError("Shell normals have no common member axis: GDLC requires a prismatic reference mesh")
    axis = V[:, 0]
    k = int(np.argmax(np.abs(axis)))
    if abs(abs(axis[k]) - 1) < 1e-6:
        axis = np.eye(3)[k]
        others = [i for i in range(3) if i != k]
        e1, e2 = np.eye(3)[others[0]], np.eye(3)[others[1]]
    else:
        axis = axis / np.linalg.norm(axis)
        e1 = np.cross(axis, [0, 0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
    s = xyz @ axis
    p = np.c_[xyz @ e1, xyz @ e2]
    size = float(max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), np.ptp(s)))
    tol = 1e-6 * size
    order = np.argsort(s)
    ss = s[order]
    brk = np.r_[True, np.diff(ss) > tol]
    rid_sorted = np.cumsum(brk) - 1
    ring_of = np.empty_like(rid_sorted)
    ring_of[order] = rid_sorted
    ring_z = np.array([ss[rid_sorted == r].mean() for r in range(rid_sorted[-1] + 1)])
    return Mesh(idx, inst, lab, xyz, E, axis, e1, e2, s, p, ring_of, ring_z, size, geometry)


@dataclass
class RingGeom:
    """Template ring: n nodes (fixed order), edges, and all basis data."""
    n: int
    p: np.ndarray                # (n,2)
    inst: np.ndarray             # (n,)
    edges: np.ndarray            # (m,2)
    rings: list                  # ring ids in this group
    node_rows: np.ndarray        # (n_rings_in_group, n) row indices into mesh arrays
    folds: list = field(default_factory=list)       # list of dict
    free_ends: list = field(default_factory=list)
    junctions: list = field(default_factory=list)
    seams: list = field(default_factory=list)
    ambiguous: list = field(default_factory=list)
    w: np.ndarray | None = None
    basis: dict | None = None
    proj: dict | None = None
    J: np.ndarray | None = None
    theta_part: str | None = None  # None | "first" | "second": arc side used for the fold rotation


def build_ring_groups(mesh: Mesh) -> list[RingGeom]:
    R = len(mesh.ring_z)
    E = mesh.elems[:, :4]
    # in-ring edges
    ed = np.vstack([E[:, [0, 1]], E[:, [1, 2]], E[:, [2, 3]], E[:, [3, 0]]])
    ed = ed[mesh.ring_of[ed[:, 0]] == mesh.ring_of[ed[:, 1]]]
    ed = ed[ed[:, 0] != ed[:, 1]]
    ed = np.sort(ed, axis=1)
    ed = np.unique(ed, axis=0)
    ring_edges = [[] for _ in range(R)]
    for a, b in ed:
        ring_edges[mesh.ring_of[a]].append((a, b))
    inst_codes = {v: i for i, v in enumerate(sorted(set(mesh.instance)))}
    tol = 1e-6 * mesh.size
    groups: dict = {}
    for r in range(R):
        rows = np.nonzero(mesh.ring_of == r)[0]
        q = np.round(mesh.p[rows] / tol).astype(np.int64)
        ic = np.array([inst_codes[mesh.instance[i]] for i in rows])
        o = np.lexsort((q[:, 1], q[:, 0], ic))
        rows = rows[o]
        pos = {g: k for k, g in enumerate(rows)}
        loc = np.array(sorted(tuple(sorted((pos[a], pos[b]))) for a, b in ring_edges[r]), dtype=np.int64).reshape(-1, 2)
        sig = hashlib.sha1(np.ascontiguousarray(ic[o]).tobytes() + np.ascontiguousarray(q[o]).tobytes()
                           + loc.tobytes()).hexdigest()
        if sig not in groups:
            groups[sig] = dict(rows=[rows], rings=[r], edges=loc)
        else:
            groups[sig]["rows"].append(rows)
            groups[sig]["rings"].append(r)
    out = []
    for g in groups.values():
        rows0 = g["rows"][0]
        out.append(RingGeom(n=len(rows0), p=mesh.p[rows0].copy(), inst=mesh.instance[rows0].copy(),
                            edges=g["edges"], rings=g["rings"], node_rows=np.array(g["rows"])))
    return out


# ----------------------------------------------------------------------------
# 5. Reference points: folds, free ends, junctions, seams
# ----------------------------------------------------------------------------
def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for k, (a, b) in enumerate(edges):
        adj[a].append((b, k))
        adj[b].append((a, k))
    return adj


def detect_reference_points(g: RingGeom, t: float, st: Settings) -> None:
    if t is None or not np.isfinite(t) or t <= 0:
        raise ValueError("Fold detection requires positive finite thickness")
    n, P, edges = g.n, g.p, g.edges
    adj = _adjacency(n, edges)
    deg = np.array([len(a) for a in adj])
    h = np.linalg.norm(P[edges[:, 1]] - P[edges[:, 0]], axis=1)
    if not len(h) or not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("Empty ring or zero/nonfinite length (degenerate) ring edge")
    if np.any(deg == 0):
        raise ValueError("Isolated ring nodes: a conforming cross-section ring mesh is required")
    g.free_ends = [int(i) for i in np.nonzero(deg == 1)[0]]
    g.junctions = [int(i) for i in np.nonzero(deg >= 3)[0]]
    turn = np.zeros(n)          # |turning angle| at deg-2 nodes
    kv = np.zeros((n, 2))       # curvature direction (independent of neighbour order)
    hbar = np.zeros(n)
    for i in range(n):
        if deg[i] == 2:
            (a, ka), (b, kb) = adj[i]
            t1 = (P[i] - P[a]) / h[ka]
            t2 = (P[b] - P[i]) / h[kb]
            turn[i] = abs(math.atan2(t1[0] * t2[1] - t1[1] * t2[0], t1 @ t2))
            kv[i] = t2 - t1
            hbar[i] = 0.5 * (h[ka] + h[kb])
    Rloc = np.full(n, np.inf)
    nz = np.abs(turn) > 1e-12
    Rloc[nz] = hbar[nz] / np.abs(turn[nz])
    Rf = st.fold_radius_factor * t
    cand = (deg == 2) & (Rloc < Rf)
    # clusters of contiguous candidates bending to the same side (kv_u . kv_v > 0)
    seen = np.zeros(n, bool)
    folds = []
    for i in range(n):
        if not cand[i] or seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adj[u]:
                if cand[v] and not seen[v] and kv[v] @ kv[u] > 0:
                    seen[v] = True
                    stack.append(v)
        # grow the arc over its transition nodes (same turning sign, >= 25 % of peak turning)
        # so that the arc's boundary edges are wall edges (needed for the virtual corner)
        peak = np.abs(turn[comp]).max()
        grown = True
        while grown:
            grown = False
            for u in list(comp):
                for v, _ in adj[u]:
                    if (v not in comp and deg[v] == 2 and not seen[v] and kv[v] @ kv[u] > 0
                            and turn[v] >= 0.25 * peak):
                        comp.append(v)
                        seen[v] = True
                        grown = True
        tot = float(np.degrees(turn[comp].sum()))
        if tot >= st.min_fold_angle_deg:
            comp = sorted(comp)
            cw = np.abs(turn[comp])
            ctr = (P[comp] * cw[:, None]).sum(0) / cw.sum()
            ref = comp[int(np.argmin(np.linalg.norm(P[comp] - ctr, axis=1)))]
            # arc-extent margin: how clearly the arc's end (tangent) nodes and the first wall
            # nodes outside it are separated by the 25 %-of-peak growth rule
            outside = [v for u in comp for v, _ in adj[u] if v not in comp and deg[v] == 2]
            folds.append(dict(nodes=[int(c) for c in comp], ref=int(ref), turning_deg=tot,
                              radius_est=float(np.median(Rloc[comp])),
                              arc_inside_min_ratio=float(np.abs(turn[comp]).min() / peak),
                              arc_outside_max_ratio=float(max((abs(turn[v]) / peak for v in outside),
                                                              default=0.0))))
    for j in g.junctions:
        folds.append(dict(nodes=[int(j)], ref=int(j), turning_deg=float("nan"), radius_est=0.0, junction=True))
    g.folds = folds
    band = (deg == 2) & (Rloc > Rf / 1.25) & (Rloc < Rf * 1.25)
    g.ambiguous = [dict(node=int(i), R=float(Rloc[i]), R_fold=Rf) for i in np.nonzero(band)[0]]


def attach_seams(g: RingGeom, mesh: Mesh, inp: InpInfo | None) -> None:
    g.seams = []
    if inp is None or not inp.pairs:
        return
    key = {(i, int(l)): k for k, (i, l) in enumerate(zip(mesh.instance, mesh.label))}
    tol_match = 1e-3 * mesh.size
    seen = {}
    for kind, a, b in inp.pairs:
        ka, kb = key.get((a[0], a[1])), key.get((b[0], b[1]))
        if ka is None or kb is None:
            continue
        pa_, pb_ = mesh.p[ka], mesh.p[kb]
        ia = np.nonzero((g.inst == a[0]) & (np.linalg.norm(g.p - pa_, axis=1) < tol_match))[0]
        ib = np.nonzero((g.inst == b[0]) & (np.linalg.norm(g.p - pb_, axis=1) < tol_match))[0]
        if len(ia) != 1 or len(ib) != 1:
            continue
        k = (int(ia[0]), int(ib[0]))
        if k in seen:
            seen[k]["n_links"] += 1
            seen[k]["z"].append(float(mesh.s[ka]))
            continue
        seen[k] = dict(A=k[0], B=k[1], kind=kind, inst=(a[0], b[0]), n_links=1, z=[float(mesh.s[ka])],
                       axial_rows=(ka, kb))
    g.seams = list(seen.values())


# ----------------------------------------------------------------------------
# 6. Constraint / energy operators on one ring (2n DOF: [x0,y0,x1,y1,...])
# ----------------------------------------------------------------------------
def _edge_rotation_rows(g: RingGeom):
    """rho_k = ((u_b - u_a) . n_ab) / h_ab for every edge k=(a,b)."""
    m = len(g.edges)
    Rm = np.zeros((m, 2 * g.n))
    Ex = np.zeros((m, 2 * g.n))
    for k, (a, b) in enumerate(g.edges):
        d = g.p[b] - g.p[a]
        hk = np.linalg.norm(d)
        tv = d / hk
        nv = np.array([-tv[1], tv[0]])
        Rm[k, 2 * b:2 * b + 2] += nv / hk
        Rm[k, 2 * a:2 * a + 2] -= nv / hk
        Ex[k, 2 * b:2 * b + 2] += tv
        Ex[k, 2 * a:2 * a + 2] -= tv
    h = np.linalg.norm(g.p[g.edges[:, 1]] - g.p[g.edges[:, 0]], axis=1)
    return Rm, Ex, h


def _node_rotation_row(g, adj, Rm, i):
    ks = [k for _, k in adj[i]]
    return Rm[ks].mean(0)


def fold_rotation_edges(g: RingGeom, f: dict) -> list[int]:
    """Internal edges of a fold arc whose chord rotation measures the arc rotation:
    those NOT touching the fold reference node (so the REF_L *Equation, whose dependent
    DOFs are the reference node's, never repeats them), else all internal edges.
    On the rigid-arc (inextensional) space every internal edge has the same rotation,
    so the choice only matters for the small non-rigid part of a measured field."""
    S = set(f["nodes"])
    internal = [k for k, (a, b) in enumerate(g.edges) if a in S and b in S]
    away = [k for k in internal if f["ref"] not in (int(g.edges[k][0]), int(g.edges[k][1]))]
    use = away or internal
    if g.theta_part and len(use) > 1:
        # sensitivity variant: rotation measured on one side of the arc only (edges ordered
        # along the arc). Both sides are equally admissible measures of a rigid arc's rotation.
        nb = {u: [] for u in S}
        for k in internal:
            a, b = int(g.edges[k][0]), int(g.edges[k][1])
            nb[a].append((b, k))
            nb[b].append((a, k))
        start = min((u for u in S if len(nb[u]) <= 1), default=min(S))
        order, prev, cur = [], None, start
        while True:
            nxt = [(v, k) for v, k in nb[cur] if v != prev]
            if not nxt:
                break
            v, k = nxt[0]
            order.append(k)
            prev, cur = cur, v
        ranked = [k for k in order if k in use] or use
        h = max(1, len(ranked) // 2)
        use = ranked[:h] if g.theta_part == "first" else (ranked[h:] or ranked[-1:])
    return use


def build_operators(g: RingGeom):
    n2 = 2 * g.n
    adj = _adjacency(g.n, g.edges)
    Rm, Ex, h = _edge_rotation_rows(g)
    # tributary weights
    w = np.zeros(g.n)
    for k, (a, b) in enumerate(g.edges):
        w[a] += 0.5 * h[k]
        w[b] += 0.5 * h[k]
    g.w = w
    # inextensibility (rows scaled to unit norm later)
    rows = [Ex]
    # rigid fold arcs: equal rotation of consecutive internal edges
    for f in g.folds:
        S = set(f["nodes"])
        if len(S) < 2:
            continue
        for i in f["nodes"]:
            ks = [k for v, k in adj[i] if v in S]
            for k1, k2 in zip(ks[:-1], ks[1:]):
                rows.append((Rm[k1] - Rm[k2])[None, :])
    Emat = np.vstack(rows)
    # bending energy rows: curvature at deg-2 nodes, pairwise at junctions
    krows = []
    for i in range(g.n):
        inc = adj[i]
        if len(inc) < 2:
            continue
        for x in range(len(inc)):
            for y in range(x + 1, len(inc)):
                k1, k2 = inc[x][1], inc[y][1]
                hb = 0.5 * (h[k1] + h[k2])
                krows.append((Rm[k1] - Rm[k2]) / math.sqrt(hb))
    Kc = np.array(krows) if krows else np.zeros((0, n2))
    # driver rows: translation of each fold's virtual sharp corner c (intersection of the
    # two adjacent wall tangents), carried rigidly by the fold arc:
    #   u_c = u_ref + theta_arc x (c - p_ref).
    # With c on both wall lines, arc rotation does not change the wall chords, so the
    # number of D/G drivers equals the sharp-corner (cFSM) count (Beregszaszi & Adany 2019).
    Qf = []
    for f in g.folds:
        r = f["ref"]
        S = set(f["nodes"])
        c_pt = g.p[r].copy()
        th_row = None
        if len(S) >= 2:
            bnd = [(a, b) if a in S else (b, a) for (a, b) in g.edges if (a in S) != (b in S)]
            th_row = Rm[fold_rotation_edges(g, f)].mean(0)
            if len(bnd) == 2:
                (i1, o1), (i2, o2) = bnd
                d1 = g.p[i1] - g.p[o1]
                d2 = g.p[i2] - g.p[o2]
                M = np.c_[d1, -d2]
                if abs(np.linalg.det(M)) > 1e-6 * np.linalg.norm(d1) * np.linalg.norm(d2):
                    s_, _ = np.linalg.solve(M, g.p[o2] - g.p[o1])
                    c_pt = g.p[o1] + s_ * d1
        f["virtual_corner"] = c_pt.tolist()
        off = c_pt - g.p[r]
        rx = np.zeros(n2); rx[2 * r] = 1.0
        ry = np.zeros(n2); ry[2 * r + 1] = 1.0
        if th_row is not None:
            rx = rx - off[1] * th_row
            ry = ry + off[0] * th_row
        Qf += [rx, ry]
    # seam jump rows
    J = []
    for s in g.seams:
        A, B = s["A"], s["B"]
        r = g.p[B] - g.p[A]
        thA = _node_rotation_row(g, adj, Rm, A)
        thB = _node_rotation_row(g, adj, Rm, B)
        jx = np.zeros(n2); jx[2 * B] += 1; jx[2 * A] -= 1; jx += r[1] * thA       # -theta*(-r_y)
        jy = np.zeros(n2); jy[2 * B + 1] += 1; jy[2 * A + 1] -= 1; jy -= r[0] * thA
        jr = (thB - thA) * max(np.linalg.norm(r), np.mean(h))
        J += [jx, jy, jr]
    Qf = np.array(Qf).reshape(-1, n2)
    J = np.array(J).reshape(-1, n2)
    return Emat, Kc, Qf, J


# ----------------------------------------------------------------------------
# 7. Basis construction
# ----------------------------------------------------------------------------
def _unit_rows(M):
    if M.shape[0] == 0:
        return M
    nr = np.linalg.norm(M, axis=1, keepdims=True)
    return M[nr[:, 0] > 0] / nr[nr[:, 0] > 0]


def null_space(M, rtol=1e-10):
    if M.shape[0] == 0:
        return np.eye(M.shape[1])
    u, s, vt = np.linalg.svd(M, full_matrices=True)
    tol = rtol * (s[0] if s.size else 1.0) * max(M.shape)
    r = int((s > tol).sum())
    return vt[r:].T.copy()


def w_orth(V, w2, against=None, rtol=1e-10):
    """W-orthonormal basis of span(V) after removing span(against) (W-orthonormal)."""
    if V.shape[1] == 0:
        return V
    sq = np.sqrt(w2)[:, None]
    V = V / np.maximum(np.sqrt((w2[:, None] * V ** 2).sum(0)), 1e-300)   # unit W-norm columns
    if against is not None and against.shape[1]:
        V = V - against @ (against.T @ (w2[:, None] * V))
        V = V - against @ (against.T @ (w2[:, None] * V))                  # re-orthogonalise
    u, s, vt = np.linalg.svd(sq * V, full_matrices=False)
    keep = s > rtol * max(V.shape) ** 0.5 * 1e2                           # absolute: columns had unit norm
    return u[:, keep] / sq


def ring_components(g: RingGeom) -> list[np.ndarray]:
    """Connected parts of the ring's wall graph (the separate pieces of a built-up section;
    seams/connectors are NOT edges, so every bolted piece is its own component)."""
    adj = _adjacency(g.n, g.edges)
    seen = np.zeros(g.n, bool)
    comps = []
    for i in range(g.n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(np.array(sorted(comp)))
    return comps


def _rigid_fields(g: RingGeom, nodes: np.ndarray) -> np.ndarray:
    """In-plane rigid translations x, y and rotation of the given nodes (zero elsewhere)."""
    n2 = 2 * g.n
    w = g.w[nodes]
    c = (g.p[nodes] * w[:, None]).sum(0) / w.sum()
    R = np.zeros((n2, 3))
    R[2 * nodes, 0] = 1
    R[2 * nodes + 1, 1] = 1
    R[2 * nodes, 2] = -(g.p[nodes, 1] - c[1])
    R[2 * nodes + 1, 2] = (g.p[nodes, 0] - c[0])
    return R


def build_basis(g: RingGeom) -> dict:
    """Unique split of the ring's in-plane space  R^2n = G + C + D + L + O.

    All classes are defined by fold-line kinematics of the WALLS only (cFSM criteria);
    the connectors between the pieces of a built-up section do not enter any definition
    (they exist only at discrete fastener rows, and between them the pieces are free):
      I = inextensional fields with rigid fold arcs;
      L = fields of I with every fold (virtual corner) translation zero;
      F = fields of I with minimum transverse bending energy for given fold translations
          (energy-orthogonal to L; contains every rigid motion of every piece);
      G = rigid in-plane motion of the whole section (3);
      C = relative rigid in-plane motion of the pieces: each piece moves as a rigid body,
          but not with the others (3*n_pieces - 3; zero for a one-piece section);
      D = F without G and C (W-orthogonal): distortion of the pieces' cross-sections;
      O = complement of I without fold translations (transverse extension, arc
          deformation, mesh noise): W-orthogonal to L inside {fold translations = 0}.
    Fixed folds suppress G/C/D only if they constrain all rigid piece motions.
    G, C, D are mutually W-orthogonal; L is
    energy-orthogonal to G+C+D; O is W-orthogonal to L."""
    Emat, Kc, Qf, J = build_operators(g)
    n2 = 2 * g.n
    w2 = np.repeat(g.w, 2)
    A = Kc.T @ Kc
    BI = null_space(_unit_rows(Emat))                      # inextensional space
    m = BI.shape[1]
    NL = null_space(_unit_rows(Qf @ BI) if Qf.shape[0] else np.zeros((0, m)))   # L coords
    AI = BI.T @ A @ BI
    notes = []
    fixed_rigid = np.zeros((n2, 0))
    if NL.shape[1]:
        S = NL.T @ AI @ NL
        ev, evec = np.linalg.eigh(0.5 * (S + S.T))
        zero = ev < 1e-10 * max(ev.max(), 1e-300)
        if zero.any():
            # e.g. a piece without folds: its rigid motion moves no fold -> belongs to F
            notes.append(f"{int(zero.sum())} zero-energy direction(s) with fixed folds moved from L to F")
            fixed_rigid = BI @ NL @ evec[:, zero]
            NL = NL @ evec[:, ~zero]
    # F = energy-orthogonal complement of L in I. NL^T AI has full row rank dim(L) (AI is
    # positive definite on the retained L), so the complement is taken with that known rank:
    # a tolerance-based rank would drop the softest L directions of long flat walls on fine meshes.
    if NL.shape[1]:
        _, _, vt_ = np.linalg.svd(NL.T @ AI, full_matrices=True)
        Fc = vt_[NL.shape[1]:].T.copy()
    else:
        Fc = np.eye(m)
    F = BI @ Fc
    L = BI @ NL
    comps = ring_components(g)
    G = w_orth(_rigid_fields(g, np.arange(g.n)), w2)
    Rp = np.hstack([_rigid_fields(g, c) for c in comps])
    C = w_orth(Rp, w2, against=G) if len(comps) > 1 else np.zeros((n2, 0))
    Fo = w_orth(F, w2)
    # every rigid motion of every piece must lie in F (zero bending energy, inextensional)
    resid = Rp - Fo @ (Fo.T @ (w2[:, None] * Rp))
    gres = float(np.sqrt((w2[:, None] * resid ** 2).sum() / (w2[:, None] * Rp ** 2).sum()))
    if gres > 1e-6:
        raise RuntimeError(f"rigid-body motion of a piece not contained in the fold-driven space "
                           f"(residual {gres:.2e}); check fold detection")
    if C.shape[1] != 3 * len(comps) - 3:
        raise RuntimeError(f"relative rigid motions of {len(comps)} pieces have rank {C.shape[1]}, "
                           f"expected {3 * len(comps) - 3}")
    GC = np.hstack([G, C])
    D = w_orth(Fo, w2, against=w_orth(GC, w2))
    # O = the non-inextensional rest, chosen WITHOUT fold translations: the W-orthogonal
    # complement of L inside {fields with all fold translations zero}. O therefore never
    # carries part of a fold motion, so a field whose folds do not move has exactly zero
    # G, C and D, and fold motions are attributed to G/C/D only (with the plain W-complement
    # of I, the small extension / arc-deformation part of a real FE mode moved the folds
    # and leaked into G/D through the oblique split).
    # (Only fold translations that inextensional fields can produce are excluded from O;
    # e.g. in a closed tube the corner translations are not independent in I.)
    if Qf.shape[0]:
        u_, s_, _ = np.linalg.svd(Qf @ BI, full_matrices=False)
        Ur = u_[:, s_ > 1e-10 * max(s_.max() if s_.size else 0.0, 1e-300) * max(Qf.shape)]
        drivers = Ur.T @ Qf
    else:
        drivers = np.zeros((0, n2))
    # A flat strip, or a piece with a single fold, has rigid motions that do not
    # move any fold. Those already belong to F, so O must exclude them as well.
    # Otherwise T contains duplicate rigid directions and is not a direct sum.
    if fixed_rigid.shape[1]:
        drivers = np.vstack([drivers, fixed_rigid.T * w2[None, :]])
    Z = null_space(_unit_rows(drivers))
    # W-orthogonal complement of L inside span(Z), via the null space of L^T W Z (rank-robust)
    O = w_orth(Z @ null_space((L * w2[:, None]).T @ Z), w2) if L.shape[1] else w_orth(Z, w2)
    T = np.hstack([G, D, L, C, O])
    if T.shape[1] != n2:
        raise RuntimeError(f"class spaces do not span the slice space: {T.shape[1]} != {n2}")
    # Equilibrate columns in the physical W metric, then solve for the dual via QR.
    # This changes neither the class spaces nor the oblique split. No singular
    # direction is dropped/regularised; an unstable split must still fail self_test.
    sq = np.sqrt(w2)
    norms = np.linalg.norm(sq[:,None]*T, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError('Nonfinite/zero class basis column; check slice geometry')
    T = T/norms
    qt, rt = np.linalg.qr(sq[:,None]*T)
    cond = float(np.linalg.cond(rt))
    Tinv = np.linalg.solve(rt, qt.T*sq[None,:])
    sizes = dict(G=G.shape[1], D=D.shape[1], L=L.shape[1], C=C.shape[1], O=O.shape[1])
    proj, i0 = {}, 0
    for k in CLASSES:
        nk = sizes[k]
        proj[k] = T[:, i0:i0 + nk] @ Tinv[i0:i0 + nk, :]
        i0 += nk
    g.basis = dict(T=T, sizes=sizes, cond=cond, notes=notes, rigid_residual=gres, J=J,
                   dim_inextensional=m, w2=w2, A=A, Qf=Qf, n_pieces=len(comps))
    g.proj = proj
    g.J = J
    return g.basis


def self_test(g: RingGeom, rng=None) -> dict:
    # A private generator makes this diagnostic reproducible across runs and threads.
    if rng is None:
        rng = np.random.default_rng(12345)
    T, w2 = g.basis["T"], g.basis["w2"]
    sizes = g.basis["sizes"]
    worst = 0.0
    for _ in range(20):
        c = rng.standard_normal(T.shape[1])
        parts, i0 = {}, 0
        for k in CLASSES:
            nk = sizes[k]
            parts[k] = T[:, i0:i0 + nk] @ c[i0:i0 + nk]
            i0 += nk
        phi = sum(parts.values())
        for k in CLASSES:
            got = g.proj[k] @ phi
            err = np.sqrt((w2 * (got - parts[k]) ** 2).sum() / max((w2 * phi ** 2).sum(), 1e-300))
            worst = max(worst, float(err) if np.isfinite(err) else float('inf'))
    # pure rigid translation must be 100 % G
    rig = np.zeros(T.shape[0]); rig[0::2] = 1.0
    gshare = float((w2 * (g.proj["G"] @ rig) ** 2).sum() / (w2 * rig ** 2).sum())
    # Check both translations and rigid rotation, not just one scalar energy share.
    rigid = _rigid_fields(g, np.arange(g.n))
    rigid_error = np.sqrt(np.sum(w2[:,None]*(g.proj['G']@rigid-rigid)**2, axis=0)
                          / np.maximum(np.sum(w2[:,None]*rigid**2, axis=0), 1e-300))
    rigid_max = float(np.max(rigid_error))
    ok = worst < 1e-8 and abs(gshare - 1) < 1e-10 and rigid_max < 1e-8
    return dict(max_rel_error=worst, rigid_translation_G_share=gshare,
                rigid_motion_max_rel_error=rigid_max, basis_condition_number=g.basis['cond'], passed=bool(ok))


# ----------------------------------------------------------------------------
# 8. Mode loading (all shards in RAM, parallel threads)
# ----------------------------------------------------------------------------
def _load_modes_npz(exp_dir: Path, meta: dict, mesh: Mesh, st: Settings, rotations: bool = False):
    """abaqus_mfsm_export.py shards: vectors (raw DOF x modes), node-major
    [U1 U2 U3 UR1 UR2 UR3] in the node order of raw_dof_map.npz (verified, not assumed).
    rotations=True also returns UR (None when the export has no rotations)."""
    order = [int(m["mode"]) for m in meta["modes"]]
    want = order[:st.max_modes] if st.max_modes else order
    mps = int(meta.get("modes_per_shard") or 0)
    shards = list(meta["shards"])
    if mps and len(order) == len(want):
        pass
    elif mps:
        shards = shards[:int(math.ceil(len(want) / mps))]
    N = len(mesh.node_index)
    d = np.load(exp_dir / "raw_dof_map.npz", allow_pickle=False)
    r_inst = np.char.upper(d["instances"].astype(str))
    r_lab = d["labels"].astype(np.int64)
    r_dof = d["dofs"].astype(np.int64)
    node_major = (len(r_dof) == 6 * N and np.array_equal(r_dof, np.tile(np.arange(1, 7), N))
                  and np.array_equal(r_lab[::6], mesh.label) and np.array_equal(r_inst[::6], mesh.instance))
    if not node_major:                                   # general mapping, still exact
        key = {(i, int(l)): k for k, (i, l) in enumerate(zip(mesh.instance, mesh.label))}
        rnode = np.array([key[(i, int(l))] for i, l in zip(r_inst, r_lab)])
        sel = r_dof <= 3
        selr = r_dof >= 4
    pos = {m: k for k, m in enumerate(want)}
    if len(set(order)) != len(order):
        raise ValueError("Duplicate mode IDs in export metadata")
    if not (len(r_inst) == len(r_lab) == len(r_dof)) or np.any((r_dof < 1) | (r_dof > 6)):
        raise ValueError("Invalid raw translation/rotation DOF map")
    raw_keys = list(zip(r_inst.tolist(), r_lab.tolist(), r_dof.tolist()))
    if len(set(raw_keys)) != len(raw_keys):
        raise ValueError("Duplicate translation/rotation entries in raw DOF map")
    required = {(str(i), int(l), d) for i, l in zip(mesh.instance, mesh.label) for d in (1, 2, 3)}
    if not required.issubset(set(raw_keys)):
        raise ValueError("Incomplete raw translation DOF map")
    U = _mode_array((len(want), N, 3))
    have_rot = rotations and {(str(i), int(l), d) for i, l in zip(mesh.instance, mesh.label)
                              for d in (4, 5, 6)}.issubset(set(raw_keys))
    UR = _mode_array((len(want), N, 3)) if have_rot else None

    def _one(name):
        z = np.load(exp_dir / name, allow_pickle=False)
        vec = z["vectors"]
        ms = [int(x) for x in z["modes"]]
        if vec.shape != (len(r_dof), len(ms)):
            raise ValueError("Mode shard shape disagrees with raw DOF map")
        for j, m in enumerate(ms):
            if m not in pos:
                continue
            if node_major:
                U[pos[m]] = vec[:, j].reshape(N, 6)[:, :3]
                if have_rot:
                    UR[pos[m]] = vec[:, j].reshape(N, 6)[:, 3:]
            else:
                U[pos[m], rnode[sel], r_dof[sel] - 1] = vec[sel, j]
                if have_rot:
                    UR[pos[m], rnode[selr], r_dof[selr] - 4] = vec[selr, j]
        return ms

    t0 = time.time()
    shard_bytes = max(1, len(r_dof) * max(1, mps or len(want)) * 8)
    workers = min(N_THREADS, len(shards), resources.batch_capacity(_available_ram(), shard_bytes))
    with cf.ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        got = [m for ms in ex.map(_one, shards) for m in ms]
    missing = sorted(set(want) - set(got))
    if missing:
        raise RuntimeError(f"modes missing from the shards: {missing[:10]}...")
    if len(got) != len(set(got)) or not np.all(np.isfinite(U)):
        raise ValueError("Duplicate modes or incomplete/nonfinite translations in shards")
    if have_rot and not np.all(np.isfinite(UR)):
        raise ValueError("Incomplete/nonfinite rotations in shards")
    log(f"  loaded {len(want)} modes x {N} nodes (npz{', U+UR' if have_rot else ''}) in "
        f"{time.time()-t0:.1f} s ({N_THREADS} threads)")
    return (want, U, UR) if rotations else (want, U)


def load_modes(exp_dir: Path, meta: dict, mesh: Mesh, st: Settings, rotations: bool = False):
    """(modes, U) or, with rotations=True, (modes, U, UR) - UR None when not exported."""
    if export_format(meta) == "npz":
        return _load_modes_npz(exp_dir, meta, mesh, st, rotations)
    shards = meta["shards"]
    modes_all = [m for sh in shards for m in sh["modes"]]
    if len(set(modes_all)) != len(modes_all):
        raise ValueError("Duplicate mode IDs in Parquet shards")
    if st.max_modes:
        modes_all = modes_all[:st.max_modes]
        shards = [sh for sh in shards if any(m in modes_all for m in sh["modes"])]
    mode_pos = {m: k for k, m in enumerate(modes_all)}
    N = len(mesh.node_index)
    pos_of_index = np.full(int(mesh.node_index.max()) + 1, -1, np.int64)
    pos_of_index[mesh.node_index] = np.arange(N)
    U = _mode_array((len(modes_all), N, 3))
    have_rot = bool(rotations and meta.get("rotations_available"))
    UR = _mode_array((len(modes_all), N, 3)) if have_rot else None
    if psutil:
        need = U.nbytes * 2
        avail = psutil.virtual_memory().available
        if need > avail:
            log(f"WARNING: mode array needs ~{need/1e9:.1f} GB, available {avail/1e9:.1f} GB")

    if pq is None:
        raise RuntimeError("this export is in parquet format: pip install pyarrow")

    def _one(sh):
        tab = pq.read_table(exp_dir / sh["parquet_file"], use_threads=True)
        ni = tab.column("node_index").to_numpy()
        if not np.array_equal(np.sort(ni), np.sort(mesh.node_index)):
            raise ValueError("Mode shard must contain each mesh node exactly once")
        rows = pos_of_index[ni]
        for m in sh["modes"]:
            if m not in mode_pos:
                continue
            for c in range(3):
                col = f"m{m:04d}_u{c+1}"
                U[mode_pos[m], rows, c] = tab.column(col).to_numpy()
                if have_rot:
                    UR[mode_pos[m], rows, c] = tab.column(f"m{m:04d}_ur{c+1}").to_numpy()
        return len(sh["modes"])

    t0 = time.time()
    largest = max(len(sh["modes"]) for sh in shards)
    shard_bytes = max(1, N * largest * (6 if have_rot else 3) * 8 * 2)
    workers = min(N_THREADS, len(shards), resources.batch_capacity(_available_ram(), shard_bytes))
    with cf.ThreadPoolExecutor(max_workers=max(workers, 1)) as ex:
        list(ex.map(_one, shards))
    if not np.all(np.isfinite(U)):
        raise ValueError("Incomplete/nonfinite modal translations in Parquet shards")
    if have_rot and not np.all(np.isfinite(UR)):
        raise ValueError("Incomplete/nonfinite modal rotations in Parquet shards")
    log(f"  loaded {len(modes_all)} modes x {N} nodes{' (U+UR)' if have_rot else ''} in "
        f"{time.time()-t0:.1f} s ({N_THREADS} threads)")
    return (modes_all, U, UR) if rotations else (modes_all, U)


# ----------------------------------------------------------------------------
# 9. Projection (CPU BLAS on all cores, or CUDA via CuPy)
# ----------------------------------------------------------------------------
def _xp(st: Settings):
    """CuPy if a CUDA device is present AND a test kernel compiles/runs; otherwise NumPy.
    (CuPy can see a GPU but still fail at the first elementwise kernel when the CUDA
    toolkit headers are missing - this is tested here so the run never aborts.)"""
    if st.gpu == "off":
        return np, False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp
            usable = []
            for device in range(cp.cuda.runtime.getDeviceCount()):
                try:
                    with cp.cuda.Device(device):
                        a = cp.arange(8.0).reshape(2, 4)
                        b = ((a * cp.sqrt(a + 1.0)) ** 2).sum(0)
                        c = cp.ascontiguousarray(cp.transpose(a)) @ a
                        float((b.sum() + c.sum()).get())
                        cp.cuda.Stream.null.synchronize()
                    usable.append(device)
                except Exception as error:
                    log(f"  CUDA device {device} unavailable: {str(error)[:120]}")
            if not usable:
                raise RuntimeError("no usable CUDA device")
        st._gpu_devices = usable
        log(f"  GPU: CuPy/CUDA test kernels OK on {len(usable)} device(s)")
        return cp, True
    except Exception as e:
        msg = str(e).splitlines()[0][:160] if str(e) else type(e).__name__
        if st.gpu == "on":
            log(f"  WARNING: --gpu on but CuPy/CUDA unusable ({msg}); using CPU x{N_THREADS}")
        elif "cupy" in sys.modules or not isinstance(e, ImportError):
            log(f"  GPU not usable ({msg}); using CPU x{N_THREADS}")
        return np, False


def _projection_chunk(g, mesh, U, rows, dz, xp, on_gpu):
    """One transactional slice block: no report accumulation until all operations succeed."""
    nm, c, n2 = len(U), len(rows), 2 * g.n
    X = np.einsum("mrnc,cd->ndrm", U[:, rows, :], np.c_[mesh.e1, mesh.e2], optimize=True).reshape(n2, c, nm)
    Xc = xp.asarray(X)
    sw = xp.asarray(np.sqrt(g.basis["w2"]))[:, None, None]
    sdz = xp.asarray(np.sqrt(dz))[None, :, None]
    Xw2 = xp.ascontiguousarray(xp.transpose(Xc * sw * sdz, (1, 0, 2))).reshape(c * n2, nm)
    total = _np(Xw2.T @ Xw2, on_gpu)
    Qf = xp.asarray(g.basis["Qf"])
    fn = np.zeros(nm)
    if Qf.shape[0]:
        qx = (Qf @ Xc.reshape(n2, c * nm)).reshape(-1, c, nm)
        fn = _np(((qx * sdz) ** 2).sum((0, 1)), on_gpu) / max(Qf.shape[0] // 2, 1)
    fd = _np(((Xc * sw * sdz) ** 2).sum((0, 1)), on_gpu) / float(g.w.sum())
    prof = np.empty((len(CLASSES) + 1, c, nm))
    prof[-1] = _np(((Xc * sw) ** 2).sum(0), on_gpu)
    H, Hs = {}, {}
    for ci, k in enumerate(CLASSES):
        Y = (xp.asarray(g.proj[k]) @ Xc.reshape(n2, c * nm)).reshape(n2, c, nm)
        Yw = xp.ascontiguousarray(xp.transpose(Y * sw * sdz, (1, 0, 2))).reshape(c * n2, nm)
        H[k] = _np(Yw.T @ Yw, on_gpu)
        Hs[k] = _np(Yw.T @ Xw2, on_gpu)
        prof[ci] = _np(((Y * sw) ** 2).sum(0), on_gpu)
        del Y, Yw
    jumps = slip = None
    if g.J.shape[0]:
        jumps = np.tensordot(g.J, X, axes=(1, 0))
        ax = np.array([[s["A"], s["B"]] for s in g.seams])
        ua = np.einsum("mrnc,c->nrm", U[:, rows[:, ax[:, 0]], :], mesh.axis, optimize=True)
        ub = np.einsum("mrnc,c->nrm", U[:, rows[:, ax[:, 1]], :], mesh.axis, optimize=True)
        slip = ub - ua
    return dict(total=total, H=H, Hs=Hs, fn=fn, fd=fd, prof=prof, jumps=jumps, slip=slip)


def _projection_range(g, mesh, U, ids, dz, xp, on_gpu, workers):
    """Adapt to live RAM/VRAM and retry allocation failures without double-counting."""
    nm, n2 = len(U), 2 * g.n
    per_ring = n2 * nm * 8 * 16
    fixed = (2 * len(CLASSES) + 1) * nm * nm * 8 + 6 * n2 * n2 * 8
    result = None
    offset, ceiling = 0, len(ids)
    while offset < len(ids):
        available = _available_ram() // workers
        if on_gpu:
            free, _ = xp.cuda.runtime.memGetInfo()
            free += xp.get_default_memory_pool().free_bytes()
            available = min(available, free)
        try:
            count = min(len(ids) - offset, ceiling, resources.batch_capacity(available, per_ring, fixed))
        except MemoryError:
            if on_gpu:
                log("  VRAM cannot hold one projection ring; completing this range on CPU")
                xp, on_gpu = np, False
                continue
            raise
        chosen = ids[offset:offset + count]
        try:
            block = _projection_chunk(g, mesh, U, g.node_rows[chosen], dz[chosen], xp, on_gpu)
        except MemoryError:
            if on_gpu:
                xp.get_default_memory_pool().free_all_blocks()
            if count > 1:
                ceiling = max(1, count // 2)
                continue
            if on_gpu:
                log("  one-ring CUDA allocation failed; completing this range on CPU")
                xp, on_gpu = np, False
                continue
            raise MemoryError("available RAM cannot hold one projection ring")
        if result is None:
            result = dict(total=np.zeros_like(block["total"]),
                          H={k: np.zeros_like(block["H"][k]) for k in CLASSES},
                          Hs={k: np.zeros_like(block["Hs"][k]) for k in CLASSES},
                          fn=np.zeros(nm), fd=np.zeros(nm),
                          prof=np.zeros((len(CLASSES) + 1, len(ids), nm)), jumps=None, slip=None)
            if block["jumps"] is not None:
                result["jumps"] = np.empty((block["jumps"].shape[0], len(ids), nm))
                result["slip"] = np.empty((block["slip"].shape[0], len(ids), nm))
        result["total"] += block["total"]
        result["fn"] += block["fn"]; result["fd"] += block["fd"]
        for k in CLASSES:
            result["H"][k] += block["H"][k]; result["Hs"][k] += block["Hs"][k]
        result["prof"][:, offset:offset + count] = block["prof"]
        if block["jumps"] is not None:
            result["jumps"][:, offset:offset + count] = block["jumps"]
            result["slip"][:, offset:offset + count] = block["slip"]
        offset += count
        del block
    return ids, result


def decompose(groups, mesh: Mesh, U: np.ndarray, st: Settings):
    """Per-mode class energies, Gram matrices (for clusters), longitudinal profiles,
    seam jump profiles. Every projection is one large GEMM (all BLAS threads / GPU)."""
    if not np.all(np.isfinite(U)):
        raise ValueError("Nonfinite modal translations cannot be decomposed")
    xp, on_gpu = _xp(st)
    nm = U.shape[0]
    R = len(mesh.ring_z)
    z = mesh.ring_z
    dz = np.zeros(R)
    dz[1:] += 0.5 * np.diff(z)
    dz[:-1] += 0.5 * np.diff(z)
    H = {k: np.zeros((nm, nm)) for k in CLASSES}
    Hs = {k: np.zeros((nm, nm)) for k in CLASSES}      # <phi_M , phi>_W  (signed, additive)
    Htot = np.zeros((nm, nm))
    # kinematic indicator: RMS virtual-corner (fold) translation / RMS mid-line displacement
    fold_num = np.zeros(nm)
    fold_den = np.zeros(nm)
    prof = np.zeros((len(CLASSES) + 1, R, nm))       # per-ring ||u_M||_W^2 (+ total)
    seam_prof = []
    for g in groups:
        devices = getattr(st, "_gpu_devices", [0]) if on_gpu else [None]
        parts = [p for p in np.array_split(np.arange(len(g.rings)), len(devices)) if len(p)]
        def work(item):
            device, ids = item
            ctx = xp.cuda.Device(device) if on_gpu else nullcontext()
            with ctx:
                return _projection_range(g, mesh, U, ids, dz[g.rings], xp, on_gpu, len(parts))
        if len(parts) > 1:
            limits = threadpool_limits(1, user_api="blas") if "threadpool_limits" in globals() else nullcontext()
            with limits, cf.ThreadPoolExecutor(max_workers=len(parts)) as pool:
                results = list(pool.map(work, zip(devices, parts)))
        else:
            results = [work((devices[0], parts[0]))]
        jumps = slip = None
        for ids, block in results:
            Htot += block["total"]; fold_num += block["fn"]; fold_den += block["fd"]
            for k in CLASSES:
                H[k] += block["H"][k]; Hs[k] += block["Hs"][k]
            prof[:, np.asarray(g.rings)[ids]] += block["prof"]
            if block["jumps"] is not None:
                if jumps is None:
                    jumps = np.empty((block["jumps"].shape[0], len(g.rings), nm))
                    slip = np.empty((block["slip"].shape[0], len(g.rings), nm))
                jumps[:, ids] = block["jumps"]; slip[:, ids] = block["slip"]
        if jumps is not None:
            seam_prof.append(dict(g=g, jumps=jumps, axial_slip=slip, z=mesh.ring_z[g.rings]))
        del results
    Ein = np.stack([np.diag(H[k]) for k in CLASSES], axis=1)
    Etot = np.diag(Htot).copy()
    # axial-slip diagnostic: rms seam slip / rms axial displacement of the whole member
    slip_ratio = None
    if seam_prof:
        ua_all = U @ mesh.axis                                     # (nm, N)
        rms_ax = np.sqrt((ua_all ** 2).mean(1))
        slip = np.concatenate([sp["axial_slip"] for sp in seam_prof], axis=0)
        slip_ratio = np.sqrt((slip ** 2).mean((0, 1))) / np.maximum(rms_ax, 1e-300)
    Es = np.stack([np.diag(Hs[k]) for k in CLASSES], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fold_ratio = np.sqrt(fold_num / fold_den)
    return dict(E=Ein, Es=Es, Hs=Hs, Etot=Etot, H=H, Htot=Htot, prof=prof, dz=dz, seam_prof=seam_prof,
                gpu=on_gpu, axial_slip_ratio=slip_ratio, fold_ratio=fold_ratio)


def _np(a, on_gpu):
    return a.get() if on_gpu else a


# ----------------------------------------------------------------------------
# 10. Clusters, labels, reports
# ----------------------------------------------------------------------------
def clusters_of(eig: np.ndarray, tol: float) -> list[list[int]]:
    """Span criterion: a group never spreads more than tol*|first eigenvalue| (no chaining)."""
    if not len(eig):
        return []
    if not np.all(np.isfinite(eig)):
        raise ValueError("Nonfinite eigenvalues cannot be clustered")
    order = np.argsort(eig, kind="stable").tolist()
    out, cur = [], [order[0]]
    for i in order[1:]:
        if abs(eig[i] - eig[cur[0]]) <= tol * max(abs(eig[cur[0]]), 1e-300):
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return out


def label_of(sh: dict, st: Settings) -> str:
    mech = [(k, v) for k, v in sh.items() if k in ("G", "D", "L", "C") and v >= st.mixed_label_pct]
    mech.sort(key=lambda kv: -kv[1])
    return "+".join(k for k, _ in mech) if mech else "UNRESOLVED"


def _assess(sh: dict, sg: dict, st: Settings, cross: float = 0.0) -> dict:
    """Dominant class, margin and confidence from the normalised (sh) and signed (sg) shares.
    confidence (conventions, see header): low  = O too large, or the two measures disagree on
    the dominant class, or margin < conf_low_margin_pct; medium = margin < conf_high_margin_pct
    or |cross term| > max_cross_pct (the class components overlap strongly, so the exact
    percentages - not the dominant class - are uncertain); high otherwise.
    margin = gap between first and second mechanism, smaller of both measures."""
    mk = ("G", "D", "L", "C")
    dom = max(mk, key=lambda k: sh[k])
    dom_s = max(mk, key=lambda k: sg[k])
    a = sorted((sh[k] for k in mk), reverse=True)
    b = sorted((sg[k] for k in mk), reverse=True)
    margin = min(a[0] - a[1], b[0] - b[1])
    flags = []
    if sh["O"] > st.max_other_pct:
        flags.append("UNRELIABLE_OTHER")
    if dom != dom_s:
        flags.append("AMBIGUOUS_DOMINANT")
    strong_cross = bool(np.isfinite(cross) and abs(cross) > st.max_cross_pct)
    if flags or margin < st.conf_low_margin_pct:
        conf = "low"
    elif margin < st.conf_high_margin_pct or strong_cross:
        conf = "medium"
    else:
        conf = "high"
    if strong_cross:
        flags.append("CROSS_TERM")
    # every grade below "high" carries the flag that explains it
    if margin < st.conf_low_margin_pct:
        flags.append("LOW_MARGIN")
    elif margin < st.conf_high_margin_pct:
        flags.append("MIXED")
    return dict(dominant=dom, margin=margin, flags=flags, confidence=conf, label=label_of(sh, st))


def halfwave_length(profile: np.ndarray, z: np.ndarray, share_pct: float, min_share_pct: float = 1.0) -> float:
    """Approximate longitudinal half-wavelength (mm) of one class component of a mode:
    member length / number of humps of its slice profile ||u_M||_W^2(z) (one hump per
    half-wave of a sine-like pattern; humps below 25 % of the highest are ignored).
    NaN when the class carries less than min_share_pct of the mode."""
    if not np.isfinite(share_pct) or share_pct < min_share_pct or profile.max() <= 0:
        return float("nan")
    p = profile / profile.max()
    pk = [i for i in range(len(p)) if p[i] >= 0.25
          and (i == 0 or p[i] >= p[i - 1]) and (i == len(p) - 1 or p[i] > p[i + 1])]
    merged = []
    for i in pk:                                  # a hump is separated from the next by a dip
        if merged and p[merged[-1]:i + 1].min() > 0.5 * min(p[merged[-1]], p[i]):
            if p[i] > p[merged[-1]]:
                merged[-1] = i
            continue
        merged.append(i)
    return float((z.max() - z.min()) / max(len(merged), 1))


def _cluster_basis(gram: np.ndarray) -> np.ndarray:
    """Whiten the in-plane eigenspace after removing arbitrary modal scale.

    Rank decisions on an unscaled Gram matrix can erase valid small-amplitude
    eigenvectors. Equilibration makes the cutoff a correlation/rank test.
    """
    diagonal = np.diag(gram)
    if not np.all(np.isfinite(gram)) or np.any(diagonal <= 0):
        raise ValueError("Cannot classify modes with zero/nonfinite in-plane displacement")
    scale = np.sqrt(diagonal)
    corr = gram / scale[:, None] / scale[None, :]
    ev, V = np.linalg.eigh(0.5 * (corr + corr.T))
    keep = ev > 1e-12 * ev.max()
    return (V[:, keep] / np.sqrt(ev[keep])) / scale[:, None]


def reliable_shares(res, eig: np.ndarray, st: Settings) -> np.ndarray:
    """(n_modes, 5) normalised shares in CLASSES order: the cluster (basis-invariant) result
    for modes in a near-degenerate group, the mode's own result otherwise - i.e. exactly
    the rel_*_pct columns, for any decomposition result `res`."""
    E = res["E"]
    out = 100 * E / np.maximum(E.sum(1, keepdims=True), 1e-300)
    for c in clusters_of(eig, st.cluster_rel_tol):
        if len(c) < 2:
            continue
        idx = np.array(c)
        Gm = res["Htot"][np.ix_(idx, idx)]
        Cm = _cluster_basis(Gm)
        en = np.array([float(np.trace(Cm.T @ res["H"][k][np.ix_(idx, idx)] @ Cm)) for k in CLASSES])
        out[idx] = 100 * en / en.sum()
    return out


def split_sensitivity(groups, mesh, U, st: Settings, eig: np.ndarray, t: float, inp) -> dict:
    """Re-run the decomposition with the other admissible measures of the fold-arc rotation
    (one side of the arc only, either side). A rounded fold is modelled as a rigid arc, but the
    FE arc is not exactly rigid, so its rotation - and thus the virtual-corner translation that
    drives G/C/D - is not unique. The spread of the reliable shares over these variants is
    the uncertainty of the split that comes from the corner kinematics (it is 0 for sharp
    folds and for exactly rigid arcs)."""
    out = {}
    for part in ("first", "second"):
        gv = []
        for g in groups:
            g2 = RingGeom(n=g.n, p=g.p, inst=g.inst, edges=g.edges, rings=g.rings, node_rows=g.node_rows)
            g2.theta_part = part
            detect_reference_points(g2, t, st)
            attach_seams(g2, mesh, inp)
            build_basis(g2)
            gv.append(g2)
        out[part] = reliable_shares(decompose(gv, mesh, U, st), eig, st)
    return out


def summarise(res, modes, eig, sigma_ref, groups, mesh, st: Settings, sens: dict | None = None):
    """Per-mode and per-cluster tables. Every mode row also carries the RELIABLE result:
    the cluster (basis-invariant) result when the mode is in a near-degenerate group, its own
    result otherwise (rel_*_pct, reliable_label, reliable_dominant, confidence)."""
    E, Etot = res["E"], res["Etot"]
    cl = clusters_of(eig, st.cluster_rel_tol)
    cl_id = {i: ci for ci, c in enumerate(cl) for i in c}
    bolt_ring_C = _bolt_ring_check(res, groups, mesh)
    # --- clusters first (needed by the mode rows) ---
    crow, cl_res = [], []
    for ci, c in enumerate(cl):
        idx = np.array(c)
        Gm = res["Htot"][np.ix_(idx, idx)]
        Cm = _cluster_basis(Gm)
        en = {k: float(np.trace(Cm.T @ res["H"][k][np.ix_(idx, idx)] @ Cm)) for k in CLASSES}
        s = sum(en.values())
        sh = {k: 100 * v / s for k, v in en.items()}
        nb = Cm.shape[1]
        sg = {k: 100 * float(np.trace(Cm.T @ res["Hs"][k][np.ix_(idx, idx)] @ Cm)) / nb for k in CLASSES}
        ccross = 100 * (1 - s / nb)
        asx = _assess(sh, sg, st, ccross)
        cl_res.append((sh, sg, asx))
        crow.append(dict(cluster=ci + 1, modes=" ".join(str(modes[i]) for i in c), size=len(c),
                         eig_min=float(eig[idx].min()), eig_max=float(eig[idx].max()),
                         G_pct=sh["G"], D_pct=sh["D"], L_pct=sh["L"], C_pct=sh["C"], O_pct=sh["O"],
                         G_signed_pct=sg["G"], D_signed_pct=sg["D"], L_signed_pct=sg["L"],
                         C_signed_pct=sg["C"], O_signed_pct=sg["O"], cross_term_pct=ccross,
                         D_assembly_pct=sh["D"] + sh["C"],
                         dominant=asx["dominant"], dominance_margin_pct=asx["margin"],
                         label=asx["label"], confidence=asx["confidence"], flags=";".join(asx["flags"])))
    # --- modes ---
    rows = []
    for i, m in enumerate(modes):
        s = E[i].sum()
        sh = {k: 100 * E[i, j] / s for j, k in enumerate(CLASSES)}
        cross = 100 * (1 - s / Etot[i]) if Etot[i] > 0 else float("nan")
        sg = {k: 100 * res["Es"][i, j] / Etot[i] for j, k in enumerate(CLASSES)}
        asx = _assess(sh, sg, st, cross)
        ci = cl_id[i]
        size = len(cl[ci])
        rsh, _, rasx = cl_res[ci] if size > 1 else (sh, sg, asx)
        # flags belong to the RELIABLE result (the cluster result for a near-degenerate group),
        # so that flags and confidence always describe the same numbers
        flags = list(rasx["flags"]) + ([f"CLUSTER_{ci+1}"] if size > 1 else [])
        rows.append(dict(mode=int(m), eigenvalue=float(eig[i]),
                         sigma_cr_MPa=float(eig[i] * sigma_ref) if sigma_ref else None,
                         G_pct=sh["G"], D_pct=sh["D"], L_pct=sh["L"], C_pct=sh["C"], O_pct=sh["O"],
                         cross_term_pct=cross,
                         G_signed_pct=sg["G"], D_signed_pct=sg["D"], L_signed_pct=sg["L"],
                         C_signed_pct=sg["C"], O_signed_pct=sg["O"],
                         D_assembly_pct=sh["D"] + sh["C"],
                         dominant=asx["dominant"], dominance_margin_pct=asx["margin"], label=asx["label"],
                         reliable_source="cluster" if size > 1 else "mode",
                         rel_G_pct=rsh["G"], rel_D_pct=rsh["D"], rel_L_pct=rsh["L"], rel_C_pct=rsh["C"],
                         rel_O_pct=rsh["O"], rel_D_assembly_pct=rsh["D"] + rsh["C"],
                         reliable_dominant=rasx["dominant"],
                         reliable_margin_pct=rasx["margin"], reliable_label=rasx["label"],
                         confidence=rasx["confidence"],
                         C_at_bolt_rings_pct=bolt_ring_C[i] if bolt_ring_C is not None else None,
                         seam_axial_slip_ratio=(float(res["axial_slip_ratio"][i])
                                                if res.get("axial_slip_ratio") is not None else None),
                         fold_motion_ratio=(float(res["fold_ratio"][i])
                                            if res.get("fold_ratio") is not None else None),
                         **{f"halfwave_{k}_mm": halfwave_length(res["prof"][CLASSES.index(k), :, i], mesh.ring_z,
                                                                sh[k]) for k in ("G", "D", "L", "C")},
                         cluster=ci + 1, cluster_size=size, flags=";".join(flags)))
    if sens:
        base = reliable_shares(res, eig, st)
        mk = [CLASSES.index(k) for k in MECH]
        for i, row in enumerate(rows):
            var = np.array([sens[p][i] for p in sens])
            allv = np.vstack([base[i][None], var])
            spread = float(np.abs(var - base[i][None]).max())
            unstable = bool(any(np.argmax(v[mk]) != np.argmax(base[i][mk]) for v in var))
            row["split_spread_pct"] = spread
            for j, k in enumerate(CLASSES):
                if k in MECH:
                    row[f"rel_{k}_min_pct"] = float(allv[:, j].min())
                    row[f"rel_{k}_max_pct"] = float(allv[:, j].max())
            fl = [f for f in row["flags"].split(";") if f]
            if unstable:
                fl.insert(0, "DOMINANT_UNSTABLE")
                row["confidence"] = "low"
            elif spread > st.max_spread_pct:
                fl.insert(0, "DEFINITION_SENSITIVE")
                if row["confidence"] == "high":
                    row["confidence"] = "medium"
            row["flags"] = ";".join(fl)
        for c in crow:                      # a cluster carries the values of its member modes
            r0 = rows[list(modes).index(int(c["modes"].split()[0]))]
            c["split_spread_pct"] = r0["split_spread_pct"]
            c["confidence"] = r0["confidence"]
            c["flags"] = ";".join(f for f in r0["flags"].split(";") if f and not f.startswith("CLUSTER_"))
    return rows, crow


def component_fields(groups, mesh: Mesh, U: np.ndarray) -> np.ndarray:
    """(5, N, 2, m) in-plane GDLC components (CLASSES order) of the modes U (m, N, 3), in the
    (e1, e2) coordinates of the slices - for a member along a global axis these are the
    global in-plane DOFs in increasing order."""
    m, N = U.shape[0], U.shape[1]
    out = np.zeros((len(CLASSES), N, 2, m))
    E12 = np.c_[mesh.e1, mesh.e2]
    for g in groups:
        rows = g.node_rows                                            # (Rg, n)
        Rg, n = rows.shape
        X = np.einsum("mrnc,cd->ndrm", U[:, rows, :], E12, optimize=True).reshape(2 * n, Rg * m)
        for k, cls in enumerate(CLASSES):
            Y = (g.proj[cls] @ X).reshape(n, 2, Rg, m).transpose(2, 0, 1, 3)   # (Rg, n, 2, m)
            out[k][rows] = Y
    return out


def energy_evaluation(run_dir: Path, groups, mesh: Mesh, inp, modes, eig, U, UR, res, st: Settings, kmeta=None) -> dict:
    """Strain-energy evaluation of the GDLC split with the exported elastic stiffness
    (gdlc_stiffness.py): every class component is completed to a full field by minimum strain
    energy, E_fg = w_f^T S w_g. Per mode: E (5 x 5), U = v^T K v of the mode. Per eigenvalue
    group (the RELIABLE result, as for the kinematic shares): Ecl (5 x 5) = trace over an
    orthonormal basis of the group (basis-invariant). Validation numbers included."""
    t0 = time.time()
    _, inplane, _ = _axis_dofs(mesh)
    if not (np.allclose(mesh.e1, np.eye(3)[inplane[0] - 1]) and np.allclose(mesh.e2, np.eye(3)[inplane[1] - 1])):
        raise RuntimeError("slice axes are not the global in-plane axes")
    if UR is None:
        raise RuntimeError("the modal export has no rotations (UR); re-export with --export always")
    if inp is None:
        raise RuntimeError("no .inp (supports and MPCs are read from it)")
    if kmeta is None:
        _, export_meta = choose_export(run_dir, st.export_dir)
        input_path = find_inp(run_dir, export_meta, st.inp)
        kmeta = gst.stiffness_status(run_dir, file_sha256(input_path)) if input_path is not None else None
    if kmeta is None:
        raise RuntimeError('no verified stiffness export for the current input')
    K = gst.load_stiffness(run_dir, mesh.instance, mesh.label, Path(kmeta['export_directory']))
    key_row = {(str(i), int(l)): r for r, (i, l) in enumerate(zip(mesh.instance, mesh.label))}
    model = gst.EnergyModel(K, mesh.xyz, key_row, inp.pairs, inp.boundaries, inplane, N_THREADS)
    nm, N = U.shape[0], U.shape[1]
    V = _mode_array((6 * N, nm))
    block = min(nm, resources.batch_capacity(_available_ram(), 6*N*8*5, nm*nm*8*4))
    cerr = np.empty(nm)
    for start in range(0, nm, block):
        stop = min(nm, start+block)
        V[:, start:stop] = np.concatenate([U[start:stop], UR[start:stop]], axis=2).reshape(stop-start, 6*N).T
        cerr[start:stop] = model.constraint_error(V[:, start:stop])
    if not np.all(cerr <= 1e-4):
        raise RuntimeError(f"the exported modes do not satisfy the MPCs/supports read from the .inp "
                           f"(max relative error {cerr.max():.2e}): stiffness/constraint mapping inconsistent")
    cl = clusters_of(eig, st.cluster_rel_tol)
    korth = gst.k_orthogonality(model, V, cl, block)
    if not np.isfinite(korth["max_normalised_offdiagonal"]) or korth["max_normalised_offdiagonal"] > 1e-2:
        raise RuntimeError("elastic matrix and modes fail K-orthogonality acceptance; energy diagnostic refused")
    nc = len(CLASSES)
    Emode = np.zeros((nm, nc, nc))
    Ecl = np.zeros((nm, nc, nc))                    # group matrix, stored for every member mode
    Umode = np.zeros(nm)
    bolt = np.zeros(nm)
    lift_res = 0.0
    # Account for component fields, prescribed/free RHS, solutions and temporary
    # products, plus the dense (nc*m)^2 Gram matrix. No percentage is reserved.
    per_mode = 8 * (nc * (4 * N + 3 * len(model.P) + 3 * len(model.F)) + 12 * N)
    quad = 8 * nc * nc * 2
    avail = _available_ram()
    budget = min(nm, int(2 * avail / (per_mode + math.sqrt(per_mode**2 + 4 * quad * avail))))
    if budget < 1 or any(len(c) > budget for c in cl):
        raise MemoryError("available RAM cannot hold a complete energy eigenspace working set")
    batches, cur = [], []
    for c in cl:
        if cur and sum(map(len, cur)) + len(c) > budget:
            batches.append(cur)
            cur = []
        cur = cur + [c]
    if cur:
        batches.append(cur)
    t1 = time.time()
    for b in batches:
        idx = np.array([i for c in b for i in c])
        pos = {i: a for a, i in enumerate(idx)}
        comps = component_fields(groups, mesh, U[idx])
        r = gst.component_energies(model, comps, V[:, idx])
        m = len(idx)
        E = r["E"].reshape(nc, m, nc, m)
        for a, i in enumerate(idx):
            Emode[i] = E[:, a, :, a]
        Umode[idx] = r["U"]
        bolt[idx] = r["bolt_incompat"]
        lift_res = max(lift_res, r["lift_residual"])
        for c in b:
            pc = [pos[i] for i in c]
            Cm = _cluster_basis(res["Htot"][np.ix_(c, c)])
            M = np.array([[float(np.trace(Cm.T @ E[f, :, g, :][np.ix_(pc, pc)] @ Cm)) for g in range(nc)]
                          for f in range(nc)])
            Ecl[c] = M
    log(f"  energy evaluation: K_FF {len(model.F)} DOF (factor {model.fill/1e6:.1f} M nonzeros, set-up "
        f"{model.setup_s:.1f} s), {nm} modes x {nc} classes lifted in {time.time()-t1:.1f} s on "
        f"{N_THREADS} threads; mode constraint error {cerr.max():.1e}, max K-coupling of distinct "
        f"modes {korth['max_normalised_offdiagonal']:.1e}")
    acceptance = gst.validate_energy_result(Emode, Umode, lift_res, korth)
    return dict(E=Emode, Ecl=Ecl, U=Umode, bolt=bolt, clusters=cl,
                validation=dict(acceptance=acceptance, mode_constraint_error_max=float(cerr.max()), k_orthogonality=korth,
                                lift_relative_residual_max=lift_res,
                                zero_stiffness_dofs_dropped=model.n_zero_stiffness,
                                n_P=int(len(model.P)), n_F=int(len(model.F)), factor_nnz=model.fill,
                                setup_s=model.setup_s, elapsed_s=time.time() - t0, threads=N_THREADS))


def _energy_shares(M: np.ndarray) -> tuple[dict, dict, float]:
    """(normalised diagonal shares, additive signed shares, cross term) in % of a 5 x 5
    class energy matrix."""
    d = np.diag(M)
    tot = float(M.sum())
    sh = {k: float(100 * d[j] / max(d.sum(), 1e-300)) for j, k in enumerate(CLASSES)}
    sg = {k: (float(100 * M[j].sum() / tot) if tot > 0 else float("nan")) for j, k in enumerate(CLASSES)}
    cross = float(100 * (1 - d.sum() / tot)) if tot > 0 else float("nan")
    return sh, sg, cross


def add_energy_columns(rows, crow, en: dict, modes, st: Settings) -> None:
    """Energy shares next to the kinematic ones (mode_participation / cluster_participation).
    E_*_pct / E_*_signed_pct / E_cross_term_pct: the mode itself; E_rel_*: the RELIABLE (group)
    result. The energy dominant class is resolved when the normalised and the signed energy
    measures agree on it (otherwise flag ENERGY_AMBIGUOUS). A resolved energy dominant class
    that differs from the kinematic reliable_dominant is flagged ENERGY_DISAGREES (confidence
    at most medium: the label depends on the measure). |energy cross term| > max_cross_pct is
    flagged ENERGY_CROSS_TERM (the class components store much more energy separately than
    together; the normalised energy percentages are then mainly a ranking)."""
    for i, row in enumerate(rows):
        row["kinematic_confidence"] = row["confidence"]
        row["energy_confidence"] = "high"
        sh, sg, cross = _energy_shares(en["E"][i])
        for k in CLASSES:
            row[f"E_{k}_pct"] = sh[k]
        for k in CLASSES:
            row[f"E_{k}_signed_pct"] = sg[k]
        row["E_cross_term_pct"] = cross
        tot = float(en["E"][i].sum())
        row["E_beyond_section_pct"] = float(100 * (1 - tot / en["U"][i])) if en["U"][i] > 0 else float("nan")
        row["bolt_incompatibility_pct"] = float(100 * en["bolt"][i])
        rsh, rsg, rcross = _energy_shares(en["Ecl"][i])
        for k in CLASSES:
            row[f"E_rel_{k}_pct"] = rsh[k]
        for k in CLASSES:
            row[f"E_rel_{k}_signed_pct"] = rsg[k]
        row["E_rel_cross_term_pct"] = rcross
        dom = max(MECH, key=lambda k: rsh[k])
        resolved = bool(dom == max(MECH, key=lambda k: rsg[k]))
        row["E_reliable_dominant"] = dom
        row["E_reliable_label"] = label_of(rsh, st)
        row["E_resolved"] = resolved
        fl = [f for f in row["flags"].split(";") if f]
        if np.isfinite(rcross) and abs(rcross) > st.max_cross_pct:
            fl.append("ENERGY_CROSS_TERM")
            row["energy_confidence"] = "medium"
        if not resolved:
            fl.append("ENERGY_AMBIGUOUS")
            row["energy_confidence"] = "low"
        elif dom != row["reliable_dominant"]:
            fl.insert(0, "ENERGY_DISAGREES")
            if row["confidence"] == "high":
                row["confidence"] = "medium"
            row["energy_confidence"] = "medium"
        if row["E_beyond_section_pct"] > st.max_energy_beyond_pct:
            fl.append("ENERGY_BEYOND_SECTION")
            row["energy_confidence"] = "low"
        if row["bolt_incompatibility_pct"] > st.max_bolt_incompat_pct:
            fl.append("ENERGY_INCOMPATIBLE_COMPONENTS")
            row["energy_confidence"] = "low"
        # A combined high-confidence label must not conceal a weak energy diagnostic.
        order = {"low": 0, "medium": 1, "high": 2}
        row["confidence"] = min((row["confidence"], row["energy_confidence"]), key=order.get)
        row["flags"] = ";".join(fl)
    for c in crow:
        members = [rows[list(modes).index(int(m))] for m in c['modes'].split()]
        i = list(modes).index(int(c["modes"].split()[0]))
        for k in CLASSES:
            c[f"E_rel_{k}_pct"] = rows[i][f"E_rel_{k}_pct"]
        c["E_rel_cross_term_pct"] = rows[i]["E_rel_cross_term_pct"]
        c["E_reliable_dominant"] = rows[i]["E_reliable_dominant"]
        c["E_resolved"] = rows[i]["E_resolved"]
        for key in ('confidence', 'kinematic_confidence', 'energy_confidence'):
            c[key] = min((row[key] for row in members), key=order.get)
        c["flags"] = ";".join(sorted({f for row in members for f in row['flags'].split(';')
                                     if f and not f.startswith('CLUSTER_')}))


def _bolt_ring_check(res, groups, mesh):
    """Share of C (relative rigid motion of the pieces) on the fastener rings."""
    zs = []
    for g in groups:
        for s in g.seams:
            zs += s["z"]
    if not zs:
        return None
    tol = 1e-6 * mesh.size
    bolt = np.zeros(len(mesh.ring_z), bool)
    for z in zs:
        bolt |= np.abs(mesh.ring_z - z) < tol * 10
    prof = res["prof"]
    ci = CLASSES.index("C")
    num = (prof[ci][bolt] * res["dz"][bolt, None]).sum(0)
    den = (prof[:-1][:, bolt] * res["dz"][None, bolt, None]).sum((0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return list(np.where(den > 0, 100 * num / den, np.nan))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def _plot_mode(args):
    (path, z, prof_m, mode, eig, row) = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for ci, k in enumerate(CLASSES):
        ax.plot(z, prof_m[ci], color=CLASS_COLORS[k], label=f"{k} {row[k+'_pct']:.1f}%")
    ax.set_xlabel("position along member (mm)")
    ax.set_ylabel("slice deformation  ||u_M||_W^2  (mm^3)")
    ax.set_title(f"Mode {mode}  lambda={eig:.4g}  {row['label']}  {row['flags']}")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_section(path: Path, g: RingGeom):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    for a, b in g.edges:
        ax.plot(*g.p[[a, b]].T, "k-", lw=0.8)
    for f in g.folds:
        ax.plot(*g.p[f["nodes"]].T, "o", color="tab:red", ms=4)
        ax.plot(*g.p[f["ref"]], "s", color="tab:red", ms=7, mfc="none")
    if g.free_ends:
        ax.plot(*g.p[g.free_ends].T, "^", color="tab:blue", ms=7, label="free end")
    for s in g.seams:
        ax.plot(*g.p[[s["A"], s["B"]]].T, "-", color="tab:green", lw=2.5)
    for a in g.ambiguous:
        ax.plot(*g.p[a["node"]], "x", color="tab:orange", ms=8)
    ax.plot([], [], "s", color="tab:red", mfc="none", label="fold (rigid arc, ref node)")
    ax.plot([], [], "-", color="tab:green", lw=2.5, label="seam link")
    ax.plot([], [], "x", color="tab:orange", label="near fold threshold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"Reference points  (n={g.n}, folds={len(g.folds)}, seams={len(g.seams)})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary(path: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = np.array([r["mode"] for r in rows])
    fig, ax = plt.subplots(figsize=(max(8, len(m) * 0.06), 4.5))
    bottom = np.zeros(len(m))
    for k in CLASSES:
        v = np.array([_rel(r, k) for r in rows])
        ax.bar(m, v, bottom=bottom, width=0.9, label=CLASS_NAMES[k], color=CLASS_COLORS[k])
        bottom += v
    ax.set_xlabel("mode")
    ax.set_ylabel("reliable share of deformation (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)



# ----------------------------------------------------------------------------
# 10b. Analytical benchmark: in-plane D-space dimensions vs cFSM/GBT
# ----------------------------------------------------------------------------
def _bench_ring(pts, closed):
    n = len(pts)
    e = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if closed else [])
    return RingGeom(n=n, p=np.array(pts, float), inst=np.array(["P"] * n), edges=np.array(e),
                    rings=[0], node_rows=np.zeros((1, n), int))


def _bench_rounded(C, R, nw=8, na=4, closed=True):
    C = np.array(C, float)
    m = len(C)
    seq = []
    for k in (range(m) if closed else range(1, m - 1)):
        a, b, c = C[k - 1], C[k], C[(k + 1) % m]
        u = (a - b) / np.linalg.norm(a - b)
        v = (c - b) / np.linalg.norm(c - b)
        ang = math.acos(np.clip(u @ v, -1, 1))
        if R <= 0:
            seq.append([b])
            continue
        d = R / math.tan(ang / 2)
        bis = (u + v) / np.linalg.norm(u + v)
        ctr = b + bis * R / math.sin(ang / 2)
        p1, p2 = b + u * d, b + v * d
        a1 = math.atan2(*(p1 - ctr)[::-1])
        a2 = math.atan2(*(p2 - ctr)[::-1])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        seq.append([ctr + R * np.array([math.cos(a1 + da * t), math.sin(a1 + da * t)])
                    for t in np.linspace(0, 1, na + 1)])
    out = [] if closed else [C[0]]
    for arc in seq:
        if out:
            p0, p1 = out[-1], arc[0]
            out += [p0 + (p1 - p0) * t for t in np.linspace(0, 1, nw + 1)[1:-1]]
        out += arc
    p0, p1 = out[-1], (out[0] if closed else C[-1])
    out += [p0 + (p1 - p0) * t for t in np.linspace(0, 1, nw + 1)[1:(-1 if closed else None)]]
    return out


def run_benchmarks(st: Settings) -> list[dict]:
    """In-plane distortional-space dimension must match cFSM/GBT for classic sections."""
    lip = [(80, 180), (80, 200), (0, 200), (0, 0), (80, 0), (80, 20)]
    cases = [("lipped channel, rounded corners", _bench_rounded(lip, 5, closed=False), False, 2),
             ("lipped channel, sharp corners", _bench_rounded(lip, 0, closed=False), False, 2),
             ("plain channel, rounded corners", _bench_rounded([(80, 200), (0, 200), (0, 0), (80, 0)], 5,
                                                               closed=False), False, 0),
             ("square tube, rounded corners", _bench_rounded([(0, 0), (200, 0), (200, 200), (0, 200)], 8),
              True, 1),
             ("square tube, sharp corners", _bench_rounded([(0, 0), (200, 0), (200, 200), (0, 200)], 0),
              True, 1)]
    out = []
    for name, pts, closed, expect in cases:
        g = _bench_ring(pts, closed)
        detect_reference_points(g, 2.0, st)
        build_basis(g)
        te = self_test(g)
        out.append(dict(case=name, D_expected=expect, D_found=g.basis["sizes"]["D"],
                        G=g.basis["sizes"]["G"], selftest=te["passed"],
                        passed=bool(g.basis["sizes"]["D"] == expect and te["passed"])))
    out.append(_bench_built_up(st, lip))
    out.append(_bench_mesh_convergence(st, lip))
    return out


def _shares(g: RingGeom, x: np.ndarray) -> dict:
    """Normalised W-shares (%) of one ring field x (2n) - the per-ring version of the run measure."""
    w2 = g.basis["w2"]
    e = {k: float((w2 * (g.proj[k] @ x) ** 2).sum()) for k in CLASSES}
    s = sum(e.values())
    return {k: 100 * v / s for k, v in e.items()}


def _bench_built_up(st: Settings, lip) -> dict:
    """Two lipped channels side by side (two pieces, no wall connection): exact pure-mode tests.
    D = 2 x 2 (per-piece cFSM count), C = 3 (relative rigid motion), G = 3, and:
      whole-section rigid motion -> 100 % G;  opposite rigid motion of the pieces -> 100 % C;
      any field with all folds fixed (random, incl. extension) -> exactly 0 % G, C and D."""
    a = np.array(_bench_rounded(lip, 5, closed=False))
    b = a * np.array([-1.0, 1.0]) + np.array([-10.0, 0.0])          # mirrored piece, 10 mm gap
    n1 = len(a)
    pts = np.vstack([a, b])
    e = [(i, i + 1) for i in range(n1 - 1)] + [(n1 + i, n1 + i + 1) for i in range(len(b) - 1)]
    g = RingGeom(n=len(pts), p=pts, inst=np.array(["P"] * n1 + ["Q"] * len(b)), edges=np.array(e),
                 rings=[0], node_rows=np.zeros((1, len(pts)), int))
    detect_reference_points(g, 2.0, st)
    build_basis(g)
    te = self_test(g)
    sz = g.basis["sizes"]
    rig = np.zeros(2 * g.n); rig[0::2] = 1.0
    rel = np.zeros(2 * g.n); rel[0:2 * n1:2] = 1.0; rel[2 * n1::2] = -1.0
    Qf = build_operators(g)[2]
    rng = np.random.default_rng(7)
    fixed = null_space(_unit_rows(Qf)) @ rng.standard_normal(2 * g.n - np.linalg.matrix_rank(Qf))
    s_rig, s_rel, s_fix = _shares(g, rig), _shares(g, rel), _shares(g, fixed)
    ok = (sz["D"] == 4 and sz["C"] == 3 and sz["G"] == 3 and te["passed"]
          and abs(s_rig["G"] - 100) < 1e-8 and abs(s_rel["C"] - 100) < 1e-8
          and s_fix["G"] + s_fix["C"] + s_fix["D"] < 1e-8)
    return dict(case="two-piece built-up, rounded corners: pure G / pure C / folds fixed => no G,C,D",
                D_expected=4, D_found=sz["D"], C_found=sz["C"], G=sz["G"], selftest=te["passed"],
                rigid_G_pct=s_rig["G"], relative_C_pct=s_rel["C"],
                folds_fixed_GCD_pct=s_fix["G"] + s_fix["C"] + s_fix["D"], passed=bool(ok))


def _bench_mesh_convergence(st: Settings, lip) -> dict:
    """The same smooth field on a coarse and a 4x finer mesh of a rounded lipped channel must
    give the same shares (the W-norm is a consistent quadrature of the mid-line integral)."""
    res = []
    for nw, na in ((16, 6), (64, 16)):
        pts = np.array(_bench_rounded(lip, 5, nw=nw, na=na, closed=False))
        g = _bench_ring(pts, False)
        detect_reference_points(g, 2.0, st)
        build_basis(g)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.r_[0.0, np.cumsum(seg)] / seg.sum()
        tng = np.gradient(pts, axis=0)
        tng /= np.linalg.norm(tng, axis=1, keepdims=True)
        nrm = np.c_[-tng[:, 1], tng[:, 0]]
        w = np.sin(2 * np.pi * s) + 0.6 * np.cos(5 * np.pi * s) + 0.3      # smooth mid-line field
        x = (w[:, None] * nrm + 0.2 * np.array([1.0, 0.5])).reshape(-1)
        res.append(_shares(g, x))
    diff = max(abs(res[0][k] - res[1][k]) for k in CLASSES)
    return dict(case="mesh convergence (coarse vs 4x finer lipped channel, same smooth field)",
                D_expected=None, D_found=None, max_share_difference_pct=diff,
                coarse=res[0], fine=res[1], passed=bool(diff < 1.5))


# ----------------------------------------------------------------------------
# 10c. Mechanism statistics, critical stresses and cross-section figures
# ----------------------------------------------------------------------------
MECH = ("G", "D", "L", "C")
CLASS_COLORS = {"G": "#1f77b4", "D": "#d62728", "L": "#2ca02c", "C": "#9467bd", "O": "#7f7f7f"}


def _rel(r: dict, k: str) -> float:
    v = r.get("rel_" + k + "_pct")
    return r[k + "_pct"] if v is None else v


def _reldom(r: dict) -> str:
    return r.get("reliable_dominant") or r["dominant"]


def confidence_by_measure(rows) -> dict:
    """Keep geometric classification confidence distinct from the auxiliary energy check."""
    def counts(key, fallback=None):
        return {c: sum(r.get(key, r.get(fallback)) == c for r in rows)
                for c in ("high", "medium", "low")}
    return dict(combined=counts("confidence"),
                kinematic=counts("kinematic_confidence", "confidence"),
                energy=counts("energy_confidence")
                if any(r.get("energy_confidence") is not None for r in rows) else None)


def dominance_stats(rows) -> dict:
    """Dominant mechanism of a set of modes = highest mean RELIABLE share (cluster result for
    near-degenerate modes); ties broken by the number of modes it dominates. Counts are also
    given for high-confidence modes only."""
    mean = {k: float(np.mean([_rel(r, k) for r in rows])) for k in MECH}
    count = {k: int(sum(_reldom(r) == k for r in rows)) for k in MECH}
    count_hi = {k: int(sum(_reldom(r) == k and r.get("confidence") == "high" for r in rows)) for k in MECH}
    conf = confidence_by_measure(rows)
    dom = max(MECH, key=lambda k: (mean[k], count[k]))
    return dict(mean_pct=mean, n_dominant=count, n_dominant_high_confidence=count_hi,
                confidence_counts=conf["combined"], confidence_by_measure=conf,
                dominant_mechanism=dom, n_modes=len(rows))


def select_top_modes(rows, mech: str, k: int) -> list[int]:
    """Indices of the k modes with the largest share of `mech`; symmetric twins
    (identical eigenvalue and share) are shown once."""
    order = sorted(range(len(rows)), key=lambda i: -_rel(rows[i], mech))
    sel = []
    for i in order:
        # members of one near-degenerate group share one (group) result: show the group once
        twin = any((rows[i].get("cluster") is not None and rows[i].get("cluster") == rows[j].get("cluster")
                    and (rows[i].get("cluster_size") or 1) > 1)
                   or (abs(rows[i]["eigenvalue"] - rows[j]["eigenvalue"]) <= 1e-6 * abs(rows[j]["eigenvalue"])
                       and abs(_rel(rows[i], mech) - _rel(rows[j], mech)) < 1e-3) for j in sel)
        if not twin:
            sel.append(i)
        if len(sel) >= k:
            break
    return sel


def mechanism_critical(rows) -> list[dict]:
    """Per mechanism: lowest eigenvalue at which it governs with confidence high/medium
    (reliable = cluster-aware result), the lowest with high confidence only, and the lowest at
    which its reliable share first reaches 50 %. These are critical values of the MIXED
    Abaqus modes, not pure-mode (constrained) values - use the REF_G / REF_L models for those."""
    out = []
    for k in MECH:
        ok = [r for r in rows if _reldom(r) == k and r.get("confidence", "medium") != "low"]
        hi = [r for r in rows if _reldom(r) == k and r.get("confidence") == "high"]
        h50 = [r for r in rows if _rel(r, k) >= 50.0]
        f = min(ok, key=lambda r: r["eigenvalue"]) if ok else None
        fh = min(hi, key=lambda r: r["eigenvalue"]) if hi else None
        f50 = min(h50, key=lambda r: r["eigenvalue"]) if h50 else None
        out.append(dict(mechanism=k,
                        first_dominant_mode=f["mode"] if f else None,
                        first_dominant_eigenvalue=f["eigenvalue"] if f else None,
                        first_dominant_sigma_cr_MPa=f["sigma_cr_MPa"] if f else None,
                        first_dominant_share_pct=_rel(f, k) if f else None,
                        first_dominant_confidence=f.get("confidence") if f else None,
                        first_high_confidence_mode=fh["mode"] if fh else None,
                        first_high_confidence_eigenvalue=fh["eigenvalue"] if fh else None,
                        first_share50_mode=f50["mode"] if f50 else None,
                        first_share50_eigenvalue=f50["eigenvalue"] if f50 else None,
                        max_share_pct=max(_rel(r, k) for r in rows),
                        mean_share_pct=float(np.mean([_rel(r, k) for r in rows])),
                        n_modes_dominant=int(sum(_reldom(r) == k for r in rows))))
    return out


def ring_inplane(U_mode: np.ndarray, g: RingGeom, mesh: Mesh) -> np.ndarray:
    """(n_rings, n_nodes, 2) in-plane translations of one mode for one ring group."""
    return U_mode[g.node_rows] @ np.c_[mesh.e1, mesh.e2]


def _plot_mode_sections(job: dict):
    """Deformed cross-section of one mode (bold = ring with largest deformation),
    every other cross-section along the member overlaid faintly, the undeformed
    section dashed; same panels for the G, D, L, C components; longitudinal
    profile underneath. One common displacement scale for all panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    p, edges, X, P = job["p"], job["edges"], job["X"].astype(float), job["proj"]
    Rg, n = X.shape[:2]
    comps = {k: (P[k].astype(float) @ X.reshape(Rg, 2 * n).T).T.reshape(Rg, n, 2) for k in MECH}
    amp = float(np.sqrt((X ** 2).sum(-1)).max())
    size = float(np.ptp(p, axis=0).max())
    sc = 0.08 * size / max(amp, 1e-300)
    rmax = int(np.argmax((X ** 2).sum((1, 2))))
    alpha = float(np.clip(6.0 / max(Rg, 1), 0.03, 0.25))
    row = job["row"]
    fig = plt.figure(figsize=(23, 7.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[2.6, 1.0], hspace=0.22, wspace=0.06,
                          left=0.03, right=0.99, top=0.88, bottom=0.08)
    und = p[edges]
    panels = [("mode", X, "black")] + [(k, comps[k], CLASS_COLORS[k]) for k in MECH]
    for j, (name, F, col) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        ax.add_collection(LineCollection(und, colors="0.6", linewidths=0.8, linestyles="--"))
        Dd = p[None] + sc * F
        ax.add_collection(LineCollection(Dd[:, edges].reshape(-1, 2, 2), colors=col,
                                         linewidths=0.5, alpha=alpha))
        ax.add_collection(LineCollection(Dd[rmax][edges], colors=col, linewidths=1.9))
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xticks([])
        ax.set_yticks([])
        if name == "mode":
            rl = row.get("reliable_label") or row["label"]
            spr = row.get("split_spread_pct")
            ax.set_title(f"Mode {row['mode']}   reliable: {rl} ({row.get('reliable_source', 'mode')})\n"
                         f"confidence {row.get('confidence', '-')}"
                         + (f", split spread {spr:.1f} %-pts" if spr is not None and np.isfinite(spr) else "")
                         + (f", fold motion {row['fold_motion_ratio']:.3f}" if row.get("fold_motion_ratio") is not None
                            else ""), fontsize=10)
        else:
            sp = row.get(name + "_signed_pct")
            if row.get("reliable_source") == "cluster":
                # a mode of a near-degenerate group: only the group result is basis-invariant
                txt = (f"{CLASS_NAMES[name]}: {_rel(row, name):.1f} % (group)\n"
                       f"this mode alone {row[name + '_pct']:.1f} %")
            else:
                txt = (f"{CLASS_NAMES[name]}: {row[name + '_pct']:.1f} %\n" +
                       (f"(signed {sp:.1f} %)" if sp is not None else ""))
            lo, hi = row.get(f"rel_{name}_min_pct"), row.get(f"rel_{name}_max_pct")
            if lo is not None and hi is not None and hi - lo >= 0.05:
                txt += f"  range {lo:.1f}-{hi:.1f}"
            ax.set_title(txt, fontsize=10, color=col)
    ax = fig.add_subplot(gs[1, :])
    for ci, k in enumerate(CLASSES):
        ax.plot(job["z"], job["prof"][ci], color=CLASS_COLORS[k], lw=1.4, label=CLASS_NAMES[k])
    for zb in job["bolt_z"]:
        ax.axvline(zb, color="0.85", lw=0.6, zorder=0)
    ax.axvline(job["z_ring"][rmax], color="k", ls=":", lw=1.2, label="bold section")
    ax.set_xlabel("position along member (mm)   [thin grey lines: connector rows]")
    ax.set_ylabel(r"$\|u_M\|_W^2$ per section")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    sig = f"   sigma_cr = {row['sigma_cr_MPa']:.2f} MPa" if row.get("sigma_cr_MPa") else ""
    fig.suptitle(f"{job['title']}\nmode {row['mode']}   lambda = {row['eigenvalue']:.4g}{sig}   "
                 f"flags: {row['flags'] or '-'}   |   displacement scale x{sc:.3g}; "
                 f"bold: section at z = {job['z_ring'][rmax]:.0f} mm (largest deformation); "
                 f"faint: all {Rg} sections; dashed: undeformed", fontsize=11)
    fig.savefig(job["path"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    return job["path"]


def plot_dominance(path: Path, rows, stats: dict, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 2.2]))
    ks = list(MECH)
    axs[0].bar(ks, [stats["mean_pct"][k] for k in ks], color=[CLASS_COLORS[k] for k in ks])
    axs[0].set_title("mean share over all modes (%)")
    axs[1].bar(ks, [stats["n_dominant"][k] for k in ks], color=[CLASS_COLORS[k] for k in ks])
    axs[1].set_title("number of modes in which it dominates")
    lam = np.array([r["eigenvalue"] for r in rows])
    for k in ks:
        axs[2].scatter(lam, [_rel(r, k) for r in rows], s=10, color=CLASS_COLORS[k], label=CLASS_NAMES[k],
                       alpha=0.8)
    axs[2].set_xlabel("eigenvalue")
    axs[2].set_ylabel("reliable share (%)")
    axs[2].set_title("share of each mechanism along the spectrum")
    axs[2].legend(ncol=2, fontsize=8)
    fig.suptitle(f"{title}   -   dominant mechanism: {stats['dominant_mechanism']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_mode_figures(out: Path, run_name: str, rows, modes, U, groups, mesh, res, st: Settings,
                      dominant: str) -> dict:
    """Figures: top-k modes of the dominant mechanism, the top mode of every other mechanism
    (gallery) and mode 1. In-plane data of the top-k modes of EVERY mechanism are stored
    (figure_data.npz, float32) so the cross-run pass can redraw them without the shards.
    Cost: a few small projections and <= k + 4 figures, drawn in parallel."""
    g = max(groups, key=lambda gg: len(gg.rings))
    z_ring = mesh.ring_z[g.rings]
    bolt_z = sorted({round(z, 6) for s in g.seams for z in s["z"]})
    proj32 = {k: g.proj[k].astype(np.float32) for k in MECH}
    store, plot = set([0]), {0: ["first"]}
    for mech in MECH:
        top = select_top_modes(rows, mech, st.top_modes)
        store.update(top)
        for i in (top if mech == dominant else top[:1]):
            plot.setdefault(i, []).append(mech)
    saved = {int(modes[i]): ring_inplane(U[i], g, mesh).astype(np.float32) for i in sorted(store)}
    jobs = []
    for i, why in sorted(plot.items()):
        for w in why:
            if w == "first":
                sub, lab = "figures_first_mode", "first mode"
            elif w == dominant:
                sub, lab = f"figures_dominant_{w}", f"top {w} mode (dominant mechanism)"
            else:
                sub, lab = "figures_gallery", f"top {w} mode"
            (out / sub).mkdir(exist_ok=True)
            name = f"mode_{modes[i]:04d}.png" if w in ("first", dominant) else f"top_{w}_mode_{modes[i]:04d}.png"
            jobs.append(dict(path=str(out / sub / name), p=g.p, edges=g.edges, X=saved[int(modes[i])],
                             proj=proj32, row=rows[i], z=mesh.ring_z, prof=res["prof"][:-1, :, i],
                             z_ring=z_ring, bolt_z=bolt_z, title=f"{run_name}   [{lab}]"))
    if jobs and st.plots != "none":
        with _process_pool(len(jobs)) as ex:
            list(ex.map(_plot_mode_sections, jobs))
    ms = sorted(saved)
    np.savez_compressed(out / "figure_data.npz", p=g.p, edges=g.edges, z_ring=z_ring, z=mesh.ring_z,
                        bolt_z=np.array(bolt_z), modes=np.array(ms), X=np.stack([saved[m] for m in ms]),
                        prof=np.stack([res["prof"][:-1, :, modes.index(m)] for m in ms]),
                        **{f"P_{k}": proj32[k] for k in MECH})
    return dict(n_figures=len(jobs), stored_modes=ms)


def fold_detection_stability(groups, t: float, st: Settings) -> dict:
    """Re-detect folds with the radius threshold changed by -20 % and +25 %; the fold
    set must not change (otherwise the D/L split depends on the threshold)."""
    import copy
    res = {}
    for fac in (0.8, 1.25):
        st2 = copy.copy(st)
        st2.fold_radius_factor = st.fold_radius_factor * fac
        same = True
        for g in groups:
            g2 = RingGeom(n=g.n, p=g.p, inst=g.inst, edges=g.edges, rings=g.rings, node_rows=g.node_rows)
            detect_reference_points(g2, t, st2)
            a = sorted(tuple(f["nodes"]) for f in g.folds)
            b = sorted(tuple(f["nodes"]) for f in g2.folds)
            same &= (a == b)
        res[f"factor_x{fac}"] = bool(same)
    res["stable"] = all(res.values())
    return res

# ----------------------------------------------------------------------------
# 11. One run
# ----------------------------------------------------------------------------
def report_flag_counts(rows) -> dict:
    """Number of modes carrying each exact diagnostic token, excluding group identifiers."""
    counts = dict.fromkeys(("UNRELIABLE_OTHER", "AMBIGUOUS_DOMINANT", "CROSS_TERM", "LOW_MARGIN",
                            "MIXED", "DEFINITION_SENSITIVE", "DOMINANT_UNSTABLE", "ENERGY_DISAGREES",
                            "ENERGY_AMBIGUOUS", "ENERGY_CROSS_TERM", "ENERGY_BEYOND_SECTION",
                            "ENERGY_INCOMPATIBLE_COMPONENTS"), 0)
    for row in rows:
        for flag in sorted(set(row.get("flags", "").split(";"))):
            if flag and not flag.startswith("CLUSTER_"):
                counts[flag] = counts.get(flag, 0) + 1
    return counts


def reliability_summary(rows) -> dict:
    """Run-level indicators of how far the reported shares can be trusted."""
    def q(key):
        v = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])], float)
        return None if not v.size else dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                                             max=float(v.max()))
    conf = confidence_by_measure(rows)
    out = dict(n_modes=len(rows), confidence=conf["combined"], confidence_by_measure=conf,
               split_spread_pct=q("split_spread_pct"), abs_cross_term_pct=None, O_pct=q("O_pct"),
               fold_motion_ratio=q("fold_motion_ratio"))
    xt = np.array([abs(r["cross_term_pct"]) for r in rows if np.isfinite(r["cross_term_pct"])])
    if xt.size:
        out["abs_cross_term_pct"] = dict(median=float(np.median(xt)), p90=float(np.percentile(xt, 90)),
                                         max=float(xt.max()))
    out["note"] = ("split_spread_pct = largest change of a reliable share when the fold-arc rotation is "
                   "measured on either side of the arc; abs_cross_term_pct = overlap of the class "
                   "components (normalised and signed shares allot it differently). "
                   "confidence is the combined grade; confidence_by_measure separates kinematic "
                   "classification from auxiliary energy evaluation (null when unavailable).")
    return out


def process_run(run_dir: Path, st: Settings) -> dict:
    t0 = time.time()
    log(f"RUN {run_dir}")
    exp_dir, meta = choose_export(run_dir, st.export_dir)
    log(f"  export: {exp_dir.name}  ({meta.get('mode_count')} modes, {export_format(meta)} format)")
    inp_path = find_inp(run_dir, meta, st.inp)
    if export_format(meta) == "npz":
        if inp_path is None or not inp_path.is_file():
            raise FileNotFoundError("npz export: the run's .inp (element connectivity) was not found")
    if inp_path and inp_path.is_file() and meta.get("source_inp_sha256"):
        if file_sha256(inp_path) != meta["source_inp_sha256"]:
            raise RuntimeError(f"{inp_path.name} is not the .inp this export was made from (sha256 differs)")
    inp = read_inp(inp_path) if inp_path and inp_path.is_file() else None
    t = st.thickness
    if t is None and inp and inp.thicknesses:
        t = min(inp.thicknesses)
    if t is None:
        bj = next(iter(run_dir.glob("*_build.json")), None)
        if bj:
            t = json.loads(bj.read_text(encoding="utf-8")).get("source_inputs", {}).get("thickness_mm")
    if t is None:
        raise RuntimeError("wall thickness unknown: pass --thickness")
    sigma_ref = None
    load_case = {"type": "compression"}
    bj = next(iter(run_dir.glob("*_build.json")), None)
    if bj:
        bjd = json.loads(bj.read_text(encoding="utf-8"))
        sigma_ref = bjd.get("reference_stress_MPa")
        load_case = bjd.get("load_case") or load_case
    m_ref = (load_case.get("reference_moment_mesh_Nmm_per_MPa")
             if load_case.get("type") in ("bending", "combined") else None)
    p_ref = (load_case.get("reference_axial_force_mesh_N_per_MPa")
             if load_case.get("type") == "combined" else None)
    if m_ref is not None:
        log(f"  load case: {load_case['type']}, neutral axis {load_case.get('neutral_axis_angle_deg')} deg"
            + (f", P-M ray {load_case.get('pm_angle_deg')} deg" if p_ref is not None else "")
            + f"; eigenvalue = extreme-fibre stress (MPa), M_cr = eigenvalue x {m_ref:.6g} N mm"
            + (f", P_cr = eigenvalue x {p_ref:.6g} N" if p_ref is not None else ""))

    import importlib.util
    if st.plots != "none" and importlib.util.find_spec("matplotlib") is None:
        log("  matplotlib not installed: plots skipped")
        st.plots = "none"
    mesh = load_mesh(exp_dir, meta, inp_path)
    if mesh.geometry.get('source') == 'inp_verified_against_export':
        log('  geometry: INP coordinates verified against export; '
            f"{mesh.geometry['fp32_rounded_coordinate_count']} FP32-rounded coordinate values recovered")
    groups = build_ring_groups(mesh)
    log(f"  mesh: {len(mesh.node_index)} nodes, {len(mesh.ring_z)} rings, "
        f"{len(groups)} distinct ring geometr{'y' if len(groups)==1 else 'ies'}; t={t}")
    out = run_dir / st.out_name
    out.mkdir(exist_ok=True)

    def _prep(g):
        detect_reference_points(g, t, st)
        attach_seams(g, mesh, inp)
        build_basis(g)
        return self_test(g)

    with cf.ThreadPoolExecutor(max_workers=max(1, min(N_THREADS, len(groups)))) as ex:
        tests = list(ex.map(_prep, groups))
    for g, te in zip(groups, tests):
        if not te["passed"]:
            raise RuntimeError(f"self-test failed: {te}; geometry={mesh.geometry}. "
                               "Projection is not reliable; no modal classification was accepted.")
    fold_stab = fold_detection_stability(groups, t, st)
    if not fold_stab["stable"]:
        log(f"  WARNING: fold detection changes when the radius threshold is varied: {fold_stab}")
    amb = [(gi, fi, round(f["arc_inside_min_ratio"], 2), round(f["arc_outside_max_ratio"], 2))
           for gi, g in enumerate(groups) for fi, f in enumerate(g.folds)
           if "arc_inside_min_ratio" in f and (f["arc_inside_min_ratio"] < 0.35 or f["arc_outside_max_ratio"] > 0.15)]
    fold_stab["arc_extent_close_to_rule"] = amb
    if amb:
        log(f"  NOTE: {len(amb)} fold arc(s) whose end nodes lie within 0.1 of the 25 %-of-peak arc rule "
            f"(group, fold, inside min, outside max): {amb[:6]} - the arc extent fixes the virtual corner; "
            "check reference_points.png")
    bench = run_benchmarks(st)
    if not all(b_["passed"] for b_ in bench):
        raise RuntimeError(f"analytical benchmark failed: {bench}")
    g0 = groups[0]
    log(f"  basis: sizes {g0.basis['sizes']}, folds {len(g0.folds)}, free ends {len(g0.free_ends)}, "
        f"seams {len(g0.seams)}, cond {g0.basis['cond']:.3g}, self-test max err "
        f"{max(te['max_rel_error'] for te in tests):.2e}")
    if st.plots != "none":
        plot_section(out / "reference_points.png", g0)

    report = dict(
        program="gdlc_mode_decomposition.py", version=VERSION, run_dir=str(run_dir),
        export_dir=str(exp_dir), inp=str(inp_path) if inp_path else None,
        source_odb_sha256=meta.get("source_odb_sha256"), source_inp_sha256=meta.get("source_inp_sha256"),
        settings={k: v for k, v in st.__dict__.items()},
        thickness_mm=t, member_axis=mesh.axis.tolist(), rings=len(mesh.ring_z),
        mesh_geometry=mesh.geometry,
        load_case={k: v for k, v in load_case.items() if k not in ("section_symmetry",)},
        ring_groups=[dict(n_rings=len(g.rings), n_nodes=g.n, sizes=g.basis["sizes"],
                          dim_inextensional=g.basis["dim_inextensional"], cond=g.basis["cond"],
                          rigid_residual=g.basis["rigid_residual"], notes=g.basis["notes"],
                          folds=g.folds, free_ends=g.free_ends, junctions=g.junctions,
                          seams=[{k: v for k, v in s.items() if k not in ("z", "axial_rows")}
                                 | {"n_rings_linked": len(set(np.round(s['z'], 6)))} for s in g.seams],
                          near_threshold_nodes=g.ambiguous) for g in groups],
        self_test=tests,
        analytical_benchmarks=bench,
        fold_detection_stability=fold_stab,
        inp_notes=inp.notes if inp else ["no .inp found: seam diagnostics unavailable"],
        methodological_scope="Project-specific in-plane kinematic split (classes); with stiffness_export the "
                             "class components are also evaluated in strain energy with the real elastic K "
                             "(minimum-energy completion; not an energy-orthogonal split). Fold-arc extent is "
                             "heuristic; internal benchmarks and constrained references are not independent "
                             "validation.",
        percentage_definition="share_M = sum_r dz_r ||phi_M,r||_W^2 / sum_N(...); in-plane translations; "
                              "share of cross-sectional deformation (not energy, not load)",
        resources=dict(threads=N_THREADS, logical_cpus=os.cpu_count(),
                       ram_GB=(psutil.virtual_memory().total / 1e9) if psutil else None),
    )
    if st.selftest_only:
        (out / "basis_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log(f"  self-test only: wrote {out/'basis_report.json'}")
        return report

    # elastic stiffness for the energy evaluation (exported after the buckling run, see
    # ensure_stiffness); it needs the rotations of the modes as well
    kmeta, energy_note = None, None
    if st.stiffness == "never":
        energy_note = "disabled (--stiffness never)"
    elif (run_dir / "gdlc_reference.json").is_file():
        energy_note = "reference model: constraints (*Equation) not part of the energy evaluation"
    elif inp_path is None or inp is None:
        energy_note = "no .inp"
    else:
        kmeta = gst.stiffness_status(run_dir, file_sha256(inp_path))
        if kmeta is None:
            energy_note = (f"no {gst.STIFFNESS_DIR}/ for the current .inp (run with --stiffness auto, which "
                           "writes and solves the matrix-generation job)")
    if energy_note and st.stiffness == "required" and not (run_dir / "gdlc_reference.json").is_file():
        raise RuntimeError(f"--stiffness required: {energy_note}")
    loaded = load_modes(exp_dir, meta, mesh, st, rotations=kmeta is not None)
    modes, U = loaded[0], loaded[1]
    UR = loaded[2] if kmeta is not None else None
    eig_map = {m["mode"]: m["eigenvalue"] for m in meta["modes"]}
    eig = np.array([eig_map[m] for m in modes], dtype=float)
    if not np.all(np.isfinite(eig)):
        raise ValueError("Missing/nonfinite eigenvalues in modal metadata")
    keep = eig > 0
    if not keep.all():
        # A negative eigenvalue buckles under the REVERSED load (e.g. the opposite moment when a
        # bending reference model fell back to subspace iteration): a different load case.
        log(f"  {int((~keep).sum())} mode(s) with eigenvalue <= 0 (reversed load) excluded")
        report["excluded_nonpositive_modes"] = [int(m) for m, k in zip(modes, keep) if not k]
        modes = [m for m, k in zip(modes, keep) if k]
        U, eig = U[keep], eig[keep]
        if UR is not None:
            UR = UR[keep]
        if not modes:
            raise RuntimeError("no mode with a positive eigenvalue")
    t1 = time.time()
    try:
        res = decompose(groups, mesh, U, st)
    except Exception as e:
        if st.gpu == "off":
            raise
        log(f"  GPU path failed ({str(e).splitlines()[0][:160]}); repeating on CPU x{N_THREADS}")
        st.gpu = "off"
        res = decompose(groups, mesh, U, st)
    log(f"  projection: {time.time()-t1:.1f} s on {'GPU' if res['gpu'] else f'CPU x{N_THREADS}'}")
    sens = None
    if st.sensitivity:
        t1 = time.time()
        sens = split_sensitivity(groups, mesh, U, st, eig, t, inp)
        log(f"  split sensitivity (fold-arc rotation from either arc side): {time.time()-t1:.1f} s")
    rows, crow = summarise(res, modes, eig, sigma_ref, groups, mesh, st, sens)
    if kmeta is not None:
        try:
            en = energy_evaluation(run_dir, groups, mesh, inp, modes, eig, U, UR, res, st, kmeta)
            add_energy_columns(rows, crow, en, modes, st)
            np.savez_compressed(out / "mode_energy.npz", modes=np.array(modes), eigenvalue=eig,
                                classes=np.array(CLASSES), E=en["E"], E_group=en["Ecl"], U=en["U"],
                                bolt_incompatibility=en["bolt"],
                                layout="E[i, f, g] = w_f^T S w_g of mode i (in-plane class components f, g, "
                                       "minimum-energy completion; units of the exported mode scaling); "
                                       "E_group[i] = the same traced over the eigenvalue group of mode i "
                                       "(basis-invariant); U[i] = v^T K v of the full mode (U and UR); "
                                       "factor 1/2 omitted everywhere")
            emean = {k: float(np.mean([r[f"E_rel_{k}_pct"] for r in rows])) for k in CLASSES}
            energy_flags = report_flag_counts(rows)
            report["energy_evaluation"] = dict(
                available=True, stiffness_export={k: kmeta.get(k) for k in (
                    "job", "node_map", "rigid_body_residual", "matrix", "source_inp_sha256", "created_at")},
                validation=en["validation"], mean_E_rel_pct=emean,
                n_energy_dominant={k: sum(r["E_reliable_dominant"] == k for r in rows) for k in MECH},
                n_energy_disagrees=energy_flags["ENERGY_DISAGREES"],
                n_energy_ambiguous=energy_flags["ENERGY_AMBIGUOUS"],
                n_energy_cross_term=energy_flags["ENERGY_CROSS_TERM"],
                n_energy_beyond_section=energy_flags["ENERGY_BEYOND_SECTION"],
                n_energy_incompatible_components=energy_flags["ENERGY_INCOMPATIBLE_COMPONENTS"],
                median_E_beyond_section_pct=_median(rows, "E_beyond_section_pct"),
                median_abs_E_cross_term_pct=_median(rows, "E_cross_term_pct", absolute=True),
                median_bolt_incompatibility_pct=_median(rows, "bolt_incompatibility_pct"),
                definition=gst.__doc__.split("Part 2")[1].strip())
            log("  energy shares (reliable, mean %): " + " ".join(f"{k}{v:.1f}" for k, v in emean.items())
                + f"; resolved energy dominant differs from the kinematic one in "
                  f"{report['energy_evaluation']['n_energy_disagrees']} mode(s)")
            if any(energy_flags[k] for k in ("ENERGY_AMBIGUOUS", "ENERGY_CROSS_TERM",
                                             "ENERGY_BEYOND_SECTION", "ENERGY_INCOMPATIBLE_COMPONENTS")):
                log(f"  WARNING: auxiliary energy split: {energy_flags['ENERGY_AMBIGUOUS']} ambiguous, "
                    f"{energy_flags['ENERGY_CROSS_TERM']} large cross term, "
                    f"{energy_flags['ENERGY_BEYOND_SECTION']} beyond-section energy, "
                    f"{energy_flags['ENERGY_INCOMPATIBLE_COMPONENTS']} incompatible components; "
                    "energy shares do not establish independent physical family fractions")
        except Exception as e:
            if st.stiffness == "required":
                raise
            energy_note = f"failed: {str(e).splitlines()[0][:300]}"
            log(f"  WARNING: energy evaluation {energy_note}")
    if energy_note:
        report["energy_evaluation"] = dict(available=False, reason=energy_note)
    if m_ref is not None:
        for r in rows:                                   # bending / P-M: critical moment (and force)
            r["M_cr_kNm"] = r["eigenvalue"] * float(m_ref) / 1e6
            if p_ref is not None:
                r["P_cr_kN"] = r["eigenvalue"] * float(p_ref) / 1e3
    write_csv(out / "mode_participation.csv", rows)
    write_csv(out / "cluster_participation.csv", crow)
    np.savez_compressed(out / "longitudinal_profiles.npz", z=mesh.ring_z, dz=res["dz"], modes=np.array(modes),
                        classes=np.array(CLASSES + ("TOTAL",)), profile=res["prof"])
    for gi, sp in enumerate(res["seam_prof"]):
        np.savez_compressed(out / f"seam_jump_profiles_group{gi+1}.npz", z=sp["z"], jumps=sp["jumps"],
                            axial_slip=sp["axial_slip"], modes=np.array(modes),
                            layout="jumps: (3*n_seams, rings, modes) rows per seam = jx, jy, jrot*scale; "
                                   "axial_slip: (n_seams, rings, modes) = u_axial(B)-u_axial(A)")
    stats = dominance_stats(rows)
    crit = mechanism_critical(rows)
    if m_ref is not None:
        for c in crit:
            for k in ("first_dominant", "first_high_confidence", "first_share50"):
                e = c.get(f"{k}_eigenvalue")
                c[f"{k}_M_cr_kNm"] = e * float(m_ref) / 1e6 if e is not None else None
                if p_ref is not None:
                    c[f"{k}_P_cr_kN"] = e * float(p_ref) / 1e3 if e is not None else None
    write_csv(out / "mechanism_critical_stresses.csv", crit)
    refc = reference_check(run_dir, rows)
    if refc:
        report["reference_check"] = refc
        log(f"  REFERENCE MODEL CHECK ({refc['kind']}): {'PASS' if refc['passed'] else 'FAIL'}  {refc}")
    report["dominance"] = stats
    report["mechanism_critical"] = crit
    log(f"  dominant mechanism: {stats['dominant_mechanism']}  mean reliable % "
        + " ".join(f"{k}{v:.1f}" for k, v in stats["mean_pct"].items())
        + f"   combined confidence high/medium/low: {stats['confidence_counts']['high']}/"
          f"{stats['confidence_counts']['medium']}/{stats['confidence_counts']['low']}")
    for measure in ("kinematic", "energy"):
        c = stats["confidence_by_measure"][measure]
        if c is not None:
            log(f"  {measure} confidence high/medium/low: {c['high']}/{c['medium']}/{c['low']}")
    if st.plots != "none":
        t2 = time.time()
        plot_dominance(out / "dominance_summary.png", rows, stats, run_dir.name)
        figinfo = make_mode_figures(out, run_dir.name, rows, modes, U, groups, mesh, res, st,
                                    stats["dominant_mechanism"])
        report["figures"] = figinfo
        log(f"  {figinfo['n_figures']} cross-section figures in {time.time()-t2:.1f} s")
    report["n_modes"] = len(modes)
    report["n_clusters"] = len(crow)
    report["flag_counts"] = report_flag_counts(rows)
    report["reliability"] = reliability_summary(rows)
    report["elapsed_s"] = time.time() - t0
    (out / "basis_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if st.plots != "none":
        plot_summary(out / "participation_summary.png", rows)
    if st.plots == "all":
        pdir = out / "mode_profiles"
        pdir.mkdir(exist_ok=True)
        jobs = [(str(pdir / f"mode_{m:04d}.png"), mesh.ring_z, res["prof"][:-1, :, i], m, eig[i], rows[i])
                for i, m in enumerate(modes)]
        with _process_pool(len(jobs)) as ex:
            list(ex.map(_plot_mode, jobs, chunksize=max(1, len(jobs) // (4 * N_THREADS) or 1)))
    log(f"  done in {time.time()-t0:.1f} s -> {out}")
    return report



# ----------------------------------------------------------------------------
# 11b. Cross-run summary (all runs processed in one call)
# ----------------------------------------------------------------------------
def _read_rows(csv_path: Path) -> list[dict]:
    import csv
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if k in ("dominant", "label", "flags"):
                    r[k] = v or ""
                    continue
                try:
                    r[k] = float(v) if v not in ("", "None") else None
                except ValueError:
                    pass
            r["mode"] = int(r["mode"])
            rows.append(r)
    return rows


def _median(rows, key, absolute=False):
    v = []
    for r in rows:
        try:
            x = float(r.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(x):
            v.append(abs(x) if absolute else x)
    return float(np.median(v)) if v else None


def cross_run_summary(runs: list[Path], st: Settings) -> None:
    """Dominant mechanism over ALL modes of ALL runs (highest mean share), a table per run,
    and cross-section figures of the top modes of that global mechanism in every run
    (redrawn from each run's figure_data.npz - no shard is read again)."""
    root = Path(os.path.commonpath([str(r.resolve()) for r in runs]))
    sdir = root / "GDLC_summary"
    sdir.mkdir(exist_ok=True)
    per_run, all_rows = [], []
    for r in runs:
        f = r / st.out_name / "mode_participation.csv"
        if not f.is_file():
            continue
        rows = _read_rows(f)
        all_rows += rows
        s = dominance_stats(rows)
        crit = {c["mechanism"]: c for c in mechanism_critical(rows)}
        first = min(rows, key=lambda x: x["eigenvalue"])
        lc = {}
        bjf = next(iter(r.glob("*_build.json")), None)
        if bjf:
            lc = json.loads(bjf.read_text(encoding="utf-8")).get("load_case") or {}
        per_run.append(dict(run=r.name, load_case=lc.get("type", "compression"),
                            pm_angle_deg=lc.get("pm_angle_deg"),
                            bending_axis_deg=lc.get("neutral_axis_angle_deg"),
                            n_modes=len(rows), dominant=s["dominant_mechanism"],
                            **{f"mean_{k}_pct": s["mean_pct"][k] for k in MECH},
                            **{f"n_dominant_{k}": s["n_dominant"][k] for k in MECH},
                            **{f"n_dominant_high_conf_{k}": s["n_dominant_high_confidence"][k] for k in MECH},
                            **{f"n_conf_{c}": s["confidence_counts"][c] for c in ("high", "medium", "low")},
                            first_mode_eigenvalue=first["eigenvalue"],
                            first_mode_label=first.get("reliable_label") or first["label"],
                            first_mode_confidence=first.get("confidence"),
                            **{f"first_{k}_dominant_eigenvalue": crit[k]["first_dominant_eigenvalue"] for k in MECH},
                            median_split_spread_pct=_median(rows, "split_spread_pct"),
                            median_abs_cross_term_pct=_median(rows, "cross_term_pct", absolute=True),
                            median_fold_motion_ratio=_median(rows, "fold_motion_ratio"),
                            **{f"mean_E_rel_{k}_pct": (float(np.mean([x[f"E_rel_{k}_pct"] for x in rows]))
                                                       if all(isinstance(x.get(f"E_rel_{k}_pct"), float) for x in rows)
                                                       else None) for k in CLASSES},
                            n_energy_disagrees=sum("ENERGY_DISAGREES" in str(x.get("flags", "")) for x in rows),
                            n_definition_sensitive=sum("DEFINITION_SENSITIVE" in str(x.get("flags", "")) for x in rows),
                            n_dominant_unstable=sum("DOMINANT_UNSTABLE" in str(x.get("flags", "")) for x in rows)))
    if not all_rows:
        return
    refs = {}
    for r in runs:
        f = r / "gdlc_reference.json"
        if f.is_file() and (r / st.out_name / "mode_participation.csv").is_file():
            k = json.loads(f.read_text(encoding="utf-8"))
            rr = _read_rows(r / st.out_name / "mode_participation.csv")
            refs.setdefault(k.get("source_run"), {})[k.get("kind")] = reference_check(r, rr)
    for pr in per_run:
        for kind in ("G", "L"):
            check = refs.get(pr["run"], {}).get(kind)
            pr[f"constrained_{kind}_reference_eigenvalue"] = check["first_eigenvalue"] if check else None
            pr[f"{kind}_reference_check_passed"] = check["passed"] if check else None
            # Retain the legacy column for readers, but never advertise a failed
            # kinematic reference check as a pure-family critical value.
            pr[f"pure_{kind}_reference_eigenvalue"] = (
                check["first_eigenvalue"] if check and check["passed"] else None)
    gstats = dominance_stats(all_rows)
    gmech = gstats["dominant_mechanism"]
    write_csv(sdir / "runs_summary.csv", per_run)
    (sdir / "global_dominance.json").write_text(json.dumps(gstats, indent=2), encoding="utf-8")
    log(f"GLOBAL dominant mechanism over {len(per_run)} runs / {len(all_rows)} modes: {gmech}  "
        + " ".join(f"{k}{v:.1f}" for k, v in gstats["mean_pct"].items()))
    if st.plots == "none":
        return
    plot_dominance(sdir / "global_dominance.png", all_rows, gstats, f"{len(per_run)} runs")
    jobs = []
    gdir = sdir / f"figures_global_top_{gmech}"
    gdir.mkdir(exist_ok=True)
    for r in runs:
        fd = r / st.out_name / "figure_data.npz"
        f = r / st.out_name / "mode_participation.csv"
        if not (fd.is_file() and f.is_file()):
            continue
        rows = _read_rows(f)
        d = np.load(fd)
        have = list(d["modes"])
        for i in select_top_modes(rows, gmech, st.top_modes):
            m = rows[i]["mode"]
            if m not in have:
                continue                                  # not stored: run figure exists in the run folder
            j = have.index(m)
            jobs.append(dict(path=str(gdir / f"{r.name}_mode_{m:04d}.png"), p=d["p"], edges=d["edges"],
                             X=d["X"][j], proj={k: d[f"P_{k}"] for k in MECH}, row=rows[i], z=d["z"],
                             prof=d["prof"][j], z_ring=d["z_ring"], bolt_z=list(d["bolt_z"]),
                             title=f"{r.name}   [global dominant mechanism {gmech}]"))
    if jobs:
        with _process_pool(len(jobs)) as ex:
            list(ex.map(_plot_mode_sections, jobs))
    log(f"  cross-run summary -> {sdir}  ({len(jobs)} figures)")


# ----------------------------------------------------------------------------
# 11c. Constrained reference models for Abaqus (pure-G and corner-restrained L)
# ----------------------------------------------------------------------------
def _axis_dofs(mesh: Mesh):
    k = int(np.argmax(np.abs(mesh.axis)))
    if abs(abs(mesh.axis[k]) - 1) > 1e-6:
        raise RuntimeError("member axis is not parallel to a global axis; reference models need that")
    inplane = [d for d in (1, 2, 3) if d != k + 1]
    return k + 1, inplane, k + 4          # axial translation dof, in-plane dofs, rotation about axis


def _nset_lines(name: str, nodes: list, inst_split=True) -> list[str]:
    """Assembly node set from (instance,label) pairs: one set per instance + a union set."""
    out, parts = [], []
    by_inst: dict = {}
    for inst, lab in nodes:
        by_inst.setdefault(inst, []).append(int(lab))
    for inst, labs in sorted(by_inst.items()):
        sub = f"{name}_{inst}"
        parts.append(sub)
        out.append(f"*Nset, nset={sub}, instance={inst}")
        labs = sorted(set(labs))
        out += [", ".join(str(x) for x in labs[i:i + 16]) for i in range(0, len(labs), 16)]
    out.append(f"*Nset, nset={name}")
    out += [", ".join(parts[i:i + 8]) for i in range(0, len(parts), 8)]
    return out


def _insert_step_bcs(lines: list[str], bc_lines: list[str]) -> list[str]:
    """Add the reference-model BCs WITHOUT writing a new *Boundary keyword in the step:
    Abaqus requires, for *BUCKLE, that every *Boundary of LOAD CASE=2 uses OP=NEW and it
    forbids mixing OP=NEW and OP=MOD in a step. The data lines are therefore appended to
    the first existing *Boundary block of every load case of the step (so they inherit its
    OP= and LOAD CASE= exactly). If the step has no *Boundary at all, the BCs go into the
    model data (before *Step), which every load case of the step inherits."""
    i_step = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("*step"))
    i_end = next(i for i in range(i_step, len(lines)) if lines[i].strip().lower().startswith("*end step"))

    def _case(line: str):
        m = re.search(r"load\s*case\s*=\s*(\d+)", line, re.I)
        return m.group(1) if m else None

    firsts = {}
    for i in range(i_step, i_end):
        l = lines[i].strip()
        if l.lower().startswith("*boundary"):
            firsts.setdefault(_case(l), i)
    tag = "** GDLC reference-model constraints (appended to this *Boundary block)"
    if not firsts:
        return lines[:i_step] + ["** GDLC reference-model constraints", "*Boundary"] + bc_lines + lines[i_step:]
    out = list(lines)
    for i in sorted(firsts.values(), reverse=True):        # insert from the bottom: indices stay valid
        j = i + 1
        while j < i_end and not out[j].lstrip().startswith("*"):
            j += 1                                          # end of this block's data lines
        while j > i + 1 and out[j - 1].startswith("**"):
            j -= 1                                          # keep comment lines with what follows
        out[j:j] = [tag] + bc_lines
    return out


def odb_status(odb: Path) -> str:
    """'ok' | 'missing' | 'locked' | 'failed' | 'stale' for <job>.odb, from the job's own
    .sta/.log (an aborted run still leaves an .odb) and the .inp modification time."""
    if not odb.is_file():
        return "missing"
    if odb.with_suffix(".lck").exists():
        return "locked"
    sta, logf = odb.with_suffix(".sta"), odb.with_suffix(".log")
    done = False
    try:
        if sta.is_file() and "COMPLETED SUCCESSFULLY" in sta.read_text(errors="replace").upper():
            done = True
        elif logf.is_file():
            t = logf.read_text(errors="replace").upper()
            done = bool(re.search(r"^\s*ABAQUS JOB\s+" + re.escape(odb.stem.upper())
                                  + r"\s+COMPLETED\s*$", t, re.M)) and "ERROR" not in t
    except OSError:
        done = False
    if not done:
        return "failed"
    inp = odb.with_suffix(".inp")
    if inp.is_file() and inp.stat().st_mtime > odb.stat().st_mtime + 1.0:
        return "stale"
    return "ok"


def lanczos_restrictions(lines: list[str]) -> list[str]:
    """Model features for which Abaqus refuses the Lanczos eigensolver in *BUCKLE
    (Abaqus Analysis Guide, 'Eigenvalue buckling prediction'): contact pairs / contact
    elements, hybrid elements, connector elements, distributing couplings. General
    contact (*Contact) is not in that list; if a solve is refused anyway,
    solve_reference_model() falls back to subspace iteration automatically."""
    found = set()
    for l in lines:
        u = l.strip().upper()
        if not u.startswith("*") or u.startswith("**"):
            continue
        if u.startswith("*CONTACT PAIR"):
            found.add("contact pair")
        elif u.startswith("*DISTRIBUTING"):
            found.add("distributing coupling")
        elif u.startswith("*ELEMENT") and not u.startswith("*ELEMENT OUTPUT"):
            m = re.search(r"TYPE\s*=\s*([A-Z0-9]+)", u)
            typ = m.group(1) if m else ""
            if typ.startswith(("GAP", "ITS", "ITT", "IRS", "ISL", "DASHPOT")):
                found.add(f"contact element {typ}")
            elif typ.startswith(("CONN", "JOINT")):
                found.add(f"connector element {typ}")
            elif typ.startswith("DCOUP"):
                found.add(f"distributing coupling element {typ}")
            elif re.fullmatch(r"(C3D|CPE|CPS|CAX|CGAX|CGPE|B2|B3|PIPE)\w*?H[TE]?", typ):
                found.add(f"hybrid element {typ}")
    return sorted(found)


def buckle_min_eigenvalue(lines: list[str]):
    """Minimum eigenvalue of interest of a Lanczos *BUCKLE (2nd data field), else None.
    The bending builder writes 0 there: only eigenvalues >= 0 (the reversed moment is a
    different load case)."""
    for i, l in enumerate(lines):
        if l.strip().lower().startswith("*buckle"):
            if "lanczos" not in l.lower().replace(" ", ""):
                return None
            fields = [x.strip() for x in lines[i + 1].split(",")]
            try:
                return float(fields[1]) if len(fields) > 1 and fields[1] else None
            except ValueError:
                return None
    return None


def set_buckle_solver(lines: list[str], n_modes: int, solver: str, min_eig=None) -> list[str]:
    """Rewrite the (first) *BUCKLE of the step for n_modes eigenvalues with the given
    eigensolver. Other *BUCKLE parameters are kept.
      lanczos : data line 'n, min_eig' (reference models: min_eig = 0, the n lowest
                eigenvalues >= 0), default block size and block steps. Without a lower
                bound Lanczos also searches negative eigenvalues (the reversed load: for a
                compression model tension), can stall there and can return a negative
                mode as mode 1; 'n' (no range) only when min_eig is None.
      subspace: 'n, , q, 1250' with q = max(4n, n+16) vectors. Many near-equal
                eigenvalues (local modes of a long member) make subspace iteration
                converge as (lambda_n/lambda_{q+1})^k, so a small subspace (q = 2n) can need
                hundreds of iterations; a larger q costs little per iteration. Subspace has
                no eigenvalue range: with min_eig given (bending) 2n eigenvalues are
                requested, because for a section with 180-degree symmetry every positive
                eigenvalue has a negative twin; the classifier drops eigenvalues <= 0."""
    out = list(lines)
    for i, l in enumerate(out):
        if l.strip().lower().startswith("*buckle"):
            parts = [x.strip() for x in l.strip().split(",")]
            keep = [x for x in parts[1:] if x and not x.lower().replace(" ", "").startswith("eigensolver")]
            out[i] = ", ".join(["*Buckle"] + keep + [f"eigensolver={solver.upper()}"])
            if solver == "lanczos":
                out[i + 1] = str(n_modes) if min_eig is None else f"{n_modes}, {min_eig:g}"
            else:
                n = 2 * n_modes if min_eig is not None else n_modes
                q = max(4 * n, n + 16)
                out[i + 1] = f"{n}, , {q}, 1250"
            return out
    raise RuntimeError("no *BUCKLE step in the source .inp")


REF_G_FORMULATION = "rigid_ring_equations_v2"                    # rigid ring, twist free
REF_G_FORMULATION_NO_TWIST = "rigid_ring_translation_equations_v1"  # rigid ring, twist restrained


def ref_g_twist_mode(run_dir: Path, setting: str, load_case=None) -> tuple[str, str]:
    """(mode, reason): whether REF_G restrains the rotation of the sections about the member axis.
    auto: restrained for compression (REF_G then buckles only in flexure - torsional and
    flexural-torsional modes are suppressed), free for bending / combined runs, whose global
    mode is lateral-torsional and needs the twist."""
    if setting in ("restrained", "free"):
        return setting, f"--ref-g-twist {setting}"
    lc = "compression"
    bj = next(iter(sorted(run_dir.glob("*_build.json"))), None)
    if load_case is not None:
        lc = load_case.get('type', 'compression')
    elif bj:
        try:
            lc = (json.loads(bj.read_text(encoding="utf-8")).get("load_case") or {}).get("type") or lc
        except (OSError, ValueError):
            pass
    if lc == "compression":
        return "restrained", "auto: compression run (global flexural buckling only)"
    return "free", f"auto: {lc} run (lateral-torsional buckling needs the twist)"


def rigid_ring_equations(ring, inplane, mpc_independent=(), bc_nodes=(), twist: bool = True):
    """*Equation lines making one cross-section ring rigid in its own plane.
    twist=False: the ring may only TRANSLATE in its plane (u_i = u_a for every node,
    2 rigid DOFs per ring): no rotation of the section about the member axis, so no
    torsional or flexural-torsional buckling; flexure and warping stay free.

    ring: [((instance, label), (p1, p2))] in-plane coordinates in the order of `inplane`.
    Rigid motion: u_i = u_a + theta * k x (r_i - r_a). Masters: node a (a node with a BC,
    e.g. the axial anchor, when present; otherwise far from the centroid) and the node b
    farthest from a; theta = (u_bj - u_aj) / (L n_j), n = k x (r_b - r_a)/L, j = argmax|n_j|.
    Dependent DOFs (first term of each equation, coefficient 1): b's other in-plane DOF and
    both in-plane DOFs of every other node; masters are never MPC-independent bolt nodes and
    no dependent DOF appears in another equation. Returns (lines, number of equations)."""
    if len(ring) < (3 if twist else 2):
        return [], 0
    pts = np.array([p for unused, p in ring], float)
    keys = [k_ for k_, unused in ring]
    free = [i for i, k_ in enumerate(keys) if k_ not in mpc_independent] or list(range(len(ring)))
    centre = pts.mean(0)
    with_bc = [i for i in free if keys[i] in bc_nodes]
    a = with_bc[0] if with_bc else max(free, key=lambda i: (np.linalg.norm(pts[i] - centre), -i))
    if not twist:                                         # pure in-plane translation of the ring
        lines = []
        for i, key_i in enumerate(keys):
            if i == a:
                continue
            for c_ in (0, 1):
                lines += ["*Equation", "2",
                          f"{key_i[0]}.{key_i[1]}, {inplane[c_]}, 1., {keys[a][0]}.{keys[a][1]}, {inplane[c_]}, -1."]
        return lines, 2 * (len(keys) - 1)
    b = max(free, key=lambda i: (np.linalg.norm(pts[i] - pts[a]), -i))
    rab = pts[b] - pts[a]
    length = float(np.linalg.norm(rab))
    if length <= 1e-9 * max(1.0, float(np.abs(pts).max())):
        raise RuntimeError("degenerate ring: master nodes coincide")
    n = np.array([-rab[1], rab[0]]) / length
    j = int(np.argmax(np.abs(n)))
    k = 1 - j
    lines, count = [], 0

    def emit(terms):
        merged = {}
        for node, comp, coef in terms:
            merged[(node, comp)] = merged.get((node, comp), 0.0) + coef
        out = [(nd, c_, v) for (nd, c_), v in merged.items() if abs(v) > 1e-12 or (nd, c_) == terms[0][:2]]
        items = [f"{nd[0]}.{nd[1]}, {inplane[c_]}, {v:.15g}" for nd, c_, v in out]
        return ["*Equation", str(len(out))] + [", ".join(items[i:i + 4]) for i in range(0, len(items), 4)]

    ka, kb = keys[a], keys[b]
    ratio = n[k] / n[j]
    lines += emit([(kb, k, 1.0), (ka, k, -1.0), (kb, j, -ratio), (ka, j, ratio)])
    count += 1
    for i, key_i in enumerate(keys):
        if i in (a, b):
            continue
        w = np.array([-(pts[i][1] - pts[a][1]), pts[i][0] - pts[a][0]])
        for c_ in (0, 1):
            f = w[c_] / (length * n[j])
            lines += emit([(key_i, c_, 1.0), (ka, c_, -1.0), (kb, j, -f), (ka, j, f)])
            count += 1
    return lines, count


def local_corner_translation_rows(g: RingGeom) -> np.ndarray:
    """Qf u = 0 with each fold-reference translation as a unique dependent DOF.

    On short arcs the chord-rotation row contains the reference node itself. Solve
    for those translations together instead of substituting the independent shell
    rotation. Removing all other dependent columns also respects Standard's rule
    that an eliminated *Equation DOF cannot occur in a subsequent constraint.
    """
    _, _, Qf, _ = build_operators(g)
    if not g.folds:
        return Qf
    pivots = np.array([2*f['ref']+c for f in g.folds for c in (0, 1)])
    block = Qf[:, pivots]
    if len(set(pivots)) != len(pivots) or np.linalg.cond(block) > 1e10:
        raise ValueError('REF_L fold translations cannot be eliminated reliably; '
                         'check fold geometry/mesh instead of substituting shell UR')
    rows = np.linalg.solve(block, Qf)
    rows[:, pivots] = np.eye(len(pivots))
    return rows


def make_reference_models(run_dir: Path, st: Settings, kinds=("G", "L"), n_modes: int = 20) -> list[Path]:
    """Writes <run>_REF_G/ and <run>_REF_L/ next to the run, each with a modified .inp
    (submit it with Abaqus, then export its modes as for any run):
      REF_G: every cross-section ring is kept rigid IN ITS PLANE by linear *Equation rows
             (no reference nodes); axial (warping) displacements and seam slip stay free.
             st.ref_g_twist (auto = restrained for compression, free for bending/combined):
             restrained -> each ring only translates in its plane (2 DOFs per ring): pure
             global FLEXURAL modes (no torsional / flexural-torsional buckling);
             free -> 3 rigid DOFs per ring: flexural, torsional, flexural-torsional or
             lateral-torsional modes. Partial composite action of the built-up member is
             kept in both; the first eigenvalue is the constrained-G critical stress.
      REF_L: the virtual sharp corner of every fold is held in the section plane along the
             whole member (fold rotation free). All load cases use the exact classifier's
             translation-based fold-driver rows, including short arcs. Seams stay as built.
             Purity still requires the output
             check (extension/arc deformation and skipped constrained folds can remain).
    The eigensolver is selected from model features before submission; a failed solve is
    preserved for diagnosis rather than silently retried. Legacy refusal metadata falls
    back to subspace iteration automatically (see solve_reference_model).
    Nodes with in-plane boundary conditions or eliminated by an *MPC are not constrained again."""
    # Geometry source: the modal export (.odb node coordinates) as before, or the .inp itself
    # for a built, not yet solved run. The source is kept for the life of the reference
    # folders (recorded as "geometry" in gdlc_reference.json). Historical INP-only
    # references need no export lookup. load_mesh now verifies export node identities
    # and coordinate precision before using the matching INP's original geometry.
    # Old solved reference inputs are still protected if regeneration changes equations.
    recorded = set()
    for k_ in ("G", "L"):
        f_ = run_dir.parent / f"{run_dir.name}_REF_{k_}" / "gdlc_reference.json"
        try:
            recorded.add(json.loads(f_.read_text(encoding="utf-8")).get("geometry", "modal_export"))
        except (OSError, ValueError):
            pass
    exp_dir, meta = None, {}
    if st.export_dir or "inp" not in recorded:
        try:
            exp_dir, meta = choose_export(run_dir, st.export_dir)
        except FileNotFoundError:
            if st.export_dir:
                raise
    inp_path = find_inp(run_dir, meta, st.inp)
    if inp_path is None or not inp_path.is_file():
        raise FileNotFoundError("no unique <job>.inp in the run folder")
    # Bind the reference to this INP's record, never an unrelated KE/other build.
    build_path = run_dir / (inp_path.stem + "_build.json")
    source_build = json.loads(build_path.read_text(encoding="utf-8")) if build_path.is_file() else {}
    load_case = source_build.get("load_case") or {"type": "compression"}
    source_hash = file_sha256(inp_path)
    inp = read_inp(inp_path)
    t = st.thickness or (min(inp.thicknesses) if inp.thicknesses else None)
    mesh = load_mesh(exp_dir, meta, inp_path)
    log(f"  geometry: {'modal export ' + exp_dir.name if exp_dir else inp_path.name + ' (no modal export yet)'}")
    groups = build_ring_groups(mesh)
    for g in groups:
        detect_reference_points(g, t, st)
    ax_dof, inplane, rot_dof = _axis_dofs(mesh)
    excl = set(inp.mpc_dependent)
    for nodes, dofs in inp.boundaries:
        if dofs & set(inplane):
            excl.update(nodes)
    mpc_independent = set(a_ for unused, a_, unused_b in inp.pairs)
    bc_nodes = set(n_ for nodes, unused in inp.boundaries for n_ in nodes)
    key = [(str(i), int(l)) for i, l in zip(mesh.instance, mesh.label)]
    lines0 = inp_path.read_text(encoding="utf-8", errors="replace").splitlines()
    i_end_asm = next(i for i, l in enumerate(lines0) if l.strip().lower().startswith("*end assembly"))
    written = []
    for kind in kinds:
        add_asm, add_bc, info = [], [], dict(kind=kind, source_run=run_dir.name, source_inp=inp_path.name,
                                             source_inp_sha256=source_hash, load_case=load_case,
                                             geometry="modal_export" if exp_dir else "inp")
        if kind == "G":
            # In-plane rigid rings written as linear *Equation rows between ring nodes (no
            # reference nodes, no *Coupling). The earlier kinematic coupling to a ring
            # reference node made Abaqus/Standard report a numerical singularity in the
            # base-state stiffness and refuse the Lanczos solve ("rigid body modes"); the
            # equations impose exactly the same 3-DOF in-plane rigid motion per ring and
            # leave every axial (warping) DOF and every BC untouched.
            twist_mode, twist_reason = ref_g_twist_mode(run_dir, st.ref_g_twist, load_case)
            twist = twist_mode == "free"
            formulation = REF_G_FORMULATION if twist else REF_G_FORMULATION_NO_TWIST
            eq_lines, n_eq, n_rings, n_constrained = [], 0, 0, 0
            for g in groups:
                for gr, r in enumerate(g.rings):
                    rows = g.node_rows[gr]
                    # global coordinates along the in-plane DOFs the equations act on
                    ring = [(key[i], mesh.xyz[i, [d - 1 for d in inplane]]) for i in rows
                            if key[i] not in excl]
                    lines_r, n_r = rigid_ring_equations(ring, inplane, mpc_independent, bc_nodes, twist)
                    if not n_r:
                        continue
                    eq_lines += lines_r
                    n_eq += n_r
                    n_rings += 1
                    n_constrained += len(ring)
            add_asm.append("** GDLC REF_G: in-plane rigid cross-sections (linear equations per ring), twist "
                           + twist_mode)
            add_asm += eq_lines
            info.update(n_rings_rigid=n_rings, n_nodes_rigid=n_constrained, n_equations=n_eq,
                        twist=twist_mode, twist_reason=twist_reason,
                        constraint=("u_i = u_a + theta k x (r_i - r_a) per ring; theta from master node b; "
                                    if twist else
                                    "u_i = u_a per ring (in-plane translation only, no rotation about the "
                                    "member axis: no torsional / flexural-torsional modes); ")
                                   + "*Equation rows, no reference nodes or couplings",
                        formulation=formulation)
            log(f"  REF_G twist {twist_mode} ({twist_reason})")
        elif kind == "L":
            # The classifier's L criterion is: translation of every fold's VIRTUAL SHARP
            # CORNER c = 0, with the fold free to rotate (cFSM). For a rounded fold c is not
            # a node, so the classifier's own kinematic statement is imposed with one
            # *Equation per in-plane direction on the fold reference node r:
            #     u_r + theta x (c - p_r) = 0,
            # theta = mean chord rotation ((u_b - u_a).n/h), preferring edges away from r
            # and using all internal edges when none are away. Eliminate the reference
            # translations from these exact Qf rows for every load case; shell UR is an
            # independent DOF and cannot substitute for chord rotation. Holding r itself would
            # let c move by theta*|c - p_r|; sharp folds (c = p_r) are held in-plane.
            held, held_dof, eq_lines, n_eq = [], {}, [], 0
            for g in groups:
                build_operators(g)                         # fills each fold's virtual_corner
                exact_rows = local_corner_translation_rows(g)
                for fi, f in enumerate(g.folds):
                    r = f["ref"]
                    off = np.asarray(f.get("virtual_corner", g.p[r])) - g.p[r]
                    sharp = float(np.linalg.norm(off)) <= 1e-9 * mesh.size or len(f["nodes"]) < 2
                    for gr in range(len(g.rings)):
                        kk = key[g.node_rows[gr][r]]
                        if kk in excl:
                            continue
                        if sharp:
                            held.append(kk)
                            continue
                        for c in (0, 1):
                            terms = [(kk, inplane[c], 1.0)]
                            row = exact_rows[2*fi+c]
                            nz = np.flatnonzero(np.abs(row) > 1e-12)
                            terms += [(key[g.node_rows[gr][j//2]], inplane[j % 2], float(row[j]))
                                      for j in nz if j != 2*r+c]
                            blocked = [n_ for n_, _, _ in terms[1:] if n_ in excl]
                            if blocked:
                                raise ValueError(f'REF_L exact fold row references constrained/MPC-dependent '
                                                 f'node {blocked[0]}; cannot safely substitute shell UR')
                            terms = terms[:1] + [t_ for t_ in terms[1:] if abs(t_[2]) > 1e-12]
                            if len(terms) == 1:
                                # Abaqus rejects one-term *Equation rows ("use *BOUNDARY"):
                                # the constraint is u_r,c = 0, written as a boundary condition
                                held_dof.setdefault(inplane[c], []).append(kk)
                                continue
                            eq_lines += ["*Equation", str(len(terms))]
                            items = [f"{n_[0]}.{n_[1]}, {d_}, {v:.12g}" for n_, d_, v in terms]
                            eq_lines += [", ".join(items[i:i + 4]) for i in range(0, len(items), 4)]
                            n_eq += 1
            add_asm.append("** GDLC REF_L: fold virtual corners held in the section plane (fold rotation free)")
            add_asm += eq_lines
            if held:
                add_asm += _nset_lines("LREF_FOLDS", held)
                add_bc += [f"LREF_FOLDS, {d}, {d}" for d in inplane]
            for d, nodes_d in sorted(held_dof.items()):
                add_asm += _nset_lines(f"LREF_FOLDS_U{d}", nodes_d)
                add_bc.append(f"LREF_FOLDS_U{d}, {d}, {d}")
            info.update(n_fold_lines=sum(len(g.folds) for g in groups), n_equations=n_eq,
                        n_equations_shell_rotation=0, n_nodes_held=len(held),
                        n_dofs_held={str(d): len(v) for d, v in sorted(held_dof.items())},
                        constraint="u_ref + theta_chord x (virtual_corner - p_ref) = 0 per fold and ring "
                                   "(exact classifier Qf rows; reference translations eliminated together)",
                        formulation='virtual_corner_exact_chord_v1')
        lines = lines0[:i_end_asm] + add_asm + lines0[i_end_asm:]
        if add_bc:
            lines = _insert_step_bcs(lines, add_bc)
        # fewer modes are needed in a reference model; Lanczos when Abaqus allows it
        out_dir = run_dir.parent / f"{run_dir.name}_REF_{kind}"
        prev = out_dir / "gdlc_reference.json"
        prev_info = json.loads(prev.read_text(encoding="utf-8")) if prev.is_file() else {}
        # a refusal recorded for an earlier REF_G formulation (kinematic coupling) does not apply
        refused = bool(prev_info.get("lanczos_refused", False)) and (
            kind != "G" or prev_info.get("formulation") == info.get("formulation"))
        solver = "lanczos" if not (lanczos_restrictions(lines) or refused) else "subspace"
        source_min = buckle_min_eigenvalue(lines0)       # bending source: eigenvalues >= 0 only
        # Lanczos always with a lower bound (see set_buckle_solver): the source's own minimum, else 0.
        lanczos_min = source_min if source_min is not None else 0.0
        min_eig = lanczos_min if solver == "lanczos" else source_min
        lines = set_buckle_solver(lines, n_modes, solver, min_eig)
        info.update(eigensolver=solver, lanczos_restrictions=lanczos_restrictions(lines),
                    lanczos_refused=bool(refused), min_eigenvalue=min_eig,
                    source_min_eigenvalue=source_min)
        out_dir.mkdir(exist_ok=True)
        job = f"{inp_path.stem}_REF_{kind}"
        new_txt = "\n".join(lines) + "\n"
        inp_out = out_dir / f"{job}.inp"
        has_solver_artifacts = any(list(out_dir.glob('*'+ext)) for ext in ('.odb','.dat','.msg','.sta','.log','.res','.sim','.lck'))
        if has_solver_artifacts and inp_out.is_file() and inp_out.read_text(encoding="utf-8", errors="replace") != new_txt:
            raise FileExistsError(f"reference input already exists with different content and solver results: "
                                  f"{inp_out}. The existing results are never overwritten: rename or move "
                                  f"{out_dir.name} (e.g. to {out_dir.name}_previous) and run again")
        if list(out_dir.glob('*.odb')):
            log(f"  reference {kind}: matching existing input/results preserved: {out_dir}")
            written.append(out_dir)
            continue
        if not inp_out.is_file() or inp_out.read_text(encoding="utf-8", errors="replace") != new_txt:
            inp_out.write_text(new_txt, encoding="utf-8")   # a changed model makes an old .odb 'stale'
        env = run_dir / "abaqus_v6.env"
        if env.is_file():                                   # same solver settings as the source run
            (out_dir / "abaqus_v6.env").write_text(env.read_text(encoding="utf-8", errors="replace"),
                                                   encoding="utf-8")
        if build_path.is_file():
            (out_dir / f"{job}_build.json").write_text(build_path.read_text(encoding="utf-8"), encoding="utf-8")
        info.update(job=job, n_modes=n_modes, inplane_dofs=inplane,
                    input_sha256=file_sha256(inp_out),
                    note="submit with Abaqus (datacheck first), then export the modes to "
                         "portable_modal_export*/ as for any run; the classifier then checks it")
        (out_dir / "gdlc_reference.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        log(f"  reference model {kind}: {out_dir / (job + '.inp')}  {info}")
        written.append(out_dir)
    return written


def reference_check(run_dir: Path, rows) -> dict | None:
    """For a REF_G / REF_L run: does the classifier recover the imposed mechanism?"""
    f = run_dir / "gdlc_reference.json"
    if not f.is_file():
        return None
    ref = json.loads(f.read_text(encoding="utf-8"))
    kind = ref.get("kind")
    if kind == "G":
        v = [r["G_pct"] for r in rows]
        res = dict(kind="G", min_G_pct=min(v), mean_G_pct=float(np.mean(v)),
                   passed=bool(min(v) >= 95.0))
    elif kind == "L":
        v = [r["G_pct"] + r["D_pct"] + r["C_pct"] for r in rows]
        c = [r["C_pct"] for r in rows]
        local = [r["L_pct"] for r in rows]
        ldom = [r["eigenvalue"] for r in rows if r.get("reliable_dominant", r.get("dominant")) == "L"]
        # Folds held => no G, D or C (every one of them moves folds): G+D+C must vanish.
        res = dict(kind="L", max_G_plus_D_plus_C_pct=max(v), mean_G_plus_D_plus_C_pct=float(np.mean(v)),
                   max_C_pct=max(c), mean_C_pct=float(np.mean(c)),
                   min_L_pct=min(local),
                   first_L_dominant_eigenvalue=min(ldom) if ldom else None,
                   passed=bool(max(v) <= 5.0 and min(local) >= 95.0))
    else:
        return None
    res.update(source_run=ref.get("source_run"), first_eigenvalue=min(r["eigenvalue"] for r in rows))
    return res

# ----------------------------------------------------------------------------
# 12. CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    global N_THREADS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folder(s) or a root folder to scan")
    ap.add_argument("--threads", type=int, default=N_THREADS)
    ap.add_argument("--gpu", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--plots", choices=("all", "summary", "none"), default="all")
    ap.add_argument("--fold-radius-factor", type=float, default=Settings.fold_radius_factor)
    ap.add_argument("--min-fold-angle", type=float, default=Settings.min_fold_angle_deg)
    ap.add_argument("--cluster-tol", type=float, default=Settings.cluster_rel_tol)
    ap.add_argument("--max-other-pct", type=float, default=Settings.max_other_pct)
    ap.add_argument("--max-split-spread", type=float, default=Settings.max_spread_pct,
                    help="split spread (%%-points over the fold-rotation variants) above which a mode is "
                         "flagged DEFINITION_SENSITIVE (confidence at most medium)")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the split-sensitivity pass (2 extra projections)")
    ap.add_argument("--max-cross-pct", type=float, default=Settings.max_cross_pct,
                    help="|cross term| (%%) above which a mode is flagged CROSS_TERM (confidence at most medium)")
    ap.add_argument("--max-energy-beyond-pct", type=float, default=Settings.max_energy_beyond_pct)
    ap.add_argument("--max-bolt-incompat-pct", type=float, default=Settings.max_bolt_incompat_pct)
    ap.add_argument("--conf-low-margin", type=float, default=Settings.conf_low_margin_pct)
    ap.add_argument("--conf-high-margin", type=float, default=Settings.conf_high_margin_pct)
    ap.add_argument("--mixed-label-pct", type=float, default=Settings.mixed_label_pct)
    ap.add_argument("--thickness", type=float, default=None)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--inp", default=None)
    ap.add_argument("--out-name", default=Settings.out_name)
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--max-modes", type=int, default=None, help="quick check on the first N modes only")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs that already have results")
    ap.add_argument("--make-reference-models", action="store_true",
                    help="write <run>_REF_G and <run>_REF_L Abaqus inputs (pure-G, corner-restrained L) and exit")
    ap.add_argument("--reference-modes", type=int, default=20, help="eigenvalues requested in reference models")
    ap.add_argument("--ref-g-twist", choices=("auto", "restrained", "free"), default=Settings.ref_g_twist,
                    help="REF_G: rotation of the sections about the member axis. auto (default) = restrained "
                         "for compression runs (global flexural modes only, no torsional / flexural-torsional "
                         "buckling), free for bending / combined runs (lateral-torsional buckling needs it)")
    ap.add_argument("--top-modes", type=int, default=Settings.top_modes,
                    help="cross-section figures for the k modes with the largest share of the dominant mechanism")
    ap.add_argument("--benchmarks", action="store_true",
                    help="only run the analytical D-dimension benchmarks (channel, tube) and exit")
    ap.add_argument("--export", choices=("auto", "always", "never"), default=Settings.export,
                    help="ODB -> modal export with abaqus_mfsm_export.py: auto = only when no export of the "
                         "current .odb exists; always = new export every time; never = use existing exports")
    ap.add_argument("--abaqus-cmd", default=Settings.abaqus_cmd, help="Abaqus launcher (default: abaqus)")
    ap.add_argument("--exporter", default=None, help="path of abaqus_mfsm_export.py (default: next to this script)")
    ap.add_argument("--export-compression", choices=("store", "zlib"), default=Settings.export_compression)
    ap.add_argument("--stiffness", choices=("auto", "always", "required", "never"), default=Settings.stiffness,
                    help="elastic stiffness export + strain-energy evaluation of the split: auto = run the "
                         "Abaqus matrix-generation job once per run (.inp sha256) and continue without the "
                         "energy columns if it fails; always = regenerate; required = stop on failure; never")
    ap.add_argument("--solve-reference-models", action="store_true",
                    help="solve <run>_REF_G / <run>_REF_L folders that have no .odb yet (Abaqus, --cpus)")
    ap.add_argument("--solve-only", action="store_true",
                    help="with --solve-reference-models: solve/reuse references only; no export, stiffness or classification")
    ap.add_argument("--cpus", type=resources.cpu_request, default=None,
                    help="CPUs of the Abaqus solves of --solve-reference-models, written '--cpus 4' "
                         "(default: all physical cores; a larger request is reduced to them)")
    if argv is None and "--benchmarks" in sys.argv:
        sys.argv.append("_")
    a = ap.parse_args(argv)
    if a.solve_only and (not a.solve_reference_models or a.make_reference_models or a.selftest_only):
        ap.error('--solve-only requires --solve-reference-models and cannot generate/classify reference models')
    if a.threads < 1:
        ap.error('--threads must be positive')
    N_THREADS = a.threads
    resources.initialise_worker(N_THREADS)
    if pq is not None:
        pa.set_cpu_count(N_THREADS)
        pa.set_io_thread_count(N_THREADS)
    st = Settings(fold_radius_factor=a.fold_radius_factor, min_fold_angle_deg=a.min_fold_angle,
                  cluster_rel_tol=a.cluster_tol, max_other_pct=a.max_other_pct, max_cross_pct=a.max_cross_pct,
                  max_spread_pct=a.max_split_spread, sensitivity=not a.no_sensitivity,
                  conf_low_margin_pct=a.conf_low_margin, conf_high_margin_pct=a.conf_high_margin,
                  mixed_label_pct=a.mixed_label_pct, thickness=a.thickness, export_dir=a.export_dir, inp=a.inp,
                  out_name=a.out_name, gpu=a.gpu, plots=a.plots, selftest_only=a.selftest_only,
                  max_modes=a.max_modes, threads=a.threads, top_modes=a.top_modes,
                  export=a.export, abaqus_cmd=a.abaqus_cmd, exporter=a.exporter, abaqus_cpus=a.cpus,
                  stiffness=a.stiffness, ref_g_twist=a.ref_g_twist,
                  max_energy_beyond_pct=a.max_energy_beyond_pct, max_bolt_incompat_pct=a.max_bolt_incompat_pct,
                  export_compression=a.export_compression)
    if a.benchmarks:
        log(f"gdlc_mode_decomposition.py v{VERSION}")
        res = run_benchmarks(st)
        for r in res:
            extra = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()
                     if k not in ("case", "passed", "D_expected", "D_found", "selftest", "coarse", "fine")}
            log(f"{'PASS' if r['passed'] else 'FAIL'}  {r['case']}: D expected {r['D_expected']}, "
                f"found {r['D_found']}  {extra}")
        sys.exit(0 if all(r["passed"] for r in res) else 1)
    log(f"gdlc_mode_decomposition.py v{VERSION}  ({Path(__file__).resolve()})")
    missing = [p for p in a.paths if not Path(p).is_dir()]
    if missing:                                          # a mistyped run folder is not "no runs"
        sys.exit("folder(s) not found: " + "; ".join(missing))
    runs = find_runs(a.paths)
    if not runs:
        sys.exit("no run folders found (need a modal export, or one <job>.inp with a *Buckle step "
                 "and <job>_build.json / gdlc_reference.json)")
    log(f"{len(runs)} run(s); threads={N_THREADS}; cpus={os.cpu_count()}; export={st.export}")

    if a.solve_only:
        failures = []
        for r in runs:
            try:
                meta = json.loads((r / 'gdlc_reference.json').read_text(encoding='utf-8-sig'))
                job = meta['job']
                previous_status = odb_status(r / (job+'.odb'))
                if not solve_reference_model(r, st) and previous_status != 'ok':
                    raise RuntimeError('Reference solve failed or is still running')
                if not (r / (job+'.inp')).is_file() or odb_status(r / (job+'.odb')) != 'ok':
                    raise RuntimeError('Reference analysis did not finish successfully or ODB is missing/locked')
                log(f'  reference ready for Step 4 (not classified): {r}')
            except Exception as error:
                log(f'  FAILED {r}: {error!r}')
                failures.append(r)
        return 1 if failures else 0

    def _prepare(r: Path) -> bool:
        """solve (reference models, on request) and export; False = nothing to classify yet."""
        if st.selftest_only:
            return bool(list_exports(r) or st.export_dir)
        if a.solve_reference_models:
            solve_reference_model(r, st)
        if st.export != "never" and not st.export_dir:
            ensure_export(r, st)
        if (list_exports(r) or st.export_dir) and not a.make_reference_models:
            ensure_stiffness(r, st)                    # elastic K of the run (energy evaluation)
        if not list_exports(r) and not st.export_dir:
            if a.make_reference_models and run_sources(r)[1] is not None:
                return True                            # reference inputs need only the .inp geometry
            log(f"skip {r.name}: no modal export and no solved .odb yet")
            return False
        return True

    if a.make_reference_models:
        bad = 0
        for r in runs:
            if (r / "gdlc_reference.json").is_file():
                continue                                   # never build a reference of a reference
            try:
                if not _prepare(r):
                    continue
                make_reference_models(r, st, n_modes=a.reference_modes)
            except Exception as e:
                log(f"  FAILED reference models for {r}: {e!r}")
                bad += 1
        sys.exit(1 if bad else 0)
    failures = []
    done = []
    for r in runs:
        if a.skip_existing and (r / st.out_name / "mode_participation.csv").is_file():
            log(f"skip (results exist): {r}")
            done.append(r)
            continue
        try:
            if not _prepare(r):
                continue
            process_run(r, st)
            done.append(r)
        except Exception as e:                          # keep going with the other runs
            log(f"  FAILED {r}: {e!r}")
            failures.append((str(r), repr(e)))
    if len(done) >= 2 and not st.selftest_only:
        try:
            cross_run_summary(done, st)
        except Exception as e:
            log(f"cross-run summary failed: {e!r}")
    if failures:
        log(f"{len(failures)} run(s) failed")
        for f in failures:
            log(f"  {f[0]}: {f[1]}")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
