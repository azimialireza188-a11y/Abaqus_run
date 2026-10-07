# Abaqus_run

## Technical audit — 7 October 2026

The existing implementation was corrected in place, preserving prior local edits.
No production analysis was submitted. The model builder remains intentionally
limited to the four-piece C4 section, uniform S4R shells and the documented
end restraints; it is not a general arbitrary-section generator.

Corrections made during this audit:

- Validate stable isotropic material constants, finite member metadata, C4
  geometry, seam attachment locations, duplicate connections and connectivity
  before entering CAE. Section-property integration is now insensitive to large
  coordinate offsets. Symmetry checks compare complete walls rather than just
  endpoints; bending-angle run names preserve the normalized angle precision.
- Protect both modal exporters against unsupported double-precision bulk access;
  fall back to precision-aware nodal values. Reject duplicate modes/nodal records,
  missing translations and local-coordinate data that would otherwise be read
  as global vectors. GDLC rejects stale source hashes for NPZ and Parquet.
- Make geometric report ratios independent of eigenvector amplitude. The fast
  suggestion path treats near-repeated eigenvalues as unresolved when it has
  not evaluated the eigenspace, even if individual labels agree.
- Repair GDLC's direct-sum basis for strips and pieces with a single fold: rigid
  directions that do not move folds must not also appear in Other. Cluster
  whitening now removes arbitrary mode scale before its rank decision and
  clusters do not depend on archive ordering. Invalid ring edges, incomplete
  archives, mixed shell thickness and composite layups fail explicitly.
- Write GDLC reference coupling DOFs separately so an axial Y DOF is not
  accidentally constrained by the range `1, 3`. Repeated node sets are additive
  and symmetry boundary DOFs are interpreted correctly. Local-reference checks
  require predominantly L content, not merely an absence of G/D/C. Failed
  reference checks no longer populate legacy "pure" eigenvalue columns.
- Use constant resultant compression loading in GMNIA, consistent with the
  reported `P = LPF * P_ref`. Reject shifted/incomplete LPF histories that would
  assign nonzero load to the initial frame. Preserve prior solver failure evidence
  during extract-only processing, and require actual job completion rather than
  a compilation-completion message. End-response extraction rejects local,
  duplicate or nonfinite displacements. Bending provenance vectors must agree
  with their recorded angle.

Scientific interpretation and remaining validation:

- The geometric report and GDLC are kinematic classifiers. Their percentages do
  not partition strain energy or critical load. GDLC uses a discrete transverse
  bending proxy, rigid-fold assumptions, and a project-specific relative-piece
  family C. Its algebraic self-tests and constrained-reference checks are useful
  consistency tests, not independent cFSM/GBT validation. See
  [GDLC methodological scope](gdlc_classifier/README_GDLC.md).
- The 25%-of-peak fold-extent rule is a mesh-dependent heuristic. Rotation-side
  sensitivity does not quantify uncertainty in that extent. Inspect detected
  corners and establish mesh/fold-threshold sensitivity for each new section
  family. A G component at a short local wavelength is not automatically a
  member-level global instability. A separated transverse component is not by
  itself a new admissible FE eigenmode or an independently solved buckling load.
- Rigid BEAM MPC/connector bolts omit bolt compliance, slip, bearing and hole
  effects. End U1/U2/UR3 restraints enforce the chosen diaphragm/twist condition.
  These are model assumptions to compare against the physical specimen, not
  software errors that can be fixed without connection/support evidence.
- Linear buckling freezes contact in the base state. Open gaps do not become
  load-bearing merely because a plotted eigenmode closes them. Eigenvectors have
  arbitrary amplitude; use GMNIA and imperfection sensitivity for collapse claims.
  These interpretations follow the
  [Abaqus eigenvalue buckling documentation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-eigenbuckling.htm).
- Independent modal validation should compare equivalent section geometry,
  restraints, loading and longitudinal wavelengths with a reference such as
  [CUFSM](https://www.ce.jhu.edu/cufsm/about/). Self-recovery of a constructed
  basis alone cannot establish agreement with that reference.

Before production: use the existing `--check-inputs` and `--build-only` paths;
inspect the generated mesh, loads, end restraints and bolt/contact definitions;
perform an Abaqus datacheck; and regenerate classifications without
`--skip-existing`. Establish mesh convergence and independent modal benchmarks
before using family-specific critical values in DSM or research conclusions.

Verification uses the repository's lightweight Python tests, numerical GDLC
benchmarks and syntax compilation with installed Abaqus 2024 Python. These do
not establish CAE API execution, solver convergence or physical calibration.
Final results: 254 Python tests passed; 64 focused tests plus two additional
data-validation regressions passed under Abaqus Python; all seven GDLC benchmarks
passed under Abaqus Python; all 38 Python files compiled syntactically; and
`git diff --check` passed. No CAE build, datacheck or solver analysis was launched.

Step 4 now supports fast geometric G/L/D suggestions directly from the
reference ODB through the existing `--suggest` command. See
[fast screening and execution](README_fast_modal_suggest.md). The exact
reference-pipeline provenance and Step-5 behavior are preserved.

This repository contains the single-model Abaqus buckling pipeline used to build the CAE/INP, submit the eigenvalue buckling job, postprocess the ODB, and optionally run the modal audit.

## Requested production command

```bat
cd /d "C:\Users\810200014.HAMI.000\Documents\Abaqus_run"

abaqus cae noGUI=abaqus_complete_model_m20.py -- ^
  --builtup-dir "C:\Users\810200014.HAMI.000\Documents\CUFSM-Single\cfs_abaqus\A4784_t3_qm1_0_0_0_R2p5_lipS_lipLen60_M80_L3600_gap10_nb19_end25-25_row15" ^
  --output-root "D:\CFS-Column\New folder (10)" ^
  --mesh-mm 5 ^
  --n-modes 250 ^
  --n-vectors 500 ^
  --max-iterations 1250 ^
  --cpus 8 ^
  --buckle-output detailed ^
  --nodal-precision full ^
  --longitudinal-lines 6 ^
  --longitudinal-line-min-spacing-mm 5 ^
  --portable-export parquet ^
  --portable-modes-per-shard 8 ^
  --portable-export-workers auto ^
  --modal-audit
```

## Bending load case (P-M workflow)

All five stages also run for bending; compression stays the default and is
unchanged. Only Step 3 needs new options; every later stage reads the load case
from `pipeline_status.json` / the Step-4 provenance.

**Reference stress (CUFSM-consistent).** Compression: uniform 1 MPa, so the
eigenvalue is the critical stress (P_cr = lambda*A). Bending: linear stress
`sigma = 1 MPa * eta/c` on both end sections, `eta` = distance from the neutral
axis through the centroid, compression positive on the `+n = (-sin t, cos t)`
side, `t = --bending-axis-deg` measured from global X, `c` = largest `|eta|` of
the CUFSM centre-line nodes. The extreme-fibre stress is exactly 1 MPa, so the
eigenvalue is the critical extreme-fibre stress in MPa, directly comparable with
a CUFSM signature curve for a unit extreme-fibre reference stress, and
`M_cr = lambda * M_ref`, `M_cr/M_y = lambda/f_y` (DSM flexure). The stress is
applied as the consistent nodal forces of the linear edge traction (exact for
the linear S4R edges); `M_ref` actually applied by the mesh is stored as
`load_case.reference_moment_mesh_Nmm_per_MPa` in the build json. Boundary
conditions are those of compression (end sections fixed in their plane and
against twist, warping free). Lanczos with eigenvalues >= 0 is used: the reversed
moment is another load case (for this section it is identical by symmetry).

**Axes needed for P-M.** `--check-inputs --load-case bending` prints the
section symmetry and `bending_axes_for_PM`. The four-piece square sections are
D4-symmetric (rotations by 90 deg and four mirror lines), so every bending axis
is equivalent to one in [0, 45] deg and +M = -M. Required: `0` (neutral axis
parallel to the faces; seams/lips at the extreme fibre) and `45` (diagonal;
corners at the extreme fibre). `22.5` is optional (only for a biaxial surface).

```bat
cd /d "C:\Users\810200014.HAMI.000\Documents\Abaqus_run"
abaqus cae noGUI=abaqus_complete_model_m20.py -- ^
  --builtup-dir "C:\Users\810200014.HAMI.000\Documents\CUFSM-Single\cfs_abaqus\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15" ^
  --output-root "D:\CFS-Column" ^
  --load-case bending ^
  --bending-axis-deg 0 ^
  --mesh-mm 20 --n-modes 250 --n-vectors 500 --max-iterations 1250 --cpus 8 ^
  --buckle-output detailed --nodal-precision full ^
  --longitudinal-lines 4 --longitudinal-line-min-spacing-mm 5 ^
  --modal-audit
```

The run folder is `<input>_BEND000` (`_BEND045` for 45 deg); `--build-only`,
`--resume-post` etc. work as for compression. `--n-vectors/--max-iterations`
are only used with `--eigensolver subspace`.
GDLC, Step 4 and Step 5 use the same commands as for compression with the
bending run folder; see `README_step4_imperfections.md` and
`README_step5_gmnia.md` (sections "Bending") for the global-mode choice and the
moment-rotation outputs. `STEP5_PM_points.csv` in every Step-5 folder (both load
cases) holds one (P, M) capacity point per model for the later P-M diagram.

## Longitudinal section lines

`--longitudinal-lines` controls longitudinal boundaries retained from the original section geometry before meshing:

- `0`: original simplification behavior.
- `2..99`: retain up to the requested number of internal lines in each curved region, with refinement concentrated where the section turning/curvature is greater.
- `100`: retain all original source lines. The minimum-spacing filter is intentionally bypassed.
- `1` is invalid.

`--longitudinal-line-min-spacing-mm 5` applies a 5 mm minimum section-arclength spacing to every retained **nonessential** longitudinal line, including low-turn lines inherited from the virtual-topology prepass and newly selected refinement lines. When soft lines compete inside the spacing neighborhood, the more curvature-sensitive line is retained.

The spacing filter never removes hard constraints: the two section-chain ends, exact bolt-row partitions, and genuine sharp corners using the existing 10 degree criterion. Therefore a spacing below 5 mm can remain only when both relevant boundaries are mandatory.

## CAE / INP / solver identity

The analysis model is constructed only once in `mdb.Model`. The same in-memory model and the same `mdb.Job` are used to:

1. write the INP with `job.writeInput(...)`;
2. save the CAE with `mdb.saveAs(...)`; and
3. submit the analysis with `job.submit(...)`.

There is no separate solver model and no presentation-only CAE model. The saved CAE is the actual model from which the INP/job are produced. Geometry, partitions, retained longitudinal lines, mesh, loads, boundary conditions, MPC bolts, contact, step, and output requests are all defined before both artifacts are written.

Use `--build-only` when you want to inspect the actual CAE and INP without submitting the solver.

## Regression checks

The repository includes regression tests for the longitudinal-line selector, spacing filter, modal pipeline, audit, validation, visuals, and physical-wall helper. In particular, the longitudinal tests verify that inherited soft lines are filtered by the requested spacing while bolt lines and sharp corners remain mandatory.


## Portable Parquet modal archive

The full solver pipeline now requests both nodal `U` and `UR`. After a successful
solve it creates a portable archive before the heavier report/audit stages. The
default portable format is Parquet; use `--portable-export off` to disable it,
`npz` for the NumPy-only archive, or `both` to keep both forms.

Parquet uses ZSTD compression (level 9) and is stored in
`portable_modal_export` under the run directory. The layout is deliberately
columnar and compact:

- `modal_export.json`: provenance, hashes, mode/eigenvalue metadata, precision,
  field availability and artifact list.
- `mesh_nodes.parquet`: one row per mesh node with instance, label and global
  undeformed coordinates.
- `raw_dof_map.parquet`: compact raw-index -> node-index/DOF mapping.
- `elements.parquet`: S4R connectivity.
- `modes.parquet`: one row per buckling mode.
- `mode_shapes_0001.parquet`, ...: one row per node; each shard stores several
  mode-component columns such as `m0001_u1` through `m0001_ur3`. This avoids
  repeating instance/label metadata for every mode/node pair.

The exporter preserves FP64 when the ODB nodal field is double precision. It
uses FP32 only when the source field explicitly reports single precision; it
does not down-cast unknown precision.

### Parquet dependency

ODB extraction is performed by Abaqus Python, but Parquet encoding can be
delegated automatically to normal Python with `pyarrow`. Install once, for
example:

```bat
py -3 -m pip install pyarrow numpy
```

If auto-detection cannot find that Python, add the exact interpreter:

```bat
--parquet-python "C:\Path\To\python.exe" ^
```

The Parquet backend is checked before an expensive solver run when Parquet is
requested.

### Export an older ODB without rebuilding or solving

The same raw archive can be created later from an ODB produced by this workflow:

```bat
cd /d "C:\Users\810200014.HAMI.000\Documents\Abaqus_run"

abaqus python abaqus_modal_export.py ^
  --odb "D:\CFS-Column\New folder (10)\YOUR_RUN\BU_BOLT_L3600_M5.odb" ^
  --output-dir "D:\CFS-Column\New folder (10)\YOUR_RUN\portable_modal_export_recovered" ^
  --format parquet ^
  --modes-per-shard 8 ^
  --workers auto
```

No CAE rebuild and no solver submission occur in this command. If a same-basename
INP is present, its SHA256 is recorded; the ODB itself is sufficient for the raw
mesh/mode archive.

Older ODBs created when the builder requested only `U` remain exportable. The
archive then contains only U1/U2/U3 and records
`rotations_available=false`; UR is never filled with zeros or reconstructed.
New runs request `U + UR`, so they produce the six-DOF portable map.

This portable archive is raw modal data. It is not an mFSM operator pack and
does not manufacture validated Local/Distortional/Global energy shares from the
ODB. Mechanical mFSM decomposition still requires independently reviewed
operators/evidence.


### Performance architecture

Portable export is intentionally split into two stages:

1. **ODB extraction remains single-owner** because the Abaqus ODB API is not
   treated as thread-safe. The exporter first tries the vectorized
   `bulkDataBlocks` interface and reuses its node-row mapping across modes. If
   an ODB does not support a stable bulk layout, it falls back automatically to
   the original per-`FieldValue` mapping.
2. **Parquet/ZSTD compression is process-parallel.** `--workers auto` uses all
   logical CPUs, capped only by the number of mode shards. Each writer process
   is constrained to one native BLAS/Arrow/OpenMP thread to avoid nested
   oversubscription.

For 250 modes with `--modes-per-shard 8`, the archive contains 32 shape
shards because `ceil(250/8)=32`: 31 shards contain 8 modes and the final shard
contains 2 modes. Increasing `--modes-per-shard` reduces the number of files
but also reduces the number of independently compressible tasks. On a machine
with many logical CPUs, 8 modes per shard is a useful compromise for a
250-mode export because it provides 32 parallel tasks.

The final console summary reports `odb_extraction_backend` and
`parquet_workers`. The fastest expected path is
`bulkDataBlocks` plus a worker count close to the machine's logical CPU count.


## Stage 4 / Stage 5 compatibility contract

The nonlinear workflow is downstream of the exact production eigenvalue model;
it is not a separate geometry/model generator.

- Step 3/reference: `abaqus_complete_model_m20.py`
- Step 4: `abaqus_step4_imperfections.py`
- Step 5: `abaqus_step5_gmnia.py`

Step 4 reads `pipeline_status.json` and binds itself to the recorded
`build.job_name`, CAE, ODB, P1..P4 instances, S4R mesh, BEAM_MPC bolt count,
BC/load count and GeneralContact definition. It stores that provenance in every
`STEP4_*` model.

Step 5 accepts only those pipeline-compatible Step-4 models. It preserves their
imperfect mesh, BCs and contact; adds elastic-perfectly-plastic material and a
Static Riks step; and converts each reference BEAM_MPC into an assembled BEAM
connector on the exact same endpoint nodes so connector force/moment output can
be requested. The conversion is explicitly recorded in model metadata.

See `README_step4_imperfections.md` and `README_step5_gmnia.md`.
