# Abaqus_run

## Solve a model already created with --build-only

Running the builder again without `--build-only` creates another run folder and
rebuilds the model. To solve the existing INP, open CMD in the folder containing
that INP, CAE and `pipeline_status.json`, then run:

```bat
abaqus job=BU_BOLT_L3600_M20 input=BU_BOLT_L3600_M20.inp cpus=4 standard_parallel=solver memory="100%" output_precision=full interactive
```

Here `4` is the capacity reported by Abaqus on the audited machine (four physical
cores, eight logical CPUs). Substitute the available solver core count on another
machine. The explicit `cpus=4` overrides an older run's `abaqus_v6.env` containing
`cpus=8`; the existing file does not need editing. The job name must match the
recorded build name. Once the analysis completes successfully, from the same CMD
directory run:

```bat
abaqus python "D:\Abaqus_run\abaqus_complete_model_m20.py" --resume-post "%CD%" --modal-audit
```

This performs the portable export and reports/audit without rebuilding or solving
again. For future builds, prefer `--cpus auto`: all available local physical cores
are used. An oversized explicit CPU request is now adjusted to local capacity
with a warning. Python postprocessing retains all logical CPUs, memory is set to
100%, and resource reserves remain zero. GPU auto-selection uses detected GPUs;
the audited machine exposed none. Actual utilization depends on which portions
of the solver support parallelism/GPU acceleration.

### `--cpus` in every stage

Write every option as `--name value` (with a space): the Abaqus launcher rejects
`--cpus=4` ("both equality sign and prepended dash") before the script starts, and
a blank line inside a `^`-continued CMD command ends the command. `--cpus N|auto`
(auto = all physical cores; a request above them is reduced with a warning) is
accepted by:

| stage | effect of `--cpus` |
|---|---|
| Step 1–3 builder `abaqus_complete_model_m20.py` | Abaqus buckling job and `abaqus_v6.env` |
| Step 4 `abaqus_step4_imperfections.py` | no solver; NumPy threads of the build, default `--suggest-cpus` |
| Step 5 `abaqus_step5_gmnia.py` | Job objects, `abaqus_v6.env` and the generated `run_step5_queue_*.bat` |
| queue `abaqus_step5_queue_nogui.py` (called by the BAT) | `cpus=` of every Riks job; `--repair-only` writes it into the BAT |
| P-M `abaqus_pm_interaction.py --mode buckling/gmnia` | buckling jobs, `run_pm_buckling.bat`, `run_pm_queue_*.bat` |
| `gdlc_classifier\gdlc_mode_decomposition.py --solve-reference-models` | `cpus=` of the REF_G/REF_L solves (`--threads` = Python threads) |

The Abaqus exporter/postprocessing scripts use all logical CPUs and have no
`--cpus` option.

Submission failures now include a short diagnostic from the job's own
`.msg/.log/.dat/.sta` files, including failures occurring before an ODB exists.

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

Follow-up review (Claude, 7 October 2026): the audit corrections above were
checked and kept, with these changes:

- Queue outcome: a 70 % crossing found inside the synchronized LPF/field data
  is now a complete result even when a controlled terminate left one trailing
  unsynchronized frame (`partial`) or when the exit status of an earlier session
  is unknown. Before, such runs were reported as `PARTIAL_RESULT` /
  `SOLVER_STATUS_UNKNOWN` and failed the queue although Pu and the crossing were
  both measured. A known solver error still wins.
- Bulk ODB reads (`abaqus_modal_wavelengths.py`, `abaqus_modal_export.py`)
  query only the local-coordinate accessor of the block's own precision. Abaqus
  raises (not `AttributeError`) for the other precision, which silently forced
  the slow per-value path for every frame.
- Step 5 writes a `<job>_step5.json` metadata sidecar per job; the queue and the
  extractor use it before the CAE (`--cae` is optional when it exists).

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

## P-M interaction of the composite section (`abaqus_pm_interaction.py`)

One entry point with four modes. Every P-M point is a **proportional** end load
along a ray of angle `psi` in the normalized `(P/Py, M/My)` plane about a
neutral-axis angle `theta` (same convention as the bending load case):

`sigma = sigma_ref * s * (cos psi + sin psi * eta/c)`, `s = 1/(cos psi + sin psi)`.

`psi = 0` is pure compression, `psi = 90` pure bending. The extreme compression
fibre always carries `sigma_ref`, so with `sigma_ref = fy` the point `LPF = 1`
is first yield, `P/Py + M/My = 1`. The stress is applied as consistent nodal
forces on the meshed end edges (same BCs, mesh, bolts and contact as the
reference model). Rays: `--axes required|all|0,45` (required = one axis per
symmetry class; equivalent axes are refused) and `--pm-angles` (default
`0,15,30,45,60,75,90`; `psi = 0` is axis-independent and built once).

| mode | runs in | produces |
|---|---|---|
| `section` | Python | `P_y`, `M_y`, `M_p` and the full-plastic P-M curve per axis (`pm_section_reference.*`) |
| `buckling` | Abaqus/CAE | one Lanczos `*BUCKLE` input per ray from the Step-3 CAE, each in its own folder with `<job>_build.json` (GDLC-ready; adds `P_cr_kN`, `M_cr_kNm`), plus `run_pm_buckling.bat` |
| `gmnia` | Abaqus/CAE | Riks inputs for every ray and every Step-4 case (same imperfect specimen for all rays), built by the Step-5 code, plus `run_pm_queue_70_pct.bat` (Step-5 queue: 70 % post-peak stop, plots, `STEP5_PM_points.csv`) |
| `collect` | Python | `PM_interaction_points.csv` and `PM_interaction_A<axis>.png` from P-M, Step-5 compression/bending folders and solved buckling folders |

The GMNIA builder prepares one CAE model per Step-4 case and writes the rays
from its INP by changing only the CF3 magnitudes of the end-node loads, so all
rays of a case share mesh, imperfection, material, bolts and Riks step. No large
CAE is saved for P-M; each job has a `<job>_step5.json` metadata sidecar (Step 5
writes them too), which the queue and the extractor read instead of a CAE.

```bat
set "RUN_DIR=D:\D-CFS-Column\D-CFS-Column\A3184_t2_qm1_0_0_0_R2p5_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15"
cd /d "D:\Abaqus_run"
abaqus cae noGUI=abaqus_pm_interaction.py -- ^
  --mode buckling ^
  --run-dir "%RUN_DIR%" ^
  --output-dir "%RUN_DIR%\PM_buckling" ^
  --axes required ^
  --pm-angles 0,15,30,45,60,75,90 ^
  --n-modes 20
abaqus cae noGUI=abaqus_pm_interaction.py -- ^
  --mode gmnia ^
  --source-cae "%RUN_DIR%\imperfection\Step4_LDG.cae" ^
  --output-dir "%RUN_DIR%\imperfection\PM_GMNIA_FY240_LDG" ^
  --fy 240 ^
  --axes required ^
  --pm-angles 0,15,30,45,60,75,90 ^
  --initial-arc 0.01 ^
  --min-arc 1e-8 ^
  --max-arc 0.05 ^
  --max-increments 5000 ^
  --field-frequency 1 ^
  --postpeak-stop-ratio 0.70
"%RUN_DIR%\imperfection\PM_GMNIA_FY240_LDG\run_pm_queue_70_pct.bat"
python abaqus_pm_interaction.py ^
  --mode collect ^
  --roots "%RUN_DIR%" ^
  --fy 240 ^
  --output-dir "%RUN_DIR%\PM_diagram"
```

`--models` (Step-5 syntax, e.g. `--models PERFECT STEP4_LDG_pp`) limits the
GMNIA cases; job names are `PM_P00_<case>` and `PM_A<axis>_P<psi>_<case>` and
must stay within 38 characters. The plastic curve is a section reference
(no local buckling, no residual stress); member capacities come from GMNIA.

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

## Engineering integration plan (October 2026)

The implementation brief is the supplied comprehensive pipeline review. Changes are
made in existing source/tests/docs only. Existing analysis artifacts are not edited
or regenerated during development; verification uses synthetic data and temporary
directories, with no production Abaqus solves.

1. **Resources:** share hardware discovery; adapt working sets to live RAM/VRAM;
   avoid nested BLAS oversubscription and the Windows process-pool limit. Preserve
   a CPU-only execution path and use no fixed resource-reservation percentage.
2. **Modal mechanics:** require consistent matrix/model provenance and numerical
   acceptance; never silently remove contact or preload. Distinguish the exact
   full-mode energy from auxiliary minimum-energy class diagnostics and their
   confidence. Reject unsupported constraint semantics explicitly.
3. **Reference workflow:** keep model generation, solve and classification distinct;
   reference classification must accept `--stiffness required` without pretending
   that the unimplemented equation-constrained energy diagnostic was evaluated.
   Self-tests must not dispatch solver/export jobs.
4. **Imperfections:** use individual versus eigenspace participation correctly;
   require complete degenerate global spaces, invariant physical orientation and
   correspondence with the source mesh. Keep amplitude/sign definitions explicit.
5. **GMNIA/queue:** preserve imperfect geometry and original results; check solver
   resource overrides (including zero GPUs), reuse only matching completed jobs,
   and keep stopping criteria consistent with the generated solver artifacts.

Each substantive behavior change gets a regression that fails before its fix.
Targeted checks and the lightweight suite run with bounded wall-clock time. Full
Abaqus matrix generation, reference solves and nonlinear capacities remain for
the user's subsequent production verification.

## Integrated execution after the engineering review

Use a **new OUTPUT_ROOT** for production verification. This review did not run
Abaqus or recalculate existing results. The sequence below is Windows CMD syntax;
keep the repository intact on the execution machine. Python needs NumPy/SciPy and
Matplotlib; CuPy and psutil are optional. Abaqus 2024 and a compatible Fortran
compiler are needed for the solver/URDFIL stages. Without a compiler, pass
`--stop-method monitor` to the Step-5 builder; monitoring stops at the first
observed crossing and may overshoot the exact increment.

```bat
set "REPO=D:\Abaqus_run"
set "CASE=YOUR_CASE"
set "BUILTUP_DIR=D:\CUFSM-Single\cfs_abaqus\%CASE%"
set "OUTPUT_ROOT=D:\D-CFS-Column\ReviewedRuns"
set "RUN_DIR=%OUTPUT_ROOT%\%CASE%"
set "REF_G=%RUN_DIR%_REF_G"
set "REF_L=%RUN_DIR%_REF_L"
cd /d "%REPO%"

rem 1-2: build and solve once; GDLC will export the required modes next.
abaqus cae noGUI=abaqus_complete_model_m20.py -- ^
  --builtup-dir "%BUILTUP_DIR%" --output-root "%OUTPUT_ROOT%" ^
  --mesh-mm 20 --n-modes 250 --n-vectors 500 --max-iterations 1250 ^
  --cpus auto --nodal-precision full ^
  --longitudinal-lines 4 --longitudinal-line-min-spacing-mm 5 --skip-post

rem 3: original-model classification and checked elastic matrix export.
python gdlc_classifier\gdlc_mode_decomposition.py "%RUN_DIR%" ^
  --export auto --stiffness required --cpus auto

rem 4: generate references only; do not generate references of references.
python gdlc_classifier\gdlc_mode_decomposition.py "%RUN_DIR%" ^
  --make-reference-models --reference-modes 20

rem 5-6: solve each missing reference once, then export/classify both.
python gdlc_classifier\gdlc_mode_decomposition.py "%REF_G%" "%REF_L%" ^
  --solve-reference-models --export auto --stiffness required --cpus auto
```

The first command combines building and solving without a duplicate wavelength/
audit pass. To inspect the model first, replace `--skip-post` with `--build-only`;
read `pipeline_status.json` → `build.job_name`, then submit that matching INP in
RUN_DIR with `abaqus job=NAME input=NAME.inp interactive`. Its generated environment
contains the detected CPU/GPU settings. Resume with command 3; no extra datacheck
or `--resume-post` is required for this GDLC workflow. Command 4 also works directly
after `--build-only` (no solved `.odb` needed): the reference inputs are then made
from the run's `.inp` geometry. RUN_DIR is always `<OUTPUT_ROOT>\<folder name of
BUILTUP_DIR>` (`_run02`, `_run03`, ... when that folder already exists; see the
`Output directory:` line of `pipeline.log`), and the reference jobs are named
`<build.job_name>_REF_G` / `_REF_L` in `<RUN_DIR>_REF_G` / `_REF_L`. A datacheck is optional,
but consumes the same solver job destination: use a separate datacheck job name
if the automatic reference solver is to run afterwards. The reference `*Buckle` uses
Lanczos for the 20 lowest eigenvalues >= 0 (`20, 0`) for compression and bending alike, so
the search never wanders into negative eigenvalues (the reversed load; for compression,
tension), which can stall the solve or make a negative mode number 1. A reference folder
that already holds results from an older input is never overwritten: move it aside and rerun.

**Select modes from the new reports before continuing.** Check individual shapes,
individual shares, group shares and confidence in original/REF reports. Neither
an old “99.99% D” nor a group-dominant label establishes individual purity. No
mode number is embedded as a universal choice. Choose flexural global modes for
the explicit X bow; torsion needs its own imperfection case without a bow angle.
The variables below accept one global mode (automatic complete-group expansion
when needed) or a comma-separated complete group. Amplitudes below preserve the
user's example; they are not asserted to be a code-prescribed universal envelope.

Replace each `SELECT_...` value below with the mode number(s) you selected before
pasting these commands into Windows cmd. These are explicit assignments, not
interactive `set /p` prompts: pasting multiple prompts together can consume the
next command as the mode value. Quote all expanded mode variables. A message
`Exit code: 0` from CAE alone does not prove Step 4 succeeded; require its `SAVED:`
message and the new output CAE. Input errors are also printed as `STEP4 INPUT ERROR`.

```bat
set "LOCAL_MODE=SELECT_REF_L_MODE"
set "DIST_MODE=SELECT_ORIGINAL_MIXED_MODE"
set "GLOBAL_MODES=SELECT_REF_G_FLEXURAL_MODE_OR_GROUP"
cd /d "%REPO%"

rem 7: nine existing built-ins plus four explicitly signed combinations.
abaqus cae noGUI=abaqus_step4_imperfections.py -- ^
  --run-dir "%RUN_DIR%" --local-source "%REF_L%" --local-mode "%LOCAL_MODE%" ^
  --local-high-t 1 --dist-mode "%DIST_MODE%" --dist-mm 3.6 ^
  --global-source "%REF_G%" --global-mode "%GLOBAL_MODES%" --global-angle-deg 0 ^
  --cases all --combo LDG_pp=0.34t,1.8t,L/1000 ^
  --combo LDG_pm=0.34t,-1.8t,L/1000 --combo LDG_mp=-0.34t,1.8t,L/1000 ^
  --combo LDG_mm=-0.34t,-1.8t,L/1000 ^
  --output-cae "%RUN_DIR%\imperfection\Step4_LDG.cae"

rem 8: new output directory; prepare inputs, no solver submission.
abaqus cae noGUI=abaqus_step5_gmnia.py -- ^
  --source-cae "%RUN_DIR%\imperfection\Step4_LDG.cae" ^
  --output-dir "%RUN_DIR%\imperfection\Step5_GMNIA_FY240_LDG" ^
  --fy 240 --initial-arc 0.01 --min-arc 1e-8 --max-arc 0.05 ^
  --max-increments 5000 --field-frequency 1 --postpeak-stop-ratio 0.70 ^
  --cpus auto --gpus auto

rem 9: execute the generated queue; resource policy is resolved again here.
cd /d "%RUN_DIR%\imperfection\Step5_GMNIA_FY240_LDG"
call run_step5_queue_70_pct.bat
```

### What the corrections establish

- A shared hardware policy uses live memory budgets, optional CUDA with CPU
  fallback, all usable visible GPUs for projection, bounded reader/process
  concurrency and parallel sparse right-hand sides. Windows process pools
  respect their platform limit; no RAM/VRAM reserve percentage is imposed.
- Matrix/model mismatches and failed numerical acceptance refuse the energy
  diagnostic. Contact is retained. Old matrix outputs are preserved; a required
  new matrix job uses a fresh numbered `stiffness_export_*` directory. Cache
  selection verifies source-input and matrix/DOF hashes.
- Kinematic participation is distinguished from full-mode energy and auxiliary
  minimum-energy component diagnostics. The latter relax in-plane bolt rows;
  they are not an admissible pure-family buckling decomposition. Confidence
  includes unexplained energy and component incompatibility. REF equation
  constraints remain outside the energy mapper and are reported as such.
- Active Buckle boundary conditions use the correct load-case precedence.
  Global imperfection orientation uses complete eigenvalue groups and a physical
  SVD solution invariant to the returned eigenvector basis. Step 4 checks source
  mesh correspondence and classifies individual shares separately from groups.
- Step-5 sidecars bind the job to its final INP hash. URDFIL source/stop ratio are
  checked before submission, including full-precision batch serialization. GPU
  zero is an explicit solver override. Existing solver artifacts are neither
  moved to backups nor overwritten. LPF prefix extraction removes duplicate
  initial zero samples before alignment; pure P/M endpoint roundoff is handled
  at the section scale.

### Verification boundary

Verification uses the existing synthetic/mock tests with temporary outputs and
bounded execution time. It checks matrix mapping, rigid motions, constraints,
energy lifting/acceptance, chunk and allocation fallback, basis invariance,
input/stop contracts and LPF extraction. These checks do **not** establish
physical validation of Abaqus capacities. Actual Abaqus-2024 matrix generation
with retained contact, reference equation solves, rigid connector equivalence
in nonlinear motion, GPU execution and full GMNIA convergence/capacity still
require the user's production runs. Sparse LU must fit in RAM; unsupported
preload/constraint/model semantics fail explicitly rather than returning a
misleading classification. The model-generation/Step-4/Step-5 builders retain
their existing uniform-thickness four-piece S4R scope; arbitrary FE topologies
are not silently accepted.

Implementation references: Abaqus documentation for
[matrix generation and MPC/DOF treatment](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-mtxgenerationperturbation.htm),
[Buckle base states and perturbation boundary conditions](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-eigenbuckling.htm),
and the Python documentation for the
[Windows process-pool limit](https://docs.python.org/3.14/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor).
The referenced Abaqus pages are the accessible 2025 documentation; actual
Abaqus-2024 execution remains to be verified.

Recorded lightweight verification (2026-10-08): the full existing suite plus the
new regressions passed **310 tests in 79.07 seconds**. After the final input-path
and startup-lock corrections, the affected GDLC suites passed again: **56 tests
in 11.80 seconds**. The two full-suite warnings intentionally exercise CPU
request clamping. Python AST parsing passed and `git diff --check` found no
whitespace errors. Relative to the pre-task file hashes, no new source/document
files or deletions were introduced. Tests used temporary synthetic/mock outputs;
no production Abaqus analysis or existing-result recalculation was performed.

### Sequential family pipeline and shared comparison

Run `abaqus_family_pipeline.py` with normal Python and the existing `--family-root`,
`--output-root`, `--repo`, mesh/eigenvalue options, `--local-mode`, `--global-modes`
and optional `--resume`. Outputs remain under
`<output-root>/<family-root-name>/<family-name>/`; each family retains its G/L
references and the PERFECT plus LG_mp Step5 cases. No added steel plates are
introduced by this pipeline.

Reference stages use `--solve-reference-models --solve-only`: successful REF_L/REF_G
analyses proceed directly to Step4, with no modal export, stiffness job, classification
or purity gate. Step4 receives `--skip-mode-quality-checks`; selected vectors are
explicitly marked unassessed. Missing modes, incompatible meshes/load cases, failed
solves and invalid/zero displacement fields still stop execution. Standalone GDLC
classification and Step4's default quality checks remain available.

Step5 execution completion and post-peak verification are distinct. A successful
analysis with synchronized results may end at an increment/displacement limit before
the requested drop; the monitor only terminates on an observed crossing. Such a run
continues the queue/pipeline as `CRITERION_NOT_REACHED`, with `capacity_verified=false`
and `peak_interpretation=OBSERVED_MAXIMUM_ONLY`. Its curves are labelled accordingly;
the maximum remains recorded, but Pu/Mu ratios and P-M capacity points are withheld.
Resume verifies positive solver completion plus saved curves and reuses these runs
without solving/extracting again. Aborted, unknown or incomplete results still fail.
The stopping ratio and Riks analysis limits are not relaxed or extended automatically.
`totalArcLength=1` is a scale factor, not a Riks termination limit. The optional
`--riks-extrapolation none` changes only the Step5 initial predictor; the default
`linear` preserves existing behavior. Resume treats a missing legacy setting as
`linear` and refuses a different requested predictor for an existing Step5 build.

Paths are taken from the command line on the current computer. `--repo` is optional:
its default is the directory containing `abaqus_family_pipeline.py`, independent
of the terminal's working directory. An explicitly supplied invalid `--repo` is
rejected; it is never silently replaced by another installation. Copy the complete
code directory, including `gdlc_classifier` and sibling helper modules.
`--abaqus-cmd` accepts a launcher name in PATH (default `abaqus`) or its full path,
including spaces. The selected launcher reaches the reference solves/exports and
the Step5 queue's job submission and termination commands. Normal-Python stages
use the Python interpreter that launched the pipeline. These options locate code
and programs; they do not rewrite saved analysis provenance. Existing `--resume`
path/hash checks still apply to results copied from another location.

The batch directory receives `STEP5_family_force_displacement.svg` and, when
normal Python has matplotlib, its `.png`. Every family has a distinct color:
PERFECT is solid and LG_mp is dashed. Axes are the existing compressive load in
kN and area-weighted end shortening in mm; saved Riks increment order is retained,
including any displacement reversal. No interpolation or normalization is added.
The plot includes verified completed families on resume and updates after each
new completion. It reads saved curve CSVs, not ODBs, and keeps one pair per family.

`--resume` verifies original input/settings, exact bolt endpoints, reference-mode
provenance, selected modes/amplitudes and final Step5 input hashes. Retouched CSV
timestamps alone do not invalidate a matching physical input snapshot. Completed
families are not rebuilt or solved; a partially solved Step5 queue reuses validated
finished curves and submits only jobs with unused solver destinations. Comparison
plot failures can be repaired from those CSVs without another solve/extraction.
Current `--cpus` and automatic supported GPU selection reach the queue directly,
independently of the saved BAT's original machine settings; existing all-core and
100% solver-memory policies are retained.

Future Step4 builds write a small `family_pipeline_step4.json` checkpoint for
recovery before Step5. Legacy Step4-only CAEs without that checkpoint or Step5
sidecars, interrupted Step5 builds lacking their final CAE/queue, running jobs,
and unsuccessful occupied solver destinations are preserved and refused rather
than overwritten or silently rerun. Lightweight checks use synthetic/mock jobs;
actual Abaqus execution and convergence still require the user's full run.
