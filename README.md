# Abaqus_run

## Sequential family CLI

Run `D:\Abaqus_run\abaqus_family_pipeline.py` with normal Python (not Abaqus
Python). Inputs are `--family-root`, `--output-root`, `--repo`, the builder mesh/mode/
CPU/output options, and `--local-mode` / `--global-modes`. `--help` lists them.
The wrapper resides in `D:\Abaqus_run`; it does not edit the existing model scripts.
All immediate family directories are checked before output creation and processed
in numeric `family_N` order. Output is `OutputRoot/FamilyRoot.name/family_N` with
sibling `family_N_REF_L` and `family_N_REF_G`. Existing destinations cause a refusal;
partial runs are preserved and automatic resume is not implemented.

The global imperfection is one eigenvalue group of the solved REF_G: an explicit
`--global-modes` list that is not one complete group (e.g. `1,2` in bending, where the
lateral-torsional mode 1 is single) is resolved per family to the complete group of its
first mode, with a WARNING, instead of stopping in Step 4.

For each family, build the original without solving, create 20-mode G/L references
(Lanczos, the 20 lowest eigenvalues >= 0; with `--resume` only references without a
completed solve are regenerated, and an unfinished attempt is preserved in
`<ref>/superseded_attempts/`, see `gdlc_classifier/README_GDLC.md`, "Eigensolver"),
solve/classify L then G with stiffness required, build only the imperfection
`LG_mp=-0.34t,0,L/1000` at global angle 0, then build Step5 and execute its generated
70% queue. The next family starts only after queue outcomes and comparison output
are checked. Step5 retains its PERFECT baseline; fy is fixed at 240 MPa, Riks arc
settings at 0.01/1e-8/0.05 and max increments at 5000. Selected mode numbers must
be within 1..20 and are passed to every family; this does not select modes from
engineering criteria automatically. `--cpus` is forwarded to every solver stage;
the existing stages handle local capacity and GPU detection. No steel plates are
added by this workflow. Both reference solves and Step5 analyses can take a long
time when the user executes this CLI.

## Family repository: transverse steel plates

In `D:\Abaqus_run_family`, add `--steel-plate-length 40` to the existing builder
command. This creates one XZ plate at y=0 per bolt row, centered on its Z
position, spanning the full section X extent. Thickness equals `gap_mm`, not
the distance between shell midsurfaces (`gap_mm + t_mm`). Plates use S4R shells,
9 section points and source E/nu. Height zero (default) retains the original model.
Out-of-member or touching/overlapping plate intervals are rejected before CAE build.

Only the two seams crossing y=0 pass through the plates. Each existing ideal
BEAM MPC there becomes two segments sharing the plate node; physical bolt
count is unchanged for this single-direction option. There are no connections to
the column beyond the existing ideal bolts and contact; no added friction, holes or pretension.
General hard/frictionless contact includes the plates and permits separation.
This remains an ideal-bolt/shell model, not a solid bolt-bearing/plate-thickness
compression model. Buckle freezes contact at the base state as before.
Because each plate is exactly as thick as the clear gap, its faces touch the walls
at zero nominal clearance, which the mesh coordinates scatter by about +-1e-6 mm.
Plate models therefore close initial openings below 1e-4 t strain-free (general
contact initialization `SnugFit`, recorded in the build report): the frozen
base-state contact of the buckling step is the bolt-clamped snug fit everywhere,
not a rounding-dependent mix of closed and open points. Real gaps stay open.
Models built before this change keep that mix; rebuild them before generating
reference models.

Add `--steel-plate-length-2 40 --steel-plate-gap-mm 0` to create connected
cross-shaped XZ/YZ plate assemblies. In this zero-gap mode the second height
parameter determines the common Z height of BOTH plates. The first parameter
still enables plates and controls their height in single-direction or
positive-gap mode. Each cross is a single native shell part, merged before
meshing, with shared translational and rotational DOFs along its entire
central intersection. The four arms must share the same intersection nodes;
the builder checks this rather than relying on visual overlap or contact.
The junction is an ideal continuous shell connection, not a weld-strength model.

Zero-gap crosses use the exact source bolt rows in ALL four seams, including
the first and last bolts. The CSV reader is the same as in the reference
builder: no pitch reconstruction, rounding, shifted bolts or extra rows.
At a plated row, each bolt is split into two ideal BEAM MPC segments through
its plate node. Both plate directions refer to the same cross instance.

Positive `--steel-plate-gap-mm` retains the previous separated, staggered
placement: each direction uses its own height, second-direction end bolts
remain bare, and its new plated rows shift inward by
`(height1 + height2)/2 + edge_gap`. Complete plates violating the clear edge
gap or colliding at the center are omitted symmetrically, as before. There is
no central connection between these separated plates.

For the supplied 3600 mm, 19-row, 25/25 mm end-offset model with heights 40/40,
gap 0 and skip 0, there are 19 cross assemblies, 76 physical bolts and 152
ideal bolt link segments. The first cross spans Z=5..45 about the bolt at 25;
the last is centered at 3575. Source and selected plate rows are recorded in
`pipeline_status.json`.
Both plate directions share `SteelPlate` and the Step5 `--steel-plate-fy` value.
The dual-direction output/job suffix is `_PL40_PL240_PG0` (`PL2` identifies the
second height); nonzero gaps or changed heights produce distinct suffixes.

Add `--steel-plate-skip-count N` to retain only selected plates in both
directions. N must be a nonnegative integer; the default 0 retains every
eligible plate and the existing output naming. Zero-gap crosses always
occupy the first bolt at each end. From each end inward, retain one cross,
skip N screw positions, retain the next cross, and repeat toward the center.
N=1 alternates; N=2 skips two positions; a value exceeding each half's
interior row count keeps only the edge plates. A shared center is kept at
most once. Positive-gap placement keeps its original outermost plates and
bare second-direction end screws.

The full screw layout, including any positive-gap second-direction rows
and original collision exclusions, is calculated before filtering plates.
No screw is moved or removed: at a skipped plate position its screw remains
as the original direct BEAM MPC instead of the two plate-connected segments.
The original screw rows and selected/skipped plate centers are recorded
separately for subsequent stages. Positive N adds `_PSN` to the output/job
suffix. For the 3600 mm zero-gap example above, N=1 gives 10 cross assemblies,
76 physical screws and 116 ideal link segments. Selected cross intervals
must not touch/overlap or extend past a column end; impossible end plates
are rejected rather than dropped or shortened.

`--output-root` appends `_PL40` to the input folder name, and the job gets the
same suffix; existing outputs are never overwritten. `--build-only` writes CAE
and INP without solving. Use the scripts in this family repository for all
subsequent stages. Generate new G/L references from this same plate model;
old four-instance reference ODBs are incompatible.

Step5 requires **`--steel-plate-fy FPRIME_Y_MPA`**, independently of `--fy` for
the column members. Plate behavior is elastic-perfectly-plastic; any member
`--stress-strain` curve does not affect the plate. The plate material remains
elastic during the buckling/reference stages. See the Step4/Step5 and GDLC
notes for the scope of imperfection normalization and reported family shares.

Verification: small CAE build-only checks are available via the optional
`CFS_STEEL_PLATE_SMOKE_DIR` integration test in `test_complete_pipeline.py`.
They operate on a temporary short fixture and never submit a solver.

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
is first yield, `P/Py + M/My = 1`. Elastic buckling (`buckling`) applies the
stress as consistent nodal forces on the meshed end edges (same BCs, mesh, bolts
and contact as the reference model). GMNIA (`gmnia`) retains the same consistent
nodal end forces and inherited supports with free end warping, following the
supplied working Step5 reference (see README_step5_gmnia.md, "End condition").
End yielding may govern capacity under this prescribed loading. Rays: `--axes required|all|0,45` (required = one axis per
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
or `--resume-post` is required for this GDLC workflow. A datacheck is optional,
but consumes the same solver job destination: use a separate datacheck job name
if the automatic reference solver is to run afterwards.

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


### Family, bending and P-M workflows with connected plates

`abaqus_family_pipeline.py` is the family entry point in this folder. It uses
its own installation by default (`--repo` can select another complete installation).
It forwards `--steel-plate-length`, `--steel-plate-length-2`,
`--steel-plate-gap-mm`, and `--steel-plate-skip-count` to the builder. The pipeline defaults
`--steel-plate-fy` to 360 MPa for the added connector/cross plates and
`--steel-column-plate-fy` to 240 MPa for the column walls. These are independent:
the column value is passed as the existing Step5/P-M `--fy`, including capacity
normalization and diagram collection. Elastic builds, reference models and Step4
keep their existing elastic properties; yielding starts in Step5. The connected-cross geometry, bolt coordinates, sparse plate
pattern, full plate mesh and contact/MPC connections are preserved.

Use external CPython for this entry point, not `abaqus python`. On machines
where `python` opens the Microsoft Store alias, select the installed interpreter
explicitly with `py -3.14`. Install NumPy into that same interpreter; installing
it into Abaqus or another Python does not satisfy GDLC's import. Matplotlib
enables family PNG plots and is required for P-M diagrams:

```bat
py -3.14 -m pip install --user --only-binary=:all: numpy matplotlib
py -3.14 -B -c "import sys, numpy, matplotlib; print(sys.executable); print(numpy.__version__, matplotlib.__version__)"
```

NumPy alone is sufficient for REF_L/REF_G input generation and solve-only
orchestration. SciPy (stiffness classification) and PyArrow (legacy Parquet
exports) are separate optional dependencies; install them with the same
`py -3.14 -m pip install --user --only-binary=:all: scipy pyarrow` command only
when using those features. Binary-only installation requires compatible wheels
and fails explicitly if the configured package index cannot supply them.
If pip reports no candidates, check its index configuration and `PIP_NO_INDEX`;
this execution session had `PIP_NO_INDEX=1` and installation was not performed.
Run installation from a normal network-enabled terminal.

The pipeline already launches GDLC and external collection subprocesses with
`sys.executable`, so they inherit this interpreter. CAE, ODB access and solver
operations continue through the Abaqus launcher. The checked Abaqus 2024
installation uses Python 3.10.5 with bundled NumPy 1.22.4; leave that environment
intact and do not add external Python's `site-packages` to `PYTHONPATH`.

For the reported installation, use
`C:\Users\810200014.HAMI.001\Documents\Abaqus_run_family - Copy\abaqus_family_pipeline.py`.
The separately named `C:\Users\810200014\Downloads\Abaqus_run_family - Copy`
was absent; changes in one copy do not update another. To continue the failed
execution, replace its initial `python` with `py -3.14`, use the Documents
script above, retain all original arguments/input/output paths, and add
`--resume`. If `--repo` was supplied, point it to that same Documents directory.
Keep `--cpus auto` for automatic solver resource selection. Interpreter/package
selection applies equally to all geometries and skip counts, including
`--steel-plate-skip-count 0,1,2,3`; retain the failed run's original selection.

Example (Windows CMD, replace input/output paths):

```bat
py -3.14 "D:\Abaqus_run_family - Copy\abaqus_family_pipeline.py" ^
  --family-root "D:\input\section_families" ^
  --output-root "D:\results" ^
  --cpus auto --mesh-mm 20 --n-modes 250 --n-vectors 500 --max-iterations 1250 ^
  --steel-plate-length 40 --steel-plate-length-2 40 ^
  --steel-plate-gap-mm 0 --steel-plate-skip-count 1 --steel-plate-fy 360
```

Add `--load-case bending --bending-axis-deg 0` for bending. Compression remains
the default. `--resume` checks geometry, plate settings/material, source inputs,
mode sources and delivered Step5 INPs before reuse; existing solver results are
not overwritten. Family plots update incrementally (perfect solid / imperfect
dashed, one color per family), using force-shortening for compression and
moment-rotation for bending. Solver completion, post-peak verification and a
usable stopped curve are separate states; unverified maxima remain provisional.

`abaqus_pm_interaction.py` retains `section`, `buckling`, `gmnia`, and `collect`
modes. GMNIA accepts the existing Step5 options, including `--steel-plate-fy`.
Its template retains the plates, contact and bolt connections; only proportional
consistent nodal CF3 forces are varied along each P-M ray. Recorded full-model symmetry controls
perfect-model axis selection. Explicit distinct orientations remain allowed for
imperfect GMNIA specimens. Collection plots dimensional P (kN) versus M (kNm)
by default (`--normalized` is optional), separates specimens/load directions,
and distinguishes verified capacity from provisional response points.

GDLC references preserve the source mesh, loads, bolt MPCs, plate flexibility
and contact. REF_G also constrains offset plate-bolt control rotations to the
rigid ring. Automatic twist is restrained for compression, as requested: REF_G
then represents global flexural buckling only. Bending/combined references keep
twist free for lateral-torsional buckling. Unbounded
reference spectra use subspace iteration without an invented eigenvalue cutoff;
bounded Lanczos searches retain the source range. The family pipeline defaults
to REF_G mode 1, without a prescribed compression bow direction or assumed
flexural pair. Explicit mode/group requests remain available. The classifier
retains the lowest-stress G candidate in reports and figures even when its
confidence is low, without upgrading that confidence. Resume checks require
the matching load-case twist policy. A twist-restrained reference does not
establish the full torsional/flexural-torsional Global spectrum; changed
references require a new output destination.
REF_L uses exactly the classifier's virtual-corner translation constraints,
including short rounded corners; it does not substitute shell rotations.
Reference mode quality, especially attachment-dominated modes, must still be
assessed on solved results before claiming a pure-family critical load. The
family pipeline solves references directly without running modal classification;
bending Step4 still checks source consistency and physical lateral direction.
GDLC uses verified full-precision input coordinates when modal archives contain
FP32-rounded geometry, and a scaled QR basis without relaxing self-test criteria.

These integration checks are synthetic/read-only or mocked and do not verify
production Abaqus convergence, reference-mode purity or ultimate capacities.
Regeneration with changed constraints requires a new output destination; old
solved inputs/results are preserved.


### Independent skip-count sweeps

The family pipeline accepts `--steel-plate-skip-count 0,1,2,3`. Each value uses
the existing scalar builder and full downstream workflow sequentially, with all
configured CPU/GPU/memory resources available to that job. Repeated/negative,
noninteger or empty list entries are rejected before execution. All configuration
destinations are checked before the first build. The plate pattern itself is
unchanged: one plated position followed by N unplated positions, symmetric from
both ends, with both first bolt positions always plated.

Sweep layout:

```text
<OutputRoot>/<FamilyRootName>/skip_0/<family model and its REF_L/REF_G folders>
<OutputRoot>/<FamilyRootName>/skip_1/<family model and its REF_L/REF_G folders>
<OutputRoot>/<FamilyRootName>/skip_2/<family model and its REF_L/REF_G folders>
<OutputRoot>/<FamilyRootName>/skip_3/<family model and its REF_L/REF_G folders>
```

Each `skip_N` has its own family comparison plots; each model has its own
imperfections, Step5 inputs, solver outputs and reports. A scalar argument keeps
its previous output layout. Resume a sweep with the same list/output root;
completed stages/configurations are reused after provenance checks. Existing
scalar outputs are left in their original locations and are not silently moved
or substituted for sweep destinations.

The supplied multi-value command works without another material argument:
**the pipeline uses plate F'y = 360 MPa**, as previously specified, and member
fy = 240 MPa. Use `--steel-plate-fy` to override the connector plate value and
`--steel-column-plate-fy` to override the column wall value. Standalone Step5
and P-M generators still require that independent plate material explicitly.

Add `--load-case bending --bending-axis-deg 0` to sweep bending configurations.
Add `--pm-interaction` to generate and solve the established proportional P-M
GMNIA rays for PERFECT and LG_mp from each configuration's own Step4 CAE.
`--pm-axes required` and `--pm-angles 0,15,30,45,60,75,90` are defaults; these
options can be overridden. Explicit opposite/orthogonal axes may be necessary
for an imperfect specimen even when its perfect section is symmetric.

P-M inputs/results are under each model's `imperfection/Step5_PM_FY240_LG_REF1`
(`_REF1`: supplied working reference loading with inherited nodal supports and free end warping;
older folders, including `_LG_EP4`, are preserved and never reused);
the FY tag in default Step5/P-M directories, job names and comparison plots follows
the column yield stress (legacy FY240 names stay unchanged). Resume rejects
incompatible yield settings or plastic tables in an existing Step5 destination;
P-M receipts bind both yield values. Changing column yield may reuse matching
elastic/reference/Step4 data while generating a separate nonlinear destination.
its `diagram` folder contains the interaction CSV/SVG/PNG. Resume verifies the
P-M request, Step4 provenance and input/metadata hashes; existing completed
curves are reused, and missing diagrams can be repaired without repeating the
queue or analyses. Stopped/unverified responses retain their provisional status.
P-M collection distinguishes both nonlinear and elastic datasets, deduplicates
overlapping search roots, and normalizes elastic results against their own
section reference. A common plastic reference is omitted when sections differ.
