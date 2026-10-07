# GDLC mode decomposition: how to run

`gdlc_mode_decomposition.py` splits every Abaqus buckling eigenmode into
**G** (global), **D** (distortional), **L** (local), **C** (relative rigid motion of the pieces of a built-up section) and **O** (other).
It reads supported prismatic, conforming four-node shell meshes from `portable_modal_export*/` and the matching `.inp`. Automatic axis detection requires at least two independent shell-normal directions; a standalone flat plate has an ambiguous axis and is rejected. The transverse bending operator assumes uniform thickness and homogeneous stiffness. Mixed thickness and composite layups are rejected; spatially varying material stiffness is not represented.

**Research interpretation (audit, October 2026).** This is a project-specific,
section-wise kinematic decomposition inspired by cFSM. It is not a validated
implementation of full cFSM/GBT, a strain-energy partition, or a separation of
the eigenvalue into independent critical loads. Algebraic reconstruction tests
establish internal consistency; they do not establish physical uniqueness.
`D_assembly = D + C` is a useful project convention, not an established equivalence
to the classical distortional family for every built-up section. Cluster results
describe the retained eigenspace, not each eigenvector independently.

Use the existing synthetic checks as regression checks and compare representative
sections against an independent cFSM/GBT implementation before treating these
shares as quantitative research evidence. Historical case-specific numbers below
were not regenerated during this audit and are not acceptance limits for new models.

## Requirements
Python ≥ 3.10, `numpy`, `pyarrow`. Optional packages:
- `matplotlib`: plots
- `psutil`: RAM-aware chunking
- `cupy`: CUDA GPU (used automatically when present)

## Integrated pipeline (ODB → export → classification)
`gdlc_mode_decomposition.py` now drives the whole chain. `abaqus_mfsm_export.py` must sit in the same folder; it is self-contained and needs no other modules. For every run folder found below the given path(s), the program:
1. **Solves reference models** (only with `--solve-reference-models`): every `<run>_REF_G` / `<run>_REF_L` folder whose `.odb` is missing, failed or older than its `.inp` is solved with `abaqus job=… cpus=<all> interactive`.
2. **Exports** `<job>.odb` by calling `abaqus python abaqus_mfsm_export.py --odb … --inp … --output-dir <run>\modal_export_gdlc`.
   - `--export auto` (default): an existing export is reused when the sha256 values it recorded match the current `.odb` and `.inp`.
   - `--export always`: a new export is written every time.
   - `--export never`: only existing exports are used.
   - The exporter never overwrites; a later export gets `_2`, `_3`, … appended to the folder name.
3. **Classifies** the run.

Which folders count as runs:
- any folder that holds a modal export (`*/modal_export.json`, in either the npz or the legacy parquet format);
- any folder with `<job>.odb` + `<job>.inp`, where the `.inp` has a `*Buckle` step, plus `<job>_build.json` or `gdlc_reference.json`.

Other `.odb` files, such as GMNIA or single-piece studies, are ignored. For npz exports, the element connectivity is read from the `.inp`, and the program checks that this `.inp` has the sha256 recorded in the export.

## Run
```
python gdlc_mode_decomposition.py "D:\CFS-Column"                  # every run below this folder
python gdlc_mode_decomposition.py "D:\CFS-Column\<run>"            # a single run
python gdlc_mode_decomposition.py "D:\CFS-Column" --skip-existing  # only runs without results
python gdlc_mode_decomposition.py <run> --selftest-only            # basis + checks, no modes read
python gdlc_mode_decomposition.py --benchmarks                     # analytical checks only
python gdlc_mode_decomposition.py "D:\CFS-Column" --make-reference-models   # write pure-G / local reference .inp files
```

## Reference models (pure G and corner-restrained L), built by Abaqus itself
`--make-reference-models` writes two folders next to every run, `<run>_REF_G\` and `<run>_REF_L\`. Each holds a modified copy of the run's `.inp` (20 eigenvalues by default, `--reference-modes`), the `_build.json` and `gdlc_reference.json`.
- **REF_G**: every cross-section ring is kept rigid in its own plane. Each ring has a kinematic coupling of its in-plane DOFs to a ring reference node. Axial displacements (warping) and seam slip stay free, so the eigenmodes are pure global: flexural, torsional or flexural-torsional, with the real partial composite action of the bolted member. The first eigenvalue is the **pure-G critical stress**.
- **REF_L**: the virtual sharp corner of every fold is held in its plane along the whole member, while the fold itself stays free to rotate. This is written as one `*Equation` per in-plane direction on the fold's reference node: u + θ×(c − p) = 0.
  - θ is the arc's chord rotation (u_b − u_a)·n/h, averaged over the arc edges that do not touch the reference node, so each equation is term for term the classifier's fold-driver row (checked to 1e-12). G and D therefore cannot occur.
  - The seams stay as built. With enough independent fold constraints on every piece, REF_L suppresses fold-driven motion. Pieces with no folds or only one fold can retain rigid motions; skipped dependent nodes and rotation-based fallback equations can also leave motion. The output check, rather than the folder name, determines whether the result is predominantly L.
  - Earlier versions: holding the arc node itself let the corners move by θ·|c − p| (up to 10 % spurious G+D). Using the shell rotation UR instead of the chord rotation left a few % D, because the FE arc is not exactly rigid.

**Eigensolver.** Both reference models use the **Lanczos** eigensolver (`*Buckle, eigensolver=LANCZOS`, same number of modes, no eigenvalue range, so the same lowest eigenvalues as before).
- The local spectrum of a long member is dense: the 40 lowest REF_L eigenvalues of the A4784 run lie within 1 %. Subspace iteration converges like (λ_n/λ_q+1)^k; with 40 vectors it needed 484 iterations, 43 min. Lanczos (shift-invert, a few factorisations of about 1 s each) is not slowed down by such clusters.
- Abaqus refuses Lanczos for contact pairs or contact elements, hybrid elements, connector elements and distributing couplings. Such models are written for subspace iteration with 4n vectors.
- If Abaqus still refuses a Lanczos solve, `--solve-reference-models` logs the `***ERROR` line, rewrites the `*Buckle` for subspace iteration (4n vectors), re-solves once and records `lanczos_refused` in `gdlc_reference.json`; later regenerations then keep subspace.

The reference-model constraints are appended to the step's existing `*Boundary, op=NEW, load case=…` blocks. Abaqus requires `OP=NEW` for load case 2 of `*BUCKLE` and forbids mixing `OP=NEW` with `OP=MOD`, so no new `*Boundary` keyword is written. The source run's `abaqus_v6.env` is copied too.

A reference model is (re-)solved by `--solve-reference-models` when its `.odb` is missing, failed (no "COMPLETED SUCCESSFULLY" in the `.sta`/`.log`) or older than its `.inp`. Re-generating identical inputs does not touch the files. Once solved, the program exports and classifies the reference model:
  - it checks each reference model automatically: REF_G modes must come out ≥ 95 % G; REF_L modes must have ≥ 95 % L and ≤ 5 % G+D+C. These are kinematic consistency checks against the imposed constraints, not independent scientific validation;
  - `GDLC_summary\runs_summary.csv` includes `constrained_G/L_reference_eigenvalue` and `G/L_reference_check_passed`. Legacy `pure_G/L_reference_eigenvalue` columns are populated only when the kinematic check passes; their names do not imply independent pure-mode validation.
By default it uses every logical CPU (BLAS, I/O threads, plotting processes) and keeps all mode shards in RAM. It uses the GPU when CuPy and CUDA are available. You can change this with `--threads N` and `--gpu on|off`.

## Bending runs (v1.7.0)
Runs built with `abaqus_complete_model_m20.py --load-case bending` are found, exported, decomposed and given reference models exactly like compression runs; the commands are the same, with the `<input>_BEND000` / `_BEND045` folders.
- The decomposition is purely kinematic (cross-section deformation shares); it does not depend on the load. A rigid lateral-torsional motion of the real A4784 section comes out as 100.0 % G, and the REF_G / REF_L checks apply unchanged. Local and distortional modes of a bending run sit in the compression zone; that changes the shapes, not the classes.
- The eigenvalue of a bending run is the critical EXTREME-FIBRE stress (MPa) of the CUFSM-consistent linear reference stress. `sigma_cr_MPa` keeps that meaning. `mode_participation.csv` adds `M_cr_kNm = eigenvalue x M_ref`, and `mechanism_critical_stresses.csv` adds `first_dominant_M_cr_kNm` etc. (`M_ref` from the build json, `load_case.reference_moment_mesh_Nmm_per_MPa`). `basis_report.json` records the load case; `GDLC_summary\runs_summary.csv` has `load_case` and `bending_axis_deg` columns.
- The bending build uses Lanczos with eigenvalues >= 0 (`*Buckle` data line `n, 0.`). The reference models keep that range. If Abaqus refuses Lanczos and falls back to subspace iteration, which has no eigenvalue range, 2n eigenvalues are requested, because for a section with 180-degree symmetry each positive eigenvalue has a negative twin (the reversed moment).
- Modes with eigenvalue <= 0 (reversed load) are excluded from the classification, and their numbers are listed in `basis_report.json` (`excluded_nonpositive_modes`). Compression runs have none, so their results are unchanged (checked byte for byte on the A4784 REF_L run).

## Output (`<run>\mode_decomposition_GDLC\`)
| file | content |
|---|---|
| `mode_participation.csv` | per mode: eigenvalue, σcr, G/D/L/C/O % (normalised, cFSM convention), the same as exactly additive signed % (`*_signed_pct`), cross term, `D_assembly_pct` (= D + C), label, flags, C at bolt rings, seam axial-slip ratio, `fold_motion_ratio`, `halfwave_<X>_mm`. Each mode also carries its **reliable result**: `rel_*_pct`, `reliable_label`, `reliable_dominant`, `reliable_source` (= the cluster result when the mode belongs to a near-degenerate group, otherwise its own) and `confidence` (high / medium / low), plus its uncertainty band: `split_spread_pct` and `rel_<X>_min_pct` / `rel_<X>_max_pct` (see "How far the numbers can be trusted"). Flags and confidence always refer to the reliable result. |
| `cluster_participation.csv` | the same shares for groups of near-equal eigenvalues, with cross term, `D_assembly_pct`, dominant mechanism, margin, label, `split_spread_pct`, confidence and flags. These shares do not depend on which vectors Abaqus picked inside the group. |
| `longitudinal_profiles.npz` | per ring, per class, per mode: slice deformation along the member |
| `seam_jump_profiles_group1.npz` | seam opening/sliding and axial slip along each seam |
| `reference_points.png` | detected folds, free ends and seams. **Check this once per section family.** |
| `participation_summary.png`, `mode_profiles/` | summary bar chart and one profile plot per mode |
| `mechanism_critical_stresses.csv` | per mechanism: lowest eigenvalue/σcr at which it robustly dominates, the lowest at which its share first reaches 50 %, its maximum and mean share |
| `dominance_summary.png` | mean share and number of dominated modes per mechanism, and the share of each mechanism along the spectrum |
| `figures_dominant_<X>/` | cross-section figures of the top-k modes (default k = 3, `--top-modes`) of the run's dominant mechanism X. Each figure shows the bold section with the largest deformation, all other sections faint, the undeformed section dashed, the same panels for G/D/L/C, and the longitudinal profile with connector rows. |
| `figures_gallery/`, `figures_first_mode/` | the same figure for the top mode of each other mechanism, and for mode 1 |
| `figure_data.npz` | compact data used to redraw those modes (no shard needed) |
| `<common parent>\GDLC_summary\` | when ≥ 2 runs are processed: `runs_summary.csv`, `global_dominance.png/.json` and `figures_global_top_<X>/`, which holds the top modes of the mechanism that dominates over ALL runs, drawn for every run. `runs_summary.csv` also lists per run the median split spread, median \|cross term\|, median fold-motion ratio and the numbers of `DEFINITION_SENSITIVE` / `DOMINANT_UNSTABLE` modes |
| `basis_report.json` | all settings, detected geometry, space dimensions, self-test and benchmark results, fold-detection and arc-extent checks, `flag_counts`, and a `reliability` block: median / 90 % / max of split spread, \|cross term\|, O share and fold-motion ratio |

## The five classes (v1.5.0)
Each cross-section ring is split exactly into five parts, defined only by how the **wall folds** move (cFSM criteria). The connectors play no part in the definitions: they act only at the fastener rows, and between them the pieces are free.

| class | definition | physical meaning |
|---|---|---|
| **G** | rigid in-plane motion of the whole section | global flexure / torsion |
| **C** | each piece moves as a rigid body, but not together with the others (3·pieces − 3 shapes; none for a one-piece section) | distortion of the **assembled** section through the connections: e.g. the four corner pieces rotating about the bolted seams like a linkage. Its stiffness comes from the connectors and the longitudinal bending of the pieces. |
| **D** | fold-driven fields of minimum transverse bending energy (cFSM), without G and C | distortion of the pieces' own cross-sections |
| **L** | inextensional fields with every fold (virtual sharp corner) fixed | local plate buckling, incl. lips flapping about held folds |
| **O** | complement carrying transverse extension and arc deformation, excluding the fold-driver motions attainable by the inextensional space | may contain real deformation omitted by the idealization as well as numerical error |

`D_assembly_pct = D + C` reports the project's assembled-section distortion convention.

**What changed and why.** Up to v1.4.0, the seams were imposed as rigid links **at every ring**, and C was defined from the seam jumps. Between fasteners the bolted lips are not connected. When they buckle locally (they flap side by side about their held folds) they produce such "jumps", so the corner-restrained REF_L model came out ~80 % C (in v1.3.2: up to 79 % G+D). Now the decomposition has these properties, checked before every run:
- where folds constrain all piece rigid motions, holding the compatible fold drivers suppresses G+C+D (historical REF_L of A4784: G+D+C ≤ 3·10⁻¹³ %, L 98.7–100 %, O ≤ 1.3 %);
- REF_G: G ≥ 99.99999 %;
- whole-section rigid motion → 100 % G; opposite rigid motion of two pieces → 100 % C (two-piece benchmark);
- the shares of a smooth field change < 1 percentage point between a mesh and a 4× finer one (benchmark);
- self-test: random combinations of the five classes are recovered to ≈ 1e-14.

**Effect on the A4784 run (250 modes).** Mode 1 (268.8 MPa) is **C 98.5 %** (D_assembly 99.5 %): the four corner pieces rotate as rigid bodies about the bolted seams over one half-wave. The previous label "D" described the same shape. Modes 2–250 (313–390 MPa) are **L-dominant**: 228 modes labelled L, 20 labelled L+D, L+C or L+D+C, and 1 labelled D+L (mode 27, 329.8 MPa, D 40 %). These are lips and walls buckling with small fold motion: the fold-motion ratio is about 0.16, against 1.05 for mode 1. The previous "D" for most of these modes came from the seam rule.

**Overlap (cross term).** The cFSM D shapes overlap strongly with L in the displacement norm: a pure cFSM distortional mode of a lipped channel has 40–55 % of its norm inside span(L). Cross terms of several 10 % are therefore normal for L–D interacting modes. The decomposition itself is still unique; the normalised and the exactly additive signed measures then differ in how they allot the overlap. `CROSS_TERM` is flagged only when the overlap exceeds the mode itself (|cross| > 100 %, `--max-cross-pct`).

**Connector check.** `C_at_bolt_rings_pct` is the C share on the fastener rings, where the pieces are bolted; it is small when the fasteners enforce composite action. In A4784 the single-node BEAM MPC leaves the shell slope almost free at the bolt node (nodal rotation ≈ 0 while the lips rotate), so the bolts act nearly as hinges. This is why mode 1 is a linkage mode. It is a property of the FE connection model; check it if composite action matters, e.g. with a fastener footprint coupling.

## What the percentages mean
A percentage is the share of the **in-plane cross-sectional deformation**, integrated along the member.
It is not a share of strain energy and not a share of the critical load.
The norm weights every millimetre of the mid-line equally. A small motion of the whole section (e.g. corners moving by 5 % of the peak displacement) therefore carries a noticeable share next to a large but localised buckle.

## How far the numbers can be trusted (v1.5.0 audit)
Checks on the A4784 run (250 modes) and its REF_G / REF_L models, and on A3984:

| check | result |
|---|---|
| shares sum to 100 % (normalised and signed) | deviation ≤ 4e-13 % |
| independent of mode scaling / sign | ≤ 5e-14 % |
| independent of the basis chosen inside each class | projectors change ≤ 8e-14 |
| pure FE modes | REF_G ≥ 99.99999 % G; REF_L G+D+C ≤ 3e-13 %, L 98.7–100 % |
| synthetic pure modes (benchmarks) | 100 % G / 100 % C / folds fixed ⇒ 0 % G+C+D, to round-off |
| longitudinal integration (every 2nd / 3rd ring only) | share change ≤ 0.22 points |
| in-plane mesh (benchmark, 4× finer) | < 1 point |
| uncorrelated noise of 1 % of the RMS displacement | median change 0.19, max 0.64 points |
| fold radius threshold 10 t … 25 t | no change (same folds) |
| **fold-arc rotation measured on one side of the arc only** | median 1.2, 90 % 2.9, max 5.1 points; no dominant class changes → reported per mode as `split_spread_pct` |
| **arc extent** (which transition nodes belong to the rigid arc) | sensitive: removing the tangent nodes changes shares up to 35 points; adding the first curved-wall node up to 21 points |

The arc extent is estimated by the rule "≥ 25 % of the peak turning". This is a mesh-dependent heuristic; it does not guarantee the exact tangent points for arbitrary curvature or grading. The program reports, per fold, how close the end nodes lie to this rule (`arc_extent_close_to_rule` in `basis_report.json`, plus a NOTE in the log). In the historical A4784 case the lip-side wall node lay at 19 % of the peak, against the 25 % rule. The rotation-side sensitivity band does not include uncertainty in arc extent.

Per-mode reliability outputs:
- `split_spread_pct`, `rel_<X>_min_pct`, `rel_<X>_max_pct`: the decomposition is repeated with the fold-arc rotation taken from either side of each arc (both are valid measures of a rigid arc's rotation; the FE arc is not exactly rigid). The spread is the largest change of a reliable share. `DEFINITION_SENSITIVE` is set above 10 points (`--max-split-spread`) and limits confidence to medium. `DOMINANT_UNSTABLE` is set if the dominant class changes under a variant, and sets confidence to low. `--no-sensitivity` skips this pass (2 extra projections).
- `fold_motion_ratio`: RMS translation of the folds' virtual corners divided by the RMS mid-line displacement (rigid translation = 1 when folds exist, pure local = 0). This checks the chosen fold geometry; a fold-free piece can move rigidly without moving any fold.
- `LOW_MARGIN` / `MIXED`: margin between the first two mechanisms below 5 % / 20 %. Every confidence below "high" now carries the flag that explains it.
- `halfwave_<X>_mm`: approximate longitudinal half-wavelength of each class component (member length / number of humps of its slice profile; empty below 1 % share). The classes are cross-sectional: a G or C component with a half-wavelength close to the local one is a rotation or translation of the section that follows the local buckles (e.g. A3984 modes 3–16: G ≈ 20 % at 360 mm with corner motions of ≈ 5 % of the peak). It is not member-level global buckling.

### Remaining limitations (read before using the numbers in an energy theory)
1. The shares are **kinematic** (mid-line displacement norm) shares, not strain-energy shares. An energy partition would need the class components' strains, including warping and the longitudinal derivatives, and the stiffness; neither is in the modal export.
2. In this norm the cFSM classes are not orthogonal: L and D overlap strongly. For L–D mixed modes, the normalised and the signed shares can differ by 10–15 points. Quote both, or the reliable share together with its cross term.
3. Rounded folds are rigid arcs. Their measured rotation is not unique (→ `split_spread_pct`), and their extent is fixed by the 25 % rule (see the arc-extent NOTE).
4. Only in-plane translations are decomposed; axial (warping) displacements and nodal rotations are not.
5. C (relative rigid motion of the pieces) is a project-specific extension of cFSM. Its size depends on how the FE model represents the fasteners.
6. The split has been verified for internal consistency, pure modes, discretisation, noise and definition variants, but not yet against an independent cFSM / GBT tool. Classifying an FE model of a single lipped channel and comparing with CUFSM's cFSM participation would close this gap.

Built-in checks:
- an exact-recovery self-test on every run;
- the analytical D-space dimensions: lipped channel 2, plain channel 0, square tube 1;
- the two-piece built-up pure-mode tests and the mesh-convergence test (see "The five classes"). The run stops if any of them fails.

Flags (default limits, which are conventions to be calibrated):
- `UNRELIABLE_OTHER`: O > 5 %
- `CROSS_TERM`: |cross term| > 100 % (`--max-cross-pct`): the class components overlap by more than the mode itself.
- `AMBIGUOUS_DOMINANT`: the normalised and the signed measures disagree on the dominant class, so the mode is truly mixed.
- `dominance_margin_pct` (column): gap between the first and second mechanism, taking the smaller of the two measures; a small value means the mode is mixed.
- Fold-detection stability: folds are re-detected with the radius threshold at −20 % and +25 %. A warning is printed and recorded in `basis_report.json` if the folds change.
- `CLUSTER_k`: the mode belongs to a group of eigenvalues whose spread is ≤ 0.1 % (span criterion, no chaining). Its reliable result is then the group result, which is applied automatically.
- `confidence`:
  - **low**: `UNRELIABLE_OTHER`, `AMBIGUOUS_DOMINANT`, `DOMINANT_UNSTABLE`, or a margin below 5 % (`--conf-low-margin`)
  - **medium**: a margin below 20 % (`--conf-high-margin`), `CROSS_TERM` or `DEFINITION_SENSITIVE`
  - **high**: otherwise

  The dominant-mechanism statistics, `mechanism_critical_stresses.csv`, the figures and `GDLC_summary` all use the reliable result and the confidence grade.

The full method, assumptions and references are in the header of the script.
