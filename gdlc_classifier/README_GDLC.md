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
- `scipy`: strain-energy evaluation with the exported stiffness (skipped, with a note, without it)
- `matplotlib`: plots
- `psutil`: optional telemetry only; RAM-aware chunking uses the shared stdlib-only `runtime_resources.py` in the repository parent directory
- `cupy`: CUDA GPU (used automatically when present)

## Integrated pipeline (ODB → export → classification)
`gdlc_mode_decomposition.py` now drives the whole chain. `abaqus_mfsm_export.py` must sit in the same folder; it is self-contained and needs no other modules. For every run folder found below the given path(s), the program:
1. **Solves reference models** (only with `--solve-reference-models`): every `<run>_REF_G` / `<run>_REF_L` folder whose `.odb` is missing and solver destination is unused is solved with `abaqus job=… cpus=<N> interactive`; N = `--cpus N` (e.g. `--cpus 4`, written with a space), default all physical cores (Abaqus counts cores, not threads; a larger request is reduced to them with a warning, as in the rest of the pipeline). `--threads` only sets the Python/NumPy threads of the classifier.

For bending, create references with `--make-reference-models` on `<family>_BEND<angle>`,
then solve the resulting `<family>_BEND<angle>_REF_L` and `_REF_G` paths. Passing only the
original run to `--solve-reference-models` does not create or solve its sibling reference folders.
The exact source INP's matching `_build.json` supplies the load case and copied provenance;
unrelated build records in the directory are ignored. New `gdlc_reference.json` records
the original and reference INP hashes plus the load case. Source geometry, loads and bolt
MPCs remain inherited; only the documented reference constraints and eigensolver setup change.
2. **Exports** `<job>.odb` by calling `abaqus python abaqus_mfsm_export.py --odb … --inp … --output-dir <run>\modal_export_gdlc`.
   - `--export auto` (default): an existing export is reused when the sha256 values it recorded match the current `.odb` and `.inp`.
   - `--export always`: a new export is written every time.
   - `--export never`: only existing exports are used.
   - The exporter never overwrites; a later export gets `_2`, `_3`, … appended to the folder name.
3. **Exports the elastic stiffness** of the run once per `.inp` (`--stiffness auto`, default; see "Strain-energy evaluation"): `<run>\stiffness_export\<job>_KE.inp` is solved with `abaqus job=<job>_KE … cpus=<N> interactive`, read and checked. Reference models are skipped.
4. **Classifies** the run (kinematic shares) and, with the stiffness, evaluates every class component in strain energy.

Which folders count as runs:
- any folder that holds a modal export (`*/modal_export.json`, in either the npz or the legacy parquet format);
- any folder with one `<job>.inp` that has a `*Buckle` step, plus `<job>_build.json` or `gdlc_reference.json`, solved (`<job>.odb`) or built only (pipeline `BUILD_ONLY`: no `.odb` yet). A built-only run is enough for `--make-reference-models` (the geometry is then read from the `.inp`, with the instance translations/rotations, by the exporter's own INP reader); classification skips it with a message until it is solved.

Other `.odb` files, such as GMNIA or single-piece studies, are ignored. For npz exports, the element connectivity is read from the `.inp`, and the program checks that this `.inp` has the sha256 recorded in the export.

## Run
```
python gdlc_mode_decomposition.py "D:\CFS-Column"                  # every run below this folder
python gdlc_mode_decomposition.py "D:\CFS-Column\<run>"            # a single run
python gdlc_mode_decomposition.py "D:\CFS-Column" --skip-existing  # only runs without results
python gdlc_mode_decomposition.py <run> --selftest-only            # basis + checks, no modes read
python gdlc_mode_decomposition.py <run> --stiffness never          # kinematic split only (no matrix job)
python gdlc_mode_decomposition.py <run> --stiffness always         # regenerate the stiffness export
python gdlc_mode_decomposition.py --benchmarks                     # analytical checks only
python gdlc_mode_decomposition.py "D:\CFS-Column" --make-reference-models   # write pure-G / local reference .inp files
```

## Reference models (pure G and corner-restrained L), built by Abaqus itself
`--make-reference-models` writes two folders next to every run, `<run>_REF_G\` and `<run>_REF_L\`; the run does not need to be solved first (geometry from its modal export when one exists, else from its `.inp`; the source is recorded as `geometry` in `gdlc_reference.json` and kept when the inputs are regenerated later, because `.odb` coordinates differ from the `.inp` ones in the last digits and would change the ring master nodes and coefficients of inputs already solved). Each holds a modified copy of the run's `.inp` (20 eigenvalues by default, `--reference-modes`), the `_build.json` and `gdlc_reference.json`.

The input must be the **original built Abaqus run containing its `.inp`**, not the CUFSM CSV input folder and not an existing reference model. `--reference-kinds L` generates only the Local reference; the default `G,L` generates both. For example, this command only writes reference inputs; it does not submit an analysis:

```bat
cd /d "C:\Users\810200014.HAMI.001\Documents\Abaqus_run\gdlc_classifier"
py -3.14 gdlc_mode_decomposition.py "D:\column-result\no_plates_bending_0\New folder\family_1_BEND000" ^
  --make-reference-models ^
  --reference-kinds L ^
  --reference-modes 20 ^
  --export never
```

Use the corresponding original run path for another load case or section. Existing solved reference inputs/results remain protected; a changed reference requires a new solve and classification performed by the user.
- **REF_G**: every cross-section ring is kept rigid in its own plane. Axial displacements (warping) and seam slip stay free, so the eigenmodes are pure global, with the real partial composite action of the bolted member. The first eigenvalue is the **constrained-G critical stress**.
  - **Twist (`--ref-g-twist`, default `auto`).** `restrained`: each ring may only **translate** in its plane (`u_i = u_a`, 2 rigid DOFs per ring), so the sections cannot rotate about the member axis: **no torsional or flexural-torsional buckling**, only global flexural modes (formulation `rigid_ring_translation_equations_v1`). `free`: 3 rigid DOFs per ring (translations + rotation), so flexural, torsional, flexural-torsional or lateral-torsional modes (formulation `rigid_ring_equations_v2`). `auto` = `restrained` for compression runs and `free` for bending / combined runs, whose global mode is lateral-torsional and needs the twist. The choice is recorded as `twist` / `twist_reason` in `gdlc_reference.json`. Only equations are added (no supports, no reference nodes), so no mechanism is introduced and Lanczos is used as before; the ends were already restrained in rotation about the axis (UR3).
  - With the twist restrained, the A3184 REF_G no longer has its torsional mode 1 (296 MPa): the flexural pair (previously modes 2, 3 at 1161 MPa) becomes modes 1, 2. Give Step 4 `--global-mode 1,2` (or a single mode with `--global-angle-deg`, which uses its equal-eigenvalue partner).
  - An existing `_REF_G` folder that already has solver results is never overwritten: rename or move it (e.g. to `<run>_REF_G_previous`) before regenerating with a different formulation.
  - Since v1.7.2 the rigid ring is written as linear `*Equation` rows between the ring's own nodes, u_i = u_a + θ k×(r_i − r_a), with θ taken from a second master node b. Each ring keeps exactly its 3 in-plane rigid DOFs; no reference nodes, couplings or extra boundary conditions are added. Master a is a node with a boundary condition when the ring has one (the axial anchor), and neither master is a bolt (MPC) node.
  - Up to v1.7.1 each ring had a kinematic coupling (DOFs 1, 2) to an extra reference node whose unused DOFs 3–5 were fixed. Abaqus then reported a numerical singularity (e.g. A3184: node P1.361, DOF 3, ratio 2.6e13) and refused the Lanczos solve ("rigid body modes … overconstrained"). A `lanczos_refused` flag recorded for that old formulation no longer forces subspace iteration.
- **REF_L**: the virtual sharp corner of every fold is held in its plane along the whole member, while the fold itself stays free to rotate. This is written as one `*Equation` per in-plane direction on the fold's reference node: u + θ×(c − p) = 0.
  - When the virtual corner lies on a global in-plane axis through the reference node (lever arm zero, e.g. a fold turned 45° to the global axes), that row reduces to u = 0. Abaqus rejects one-term `*Equation` rows ("use *BOUNDARY"), so since v1.7.3 such DOFs are held by boundary conditions on the node sets `LREF_FOLDS_U1` / `LREF_FOLDS_U2` in every load case; `n_dofs_held` in `gdlc_reference.json` counts them. (A3184 up to v1.7.2: 732 one-term rows, input-processor error.)
  - θ is the arc's chord rotation (u_b − u_a)·n/h, averaged over edges away from the reference node, or over all internal edges when none are away. For **all load cases**, the equations impose the classifier's exact `Qf u = 0`: the fold-reference translations are eliminated together, including on two/three-node arcs. No shell `UR` substitution is used (`formulation: virtual_corner_exact_chord_v1`, `n_equations_shell_rotation: 0`). Rotation about the virtual corner remains free; axial DOFs, applied loads and connections are unchanged. Since v1.10.2, existing modal boundaries are handled per DOF: fixed zero terms are substituted, fully implied rows are counted, and unsafe partially fixed pivots/MPC dependencies are rejected. A boundary on one translation cannot silently remove both corner rows. Older references are not updated by changing the code.
  - The seams stay as built. Before writing a Local reference, the generator checks that the corner rows suppress all three planar rigid motions of **each separate piece**. A corner-free or single-corner piece fails this coverage check instead of being labelled Local. Source `*Equation`, `*Coupling`, `*Tie` and `*Rigid Body` dependencies are rejected because this writer cannot safely track their eliminations. `gdlc_reference.json` records `rigid_motion_coverage` and the counts of added/implied/required rows in `constraint_coverage`.
  - **Scope of the Local reference.** Holding corner drivers suppresses G/D/C, but does not impose the classifier's transverse-inextensibility/rigid-arc operator. L and some O deformation remain admissible. Consequently the result is a corner-restrained, Local-dominated candidate after a passing output check; this is not independent validation of a pure cFSM/GBT mode. Rounded-corner idealizations can change stiffness and the local spectrum; see [Beregszászi and Ádány (2019)](https://pp.bme.hu/ci/article/download/12949/8223).
  - Assembly equations and the added boundary conditions also act in the stress-generation load case. Keeping the applied loads does **not** guarantee unchanged prestress: restrained transverse/Poisson response can change it. The eigenvalue is that of the constrained reference model, not an independently verified local critical stress of the original member. Inspect the selected shape and the reported `DEFINITION_SENSITIVE`/split-sensitivity flags before using it for an imperfection. No extra inextensibility constraints are added automatically.
  - Earlier versions: holding the arc node itself let the corners move by θ·|c − p| (up to 10 % spurious G+D). Using the shell rotation UR instead of the chord rotation left a few % D, because the FE arc is not exactly rigid.
  - Existing solved references are never rewritten. Use a new source run/output folder, or preserve/move the old reference folder yourself before generating a changed formulation. Verify the new Abaqus modes against the unchanged 95 % L / 5 % G+D+C criteria; exact fold constraints do not guarantee that the remaining O component is below 5 %.

**Eigensolver.** Both reference models use the **Lanczos** eigensolver (`*Buckle, eigensolver=LANCZOS`, same number of modes, no eigenvalue range, so the same lowest eigenvalues as before).
- The local spectrum of a long member is dense: the 40 lowest REF_L eigenvalues of the A4784 run lie within 1 %. Subspace iteration converges like (λ_n/λ_q+1)^k; with 40 vectors it needed 484 iterations, 43 min. Lanczos (shift-invert, a few factorisations of about 1 s each) is not slowed down by such clusters.
- Abaqus refuses Lanczos for contact pairs or contact elements, hybrid elements, connector elements and distributing couplings. Such models are written for subspace iteration with 4n vectors.
- If Abaqus still refuses a Lanczos solve, `--solve-reference-models` logs the `***ERROR` line, rewrites the `*Buckle` for subspace iteration (4n vectors), re-solves once and records `lanczos_refused` in `gdlc_reference.json`; later regenerations of the same formulation then keep subspace.

The reference-model constraints are appended to the step's existing `*Boundary, op=NEW, load case=…` blocks. Abaqus requires `OP=NEW` for load case 2 of `*BUCKLE` and forbids mixing `OP=NEW` with `OP=MOD`, so no new `*Boundary` keyword is written. The source run's `abaqus_v6.env` is copied too.

A reference model is (re-)solved by `--solve-reference-models` when its `.odb` is missing, failed (no "COMPLETED SUCCESSFULLY" in the `.sta`/`.log`) or older than its `.inp`. Re-generating identical inputs does not touch the files. Once solved, the program exports and classifies the reference model:
  - it checks each reference model automatically: all five modal shares must be finite, within 0–100 %, sum to 100 % within 0.001 percentage points, and accompany a positive finite eigenvalue. Empty/incomplete/nonfinite reports fail. REF_G modes must come out ≥ 95 % G; REF_L modes must have ≥ 95 % L, ≤ 5 % G+D+C and ≤ 5 % O. These are kinematic consistency checks against the imposed constraints, not independent scientific validation;
  - optional eigenspace purity bounds supplied in external/archived reports are also checked. This implementation does not compute those bounds; `eigenspace_purity_assessed` is true only if every row supplies one. Definition-sensitivity bounds are not eigenspace bounds;
  - Step 4 applies the same Local criterion to a selected, matching classified REF_L vector for compression, bending and combined loading. Missing/stale classifications and original mixed-mode sources retain their warning path; the explicit `--skip-mode-quality-checks` override remains available;
  - `GDLC_summary\runs_summary.csv` includes `constrained_G/L_reference_eigenvalue` and `G/L_reference_check_passed`. Legacy `pure_G/L_reference_eigenvalue` columns are populated only when the kinematic check passes; their names do not imply independent pure-mode validation.
By default it uses every available logical CPU for useful Python/BLAS work. Mode arrays spill to temporary memory maps when live RAM cannot hold them; shard readers, projection chunks and energy batches use live memory budgets without a reserved percentage. Plot workers share the CPU budget and respect the Windows process-pool ceiling. Every usable visible CUDA GPU participates in projection; unavailable CUDA, failed probes or insufficient VRAM use the CPU path. CuPy is optional. `--threads N` and `--gpu on|off` override classifier settings; `--cpus auto|N` controls Abaqus separately. Sparse LU factorization uses SciPy SuperLU (sequential factorization); independent right-hand sides run concurrently. This does not imply that every phase scales with every core. Sparse factorization must still fit in RAM, and CUDA execution requires production verification on actual GPU hardware.

## Bending runs (v1.7.0)
Runs built with `abaqus_complete_model_m20.py --load-case bending` are found, exported, decomposed and given reference models exactly like compression runs; the commands are the same, with the `<input>_BEND000` / `_BEND045` folders.
- The decomposition is purely kinematic (cross-section deformation shares); it does not depend on the load. A rigid lateral-torsional motion of the real A4784 section comes out as 100.0 % G, and the REF_G / REF_L checks apply unchanged. Local and distortional modes of a bending run sit in the compression zone; that changes the shapes, not the classes.
- The eigenvalue of a bending run is the critical EXTREME-FIBRE stress (MPa) of the CUFSM-consistent linear reference stress. `sigma_cr_MPa` keeps that meaning. `mode_participation.csv` adds `M_cr_kNm = eigenvalue x M_ref`, and `mechanism_critical_stresses.csv` adds `first_dominant_M_cr_kNm` etc. (`M_ref` from the build json, `load_case.reference_moment_mesh_Nmm_per_MPa`). `basis_report.json` records the load case; `GDLC_summary\runs_summary.csv` has `load_case` and `bending_axis_deg` columns.
- The bending build uses Lanczos with eigenvalues >= 0 (`*Buckle` data line `n, 0.`). The reference models keep that range. If Abaqus refuses Lanczos and falls back to subspace iteration, which has no eigenvalue range, 2n eigenvalues are requested, because for a section with 180-degree symmetry each positive eigenvalue has a negative twin (the reversed moment).
- Modes with eigenvalue <= 0 (reversed load) are excluded from the classification, and their numbers are listed in `basis_report.json` (`excluded_nonpositive_modes`). Compression runs have none, so their results are unchanged (checked byte for byte on the A4784 REF_L run).

## P-M buckling folders
Folders written by `abaqus_pm_interaction.py --mode buckling` (`PMB_*`, one proportional P-M ray each) are ordinary runs for this program once solved: their build json has `load_case.type = combined`, so `mode_participation.csv` adds `P_cr_kN` and `M_cr_kNm`, `mechanism_critical_stresses.csv` adds `first_dominant_P_cr_kN` / `_M_cr_kNm`, and `runs_summary.csv` records `pm_angle_deg`.

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
| `mode_participation.csv` (energy columns, when `stiffness_export\` exists) | `E_<X>_pct`, `E_<X>_signed_pct`, `E_cross_term_pct` (the mode), `E_rel_<X>_pct`, `E_rel_<X>_signed_pct`, `E_rel_cross_term_pct`, `E_reliable_dominant`, `E_reliable_label`, `E_resolved` (group result), `E_beyond_section_pct`, `bolt_incompatibility_pct`; flags `ENERGY_DISAGREES`, `ENERGY_AMBIGUOUS`, `ENERGY_CROSS_TERM` |
| `mode_energy.npz` | per mode the 5 × 5 class energy matrix `E[i,f,g] = w_f^T S w_g`, its group version `E_group`, the mode's own energy `U = v^T K v`, `bolt_incompatibility` |
| `<run>\stiffness_export\` | the matrix job and Abaqus' `.mtx` files, `stiffness_raw.npz` (scipy.sparse CSR, full symmetric, N/mm), `stiffness_dofs.npz` (instance, label, DOF, coordinates of every row), `stiffness_export.json` (source `.inp` sha256, node map, rigid-body check) |
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
- self-test: deterministic random combinations must be recovered with relative error below `1e-8`, rigid translation G share must differ from one by less than `1e-10`, and both translations plus rigid rotation must be recovered within `1e-8`. Nonfinite results fail; tolerances are not relaxed for an ill-conditioned section.

**Geometry precision (v1.10.1).** With a matching INP, classification and reference
generation use its assembly coordinates at the original FP64 precision. Export
node order and modal vectors are preserved. Every instance/label must match;
exported coordinates must agree to FP64 transformation roundoff or equal the
exact FP32 rounding of the input. Other differences, missing nodes and duplicate
identities are rejected. Converting rounded ODB coordinates to FP64 alone can
create artificial nearly dependent fold-driver directions and make the split
ill-conditioned. This recovery changes neither the source geometry nor the
accuracy of the exported U/UR fields. Legacy parquet without an INP retains
exported coordinates and must still pass the same self-test.

Class-basis columns are equilibrated in the W metric, and their dual is computed
with a weighted QR solve instead of an unscaled inverse. Class definitions and
all singular directions are retained: there is no smoothing, regularisation or
rank truncation added to make a failed self-test pass. The condition number now
refers to this equilibrated basis. `basis_report.json` records `mesh_geometry`,
including how coordinates were verified; self-test failures include the
condition number and geometry source. The exporter's
`scientifically_eligible: false` describes raw modal data rather than a validated
classification and is not the cause of a basis self-test failure.

**Effect on the A4784 run (250 modes).** Mode 1 (268.8 MPa) is **C 98.5 %** (D_assembly 99.5 %): the four corner pieces rotate as rigid bodies about the bolted seams over one half-wave. The previous label "D" described the same shape. Modes 2–250 (313–390 MPa) are **L-dominant**: 228 modes labelled L, 20 labelled L+D, L+C or L+D+C, and 1 labelled D+L (mode 27, 329.8 MPa, D 40 %). These are lips and walls buckling with small fold motion: the fold-motion ratio is about 0.16, against 1.05 for mode 1. The previous "D" for most of these modes came from the seam rule.

**Overlap (cross term).** The cFSM D shapes overlap strongly with L in the displacement norm: a pure cFSM distortional mode of a lipped channel has 40–55 % of its norm inside span(L). Cross terms of several 10 % are therefore normal for L–D interacting modes. The decomposition itself is still unique; the normalised and the exactly additive signed measures then differ in how they allot the overlap. `CROSS_TERM` is flagged only when the overlap exceeds the mode itself (|cross| > 100 %, `--max-cross-pct`).

**Connector check.** `C_at_bolt_rings_pct` is the C share on the fastener rings, where the pieces are bolted; it is small when the fasteners enforce composite action. In A4784 the single-node BEAM MPC leaves the shell slope almost free at the bolt node (nodal rotation ≈ 0 while the lips rotate), so the bolts act nearly as hinges. This is why mode 1 is a linkage mode. It is a property of the FE connection model; check it if composite action matters, e.g. with a fastener footprint coupling.

## Strain-energy evaluation (v1.9.0)
The five classes stay kinematic (they are defined on the in-plane slice field, as in cFSM). What is new is that every class component of a mode is also **evaluated with the real elastic stiffness of the shell model**, including longitudinal bending, warping, transverse shear, the corners as modelled and the bolts.

**What Abaqus has to provide.** After the buckling run, a small companion job is written from the run's own `.inp` and solved once (`<run>\stiffness_export\<job>_KE.inp`):
```
model data of the run, unchanged except the model-data *Boundary blocks
*Step, name=GDLC_KE
*Matrix Generate, STIFFNESS, LOAD, MPC=NO
*Matrix Output, STIFFNESS, LOAD, FORMAT=MATRIX INPUT
*Cload          one load per node on DOF 1, value = instance code + node label
*End Step
```
- `MPC=NO` and no boundary condition give the **raw assembled element stiffness** K (6 DOF per node) in a known DOF space; supports and the BEAM/TIE/PIN MPCs are applied afterwards from the `.inp`. K is the stiffness of the buckling problem K φ = λ(−K_g) φ: `*BUCKLE` starts from the unloaded initial state, so there is no initial-stress term.
- Abaqus writes matrices with internal node numbers. The coded `*Cload` values come back in the load matrix next to those numbers, which gives the exact map to (instance, label). The fallback (instance order, labels ascending) is used only if it passes the same check.
- **Acceptance before use:** K symmetry, rigid-body null space, exact support/MPC compatibility, and K-orthogonality of distinct eigenvalue groups are checked. K-coupling above 1 %, excessive lift residual, nonpositive full-mode energy, nonfinite values or an indefinite component energy Gram matrix refuse the diagnostic. These are necessary checks, not sufficient proof of the correct physical operator (for example, uniformly scaling K preserves orthogonality).
- Contact is retained. Matrix-export failure never triggers a solve with contact removed. Preloaded/multiple-step models and unmapped constraint/initial-state semantics are rejected instead of being silently replaced by a virgin elastic model. The supported case is a single Buckle step from an unloaded initial state. Active perturbation boundary conditions follow the Buckle load-case precedence, including `OP=NEW`.
- Abaqus may emit negative internal node labels for contact or other internal variables. Exactly-zero entries involving these labels are ignored and their counts are recorded; any nonzero internal stiffness/coupling or load is rejected because it requires explicit internal-DOF treatment. No magnitude cutoff is used and contact is not removed. After a matrix-ingestion failure, `auto`/`required` can reuse the completed, unprocessed matrix job if its full companion INP and node-code definition match the current model, its STA reports completion and no lock exists. Existing `.mtx`/solver files are preserved; only previously missing processed exports are produced. `always` requests a fresh solve instead.
- Outputs: `stiffness_raw.npz`, `stiffness_dofs.npz`, `stiffness_export.json` (sha256 of the `.inp`; regenerated automatically when the `.inp` changes). `--stiffness always` creates a fresh numbered export; existing artifacts stay intact, `required` stops on any failure, `never` switches the whole evaluation off; with `auto` a failure is logged and the classification runs without the energy columns.

**How a class component gets an energy.** A component w_f is an in-plane field. It is completed to a full displacement field by **minimum strain energy**: the in-plane translations of all nodes are prescribed, the axial translations and rotations follow from K (supports and the axial / rotational rows of the bolt MPCs kept, i.e. axial slip and rotation continuity = composite action). With S the resulting Schur complement (never formed; one sparse factorisation, concurrent independent right-hand sides):
- `E_fg = w_f^T S w_g` (5 × 5 per mode, exactly additive: Σ E_fg = w^T S w);
- `E_<X>_pct` = E_ff / Σ E_gg; `E_<X>_signed_pct` = Σ_g E_fg / Σ E (sums to exactly 100 %); `E_cross_term_pct` = 1 − Σ E_ff / Σ E;
- `E_rel_*` = the group (near-degenerate eigenvalues) result, basis-invariant like `rel_*`;
- `E_beyond_section_pct` = 1 − w^T S w / v^T K v: share of the mode's own energy (with its exported U3 and UR) that its in-plane field does not imply. Small values mean the in-plane classes capture the mechanics of the mode.
- `bolt_incompatibility_pct`: RMS relative motion along the bolt axes that the class components carry individually, relative to the RMS in-plane displacement of the mode. The bolts forbid it; it cancels in the sum of a compatible mode. The in-plane bolt rows are relaxed for the components on purpose: forcing a component's bolt node to follow the MPC would create a kink and spurious energy.

**Flags.** `ENERGY_DISAGREES`: the energy dominant class (resolved: normalised and signed energy agree) differs from the kinematic `reliable_dominant`; confidence at most medium. `ENERGY_AMBIGUOUS`: normalised and signed energy disagree. `ENERGY_CROSS_TERM`: |energy cross term| > `--max-cross-pct`; the class components store much more energy separately than together (the kinematic classes are not energy-orthogonal), so the normalised energy percentages are then mainly a ranking and the signed ones should be quoted.

**Interpretation and confidence.** `v^T K v` uses the full exported U/UR. Class energies are auxiliary minimum-energy completions of transverse kinematic components. Their relaxed in-plane bolt constraints mean they are not individually admissible physical buckling modes. They do not establish pure-family critical stresses or a unique energy partition. `kinematic_confidence` and `energy_confidence` are reported separately; combined confidence is the weaker. More than `--max-energy-beyond-pct` (default 10 %) unexplained full-mode energy or `--max-bolt-incompat-pct` (default 5 %) incompatible component motion gives low energy confidence. These cutoffs are explicit screening conventions, not code provisions. Reference classification accepts `--stiffness required` but explicitly reports energy unavailable for its unsupported equation constraints. No timing estimate is guaranteed without running the actual model.

Since v1.9.1, `basis_report.json` includes `confidence_by_measure` under both `dominance` and `reliability`: combined, kinematic and energy counts (energy is null when unavailable). Existing `confidence_counts` / `confidence` fields retain their combined meaning. `flag_counts` counts exact semicolon-delimited tokens once per mode, excludes `CLUSTER_*` membership tags, and includes `ENERGY_BEYOND_SECTION` and `ENERGY_INCOMPATIBLE_COMPONENTS`. Earlier reports could overcount `CROSS_TERM` by also matching `ENERGY_CROSS_TERM`, and omitted these two energy warnings from the summary. The energy block and terminal log now state the limitation counts explicitly. Existing saved results and reports are not retroactively changed.

**What is deliberately not done.**
- The geometric stiffness K_g is not exported: Abaqus has no direct output for it, and differencing a preloaded tangent stiffness mixes in initial-displacement terms. REF_G / REF_L provide constrained reference stresses/shapes; verify their reported purity before treating them as family-specific critical stresses.
- Section outputs (SE, SK, SF) are not requested in the buckling step: the energy is computed exactly from K and the nodal U/UR, and section output for 250 modes would multiply the ODB size without entering the split.
- The class definitions themselves are unchanged (cFSM-type kinematics, D from the transverse-bending proxy). An energy-orthogonal split of the whole member would be a different, non-standard method; the energy columns show where the kinematic classes interact mechanically.
- Reference models (REF_G / REF_L) are not evaluated in energy: their `*Equation` constraints are not part of the constraint mapping.

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
1. The `G/D/L/C/O_pct` shares are **kinematic** (mid-line displacement norm) shares. Their strain-energy counterparts are the `E_*` columns (v1.8.0, with the exported stiffness); the classes are not energy-orthogonal, so quote the energy cross term or the signed energy shares with them.
2. In this norm the cFSM classes are not orthogonal: L and D overlap strongly. For L–D mixed modes, the normalised and the signed shares can differ by 10–15 points. Quote both, or the reliable share together with its cross term.
3. Rounded folds are rigid arcs. Their measured rotation is not unique (→ `split_spread_pct`), and their extent is fixed by the 25 % rule (see the arc-extent NOTE).
4. Only in-plane translations are decomposed; axial (warping) displacements and nodal rotations enter only through the energy evaluation (minimum-energy completion of each component, and the mode's own U3/UR in `E_beyond_section_pct`).
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
- `ENERGY_DISAGREES` / `ENERGY_AMBIGUOUS` / `ENERGY_CROSS_TERM`: see "Strain-energy evaluation".
- `dominance_margin_pct` (column): gap between the first and second mechanism, taking the smaller of the two measures; a small value means the mode is mixed.
- Fold-detection stability: folds are re-detected with the radius threshold at −20 % and +25 %. A warning is printed and recorded in `basis_report.json` if the folds change.
- `CLUSTER_k`: the mode belongs to a group of eigenvalues whose spread is ≤ 0.1 % (span criterion, no chaining). Its reliable result is then the group result, which is applied automatically.
- `confidence`:
  - **low**: `UNRELIABLE_OTHER`, `AMBIGUOUS_DOMINANT`, `DOMINANT_UNSTABLE`, or a margin below 5 % (`--conf-low-margin`)
  - **medium**: a margin below 20 % (`--conf-high-margin`), `CROSS_TERM` or `DEFINITION_SENSITIVE`
  - **high**: otherwise

  The dominant-mechanism statistics, `mechanism_critical_stresses.csv`, the figures and `GDLC_summary` all use the reliable result and the confidence grade.

The full method, assumptions and references are in the header of the script.
