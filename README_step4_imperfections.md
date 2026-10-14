# Step 4 — reference-compatible geometric imperfections

For steel-plate runs in `D:\Abaqus_run_family`, all source ODBs must contain the
same P1..P4 **and** STEEL_PLATE_* meshes as the reference CAE. Recreate G/L
reference models from the plate run. Default amplitude gauges and global
orientation/normalization use the column walls only. Identical eigenmode
coefficients, signs and scale factors are then applied to plate nodes too;
the plates are neither left perfect nor separately renormalized. As before,
imperfection coordinates use transverse translations only. The existing
clearance screen covers member-to-member gaps, not plate/member face contact;
inspect Abaqus initial contact adjustments for the chosen amplitudes.

`abaqus_step4_imperfections.py` now consumes the exact completed run created by
`abaqus_complete_model_m20.py`. The run identity comes from
`pipeline_status.json`; the script no longer selects arbitrary `*.cae` or
`*.odb` files by wildcard order. A run that was only built (`--build-only`,
status `BUILT`, no `.odb`) is accepted as well: its CAE is the perfect model,
but it cannot supply mode shapes, so every needed component must then name its
own solved source (e.g. `--local-source <run>_REF_L --global-source <run>_REF_G`);
otherwise Step 4 stops with a message. The manifest records
`reference_run_status` and `source_odb_sha256 = null` in that case.

## 1. Update the repository

```bat
cd /d "C:\Users\810200014.HAMI.000\Documents\Abaqus_run"
git pull --ff-only
```

## 2. Inspect candidate modes

```bat
set "RUN_DIR=D:\CFS-Column\A4784_t2_qm1_0_0_0_R8_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15_run05"

abaqus python abaqus_step4_imperfections.py ^
  --run-dir "%RUN_DIR%" ^
  --suggest
```

The suggestion is heuristic. It reads the enhanced report belonging to the same
recorded reference job. Mode numbers must still be confirmed from the physical
mode shapes before they are used as imperfection components.

## 3. Build the imperfect CAE

Example only:

```bat
abaqus cae noGUI=abaqus_step4_imperfections.py -- ^
  --run-dir "%RUN_DIR%" ^
  --local-mode 2 ^
  --dist-mode 1 ^
  --local-high-t 1 ^
  --dist-mm 3.6 ^
  --output-cae "D:\CFS-Column\Step4_imperfections.cae"
```

### Command-line form and `--cpus`

Every option is written `--name value` with a space. The Abaqus launcher rejects `--name=value`
("has both equality sign and prepended dash") before the script starts. A blank line inside a `^`-continued
command also ends it.

`--cpus N` (or `auto`, the default) is accepted by Step 4 for a uniform command line across the pipeline.
Step 4 launches no Abaqus solver: the value sets the NumPy/BLAS threads of the build (when NumPy is not
already loaded by Abaqus/CAE) and is the default worker count of `--suggest` (`--suggest-cpus` overrides it).

### Selectable source and mode for each component

Every component can come from its own completed buckling analysis. If you give no source, the component comes from the `--run-dir` reference ODB (the earlier behaviour).

| component | source option | mode option | default source |
|---|---|---|---|
| local (L) | `--local-source` | `--local-mode` | `--run-dir` reference ODB |
| distortional (D) | `--dist-source` | `--dist-mode` | `--run-dir` reference ODB |
| global (G) | `--global-source` | `--global-mode` | `analytical` (half-sine bow, `--global-angle-deg`) |

A source can be any of the following:
- a pipeline run folder (with `pipeline_status.json`), which gives its recorded ODB;
- a folder holding exactly one completed `<job>.odb` with its `<job>.inp`, such as the GDLC `<run>_REF_L` / `<run>_REF_G` models;
- the path of an `.odb` file.

Every source ODB is checked before use:
- the analysis must have completed successfully (`.sta`/`.log`) and the ODB must not be locked;
- it must have the reference mesh: same instances, node labels, coordinates and connectivity, verified node by node.

An eigenmode used as the global shape is normalized to unit peak transverse displacement, so `G1000` / `G3000` still mean a peak of L/1000 / L/3000.

Only the components used by the selected `--cases` are required. Two components may use the same mode number only if they come from different sources.

The untouched reference model always stays in the output CAE as the **perfect baseline**. Step 5 builds it as `STEP5_PERFECT_FY…` by default (token `PERFECT`). The CAE is never re-meshed: only the existing mesh-node coordinates are moved.

The sources, modes, ODB checksums and the normalization of every component are stored in each model's description (`component_sources`).

Example: L from REF_L, D from the reference run, G from REF_G:

```bat
abaqus cae noGUI=abaqus_step4_imperfections.py -- ^
  --run-dir "%RUN_DIR%" ^
  --local-source "%RUN_DIR%_REF_L" ^
  --local-mode 1 ^
  --dist-mode 2 ^
  --global-source "%RUN_DIR%_REF_G" ^
  --global-mode 1 ^
  --local-high-t 1 ^
  --dist-mm 3.6 ^
  --output-cae "%RUN_DIR%\imperfection\Step4_imperfections_pipeline.cae"
```

### Combined L + D + G imperfections and amplitude studies (`--combo`)

When running from Windows cmd, assign the selected mode numbers using
`set "LOCAL_MODE=number"`, `set "DIST_MODE=number"` and
`set "GLOBAL_MODES=number,number"`, then pass `--local-mode "%LOCAL_MODE%"`,
`--dist-mode "%DIST_MODE%"` and `--global-mode "%GLOBAL_MODES%"`.
Do not paste a block of interactive `set /p` prompts: the next command can become
the answer to the preceding prompt. Mode numbers in examples are illustrative;
select them from the current original/reference reports. Step 4 prints argument
errors to stdout as `STEP4 INPUT ERROR`, including in CAE, and treats invalid
arguments as a failed CAE script. Successful generation requires `SAVED:` and
the new output CAE; CAE's `Exit code: 0` by itself is insufficient.
If CAE hides stdout/stderr, its final `RuntimeError` also includes the exact bad
option and value. Changing a source folder cannot fix a mode variable containing
command text. Mode numbers can be passed directly, for example `--local-mode 1`
and `--global-mode 2,3`, instead of using environment variables; choose the numbers
from the current reports. This avoids stale `LOCAL_MODE` / `GLOBAL_MODES` values.

Every case is the field

    u0 = aL*phiL + aD*phiD + aG*phiG

where each phi is normalized on its own (local: peak normal displacement, distortional: peak transverse
displacement, global: peak transverse displacement) and the three terms are added without rescaling.
The built-in cases cover L, D, L+D and G alone. Any other combination, for example L+D+G, or an amplitude
sweep for a sensitivity study, is given with `--combo NAME=aL,aD,aG` (repeat the option once per case):

| amplitude form | meaning | example |
|---|---|---|
| number or `…mm` | millimetres | `3.6`, `-3.6mm` |
| `…t` | multiple of the shell thickness | `0.34t`, `-0.94t` |
| `L/…` | fraction of the member length | `L/1000`, `-L/1500` |
| `0` | component absent | `0` |

- Any mix of families in one command, in any number and order: three non-zero amplitudes (LDG), two (LD, LG, DG) or one (L, D, G), e.g. `--combo LDG_mp=-0.34t,1.8t,L/1000 --combo LG_mp=-0.34t,0,L/1000 --combo L_mp=-0.34t,0,0`; add or remove `--combo` options freely.
- Names: 1–20 letters, digits or `_`, starting with a letter; they become `STEP4_<NAME>` and, in Step 5, `STEP5_<NAME>_FY…`. Names are compared case-insensitively (Windows file names); they must differ from each other and from the built-in cases selected with `--cases` (with `--cases none`, `--combo LD_mp=…` is fine).
- With `--combo` and no `--cases`, only the `--combo` cases are built; add `--cases L_low,D,…` (or `all`) to build built-in cases too.
- Only the components with a non-zero amplitude are required, e.g. a `--combo` with aD = 0 needs no `--dist-mode`.
- A non-zero aD needs `--dist-mode` and a solved distortional source. There is no GDLC distortional reference model: without `--dist-source` the mode is read from the `--run-dir` run, which must then be solved and recorded (`--resume-post`, status `COMPLETED`); a `--build-only` run cannot supply it.
- A negative amplitude flips that component's normalized shape (the sign rule is recorded in the manifest).

```bat
abaqus cae noGUI=abaqus_step4_imperfections.py -- ^
  --run-dir "%RUN_DIR%" ^
  --local-source "%RUN_DIR%_REF_L" ^
  --local-mode 1 ^
  --dist-mode 1 ^
  --global-source "%RUN_DIR%_REF_G" ^
  --global-mode 1,2 ^
  --global-angle-deg 0 ^
  --combo LDG_pp=0.34t,1.8t,L/1000 ^
  --combo LDG_pm=0.34t,-1.8t,L/1000 ^
  --combo LDG_mp=-0.34t,1.8t,L/1000 ^
  --combo LDG_mm=-0.34t,-1.8t,L/1000 ^
  --output-cae "%RUN_DIR%\imperfection\Step4_LDG.cae"
```

### Global eigenmode of a doubly-symmetric section (`--global-mode 1,2`)

A square or other doubly-symmetric column has two global flexural modes with equal eigenvalues. Inside such a
pair Abaqus returns an arbitrary basis, so the bow direction of "mode 1" alone can change from one solve to the
next. `--global-mode 1,2` takes the combination of the pair (again an exact eigenmode) whose mean transverse
displacement points along `--global-angle-deg` (degrees from the first transverse axis, X for a Z-axis
column), then normalizes it to unit peak. The modes must form a complete span-based eigenvalue group (relative span 0.1 %, consistent with default GDLC). For a single
mode, Step 4 prints a warning when a neighbouring mode has the same eigenvalue (within 0.1 %), and records
the achieved bow angle and a translation ratio (about 1 for a flexural bow, about 0 for a torsional mode).

A single `--global-mode` together with an explicit `--global-angle-deg` is accepted:
- if that mode has exactly one equal-eigenvalue partner in its ODB (within 0.1 %), the pair is used and
  aligned with the angle, and a `NOTE` says so (A3184 REF_G: `--global-mode 2` becomes the flexural pair
  2,3; mode 1 there is torsional). Writing the pair, `--global-mode 2,3`, gives the same field;
- with several partners the complete eigenspace is used. An explicit incomplete group is rejected;
- the oriented shape minimizes its physical displacement norm subject to the requested mean bow, using an orthonormal SVD basis. It is invariant under nonsingular changes of the eigenvector basis, including groups larger than two;
- without a partner the angle can only fix the sign of an already aligned bow. A direction outside the single-mode span, or an attempt to orient a torsional mode by a bow angle, is rejected. Omit the angle to use that eigenmode unchanged apart from normalization.

Only an exactly repeated eigenvalue makes every combination an exact eigenmode.
The 0.1 % grouping tolerance is a numerical convention for nearly repeated
modes. A combined imperfection from a nearly repeated group is an oriented
imperfection field, not a newly solved exact eigenpair; source eigenvalues are
retained in the manifest.

### Bending runs

Step 4 detects a bending reference run (Step 3 `--load-case bending`) from
`pipeline_status.json`; nothing else changes and the provenance carries the load case to Step 5.

- Local and distortional shapes: take them from the BENDING run / its REF_L (they are concentrated in the
  compression zone and differ from the compression modes). Choose the mode numbers again from the bending
  GDLC results (`mechanism_critical_stresses.csv`, `first_dominant_mode` of L and D); the compression
  mode numbers do not carry over.
- Global shape: the lateral-torsional mode, preferably REF_G mode 1 of the bending run as a SINGLE mode
  (`--global-mode 1`, no `--global-angle-deg`). Bending breaks the equal-eigenvalue flexural pair, so
  `--global-mode 1,2` is refused when the two eigenvalues differ. `G1000`/`L/1000` is the peak transverse
  displacement of that mode (it includes the twist component).
- Analytical bow (no `--global-source`): the default direction is the lateral direction, i.e. along the
  neutral axis (`--bending-axis-deg`); `--global-angle-deg` still overrides it.
- Amplitudes and `--combo` work unchanged (0.34t, 1.8t, L/1000 are geometric).

```bat
set "RUN_DIR=D:\CFS-Column\<input>_BEND000"
cd /d "C:\Users\810200014.HAMI.000\Documents\Abaqus_run"
abaqus cae noGUI=abaqus_step4_imperfections.py -- ^
  --run-dir "%RUN_DIR%" ^
  --local-source "%RUN_DIR%_REF_L" --local-mode 1 --local-high-t 1 ^
  --dist-mode <first D-dominant mode of the bending run> --dist-mm 3.6 ^
  --global-source "%RUN_DIR%_REF_G" --global-mode 1 ^
  --cases all ^
  --combo LDG_pp=0.34t,1.8t,L/1000 --combo LDG_pm=0.34t,-1.8t,L/1000 ^
  --combo LDG_mp=-0.34t,1.8t,L/1000 --combo LDG_mm=-0.34t,-1.8t,L/1000 ^
  --output-cae "%RUN_DIR%\imperfection\Step4_LDG.cae"
```

### Purity of the selected modes (GDLC)

When the source folder contains `mode_decomposition_GDLC\mode_participation.csv` from
`gdlc_mode_decomposition.py`, Step 4 looks up every selected mode. The lookup is used only if the eigenvalue
in the CSV matches the ODB frame and `basis_report.json` identifies the same ODB SHA256 (otherwise the classification is stale or unverified and is reported as
stale). Step 4 prints its G/D/L/C/O shares and warns when the local, distortional or global component is not
dominated by L, D or G respectively, with at least 80 %. The shares are stored under
`component_sources.<component>.gdlc`. Individual `*_pct` shares assess the selected vector; `rel_*_pct` group shares are retained separately and never substituted for individual purity. Old reports without ODB provenance must be reclassified before relying on their purity. REF_L / REF_G are constrained reference shapes, whose purity must still be checked from their outputs.

### Inter-piece clearance screen

For every case, Step 4 compares the distance between the midsurface nodes of different pieces before and
after the imperfection. Two midsurfaces closer than one shell thickness overlap. Abaqus general contact
removes such initial overclosures by moving nodes without strain, which would locally change the imposed
imperfection. Step 4 then prints a warning and records the pairs under `cases.<model>.interpiece_clearance`.
The screen compares node-to-node distances and is always indicative: tangential slip and face interiors can hide penetration even with matching meshes. Zero flagged pairs is not proof of clearance. After a Step 5
solve, the `.msg` file lists any adjustment that was made.

The local/distortional names here identify the selected imperfection components;
they do not override the accepted mode-family classification from the reference
modal audit.

## Compatibility checks

Before any nodal coordinate is modified, Step 4 verifies:

- `pipeline_status.json` reports a completed reference run;
- source model name equals the recorded `build.job_name`;
- exact instances are `P1..P4`;
- mesh elements are `S4R`;
- BC/load counts match the reference build;
- bolts are the reference `BEAM_MPC` constraints and their count equals
  `source_inputs.expected_links`;
- `GeneralContact` / `Hard_Frictionless` are present;
- CAE and ODB belong to the same recorded job.

The imperfect models preserve the original mesh connectivity, BCs, loads,
contact, material and BEAM_MPC constraints. Only mesh-node coordinates are
changed. Do not remesh.

Each `STEP4_*` model stores the reference settings and source hashes in its
Model description. Step 5 requires this provenance and rejects independent or
legacy Step-4 CAEs that do not contain it.
