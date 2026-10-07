# Step 4 — reference-compatible geometric imperfections

`abaqus_step4_imperfections.py` now consumes the exact completed run created by
`abaqus_complete_model_m20.py`. The run identity comes from
`pipeline_status.json`; the script no longer selects arbitrary `*.cae` or
`*.odb` files by wildcard order.

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

- Names: 1–20 letters, digits or `_`, starting with a letter; they become `STEP4_<NAME>` and, in Step 5, `STEP5_<NAME>_FY…`. Names are compared case-insensitively (Windows file names).
- With `--combo` and no `--cases`, only the `--combo` cases are built; add `--cases L_low,D,…` (or `all`) to build built-in cases too.
- Only the components with a non-zero amplitude are required, e.g. a `--combo` with aD = 0 needs no `--dist-mode`.
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
column), then normalizes it to unit peak. The pair must have equal eigenvalues (within 1 %). For a single
mode, Step 4 prints a warning when a neighbouring mode has the same eigenvalue (within 0.1 %), and records
the achieved bow angle and a translation ratio (about 1 for a flexural bow, about 0 for a torsional mode).

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
in the CSV matches the ODB frame (otherwise the classification belongs to another solve and is reported as
stale). Step 4 prints its G/D/L/C/O shares and warns when the local, distortional or global component is not
dominated by L, D or G respectively, with at least 80 %. The shares are stored under
`component_sources.<component>.gdlc`. The pure REF_L / REF_G reference models give the cleanest L and G shapes.

### Inter-piece clearance screen

For every case, Step 4 compares the distance between the midsurface nodes of different pieces before and
after the imperfection. Two midsurfaces closer than one shell thickness overlap. Abaqus general contact
removes such initial overclosures by moving nodes without strain, which would locally change the imposed
imperfection. Step 4 then prints a warning and records the pairs under `cases.<model>.interpiece_clearance`.
The screen compares node-to-node distances: exact for matching meshes, indicative otherwise. After a Step 5
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
