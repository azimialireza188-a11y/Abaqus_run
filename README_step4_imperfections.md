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
