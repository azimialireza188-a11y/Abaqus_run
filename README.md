# Abaqus_run

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
  --modal-audit
```

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
