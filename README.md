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
  --portable-export parquet ^
  --portable-modes-per-shard 8 ^
  --portable-export-workers auto ^
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
