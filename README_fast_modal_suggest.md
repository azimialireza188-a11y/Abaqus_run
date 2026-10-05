# Fast Step-4 buckling-mode suggestions

The existing command now screens mode shapes directly from the completed ODB:

```bat
cd /d "C:\Users\810200014.HAMI.000\Documents\Abaqus_run"
git pull --ff-only
set "RUN_DIR=D:\CFS-Column\A4784_t2_qm1_0_0_0_R8_lipR20t_lipLen60_M80_L3600_gap10_nb19_end25-25_row15_run05"
abaqus python abaqus_step4_imperfections.py --run-dir "%RUN_DIR%" --suggest
```

No eigenvalue/GMNIA job is submitted, no CAE is opened, and no imperfection
amplitude or mode is selected automatically. The original model and ODB are
read-only. Model construction without `--suggest` retains its previous behavior.

This repository resolves the exact completed reference job through
`pipeline_status.json`, as before. It checks the reference model/instance/ODB
identity before screening and records its provenance without hashing the entire
ODB. The Step-4 model builder, Step-5 PERFECT case and 70% post-peak stop remain
unchanged.

Version 2 separates coordinate precision from cross-section discretization.
Free piece ends must match source coordinates within the numeric tolerance.
Interior physical wall boundaries may lie between mesh nodes; nearest-node
mapping is accepted within half the local adjacent mesh interval plus numeric
tolerance, with `wall_boundary_mesh_snapped` retained as an informational flag.
All expected physical walls must still be present and topology/rank checks must
pass. The JSON records each endpoint's source/mesh position, error, allowance
and pass/fail result. This permits approximate geometric screening, not exact
physical fold reconstruction or a mesh-convergence claim.

## What changes

The old suggestion routine only read the enhanced CSV family labels. The default
now evaluates global transverse nodal U on all resolved longitudinal stations.
No sine fit, matrix export, mFSM eigenproblem or chart generation is required.

The existing physical-wall method is reused:

- LOCAL: normal wall bending relative to endpoint-driven wall motion.
  Interpolation reproduces infinitesimal rigid rotation of initially curved walls.
- GLOBAL: whole-section translation/twist fitted from physical wall anchors.
  Flexural, torsional and flexural-torsional subtypes are geometric observations.
- DISTORTIONAL: remaining inextensional anchor/wall deformation.
- Independent rigid motion of the four pieces and extension/other residuals
  are diagnostics, never additional physical buckling families.

The report always retains the dominant G/D/L family, but only
`CANDIDATE` rows appear in the family suggestion list. `UNRESOLVED` is a diagnostic
status. Dominance must be at least 80% of the L/D/G component self-norm sum;
piece-relative or other self-norm at/above 25% blocks a suggestion. These
thresholds are screening settings, not calibrated probabilities or code limits.
Nonpositive eigenvalues, significant transverse end motion and fewer than four
mesh intervals per estimated half-wave also block a suggestion.

Half-wave estimates use sign changes of the strongest transverse track and are
only a resolution aid; they do not determine L/D/G. Unlike the enhanced harmonic
report, this estimate is not a fitted spectral wavelength.

Nearby eigenvalues are flagged using a relative adjacent gap of 0.02%. If their
dominant families disagree, that cluster is unresolved. Observed global subtypes
in a nearby cluster are retained only as diagnostics, since its eigenvectors may
rotate within the eigenspace. No expensive subspace decomposition is added.

## Geometry and output

The command requires four prismatic, axis-aligned linear S3/S4 shell instances,
at least 98% node coverage by full longitudinal tracks, common end coordinates,
and common longitudinal stations. It refuses to invent an interpolation for an
unsupported mesh. Use `--instances P1 P2 P3 P4`, `--axis`, `--step`, or `--odb`
when automatic discovery is ambiguous.

Validated `source_inputs.section_segments` from the reference
`pipeline_status.json` defines physical walls. Extra historical build JSON
files cannot override this reference geometry. Source endpoint mapping must pass
the documented local mesh checks.
Missing/unconfirmed source walls permit conservative mesh diagnostics, but
Local/Distortional suggestions remain unresolved. Limited physical wall coverage
is flagged. The source bend detector remains a geometric screening assumption;
its wall layout and mapping error are recorded for inspection.

Outputs beside the ODB:

- `<job>_fast_mode_screening.csv`: every mode, family, status, component indices,
  estimated half-wave and flags.
- `<job>_fast_mode_screening.json`: full geometry, settings/provenance, diagnostics,
  resource selection, elapsed time and all modes.

Percentages are normalized squared displacement component norms. Components are
not necessarily orthogonal, so these are **not strain energy, statistical
confidence, or mechanically validated modal participation**. The component-norm
sum/input-norm ratio is also reported. The command is for fast imperfection-mode
screening; it does not establish DSM critical loads or validate mFSM. Visually
check selected shapes before creating the imperfect models.

## Resource policy and repeat runs

CPU selection uses all available logical CPUs by default. Detached NumPy mode
arrays are processed in a dynamic thread queue; ODB API reads stay on the main
thread. BLAS uses one thread per mode worker to avoid nested CPU teams. The
standalone command configures this before importing NumPy; if imported into an
already-running Python session without runtime BLAS control, it may fall back
to a single mode worker and records that choice.

There is no fixed worker cap or reserved-memory percentage. Concurrency is
bounded by actual live available memory and the estimated mode working set;
allocating beyond available RAM does not accelerate the work. Geometry is
prepared once, unnecessary dense SVD bases are skipped, and dense temporary
section operators are released after compiling sparse wall and fold operators.

If compatible CuPy/CUDA is present in Abaqus Python, visible GPUs are tested in
FP64 against CPU results. GPU batch throughput, including host transfers, must
beat the actual parallel CPU alternative. Selected GPUs each run one in-flight
mode; otherwise CPU is used and the reason is reported. No additional package
installation is required for the CPU path. GPU acceleration is optional and has
not been validated on a physical GPU in the development environment.

Optional explicit resource overrides:

```bat
abaqus python abaqus_step4_imperfections.py --run-dir "%RUN_DIR%" --suggest --suggest-cpus auto --suggest-gpus auto
```

An unchanged repeat command reads the completed cache. Its signature includes
ODB path/size/modification time, build geometry, classification source files,
axis, step and selected instances. ODB contents are not hashed or rescanned for
cache lookup. For a forced fresh read (including an ODB replaced while preserving
its file size and modification time):

```bat
abaqus python abaqus_step4_imperfections.py --run-dir "%RUN_DIR%" --suggest --suggest-refresh
```

The prior CSV-only routine remains explicitly available:

```bat
abaqus python abaqus_step4_imperfections.py --run-dir "%RUN_DIR%" --suggest --suggest-source csv
```

## Verification

```bat
py -3 -m unittest -v test_fast_modal_suggest test_step4_imperfections test_modal_report test_physical_walls
```

Tests include rigid translation/twist, Local panel bending, inextensional panel
angle change, piece-relative motion, curved-reference rigid rotation, sign/scale
invariance, agreement with the original geometric operators, repeated-eigenvalue
flags, memory/CPU selection, simulated GPU batch selection, and mocked ODB
extraction/cache invalidation.

These tests do not substitute for a real Abaqus 2024 ODB run. Actual speed and
improved family accuracy on the project's modes remain to be measured on the
Windows/Abaqus system; no accuracy percentage or total runtime is promised.
