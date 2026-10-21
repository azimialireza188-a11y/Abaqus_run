# -*- coding: utf-8 -*-
"""STEP 4 ONLY: create a new CAE containing documented imperfect meshes from the exact completed reference pipeline run.

Reference identity comes from run-dir/pipeline_status.json; arbitrary CAE/ODB glob selection is forbidden.

Suggestions (screen actual step-3 ODB shapes; no CAE/solver needed):
  abaqus python abaqus_step4_imperfections.py --run-dir "completed run" --suggest

The default fast screen reads U and persisted physical walls, with geometric
G/L/D candidates and separate relative-piece/QC diagnostics. --suggest-source csv
retains the old report-only routine. See README_fast_modal_suggest.md.

Build after YOU confirm mode IDs and amplitudes:
  abaqus cae noGUI=abaqus_step4_imperfections.py -- --run-dir "completed run"
    --local-mode INTEGER --dist-mode INTEGER --local-high-t FACTOR
    --dist-mm MILLIMETRES --output-cae "new output.cae"
Every option is written '--name value' with a space: the Abaqus launcher rejects
'--name=value' before this script starts.
--cpus N|auto (default auto): CPU threads of this stage. Step 4 launches no solver;
the value sets the NumPy/BLAS threads of the build and is the default worker count
of --suggest (--suggest-cpus overrides it there).

local-high-t and dist-mm have NO assumed defaults. The low local level defaults
to 0.34*t (the research plan). With --local-metric normal, its scale is the max
absolute displacement along averaged original shell normals on the gauge nodes.
The default gauge is all shell nodes; use --local-gauge INSTANCE:LABEL,... to
select the stiffened-panel measurement points relevant to the cited statistic.
Numerical modal scaling alone does not establish statistical equivalence to a
published imperfection measurement. --dist-gauge uses max transverse norm.

Selectable mode sources (all optional; defaults reproduce the earlier behaviour):
  --local-source / --dist-source  folder or .odb to take that component's eigenmode
      from (default: the --run-dir reference ODB). A folder is either a reference
      pipeline run (pipeline_status.json -> its recorded ODB) or a folder with one
      completed <job>.odb + <job>.inp, e.g. the GDLC <run>_REF_L / <run>_REF_G models.
  --global-source analytical|folder|.odb  (default analytical half-sine bow);
      with a folder/.odb, --global-mode selects the eigenmode (e.g. REF_G mode 1),
      normalized to unit peak transverse displacement, so G1000/G3000 still mean
      peak L/1000 and L/3000. --global-mode 1,2 combines a DEGENERATE pair
      (equal eigenvalues, e.g. the two flexural modes of a square section, whose
      direction inside the pair is arbitrary in Abaqus) into the eigenmode whose
      bow points along --global-angle-deg, so the direction is reproducible.
      A single --global-mode with an explicit --global-angle-deg: when that mode has
      exactly one equal-eigenvalue partner in its ODB, the pair is used (the requested
      direction is otherwise not reproducible); without a partner the single mode is
      kept and the angle only fixes its sign (bow towards the angle), with a warning
      when the mode does not bow along that direction.
  Every source ODB must carry exactly the reference mesh (same instances, node
  labels, coordinates and connectivity - verified node by node) and a completed
  analysis. Only the components used by the selected cases are required.
  When the source folder holds a GDLC classification
  (mode_decomposition_GDLC/mode_participation.csv) with matching eigenvalues, the
  G/D/L/C/O shares of every selected mode are recorded and a mode that is not
  dominated (>= 80 %) by its intended mechanism is reported.
The untouched reference model is always kept in the output CAE as the perfect
(no-imperfection) baseline; Step 5 builds it as STEP5_PERFECT_* (token PERFECT).

Imperfection field of every case:  u0 = aL*phiL + aD*phiD + aG*phiG
Each component phi is normalized independently, THEN combined without rescaling.
Built-in cases (--cases): L_low, L_high, D, LD_pp, LD_pm, G1000, G3000 (default);
optional LD_mp and LD_mm; --cases all for all nine; --cases none for only --combo.
User-defined cases (repeatable; any L/D/G combination and amplitude):
  --combo NAME=aL,aD,aG    each amplitude: mm (3.6), thickness multiple (0.34t),
                           length fraction (L/1000), signed (-0.34t); 0 = absent.
  e.g. --combo LDG_pp=0.34t,3.6,L/1000 --combo LG_mp=-0.34t,0,L/1000 --combo L_mp=-0.34t,0,0
  (any number, any order; names differ case-insensitively from each other and from
  the selected --cases)
  With --combo and no explicit --cases, only the --combo cases are built.
Global bow = sin(pi*(z-z0)/L) in the --global-angle-deg direction (default global
X for a Z-axis column), with analytical peaks L/1000 and L/3000. This is not an
automatically chosen weak axis or an imported global eigenmode.
Bending runs (--load-case bending in Step 3) are detected from pipeline_status.json:
the default bow direction is then the LATERAL direction, i.e. along the neutral
axis (--bending-axis-deg), which is the direction of lateral-torsional buckling;
an explicit --global-angle-deg still wins. For bending the preferred global shape
is the REF_G eigenmode (lateral-torsional, with its twist): --global-source
<run>_REF_G --global-mode 1 (a single mode: bending breaks the flexural pair).
Each case reports a midsurface inter-piece clearance screen: node pairs of
different pieces that the imperfection brings closer than one shell thickness
would be initial contact overclosures, which Abaqus general contact removes by
strain-free node adjustment (i.e. it would silently alter the imperfection).

Only transverse eigenvector translations modify coordinates; axial U and nodal
rotations are not used as geometric imperfections. Ends stay at original nodes.
Native part geometry remains perfect; DISPLAY THE MESH to inspect imperfections.
Do NOT remesh/regenerate geometry: doing so can erase the imposed nodal offsets.
The source model's sections, contact, BCs, MPCs and steps/loads are preserved in
each copy. Existing Buckle steps are reference setup, NOT a GMNIA/Riks analysis.
No material plasticity, new step, INP, ODB or solver job is created/submitted.
All saved Job objects are removed from the NEW database to avoid submitting a
perfect reference job by mistake. The original CAE/ODB files are never saved over.
The full input specification and each case's amplitudes are stored in the Model
description inside the new CAE. No separate output data files are required.

Scope: four matching, straight, meshed shell instances, axis aligned with X/Y/Z;
uniform thickness. A matching step-3 CAE and ODB are required. Abaqus 2024/Python 3.
"""
import argparse
import builtins as python_builtins
import csv
import json
import math
import os
import re
import sys


def _early_option(argv, name):
    """Value of '--name value' (or '--name=value') before argparse runs; None when absent."""
    for i, item in enumerate(argv):
        if item == name and i+1 < len(argv):
            return argv[i+1]
        if item.startswith(name+'='):
            return item[len(name)+1:]
    return None


# Configure before NumPy loads. --suggest: CPU workers across modes without nesting
# all-core BLAS teams (the worker count still uses all available CPUs). Build: --cpus N
# sets the BLAS threads; auto leaves the library default (all cores).
from runtime_resources import configure_threads, cpu_request
if '--suggest' in sys.argv and _early_option(sys.argv, '--suggest-source') != 'csv':
    configure_threads(1)
else:
    try:
        _cpus = cpu_request(_early_option(sys.argv, '--cpus') or 'auto')
    except Exception:
        _cpus = 'auto'                      # argparse reports the invalid value later
    if _cpus != 'auto':
        configure_threads(_cpus)
import numpy as np
import abaqus_pipeline_contract as pipeline_contract

DEFAULT_CASES = ('L_low', 'L_high', 'D', 'LD_pp', 'LD_pm', 'G1000', 'G3000')


class Step4ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # CAE can route argparse's stderr only to abaqus.rpy, leaving the console with
        # a misleading "Exit code: 0". Keep the actual error visible before exiting.
        print('STEP4 INPUT ERROR: '+message)
        if 'mode' in message.lower():
            print('Use positive mode numbers (GLOBAL_MODES may be comma-separated). '
                  'Assign Windows variables with set "LOCAL_MODE=number", '
                  'set "DIST_MODE=number", set "GLOBAL_MODES=number,number". '
                  'Do not paste the next command into a set /p prompt.')
        sys.stdout.flush()
        try:
            super().error(message)
        except SystemExit as error:
            # CAE may hide both console streams and show only the final exception.
            # Carry the actual diagnostic through without changing argparse's exit code.
            error.step4_argument_error = message
            raise


def parse_arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' in argv:
        argv = argv[argv.index('--')+1:]
    elif '-cae' in argv:
        clean, i = [], 0
        while i < len(argv):
            if argv[i] == '-cae':
                i += 1
            elif argv[i] in ('-noGUI', '-tmpdir', '-lmlog'):
                i += 2
            else:
                clean.append(argv[i])
                i += 1
        argv = clean
    p = Step4ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--suggest', action='store_true', help='Screen actual ODB shapes; never select a mode automatically')
    p.add_argument('--suggest-source', choices=('odb','csv'), default='odb')
    p.add_argument('--cpus', type=cpu_request, default='auto',
                   help="CPU threads of this stage: 'auto' (all) or a positive integer. Step 4 runs no "
                        "solver; it sets the NumPy threads and the default --suggest-cpus")
    p.add_argument('--suggest-cpus', default=None, help='Default: --cpus (all available logical CPUs)')
    p.add_argument('--suggest-gpus', default='auto', help='Benchmark compatible GPUs against parallel CPU')
    p.add_argument('--suggest-refresh', action='store_true', help='Ignore cached geometric screening')
    p.add_argument('--source-cae', help='Optional explicit reference CAE path; must equal the artifact resolved from --run-dir')
    p.add_argument('--odb', help='Optional explicit reference ODB path; must equal the artifact resolved from --run-dir')
    p.add_argument('--model', help='Optional explicit model name; must equal pipeline_status build.job_name')
    p.add_argument('--step', default='Buckle')
    p.add_argument('--load-case', choices=pipeline_contract.LOAD_CASES, default=None,
                   help='Optional assertion of the source load case; loading is inherited, never converted')
    p.add_argument('--instances', nargs=4, help='Optional explicit instances; must be exactly P1 P2 P3 P4')
    p.add_argument('--local-mode', type=int)
    p.add_argument('--dist-mode', type=int)
    p.add_argument('--local-source', help='Folder or .odb for the LOCAL mode (default: --run-dir reference ODB)')
    p.add_argument('--dist-source', help='Folder or .odb for the DISTORTIONAL mode (default: --run-dir reference ODB)')
    p.add_argument('--global-source', default='analytical',
                   help="'analytical' (half-sine bow, default) or a folder/.odb whose --global-mode is used")
    p.add_argument('--global-mode',
                   help='Eigenmode of --global-source (required unless analytical): one mode, e.g. 1, '
                        'or a complete degenerate group, e.g. 1,2, aligned with --global-angle-deg')
    p.add_argument('--local-low-t', type=float, default=.34)
    p.add_argument('--local-high-t', type=float)
    p.add_argument('--dist-mm', type=float)
    p.add_argument('--local-metric', choices=('normal', 'transverse'), default='normal')
    p.add_argument('--local-gauge', help='Comma-separated INSTANCE:LABEL pairs; otherwise all selected nodes')
    p.add_argument('--dist-gauge', help='Comma-separated INSTANCE:LABEL pairs; otherwise all selected nodes')
    p.add_argument('--axis', choices=('x', 'y', 'z'), default='z')
    p.add_argument('--global-angle-deg', type=float, default=None,
                   help='Bow direction in degrees from the first transverse axis (default 0); '
                        'analytical bow or a --global-mode pair')
    p.add_argument('--cases', default=None,
                   help='Built-in cases (default: the seven DEFAULT_CASES, or none when --combo is given); '
                        'all, none, or a comma list')
    p.add_argument('--combo', action='append', default=[], metavar='NAME=aL,aD,aG',
                   help='User-defined case u0=aL*phiL+aD*phiD+aG*phiG; amplitudes in mm, t multiples '
                        '(0.34t) or length fractions (L/1000); repeatable')
    p.add_argument('--output-cae')
    p.add_argument('--skip-mode-quality-checks', action='store_true',
                   help='Use explicitly selected reference vectors without GDLC classification/purity checks; '
                        'file, mesh, load-case and nonzero-field requirements still apply')
    args = p.parse_args(argv)
    for name in ('run_dir', 'source_cae', 'odb', 'local_source', 'dist_source', 'global_source', 'output_cae'):
        value = getattr(args, name)
        unresolved = re.search(r'%([A-Za-z_][A-Za-z0-9_]*)%', value or '')
        if unresolved:
            p.error('--%s contains unexpanded Windows variable %s. Define it in the same CMD window '
                    'with set "%s=full-path", or enter a full literal path.' %
                    (name.replace('_', '-'), unresolved.group(0), unresolved.group(1)))
    args.global_modes = None
    if args.global_mode is not None:
        try:
            args.global_modes = tuple(int(v) for v in str(args.global_mode).replace(' ', '').split(','))
        except ValueError:
            p.error('--global-mode must contain comma-separated mode numbers; received %r' % args.global_mode)
        if not args.global_modes or len(set(args.global_modes)) != len(args.global_modes):
            p.error('--global-mode must contain distinct modes of one complete degenerate group')
    try:
        args.combos = parse_combos(args.combo)
    except ValueError as error:
        p.error(str(error))
    if args.suggest_cpus is None:
        args.suggest_cpus = str(args.cpus)
    for name, minimum in (('suggest_cpus',1),('suggest_gpus',0)):
        value=getattr(args,name)
        if value != 'auto':
            try:
                if int(value)<minimum: raise ValueError()
            except ValueError:
                p.error('--%s must be auto or an integer >= %d'%(name.replace('_','-'),minimum))
    args.run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    for name in ('local_source', 'dist_source'):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(os.path.expanduser(getattr(args, name))))
    if str(args.global_source).strip().lower() == 'analytical':
        args.global_source = 'analytical'
    else:
        args.global_source = os.path.abspath(os.path.expanduser(args.global_source))
    if not args.suggest:
        text = None if args.cases is None else args.cases.strip()
        if text is None:
            selected = [] if args.combos else list(DEFAULT_CASES)
        elif text.lower() == 'none':
            selected = []
        elif text.lower() == 'all':
            selected = list(CASE_COMPONENTS)
        else:
            selected = [name.strip() for name in text.split(',')]
            unknown = [name for name in selected if name not in CASE_COMPONENTS]
            if unknown or not selected or len(set(selected)) != len(selected):
                p.error('Invalid/duplicate --cases. Available: '+','.join(CASE_COMPONENTS)+' (or all/none)')
        if not selected and not args.combos:
            p.error('No case selected: give --cases and/or at least one --combo')
        clash = sorted(name for name in args.combos if name.upper() in {s.upper() for s in selected})
        if clash:                    # Windows file names (INP/ODB) are case-insensitive
            p.error('--combo name(s) %s repeat a selected --cases case; rename the combo or drop that case'
                    % ','.join(clash))
        args.selected_cases = selected+list(args.combos)
        need = set(c for name in selected for c in CASE_COMPONENTS[name])
        for specs in args.combos.values():
            need.update(c for c, spec in zip(COMPONENTS, specs) if spec[1] != 0.)
        args.needed_components = sorted(need)
        required = ['output_cae']
        if 'local' in need:
            required.append('local_mode')
        if 'L_high' in selected:
            required.append('local_high_t')
        if 'dist' in need:
            required.append('dist_mode')
        if any('dist' in CASE_COMPONENTS[name] for name in selected):
            required.append('dist_mm')
        if 'global' in need and args.global_source != 'analytical':
            required.append('global_mode')
        for name in required:
            if getattr(args, name) is None:
                p.error('--%s must be supplied explicitly for the selected cases; use --suggest first if needed'
                        % name.replace('_', '-'))
        for name in ('local_mode', 'dist_mode'):
            if getattr(args, name) is not None and getattr(args, name) < 1:
                p.error('--%s must be a positive mode number' % name.replace('_', '-'))
        if args.global_modes is not None and min(args.global_modes) < 1:
            p.error('--global-mode must contain positive mode numbers')
        if args.global_mode is not None and args.global_source == 'analytical':
            p.error('--global-mode needs --global-source <folder or .odb>')
        mode_sets = [(c, getattr(args, c+'_source') or args.run_dir,
                      set(args.global_modes or ()) if c == 'global' else {getattr(args, c+'_mode')})
                     for c in ('local', 'dist', 'global') if c in need
                     and (args.global_modes if c == 'global' else getattr(args, c+'_mode')) is not None]
        for i in range(len(mode_sets)):
            for j in range(i+1, len(mode_sets)):
                common = mode_sets[i][2] & mode_sets[j][2]
                if os.path.normcase(mode_sets[i][1]) == os.path.normcase(mode_sets[j][1]) and common:
                    p.error('%s and %s use the same mode %d of the same source; choose distinct modes'
                            % (mode_sets[i][0], mode_sets[j][0], min(common)))
        for name in ('local_low_t', 'local_high_t', 'dist_mm'):
            value = getattr(args, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                p.error('--%s must be positive and finite' % name.replace('_', '-'))
        if args.local_high_t is not None and args.local_high_t <= args.local_low_t:
            p.error('--local-high-t must exceed --local-low-t')
        if args.global_angle_deg is not None and not math.isfinite(args.global_angle_deg):
            p.error('--global-angle-deg must be finite')
        args.output_cae = os.path.abspath(os.path.expanduser(args.output_cae))
        if not args.output_cae.lower().endswith('.cae'):
            args.output_cae += '.cae'
    return args


def suggest_modes(run_dir):
    reference = pipeline_contract.load_reference_run(run_dir, require_completed=True)
    path = os.path.join(reference['run_dir'],
                        reference['job_name']+'_modal_wavelengths_enhanced_modes.csv')
    if not os.path.isfile(path):
        print('NO AUTOMATIC CANDIDATES: the reference enhanced-mode CSV is missing: '+path)
        print('Run the normal reference postprocessing first, or inspect the matching ODB manually. No CAE was created.')
        return
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    print('Reference job: '+reference['job_name'])
    print('Source: '+path)
    print('Heuristic suggestions only, NOT confirmed physical mode families. Verify full shapes in the step-3 ODB.')
    for family, flag in (('Local-like', '--local-mode'), ('Distortional-like', '--dist-mode')):
        candidates = [r for r in rows if r.get('family') == family and float(r['eigenvalue']) > 0]
        candidates.sort(key=lambda r: float(r['eigenvalue']))
        if not candidates:
            print('NO %s CANDIDATE in the step-3 classification; this does NOT prove that the ODB contains no such mode.' % family.upper())
            print('Manual mode-shape inspection is required before supplying '+flag)
        else:
            print('%s candidates (lowest positive eigenvalues first):' % family)
            for row in candidates[:5]:
                print('  mode=%s, eigenvalue=%s, half-wave=%s mm, flags=%s' %
                    (row['mode'], row['eigenvalue'], row['half_wavelength_mm'], row.get('enhanced_flags', row.get('status', ''))))
    print('No mode/amplitude was selected. Supply --local-high-t and --dist-mm explicitly when building.')


CASE_COMPONENTS = dict(L_low=('local',), L_high=('local',), D=('dist',),
                       LD_pp=('local', 'dist'), LD_pm=('local', 'dist'),
                       LD_mp=('local', 'dist'), LD_mm=('local', 'dist'),
                       G1000=('global',), G3000=('global',))


COMPONENTS = ('local', 'dist', 'global')
COMBO_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,19}$')
_NUMBER = r'(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'


def parse_amplitude(token):
    """'3.6' or '3.6mm' -> ('mm', 3.6); '0.34t' -> ('t', 0.34); 'L/1000' -> ('L', 0.001).
    A leading sign is allowed on every form; the result is (unit, factor, original text)."""
    text = str(token).strip().replace(' ', '')
    match = re.fullmatch(r'([+-]?)L/(%s)' % _NUMBER, text, re.I)
    if match:
        divisor = float(match.group(2))
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError('Invalid length-fraction amplitude: '+token)
        return ('L', (-1. if match.group(1) == '-' else 1.)/divisor, text)
    match = re.fullmatch(r'([+-]?%s)(t|mm)?' % _NUMBER, text, re.I)
    if not match:
        raise ValueError('Invalid amplitude %r: use mm (3.6), thickness multiples (0.34t) or L/1000' % token)
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError('Amplitude must be finite: '+token)
    return ('t' if (match.group(2) or '').lower() == 't' else 'mm', value, text)


def amplitude_mm(spec, thickness, length):
    unit, factor = spec[0], spec[1]
    return factor*(thickness if unit == 't' else length if unit == 'L' else 1.)


def parse_combos(texts):
    """--combo NAME=aL,aD,aG entries -> ordered {NAME: (specL, specD, specG)}.
    Any mix of the three families: a 0 amplitude leaves that family out (L, D, G, LD, LG,
    DG or LDG). Names must differ case-insensitively among the combos; a clash with a
    built-in case name matters only when that built-in case is selected too (checked by
    parse_arguments), so e.g. --cases none --combo LD_mp=... is allowed."""
    combos, folded = {}, set()
    for text in texts or ():
        if '=' not in text:
            raise ValueError('--combo must be NAME=aL,aD,aG, got %r' % text)
        name, values = text.split('=', 1)
        name = name.strip()
        if not COMBO_NAME.match(name):
            raise ValueError('--combo name %r: 1-20 letters/digits/underscores, starting with a letter' % name)
        if name.upper() in folded:   # Windows file names (INP/ODB) are case-insensitive
            raise ValueError('--combo name %r is given twice (names are compared case-insensitively)' % name)
        parts = values.split(',')
        if len(parts) != 3:
            raise ValueError('--combo %s needs exactly three amplitudes aL,aD,aG' % name)
        specs = tuple(parse_amplitude(part) for part in parts)
        if all(spec[1] == 0. for spec in specs):
            raise ValueError('--combo %s has no imperfection; the perfect baseline is always kept' % name)
        folded.add(name.upper())
        combos[name] = specs
    return combos


def case_definitions(thickness, low_factor, high_factor, dist_mm, length):
    high_factor = 0. if high_factor is None else high_factor
    dist_mm = 0. if dist_mm is None else dist_mm
    low, high = thickness*low_factor, thickness*high_factor
    return dict(L_low=(low, 0., 0.), L_high=(high, 0., 0.), D=(0., dist_mm, 0.),
        LD_pp=(low, dist_mm, 0.), LD_pm=(low, -dist_mm, 0.),
        LD_mp=(-low, dist_mm, 0.), LD_mm=(-low, -dist_mm, 0.),
        G1000=(0., 0., length/1000.), G3000=(0., 0., length/3000.))


def normalize_mode(displacement, normals, axis, gauge, metric):
    u = np.asarray(displacement, dtype=float).copy()
    if not np.all(np.isfinite(u)):
        raise ValueError('Non-finite mode displacements')
    u[:, axis] = 0.
    gauge = np.asarray(gauge, dtype=int)
    if not len(gauge):
        raise ValueError('Empty amplitude gauge')
    if metric == 'normal':
        signed = np.sum(u*normals, axis=1)[gauge]
        peak = int(np.argmax(np.abs(signed)))
        denominator = abs(float(signed[peak]))
        sign = 1. if signed[peak] >= 0 else -1.
    else:
        denominator = float(np.max(np.linalg.norm(u[gauge], axis=1)))
        signed = u[gauge].ravel()
        peak = int(np.argmax(np.abs(signed)))
        sign = 1. if signed[peak] >= 0 else -1.
    if denominator <= np.finfo(float).tiny:
        raise ValueError('Selected mode has no displacement in the chosen gauge/metric')
    return u*sign/denominator, dict(metric=metric, raw_denominator=denominator,
        eigenvector_sign_multiplier=sign, gauge_node_count=len(gauge))


def global_bow(coordinates, axis, angle_degrees):
    coordinates = np.asarray(coordinates)
    z = coordinates[:, axis]
    length = float(z.max()-z.min())
    if length <= 0:
        raise ValueError('Invalid column length')
    transverse = [i for i in range(3) if i != axis]
    wave = np.sin(np.pi*(z-z.min())/length)
    wave[(z == z.min()) | (z == z.max())] = 0.
    result = np.zeros_like(coordinates, dtype=float)
    angle = math.radians(angle_degrees)
    result[:, transverse[0]] = wave*math.cos(angle)
    result[:, transverse[1]] = wave*math.sin(angle)
    return result


def global_eigen_shape(fields, axis, angle_degrees=None, check_direction=True):
    """Unit-peak global imperfection from one eigenmode, or from a degenerate pair.

    One mode: transverse U normalized to unit peak (same sign rule as normalize_mode);
    with angle_degrees the sign is chosen instead so that the bow (mean transverse
    displacement) points towards that direction - a single mode cannot be rotated.
    Two modes (equal eigenvalues): any combination is again an eigenmode; the one whose
    mean transverse displacement (the bow of the member) points along angle_degrees is
    taken, then normalized to unit peak, so the bow direction no longer depends on the
    arbitrary basis Abaqus returns inside the degenerate eigenspace.
    Returns (shape, metadata with the achieved bow angle and a translation ratio:
    ~1 for a flexural bow, ~0 for a torsional mode)."""
    transverse = [i for i in range(3) if i != axis]
    shapes = []
    for field in fields:
        u = np.asarray(field, dtype=float).copy()
        if not np.all(np.isfinite(u)):
            raise ValueError('Non-finite global mode displacements')
        u[:, axis] = 0.
        shapes.append(u)
    if len(shapes) == 1:
        u, coefficients = shapes[0], [1.]
    elif len(shapes) >= 2:
        angle = math.radians(0. if angle_degrees is None else angle_degrees)
        target = np.array([math.cos(angle), math.sin(angle)])
        shape_norms = np.array([np.linalg.norm(s) for s in shapes])
        if np.any(shape_norms <= np.finfo(float).tiny):
            raise ValueError('Global eigenspace contains a zero transverse vector')
        basis = np.column_stack([s.ravel()/norm for s, norm in zip(shapes, shape_norms)])
        q, singular, right = np.linalg.svd(basis, full_matrices=False)
        if singular[-1] <= 1e-10*singular[0]:
            raise ValueError('Global eigenvectors are linearly dependent')
        means = np.column_stack([q[:, j].reshape(shapes[0].shape)[:, transverse].mean(axis=0)
                                 for j in range(q.shape[1])])
        bow_singular = np.linalg.svd(means, compute_uv=False)
        if check_direction and (bow_singular[-1] <= 1e-6*bow_singular[0] or bow_singular[0] <= 1e-12):
            raise ValueError('Global eigenspace does not span two independent bow directions')
        # Minimum physical displacement norm, not minimum coefficients in an arbitrary basis.
        weights = np.linalg.lstsq(means, target, rcond=1e-10)[0]
        if not check_direction and (bow_singular[0] <= 1e-12 or np.linalg.norm(means @ weights) <= 1e-12):
            # No usable mean bow in the requested direction. The user explicitly
            # selected these vectors without quality screening: retain their equal
            # unit-L2 superposition, rather than invent a pure/global aligned shape.
            coefficients = (1./shape_norms).tolist()
            u = python_builtins.sum((s/norm for s, norm in zip(shapes, shape_norms)), np.zeros_like(shapes[0]))
            print('WARNING: selected reference vectors cannot supply the requested mean-bow direction; '
                  'using their equal unit-L2 superposition without a direction/purity claim')
        else:
            coefficients = ((right.T @ (weights/singular))/shape_norms).tolist()
            u = (q @ weights).reshape(shapes[0].shape)
            if not check_direction and np.linalg.norm(means @ weights-target) > 1e-6:
                print('WARNING: requested mean-bow direction is outside the selected reference space; '
                      'using its projection onto that space without a direction/purity claim')
    else:
        raise ValueError('Global shape needs one mode or one degenerate pair')
    norms = np.linalg.norm(u, axis=1)
    peak = float(np.max(norms))
    if peak <= np.finfo(float).tiny:
        raise ValueError('Selected global mode has no transverse displacement')
    sign = 1.
    if len(shapes) == 1:
        bow = u[:, transverse].mean(axis=0)
        if angle_degrees is not None and np.linalg.norm(bow) > 1e-6*peak:
            angle = math.radians(angle_degrees)
            target = np.array([math.cos(angle), math.sin(angle)])
            alignment = float(np.dot(bow, target))/float(np.linalg.norm(bow))
            if check_direction and abs(alignment) < 1.-1e-6:
                raise ValueError('Requested global bow direction is outside this single-mode space; '
                                 'use its complete degenerate group or omit --global-angle-deg')
            sign = 1. if alignment >= 0 else -1.
        elif angle_degrees is not None and check_direction:
            raise ValueError('Selected global mode has no mean bow; a bow angle cannot orient torsion')
        else:
            flat = u.ravel()
            sign = 1. if flat[int(np.argmax(np.abs(flat)))] >= 0 else -1.
    u = u*sign/peak
    mean = u[:, transverse].mean(axis=0)
    mean_abs = float(np.mean(np.linalg.norm(u, axis=1)))
    return u, dict(metric='transverse', modes_combined=len(shapes), raw_denominator=peak,
                   eigenvector_sign_multiplier=sign, combination_coefficients=coefficients,
                   requested_bow_angle_deg=angle_degrees,
                   bow_angle_deg=float(math.degrees(math.atan2(mean[1], mean[0]))),
                   translation_ratio=float(np.linalg.norm(mean)/mean_abs) if mean_abs else 0.,
                   gauge_node_count=len(u))


BENDING_MIN_TRANSLATION_RATIO = .5    # |mean transverse u| / mean |u|: ~1 bow, ~0 twist or even half-waves
BENDING_MAX_BOW_DEVIATION_DEG = 10.


def orient_bending_global(shape, meta, axis, angle_degrees):
    """Bending global eigenmode = lateral-torsional buckling: the sections translate along the
    neutral axis (lateral direction) and twist about a centre on the tension side. Check that the
    selected single mode really is that shape - a net lateral bow, not a torsional or
    antisymmetric (even half-wave) mode, along the lateral direction angle_degrees - and point
    the bow towards +angle_degrees, so the sign of aG is reproducible ('p' = towards the lateral
    direction). Abaqus/FE asymmetry leaves the bow a fraction of a degree off the axis, so the
    direction test is a tolerance, not exact alignment. Returns (shape, meta)."""
    transverse = [i for i in range(3) if i != axis]
    ratio = float(meta['translation_ratio'])
    if not ratio >= BENDING_MIN_TRANSLATION_RATIO:
        raise ValueError('Bending global mode has no net lateral bow (translation ratio %.3f < %.2f): it is a '
                         'torsional or antisymmetric (even half-wave) mode, not the lateral-torsional buckling '
                         'shape; use the single-half-wave REF_G mode (normally --global-mode 1)'
                         % (ratio, BENDING_MIN_TRANSLATION_RATIO))
    deviation = abs((float(meta['bow_angle_deg'])-angle_degrees+90.) % 180.-90.)
    if deviation > BENDING_MAX_BOW_DEVIATION_DEG:
        raise ValueError('Bending global mode bows at %.2f deg, %.1f deg away from the lateral direction %g deg '
                         '(neutral axis of the applied moment); it is not the lateral-torsional mode of this '
                         'moment' % (meta['bow_angle_deg'], deviation, angle_degrees))
    target = np.array([math.cos(math.radians(angle_degrees)), math.sin(math.radians(angle_degrees))])
    if float(np.dot(shape[:, transverse].mean(axis=0), target)) < 0.:
        shape = -shape
        meta = dict(meta, eigenvector_sign_multiplier=-meta['eigenvector_sign_multiplier'],
                    bow_angle_deg=(meta['bow_angle_deg']+360.) % 360.-180.)
    meta = dict(meta, requested_bow_angle_deg=angle_degrees, bending_lateral_check=dict(
        deviation_deg=deviation, translation_ratio=ratio, max_deviation_deg=BENDING_MAX_BOW_DEVIATION_DEG,
        min_translation_ratio=BENDING_MIN_TRANSLATION_RATIO))
    return shape, meta


def default_global_angle(load_case):
    """Default bow direction: global X for compression (unchanged); for bending the lateral
    direction = the neutral-axis direction (angle measured from X like --bending-axis-deg)."""
    if (load_case or {}).get('type') == 'bending':
        return float(load_case.get('neutral_axis_angle_deg', 0.)) % 180.
    return 0.


def bending_global_eigen_shape(fields, axis, load_case, angle_degrees=None):
    """Check single modes and degenerate combinations against the applied moment.

    A requested imperfection angle cannot redefine the moment's lateral direction.
    Keep the lateral/torsional eigenfield intact; only its scale and sign change.
    """
    lateral = default_global_angle(load_case)
    angle = lateral if angle_degrees is None else angle_degrees
    deviation = abs((angle-lateral+90.) % 180.-90.)
    if not np.isfinite(angle) or deviation > BENDING_MAX_BOW_DEVIATION_DEG:
        raise ValueError('Bending eigenmode bow angle must follow the applied moment neutral axis '
                         '(lateral direction %g deg); --global-angle-deg cannot rotate this load case' % lateral)
    shape, meta = global_eigen_shape(fields, axis, angle if len(fields) > 1 else None)
    return orient_bending_global(shape, meta, axis, lateral)


def frame_eigenvalue(frame):
    """Eigenvalue of an Abaqus buckling frame (description 'EigenValue = ...', else frameValue)."""
    match = re.search(r'EigenValue\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)',
                      str(getattr(frame, 'description', '')), re.I)
    if match:
        return float(match.group(1))
    value = getattr(frame, 'frameValue', None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def degenerate_partners(eigenvalues, mode, tolerance=1e-3):
    """Span-based groups, consistent with GDLC; no transitive chaining of close pairs."""
    ordered = sorted((v, m) for m, v in eigenvalues.items() if v is not None and math.isfinite(v))
    groups = []
    for value, member in ordered:
        if not groups or abs(value-groups[-1][0][0]) > tolerance*max(abs(groups[-1][0][0]), 1e-300):
            groups.append([])
        groups[-1].append((value, member))
    return next((sorted(m for _, m in g if m != mode) for g in groups if any(m == mode for _, m in g)), [])


GDLC_EXPECTED = {'local': 'L', 'dist': 'D', 'global': 'G'}
GDLC_PURITY_PCT = 80.


def gdlc_mode_shares(folder, mode, eigenvalue, odb_sha256=None):
    """G/D/L/C/O shares of `mode` from <folder>/mode_decomposition_GDLC/mode_participation.csv.

    Used only when that classification exists and its eigenvalue matches the ODB frame
    (otherwise it belongs to a different solve and is reported as stale)."""
    path = os.path.join(folder, 'mode_decomposition_GDLC', 'mode_participation.csv')
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline='', encoding='utf-8-sig') as stream:
            row = next((r for r in csv.DictReader(stream) if str(r.get('mode', '')).strip() == str(mode)), None)
    except (OSError, csv.Error) as error:
        return dict(status='unreadable', csv=path, error=str(error))
    if row is None:
        return dict(status='mode_not_classified', csv=path)
    result = dict(status='ok', csv=path, csv_mtime=os.path.getmtime(path))
    try:
        result['eigenvalue'] = float(row['eigenvalue'])
        if eigenvalue is not None and not math.isclose(result['eigenvalue'], eigenvalue, rel_tol=2e-4, abs_tol=1e-9):
            result['status'] = 'stale_eigenvalue_mismatch'
            result['odb_eigenvalue'] = eigenvalue
            return result
        result['group_shares_pct'] = {k: float(row['rel_'+k+'_pct']) for k in 'GDLCO'
                                      if row.get('rel_'+k+'_pct') not in (None, '')}
        result['shares_pct'] = {k: float(row[k+'_pct']) for k in 'GDLCO' if row.get(k+'_pct') not in (None, '')}
        # Legacy files with only reliable shares cannot establish individual-mode purity.
        result['individual_shares_available'] = bool(result['shares_pct'])
        if not result['shares_pct'] and row.get('reliable_source', 'mode') == 'mode':
            result['shares_pct'] = dict(result['group_shares_pct'])
    except (KeyError, ValueError) as error:
        return dict(status='unreadable', csv=path, error=str(error))
    result['dominant'] = row.get('dominant') or (max(result['shares_pct'], key=result['shares_pct'].get) if result['shares_pct'] else None)
    result['label'] = row.get('label')
    result['reliable_source'] = row.get('reliable_source')
    result['cluster'] = row.get('cluster')
    report_path = os.path.join(os.path.dirname(path), 'basis_report.json')
    source_sha = None
    if os.path.isfile(report_path):
        try:
            with open(report_path, encoding='utf-8') as stream:
                report = json.load(stream)
            source_sha = report.get('source_odb_sha256')
            # REF_G / REF_L: did the classifier recover the imposed mechanism in every mode?
            result['reference_check'] = report.get('reference_check')
        except (OSError, ValueError):
            pass
    result['odb_source_verified'] = bool(source_sha and odb_sha256 and source_sha == odb_sha256)
    if odb_sha256 is not None and not result['odb_source_verified']:
        result['status'] = 'stale_odb_mismatch' if source_sha else 'unverified_odb_source'
    result['confidence'] = row.get('confidence')
    return result


def validate_bending_reference_mode(load_case, component, mode, classification):
    """A known impure reference vector must not pass the generic mixed-mode threshold.

    Check the selected vector, not the whole reference run: another eigenmode's
    failure does not invalidate a pure selected one. REF_L uses the same criterion
    for every load case; the REF_G criterion remains specific to bending. Original
    mixed sources and unavailable classifications retain their warning path.
    """
    expected = {'local': 'L', 'global': 'G'}.get(component)
    if (expected is None or (expected == 'G' and load_case.get('type') != 'bending') or not classification
            or classification.get('status') != 'ok'
            or (classification.get('reference_check') or {}).get('kind') != expected):
        return
    shares = classification.get('shares_pct', {})
    if expected == 'L':
        try:
            shares = {k: float(shares[k]) for k in 'GDLCO'}
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError('REF_L mode %d requires complete finite individual G/D/L/C/O shares' % mode)
        if (any(not np.isfinite(v) or v < 0. or v > 100. for v in shares.values())
                or not math.isclose(python_builtins.sum(shares.values()), 100., rel_tol=0., abs_tol=1e-3)):
            raise ValueError('REF_L mode %d has invalid individual G/D/L/C/O shares; '
                             'each must be finite in [0, 100] and their sum must be 100%%' % mode)
    primary = shares.get(expected, float('nan'))
    # CAE's execution namespace can shadow Python's sum with its own API.
    nonlocal_share = python_builtins.sum(shares.get(k, 0.) for k in ('G', 'D', 'C'))
    if (not np.isfinite(primary) or primary < 95.
            or (expected == 'L' and (not np.isfinite(nonlocal_share) or nonlocal_share > 5. or shares['O'] > 5.))):
        raise ValueError('REF_%s mode %d does not meet the reference criterion '
                         '(%s >= 95%%%s): %s=%.4g%%. Select a verified pure mode; '
                         'a changed reference requires new solved/classified results. The 80%% '
                         'mixed-mode warning threshold is not a reference acceptance criterion.' %
                         (expected, mode, expected, ' and G+D+C <= 5% and O <= 5%' if expected == 'L' else '',
                          expected, primary))


def interpiece_candidates(keys, xyz, cutoff):
    """Node pairs (i < j) of DIFFERENT pieces closer than cutoff in the perfect mesh."""
    xyz = np.asarray(xyz, dtype=float)
    names = sorted(set(k[0] for k in keys))
    piece = np.array([names.index(k[0]) for k in keys])
    cells = np.floor(xyz/cutoff).astype(np.int64)
    table = {}
    for i, cell in enumerate(map(tuple, cells)):
        table.setdefault(cell, []).append(i)
    neighbours = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    first, second = [], []
    for cell, members in table.items():
        nearby = np.asarray([j for o in neighbours for j in
                             table.get((cell[0]+o[0], cell[1]+o[1], cell[2]+o[2]), ())], dtype=np.int64)
        for i in members:
            j = nearby[(nearby > i) & (piece[nearby] != piece[i])]
            if len(j):
                j = j[np.linalg.norm(xyz[j]-xyz[i], axis=1) < cutoff]
                first.extend([i]*len(j))
                second.extend(j.tolist())
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


def clearance_screen(xyz, offsets, pairs, thickness):
    """Indicative inter-piece node distances before/after the imperfection.

    Distances below one thickness flag possible overclosure. Contact adjustment
    could change the imposed imperfection by moving nodes strain-free.
    Node-to-node distances are always indicative: tangential slip and face interiors
    can hide penetration, even with matching initial meshes. Zero flags is not clearance proof."""
    i, j = pairs
    if not len(i):
        return dict(candidate_pairs=0, perfect_min_mm=None, imperfect_min_mm=None, new_overclosure_pairs=0,
                    indicative_only=True)
    perfect = np.linalg.norm(xyz[j]-xyz[i], axis=1)
    moved = xyz+offsets
    imperfect = np.linalg.norm(moved[j]-moved[i], axis=1)
    bad = (imperfect < thickness*(1.-1e-3)) & (imperfect < perfect-1e-9)
    result = dict(indicative_only=True, candidate_pairs=int(len(i)), perfect_min_mm=float(perfect.min()),
                  imperfect_min_mm=float(imperfect.min()), new_overclosure_pairs=int(bad.sum()))
    if bad.any():
        k = int(np.argmin(np.where(bad, imperfect, np.inf)))
        result['worst_pair_indices'] = [int(i[k]), int(j[k])]
        result['worst_overclosure_mm'] = float(thickness-imperfect[k])
    return result


def shell_instances(model):
    return sorted(name for name, inst in model.rootAssembly.instances.items()
                  if len(inst.elements) and all(str(e.type).startswith(('S3', 'S4', 'S6', 'S8', 'S9', 'STRI')) for e in inst.elements))


def gauge_indices(text, keys):
    if not text:
        return list(range(len(keys)))
    lookup = {(name.upper(), label): i for i, (name, label) in enumerate(keys)}
    result = []
    for token in text.split(','):
        name, label = token.strip().rsplit(':', 1)
        key = (name.upper(), int(label))
        if key not in lookup:
            raise ValueError('Amplitude gauge node not found: '+token)
        result.append(lookup[key])
    return sorted(set(result))


def read_modes_and_mesh(model, odb, names, args, modes=None, require_positive=False):
    """Mesh (from the CAE, verified node-by-node against this ODB), the transverse U of
    the requested modes and the eigenvalue of every mode frame in the step.
    modes defaults to (local_mode, dist_mode)."""
    if modes is None:
        modes = (args.local_mode, args.dist_mode)
    axis = 'xyz'.index(args.axis)
    keys, xyz, normals, element_topology = [], [], [], {}
    cae_instances = model.rootAssembly.instances
    odb_names = {name.upper(): name for name in odb.rootAssembly.instances.keys()}
    instance_map = {}
    for name in names:
        if name.upper() not in odb_names:
            raise ValueError('Instance missing from ODB: '+name)
        odb_name = odb_names[name.upper()]
        instance_map[odb_name] = name
        ci, oi = cae_instances[name], odb.rootAssembly.instances[odb_name]
        cn = {node.label: np.asarray(node.coordinates, dtype=float) for node in ci.nodes}
        on = {node.label: np.asarray(node.coordinates, dtype=float) for node in oi.nodes}
        if set(cn) != set(on):
            raise ValueError('CAE/ODB node labels differ in '+name)
        tolerance = max(1e-5, 1e-6*max(np.max(np.abs(v)) for v in cn.values()))
        if any(np.max(np.abs(cn[label]-on[label])) > tolerance for label in cn):
            raise ValueError('CAE/ODB coordinates differ; use the matching PERFECT step-3 model: '+name)
        ce = {e.label: tuple(n.label for n in e.getNodes()) for e in ci.elements}
        oe = {e.label: tuple(e.connectivity) for e in oi.elements}
        if ce != oe:
            raise ValueError('CAE/ODB element connectivity differs in '+name)
        element_topology[name] = ce
        sums = {label: np.zeros(3) for label in cn}
        for element in oi.elements:
            kind = str(element.type)
            corners = element.connectivity[:3 if kind.startswith(('S3', 'S6', 'STRI')) else 4]
            points = [on[label] for label in corners]
            # CAE noGUI injects its own sum into the script's global namespace.
            vector = python_builtins.sum((np.cross(points[i]-points[0], points[i+1]-points[0])
                          for i in range(1, len(points)-1)), np.zeros(3))
            for label in element.connectivity:
                sums[label] += vector
        for label in sorted(cn):
            keys.append((name, label))
            xyz.append(cn[label])
            norm = np.linalg.norm(sums[label])
            normals.append(sums[label]/norm if norm else np.zeros(3))
    xyz, normals = np.asarray(xyz), np.asarray(normals)
    lookup = {key: i for i, key in enumerate(keys)}
    frames = mode_frames(odb, args.step)
    eigenvalues = {mode: frame_eigenvalue(frame) for mode, frame in frames.items()}
    fields, descriptions = {}, {}
    for mode in modes:
        if mode not in frames:
            raise ValueError('Requested mode %d absent from ODB %s' % (mode, getattr(odb, 'path', '')))
        value = eigenvalues[mode]
        if require_positive and (value is None or not math.isfinite(value) or value <= 0.):
            raise ValueError('Bending imperfection requires a finite positive eigenvalue; '
                             'mode %d is not a buckling mode for the applied moment direction' % mode)
        frame = frames[mode]
        if 'U' not in frame.fieldOutputs:
            raise ValueError('Mode has no nodal U field: %d' % mode)
        values = np.full((len(keys), 3), np.nan)
        for value in frame.fieldOutputs['U'].values:
            if value.instance is None or value.instance.name not in instance_map:
                continue
            key = (instance_map[value.instance.name], value.nodeLabel)
            if key not in lookup:
                continue
            double = str(value.precision) == 'DOUBLE_PRECISION'
            system = value.localCoordSystemDouble if double else value.localCoordSystem
            if system is not None and len(system):
                raise ValueError('U uses a local coordinate system; export global nodal U first')
            values[lookup[key]] = value.dataDouble if double else value.data
        if not np.all(np.isfinite(values)):
            raise ValueError('Incomplete U data for mode %d' % mode)
        values[:, axis] = 0.
        z = xyz[:, axis]
        tolerance = max(1e-5, 1e-6*float(z.max()-z.min()))
        ends = (abs(z-z.min()) <= tolerance) | (abs(z-z.max()) <= tolerance)
        peak = np.max(np.linalg.norm(values, axis=1))
        if np.max(np.linalg.norm(values[ends], axis=1)) > 1e-4*peak:
            raise ValueError('Mode %d does not satisfy zero transverse end displacement' % mode)
        values[ends] = 0.
        fields[mode] = values
        descriptions[str(mode)] = frame.description
    return keys, xyz, normals, fields, descriptions, element_topology, eigenvalues


def mode_frames(odb, step):
    """{mode number: frame} of the buckling step (frame 0 = base state is skipped)."""
    if step not in odb.steps:
        raise ValueError('Buckling step not found: '+step)
    frames = {}
    for frame in odb.steps[step].frames:
        mode = int(getattr(frame, 'mode', 0) or 0)
        if mode <= 0:
            match = re.search(r'\bMode\s*[:=]?\s*(\d+)', str(frame.description), re.I)
            mode = int(match.group(1)) if match else 0
        if mode > 0:
            if mode in frames:
                raise ValueError('Duplicate mode IDs in selected step')
            frames[mode] = frame
    return frames


def global_modes_for_angle(modes, eigenvalues, angle_requested):
    """Effective global modes. A single mode with an explicitly requested bow angle cannot be
    rotated; when it has exactly one equal-eigenvalue partner (a degenerate flexural pair, whose
    basis Abaqus returns arbitrarily) the pair is used so that the requested direction is met.
    Returns (modes, note)."""
    modes = tuple(modes)
    partners = degenerate_partners(eigenvalues, modes[0])
    complete = (modes[0],) + tuple(partners)
    if len(modes) > 1:
        if set(modes) != set(complete):
            raise ValueError('--global-mode must select one complete eigenvalue group (relative span 0.001): %s'
                             % ','.join(map(str, complete)))
    elif angle_requested and partners:
        return complete, 'complete equal-eigenvalue group %s used for the requested bow direction' % ','.join(map(str, complete))
    return modes, None


def _analysis_completed(odb_path):
    """True when the job that wrote this ODB finished successfully (.sta/.log)."""
    stem = os.path.splitext(odb_path)[0]
    job = re.escape(os.path.basename(stem).upper())
    for suffix, pattern in (('.sta', r'THE ANALYSIS HAS COMPLETED SUCCESSFULLY'),
                            ('.log', r'^\s*ABAQUS JOB\s+'+job+r'\s+COMPLETED\s*$')):
        path = stem+suffix
        if os.path.isfile(path):
            with open(path, errors='replace') as stream:
                text = stream.read().upper()
            if re.search(pattern, text, re.M) and 'EXITED WITH ERRORS' not in text:
                return True
    return False


def validate_bending_mode_source(info, reference, component):
    """A matching mesh alone cannot establish the bending load or bolt configuration."""
    if pipeline_contract.reference_load_case(reference['build'])['type'] != 'bending':
        return  # Preserve compression's established same-mesh source flexibility.
    source_build = info.get('source_build')
    if not isinstance(source_build, dict):
        raise ValueError('Bending mode source has no build provenance: '+info['folder'])
    if pipeline_contract.load_case_snapshot(source_build) != pipeline_contract.load_case_snapshot(reference['build']):
        raise ValueError('Mode source load case/axis differs from the bending run: '+info['folder'])
    expected_source = reference['source_inputs']
    if source_build.get('source_inputs') != expected_source:
        raise ValueError('Mode source structural configuration (geometry/material/bolts) differs: '+info['folder'])
    for key in ('length_mm', 'nodes', 'elements', 'rigid_links', 'boundary_conditions', 'loads'):
        if source_build.get(key) != reference['build'].get(key):
            raise ValueError('Mode source configuration differs (%s): %s' % (key, info['folder']))
    marker = info.get('gdlc_reference')
    if marker is None and source_build.get('job_name') != info.get('source_job'):
        raise ValueError('Bending reference provenance is missing: copied source build requires '
                         'gdlc_reference.json in '+info['folder'])
    if marker is not None:
        expected_kind = GDLC_EXPECTED[component]
        original = reference['paths']['inp']
        if (marker.get('kind') != expected_kind or marker.get('source_run') != os.path.basename(reference['run_dir'])
                or marker.get('source_inp') != os.path.basename(original)):
            raise ValueError('Wrong bending reference family/kind; use <bending-run>_REF_%s: %s' %
                             (expected_kind, info['folder']))
        if marker.get('source_inp_sha256') and marker['source_inp_sha256'] != pipeline_contract.file_sha256(original):
            raise ValueError('Bending reference was generated from a different original INP: '+info['folder'])
        if expected_kind == 'G' and marker.get('twist') == 'restrained':
            raise ValueError('Bending REF_G restrains twist and cannot supply lateral-torsional buckling: '+info['folder'])


def resolve_mode_source(spec, reference_odb, reference_run_dir, reference=None, component=None):
    """Resolve a --*-source argument to one completed buckling ODB.
    None (or the --run-dir folder) -> the reference ODB. A folder with
    pipeline_status.json -> the ODB recorded there. Any other folder -> its single
    completed <job>.odb with a matching <job>.inp (e.g. GDLC REF_G / REF_L models).
    A path to a .odb is used directly."""
    bending = reference is not None and pipeline_contract.reference_load_case(reference['build'])['type']=='bending'
    if not spec or os.path.normcase(os.path.abspath(spec)) == os.path.normcase(os.path.abspath(reference_run_dir)):
        if reference_odb is None:
            raise ValueError('The --run-dir reference run is only BUILT (no verified solved .odb), so it cannot '
                             'supply mode shapes: give every needed component its own solved source '
                             '(--local-source <run>_REF_L, --global-source <run>_REF_G; a distortional mode has '
                             'no GDLC reference model, so --dist-source must name another solved model), or solve '
                             'the reference run and record it with abaqus_complete_model_m20.py --resume-post')
        return reference_odb, dict(kind='reference_run', folder=reference_run_dir)
    path = os.path.abspath(spec)
    info = dict(kind='odb', folder=os.path.dirname(path))
    if os.path.isdir(path):
        info['folder'] = path
        if os.path.isfile(os.path.join(path, pipeline_contract.STATUS_FILE)):
            other = pipeline_contract.load_reference_run(path, require_completed=True)
            path = pipeline_contract.resolve_reference_artifact(other, None, 'odb')
            info['kind'] = 'pipeline_run'
            if bending:
                info['source_build'] = dict(other['build'], source_inputs=other['source_inputs'])
        else:
            candidates = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith('.odb')
                          and os.path.isfile(os.path.join(path, f[:-4]+'.inp'))
                          and _analysis_completed(os.path.join(path, f))]
            if len(candidates) != 1:
                raise ValueError('Mode source folder must hold exactly one completed <job>.odb with its '
                                 '<job>.inp; found %d in %s' % (len(candidates), path))
            path = candidates[0]
            info['kind'] = 'buckling_model'
    if not os.path.isfile(path) or not path.lower().endswith('.odb'):
        raise ValueError('Mode source is not an ODB file or a folder with one: '+str(spec))
    if os.path.exists(os.path.splitext(path)[0]+'.lck'):
        raise ValueError('Mode source ODB is locked (analysis running?): '+path)
    if info['kind'] != 'pipeline_run' and not _analysis_completed(path):
        raise ValueError('Mode source analysis did not complete successfully (.sta/.log): '+path)
    # Also inspect metadata for a direct .odb argument; it must not bypass the folder checks.
    marker_path = os.path.join(info['folder'], 'gdlc_reference.json')
    if os.path.isfile(marker_path):
        with open(marker_path, encoding='utf-8') as stream:
            marker = json.load(stream)
        info.update(gdlc_reference=marker, kind='gdlc_reference_'+str(marker.get('kind', '?')))
        if bending:
            if marker.get('job') != os.path.splitext(os.path.basename(path))[0]:
                raise ValueError('Bending reference metadata belongs to another ODB: '+path)
            if marker.get('input_sha256'):
                inp = os.path.splitext(path)[0]+'.inp'
                if not os.path.isfile(inp) or marker['input_sha256'] != pipeline_contract.file_sha256(inp):
                    raise ValueError('Bending reference INP changed after generation: '+inp)
        src = str(marker.get('source_run', ''))
        if src and src != os.path.basename(os.path.abspath(reference_run_dir)) and not bending:
            print('WARNING: %s was built from run %s, not from %s; the node-by-node mesh '
                  'check still decides compatibility' % (info['folder'], src, os.path.basename(reference_run_dir)))
    if bending:
        info['source_job'] = os.path.splitext(os.path.basename(path))[0]
        inp = os.path.splitext(path)[0]+'.inp'
        if not os.path.isfile(inp):
            raise ValueError('Bending mode source INP is missing: '+inp)
        build_path = os.path.splitext(path)[0]+'_build.json'
        if 'source_build' not in info and os.path.isfile(build_path):
            with open(build_path, encoding='utf-8') as stream:
                info['source_build'] = json.load(stream)
        validate_bending_mode_source(info, reference, component)
        info.pop('source_build', None)  # Validate once; avoid embedding the entire mesh record per component.
    return path, info


def apply_offsets(model, names, keys, reference_coordinates, offsets, topology):
    from abaqusConstants import ON, OFF
    import meshEdit
    assembly = model.rootAssembly
    dependent = tuple(assembly.instances[name] for name in names if assembly.instances[name].dependent == ON)
    if dependent:
        assembly.makeIndependent(instances=dependent)
    index = {key: i for i, key in enumerate(keys)}
    for name in names:
        instance = assembly.instances[name]
        nodes = instance.nodes  # MeshNodeArray is ordered by mesh index, not node label.
        rows = [index[(name, node.label)] for node in nodes]
        coordinates = reference_coordinates[rows]+offsets[rows]
        assembly.editNode(nodes=nodes, coordinates=tuple(tuple(float(v) for v in row) for row in coordinates),
                          projectToGeometry=OFF)
        actual = np.asarray([node.coordinates for node in assembly.instances[name].nodes])
        tolerance = max(1e-6, 1e-7*float(np.max(np.abs(reference_coordinates))))
        if actual.shape != coordinates.shape or np.max(np.abs(actual-coordinates)) > tolerance:
            raise RuntimeError('Edited node coordinates did not match the requested imperfection: '+name)
        current = {e.label: tuple(n.label for n in e.getNodes()) for e in assembly.instances[name].elements}
        if current != topology[name]:
            raise RuntimeError('Mesh connectivity changed while applying offsets: '+name)


def build(args):
    import caeModules
    from abaqus import openMdb, session
    from abaqusConstants import ON, BEAM_MPC
    from odbAccess import openOdb

    # A BUILT (unsolved) reference run is accepted: the perfect model is its CAE, and every mode
    # shape then comes from an explicitly given solved source of the same mesh (checked below).
    reference = pipeline_contract.load_reference_run(args.run_dir, require_completed=True, allow_built=True)
    load_case = pipeline_contract.load_case_snapshot(reference['build'],getattr(args,'load_case',None))
    source_cae = pipeline_contract.resolve_reference_artifact(reference, args.source_cae, 'cae')
    if reference['solved']:
        odb_path = pipeline_contract.resolve_reference_artifact(reference, args.odb, 'odb')
    elif args.odb:
        raise ValueError('--odb given, but the reference run is only BUILT (no solved .odb)')
    else:
        odb_path = None
        print('NOTE: reference run %s is BUILT, not solved: its CAE is the perfect model; mode shapes come '
              'only from the explicit component sources' % reference['job_name']); sys.stdout.flush()
    if os.path.exists(args.output_cae):
        raise ValueError('Output CAE already exists; select a new filename: '+args.output_cae)

    database = openMdb(pathName=source_cae)
    model_name = args.model or reference['job_name']
    if model_name != reference['job_name']:
        raise ValueError('STEP4 must use the reference pipeline model %s, not %s' %
                         (reference['job_name'], model_name))
    if model_name not in database.models:
        raise ValueError('Reference model is missing from source CAE: '+model_name)
    source = database.models[model_name]
    names = list(args.instances or reference['instances'])
    if tuple(names) != tuple(reference['instances']):
        raise ValueError('STEP4 instances must be exactly '+','.join(reference['instances']))
    if shell_instances(source) != sorted(reference['instances']):
        raise ValueError('Source CAE shell instances differ from the reference pipeline')

    # Enforce the exact upstream production contract before perturbing any node.
    if len(source.boundaryConditions) != int(reference['build']['boundary_conditions']):
        raise ValueError('Source CAE boundary conditions differ from pipeline_status.json')
    if len(source.loads) != int(reference['build']['loads']):
        raise ValueError('Source CAE loads differ from pipeline_status.json')
    if len(source.constraints) != int(reference['build']['rigid_links']):
        raise ValueError('Source CAE bolt constraint count differs from pipeline_status.json')
    bolt_names = sorted(name for name in source.constraints.keys() if name.startswith('BOLT_'))
    if len(bolt_names) != int(reference['source_inputs']['expected_links']):
        raise ValueError('Source CAE BOLT_* count differs from the exported seam/bolt definition')
    if any(getattr(source.constraints[name], 'mpcType', None) != BEAM_MPC for name in bolt_names):
        raise ValueError('Source CAE no longer uses the reference BEAM_MPC bolt model')
    if 'GeneralContact' not in source.interactions or 'Hard_Frictionless' not in source.interactionProperties:
        raise ValueError('Source CAE no longer contains the reference general hard/frictionless contact')
    thicknesses = set()
    for name in names:
        instance = source.rootAssembly.instances[name]
        assignments = source.parts[instance.partName].sectionAssignments
        if not assignments:
            raise ValueError('Missing shell section assignment: '+name)
        for assignment in assignments:
            thickness = getattr(source.sections[assignment.sectionName], 'thickness', None)
            if thickness is None or thickness <= 0:
                raise ValueError('A constant positive shell thickness is required')
            thicknesses.add(float(thickness))
    if len(thicknesses) != 1:
        raise ValueError('This step-4 builder requires uniform shell thickness')
    thickness = next(iter(thicknesses))
    # ---- component sources: each component may come from its own ODB (same mesh) ----
    need = set(args.needed_components)
    components = []
    for comp in COMPONENTS:
        if comp not in need or (comp == 'global' and args.global_source == 'analytical'):
            continue
        spec = getattr(args, comp+'_source')
        path, info = resolve_mode_source(spec, odb_path, reference['run_dir'],reference=reference,component=comp)
        modes = tuple(args.global_modes) if comp == 'global' else (getattr(args, comp+'_mode'),)
        if (not args.skip_mode_quality_checks and comp == 'global' and
                (len(modes) > 1 or args.global_angle_deg is not None)):
            odb = openOdb(path=path, readOnly=True)
            try:
                values = {m: frame_eigenvalue(f) for m, f in mode_frames(odb, args.step).items()}
            finally:
                odb.close()
            modes, note = global_modes_for_angle(modes, values, args.global_angle_deg is not None)
            if note:
                print('NOTE: global '+note); sys.stdout.flush()
                info = dict(info, automatic_degenerate_pair=note)
        components.append((comp, path, modes, info))
    global_modes_used = next((list(ms) for c, _, ms, _ in components if c == 'global'), None)
    for i in range(len(components)):
        for j in range(i+1, len(components)):
            common = set(components[i][2]) & set(components[j][2])
            if os.path.normcase(components[i][1]) == os.path.normcase(components[j][1]) and common:
                raise ValueError('%s and %s use the same mode %d of the same ODB; choose distinct modes'
                                 % (components[i][0], components[j][0], min(common)))
    by_odb = {}
    for comp, path, modes, info in components:
        by_odb.setdefault(os.path.normcase(path), [path, []])[1].append((comp, modes, info))
    keys = xyz = normals = topology = None
    raw, descriptions, sources = {}, {}, {}
    for norm_path, (path, wanted) in by_odb.items():
        modes = sorted(set(m for _, ms, _ in wanted for m in ms))
        print('Reading mode(s) %s from %s' % (','.join(str(m) for m in modes), path)); sys.stdout.flush()
        odb = openOdb(path=path, readOnly=True)
        try:
            k, x, n, fields, desc, topo, eigenvalues = read_modes_and_mesh(
                source, odb, names, args, modes=modes,
                require_positive=load_case['type']=='bending' and not args.skip_mode_quality_checks)
        finally:
            odb.close()
        if keys is None:
            keys, xyz, normals, topology = k, x, n, topo
        elif k != keys or not np.array_equal(x, xyz):
            raise ValueError('Mode source mesh differs from the reference mesh: '+path)
        sha = pipeline_contract.file_sha256(path)
        for comp, ms, info in wanted:
            raw[comp] = [fields[m] for m in ms]
            frames = [desc[str(m)] for m in ms]
            descriptions[comp] = frames[0] if len(ms) == 1 else frames
            values = [eigenvalues.get(m) for m in ms]
            partners = sorted(set(p for m in ms for p in degenerate_partners(eigenvalues, m)) - set(ms))
            classification = ([None]*len(ms) if args.skip_mode_quality_checks else
                              [gdlc_mode_shares(info['folder'], m, eigenvalues.get(m), sha) for m in ms])
            if not args.skip_mode_quality_checks:
                for m, item in zip(ms, classification):
                    validate_bending_reference_mode(load_case, comp, m, item)
            sources[comp] = dict(info, odb=path, odb_sha256=sha, odb_size=os.path.getsize(path),
                                 odb_mtime=os.path.getmtime(path),
                                 mode=int(ms[0]) if len(ms) == 1 else [int(m) for m in ms],
                                 frame=descriptions[comp], eigenvalue=values[0] if len(ms) == 1 else values,
                                 degenerate_partner_modes=partners,
                                 mode_quality_assessed=not args.skip_mode_quality_checks,
                                 gdlc=classification[0] if len(ms) == 1 else classification)
            if not args.skip_mode_quality_checks and comp == 'global' and len(ms) > 1:
                global_modes_for_angle(ms, eigenvalues, args.global_angle_deg is not None)
            if partners:
                print('WARNING: %s mode %s has (nearly) equal eigenvalue(s) in mode(s) %s: inside such a group '
                      'Abaqus returns an arbitrary basis, so the shape/direction of this single mode is not '
                      'reproducible.%s' % (comp, ','.join(str(m) for m in ms), ','.join(str(m) for m in partners),
                      ' Give the pair, e.g. --global-mode %d,%d, with --global-angle-deg.' % tuple(sorted((ms[0], partners[0])))
                      if comp == 'global' and len(ms) == 1 else ''))
            check = next((item.get('reference_check') for item in classification
                          if item and item.get('status') == 'ok' and item.get('reference_check')), None)
            if check is not None and not check.get('passed'):
                print('WARNING: the %s source %s FAILED its GDLC reference-model check (%s): the reference '
                      'model does not isolate one mechanism in all its modes; inspect the selected mode.'
                      % (comp, info['folder'], ', '.join('%s=%s' % (k, round(v, 3) if isinstance(v, float) else v)
                      for k, v in check.items() if k.startswith(('max_', 'min_')))))
            for m, item in zip(ms, classification):
                if item is None:
                    continue
                if item['status'] != 'ok':
                    print('NOTE: GDLC classification for %s mode %d not used (%s): %s'
                          % (comp, m, item['status'], item['csv']))
                    continue
                shares = item.get('shares_pct', {})
                expected = GDLC_EXPECTED[comp]
                text = ', '.join('%s %.1f%%' % (k, v) for k, v in shares.items())
                print('  GDLC %s mode %d: %s (dominant %s, confidence %s)'
                      % (comp, m, text, item.get('dominant'), item.get('confidence')))
                if item.get('dominant') != expected or shares.get(expected, 0.) < GDLC_PURITY_PCT:
                    print('WARNING: the %s component (mode %d) is not a clean %s mode (%s); the imperfection '
                          'will carry this mixture of mechanisms.' % (comp, m, expected, text))
    if keys is None:                                    # e.g. only analytical global cases
        if odb_path is None:
            raise ValueError('Only analytical components: the mesh is read from the reference .odb, but the '
                             'reference run is only BUILT; solve it or give a solved mode source')
        odb = openOdb(path=odb_path, readOnly=True)
        try:
            keys, xyz, normals, unused, unused2, topology, unused3 = read_modes_and_mesh(
                source, odb, names, args, modes=())
        finally:
            odb.close()
    axis = 'xyz'.index(args.axis)
    length = float(np.ptp(xyz[:, axis]))
    for name in names:
        z = xyz[[i for i, key in enumerate(keys) if key[0] == name], axis]
        if not np.allclose([z.min(), z.max()], [xyz[:, axis].min(), xyz[:, axis].max()], atol=1e-5, rtol=0):
            raise ValueError('The four pieces must have common ends')
    zero = np.zeros_like(xyz, dtype=float)
    local, local_meta = zero, None
    dist, dist_meta = zero, None
    if 'local' in raw:
        local, local_meta = normalize_mode(raw['local'][0], normals, axis,
            gauge_indices(args.local_gauge, keys), args.local_metric)
    if 'dist' in raw:
        dist, dist_meta = normalize_mode(raw['dist'][0], normals, axis,
            gauge_indices(args.dist_gauge, keys), 'transverse')
    angle = default_global_angle(load_case) if args.global_angle_deg is None else args.global_angle_deg
    if 'global' in raw:
        # eigenmode global shape (one mode or an aligned degenerate pair), unit peak transverse displacement
        explicit = args.global_angle_deg is not None
        lateral = load_case['type'] == 'bending'
        if args.skip_mode_quality_checks:
            global_shape, global_meta = global_eigen_shape(raw['global'], axis,
                angle if (len(raw['global']) >= 2 or explicit) else None, check_direction=False)
        elif lateral:          # same mechanical checks for single modes and degenerate groups
            global_shape, global_meta = bending_global_eigen_shape(raw['global'], axis, load_case,
                                                                   args.global_angle_deg)
        else:
            global_shape, global_meta = global_eigen_shape(raw['global'], axis,
                angle if (len(raw['global']) >= 2 or explicit) else None)
        global_text = ('Eigenmode(s) %s of %s, %s; unit peak transverse displacement; bow angle %.2f deg, '
                       'translation ratio %.3f' % (','.join(str(m) for m in global_modes_used),
                       sources['global']['odb'], 'complete degenerate group aligned with --global-angle-deg'
                       if len(raw['global']) >= 2 else ('single lateral-torsional mode (bending), checked and '
                       'signed towards the lateral direction %g deg' % angle if lateral else
                       'single mode, sign towards --global-angle-deg' if explicit else 'single mode'),
                       global_meta['bow_angle_deg'],
                       global_meta['translation_ratio']))
        if args.skip_mode_quality_checks:
            global_text = ('Explicitly selected reference vector(s) %s; unit peak transverse displacement; '
                           'mode quality NOT assessed; achieved bow angle %.2f deg' %
                           (','.join(str(m) for m in global_modes_used), global_meta['bow_angle_deg']))
        print('  global shape: '+global_text)
        if len(raw['global']) == 1 and explicit and not lateral:
            deviation = abs((global_meta['bow_angle_deg']-angle+90.) % 180.-90.)
            if global_meta['translation_ratio'] < .5 or deviation > 10.:
                print('WARNING: global mode %d cannot be rotated: its bow points at %.1f deg (translation ratio '
                      '%.2f), not along --global-angle-deg %g; the requested direction is NOT met.'
                      % (global_modes_used[0], global_meta['bow_angle_deg'], global_meta['translation_ratio'],
                         angle))
    else:
        global_shape, global_meta = global_bow(xyz, axis, angle), None
        global_text = 'Analytical half-sine bow; prescribed transverse direction, not inferred weak axis'
        if load_case['type'] == 'bending':
            global_text += ('; bending run: %s %.6g deg (lateral = along the neutral axis), no twist component'
                            % ('explicit angle' if args.global_angle_deg is not None else 'default lateral direction',
                               angle))
    cases = case_definitions(thickness, args.local_low_t, args.local_high_t, args.dist_mm, length)
    specs = {}
    for name, triple in args.combos.items():
        cases[name] = tuple(amplitude_mm(spec, thickness, length) for spec in triple)
        specs[name] = [spec[2] for spec in triple]
    selected = list(args.selected_cases)
    if any(name not in cases for name in selected):
        raise ValueError('Invalid case selection. Available: '+','.join(cases))
    case_names = ['STEP4_'+name for name in selected]
    if any(name in database.models for name in case_names):
        raise ValueError('STEP4 model names already exist in source; start from the perfect step-3 CAE')
    for name in selected:
        missing = [c for c, a in zip(COMPONENTS, cases[name]) if a != 0. and c not in need]
        if missing:
            raise RuntimeError('Internal error: case %s uses unread component(s) %s' % (name, missing))
    offsets_by_case = {name: cases[name][0]*local+cases[name][1]*dist+cases[name][2]*global_shape
                       for name in selected}
    # Inter-piece clearance screen: only pairs that can come within one thickness are kept.
    reach = max(float(np.max(np.linalg.norm(o, axis=1))) for o in offsets_by_case.values())
    pairs = interpiece_candidates(keys, xyz, thickness+2.*reach+1e-6)
    source_counts = {key: len(getattr(source, key)) for key in
        ('boundaryConditions', 'constraints', 'interactions', 'loads', 'materials', 'sections')}
    manifest = dict(stage=4, source_cae=source_cae, source_model=model_name, source_odb=odb_path,
        source_odb_size=os.path.getsize(odb_path) if odb_path else None,
        source_odb_mtime=os.path.getmtime(odb_path) if odb_path else None,
        reference_run_status='COMPLETED' if reference['solved'] else 'BUILT',
        reference_pipeline=pipeline_contract.reference_snapshot(reference, include_hashes=True),
        source_repository_counts=source_counts,
        settings=vars(args), thickness_mm=thickness, length_mm=length, source_frames=descriptions,
        load_case=load_case, global_angle_deg_used=angle, global_modes_used=global_modes_used,
        component_sources=sources,
        perfect_baseline_model=model_name,
        imperfection_field='u0 = aL*phiL + aD*phiD + aG*phiG (transverse nodal translations)',
        local_normalization=local_meta, distortional_normalization=dist_meta,
        global_normalization=global_meta,
        global_shape=global_text,
        combo_amplitude_specs=specs,
        convention='Only transverse nodal translations; independent component normalization; no rescaling after addition',
        scope='Imperfection specimens only. Existing elastic steps/loads retained as reference; no GMNIA setup or solver jobs.',
        measurement_warning='0.34t is the plan input, not proof of statistical equivalence. Select a relevant local gauge.',
        mesh_warning='Imperfections are in node coordinates. Show mesh. Do not remesh.', cases={})
    counts = dict(source_counts)
    for case_name in selected:
        a_local, a_dist, a_global = cases[case_name]
        offsets = offsets_by_case[case_name]
        name = 'STEP4_'+case_name
        print('Creating %s: local=%g mm, distortional=%g mm, global=%g mm' % (name, a_local, a_dist, a_global))
        sys.stdout.flush()
        model = database.Model(name=name, objectToCopy=source)
        apply_offsets(model, names, keys, xyz, offsets, topology)
        if any(len(getattr(model, key)) != value for key, value in counts.items()):
            raise RuntimeError('Copied model lost a boundary condition, connection or property')
        screen = clearance_screen(xyz, offsets, pairs, thickness)
        if 'worst_pair_indices' in screen:
            screen['worst_pair_nodes'] = [list(keys[i]) for i in screen.pop('worst_pair_indices')]
            print('WARNING: %s brings %d inter-piece node pair(s) closer than t=%g mm (worst %s, overclosure '
                  '%.3g mm). General contact will remove these overclosures by strain-free node adjustment, '
                  'which locally changes the imperfection; check the .msg file after the solve.'
                  % (name, screen['new_overclosure_pairs'], thickness, screen['worst_pair_nodes'],
                     screen['worst_overclosure_mm']))
        item = dict(local_component_mm=a_local, distortional_component_mm=a_dist, global_component_mm=a_global,
            amplitude_spec=specs.get(case_name),
            actual_max_transverse_offset_mm=float(np.max(np.linalg.norm(offsets, axis=1))),
            actual_max_normal_offset_mm=float(np.max(np.abs(np.sum(offsets*normals, axis=1)))),
            interpiece_clearance=screen)
        manifest['cases'][name] = item
    for name in case_names:
        database.models[name].setValues(description='STEP4_PIPELINE_COMPATIBLE '+json.dumps(
            dict(specification=manifest, current_case=manifest['cases'][name]), ensure_ascii=True))
    # Remove jobs only in the new in-memory database; source disk file is unchanged.
    for name in list(database.jobs.keys()):
        del database.jobs[name]
    source.setValues(description=str(source.description or '')+'\nSTEP4_MANIFEST='+json.dumps(manifest, ensure_ascii=True))
    os.makedirs(os.path.dirname(args.output_cae), exist_ok=True)
    if session.viewports:
        viewport = session.viewports[session.currentViewportName]
        viewport.setValues(displayedObject=database.models[case_names[0]].rootAssembly)
        viewport.assemblyDisplay.setValues(mesh=ON)
        viewport.view.fitView()
    database.saveAs(pathName=args.output_cae)
    print('SAVED: %s (%d imperfect models plus the untouched perfect reference model %s). '
          'No analysis submitted; mesh not regenerated.' % (args.output_cae, len(selected), model_name))
    for comp in COMPONENTS:
        if comp in sources:
            print('  %-6s mode %s from %s' % (comp, sources[comp]['mode'], sources[comp]['odb']))
    if 'global' in need and 'global' not in sources:
        print('  global analytical half-sine bow (angle %g deg)' % angle)
    if load_case['type'] == 'bending':
        print('  load case: bending, neutral axis %g deg (%s)' % (load_case.get('neutral_axis_angle_deg'),
                                                               load_case.get('tag')))
    print('  %-14s %9s %9s %9s %10s' % ('case', 'aL [mm]', 'aD [mm]', 'aG [mm]', 'max|u0|'))
    for case_name in selected:
        print('  %-14s %9.4g %9.4g %9.4g %10.4g' % ((case_name,)+tuple(cases[case_name])+(
            manifest['cases']['STEP4_'+case_name]['actual_max_transverse_offset_mm'],)))
    return manifest


def main(argv=None):
    try:
        args = parse_arguments(argv)
    except SystemExit as error:
        if error.code not in (None, 0) and ('-cae' in sys.argv or 'abaqus' in sys.modules):
            detail = getattr(error, 'step4_argument_error', None)
            message = ('STEP4 FAILED: invalid arguments'+(': '+detail if detail else '')+
                       '. No imperfection CAE was created. Correct the command-line values '
                       'and rerun Step 4 only.')
            if detail and 'mode' in detail.lower():
                message += (' Supply mode numbers directly or use explicit set assignments; '
                            'do not paste a block of set /p prompts.')
            print(message)
            sys.stdout.flush()
            # CAE reports SystemExit(2) as "Exit code: 0" on some installations.
            # A runtime exception keeps this an explicit failed script, not a clean exit.
            raise RuntimeError(message) from error
        raise
    if args.suggest:
        if args.suggest_source == 'csv':
            return suggest_modes(args.run_dir)
        # Preserve the exact upstream identity/provenance contract used by
        # Step 4/5; do not redirect to an arbitrary ODB in the folder.
        reference=pipeline_contract.load_reference_run(args.run_dir,require_completed=True)
        args.odb=pipeline_contract.resolve_reference_artifact(reference,args.odb,'odb')
        if args.model and args.model != reference['job_name']:
            raise ValueError('Suggestion model must match the reference pipeline')
        if args.instances and tuple(args.instances) != tuple(reference['instances']):
            raise ValueError('Suggestion instances must match the reference pipeline')
        args.instances=list(reference['instances'])
        args.suggest_reference=pipeline_contract.reference_snapshot(reference,include_hashes=False)
        args.suggest_section_segments=reference['source_inputs']['section_segments']
        from abaqus_fast_modal_suggest import suggest
        return suggest(args)
    return build(args)


if __name__ == '__main__':
    main()
