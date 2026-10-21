# -*- coding: utf-8 -*-
"""Extract the exact STEP5 Riks force-shortening curve from CAE + ODB.

Run with Abaqus/CAE because the STEP5 model description in the CAE contains the
reference force and end-area weights used by abaqus_step5_gmnia.py.

Example:
  abaqus cae noGUI=abaqus_step5_force_displacement.py -- ^
    --cae "Step5_GMNIA_FY240.cae" ^
    --odb "STEP5_D_FY240.odb"

Outputs: CSV, JSON summary, PNG and a dependency-free SVG plot.
Bending models (STEP5 metadata load_case.type == 'bending') give the moment -
end-rotation curve instead: M = LPF*reference_moment_Nmm and the work-conjugate
rotation theta = sum k_i*U3_i over both ends (files *_moment_rotation.*).
Combined P-M models (abaqus_pm_interaction.py) give P = LPF*P_ref and M = LPF*M_ref
together with delta, theta and the extreme-fibre shortening delta + theta*c,
plotted as the first-yield utilization P/P_y + M/M_y (files *_pm_response.*).
The metadata comes from the <model>_step5.json sidecar next to the ODB when it
exists (always written by the current builders); --cae is then optional.
No analysis is submitted; the CAE is closed without saving and the ODB is read-only.
"""
import argparse
import builtins as python_builtins
import csv
import json
import math
import os
import sys
from html import escape

SCRIPT_DIR = os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0,SCRIPT_DIR)

import abaqus_pipeline_contract as pipeline_contract

STEP5_PREFIX = 'STEP5_GMNIA '

# Plot/CSV conventions per load case. Compression strings are the production ones.
RESPONSE_AXES = dict(
    compression=dict(kind='compression', x='shortening_mm', y='force_kN', stem='_force_displacement',
                     title='Riks force-shortening', response='force-displacement', x_label='Axial shortening, delta (mm)',
                     y_label='Axial load, P (kN)', queue_y_label='Axial compressive load, P (kN)',
                     queue_x_label='Axial shortening (mm)', queue_y_short='Axial compressive load (kN)',
                     capacity_label='AsFy = %.3f kN', combined='STEP5_all_force_displacement_combined',
                     combined_title='STEP5 GMNIA - Combined force-shortening curves',
                     peak_text='Peak %.3f kN @ %.4g mm'),
    bending=dict(kind='bending', x='rotation_mrad', y='moment_kNm', stem='_moment_rotation',
                 title='Riks moment-rotation', response='moment-rotation', x_label='End rotation, theta (mrad)',
                 y_label='Bending moment, M (kN m)', queue_y_label='Bending moment, M (kN m)',
                 queue_x_label='End rotation (mrad)', queue_y_short='Bending moment (kN m)',
                 capacity_label='My = %.3f kN m', combined='STEP5_all_moment_rotation_combined',
                 combined_title='STEP5 GMNIA - Combined moment-rotation curves',
                 peak_text='Peak %.3f kN m @ %.4g mrad'),
    combined=dict(kind='combined', x='extreme_fibre_shortening_mm', y='utilization', stem='_pm_response',
                  title='Riks P-M response', response='P-M response',
                  x_label='Extreme-fibre end shortening, delta + theta*c (mm)',
                  y_label='First-yield utilization, P/Py + M/My (-)',
                  queue_y_label='First-yield utilization, P/Py + M/My (-)',
                  queue_x_label='Extreme-fibre shortening (mm)', queue_y_short='P/Py + M/My (-)',
                  capacity_label='First yield = %.3f', combined='STEP5_all_pm_response_combined',
                  combined_title='P-M GMNIA - Combined response curves',
                  peak_text='Peak utilization %.4f @ %.4g mm'))


def response_kind(info):
    """'bending' / 'combined' from the STEP5 metadata load case, else 'compression'."""
    kind = ((info or {}).get('load_case') or {}).get('type')
    return kind if kind in ('bending', 'combined') else 'compression'


METADATA_SUFFIX = '_step5.json'


def load_metadata(odb_path, model_name, cae=None):
    """STEP5 metadata: the <model>_step5.json sidecar beside the ODB, else the CAE description."""
    sidecar = os.path.join(os.path.dirname(os.path.abspath(odb_path)), model_name+METADATA_SUFFIX)
    if os.path.isfile(sidecar):
        with open(sidecar, encoding='utf-8') as stream:
            info = json.load(stream)
        if not isinstance(info, dict) or info.get('stage') != 5:
            raise ValueError('Invalid STEP5 metadata sidecar: '+sidecar)
        return info, sidecar
    if not cae:
        raise ValueError('No metadata sidecar %s and no --cae given' % sidecar)
    import caeModules
    database = pipeline_contract.open_working_mdb(cae)
    try:
        if model_name not in database.models.keys():
            raise ValueError('STEP5 model not found in CAE: '+model_name)
        return pipeline_contract.parse_prefixed_json(
            database.models[model_name].description, STEP5_PREFIX), cae
    finally:
        database.close()


def response_axes(info):
    return RESPONSE_AXES[response_kind(info)]


def parse_arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    elif '-cae' in argv:
        clean, i = [], 0
        while i < len(argv):
            if argv[i] == '-cae':
                i += 1
            elif argv[i] in ('-noGUI', '-tmpdir', '-lmlog'):
                i += 2
            else:
                clean.append(argv[i]); i += 1
        argv = clean
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cae', help='STEP5_GMNIA_FY*.cae; only needed when no <model>_step5.json sidecar exists')
    p.add_argument('--odb', required=True, help='Solved STEP5_*.odb')
    p.add_argument('--model', help='STEP5 model name; default is ODB basename')
    p.add_argument('--step', default='GMNIA')
    p.add_argument('--output-dir', help='Default: directory containing the ODB')
    p.add_argument('--cufsm-report-dir', help='Built-up CUFSM report directory for plotted SFy/ZFy/AsFy benchmarks')
    p.add_argument('--allow-partial', action='store_true',
                   help='Allow extraction from an in-progress ODB by using only the common complete frame/LPF prefix')
    args = p.parse_args(argv)
    args.cae = os.path.abspath(os.path.expanduser(args.cae)) if args.cae else None
    args.odb = os.path.abspath(os.path.expanduser(args.odb))
    if args.cufsm_report_dir:
        args.cufsm_report_dir = os.path.abspath(os.path.expanduser(args.cufsm_report_dir))
    args.output_dir = os.path.abspath(os.path.expanduser(
        args.output_dir or os.path.dirname(args.odb)))
    if args.cae and not os.path.isfile(args.cae):
        p.error('CAE not found: '+args.cae)
    if not os.path.isfile(args.odb):
        p.error('ODB not found: '+args.odb)
    if os.path.exists(os.path.splitext(args.odb)[0]+'.lck') and not args.allow_partial:
        p.error('ODB is locked; wait for the job to finish')
    return args


def _precision_vector(value):
    double = str(getattr(value, 'precision', '')) == 'DOUBLE_PRECISION'
    system = getattr(value, 'localCoordSystemDouble' if double else 'localCoordSystem', None)
    if system is not None and len(system):
        raise ValueError('End response requires global-coordinate nodal displacement')
    data = value.dataDouble if double else value.data
    if len(data) < 3 or not all(math.isfinite(float(v)) for v in data[:3]):
        raise ValueError('Incomplete/nonfinite nodal displacement')
    return data


def weight_map(rows):
    result = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError('Invalid end-area weight row')
        instance, label, area = row
        key = (str(instance).upper(), int(label))
        area = float(area)
        if not math.isfinite(area) or area <= 0:
            raise ValueError('End-area weights must be positive and finite')
        if key in result:
            raise ValueError('Duplicate end-area weight node: %s:%s' % key)
        result[key] = area
    if not result:
        raise ValueError('Empty end-area weight table')
    return result


def coefficient_map(rows):
    """Signed rotation coefficients [instance, label, k] -> {(INSTANCE, label): k}."""
    result = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError('Invalid end-rotation weight row')
        instance, label, value = row
        key = (str(instance).upper(), int(label))
        value = float(value)
        if not math.isfinite(value) or value == 0.0:
            raise ValueError('End-rotation weights must be finite and nonzero')
        if key in result:
            raise ValueError('Duplicate end-rotation weight node: %s:%s' % key)
        result[key] = value
    if not result:
        raise ValueError('Empty end-rotation weight table')
    return result


def _u3_values(field, region, keys):
    values = field.getSubset(region=region).values
    found = {}
    for value in values:
        instance = getattr(value, 'instance', None)
        if instance is None:
            continue
        key = (str(instance.name).upper(), int(value.nodeLabel))
        if key in keys:
            if key in found:
                raise ValueError('Duplicate end-node displacement: %s:%d' % key)
            found[key] = float(_precision_vector(value)[2])
    missing = sorted(set(keys)-set(found))
    if missing:
        preview = ', '.join('%s:%d' % item for item in missing[:8])
        raise ValueError('U3 missing for %d weighted end nodes; first: %s' % (len(missing), preview))
    return found


def coefficient_u3(field, region, coefficients):
    """sum_i k_i * U3_i (no normalization): the end rotation for bending weights."""
    found = _u3_values(field, region, coefficients)
    return python_builtins.sum(coefficients[key]*found[key] for key in coefficients)


def weighted_u3(field, region, weights):
    found = _u3_values(field, region, weights)
    total = python_builtins.sum(weights.values())
    return python_builtins.sum(weights[key]*found[key] for key in weights)/total


def lpf_history(step):
    """Return automatic Static-Riks LPF history as (frameValue, LPF) pairs."""
    matches = []
    for region_name in step.historyRegions.keys():
        region = step.historyRegions[region_name]
        for output_name in region.historyOutputs.keys():
            output = region.historyOutputs[output_name]
            key = str(output_name).strip().upper()
            description = str(getattr(output, 'description', '')).upper()
            if key == 'LPF' or 'LOAD PROPORTIONALITY FACTOR' in description:
                raw = output.data
                if raw is None:
                    continue  # Failed/in-progress increment may have an unwritten LPF repository.
                data = [(float(x), float(y)) for x, y in raw]
                if data:
                    matches.append((str(region_name), str(output_name), data))
    if not matches:
        raise ValueError('Automatic Riks LPF history output was not found in the ODB')
    reference = matches[0][2]
    for region_name, output_name, data in matches[1:]:
        if len(data) != len(reference):
            raise ValueError('Multiple inconsistent LPF histories found in the ODB')
        for (xa, ya), (xb, yb) in zip(reference, data):
            tol_x = 1e-10*max(1.0, abs(xa), abs(xb))
            tol_y = 1e-10*max(1.0, abs(ya), abs(yb))
            if abs(xa-xb) > tol_x or abs(ya-yb) > tol_y:
                raise ValueError('Multiple inconsistent LPF histories found in the ODB')
    return reference


def align_lpf_history(frames, history):
    """Align automatic Riks LPF history to field-output frames by sequence.

    In Abaqus/Standard Riks output the history abscissa and ODB frameValue can
    be offset by an increment even though the samples describe the same ordered
    converged increments. Therefore numeric nearest-x matching is not valid.
    This function accepts only the three auditable count patterns expected with
    frequency=1: same count, one missing initial history sample, or one extra
    initial zero history sample.
    """
    frames = list(frames)
    history = list(history)
    nf, nh = len(frames), len(history)
    if not nf or not nh:
        raise ValueError('Empty Riks frame sequence or LPF history')

    if nh == nf:
        if abs(float(frames[0].frameValue)) <= 1e-10 and abs(float(history[0][1])) > 1e-10:
            raise ValueError('LPF sequence would assign nonzero load to the initial field frame')
        offset = 0
        mode = 'same_count_by_sequence'
        pairs = [(history[i][0], history[i][1]) for i in range(nf)]
    elif nh == nf-1:
        first_value = float(frames[0].frameValue)
        if abs(first_value) > 1e-10*max(1.0, abs(first_value)):
            raise ValueError('LPF history has one fewer sample but the first field frame is not the initial frame')
        if abs(float(history[0][0])) <= 1e-10 and abs(float(history[0][1])) <= 1e-10:
            raise ValueError('LPF history already contains the initial zero; a later sample is missing')
        offset = -1
        mode = 'synthesized_initial_zero_then_sequence'
        pairs = [(first_value, 0.0)] + [(history[i][0], history[i][1]) for i in range(nh)]
    elif nh == nf+1:
        hx, hy = history[0]
        tolerance = 1e-10*max(1.0, abs(hx), abs(hy))
        if abs(hx) > tolerance or abs(hy) > tolerance:
            raise ValueError('LPF history has one extra sample but it is not an initial zero sample')
        if abs(history[1][1]) > 1e-10:
            raise ValueError('Extra LPF history would assign nonzero load to the initial field frame')
        offset = 1
        mode = 'skipped_extra_initial_zero_then_sequence'
        pairs = [(history[i+1][0], history[i+1][1]) for i in range(nf)]
    else:
        raise ValueError(
            'Cannot align Riks LPF history to field frames by sequence: %d frames, %d LPF samples' %
            (nf, nh))

    diagnostics = []
    for i, (history_x, lpf) in enumerate(pairs):
        frame_value = float(frames[i].frameValue)
        if not all(math.isfinite(v) for v in (frame_value, float(history_x), float(lpf))):
            raise ValueError('Non-finite Riks frame/LPF data at sequence index %d' % i)
        diagnostics.append(abs(frame_value-float(history_x)))
    return [float(pair[1]) for pair in pairs], dict(
        mode=mode, frame_count=nf, history_count=nh, history_index_offset=offset,
        max_abs_frameValue_minus_historyX=max(diagnostics),
        final_abs_frameValue_minus_historyX=diagnostics[-1])


def curve_rows(odb, step_name, info, allow_partial=False):
    if int(info.get('settings', {}).get('field_frequency', 1)) != 1:
        raise ValueError('field_frequency must be 1 for auditable LPF/field alignment')
    if step_name not in odb.steps:
        raise ValueError('ODB step not found: '+step_name)
    step = odb.steps[step_name]
    if not step.frames:
        raise ValueError('ODB step has no frames: '+step_name)
    kind = response_kind(info)
    bending = kind == 'bending'
    combined = kind == 'combined'
    if combined:
        tables = [info.get('end_area_weights'), info.get('end_rotation_weights')]
        if any(not isinstance(t, list) or len(t) != 2 for t in tables):
            raise ValueError('P-M metadata lacks the end_area_weights / end_rotation_weights tables')
        area_maps = [weight_map(t) for t in tables[0]]
        rotation_maps = [coefficient_map(t) for t in tables[1]]
        section = info.get('section_for_PM') or {}
        p_ref = float(info.get('reference_axial_force_N', float('nan')))
        m_ref = float(info.get('reference_moment_Nmm', float('nan')))
        p_y = float(section.get('squash_load_N') or float('nan'))
        m_y = float(section.get('first_yield_moment_Nmm') or float('nan'))
        c_extreme = float(section.get('c_extreme_mm') or float('nan'))
        # Summing consistent nodal forces leaves roundoff at pure-P / pure-M endpoints.
        if abs(p_ref) <= 1e-10*abs(p_y):
            p_ref = 0.
        if abs(m_ref) <= 1e-10*abs(m_y):
            m_ref = 0.
        if (not all(math.isfinite(v) for v in (p_ref, m_ref, p_y, m_y, c_extreme)) or min(p_ref, m_ref) < 0
                or p_ref+m_ref <= 0 or min(p_y, m_y, c_extreme) <= 0):
            raise ValueError('Invalid P-M reference quantities in STEP5 metadata')
    elif bending:
        weights = info.get('end_rotation_weights')
        if not isinstance(weights, list) or len(weights) != 2:
            raise ValueError('STEP5 bending metadata lacks the two end_rotation_weights tables')
        bottom_weights, top_weights = coefficient_map(weights[0]), coefficient_map(weights[1])
        pref = float(info.get('reference_moment_Nmm', float('nan')))
        if not math.isfinite(pref) or pref <= 0:
            raise ValueError('Invalid reference_moment_Nmm in STEP5 bending metadata')
    else:
        weights = info.get('end_area_weights')
        if not isinstance(weights, list) or len(weights) != 2:
            raise ValueError('STEP5 metadata lacks the two end_area_weights tables')
        bottom_weights, top_weights = weight_map(weights[0]), weight_map(weights[1])
    node_sets = odb.rootAssembly.nodeSets
    if 'STEP5_BOTTOM' not in node_sets.keys() or 'STEP5_TOP' not in node_sets.keys():
        raise ValueError('ODB lacks STEP5_BOTTOM/STEP5_TOP node sets')
    bottom = node_sets['STEP5_BOTTOM']
    top = node_sets['STEP5_TOP']
    if not bending and not combined:
        pref = float(info['reference_force_N_per_end'])
        if not math.isfinite(pref) or pref <= 0:
            raise ValueError('Invalid reference_force_N_per_end in STEP5 metadata')
    reference_stress = float(info.get('settings', {}).get('reference_stress', float('nan')))

    lpf_data = lpf_history(step)
    frames = list(step.frames)
    partial = False
    dropped_field_frames = 0
    dropped_history_samples = 0
    duplicate_initial_zero_samples = 0
    while (len(lpf_data) > 1 and all(abs(float(v)) <= 1e-10 for v in lpf_data[0])
           and all(abs(float(v)) <= 1e-10 for v in lpf_data[1])):
        lpf_data = lpf_data[1:]
        duplicate_initial_zero_samples += 1
    initial_history_zero = bool(lpf_data and abs(lpf_data[0][0]) <= 1e-10 and abs(lpf_data[0][1]) <= 1e-10)
    available_history_slots = len(frames) if initial_history_zero else max(0,len(frames)-1)
    if allow_partial and len(lpf_data) > available_history_slots:
        extra_initial_zero = (len(lpf_data) == len(frames)+1 and initial_history_zero
                              and abs(lpf_data[1][1]) <= 1e-10)
        if not extra_initial_zero:
            target = len(frames) if initial_history_zero else max(0,len(frames)-1)
            dropped_history_samples = len(lpf_data)-target
            lpf_data = lpf_data[:target]
            partial = True
    if len(frames) > len(lpf_data)+(0 if initial_history_zero else 1):
        if not allow_partial:
            raise ValueError(
                'ODB appears in-progress or incompletely flushed: %d field frames but %d LPF samples. '
                'Re-run after the job finishes, or pass --allow-partial to extract only the common complete prefix.' %
                (len(frames), len(lpf_data)))
        partial = True
        # Preserve the initial field frame plus one frame per available LPF sample when
        # the LPF history omits the initial zero sample. If the first LPF sample is zero
        # and counts align differently, fall back to an equal-length common prefix.
        first_lpf_zero = False
        if lpf_data:
            hx0, hy0 = lpf_data[0]
            tol0 = 1e-10*max(1.0, abs(hx0), abs(hy0))
            first_lpf_zero = abs(hx0) <= tol0 and abs(hy0) <= tol0
        target = len(lpf_data) if first_lpf_zero else min(len(frames), len(lpf_data)+1)
        dropped_field_frames = len(frames)-target
        frames = frames[:target]
    aligned_lpf, lpf_alignment = align_lpf_history(frames, lpf_data)
    lpf_alignment['partial'] = partial
    lpf_alignment['dropped_field_frames'] = dropped_field_frames
    lpf_alignment['dropped_history_samples'] = dropped_history_samples
    lpf_alignment['duplicate_initial_zero_samples'] = duplicate_initial_zero_samples
    rows = []
    for index, frame in enumerate(frames):
        if 'U' not in frame.fieldOutputs:
            raise ValueError('Frame %d has no U field output' % index)
        frame_value = float(frame.frameValue)
        lpf = aligned_lpf[index]
        if combined:
            field = frame.fieldOutputs['U']
            u3 = [weighted_u3(field, region, area_maps[side]) for side, region in ((0, bottom), (1, top))]
            rot = [coefficient_u3(field, region, rotation_maps[side]) for side, region in ((0, bottom), (1, top))]
            shortening, rotation = u3[0]-u3[1], rot[0]+rot[1]
            force_n, moment = lpf*p_ref, lpf*m_ref
            rows.append(dict(
                frame=index, frame_value=frame_value, lpf=lpf,
                shortening_mm=shortening, rotation_rad=rotation, rotation_mrad=1000.0*rotation,
                extreme_fibre_shortening_mm=shortening+rotation*c_extreme,
                force_N=force_n, force_kN=force_n/1000.0, moment_Nmm=moment, moment_kNm=moment/1e6,
                P_over_Py=force_n/p_y, M_over_My=moment/m_y, utilization=force_n/p_y+moment/m_y,
                nominal_extreme_stress_MPa=lpf*reference_stress if math.isfinite(reference_stress) else None))
            continue
        if bending:
            rot_bottom = coefficient_u3(frame.fieldOutputs['U'], bottom, bottom_weights)
            rot_top = coefficient_u3(frame.fieldOutputs['U'], top, top_weights)
            moment = lpf*pref
            rows.append(dict(
                frame=index, frame_value=frame_value, lpf=lpf,
                rotation_bottom_rad=rot_bottom, rotation_top_rad=rot_top,
                rotation_rad=rot_bottom+rot_top, rotation_mrad=1000.0*(rot_bottom+rot_top),
                moment_Nmm=moment, moment_kNm=moment/1e6,
                nominal_extreme_stress_MPa=lpf*reference_stress if math.isfinite(reference_stress) else None))
            continue
        u3_bottom = weighted_u3(frame.fieldOutputs['U'], bottom, bottom_weights)
        u3_top = weighted_u3(frame.fieldOutputs['U'], top, top_weights)
        shortening = u3_bottom-u3_top
        force_n = lpf*pref
        nominal_stress = lpf*reference_stress if math.isfinite(reference_stress) else None
        rows.append(dict(
            frame=index, frame_value=frame_value, lpf=lpf,
            u3_bottom_mm=u3_bottom, u3_top_mm=u3_top,
            shortening_mm=shortening,
            force_N=force_n, force_kN=force_n/1000.0,
            nominal_stress_MPa=nominal_stress))
    return rows, lpf_alignment


def peak_summary(rows):
    if not rows:
        raise ValueError('No curve rows')
    if 'utilization' in rows[0]:
        # proportional loading: P and M peak together at the largest LPF
        peak = max(rows, key=lambda row: row['lpf'])
        return dict((key, peak[key]) for key in (
            'frame', 'lpf', 'force_N', 'force_kN', 'moment_Nmm', 'moment_kNm', 'P_over_Py', 'M_over_My',
            'utilization', 'shortening_mm', 'rotation_mrad', 'extreme_fibre_shortening_mm',
            'nominal_extreme_stress_MPa'))
    if 'moment_Nmm' in rows[0]:
        peak = max(rows, key=lambda row: row['moment_Nmm'])
        return dict(
            frame=peak['frame'], lpf=peak['lpf'],
            moment_Nmm=peak['moment_Nmm'], moment_kNm=peak['moment_kNm'],
            rotation_mrad=peak['rotation_mrad'],
            nominal_extreme_stress_MPa=peak['nominal_extreme_stress_MPa'])
    peak = max(rows, key=lambda row: row['force_N'])
    return dict(
        frame=peak['frame'], lpf=peak['lpf'],
        force_N=peak['force_N'], force_kN=peak['force_kN'],
        shortening_mm=peak['shortening_mm'],
        nominal_stress_MPa=peak['nominal_stress_MPa'])


def _nice_ticks(lo, hi, count=6):
    if not math.isfinite(lo) or not math.isfinite(hi):
        return []
    if math.isclose(lo, hi):
        span = max(1.0, abs(lo)*0.1)
        lo, hi = lo-span, hi+span
    raw = abs(hi-lo)/max(1, count-1)
    exponent = math.floor(math.log10(raw)) if raw > 0 else 0
    fraction = raw/(10**exponent)
    nice = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    step = nice*(10**exponent)
    start = math.floor(lo/step)*step
    end = math.ceil(hi/step)*step
    values = []
    x = start
    for unused in range(100):
        if x > end+0.5*step:
            break
        values.append(x)
        x += step
    return values


def write_svg(path, rows, peak, title, axes=None, yield_lines=None):
    axes = axes or RESPONSE_AXES['compression']
    width, height = 1200, 800
    left, right, top, bottom = 115, 45, 75, 105
    xs = [row[axes['x']] for row in rows]
    ys = [row[axes['y']] for row in rows]
    references = normalize_capacity_lines(yield_lines or [], axes)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(0.0, min(ys)), max(ys+[line['value'] for line in references])
    if math.isclose(xmin, xmax):
        xmax = xmin+1.0
    if math.isclose(ymin, ymax):
        ymax = ymin+1.0
    xpad = 0.04*(xmax-xmin)
    ypad = 0.06*(ymax-ymin)
    xmin, xmax = xmin-xpad, xmax+xpad
    ymin, ymax = ymin, ymax+ypad
    plot_w, plot_h = width-left-right, height-top-bottom

    def sx(x):
        return left+(x-xmin)/(xmax-xmin)*plot_w

    def sy(y):
        return top+(ymax-y)/(ymax-ymin)*plot_h

    poly = ' '.join('%.2f,%.2f' % (sx(x), sy(y)) for x, y in zip(xs, ys))
    xticks, yticks = _nice_ticks(xmin, xmax), _nice_ticks(ymin, ymax)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">' %
        (width, height, width, height),
        '<rect width="100%%" height="100%%" fill="white"/>',
        '<text x="%d" y="38" text-anchor="middle" font-family="Arial" font-size="24">%s</text>' %
        (width//2, title),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black" stroke-width="2"/>' %
        (left, top+plot_h, left+plot_w, top+plot_h),
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="black" stroke-width="2"/>' %
        (left, top, left, top+plot_h),
    ]
    for value in xticks:
        if xmin <= value <= xmax:
            x = sx(value)
            lines.append('<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="#dddddd"/>' %
                         (x, top, x, top+plot_h))
            lines.append('<text x="%.2f" y="%d" text-anchor="middle" font-family="Arial" font-size="15">%.4g</text>' %
                         (x, top+plot_h+27, value))
    for value in yticks:
        if ymin <= value <= ymax:
            y = sy(value)
            lines.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#dddddd"/>' %
                         (left, y, left+plot_w, y))
            lines.append('<text x="%d" y="%.2f" text-anchor="end" dominant-baseline="middle" font-family="Arial" font-size="15">%.4g</text>' %
                         (left-12, y, value))
    for reference in references:
        y = sy(reference['value'])
        lines.append('<line x1="%d" y1="%.2f" x2="%d" y2="%.2f" stroke="#1676c3" stroke-dasharray="8 5"/>' %
                     (left, y, left+plot_w, y))
        lines.append('<text x="%d" y="%.2f" text-anchor="end" font-family="Arial" font-size="15">%s</text>' %
                     (left+plot_w-8, y-6, escape(reference['label'])))
    lines.extend([
        '<polyline points="%s" fill="none" stroke="#1f5f99" stroke-width="3"/>' % poly,
        '<circle cx="%.2f" cy="%.2f" r="6" fill="#b42318"/>' %
        (sx(peak[axes['x']]), sy(peak[axes['y']])),
        ('<text x="%.2f" y="%.2f" font-family="Arial" font-size="16">'+axes['peak_text']+'</text>') %
        (sx(peak[axes['x']])+12, sy(peak[axes['y']])-12,
         peak[axes['y']], peak[axes['x']]),
        '<text x="%d" y="%d" text-anchor="middle" font-family="Arial" font-size="19">%s</text>' %
        (left+plot_w//2, height-34, axes['x_label']),
        '<text transform="translate(30,%d) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="19">%s</text>' %
        (top+plot_h//2, axes['y_label']),
        '</svg>'
    ])
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))


RESPONSE_PAIR_VERSION = 'axial_bending_frames_v2'


def response_pair_samples(rows, key, selected_frame=None, count=6):
    """Sparse exact sample labels, not a fictitious inverse displacement/frame scale."""
    n = len(rows)
    peak = max(range(n), key=lambda i: max(abs(rows[i]['force_kN']), abs(rows[i]['moment_kNm'])))
    priority = [0, n-1, peak]
    priority += [i for i, row in enumerate(rows) if row['frame'] == selected_frame]
    lo, hi = min(row[key] for row in rows), max(row[key] for row in rows)
    span = hi-lo
    # Adaptive solvers may spend hundreds of increments in a tiny response
    # interval. Pick real samples across physical response, not uniform frames.
    priority += [min(range(n),key=lambda j:abs(rows[j][key]-(lo+i*span/max(1,count-1))))
                 for i in range(count)]
    groups = []
    for i in dict.fromkeys(priority):
        x = rows[i][key]
        exact = next((g for g in groups if g['x'] == x), None)
        if exact is not None:
            exact['indices'].append(i)
        elif not any(abs(g['x']-x) < .13*span for g in groups):
            groups.append(dict(x=x, indices=[i]))
    return sorted(groups, key=lambda g: g['x'])


def response_pair_rows(rows):
    """Shared numeric/frame validation for paired figures and Excel exports."""
    keys = ('shortening_mm', 'rotation_mrad', 'force_kN', 'moment_kNm')
    data = []
    for row in rows:
        try:
            frame = float(row['frame'])
            values = {key: float(row[key]) for key in keys}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Incomplete paired-response row: '+str(error))
        if (not math.isfinite(frame) or frame < 0 or frame != int(frame) or
                not all(math.isfinite(v) for v in values.values())):
            raise ValueError('Nonfinite response or invalid ODB frame index')
        data.append(dict(values, frame=int(frame)))
    if len(data) < 2 or any(b['frame'] <= a['frame'] for a, b in zip(data, data[1:])):
        raise ValueError('Paired response needs at least two frames in strict ODB order')
    return data


def write_response_pair(stem, rows, model, result=None, dataset_label=None, source_csv=None):
    """Plot synchronized P-delta/M-theta histories in ODB order, including reversals."""
    data = response_pair_rows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = result or {}
    selected = result.get('pm_point') or {}
    selected_frame = selected.get('frame')
    selected_row = next((r for r in data if r['frame'] == selected_frame), None)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    files, frame_labels = [], {}
    try:
        fig.subplots_adjust(left=.075, right=.97, bottom=.29, top=.75, wspace=.28)
        fig.suptitle(model+' | axial and bending response', fontsize=16, y=.975, weight='bold')
        status = result.get('outcome', 'RESPONSE_ONLY')
        fig.text(.075, .90, (dataset_label+' | ' if dataset_label else '')+
                 'Step: GMNIA | '+status+' | %d saved frames' % len(data), fontsize=10, color='#334155')
        detail = ('P-M selection: %s | Frame %s | capacity %s' %
                  (selected.get('selection_kind', 'none'), selected_frame,
                   'verified by current checks' if selected.get('capacity_verified') else 'unverified'))
        fig.text(.075, .84, detail, fontsize=10, color='#536270')
        response = {'shortening_mm': 'P-shortening', 'rotation_mrad': 'M-rotation'}.get(
            selected.get('selection_response'), selected.get('selection_response', 'none'))
        fig.text(.075, .80, 'Selection response: '+response+' | Policy: '+
                 selected.get('selection_policy', 'not recorded'), fontsize=9, color='#536270')
        for ax, xkey, ykey, xlabel, ylabel, title in (
                (axes[0], 'shortening_mm', 'force_kN', 'Axial shortening, delta (mm)', 'Axial force, P (kN)', 'Axial response'),
                (axes[1], 'rotation_mrad', 'moment_kNm', 'End rotation, theta (mrad)', 'Moment, M (kN m)', 'Bending response')):
            line, = ax.plot([r[xkey] for r in data], [r[ykey] for r in data], color='#176a9c', lw=1.8)
            line.set_gid('response-pair-curve')
            groups = response_pair_samples(data, xkey, selected_frame)
            labels = []
            frame_labels[xkey] = []
            for group in groups:
                samples = [data[i] for i in group['indices']]
                frames = [r['frame'] for r in samples]
                labels.append('%.5g\n%s' % (group['x'], ','.join('F%d' % f for f in frames)))
                frame_labels[xkey].append(dict(value=group['x'], frames=frames))
                ax.plot([r[xkey] for r in samples], [r[ykey] for r in samples], 'o',
                        ms=3.8, mfc='white', mec='#176a9c', ls='none', zorder=4)
            ax.set_xticks([g['x'] for g in groups])
            ax.set_xticklabels(labels, fontsize=9)
            ax.set(xlabel=xlabel+'\nODB frame index (marked samples)', ylabel=ylabel, title=title)
            ax.grid(True, color='#e5eaf0', lw=.65)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(axis='y', labelsize=9)
            ax.margins(x=.06, y=.10)
            if selected_row is not None:
                ax.plot(selected_row[xkey], selected_row[ykey], marker='D', ms=7,
                        mec='#b42318', mfc='#b42318' if selected.get('capacity_verified') else 'white',
                        ls='none', zorder=6)
                ax.annotate('P-M point: F%d' % selected_frame,
                            (selected_row[xkey], selected_row[ykey]), xytext=(6, 9),
                            textcoords='offset points', fontsize=8, color='#b42318')
        fig.text(.075, .12, 'Frame means steps["GMNIA"].frames[index] (zero-based); it is not increment number or LPF.',
                 fontsize=9, color='#536270')
        fig.text(.075, .075, 'Frame labels refer only to marked samples; repeated/reversed coordinates are retained. Red diamond: selected P-M point.',
                 fontsize=9, color='#536270')
        if source_csv:
            fig.text(.075, .035, 'Every frame and its P, M, delta and theta: '+os.path.basename(source_csv),
                     fontsize=8.5, color='#536270')
        os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
        for extension in ('png', 'svg'):
            path = stem+'.'+extension
            fig.savefig(path, dpi=180, facecolor='white')
            files.append(path)
    finally:
        plt.close(fig)
    return dict(files=files, frames=len(data), frame_labels=frame_labels, version=RESPONSE_PAIR_VERSION)


RESPONSE_WORKBOOK_VERSION = 'axial_bending_excel_v2'


def response_workbook_valid(path):
    """A missing/interrupted workbook is a cache miss, never a reason to solve again."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as archive:
            required = {'[Content_Types].xml', '_rels/.rels', 'xl/workbook.xml',
                        'xl/_rels/workbook.xml.rels', 'xl/styles.xml', 'xl/worksheets/sheet1.xml',
                        'xl/worksheets/sheet2.xml', 'xl/worksheets/sheet3.xml'}
            return required.issubset(archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def write_response_workbook(path, rows, model, result=None, dataset_label=None, source_csv=None, info=None):
    """Dependency-free XLSX with both complete, numeric response pairs and provenance."""
    import zipfile
    import xml.etree.ElementTree as ET
    data = response_pair_rows(rows)
    lpf = []
    for row in rows:
        raw = row.get('lpf')
        value = None if raw in (None, '') else float(raw)
        if value is not None and not math.isfinite(value):
            raise ValueError('Nonfinite LPF in response workbook')
        lpf.append(value)
    result, info = result or {}, info or {}
    selected = result.get('pm_point') or {}
    verified = str(selected.get('capacity_verified', False)).lower() in ('true', '1')
    metadata = [
        ('Model', model), ('Dataset', dataset_label or ''), ('Step', 'GMNIA'),
        ('Frame convention', 'Zero-based steps["GMNIA"].frames[index]; not increment number or LPF'),
        ('Solver outcome', result.get('outcome', 'RESPONSE_ONLY')),
        ('Selected point verified', str(verified)), ('Selected frame', selected.get('frame')),
        ('Selection kind', selected.get('selection_kind', 'none')),
        ('Selection response', selected.get('selection_response', '')),
        ('Selection policy', selected.get('selection_policy', '')),
        ('Selection diagnostics', selected.get('selection_diagnostics', '')), ('Saved frames', len(data)),
        ('Selected P (kN)', selected.get('P_kN')), ('Selected M (kN m)', selected.get('M_kNm')),
        ('Model case', info.get('source_case', '')), ('Bending axis (deg)', info.get('bending_axis_deg')),
        ('P-M angle (deg)', info.get('pm_angle_deg')), ('Source CSV', source_csv or ''),
        ('Source ODB', os.path.join(os.path.dirname(source_csv), model+'.odb') if source_csv else ''),
        ('Response convention', 'Original synchronized frames, in solver order; no smoothing or resampling'),
        ('Verification note', 'Exporting a stopped response does not certify capacity or solver completion')]
    sheets = [
        ('Axial', ['Frame (GMNIA, 0-based)', 'Shortening (mm)', 'Axial force (kN)', 'LPF'],
         [[r['frame'], r['shortening_mm'], r['force_kN'], p] for r, p in zip(data, lpf)]),
        ('Bending', ['Frame (GMNIA, 0-based)', 'End rotation (mrad)', 'Moment (kN m)', 'LPF'],
         [[r['frame'], r['rotation_mrad'], r['moment_kNm'], p] for r, p in zip(data, lpf)]),
        ('Metadata', ['Property', 'Value'], metadata)]
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    package = 'http://schemas.openxmlformats.org/package/2006/relationships'
    def xml(element):
        return ET.tostring(element, encoding='utf-8', xml_declaration=True)
    workbook = ET.Element('workbook', {'xmlns': ns, 'xmlns:r': rel})
    ET.SubElement(ET.SubElement(workbook, 'bookViews'), 'workbookView')
    sheet_list = ET.SubElement(workbook, 'sheets')
    relationships = ET.Element('Relationships', {'xmlns': package})
    types = ET.Element('Types', {'xmlns': 'http://schemas.openxmlformats.org/package/2006/content-types'})
    ET.SubElement(types, 'Default', Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
    ET.SubElement(types, 'Default', Extension='xml', ContentType='application/xml')
    ET.SubElement(types, 'Override', PartName='/xl/workbook.xml',
                  ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml')
    root_rels = ET.Element('Relationships', {'xmlns': package})
    ET.SubElement(root_rels, 'Relationship', Id='rId1', Type=rel+'/officeDocument', Target='xl/workbook.xml')
    # Keep a small, ordinary header style; data cells remain numeric General cells.
    styles = ET.fromstring('<styleSheet xmlns="'+ns+'"><fonts count="2">'
        '<font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>'
        '<fills count="3"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>'
        '<borders count="1"><border/></borders><cellStyleXfs count="1">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
    ET.SubElement(types, 'Override', PartName='/xl/styles.xml',
                  ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml')
    ET.SubElement(relationships, 'Relationship', Id='rIdStyles', Type=rel+'/styles', Target='styles.xml')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for index, (name, headers, values) in enumerate(sheets, 1):
            ET.SubElement(sheet_list, 'sheet', {'name': name, 'sheetId': str(index), 'r:id': 'rId%d' % index})
            ET.SubElement(relationships, 'Relationship', Id='rId%d' % index, Type=rel+'/worksheet',
                          Target='worksheets/sheet%d.xml' % index)
            ET.SubElement(types, 'Override', PartName='/xl/worksheets/sheet%d.xml' % index,
                          ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
            sheet = ET.Element('worksheet', {'xmlns': ns})
            extent = 'A1:%s%d' % (chr(64+len(headers)), len(values)+1)
            ET.SubElement(sheet, 'dimension', ref=extent)
            view = ET.SubElement(ET.SubElement(sheet, 'sheetViews'), 'sheetView', workbookViewId='0')
            ET.SubElement(view, 'pane', ySplit='1', topLeftCell='A2', activePane='bottomLeft', state='frozen')
            ET.SubElement(sheet, 'sheetFormatPr', defaultRowHeight='15')
            columns = ET.SubElement(sheet, 'cols')
            for column in range(1, len(headers)+1):
                width = 100 if name == 'Metadata' and column == 2 else 28
                ET.SubElement(columns, 'col', min=str(column), max=str(column), width=str(width), customWidth='1')
            table = ET.SubElement(sheet, 'sheetData')
            for row_index, line in enumerate([headers]+list(values), 1):
                record = ET.SubElement(table, 'row', r=str(row_index))
                for column, value in enumerate(line, 1):
                    if value is None:
                        continue
                    attributes = dict(r=chr(64+column)+str(row_index))
                    if row_index == 1:
                        attributes['s'] = '1'
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if not math.isfinite(value):
                            raise ValueError('Nonfinite numeric workbook cell')
                        cell = ET.SubElement(record, 'c', attributes)
                        ET.SubElement(cell, 'v').text = repr(value)
                    else:
                        cell = ET.SubElement(record, 'c', dict(attributes, t='inlineStr'))
                        ET.SubElement(ET.SubElement(cell, 'is'), 't',
                                      {'xml:space': 'preserve'}).text = str(value)
            ET.SubElement(sheet, 'autoFilter', ref=extent)
            archive.writestr('xl/worksheets/sheet%d.xml' % index, xml(sheet))
        archive.writestr('[Content_Types].xml', xml(types))
        archive.writestr('_rels/.rels', xml(root_rels))
        archive.writestr('xl/workbook.xml', xml(workbook))
        archive.writestr('xl/_rels/workbook.xml.rels', xml(relationships))
        archive.writestr('xl/styles.xml', xml(styles))
    return path


def collect_response_pairs(output_dir, points, in_place=False):
    """Render/cache per-job histories from validated CSVs; never extract ODBs or submit jobs."""
    import hashlib
    import re
    from pathlib import Path
    import abaqus_step5_queue as queue
    output = Path(output_dir)
    manifest_path = output/'PM_model_response_plots.json'
    previous = {}
    if manifest_path.is_file():
        try:
            with manifest_path.open(encoding='utf-8') as stream:
                saved = json.load(stream)
            if isinstance(saved,dict) and saved.get('version') == RESPONSE_PAIR_VERSION:
                previous = {entry['identity']: entry for entry in saved.get('models', [])}
        except (OSError, ValueError, KeyError, TypeError):
            # This is a derived plot cache only. Validate scientific source
            # artifacts below, then regenerate images after an interrupted write.
            previous = {}
    entries, files, seen, statuses = [], [], set(), {}
    for point in points:
        if point.get('source') != 'GMNIA':
            continue
        folder = Path(point['folder']).resolve()
        name = str(point['model'])
        if os.path.basename(name) != name or name in ('.', '..'):
            raise ValueError('Invalid response model name: '+name)
        identity = os.path.normcase(str(folder))+'|'+name
        if identity in seen:
            continue
        seen.add(identity)
        metadata, status_path = folder/(name+METADATA_SUFFIX), folder/'STEP5_queue_status.json'
        if not metadata.is_file() or not status_path.is_file():
            continue  # Historic point-only imports have no synchronized response history.
        info = json.loads(metadata.read_text(encoding='utf-8-sig'))
        if response_kind(info) != 'combined':
            continue
        if folder not in statuses:
            statuses[folder] = json.loads(status_path.read_text(encoding='utf-8-sig'))
        records = [r for r in statuses[folder] if r.get('model') == name]
        if len(records) != 1:
            raise ValueError('Missing/duplicate paired-response queue record: '+name)
        ratio = records[0].get('postpeak_stop_ratio',(info.get('settings') or {}).get('postpeak_stop_ratio',.70))
        entry = queue.completed_curve(folder, name, info, float(ratio), allow_incomplete=True, record=records[0])
        if entry is None:
            raise ValueError('Paired-response history is not safely reusable: '+name)
        result, unused_curve = entry
        csv_path = folder/(name+'_pm_response'+('_partial' if result.get('partial') else '')+'.csv')
        payload = csv_path.read_bytes()
        label = str(point.get('dataset_label') or folder.name)
        signature = hashlib.sha256(json.dumps(dict(version=RESPONSE_PAIR_VERSION, dataset_label=label,
            csv_sha256=hashlib.sha256(payload).hexdigest(), input_sha256=info.get('input_sha256'),
            outcome=result.get('outcome'), selected=result.get('pm_point'),
            partial=result.get('partial')), sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()
        slug = re.sub('[^A-Za-z0-9_-]+', '_', label).strip('_')[:64] or 'dataset'
        tag = hashlib.sha256(os.path.normcase(str(folder)).encode('utf-8')).hexdigest()[:8]
        stem = (output/(name+'_axial_bending_response') if in_place else
                output/'response_curves'/(slug+'_'+tag)/(name+'_axial_bending_response'))
        paths = [str(stem)+'.'+extension for extension in ('png', 'svg')]
        cached = previous.get(identity)
        rows = None
        if not (cached and cached.get('signature') == signature and cached.get('files') == paths and
                all(Path(path).is_file() and Path(path).stat().st_size for path in paths)):
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            plot = write_response_pair(str(stem), rows, name, result, label, str(csv_path))
            cached = dict(identity=identity, signature=signature, model=name, dataset=str(folder),
                          source_csv=str(csv_path), source_odb=str(folder/(name+'.odb')), step='GMNIA',
                          outcome=result.get('outcome'),
                          selected_frame=(result.get('pm_point') or {}).get('frame'), **plot)
        workbook_path = str(stem)+'.xlsx'
        workbook_signature = hashlib.sha256((signature+RESPONSE_WORKBOOK_VERSION).encode('utf-8')).hexdigest()
        if not (cached.get('workbook') == workbook_path and
                cached.get('workbook_signature') == workbook_signature and response_workbook_valid(workbook_path)):
            if rows is None:
                with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
            write_response_workbook(workbook_path,rows,name,result,label,str(csv_path),info)
        cached.update(workbook=workbook_path,workbook_signature=workbook_signature,
                      workbook_version=RESPONSE_WORKBOOK_VERSION, dataset_label=label,
                      selection_kind=(result.get('pm_point') or {}).get('selection_kind'),
                      selection_response=(result.get('pm_point') or {}).get('selection_response'),
                      bending_axis_deg=info.get('bending_axis_deg'), pm_angle_deg=info.get('pm_angle_deg'),
                      model_case=info.get('source_case'), fy_MPa=(info.get('settings') or {}).get('fy'))
        entries.append(cached); files.extend(paths+[workbook_path])
    if entries:
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(dict(version=RESPONSE_PAIR_VERSION, models=entries),
                                            indent=2, allow_nan=False), encoding='utf-8')
    return files


def write_response_path_overview(output_dir, entries, stem='STEP5_response_paths'):
    """All finished histories: response versus LPF and actual stored frame versus LPF.

    Paginate instead of shrinking an arbitrary number of jobs into unreadable axes.
    No curve is sorted, smoothed, certified, or omitted for lack of convergence.
    """
    import textwrap
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for start in range(0, len(entries), 6):
        page = entries[start:start+6]
        fig, axes = plt.subplots(len(page), 2, figsize=(13, 3.2*len(page)+1.2), squeeze=False)
        try:
            fig.subplots_adjust(left=.075, right=.97, bottom=.65/(3.2*len(page)+1.2),
                                top=1.-.65/(3.2*len(page)+1.2), hspace=.85, wspace=.25)
            fig.suptitle('Response paths | load reduction alone does not verify collapse', fontsize=13, weight='bold')
            for pair, entry in zip(axes, page):
                with open(entry['source_csv'], encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                data = response_pair_rows(rows)
                lpf = [float(row['lpf']) for row in rows]
                if not all(math.isfinite(value) for value in lpf):
                    raise ValueError('Nonfinite LPF: '+entry['source_csv'])
                key = entry.get('selection_response')
                if key not in ('shortening_mm', 'rotation_mrad'):
                    key = 'rotation_mrad' if max(abs(row['force_kN']) for row in data) < 1e-9 else 'shortening_mm'
                label = 'End rotation (mrad)' if key == 'rotation_mrad' else 'Axial shortening (mm)'
                peak = max(range(len(lpf)), key=lpf.__getitem__)
                selected = next((i for i, row in enumerate(data) if row['frame'] == entry.get('selected_frame')), None)
                title = (entry.get('dataset_label', '')+' | '+entry['model']+' | '+entry.get('outcome','UNKNOWN')+
                         '\naxis=%s deg; ray=%s deg; Fy=%s MPa; %d frames; selected F%s | %s' %
                         (entry.get('bending_axis_deg','?'), entry.get('pm_angle_deg','?'), entry.get('fy_MPa','?'),
                          len(data), entry.get('selected_frame'), entry.get('selection_kind','none')))
                for ax, x, xlabel in ((pair[0], [row[key] for row in data], label),
                                      (pair[1], [row['frame'] for row in data], 'Stored ODB frame (zero-based)')):
                    line, = ax.plot(x, lpf, 'o-', color='#0072b2', ms=2, lw=1.25)
                    line.set_gid('response-path-curve')
                    ax.plot(x[peak], lpf[peak], 'o', mfc='white', mec='#b2182b', ms=6)
                    if selected is not None:
                        ax.plot(x[selected], lpf[selected], 'D', mfc='white', mec='#b2182b', ms=5)
                    ax.set(xlabel=xlabel, ylabel='Load proportionality factor', title='\n'.join(textwrap.wrap(title.split('\n')[0], 72))+ '\n'+title.split('\n')[1])
                    ax.title.set_fontsize(7.5)
                    ax.tick_params(labelsize=8)
                    ax.grid(color='#e7ebef')
            fig.text(.075, .012, 'Circle: attained maximum; diamond: selected P-M point. Failed/stopped histories remain unverified. Original solver order.', fontsize=8, color='#536270')
            name = stem+('' if start == 0 else '_%03d' % (start//6+1))
            for extension in ('png','svg'):
                path = str(output/(name+'.'+extension))
                fig.savefig(path, dpi=160, facecolor='white'); files.append(path)
        finally:
            plt.close(fig)
    return files


def publish_queue_responses(run_dir):
    """Incremental reports from saved CSV only; no ODB opening or solver submission."""
    from pathlib import Path
    run_dir = Path(run_dir).resolve()
    context = json.loads(os.environ.get('STEP5_LIVE_REPORT_CONTEXT', '{}'))
    datasets = context.get('datasets') or [dict(root=str(run_dir), label=run_dir.name)]
    current = next((entry for entry in datasets if Path(entry['root']).resolve() == run_dir),
                   dict(root=str(run_dir), label=run_dir.name))

    def points_for(entries):
        points = []
        for dataset in entries:
            folder = Path(dataset['root'])
            status = folder/'STEP5_queue_status.json'
            if not status.is_file():
                continue
            for record in json.loads(status.read_text(encoding='utf-8-sig')):
                if not record.get('curve_available') and not record.get('pm_point'):
                    continue
                points.append(dict(source='GMNIA', folder=str(folder), model=record['model'],
                                   dataset_label=dataset.get('label',folder.name)))
        return points

    def publish(destination, points, in_place, stem):
        files = collect_response_pairs(str(destination), points, in_place=in_place)
        manifest = destination/'PM_model_response_plots.json'
        if not manifest.is_file():
            return
        entries = json.loads(manifest.read_text(encoding='utf-8'))['models']
        files += write_response_path_overview(str(destination), entries, stem=stem)
        fields = ('dataset_label','model','outcome','frames','selected_frame','selection_kind',
                  'selection_response','bending_axis_deg','pm_angle_deg','model_case','fy_MPa','source_csv','source_odb','workbook')
        with (destination/'PM_response_report_index.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(entries)
        print('RESPONSE REPORTS: %d finished histories -> %s (PNG/SVG/XLSX; path overview)' %
              (len(entries), destination), flush=True)

    publish(run_dir, points_for([current]), True, 'STEP5_response_paths')
    if context.get('output_dir'):
        publish(Path(context['output_dir']), points_for(datasets), False, '04_PM_response_paths')


def normalize_capacity_lines(values, axes):
    """Accept historic numeric yield lines and explicitly labelled plotting benchmarks."""
    result = []
    for item in values:
        value = float(item['value'] if isinstance(item, dict) else item)
        if not math.isfinite(value):
            raise ValueError('Capacity line value must be finite')
        label = str(item.get('label') or axes['capacity_label'] % value) if isinstance(item, dict) else axes['capacity_label'] % value
        result.append(dict(value=value, label=label))
    return result


def with_report_reference(info, directory):
    """Overlay plotting metadata in memory; never change the section/load normalization."""
    if not directory:
        from abaqus_report_reference import step5_reference
        reference = step5_reference(info)
        return dict(info,report_reference=reference) if reference else info
    from abaqus_report_reference import read_reference
    return dict(info, report_reference=read_reference(directory, fy=(info.get('settings') or {}).get('fy')))


def capacity_lines(info):
    """Plot gross AsFy or SFy/ZFy from model geometry or an explicit legacy report."""
    fy = float((info.get('settings') or {}).get('fy', 0) or 0)
    if response_kind(info) == 'combined':
        return [1.0]
    reference = info.get('report_reference')
    if reference:
        from abaqus_report_reference import reference_origin
        origin = reference_origin(reference)
        if response_kind(info) == 'bending':
            from abaqus_report_reference import moment_reference
            angle = info.get('bending_axis_deg')
            if angle is None:
                angle = (info.get('load_case') or {}).get('neutral_axis_angle_deg', 0.)
            selected = moment_reference(reference, angle)
            return [dict(value=float(selected[key]), label='%s (%s %s) = %.3f kN m' %
                         (name, origin, selected['axis_label'], float(selected[key])))
                    for name, key in (('SFy', 'SFy_kNm'), ('ZFy', 'ZFy_kNm'))]
        value = float(reference['AsFy_kN'])
        return [dict(value=value, label='AsFy (%s built-up) = %.3f kN' % (origin,value))]
    if response_kind(info) == 'bending':
        my = float(info.get('first_yield_moment_Nmm') or 0)
        return [my/1e6] if my > 0 else []
    area = float(info.get('end_area_mm2', 0) or 0)
    return [area*fy/1000] if area > 0 and fy > 0 else []


def extract(args):
    from odbAccess import openOdb

    locked_at_start = os.path.exists(os.path.splitext(args.odb)[0]+'.lck')
    if locked_at_start and not args.allow_partial:
        raise ValueError('ODB is locked; use --allow-partial for a read-only snapshot')
    model_name = args.model or os.path.splitext(os.path.basename(args.odb))[0]
    info, metadata_source = load_metadata(args.odb, model_name, getattr(args, 'cae', None))
    info = with_report_reference(info, getattr(args, 'cufsm_report_dir', None))

    odb = openOdb(path=args.odb, readOnly=True)
    try:
        diagnostic = getattr(odb, 'diagnosticData', None)
        odb_job_status = str(getattr(diagnostic, 'jobStatus', '') or '') or None
        rows, lpf_alignment = curve_rows(odb, args.step, info, allow_partial=args.allow_partial)
    finally:
        odb.close()

    live = locked_at_start or os.path.exists(os.path.splitext(args.odb)[0]+'.lck')
    lpf_alignment['partial'] = bool(lpf_alignment.get('partial') or live)
    lpf_alignment['live_odb_lock_observed'] = live
    os.makedirs(args.output_dir, exist_ok=True)
    axes = response_axes(info)
    stem = os.path.splitext(os.path.basename(args.odb))[0]+axes['stem']+('_partial' if lpf_alignment['partial'] else '')
    csv_path = os.path.join(args.output_dir, stem+'.csv')
    json_path = os.path.join(args.output_dir, stem+'.json')
    svg_path = os.path.join(args.output_dir, stem+'.svg')
    png_path = os.path.join(args.output_dir, stem+'.png')
    fields = list(rows[0].keys()) if axes['kind'] in ('bending', 'combined') else [
        'frame', 'frame_value', 'lpf', 'u3_bottom_mm', 'u3_top_mm', 'shortening_mm',
        'force_N', 'force_kN', 'nominal_stress_MPa']
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    peak = peak_summary(rows)
    odb_stat = os.stat(args.odb)
    summary = dict(
        odb_job_status=odb_job_status, source_odb_size=odb_stat.st_size,
        source_odb_mtime_ns=odb_stat.st_mtime_ns,
        source_cae=args.cae, metadata_source=metadata_source, source_odb=args.odb, model=model_name, step=args.step,
        convention=info.get('force_formula'),
        shortening_convention=info.get('shortening_formula'),
        reference_force_N_per_end=info.get('reference_force_N_per_end'),
        end_area_mm2=info.get('end_area_mm2'),
        frames=len(rows), partial=bool(lpf_alignment.get('partial')),
        lpf_source='Automatic ODB history output LPF',
        lpf_alignment=lpf_alignment, peak=peak,
        peak_definition='maximum recorded load in this snapshot; final capacity requires completed validation',
        csv=csv_path, svg=svg_path, png=png_path)
    if axes['kind'] == 'bending':
        summary.update(convention=info.get('moment_formula'), rotation_convention=info.get('rotation_formula'),
                       load_case=info.get('load_case'), bending_axis_deg=info.get('bending_axis_deg'),
                       reference_moment_Nmm=info.get('reference_moment_Nmm'),
                       section_for_PM=info.get('section_for_PM'))
        summary.pop('shortening_convention'); summary.pop('reference_force_N_per_end')
    if axes['kind'] == 'combined':
        summary.update(convention=info.get('pm_formula'), load_case=info.get('load_case'),
                       bending_axis_deg=info.get('bending_axis_deg'), pm_angle_deg=info.get('pm_angle_deg'),
                       reference_axial_force_N=info.get('reference_axial_force_N'),
                       reference_moment_Nmm=info.get('reference_moment_Nmm'),
                       section_for_PM=info.get('section_for_PM'))
        summary.pop('shortening_convention'); summary.pop('reference_force_N_per_end')
    if info.get('report_reference'):
        summary['report_reference'] = info['report_reference']
    with open(json_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    title=model_name+' - '+axes['title']+(' (PARTIAL SNAPSHOT)' if lpf_alignment['partial'] else '')
    yield_lines=capacity_lines(info)
    write_svg(svg_path, rows, peak, title, axes,
              yield_lines=yield_lines if info.get('report_reference') else None)
    if not getattr(args,'skip_png',False):
        from abaqus_step5_queue import plot_png
        plot_png(png_path, [(model_name,[(r[axes['x']],r[axes['y']]) for r in rows])],yield_lines,title,axes=axes)

    print('FORCE-DISPLACEMENT EXTRACTED')
    print('Status : %s' % ('PARTIAL / IN-PROGRESS' if lpf_alignment.get('partial') else 'COMPLETE'))
    print('Frames : %d' % len(rows))
    print('LPF map: %s; frames=%d history=%d; max |frameValue-historyX|=%.6g' %
          (lpf_alignment['mode'], lpf_alignment['frame_count'],
           lpf_alignment['history_count'],
           lpf_alignment['max_abs_frameValue_minus_historyX']))
    if lpf_alignment.get('partial'):
        print('Partial: dropped %d field frame(s) and %d history sample(s) outside the synchronized prefix' %
              (lpf_alignment.get('dropped_field_frames', 0),
               lpf_alignment.get('dropped_history_samples', 0)))
    if axes['kind'] == 'combined':
        print('Peak   : P=%.6g kN, M=%.6g kN m, P/Py+M/My=%.4f (LPF %.6g)' %
              (peak['force_kN'], peak['moment_kNm'], peak['utilization'], peak['lpf']))
    elif axes['kind'] == 'bending':
        print('Peak   : %.6g kN m at %.6g mrad (LPF %.6g)' %
              (peak['moment_kNm'], peak['rotation_mrad'], peak['lpf']))
    else:
        print('Peak   : %.6g kN at %.6g mm (LPF %.6g)' %
              (peak['force_kN'], peak['shortening_mm'], peak['lpf']))
    print('CSV    : '+csv_path)
    print('SVG    : '+svg_path)
    if not getattr(args,'skip_png',False):
        print('PNG    : '+png_path)
    print('JSON   : '+json_path)
    return summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--publish-responses' in argv:
        parser = argparse.ArgumentParser(description='Publish saved responses without opening ODBs')
        parser.add_argument('--publish-responses', action='store_true', required=True)
        parser.add_argument('--run-dir', required=True)
        return publish_queue_responses(parser.parse_args(argv).run_dir)
    return extract(parse_arguments(argv))


if __name__ == '__main__':
    main()
