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
No analysis is submitted; the CAE is closed without saving and the ODB is read-only.
"""
import argparse
import builtins as python_builtins
import csv
import json
import math
import os
import sys

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
                 peak_text='Peak %.3f kN m @ %.4g mrad'))


def response_kind(info):
    """'bending' for STEP5 models built from a bending run, else 'compression'."""
    return 'bending' if ((info or {}).get('load_case') or {}).get('type') == 'bending' else 'compression'


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
    p.add_argument('--cae', required=True, help='STEP5_GMNIA_FY*.cae')
    p.add_argument('--odb', required=True, help='Solved STEP5_*.odb')
    p.add_argument('--model', help='STEP5 model name; default is ODB basename')
    p.add_argument('--step', default='GMNIA')
    p.add_argument('--output-dir', help='Default: directory containing the ODB')
    p.add_argument('--allow-partial', action='store_true',
                   help='Allow extraction from an in-progress ODB by using only the common complete frame/LPF prefix')
    args = p.parse_args(argv)
    args.cae = os.path.abspath(os.path.expanduser(args.cae))
    args.odb = os.path.abspath(os.path.expanduser(args.odb))
    args.output_dir = os.path.abspath(os.path.expanduser(
        args.output_dir or os.path.dirname(args.odb)))
    if not os.path.isfile(args.cae):
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
                data = [(float(x), float(y)) for x, y in output.data]
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
    bending = response_kind(info) == 'bending'
    if bending:
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
    if not bending:
        pref = float(info['reference_force_N_per_end'])
        if not math.isfinite(pref) or pref <= 0:
            raise ValueError('Invalid reference_force_N_per_end in STEP5 metadata')
    reference_stress = float(info.get('settings', {}).get('reference_stress', float('nan')))

    lpf_data = lpf_history(step)
    frames = list(step.frames)
    partial = False
    dropped_field_frames = 0
    dropped_history_samples = 0
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
    rows = []
    for index, frame in enumerate(frames):
        if 'U' not in frame.fieldOutputs:
            raise ValueError('Frame %d has no U field output' % index)
        frame_value = float(frame.frameValue)
        lpf = aligned_lpf[index]
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


def write_svg(path, rows, peak, title, axes=None):
    axes = axes or RESPONSE_AXES['compression']
    width, height = 1200, 800
    left, right, top, bottom = 115, 45, 75, 105
    xs = [row[axes['x']] for row in rows]
    ys = [row[axes['y']] for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(0.0, min(ys)), max(ys)
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


def capacity_lines(info):
    """Reference lines in plot units: AsFy (kN) for compression, My (kN m) for bending."""
    fy = float((info.get('settings') or {}).get('fy', 0) or 0)
    if response_kind(info) == 'bending':
        my = float(info.get('first_yield_moment_Nmm') or 0)
        return [my/1e6] if my > 0 else []
    area = float(info.get('end_area_mm2', 0) or 0)
    return [area*fy/1000] if area > 0 and fy > 0 else []


def extract(args):
    import caeModules
    from abaqus import openMdb
    from odbAccess import openOdb

    locked_at_start = os.path.exists(os.path.splitext(args.odb)[0]+'.lck')
    if locked_at_start and not args.allow_partial:
        raise ValueError('ODB is locked; use --allow-partial for a read-only snapshot')
    model_name = args.model or os.path.splitext(os.path.basename(args.odb))[0]
    database = openMdb(pathName=args.cae)
    try:
        if model_name not in database.models.keys():
            raise ValueError('STEP5 model not found in CAE: '+model_name)
        info = pipeline_contract.parse_prefixed_json(
            database.models[model_name].description, STEP5_PREFIX)
    finally:
        database.close()

    odb = openOdb(path=args.odb, readOnly=True)
    try:
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
    fields = list(rows[0].keys()) if axes['kind'] == 'bending' else [
        'frame', 'frame_value', 'lpf', 'u3_bottom_mm', 'u3_top_mm', 'shortening_mm',
        'force_N', 'force_kN', 'nominal_stress_MPa']
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    peak = peak_summary(rows)
    summary = dict(
        source_cae=args.cae, source_odb=args.odb, model=model_name, step=args.step,
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
    with open(json_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    title=model_name+' - '+axes['title']+(' (PARTIAL SNAPSHOT)' if lpf_alignment['partial'] else '')
    write_svg(svg_path, rows, peak, title, axes)
    if not getattr(args,'skip_png',False):
        from abaqus_step5_queue import plot_png
        yield_lines=capacity_lines(info)
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
    if axes['kind'] == 'bending':
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
    return extract(parse_arguments(argv))


if __name__ == '__main__':
    main()
