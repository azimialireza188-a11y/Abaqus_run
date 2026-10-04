# -*- coding: utf-8 -*-
"""Extract the exact STEP5 Riks force-shortening curve from CAE + ODB.

Run with Abaqus/CAE because the STEP5 model description in the CAE contains the
reference force and end-area weights used by abaqus_step5_gmnia.py.

Example:
  abaqus cae noGUI=abaqus_step5_force_displacement.py -- ^
    --cae "Step5_GMNIA_FY240.cae" ^
    --odb "STEP5_D_FY240.odb"

Outputs: CSV, JSON summary and a dependency-free SVG plot.
No analysis is submitted and the source CAE/ODB are opened read-only.
"""
import argparse
import builtins as python_builtins
import csv
import json
import math
import os
import sys

import abaqus_pipeline_contract as pipeline_contract

STEP5_PREFIX = 'STEP5_GMNIA '


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
    args = p.parse_args(argv)
    args.cae = os.path.abspath(os.path.expanduser(args.cae))
    args.odb = os.path.abspath(os.path.expanduser(args.odb))
    args.output_dir = os.path.abspath(os.path.expanduser(
        args.output_dir or os.path.dirname(args.odb)))
    if not os.path.isfile(args.cae):
        p.error('CAE not found: '+args.cae)
    if not os.path.isfile(args.odb):
        p.error('ODB not found: '+args.odb)
    if os.path.exists(os.path.splitext(args.odb)[0]+'.lck'):
        p.error('ODB is locked; wait for the job to finish')
    return args


def _precision_vector(value):
    if str(getattr(value, 'precision', '')) == 'DOUBLE_PRECISION':
        return value.dataDouble
    return value.data


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


def weighted_u3(field, region, weights):
    values = field.getSubset(region=region).values
    found = {}
    for value in values:
        instance = getattr(value, 'instance', None)
        if instance is None:
            continue
        key = (str(instance.name).upper(), int(value.nodeLabel))
        if key not in weights:
            continue
        data = _precision_vector(value)
        found[key] = float(data[2])
    missing = sorted(set(weights)-set(found))
    if missing:
        preview = ', '.join('%s:%d' % item for item in missing[:8])
        raise ValueError('U3 missing for %d weighted end nodes; first: %s' %
                         (len(missing), preview))
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
        offset = 0
        mode = 'same_count_by_sequence'
        pairs = [(history[i][0], history[i][1]) for i in range(nf)]
    elif nh == nf-1:
        first_value = float(frames[0].frameValue)
        if abs(first_value) > 1e-10*max(1.0, abs(first_value)):
            raise ValueError('LPF history has one fewer sample but the first field frame is not the initial frame')
        offset = -1
        mode = 'synthesized_initial_zero_then_sequence'
        pairs = [(first_value, 0.0)] + [(history[i][0], history[i][1]) for i in range(nh)]
    elif nh == nf+1:
        hx, hy = history[0]
        tolerance = 1e-10*max(1.0, abs(hx), abs(hy))
        if abs(hx) > tolerance or abs(hy) > tolerance:
            raise ValueError('LPF history has one extra sample but it is not an initial zero sample')
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


def curve_rows(odb, step_name, info):
    if step_name not in odb.steps:
        raise ValueError('ODB step not found: '+step_name)
    step = odb.steps[step_name]
    if not step.frames:
        raise ValueError('ODB step has no frames: '+step_name)
    weights = info.get('end_area_weights')
    if not isinstance(weights, list) or len(weights) != 2:
        raise ValueError('STEP5 metadata lacks the two end_area_weights tables')
    bottom_weights, top_weights = weight_map(weights[0]), weight_map(weights[1])
    node_sets = odb.rootAssembly.nodeSets
    if 'STEP5_BOTTOM' not in node_sets.keys() or 'STEP5_TOP' not in node_sets.keys():
        raise ValueError('ODB lacks STEP5_BOTTOM/STEP5_TOP node sets')
    bottom = node_sets['STEP5_BOTTOM']
    top = node_sets['STEP5_TOP']
    pref = float(info['reference_force_N_per_end'])
    if not math.isfinite(pref) or pref <= 0:
        raise ValueError('Invalid reference_force_N_per_end in STEP5 metadata')
    reference_stress = float(info.get('settings', {}).get('reference_stress', float('nan')))

    lpf_data = lpf_history(step)
    aligned_lpf, lpf_alignment = align_lpf_history(step.frames, lpf_data)
    rows = []
    for index, frame in enumerate(step.frames):
        if 'U' not in frame.fieldOutputs:
            raise ValueError('Frame %d has no U field output' % index)
        frame_value = float(frame.frameValue)
        lpf = aligned_lpf[index]
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


def write_svg(path, rows, peak, title):
    width, height = 1200, 800
    left, right, top, bottom = 115, 45, 75, 105
    xs = [row['shortening_mm'] for row in rows]
    ys = [row['force_kN'] for row in rows]
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
        (sx(peak['shortening_mm']), sy(peak['force_kN'])),
        '<text x="%.2f" y="%.2f" font-family="Arial" font-size="16">Peak %.3f kN @ %.4g mm</text>' %
        (sx(peak['shortening_mm'])+12, sy(peak['force_kN'])-12,
         peak['force_kN'], peak['shortening_mm']),
        '<text x="%d" y="%d" text-anchor="middle" font-family="Arial" font-size="19">Axial shortening, delta (mm)</text>' %
        (left+plot_w//2, height-34),
        '<text transform="translate(30,%d) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="19">Axial load, P (kN)</text>' %
        (top+plot_h//2),
        '</svg>'
    ])
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))


def extract(args):
    import caeModules
    from abaqus import openMdb
    from odbAccess import openOdb

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
        rows, lpf_alignment = curve_rows(odb, args.step, info)
    finally:
        odb.close()

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.odb))[0]+'_force_displacement'
    csv_path = os.path.join(args.output_dir, stem+'.csv')
    json_path = os.path.join(args.output_dir, stem+'.json')
    svg_path = os.path.join(args.output_dir, stem+'.svg')
    fields = ['frame', 'frame_value', 'lpf', 'u3_bottom_mm', 'u3_top_mm', 'shortening_mm',
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
        frames=len(rows), lpf_source='Automatic ODB history output LPF',
        lpf_alignment=lpf_alignment, peak=peak,
        csv=csv_path, svg=svg_path)
    with open(json_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    write_svg(svg_path, rows, peak, model_name+' — Riks force-shortening')

    print('FORCE-DISPLACEMENT EXTRACTED')
    print('Frames : %d' % len(rows))
    print('LPF map: %s; frames=%d history=%d; max |frameValue-historyX|=%.6g' %
          (lpf_alignment['mode'], lpf_alignment['frame_count'],
           lpf_alignment['history_count'],
           lpf_alignment['max_abs_frameValue_minus_historyX']))
    print('Peak   : %.6g kN at %.6g mm (LPF %.6g)' %
          (peak['force_kN'], peak['shortening_mm'], peak['lpf']))
    print('CSV    : '+csv_path)
    print('SVG    : '+svg_path)
    print('JSON   : '+json_path)
    return summary


def main(argv=None):
    return extract(parse_arguments(argv))


if __name__ == '__main__':
    main()
