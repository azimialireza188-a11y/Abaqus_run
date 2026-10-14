# -*- coding: utf-8 -*-
"""Run/repair existing Step5 inputs, monitor 70% drop, extract and compare curves.

abaqus cae noGUI=abaqus_step5_queue_nogui.py -- --run-dir DIR --repair-only --cpus 4
Then run the regenerated run_step5_queue_<ratio>_pct.bat. No model rebuild needed.
--cpus N|auto: CPUs of every Abaqus job of the queue (auto = all physical cores; a
request above the physical cores of this machine uses all of them). Options are
written '--name value': the Abaqus launcher rejects '--name=value'.
Monitor mode needs no Fortran compiler; it stops at first OBSERVED crossing.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from html import escape
from types import SimpleNamespace

# CAE noGUI does not always add the script directory to its import path.
# Resolve without __file__, before sibling imports and directory changes.
SCRIPT_PATH = os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename))
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import abaqus_step5_force_displacement_core as fd
import abaqus_pipeline_contract as contract
import runtime_resources as resources
from abaqus_step5_progress import report, elapsed_text


def crossing(values, ratio):
    if not 0 < ratio < 1:
        raise ValueError('Require 0 < ratio < 1')
    peak, peak_index = 0.0, -1
    for i, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError('Nonfinite LPF')
        if value > peak:
            peak, peak_index = value, i
        elif peak > 0 and i > peak_index and value <= ratio*peak:
            return dict(index=i, lpf=value, peak_lpf=peak, peak_index=peak_index,
                        ratio=value/peak)
    return None


def job_command(name, cpus, gpus, mode, abaqus_cmd='abaqus'):
    if not re.fullmatch(r'(STEP5|PM)_[A-Za-z0-9_]+', name):
        raise ValueError('Invalid STEP5 job name: '+name)
    cmd = [abaqus_cmd, 'job='+name, 'input='+name+'.inp', 'cpus='+str(cpus),
           'memory=100%', 'standard_parallel=all', 'interactive']
    cmd.append('gpus='+str(gpus))  # zero must override a GPU-built abaqus_v6.env on CPU-only hosts
    if mode == 'urdfil':
        cmd.append('user=step5_postpeak_stop.for')
    return cmd


def outcome(result, reached):
    """A known solver error wins. Otherwise a post-peak crossing found in the synchronized
    LPF/field prefix is a complete capacity result, even when trailing frames were dropped
    (a controlled terminate can leave one unsynchronized increment) or when the exit status
    of an earlier session is unknown: Pu and the crossing are both inside consistent data."""
    code = result.get('solver_exit_code')
    state = result.get('analysis_state')
    if state == 'FAILED':
        return 'SOLVER_ERROR'
    if code is not None and code != 0 and not (result.get('termination_requested') and reached):
        return 'SOLVER_ERROR'
    if state == 'RUNNING' or (state in ('INCOMPLETE', 'TERMINATED') and
                             not (result.get('termination_requested') and reached)):
        return 'INCOMPLETE_ANALYSIS'
    if reached:
        return 'CONTROLLED_TERMINATION' if result.get('termination_requested') else 'CRITERION_REACHED'
    if result.get('partial'):
        return 'PARTIAL_RESULT'
    if state == 'UNKNOWN' or (code is None and not result.get('analysis_completed')):
        return 'SOLVER_STATUS_UNKNOWN'
    if result.get('analysis_completed') is False or result.get('termination_requested'):
        return 'INCOMPLETE_ANALYSIS'
    return 'CRITERION_NOT_REACHED'


def completion_details(directory, name, odb_status=None):
    """Combine terminal evidence of this job only; a missing footer is not an abort.

    Explicit errors/incomplete endings override positive markers. Native monitor
    termination is distinguished from numerical failure by its external-request
    diagnostic, and must still be justified by this job's recorded request/crossing.
    """
    evidence, diagnostics = [], []
    for suffix in ('.sta', '.msg', '.log', '_queue_solver.log'):
        path = os.path.join(directory, name+suffix)
        if not os.path.isfile(path):
            continue
        tag = name+suffix
        try:
            with open(path, 'rb') as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell()-131072))
                text = stream.read().decode('utf-8', errors='replace').upper()
        except OSError as error:
            diagnostics.append(tag+': unreadable during completion check: '+str(error))
            continue
        if re.search(r'^\s*THE ANALYSIS HAS (?:BEEN )?COMPLETED(?: SUCCESSFULLY)?\s*$', text, re.M):
            evidence.append((tag, 'COMPLETED'))
        if re.search(r'\bABAQUS JOB\s+'+re.escape(name.upper())+r'\s+COMPLETED\s*$', text, re.M):
            evidence.append((tag, 'COMPLETED'))
        external = re.search(r'PROCESS TERMINATED BY EXTERNAL REQUEST|TERMINATE REQUEST RECEIVED', text)
        if external or re.search(r'SENT TERMINATE MESSAGE TO ABAQUS JOB\s+'+re.escape(name.upper())+r'\s*$',text,re.M):
            evidence.append((tag, 'TERMINATED'))
        if re.search(r'THE ANALYSIS HAS NOT BEEN COMPLETED|\bANALYSIS ABORTED\b', text):
            evidence.append((tag, 'INCOMPLETE'))
        # A monitor SIGTERM is reported as ***ERROR in Standard; it is not a
        # numerical convergence error. Other explicit error blocks remain fatal.
        errors = re.findall(r'^\s*\*{2,3}ERROR:\s*([^\n]*(?:\n[^\S\n]*[^\n]+){0,2})', text, re.M)
        fatal = [error for error in errors if not re.search(
            r'TERMINATED BY (?:EXTERNAL REQUEST|THE USER)|TERMINATE REQUEST', error)]
        if fatal:
            evidence.append((tag, 'FAILED'))
            diagnostics.append(tag+': '+fatal[0][:500].replace('\n',' '))
        if re.search(r'ABAQUS/ANALYSIS EXITED WITH ERRORS|ABAQUS JOB\s+'+re.escape(name.upper())+r'\s+ABORTED\b',text):
            evidence.append((tag, 'TERMINATED' if external else 'INCOMPLETE'))
    status = str(odb_status or '').upper()
    if status in ('JOB_STATUS_COMPLETED_SUCCESSFULLY', 'COMPLETED_SUCCESSFULLY'):
        evidence.append(('ODB', 'COMPLETED'))
    elif status == 'JOB_STATUS_COMPLETED_WITH_ERRORS':
        evidence.append(('ODB', 'FAILED'))
    elif status in ('JOB_STATUS_ABORTED','ABORTED'):
        evidence.append(('ODB', 'INCOMPLETE'))
    elif status in ('JOB_STATUS_TERMINATED','TERMINATED'):
        evidence.append(('ODB', 'TERMINATED'))
    elif status in ('JOB_STATUS_RUNNING','RUNNING','JOB_STATUS_IN_PROGRESS'):
        evidence.append(('ODB', 'RUNNING'))
    locked = os.path.exists(os.path.join(directory, name+'.lck'))
    if locked:
        evidence.append(('LOCK', 'RUNNING'))
    states = {state for _, state in evidence}
    # A live lock wins. A nonterminal ODB header alone must not veto later,
    # consistent completion footers written by the solver/wrap-up process.
    state = 'RUNNING' if locked else next((s for s in
        ('FAILED','TERMINATED','INCOMPLETE','COMPLETED','RUNNING') if s in states),'UNKNOWN')
    return dict(analysis_state=state, analysis_completed=state=='COMPLETED',
                completion_sources=[source+':'+state for source,state in evidence],
                completion_diagnostics=diagnostics)


def analysis_finished(directory, name):
    return completion_details(directory, name)['analysis_completed']


def finished_result(result):
    """Execution may finish without a verified capacity; partial/failed solves may not."""
    reached = bool(result.get('criterion_reached'))
    if outcome(result, reached) != result.get('outcome'):
        return False
    return (reached and result['outcome'] in ('CRITERION_REACHED', 'CONTROLLED_TERMINATION') or
            result.get('outcome') == 'CRITERION_NOT_REACHED' and
            result.get('analysis_completed') is True and not result.get('partial') and
            not result.get('termination_requested'))


def capacity_status(result):
    """Keep observed maxima, but never publish unverified maxima as Pu/Mu or P-M capacity."""
    verified = (bool(result.get('criterion_reached')) and
                result.get('outcome') in ('CRITERION_REACHED', 'CONTROLLED_TERMINATION'))
    result['capacity_verified'] = verified
    result['peak_interpretation'] = 'VERIFIED_POSTPEAK' if verified else 'OBSERVED_MAXIMUM_ONLY'
    if not verified:
        for key in ('Pu_over_Py', 'Mu_over_My', 'Mu_over_Mp'):
            result.pop(key, None)
        result['pm_point'] = None
    return result


def curve_label(name, result):
    label = name.replace('STEP5_', '')
    return label if result.get('capacity_verified') else label+' (capacity unverified: '+result.get('outcome','UNKNOWN')+')'


def completed_curve(directory, name, info, ratio=.70):
    """Reuse a verified, finished CSV in solver order; never open/re-extract its ODB.

    None means no successful saved result. Occupied unsuccessful solver destinations
    remain protected by the submission guard; they are never silently resubmitted.
    """
    from pathlib import Path
    directory=Path(directory)
    status=directory/'STEP5_queue_status.json'
    if not status.is_file():
        return None
    with status.open(encoding='utf-8-sig') as stream:
        records=json.load(stream)
    if not isinstance(records,list) or any(not isinstance(r,dict) for r in records):
        raise ValueError('Invalid queue status: '+str(status))
    rows=[r for r in records if r.get('model')==name]
    if len(rows)>1:
        raise ValueError('Duplicate saved queue result: '+name)
    if not rows:
        return None
    result=dict(rows[0])
    # A plot exception recorded AFTER a valid outcome must not force another
    # solve. Revalidate the saved numerical curve, then repair plots from CSV.
    if result.get('outcome') not in ('CRITERION_REACHED','CONTROLLED_TERMINATION','CRITERION_NOT_REACHED',
                                   'INCOMPLETE_ANALYSIS','SOLVER_STATUS_UNKNOWN'):
        return None
    axes=fd.response_axes(info)
    stem=directory/(name+axes['stem']+('_partial' if result.get('partial') else ''))
    csv_path=stem.with_suffix('.csv')
    odb=directory/(name+'.odb')
    if (directory/(name+'.lck')).exists():
        raise ValueError('Job is locked/running: '+name)
    for path in (csv_path,stem.with_suffix('.json'),odb):
        if not path.is_file():
            raise ValueError('Missing completed result artifact: '+str(path))
    if csv_path.stat().st_mtime_ns < odb.stat().st_mtime_ns:
        raise ValueError('Stale curve CSV (ODB is newer): '+str(csv_path))
    with stem.with_suffix('.json').open(encoding='utf-8-sig') as stream:
        summary=json.load(stream)
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:
        data=list(csv.DictReader(stream))
    if (summary.get('model')!=name or summary.get('step')!='GMNIA' or
            summary.get('frames')!=len(data) or len(data)<2 or
            bool(summary.get('partial')) != bool(result.get('partial'))):
        raise ValueError('Curve summary/CSV mismatch: '+str(csv_path))
    points=[(float(r[axes['x']]),float(r[axes['y']])) for r in data]
    lpf=[float(r['lpf']) for r in data]
    if not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('Nonfinite completed curve: '+str(csv_path))
    if 'frame' in data[0]:
        frames=[int(r['frame']) for r in data]
        if any(b<=a for a,b in zip(frames,frames[1:])):
            raise ValueError('Curve frames are not in solver order: '+str(csv_path))
    if axes['kind']=='compression':
        scale=float(info['reference_force_N_per_end'])/1000.
        if not all(math.isclose(y,f*scale,rel_tol=1e-7,abs_tol=1e-8) for (_,y),f in zip(points,lpf)):
            raise ValueError('Curve force differs from LPF times reference force: '+str(csv_path))
    reached=bool(crossing(lpf,ratio))
    if reached != bool(result.get('criterion_reached')):
        raise ValueError('Saved result does not verify the post-peak criterion: '+name)
    stat = odb.stat()
    cached_status = (summary.get('odb_job_status') if
                     summary.get('source_odb_size')==stat.st_size and
                     summary.get('source_odb_mtime_ns')==stat.st_mtime_ns else None)
    result.update(completion_details(str(directory), name, cached_status))
    reassessed = outcome(result,reached)
    result['completion_reassessed'] = reassessed != result['outcome']
    if reassessed != result['outcome']:
        result['previous_outcome'] = result['outcome']
        result['outcome'] = reassessed
    if not finished_result(result):
        return None
    ykey=axes['y']
    if ykey in result and not math.isclose(float(result[ykey]),max(p[1] for p in points),rel_tol=1e-7,abs_tol=1e-8):
        raise ValueError('Saved peak differs from curve: '+name)
    return capacity_status(result),points


def launch(cmd, **kwargs):
    # Abaqus on Windows is a .bat launcher; quote arguments before cmd.exe.
    return subprocess.Popen(subprocess.list2cmdline(cmd) if os.name == 'nt' else cmd,
                            shell=os.name == 'nt', **kwargs)


def plot_png(path, curves, yield_lines, title, axes=None, styles=None):
    axes = axes or fd.RESPONSE_AXES['compression']
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        # Abaqus/CAE supplies its own plotting engine, so pip is not required.
        import visualization
        from abaqus import session
        from abaqusConstants import PNG, OFF
        key = 'STEP5_QUEUE_PLOT'
        if key in session.xyPlots:
            del session.xyPlots[key]
        plot = session.XYPlot(name=key)
        chart = plot.charts[list(plot.charts.keys())[0]]
        abq_curves, names = [], []
        for i, (label, points) in enumerate(curves):
            name = 'STEP5_QUEUE_CURVE_%d' % i
            if name in session.xyDataObjects:
                del session.xyDataObjects[name]
            xy = session.XYData(name=name, data=tuple(points),
                                xValuesLabel=axes['queue_x_label'],
                                yValuesLabel=axes['queue_y_short'], legendLabel=label)
            abq_curves.append(session.Curve(xyData=xy)); names.append(name)
        xs = [x for _, points in curves for x, _ in points]
        for i, value in enumerate(yield_lines):
            name = 'STEP5_QUEUE_YIELD_%d' % i
            if name in session.xyDataObjects:
                del session.xyDataObjects[name]
            xy = session.XYData(name=name, data=((min(xs),value),(max(xs),value)),
                                legendLabel=axes['capacity_label'] % value,
                                xValuesLabel=axes['queue_x_label'], yValuesLabel=axes['queue_y_short'])
            abq_curves.append(session.Curve(xyData=xy)); names.append(name)
        chart.setValues(curvesToPlot=tuple(abq_curves))
        chart.autoColor(lines=True, symbols=True)
        plot.title.setValues(text=title)
        viewport = session.viewports[list(session.viewports.keys())[0]]
        viewport.setValues(displayedObject=plot)
        session.printOptions.setValues(vpDecorations=OFF)
        session.pngOptions.setValues(imageSize=(1600,1000))
        session.printToFile(fileName=os.path.splitext(path)[0], format=PNG,
                            canvasObjects=(viewport,))
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    for i,(label, points) in enumerate(curves):
        style = styles[i] if styles is not None else {}
        ax.plot([p[0] for p in points], [p[1] for p in points], label=label, lw=1.6,
                color=style.get('color'), ls='--' if style.get('dash') else '-')
    for y in yield_lines:
        ax.axhline(y, ls='--', label=axes['capacity_label'] % y)
    ax.set(xlabel=axes['x_label'], ylabel=axes['queue_y_label'], title=title)
    ax.grid(alpha=.25)
    if styles is None:
        ax.legend(loc='best',fontsize=9)
    else:
        # Batch comparisons can have many families: keep the legend off the data.
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300,bbox_inches='tight'); plt.close(fig)


def plot_svg(path, curves, yield_lines, title, axes=None, styles=None):
    axes = axes or fd.RESPONSE_AXES['compression']
    xs = [x for _, points in curves for x, _ in points]
    ys = [y for _, points in curves for _, y in points]+list(yield_lines)+[0.0]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    dx, dy = max(1e-6, xmax-xmin), max(1e-6, ymax-ymin)
    xmin -= .04*dx; xmax += .04*dx; ymax += .08*dy
    sx = lambda x: 110+(x-xmin)/(xmax-xmin)*1050
    sy = lambda y: 670-(y-ymin)/(ymax-ymin)*580
    height=max(850,805+((len(curves)+2)//3)*22)
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="%d" viewBox="0 0 1400 %d">'%(height,height),
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial" font-size="16">',
             '<text x="700" y="40" text-anchor="middle">%s</text>' % escape(title)]
    for value in fd._nice_ticks(xmin, xmax):
        if xmin <= value <= xmax:
            x = sx(value)
            lines += ['<path d="M %.2f 90 V 670" stroke="#ddd"/>' % x,
                      '<text x="%.2f" y="700" text-anchor="middle">%.4g</text>' % (x,value)]
    for value in fd._nice_ticks(ymin, ymax):
        if ymin <= value <= ymax:
            y = sy(value)
            lines += ['<path d="M 110 %.2f H 1160" stroke="#ddd"/>' % y,
                      '<text x="95" y="%.2f" text-anchor="end">%.4g</text>' % (y,value)]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#222222']
    for i, (label, points) in enumerate(curves):
        style = styles[i] if styles is not None else {}
        color = style.get('color',colors[i % len(colors)])
        dash = ' stroke-dasharray="%s"'%escape(style['dash'],quote=True) if style.get('dash') else ''
        poly = ' '.join('%.2f,%.2f' % (sx(x),sy(y)) for x,y in points)
        lines.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"%s/>' % (poly,color,dash))
        lx,ly=115+(i%3)*420,775+(i//3)*22
        lines.append('<path d="M %d %d h 35" stroke="%s" stroke-width="2"%s/>'%(lx,ly-5,color,dash))
        lines.append('<text x="%d" y="%d" fill="%s">%s</text>' % (lx+42,ly,color,escape(label)))
    for value in yield_lines:
        lines.append('<path d="M 110 %.2f H 1160" stroke="#1676c3" stroke-dasharray="8 5"/>' % sy(value))
        lines.append('<text x="1170" y="%.2f">%s</text>' % (sy(value),escape(axes['capacity_label'].replace(' = ', ' ') % value)))
    lines += ['<path d="M 110 90 V 670 H 1160" fill="none" stroke="black"/>',
              '<text x="630" y="740" text-anchor="middle">%s</text>' % escape(axes['x_label']),
              '<text transform="translate(30,380) rotate(-90)" text-anchor="middle">%s</text>' % escape(axes['queue_y_label']),
              '</g></svg>']
    with open(path,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines))


PM_FIELDS = ['model','load_case','bending_axis_deg','pm_angle_deg','source_case','P_kN','M_kNm','Py_kN','My_kNm',
             'Mp_kNm','P_over_Py','M_over_My','M_over_Mp','a_L_mm','a_D_mm','a_G_mm','criterion_reached','outcome']


def write_comparison(directory, curves, yield_lines, summaries, axes=None):
    axes = axes or fd.RESPONSE_AXES['compression']
    if axes['kind'] == 'combined':
        fields = ['model','source_case','bending_axis_deg','pm_angle_deg','a_L_mm','a_D_mm','a_G_mm','force_kN',
                  'moment_kNm','P_over_Py','M_over_My','utilization','criterion_reached','outcome',
                  'solver_exit_code','error']
    elif axes['kind'] == 'bending':
        fields = ['model','a_L_mm','a_D_mm','a_G_mm','moment_kNm','My_kNm','Mu_over_My','Mp_kNm','Mu_over_Mp',
                  'rotation_mrad','bending_axis_deg','criterion_reached','outcome','solver_exit_code','error']
    else:
        fields = ['model','a_L_mm','a_D_mm','a_G_mm','force_kN','Py_kN','Pu_over_Py','shortening_mm',
                  'criterion_reached','outcome','solver_exit_code','error']
    fields += ['analysis_completed','analysis_state','partial','capacity_verified','peak_interpretation']
    with open(os.path.join(directory,'STEP5_peak_summary.csv'),'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        writer.writerows(summaries)
    # One P-M point per model (P = peak axial load for compression, M = peak moment for
    # bending): the files of all compression and bending folders combine into P-M diagrams.
    with open(os.path.join(directory,'STEP5_PM_points.csv'),'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=PM_FIELDS,extrasaction='ignore');writer.writeheader()
        for row in summaries:
            if row.get('pm_point'):
                writer.writerow(dict(row['pm_point'], model=row.get('model'),
                                     criterion_reached=row.get('criterion_reached'), outcome=row.get('outcome'),
                                     a_L_mm=row.get('a_L_mm'), a_D_mm=row.get('a_D_mm'), a_G_mm=row.get('a_G_mm')))
    if not curves:
        return
    stem = os.path.join(directory, axes['combined'])
    plot_svg(stem+'.svg',curves,yield_lines,axes['combined_title'],axes)
    plot_png(stem+'.png',curves,yield_lines,axes['combined_title'],axes)


def pm_point(info, peak):
    """Capacity point for P-M: (P, M) at the peak of this model with its section references."""
    section = info.get('section_for_PM') or {}
    fy = float((info.get('settings') or {}).get('fy', 0) or 0)
    area = float(section.get('area_mm2') or info.get('end_area_mm2') or 0)
    point = dict(load_case=fd.response_kind(info), Py_kN=area*fy/1000 if area and fy else None,
                 bending_axis_deg=info.get('bending_axis_deg'), pm_angle_deg=info.get('pm_angle_deg'),
                 source_case=(info.get('source_case') or
                              (((info.get('source_step4') or {}).get('case') or {}).get('label'))))
    my = (section.get('first_yield_moment_Nmm') or 0)/1e6 or None
    mp = (section.get('plastic_moment_Nmm') or 0)/1e6 or None
    if point['load_case'] == 'combined':
        point.update(P_kN=peak.get('force_kN'), M_kNm=peak.get('moment_kNm'), My_kNm=my, Mp_kNm=mp)
    elif point['load_case'] == 'bending':
        point.update(P_kN=0.0, M_kNm=peak.get('moment_kNm'),
                     My_kNm=(section.get('first_yield_moment_Nmm') or 0)/1e6 or None,
                     Mp_kNm=(section.get('plastic_moment_Nmm') or 0)/1e6 or None)
    else:
        point.update(P_kN=peak.get('force_kN'), M_kNm=0.0)
    for key, numerator, denominator in (('P_over_Py', 'P_kN', 'Py_kN'), ('M_over_My', 'M_kNm', 'My_kNm'),
                                        ('M_over_Mp', 'M_kNm', 'Mp_kNm')):
        if point.get(numerator) is not None and point.get(denominator):
            point[key] = float(point[numerator])/float(point[denominator])
    return point


def metadata(cae, names, run_dir=None):
    """STEP5 metadata per job: <job>_step5.json sidecars when all exist, else the CAE."""
    if run_dir:
        sidecars = {name: os.path.join(run_dir, name+fd.METADATA_SUFFIX) for name in names}
        if all(os.path.isfile(path) for path in sidecars.values()):
            result = {}
            for name, path in sidecars.items():
                with open(path, encoding='utf-8') as stream:
                    result[name] = json.load(stream)
            return result
    if not cae:
        raise ValueError('No <job>_step5.json metadata sidecars and no unique Step5_GMNIA_FY*.cae; use --cae')
    import caeModules
    from abaqus import openMdb
    db = openMdb(pathName=cae)
    try:
        return {name: contract.parse_prefixed_json(db.models[name].description,fd.STEP5_PREFIX)
                for name in names}
    finally:
        db.close()


def _printed_half_unit(token):
    """Half of the last printed unit, used to avoid early stop from STA rounding."""
    text = str(token).strip().upper()
    if 'E' in text:
        mantissa, exponent = text.split('E', 1)
    else:
        mantissa, exponent = text, '0'
    decimals = len(mantissa.split('.', 1)[1]) if '.' in mantissa else 0
    return 0.5 * (10.0 ** (int(exponent) - decimals))


def read_sta_lpf(path):
    """Read converged Static-Riks LPF values from the append-only .sta file."""
    records = []
    try:
        with open(path, 'r', errors='replace') as stream:
            for raw in stream:
                parts = raw.split()
                if len(parts) < 9 or str(parts[-1]).upper() != 'R':
                    continue
                try:
                    step = int(parts[0]); increment = int(parts[1])
                except (TypeError, ValueError):
                    continue
                attempt = str(parts[2]).upper()
                if not attempt.isdigit():
                    continue
                token = parts[6]
                try:
                    value = float(token)
                    half_unit = _printed_half_unit(token)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(value) and math.isfinite(half_unit) and half_unit >= 0:
                    records.append(dict(lpf=value, half_unit=half_unit, step=step,
                                        increment=increment, attempt=int(attempt)))
    except OSError:
        return []
    return records


def crossing_sta(records, ratio):
    """Conservative post-peak crossing using rounded .sta LPF values."""
    if not 0 < ratio < 1:
        raise ValueError('Require 0 < ratio < 1')
    peak_lower = 0.0
    peak_value = 0.0
    peak_index = -1
    for i, record in enumerate(records):
        value = float(record['lpf'])
        half = float(record.get('half_unit', 0.0))
        if not math.isfinite(value) or not math.isfinite(half):
            raise ValueError('Nonfinite STA LPF')
        lower = value - half
        upper = value + half
        if lower > peak_lower:
            peak_lower = lower
            peak_value = value
            peak_index = i
        elif peak_lower > 0 and i > peak_index and upper <= ratio * peak_lower:
            return dict(index=i, lpf=value, peak_lpf=peak_value,
                        peak_index=peak_index, ratio=value/peak_value,
                        conservative_ratio=upper/peak_lower,
                        monitor_source='STA_RIKS_TOTAL_LPF')
    return None


def backup_failed_artifacts(directory, name):
    """Legacy function name; refuse occupied destinations without moving old results."""
    suffixes = ('.log','.dat','.msg','.sta','.com','.prt','.sim','.res','.stt',
                '.mdl','.pac','.023','.ipm','.cid','.fil','.abq','.sel','.odb','_queue_solver.log')
    paths = [os.path.join(directory,name+ext) for ext in suffixes
             if os.path.isfile(os.path.join(directory,name+ext))]
    if paths:
        details=completion_details(directory,name)
        reasons=['job='+name,'analysis_state='+details['analysis_state']]
        status_path=os.path.join(directory,'STEP5_queue_status.json')
        try:
            with open(status_path,encoding='utf-8-sig') as stream:
                records=json.load(stream)
            rows=[r for r in records if isinstance(r,dict) and r.get('model')==name] if isinstance(records,list) else []
            if len(rows)==1:
                row=rows[0]
                reasons.extend(key+'='+str(row.get(field)) for key,field in (
                    ('saved_outcome','outcome'),('solver_exit_code','solver_exit_code'),
                    ('criterion_reached','criterion_reached'),('termination_requested','termination_requested')))
            else:
                reasons.append('saved_status=missing or ambiguous job record')
        except FileNotFoundError:
            reasons.append('saved_status=missing')
        except (OSError,ValueError) as error:
            reasons.append('saved_status=unreadable: '+str(error))
        reasons.extend(details['completion_diagnostics'] or details['completion_sources'] or
                       ['No terminal completion evidence found in job logs.'])
        raise ValueError('Existing solver artifacts are preserved; no solver was submitted.\n'
                         +'\n'.join(reasons)+'\nFolder: '+str(directory)+
                         '\nResume requires a verified saved curve. Do not delete solver artifacts to bypass this check. '
                         'If only extraction/status is missing, inspect the existing ODB with --extract-only; '
                         'a genuinely failed/incomplete analysis needs a corrected new run, not an overwrite.')
    return []


def validate_submission(directory, name, info, stop_method, ratio):
    """Bind future solves to the final verified INP and the compiled stopping criterion."""
    path = os.path.join(directory, name+'.inp')
    if info.get('job_name') != name or not info.get('input_sha256'):
        raise ValueError('Unverified legacy metadata: '+name+'; rebuild in a new folder or use --extract-only')
    if contract.file_sha256(path) != info['input_sha256']:
        raise ValueError('Input hash differs from STEP5 metadata: '+name)
    if stop_method == 'urdfil':
        built_ratio = info.get('settings', {}).get('postpeak_stop_ratio')
        if built_ratio is None or not math.isclose(float(built_ratio), ratio, rel_tol=0., abs_tol=1e-12):
            raise ValueError('URDFIL stop ratio differs from the generated subroutine; rebuild or use --stop-method monitor')
        if info.get('stop_subroutine_sha256'):
            from abaqus_step5_gmnia import STOP_SUBROUTINE_NAME
            if contract.file_sha256(os.path.join(directory, STOP_SUBROUTINE_NAME)) != info['stop_subroutine_sha256']:
                raise ValueError('URDFIL source differs from STEP5 metadata: '+name)
    return True


def latest_line(path):
    """Read only a bounded tail; solver-owned files may be absent or busy."""
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 16384))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        if size > 16384:
            lines = lines[1:]
        return next((line.strip()[:300] for line in reversed(lines) if line.strip()), '')
    except OSError:
        return ''


def solver_progress(directory, name, elapsed, values, requested):
    parts = [name, 'elapsed='+elapsed_text(elapsed)]
    if requested:
        parts.append('Stop requested; waiting for solver exit')
    elif not os.path.isfile(os.path.join(directory, name+'.odb')):
        parts.append('Waiting for ODB (startup/preprocessing/solver)')
    if values:
        peak = max(values)
        parts.append('LPF=%g; peak LPF=%g' % (values[-1], peak))
        if peak > 0:
            parts.append('P/Pmax=%.3f (latest available ODB sample)' % (values[-1]/peak))
    status = latest_line(os.path.join(directory, name+'.sta'))
    if status:
        parts.append('STA latest: '+status)
    else:
        status = latest_line(os.path.join(directory, name+'_queue_solver.log'))
        if status:
            parts.append('Solver log: '+status)
    return ' | '.join(parts)


def run_job(name, args, policy):
    path = os.path.join(args.run_dir,name+'.odb')
    if os.path.exists(path):
        raise ValueError('Existing ODB: '+path+'; use --extract-only to preserve/replot results, or a new run folder')
    result = dict(model=name,criterion_reached=False,stop_method=args.stop_method,
                  previous_artifacts=backup_failed_artifacts(args.run_dir,name))
    abaqus_cmd = getattr(args, 'abaqus_cmd', 'abaqus')
    cmd = job_command(name,policy.cpus,policy.gpus,args.stop_method,abaqus_cmd)
    report('SUBMIT', subprocess.list2cmdline(cmd))
    log_path = os.path.join(args.run_dir, name+'_queue_solver.log')
    report('SOLVER LOG', log_path)
    started = time.monotonic()
    next_progress = started
    values = []
    with open(log_path,'w') as log:
        process = launch(cmd,cwd=args.run_dir,stdout=log,stderr=subprocess.STDOUT)
        requested = False
        last_warning = ''
        try:
            while process.poll() is None:
                if args.stop_method == 'monitor' and not requested:
                    try:
                        sta_path = os.path.join(args.run_dir, name+'.sta')
                        sta_records = read_sta_lpf(sta_path)
                        values = [record['lpf'] for record in sta_records]
                        hit = crossing_sta(sta_records, args.ratio)
                        if hit:
                            report('STOP REQUEST', '%s | STA post-peak ratio %.3f (conservative %.3f) <= %.3f; requesting Abaqus termination' %
                                   (name, hit['ratio'], hit['conservative_ratio'], args.ratio))
                            # Native job control; do not kill the launcher or edit solver files.
                            control = launch([abaqus_cmd,'terminate','job='+name],cwd=args.run_dir)
                            code = control.wait(timeout=30)
                            result['termination_command_exit_code'] = code
                            if code == 0:
                                requested = True
                                result['observed_crossing'] = hit
                                result['monitor_source'] = 'STA_RIKS_TOTAL_LPF'
                                report('STOP ACCEPTED', name+'; waiting for solver shutdown')
                            else:
                                raise RuntimeError('Abaqus terminate command failed: %d' % code)
                    except Exception as error:
                        message = str(error)
                        if message != last_warning:
                            report('MONITOR', name+': '+message);last_warning=message
                now = time.monotonic()
                if now >= next_progress:
                    report('SOLVING', solver_progress(args.run_dir, name, now-started, values, requested))
                    next_progress = now + getattr(args, 'progress_seconds', 15.0)
                time.sleep(args.poll_seconds)
        except BaseException:
            control = launch([abaqus_cmd,'terminate','job='+name],cwd=args.run_dir)
            control.wait(timeout=30)
            process.wait(timeout=60)
            raise
        result['solver_exit_code'] = process.returncode
        result['termination_requested'] = requested
    report('SOLVER EXIT', '%s | code=%s | elapsed=%s' %
           (name, process.returncode, elapsed_text(time.monotonic()-started)))
    if not os.path.isfile(path):
        with open(log_path,errors='replace') as log:
            diagnostic=log.read()[-6000:].strip()
        result['outcome']='SOLVER_ERROR'
        result['error']='Solver produced no ODB. '+(diagnostic or 'See '+name+'_queue_solver.log')
    result.update(completion_details(args.run_dir, name))
    return result


def run(args):
    from abaqus_step5_gmnia import write_queue_batch
    report('VALIDATE', 'Checking queue inputs in '+args.run_dir)
    names = args.jobs or [os.path.splitext(os.path.basename(p))[0]
                         for pattern in ('STEP5_*.inp', 'PM_*.inp')
                         for p in sorted(glob.glob(os.path.join(args.run_dir,pattern)))]
    names = sorted(names, key=lambda name: (not name.startswith('STEP5_PERFECT_'), name))
    if not names:
        raise ValueError('No STEP5 inputs found in '+args.run_dir)
    for name in names:
        job_command(name,1,0,'monitor')
        if not os.path.isfile(os.path.join(args.run_dir,name+'.inp')):
            raise ValueError('Missing input: '+name)
        if os.path.exists(os.path.join(args.run_dir,name+'.lck')):
            raise ValueError('Job is locked/running: '+name)
    caes = glob.glob(os.path.join(args.run_dir,'Step5_GMNIA_FY*.cae'))
    cae = args.cae or (caes[0] if len(caes)==1 else None)
    report('METADATA', 'Reading STEP5 metadata (sidecars, else %s)' % (cae or 'no CAE'))
    infos = metadata(cae,names,args.run_dir)
    # Metadata is validated before any submission; all plots use P=LPF*P_ref.
    for name,info in infos.items():
        if int(info.get('settings',{}).get('field_frequency',1)) != 1:
            raise ValueError('Automatic plots require field_frequency=1: '+name)
        kind = fd.response_kind(info)
        reference_key = {'bending': 'reference_moment_Nmm', 'combined': 'pm_angle_deg'}.get(
            kind, 'reference_force_N_per_end')
        valid = info.get(reference_key) is not None and (kind == 'combined' or float(info.get(reference_key))>0)
        if info.get('stage') != 5 or not valid:
            raise ValueError('Invalid STEP5 metadata: '+name)
        if not args.extract_only:
            validate_submission(args.run_dir, name, info, args.stop_method, args.ratio)
    kinds = set(fd.response_kind(info) for info in infos.values())
    if len(kinds) != 1:
        raise ValueError('One queue folder must hold one load case; found '+', '.join(sorted(kinds)))
    axes = fd.RESPONSE_AXES[kinds.pop()]
    report('RESOURCES', 'Detecting available CPUs and GPUs')
    policy=resources.resolve_solver_policy(resources.detect_resources(),args.cpus,args.gpus)
    report('RESOURCES', 'cpus=%d gpus=%d memory=100%%; reserves=0' % (policy.cpus,policy.gpus))
    report('QUEUE', '%d jobs | mode=%s | post-peak stop=%.1f%% | progress every %g s' %
           (len(names), 'extract-only' if args.extract_only else args.stop_method,
            args.ratio*100, getattr(args, 'progress_seconds', 15.0)))
    if args.repair_only:
        # Same file name as the builder writes for this ratio (STEP5 or P-M queue).
        prefix = 'run_pm_queue_' if all(n.startswith('PM_') for n in names) else 'run_step5_queue_'
        path=os.path.join(args.run_dir,prefix+('%g' % (100.*args.ratio)).replace('.', 'p')+'_pct.bat')
        write_queue_batch(path,names,args.cpus,cae=cae,
                          script=os.path.join(SCRIPT_DIR,'abaqus_step5_queue_nogui.py'),
                          ratio=args.ratio,stop_method=args.stop_method,gpus=args.gpus)
        print('REPAIRED: '+path+'; inputs and CAE preserved; no analysis submitted')
        return
    # Refuse overwrite up front, rather than launch some jobs then discover old ODBs.
    if not args.extract_only:
        saved={}
        if getattr(args,'resume',False):
            for name in names:
                entry=completed_curve(args.run_dir,name,infos[name],args.ratio)
                if entry is not None:
                    saved[name]=entry
        for name in names:
            if name not in saved:
                backup_failed_artifacts(args.run_dir, name)
    else:
        saved={}
    if (args.stop_method=='urdfil' and not args.extract_only and len(saved)<len(names)
            and not shutil.which('ifort')):
        raise ValueError('ifort is unavailable. Initialize a compatible Intel Fortran/Visual Studio environment, or use --stop-method monitor')
    prior_results = {}
    status_path = os.path.join(args.run_dir, 'STEP5_queue_status.json')
    if args.extract_only and os.path.isfile(status_path):
        with open(status_path, encoding='utf-8') as stream:
            prior_results = {row['model']: row for row in json.load(stream)}
    previous=os.getcwd();os.chdir(args.run_dir)
    curves, summaries, yield_lines=[],[],[]
    try:
        for name,(result,points) in saved.items():
            report('REUSE COMPLETED',name+' | '+result['peak_interpretation']+' | existing CSV; no solve or ODB extraction')
            summaries.append(result)
            curves.append((curve_label(name,result),points))
            y=(1. if axes['kind']=='combined' else float(infos[name]['first_yield_moment_Nmm'])/1e6
               if axes['kind']=='bending' else float(infos[name]['end_area_mm2'])*float(infos[name]['settings']['fy'])/1000.)
            if not any(math.isclose(y,old,rel_tol=1e-10) for old in yield_lines):yield_lines.append(y)
            if result.get('error'):
                stem=name+axes['stem']+('_partial' if result.get('partial') else '')
                label=curve_label(name,result)
                plot_svg(stem+'.svg',[(label,points)],[y],name+' - '+axes['title'],axes)
                plot_png(stem+'.png',[(label,points)],[y],name+' - '+axes['title'],axes)
                result.pop('error',None)
            result.pop('comparison_error',None)
        if len(saved)==len(names):
            # No solver or ODB work: only repair/recreate small comparison outputs.
            write_comparison(args.run_dir,curves,yield_lines,summaries,axes)
            with open(status_path,'w') as stream:json.dump(summaries,stream,indent=2,allow_nan=False)
            report('QUEUE FINISHED','Saved curves reused; no analyses or ODB extraction.')
            return
        for index,name in enumerate(names,1):
            if name in saved:
                continue
            report('JOB %d/%d' % (index,len(names)), name)
            result=dict(model=name,criterion_reached=False)
            if args.extract_only:
                # Replotting cannot turn a previous solver failure into success.
                prior = prior_results.get(name, {})
                for key in ('solver_exit_code', 'termination_requested', 'termination_command_exit_code',
                            'observed_crossing', 'monitor_source', 'stop_method'):
                    if key in prior:
                        result[key] = prior[key]
                result.update(completion_details(args.run_dir, name))
            # Step-4 imperfection amplitudes of this model (u0 = aL*phiL + aD*phiD + aG*phiG).
            case=((infos.get(name) or {}).get('source_step4') or {}).get('case') or {}
            result.update(a_L_mm=case.get('local_component_mm'),a_D_mm=case.get('distortional_component_mm'),
                          a_G_mm=case.get('global_component_mm'))
            try:
                if not args.extract_only:
                    result.update(run_job(name,args,policy))
                    if result.get('error'):
                        raise RuntimeError(result['error'])
                path=os.path.join(args.run_dir,name+'.odb')
                # Wait briefly for file handles to be released; never delete a lock.
                report('WAIT FOR ODB', name+' | checking solver file locks')
                for _ in range(30):
                    if not os.path.exists(os.path.splitext(path)[0]+'.lck'):
                        break
                    time.sleep(1)
                if os.path.exists(os.path.splitext(path)[0]+'.lck'):
                    raise ValueError('ODB remains locked: '+path)
                report('EXTRACT', name+' | reading %s results and writing CSV/SVG' % axes['response'])
                summary=fd.extract(SimpleNamespace(cae=cae,odb=path,model=name,step='GMNIA',
                                    output_dir=args.run_dir,allow_partial=True,skip_png=True))
                # Re-read after solver locks were released and extraction closed the
                # ODB; do not freeze an early, incomplete text-file snapshot.
                result.update(completion_details(args.run_dir, name, summary.get('odb_job_status')))
                with open(summary['csv'],encoding='utf-8-sig') as f:
                    rows=list(csv.DictReader(f))
                points=[(float(r[axes['x']]),float(r[axes['y']])) for r in rows]
                hit=crossing([float(r['lpf']) for r in rows],args.ratio)
                result['criterion_reached']=bool(hit)
                result['first_crossing']=hit
                result['partial']=bool(summary.get('partial'))
                result.update(summary['peak'])
                result['outcome']=outcome(result,bool(hit))
                report('ANALYSIS STATUS', name+' | '+result['analysis_state']+' | '+
                       '; '.join(result['completion_diagnostics'] or result['completion_sources']))
                valid = result['outcome'] in ('CRITERION_REACHED', 'CONTROLLED_TERMINATION')
                capacity_status(result)
                label=curve_label(name,result)
                curves.append((label,points))
                info=infos[name]
                result['pm_point']=pm_point(info, summary['peak']) if valid else None
                if axes['kind'] == 'combined':
                    y=1.0
                    result.update(bending_axis_deg=info.get('bending_axis_deg'), pm_angle_deg=info.get('pm_angle_deg'),
                                  source_case=(result['pm_point'] or {}).get('source_case'))
                elif axes['kind'] == 'bending':
                    y=float(info['first_yield_moment_Nmm'])/1e6
                    result['My_kNm']=y
                    result['bending_axis_deg']=info.get('bending_axis_deg')
                    mp=(result['pm_point'] or {}).get('Mp_kNm')
                    result['Mp_kNm']=mp
                    if valid and result.get('moment_kNm') is not None:
                        result['Mu_over_My']=float(result['moment_kNm'])/y
                        if mp:
                            result['Mu_over_Mp']=float(result['moment_kNm'])/mp
                else:
                    y=float(info['end_area_mm2'])*float(info['settings']['fy'])/1000
                    result['Py_kN']=y
                    if valid and result.get('force_kN') is not None:
                        result['Pu_over_Py']=float(result['force_kN'])/y
                if result['outcome']=='CRITERION_NOT_REACHED':
                    report('POSTPEAK NOT VERIFIED', name+' | analysis completed; maximum is observed only, not verified capacity; continuing queue')
                if not any(math.isclose(y,old,rel_tol=1e-10) for old in yield_lines):
                    yield_lines.append(y)
                stem=os.path.splitext(summary['csv'])[0]
                report('PLOT', name+' | writing %s PNG' % axes['response'])
                plot_png(stem+'.png',[(label,points)],[y],name+' - '+axes['title'],axes)
                summary.update(queue_status=result,png=stem+'.png')
                with open(os.path.splitext(summary['csv'])[0]+'.json','w',encoding='utf-8') as f:
                    json.dump(summary,f,indent=2,allow_nan=False)
            except Exception as error:
                result['error']=str(error)
                report('FAILED', name+': '+str(error))
            summaries.append(result)
            with open(os.path.join(args.run_dir,'STEP5_queue_status.json'),'w') as f:
                json.dump(summaries,f,indent=2,allow_nan=False)
            try:
                report('COMPARISON', 'Updating combined plots and peak summary (%d/%d jobs processed)' % (index,len(names)))
                write_comparison(args.run_dir,curves,yield_lines,summaries,axes)
            except Exception as error:
                report('COMPARISON FAILED', str(error))
                summaries[-1]['comparison_error']=str(error)
            with open(os.path.join(args.run_dir,'STEP5_queue_status.json'),'w') as f:
                json.dump(summaries,f,indent=2,allow_nan=False)
            report('JOB %d/%d FINISHED' % (index,len(names)), name+' | '+
                   ('FAILED' if result.get('error') else result.get('outcome','UNKNOWN')))
        report('QUEUE FINISHED', 'Check STEP5_queue_status.json and STEP5_peak_summary.csv.')
        if any(r.get('error') or r.get('comparison_error') or not finished_result(r) for r in summaries):
            raise RuntimeError('Some runs/plots failed or analyses are incomplete; see queue status')
    finally:
        os.chdir(previous)


def main():
    argv=sys.argv[1:]
    if '--' in argv:
        argv=argv[argv.index('--')+1:]
    elif '-cae' in argv:
        # Match the existing builder's handling of CAE launcher arguments.
        clean, i = [], 0
        while i < len(argv):
            if argv[i] == '-cae':
                i += 1
            elif argv[i] in ('-noGUI', '-tmpdir', '-lmlog'):
                i += 2
            else:
                clean.append(argv[i]); i += 1
        argv = clean
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--cae')
    p.add_argument('--abaqus-cmd',default='abaqus',
                   help='Abaqus launcher name in PATH or full path; used for submission and termination')
    p.add_argument('--jobs',nargs='+')
    p.add_argument('--cpus',type=resources.cpu_request,default='auto',
                   help="Abaqus CPUs per job: 'auto' (all physical cores) or a positive integer")
    p.add_argument('--gpus',type=resources.gpu_request,default='auto',
                   help="Abaqus GPUs per job: 'auto' (all visible) or an integer >= 0")
    p.add_argument('--ratio',type=float,default=.70)
    p.add_argument('--poll-seconds',type=float,default=1.0)
    p.add_argument('--progress-seconds',type=float,default=15.0,
                   help='Console heartbeat interval; independent of ODB stop polling (default: 15)')
    p.add_argument('--stop-method',choices=['monitor','urdfil'],default='monitor')
    p.add_argument('--repair-only',action='store_true')
    p.add_argument('--extract-only',action='store_true')
    p.add_argument('--resume',action='store_true',
                   help='Reuse successful saved curves and solve only jobs with unused solver destinations')
    args=p.parse_args(argv)
    if args.resume and (args.extract_only or args.repair_only):
        p.error('--resume cannot be combined with --extract-only or --repair-only')
    if not 0 < args.ratio < 1 or not math.isfinite(args.poll_seconds) or args.poll_seconds<=0:
        p.error('Invalid ratio/poll interval')
    if not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0:
        p.error('Progress interval must be finite and positive')
    args.run_dir=os.path.abspath(args.run_dir)
    if args.cae:
        args.cae=os.path.abspath(args.cae)
    run(args)


if __name__=='__main__':
    main()
