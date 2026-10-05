# -*- coding: utf-8 -*-
"""Run/repair existing Step5 inputs, monitor 70% drop, extract and compare curves.

abaqus cae noGUI=abaqus_step5_queue_nogui.py -- --run-dir DIR --repair-only
Then run the regenerated run_step5_queue_70_pct.bat. No model rebuild needed.
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

import abaqus_step5_force_displacement as fd
import abaqus_pipeline_contract as contract
import runtime_resources as resources


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


def job_command(name, cpus, gpus, mode):
    if not re.fullmatch(r'STEP5_[A-Za-z0-9_]+', name):
        raise ValueError('Invalid STEP5 job name: '+name)
    cmd = ['abaqus', 'job='+name, 'input='+name+'.inp', 'cpus='+str(cpus),
           'memory=100%', 'standard_parallel=all', 'interactive']
    if gpus:
        cmd.append('gpus='+str(gpus))
    if mode == 'urdfil':
        cmd.append('user=step5_postpeak_stop.for')
    return cmd


def outcome(result, reached):
    code = result.get('solver_exit_code', 0)
    if code != 0 and not (result.get('termination_requested') and reached):
        return 'SOLVER_ERROR'
    if not reached:
        return 'CRITERION_NOT_REACHED'
    return 'CONTROLLED_TERMINATION' if result.get('termination_requested') else 'CRITERION_REACHED'


def launch(cmd, **kwargs):
    # Abaqus on Windows is a .bat launcher; quote arguments before cmd.exe.
    return subprocess.Popen(subprocess.list2cmdline(cmd) if os.name == 'nt' else cmd,
                            shell=os.name == 'nt', **kwargs)


def plot_png(path, curves, yield_lines, title):
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
                                xValuesLabel='Axial shortening (mm)',
                                yValuesLabel='Axial compressive load (kN)', legendLabel=label)
            abq_curves.append(session.Curve(xyData=xy)); names.append(name)
        xs = [x for _, points in curves for x, _ in points]
        for i, value in enumerate(yield_lines):
            name = 'STEP5_QUEUE_YIELD_%d' % i
            if name in session.xyDataObjects:
                del session.xyDataObjects[name]
            xy = session.XYData(name=name, data=((min(xs),value),(max(xs),value)),
                                legendLabel='AsFy = %.3f kN' % value,
                                xValuesLabel='Axial shortening (mm)', yValuesLabel='Axial compressive load (kN)')
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
    for label, points in curves:
        ax.plot([p[0] for p in points], [p[1] for p in points], label=label, lw=1.6)
    for y in yield_lines:
        ax.axhline(y, ls='--', label='AsFy = %.3f kN' % y)
    ax.set(xlabel='Axial shortening, delta (mm)', ylabel='Axial compressive load, P (kN)', title=title)
    ax.grid(alpha=.25); ax.legend(loc='best', fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_svg(path, curves, yield_lines, title):
    xs = [x for _, points in curves for x, _ in points]
    ys = [y for _, points in curves for _, y in points]+list(yield_lines)+[0.0]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    dx, dy = max(1e-6, xmax-xmin), max(1e-6, ymax-ymin)
    xmin -= .04*dx; xmax += .04*dx; ymax += .08*dy
    sx = lambda x: 110+(x-xmin)/(xmax-xmin)*1050
    sy = lambda y: 670-(y-ymin)/(ymax-ymin)*580
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="850" viewBox="0 0 1400 850">',
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
        color = colors[i % len(colors)]
        poly = ' '.join('%.2f,%.2f' % (sx(x),sy(y)) for x,y in points)
        lines.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (poly,color))
        lines.append('<text x="%d" y="%d" fill="%s">%s</text>' % (115+(i%3)*420,775+(i//3)*22,color,escape(label)))
    for value in yield_lines:
        lines.append('<path d="M 110 %.2f H 1160" stroke="#1676c3" stroke-dasharray="8 5"/>' % sy(value))
        lines.append('<text x="1170" y="%.2f">AsFy %.3f kN</text>' % (sy(value),value))
    lines += ['<path d="M 110 90 V 670 H 1160" fill="none" stroke="black"/>',
              '<text x="630" y="740" text-anchor="middle">Axial shortening, delta (mm)</text>',
              '<text transform="translate(30,380) rotate(-90)" text-anchor="middle">Axial compressive load, P (kN)</text>',
              '</g></svg>']
    with open(path,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_comparison(directory, curves, yield_lines, summaries):
    fields = ['model','force_kN','shortening_mm','criterion_reached','outcome','solver_exit_code','error']
    with open(os.path.join(directory,'STEP5_peak_summary.csv'),'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        writer.writerows(summaries)
    if not curves:
        return
    stem = os.path.join(directory, 'STEP5_all_force_displacement_combined')
    plot_svg(stem+'.svg',curves,yield_lines,'STEP5 GMNIA - Combined force-shortening curves')
    plot_png(stem+'.png',curves,yield_lines,'STEP5 GMNIA - Combined force-shortening curves')


def metadata(cae, names):
    import caeModules
    from abaqus import openMdb
    db = openMdb(pathName=cae)
    try:
        return {name: contract.parse_prefixed_json(db.models[name].description,fd.STEP5_PREFIX)
                for name in names}
    finally:
        db.close()


def read_live_lpf(path):
    from odbAccess import openOdb
    odb = openOdb(path=path, readOnly=True)
    try:
        return [y for _,y in fd.lpf_history(odb.steps['GMNIA'])]
    finally:
        odb.close()


def backup_failed_artifacts(directory, name):
    # Retain diagnostics from failed compilation and prevent Abaqus overwrite prompts.
    suffixes = ('.log','.dat','.msg','.sta','.com','.prt','.sim','.res','.stt',
                '.mdl','.pac','.023','.ipm','.cid','.fil','.abq','.sel')
    paths = [os.path.join(directory,name+ext) for ext in suffixes
             if os.path.isfile(os.path.join(directory,name+ext))]
    if not paths:
        return []
    import tempfile
    target = tempfile.mkdtemp(prefix=name+'_previous_',dir=directory)
    result=[]
    for path in paths:
        destination=os.path.join(target,os.path.basename(path))
        shutil.move(path,destination);result.append(destination)
    return result


def run_job(name, args, policy):
    path = os.path.join(args.run_dir,name+'.odb')
    if os.path.exists(path):
        raise ValueError('Existing ODB: '+path+'; use --extract-only to preserve/replot results, or a new run folder')
    result = dict(model=name,criterion_reached=False,stop_method=args.stop_method,
                  previous_artifacts=backup_failed_artifacts(args.run_dir,name))
    cmd = job_command(name,policy.cpus,policy.gpus,args.stop_method)
    print('RUN: '+subprocess.list2cmdline(cmd));sys.stdout.flush()
    with open(name+'_queue_solver.log','w') as log:
        process = launch(cmd,cwd=args.run_dir,stdout=log,stderr=subprocess.STDOUT)
        requested = False
        last_warning = ''
        try:
            while process.poll() is None:
                if args.stop_method == 'monitor' and not requested and os.path.isfile(path):
                    try:
                        hit = crossing(read_live_lpf(path), args.ratio)
                        if hit:
                            # Native job control; do not kill the launcher or edit solver files.
                            control = launch(['abaqus','terminate','job='+name],cwd=args.run_dir)
                            code = control.wait(timeout=30)
                            result['termination_command_exit_code'] = code
                            if code == 0:
                                requested = True
                                result['observed_crossing'] = hit
                                print('70% crossing observed: '+name);sys.stdout.flush()
                            else:
                                raise RuntimeError('Abaqus terminate command failed: %d' % code)
                    except Exception as error:
                        # ODB can be unavailable/not yet flushed while Standard is writing it.
                        message = str(error)
                        if message != last_warning:
                            print('MONITOR: '+name+': '+message);sys.stdout.flush();last_warning=message
                time.sleep(args.poll_seconds)
        except BaseException:
            control = launch(['abaqus','terminate','job='+name],cwd=args.run_dir)
            control.wait(timeout=30)
            process.wait(timeout=60)
            raise
        result['solver_exit_code'] = process.returncode
        result['termination_requested'] = requested
    return result


def run(args):
    from abaqus_step5_gmnia import write_queue_batch
    names = args.jobs or [os.path.splitext(os.path.basename(p))[0]
                         for p in sorted(glob.glob(os.path.join(args.run_dir,'STEP5_*.inp')))]
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
    if not cae:
        raise ValueError('Specify --cae: expected exactly one Step5_GMNIA_FY*.cae')
    infos = metadata(cae,names)
    # Metadata is validated before any submission; all plots use P=LPF*P_ref.
    for name,info in infos.items():
        if int(info.get('settings',{}).get('field_frequency',1)) != 1:
            raise ValueError('Automatic plots require field_frequency=1: '+name)
        if info.get('stage') != 5 or float(info.get('reference_force_N_per_end',0))<=0:
            raise ValueError('Invalid STEP5 metadata: '+name)
    policy=resources.resolve_policy(resources.detect_resources(),args.cpus,args.gpus)
    print('RESOURCE REQUEST: cpus=%d gpus=%d memory=100%%; reserves=0' % (policy.cpus,policy.gpus))
    if args.repair_only:
        path=os.path.join(args.run_dir,'run_step5_queue_70_pct.bat')
        if os.path.exists(path):
            shutil.copy2(path,path+'.backup_'+time.strftime('%Y%m%d_%H%M%S'))
        write_queue_batch(path,names,'auto',cae=cae,
                          script=os.path.join(SCRIPT_DIR,'abaqus_step5_queue_nogui.py'),
                          ratio=args.ratio,stop_method=args.stop_method)
        print('REPAIRED: '+path+'; inputs and CAE preserved; no analysis submitted')
        return
    if args.stop_method == 'urdfil' and not args.extract_only and not shutil.which('ifort'):
        raise ValueError('ifort is unavailable. Initialize a compatible Intel Fortran/Visual Studio environment, or use --stop-method monitor')
    # Refuse overwrite up front, rather than launch some jobs then discover old ODBs.
    if not args.extract_only:
        for name in names:
            if os.path.exists(os.path.join(args.run_dir,name+'.odb')):
                raise ValueError('Existing ODB: '+name+'; use --extract-only or a new run folder')
    previous=os.getcwd();os.chdir(args.run_dir)
    curves, summaries, yield_lines=[],[],[]
    try:
        for index,name in enumerate(names,1):
            print('[%d/%d] %s' % (index,len(names),name));sys.stdout.flush()
            result=dict(model=name,criterion_reached=False)
            try:
                if not args.extract_only:
                    result.update(run_job(name,args,policy))
                path=os.path.join(args.run_dir,name+'.odb')
                # Wait briefly for file handles to be released; never delete a lock.
                for _ in range(30):
                    if not os.path.exists(os.path.splitext(path)[0]+'.lck'):
                        break
                    time.sleep(1)
                if os.path.exists(os.path.splitext(path)[0]+'.lck'):
                    raise ValueError('ODB remains locked: '+path)
                summary=fd.extract(SimpleNamespace(cae=cae,odb=path,model=name,step='GMNIA',
                                    output_dir=args.run_dir,allow_partial=True))
                with open(summary['csv'],encoding='utf-8-sig') as f:
                    rows=list(csv.DictReader(f))
                points=[(float(r['shortening_mm']),float(r['force_kN'])) for r in rows]
                hit=crossing([float(r['lpf']) for r in rows],args.ratio)
                result['criterion_reached']=bool(hit)
                result['first_crossing']=hit
                result.update(summary['peak'])
                result['outcome']=outcome(result,bool(hit))
                label=name.replace('STEP5_','')+(' (solver error)' if result['outcome']=='SOLVER_ERROR' else '' if hit else ' (70% not reached)')
                curves.append((label,points))
                info=infos[name]
                y=float(info['end_area_mm2'])*float(info['settings']['fy'])/1000
                if not any(math.isclose(y,old,rel_tol=1e-10) for old in yield_lines):
                    yield_lines.append(y)
                stem=os.path.splitext(summary['csv'])[0]
                plot_png(stem+'.png',[(label,points)],[y],name+' - Riks force-shortening')
                summary.update(queue_status=result,png=stem+'.png')
                with open(os.path.splitext(summary['csv'])[0]+'.json','w',encoding='utf-8') as f:
                    json.dump(summary,f,indent=2,allow_nan=False)
            except Exception as error:
                result['error']=str(error)
                print('FAILED: '+name+': '+str(error));sys.stdout.flush()
            summaries.append(result)
            with open(os.path.join(args.run_dir,'STEP5_queue_status.json'),'w') as f:
                json.dump(summaries,f,indent=2,allow_nan=False)
            try:
                write_comparison(args.run_dir,curves,yield_lines,summaries)
            except Exception as error:
                print('COMPARISON PLOT FAILED: '+str(error));sys.stdout.flush()
                summaries[-1]['comparison_error']=str(error)
            with open(os.path.join(args.run_dir,'STEP5_queue_status.json'),'w') as f:
                json.dump(summaries,f,indent=2,allow_nan=False)
        print('QUEUE FINISHED. Check STEP5_queue_status.json and STEP5_peak_summary.csv.')
        if any(r.get('error') or r.get('comparison_error') or r.get('outcome')=='SOLVER_ERROR' or not r.get('criterion_reached') for r in summaries):
            raise RuntimeError('Some runs/plots failed or did not reach the post-peak criterion; see queue status')
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
    p.add_argument('--jobs',nargs='+')
    p.add_argument('--cpus',default='auto')
    p.add_argument('--gpus',default='auto')
    p.add_argument('--ratio',type=float,default=.70)
    p.add_argument('--poll-seconds',type=float,default=1.0)
    p.add_argument('--stop-method',choices=['monitor','urdfil'],default='monitor')
    p.add_argument('--repair-only',action='store_true')
    p.add_argument('--extract-only',action='store_true')
    args=p.parse_args(argv)
    if not 0 < args.ratio < 1 or not math.isfinite(args.poll_seconds) or args.poll_seconds<=0:
        p.error('Invalid ratio/poll interval')
    args.run_dir=os.path.abspath(args.run_dir)
    if args.cae:
        args.cae=os.path.abspath(args.cae)
    run(args)


if __name__=='__main__':
    main()
