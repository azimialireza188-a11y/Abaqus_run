"""Fast read-only geometric mode screening for Step 4 (Abaqus Python 3).

U-only screening; not mFSM, cFSM, strain energy or DSM capacity acceptance.
The full existing curved-wall split is retained, with sparse wall residuals
and low-dimensional fold operators replacing dense per-mode multiplication.
ODB reads stay in the main thread; detached NumPy fields can run concurrently.
"""
import csv
import glob
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import nullcontext
import numpy as np
import abaqus_modal_wavelengths as base
from abaqus_modal_report import SectionProjector, load_physical_segments
from runtime_resources import detect_resources, resolve_policy, available_memory

VERSION = 'fast-curved-wall-screening-2'
FAMILIES = ('GLOBAL', 'DISTORTIONAL', 'LOCAL')
KEYS = ('G', 'A', 'D', 'L', 'O')
LIMITATION = ('Geometric displacement screening, not energy/modal participation or DSM acceptance. '
              'Percentages are normalized component self-norms; components need not be orthogonal. '
              'Inspect selected mode shapes before building imperfections.')


def longitudinal_weights(z):
    z = np.asarray(z, dtype=float)
    if len(z) < 2 or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError('Distinct ordered finite longitudinal stations required')
    w = np.zeros(len(z)); w[1:] += np.diff(z)/2.; w[:-1] += np.diff(z)/2.
    return w


def worker_count(cpus, tasks, available, bytes_per_task):
    if min(cpus, tasks, bytes_per_task) < 1:
        raise ValueError('Positive CPU/task/working-set counts required')
    capacity = int(available)//int(bytes_per_task)
    if capacity < 1: raise MemoryError('Available memory cannot fit one mode')
    return min(int(cpus), int(tasks), capacity)


def benchmark_batch(operation, workers, count):
    """Time steady batch throughput, including device input/output transfers."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(operation,range(workers)))  # create workers and warm caches
        timings=[]
        for unused in range(2):
            begin=time.perf_counter()
            list(pool.map(operation,range(count)))
            timings.append(time.perf_counter()-begin)
    return min(timings)


def select_backend(kernel, first, gpu_count, workers, count, working_bytes):
    info=dict(backend='numpy_cpu',reason='NO_SUPPORTED_GPU')
    if not gpu_count: return np,(),info
    try:
        import cupy as cp
        devices=[]
        expected=kernel.vector(first)
        for device in range(min(gpu_count,cp.cuda.runtime.getDeviceCount())):
            with cp.cuda.Device(device):
                if cp.cuda.runtime.memGetInfo()[0] < working_bytes: continue
                kernel.prepare_device(cp,device)
                actual=kernel.vector(first,cp,device)
                cp.cuda.Stream.null.synchronize()
                if not np.allclose(actual,expected,rtol=1e-8,atol=1e-10):
                    raise ValueError('GPU/CPU screening mismatch')
            devices.append(device)
        if not devices: return np,(),dict(info,reason='GPU_WORKING_SET_UNAVAILABLE')
        probe_count=min(count,max(workers,len(devices))*2)
        cpu_seconds=benchmark_batch(lambda i:kernel.vector(first),workers,probe_count)
        # One executor per device prevents concurrent allocations on that device.
        def device_batch():
            def run(item):
                position,device=item
                with cp.cuda.Device(device):
                    for i in range(position,probe_count,len(devices)):
                        kernel.vector(first,cp,device)
                    cp.cuda.Stream.null.synchronize()
            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                list(pool.map(run,enumerate(devices)))
        timings=[]
        for unused in range(2):
            begin=time.perf_counter(); device_batch()
            timings.append(time.perf_counter()-begin)
        gpu_seconds=min(timings)
        info.update(cpu_batch_seconds=cpu_seconds,gpu_batch_seconds=gpu_seconds,
                    probe_modes=probe_count,cpu_workers=workers,gpu_devices=devices)
        if gpu_seconds < cpu_seconds:
            info.update(backend='cupy_gpu',reason='MEASURED_FASTER_BATCH_THROUGHPUT')
            return cp,tuple(devices),info
        info['reason']='PARALLEL_CPU_MEASURED_FASTER'
        kernel.device_arrays.clear()
        return np,(),info
    except (ImportError,OSError,RuntimeError,MemoryError) as exc:
        kernel.device_arrays.clear()
        return np,(),dict(info,reason='GPU_UNAVAILABLE: '+str(exc))


class ScreeningKernel:
    def __init__(self, fit):
        self.sqrtw = fit.sqrtw
        self.fold = np.array([2*i+j for i in fit.metadata['fold_nodes'] for j in (0, 1)])
        # G/A/D depend only on folds: local residual vanishes at every anchor.
        ops = (fit.pglobal, fit.passembly, fit.pdist)
        off = np.ones(len(self.sqrtw), dtype=bool); off[self.fold] = False
        if any(np.max(np.abs(op[:, off]), initial=0.) > 1e-10 for op in ops):
            raise ValueError('Fold compression would omit non-anchor dependence')
        self.coarse = np.vstack([op[:, self.fold] for op in ops])
        rows, cols = np.nonzero(fit.plocal)
        self.local_rows, counts = np.unique(rows, return_counts=True)
        width = int(counts.max(initial=0))
        self.local_cols = np.zeros((len(counts), width), dtype=int)
        self.local_values = np.zeros_like(self.local_cols, dtype=float)
        start = 0
        for i, (row, count) in enumerate(zip(self.local_rows, counts)):
            self.local_cols[i, :count] = cols[start:start+count]
            self.local_values[i, :count] = fit.plocal[row, cols[start:start+count]]
            start += count
        centered = fit.xy-np.average(fit.xy, axis=0, weights=fit.weights)
        self.twist = np.c_[-centered[:, 1], centered[:, 0]].ravel()*self.sqrtw
        self.twist /= np.linalg.norm(self.twist)
        self.device_arrays = {}

    def prepare_device(self, xp, device):
        with xp.cuda.Device(device):
            self.device_arrays[device] = tuple(xp.asarray(a) for a in
                (self.sqrtw, self.fold, self.coarse, self.local_rows,
                 self.local_cols, self.local_values, self.twist))

    def vector(self, x, xp=np, device=None):
        arrays = ((self.sqrtw, self.fold, self.coarse, self.local_rows,
                   self.local_cols, self.local_values, self.twist)
                  if xp is np else self.device_arrays[device])
        sqrtw, fold, coarse, lr, lc, lv, twist = arrays
        x = xp.asarray(x, dtype=float)
        # Normalize before squaring: eigenvector amplitude/sign are arbitrary.
        scale = xp.max(xp.abs(x))
        if not bool(xp.isfinite(x).all()) or float(scale) <= 0:
            raise ValueError('Missing/nonfinite/zero transverse mode field')
        x = x/scale
        local = xp.zeros_like(x)
        if len(self.local_rows): local[:, lr] = xp.sum(x[:, lc]*lv[None, :, :], axis=2)
        projected = x[:, fold] @ coarse.T
        g, a, d = xp.split(projected, 3, axis=1)
        o = x-local-g-a-d
        parts = (g, a, d, local, o)
        norms = xp.stack([xp.sum((p*sqrtw[None, :])**2) for p in parts])
        input_norm = xp.sum((x*sqrtw[None, :])**2)
        error = xp.linalg.norm((x-sum(parts))*sqrtw[None, :])/xp.sqrt(input_norm)
        twist_norm = xp.sum(((g*sqrtw[None, :]) @ twist)**2)
        result = xp.concatenate((norms, xp.stack((input_norm, error, twist_norm))))
        return result if xp is np else xp.asnumpy(result)

    def evaluate(self, x, xp=np, device=None):
        v = self.vector(x, xp, device)
        g, a, d, l, o, input_norm, error, twist_norm = map(float, v)
        ldg = g+d+l; total = ldg+a+o
        if ldg <= 1e-24*total: g=d=l=ldg=0.
        return dict(global_percent=100*g/max(ldg, 1e-250),
                    distortional_percent=100*d/max(ldg, 1e-250),
                    local_percent=100*l/max(ldg, 1e-250),
                    assembly_percent=100*a/max(total, 1e-250),
                    other_percent=100*o/max(total, 1e-250),
                    reconstruction_relative_error=error,
                    component_norm_sum_over_input=total/max(input_norm, 1e-250),
                    global_twist_fraction=twist_norm/max(g, 1e-250))


def classify(split, supported, threshold=80.):
    scores = [split[k] for k in ('global_percent', 'distortional_percent', 'local_percent')]
    family = FAMILIES[int(np.argmax(scores))]
    flags = []
    if max(scores) < threshold: flags.append('mixed_family')
    if split['assembly_percent'] >= 25.: flags.append('relative_piece_motion')
    if split['other_percent'] >= 25.: flags.append('extension_or_unmodelled_wall_motion')
    if not supported and family != 'GLOBAL': flags.append('wall_geometry_unresolved')
    if split['reconstruction_relative_error'] > 1e-7: flags.append('reconstruction_error')
    twist = split.get('global_twist_fraction', 0.)
    subtype = ('torsional' if twist >= .8 else 'flexural' if twist <= .2
               else 'flexural-torsional') if family == 'GLOBAL' else None
    return dict(family=family, status='UNRESOLVED' if flags else 'CANDIDATE',
                dominance_percent=max(scores), global_subtype=subtype, flags=flags)


def flag_clusters(rows, relative_gap=2e-4):
    positive = sorted((r for r in rows if r['eigenvalue'] > 0), key=lambda r:r['eigenvalue'])
    clusters = []
    for row in positive:
        if not clusters or row['eigenvalue']-clusters[-1][-1]['eigenvalue'] > relative_gap*row['eigenvalue']:
            clusters.append([])
        clusters[-1].append(row)
    for cluster in clusters:
        if len(cluster) < 2: continue
        disagree = len({r['family'] for r in cluster}) > 1
        for row in cluster:
            row['flags'].append('near_repeated_eigenvalue')
            row['cluster_modes'] = [r['mode'] for r in cluster]
            if disagree:
                row['flags'].append('cluster_family_disagreement'); row['status']='UNRESOLVED'
            if row.get('global_subtype'):
                row['observed_global_subtype'] = row['global_subtype']
                row['global_subtype'] = None


def wall_mapping_status(metadata, names, coordinate_tolerance):
    """Separate coordinate precision from bounded discretization of a fold.

    Free section ends must match coordinates. An interior physical boundary
    can lie between mesh nodes: nearest-node snapping is accepted only within
    half its local adjacent mesh interval. Approximation remains diagnostic.
    """
    numeric=10*coordinate_tolerance
    checks=[]; snapped=False; reasons=[]
    expected=sum(len(v['walls']) for v in metadata['source_wall_layouts'].values())
    if not metadata['geometry_supported']: reasons.append('unsupported_section_topology')
    if set(metadata['source_wall_layouts'])!=set(names): reasons.append('missing_piece_wall_layout')
    if expected==0 or metadata['physical_wall_count']!=expected: reasons.append('missing_or_collapsed_physical_walls')
    endpoints=metadata.get('wall_endpoint_mapping',[])
    if not endpoints: reasons.append('endpoint_mapping_missing')
    for endpoint in endpoints:
        allowance=numeric if endpoint['piece_end'] else .5*endpoint['adjacent_edge_mm']+numeric
        accepted=endpoint['error_mm']<=allowance
        checks.append(dict(endpoint,allowed_error_mm=allowance,accepted=accepted))
        if not accepted: reasons.append('endpoint_outside_local_mesh_interval')
        if endpoint['error_mm']>numeric: snapped=True
    return dict(supported=not reasons,mesh_snapped=snapped,checks=checks,
                reasons=sorted(set(reasons)),coordinate_tolerance_mm=coordinate_tolerance,
                definition='exact free ends; nearest-node interior boundaries within half local adjacent edge')


def prepare_geometry(odb, names, axis, run_dir, physical_segments=None):
    tracks, mesh, keys, areas, origin, length, tol = base.collect_tracks(odb, names, axis, 1e-5)
    if any(m['coverage'] < .98 for m in mesh):
        raise ValueError('Fast screening requires at least 98% full-track node coverage')
    z = tracks[0]['z']
    if any(len(t['z']) != len(z) or not np.allclose(t['z'], z, atol=tol, rtol=0) for t in tracks):
        raise ValueError('Fast screening requires common longitudinal stations; no mesh interpolation is invented')
    transverse = [i for i in range(3) if i != axis]
    lookup = {key:i for i,key in enumerate(keys)}
    node_track = {i:j for j,t in enumerate(tracks) for i in t['indices']}
    coordinates = {(name,n.label):np.asarray(n.coordinates,dtype=float)
                   for name in names for n in odb.rootAssembly.instances[name].nodes}
    xy = np.array([coordinates[keys[t['indices'][0]]][transverse] for t in tracks])
    edges = set()
    for name in names:
        for e in odb.rootAssembly.instances[name].elements:
            if str(e.type).upper() not in ('S4', 'S4R', 'S3', 'S3R'):
                raise ValueError('Fast track topology currently supports linear S3/S4 shells: '+str(e.type))
            corners = base.shell_corners(e)
            for a,b in zip(corners, corners[1:]+corners[:1]):
                ka,kb = (name,a),(name,b)
                if ka in lookup and kb in lookup and abs(coordinates[ka][axis]-coordinates[kb][axis]) <= tol:
                    ia,ib = node_track[lookup[ka]],node_track[lookup[kb]]
                    if ia != ib: edges.add(tuple(sorted((ia,ib))))
    source = physical_segments if physical_segments is not None else load_physical_segments(run_dir)
    fit = SectionProjector(xy, sorted(edges), [t['instance'] for t in tracks],
                           [t['weight'] for t in tracks], physical_segments=source, compute_bases=False)
    mapping=wall_mapping_status(fit.metadata,names,tol)
    fit.metadata['fast_wall_mapping']=mapping
    supported=source is not None and mapping['supported']
    warnings = []
    if source is None: warnings.append('source_walls_missing_mesh_fallback')
    if not supported: warnings.append('physical_wall_mapping_unconfirmed')
    elif mapping['mesh_snapped']: warnings.append('wall_boundary_mesh_snapped')
    if fit.metadata['curved_panel_proxy']: warnings.append('limited_physical_wall_coverage')
    # Keep only compiled sparse/fold operators and metadata. Release the dense
    # temporary projector before reading mode fields or choosing batch capacity.
    return dict(metadata=fit.metadata, kernel=ScreeningKernel(fit), keys=keys, lookup=lookup,
                tables=base.label_tables(keys), indices=np.array([t['indices'] for t in tracks]).T,
                transverse=transverse, z=z, sqrtz=np.sqrt(longitudinal_weights(z)/length),
                length=length, supported=supported, warnings=warnings, mesh=mesh, tolerance=tol)


def read_mode(frame, geometry):
    u = base.read_displacements(frame, geometry['lookup'], len(geometry['keys']), geometry['tables'])
    transverse = u[geometry['indices']][:, :, geometry['transverse']]
    peak = float(np.max(np.linalg.norm(transverse, axis=2)))
    if peak <= 0: raise ValueError('Mode has no transverse displacement')
    values = transverse.reshape(len(geometry['z']), -1)
    end_ratio = float(np.max(np.linalg.norm(transverse[[0,-1]], axis=2))/peak)
    # A supporting resolution estimate only. Never determines L/D/G.
    column = values[:, int(np.argmax(np.max(np.abs(values), axis=0)))]
    active = column[np.abs(column) > 1e-5*np.max(np.abs(column))]
    halfwaves = 1+int(np.count_nonzero(active[1:]*active[:-1] < 0))
    intervals = geometry['length']/halfwaves/np.max(np.diff(geometry['z']))
    return values*geometry['sqrtz'][:, None], dict(end_displacement_ratio=end_ratio,
        estimated_halfwaves=halfwaves, estimated_half_wavelength_mm=geometry['length']/halfwaves,
        intervals_per_estimated_halfwave=float(intervals))


def cache_signature(path, args):
    s = os.stat(path)
    digest = hashlib.sha256()
    for name in sorted(glob.glob(os.path.join(args.run_dir, '*_build.json'))):
        with open(name, 'rb') as stream: digest.update(stream.read())
    status=os.path.join(args.run_dir,'pipeline_status.json')
    if os.path.isfile(status):
        with open(status,'rb') as stream: digest.update(stream.read())
    for name in ('abaqus_fast_modal_suggest.py','abaqus_modal_report.py','abaqus_physical_walls.py',
                 'abaqus_modal_wavelengths.py','abaqus_step4_imperfections.py','abaqus_pipeline_contract.py'):
        with open(os.path.join(os.path.dirname(__file__),name),'rb') as stream: digest.update(stream.read())
    return dict(version=VERSION, odb=os.path.realpath(path), size=s.st_size,
                mtime_ns=s.st_mtime_ns, axis=args.axis, step=args.step,
                instances=args.instances, definition_hash=digest.hexdigest())


def write_report(path, report):
    # Each file is replaced atomically; JSON is the final cache commit marker.
    descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    csv_temporary=None
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        csv_descriptor, csv_temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
        fields = ['mode','eigenvalue','family','status','global_subtype','dominance_percent',
                  'global_percent','distortional_percent','local_percent','assembly_percent','other_percent',
                  'estimated_half_wavelength_mm','component_norm_sum_over_input','flags']
        with os.fdopen(csv_descriptor,'w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
            for row in report['modes']:
                writer.writerow(dict(row,flags=';'.join(row['flags'])))
        os.replace(csv_temporary,os.path.splitext(path)[0]+'.csv')
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.remove(temporary)
        if csv_temporary is not None and os.path.exists(csv_temporary): os.remove(csv_temporary)


def print_report(report, cached=False):
    print('FAST ODB SHAPE SCREENING%s | %s' % (' (cache)' if cached else '',VERSION))
    print(LIMITATION)
    mapping=report['geometry'].get('fast_wall_mapping',{})
    print('Wall mapping: supported=%s, max error=%.6g mm, mesh-snapped=%s'%
          (mapping.get('supported'),report['geometry']['maximum_wall_mapping_error_mm'],
           mapping.get('mesh_snapped')))
    for item in [c for c in mapping.get('checks',[]) if not c['accepted']][:3]:
        print('  failed boundary: piece=%s node=%s error=%.6g mm allowed=%.6g mm'%
              (item['piece'],item['node'],item['error_mm'],item['allowed_error_mm']))
    for family in FAMILIES:
        rows=sorted((r for r in report['modes'] if r['family']==family and r['status']=='CANDIDATE'
                     and r['eigenvalue']>0), key=lambda r:r['eigenvalue'])
        print('%s candidates:'%family)
        if not rows: print('  NONE confirmed by this geometric screen; inspect unresolved shapes.')
        for r in rows[:5]:
            print('  mode=%d eigenvalue=%.8g G/D/L=%.1f/%.1f/%.1f%% relative=%.1f%% other=%.1f%% flags=%s'%
                  (r['mode'],r['eigenvalue'],r['global_percent'],r['distortional_percent'],
                   r['local_percent'],r['assembly_percent'],r['other_percent'],';'.join(r['flags']) or 'none'))
    unresolved=sorted((r for r in report['modes'] if r['status']=='UNRESOLVED'),
                      key=lambda r:r['eigenvalue'])
    print('Unresolved: %d/%d modes (lowest positive eigenvalues below)'%(len(unresolved),len(report['modes'])))
    for r in [r for r in unresolved if r['eigenvalue']>0][:5]:
        print('  mode=%d dominant=%s flags=%s'%(r['mode'],r['family'],';'.join(r['flags'])))
    print('JSON/CSV: '+report['report_path'])
    print('No model/job created; no mode or amplitude selected automatically.')


def suggest(args):
    started=time.perf_counter()
    paths=glob.glob(os.path.join(args.run_dir,'*.odb')) if not args.odb else [os.path.abspath(args.odb)]
    if len(paths)!=1: raise ValueError('Expected one ODB; use --odb explicitly')
    path=paths[0]
    if os.path.exists(os.path.splitext(path)[0]+'.lck'):
        raise ValueError('ODB is locked; wait for the buckling job to finish')
    signature=cache_signature(path,args)
    output=os.path.join(args.run_dir,os.path.splitext(os.path.basename(path))[0]+'_fast_mode_screening.json')
    if (not args.suggest_refresh and os.path.isfile(output)
            and os.path.isfile(os.path.splitext(output)[0]+'.csv')):
        try:
            with open(output,encoding='utf-8') as stream: cached=json.load(stream)
            if cached.get('signature')==signature:
                print_report(cached,True); return cached
        except (OSError,ValueError): pass
    from odbAccess import openOdb
    policy=resolve_policy(detect_resources(),args.suggest_cpus,args.suggest_gpus)
    odb=openOdb(path=path,readOnly=True)
    try:
        if args.step not in odb.steps: raise ValueError('Buckling step missing: '+args.step)
        names=args.instances or sorted(name for name,inst in odb.rootAssembly.instances.items()
            if len(inst.elements) and all(base.shell_corners(e) for e in inst.elements))
        if len(names)!=4 or len(set(names))!=4: raise ValueError('Select exactly four distinct shell instances')
        geometry=prepare_geometry(odb,names,'xyz'.index(args.axis),args.run_dir,
                                  getattr(args,'suggest_section_segments',None))
        frames=[(base.frame_eigen(f),f) for f in odb.steps[args.step].frames if base.frame_eigen(f)]
        if not frames: raise ValueError('No eigenmode frames found')
        if len({item[0] for item,f in frames})!=len(frames): raise ValueError('Duplicate ODB mode IDs')
        bytes_per_mode=geometry['indices'].size*2*8
        # Working set includes input, local/coarse/residual and sparse gather arrays.
        working=bytes_per_mode*(12+geometry['kernel'].local_cols.shape[1])
        workers=worker_count(policy.cpus,len(frames),available_memory()[1],working)
        try:
            from threadpoolctl import threadpool_limits
            limits=threadpool_limits(limits=1)
            thread_mode='parallel_modes_single_thread_blas'
        except ImportError:
            limits=nullcontext()
            if all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
                thread_mode='parallel_modes_startup_single_thread_blas'
            else:
                workers=1; thread_mode='single_mode_runtime_blas'
        kernel=geometry['kernel']
        first,first_info=read_mode(frames[0][1],geometry)
        # Compare supported GPU batch throughput with the actual parallel CPU
        # alternative, rather than comparing against only one CPU worker.
        xp,devices,backend=select_backend(kernel,first,policy.gpus,workers,len(frames),working)
        if devices:
            workers=worker_count(len(devices),len(frames),available_memory()[1],working)
            devices=devices[:workers]; thread_mode='one_mode_per_gpu'
        print('Fast screening: %d modes, %d logical CPUs, %d active workers, %s'%
              (len(frames),policy.cpus,workers,backend['backend']),flush=True)
        rows=[]
        def evaluate(item, values, info, device):
            if xp is np: split=kernel.evaluate(values)
            else:
                with xp.cuda.Device(device): split=kernel.evaluate(values,xp,device)
            row=dict(mode=item[0],eigenvalue=item[1],**split,**info,**classify(split,geometry['supported']))
            row['flags'].extend(geometry['warnings'])
            if item[1]<=0: row['flags'].append('nonpositive_eigenvalue'); row['status']='UNRESOLVED'
            if info['end_displacement_ratio']>.05:
                row['flags'].append('nonzero_transverse_ends'); row['status']='UNRESOLVED'
            if info['intervals_per_estimated_halfwave']<4:
                row['flags'].append('near_resolution_limit'); row['status']='UNRESOLVED'
            return row
        iterator=iter(enumerate(frames)); pending={}
        last_progress=0; progress_time=time.perf_counter()
        with limits, ThreadPoolExecutor(max_workers=workers) as pool:
            def submit(device=None):
                try: index,(item,frame)=next(iterator)
                except StopIteration: return False
                values,info=(first,first_info) if index==0 else read_mode(frame,geometry)
                pending[pool.submit(evaluate,item,values,info,device)]=device
                return True
            for i in range(workers): submit(devices[i] if devices else None)
            while pending:
                done,unused=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    device=pending.pop(future); rows.append(future.result()); submit(device)
                if (len(rows)-last_progress>=25 or not pending or
                        time.perf_counter()-progress_time>=5.):
                    print('Screened %d/%d modes'%(len(rows),len(frames)),flush=True)
                    last_progress=len(rows); progress_time=time.perf_counter()
        rows.sort(key=lambda r:r['mode']); flag_clusters(rows)
        if cache_signature(path,args)!=signature: raise ValueError('ODB or geometry changed during screening')
        report=dict(signature=signature,limitation=LIMITATION,report_path=output,modes=rows,
                    geometry=geometry['metadata'],mesh=geometry['mesh'],
                    reference_pipeline=getattr(args,'suggest_reference',None),
                    resources=dict(requested_cpus=policy.cpus,workers=workers,thread_mode=thread_mode,
                                   memory_reserve_bytes=0,gpu_reserve_bytes=0,backend=backend),
                    elapsed_seconds=time.perf_counter()-started)
        write_report(output,report); print_report(report); return report
    finally:
        odb.close()
