# -*- coding: ascii -*-
"""Read-only ODB postprocessor for straight four-piece shell columns.

abaqus python abaqus_modal_wavelengths.py model.odb --modes 20 --sigma-ref 1

Requires Abaqus Python (odbAccess + NumPy). No model builder or CSV geometry
files are needed. No job is submitted. Coordinates must be in mm for mm output.
Cross-section shape, instance names, symmetry and bolt layout are not assumed.
The column must have common ends and longitudinal mesh tracks. Default axis Z;
use --axis x/y for other global orientations. Transverse displacements are fitted
to simply-supported sine harmonics at their actual, possibly nonuniform stations.
Spectral shares measure squared displacement coefficients, NOT strain energy.
Outputs describe modes of ONE finite column; they are not a classical signature
curve, and do not classify local/distortional/global cross-sectional mode families.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import numpy as np


_EIGEN = re.compile(r'eigen\s*value\s*[:=]\s*([-+]?(\d*)\.?(\d*)(?:[eEdD]([-+]?\d+))?)', re.I)


def parse_description(description):
    mode = re.search(r'\bmode\s*[:=]?\s*(\d+)', description, re.I)
    value = _EIGEN.search(description)
    if mode and value and (value.group(2) or value.group(3)):
        return int(mode.group(1)), float(value.group(1).replace('D','E').replace('d','e'))
    return None


def description_resolution(description):
    """Half a unit in the last digit of the printed eigenvalue."""
    value = _EIGEN.search(description)
    return .5*10.**(int(value.group(4) or 0)-len(value.group(3)))


def frame_eigen(frame):
    """(mode, eigenvalue). The description is rounded to ~5 digits; use the
    full-precision frameValue only when it agrees with the printed value."""
    parsed = parse_description(frame.description)
    if parsed is None:
        return None
    try:
        exact = float(frame.frameValue)
    except (AttributeError, TypeError, ValueError):
        return parsed
    if math.isfinite(exact) and abs(exact-parsed[1]) <= 1.01*description_resolution(frame.description):
        return parsed[0], exact
    return parsed


def cluster_ids(values, tolerance):
    """1-D clusters of sorted values with gaps <= tolerance; no rounding-bin edge splits."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ids = np.empty(len(values), dtype=int)
    ids[order] = np.cumsum(np.r_[True, np.diff(values[order]) > tolerance])-1
    return ids


class SineProjector(object):
    """Cache a weighted least-squares projection for one longitudinal grid."""
    def __init__(self, z, origin, length, max_n):
        z = np.asarray(z, dtype=float)
        self.order = np.argsort(z)
        z = z[self.order]
        if len(z) < 5 or length <= 0 or np.any(np.diff(z) <= 0):
            raise ValueError('Each track needs at least 5 distinct, ordered stations')
        # At least four intervals per half-wave, even in the sparsest region.
        self.max_n = min(int(max_n), int(math.floor(length/(4*np.max(np.diff(z)))+1e-8)),
                         (len(z)-2)//2)
        if self.max_n < 1:
            raise ValueError('Longitudinal mesh is too coarse for a half-wave fit')
        weights = np.zeros(len(z))
        weights[1:] += .5*np.diff(z)
        weights[:-1] += .5*np.diff(z)
        self.w = weights / length
        self.basis = np.sin(np.outer((z-origin)/length, np.arange(1,self.max_n+1))*np.pi)
        weighted = self.basis * np.sqrt(self.w)[:,None]
        if np.linalg.matrix_rank(weighted) != self.max_n:
            raise ValueError('Sine fit is rank deficient; reduce --max-halfwaves')
        self.project = np.linalg.pinv(weighted) * np.sqrt(self.w)[None,:]

    def evaluate(self, y):
        y = np.asarray(y, dtype=float)[self.order]
        if y.ndim == 1:
            y = y[:,None]
        if not np.all(np.isfinite(y)):
            raise ValueError('Missing or non-finite nodal displacements')
        coefficients = self.project.dot(y)
        residual = y-self.basis.dot(coefficients)
        return (np.sum(coefficients**2,axis=1),
                float(np.sum(self.w[:,None]*residual**2)),
                float(np.sum(self.w[:,None]*y**2)))


def spectral_summary(power, error2, norm2, length):
    total = float(np.sum(power))
    result = dict(max_n=len(power), dominant_n=None, second_n=None,
                  half_wavelength=None, full_wavelength=None, dominant_share=0.,
                  second_share=0., relative_error=None, shares=[0.]*len(power))
    if total <= np.finfo(float).tiny or norm2 <= np.finfo(float).tiny:
        return result
    shares = np.asarray(power)/total
    order = np.argsort(shares)[::-1]
    n = int(order[0])+1
    result.update(dominant_n=n, half_wavelength=length/n, full_wavelength=2*length/n,
                  dominant_share=float(shares[n-1]), relative_error=math.sqrt(error2/norm2),
                  shares=shares.tolist())
    if len(order) > 1:
        second = int(order[1])+1
        result.update(second_n=second, second_share=float(shares[second-1]))
    return result


def fit_sines(z, y, origin, length, max_n=100):
    fit = SineProjector(z,origin,length,max_n)
    return spectral_summary(*fit.evaluate(y), length=length)


def shell_corners(element):
    kind = str(element.type).upper()
    if kind.startswith(('S4','S8','S9')):
        return list(element.connectivity[:4])
    if kind.startswith(('S3','S6','STRI')):
        return list(element.connectivity[:3])
    return []


def collect_tracks(odb, names, axis, tolerance):
    transverse = [i for i in range(3) if i != axis]
    tracks, metadata, node_keys, node_areas = [], [], [], []
    all_coords = {name: {n.label: np.asarray(n.coordinates,dtype=float)
                         for n in odb.rootAssembly.instances[name].nodes} for name in names}
    # ODB coordinates are usually float32: never match tighter than ~8 ulp of the model size.
    scale = max(float(np.max(np.abs(list(c.values())))) for c in all_coords.values())
    tolerance = max(tolerance, 1e-6*scale)
    for name in names:
        inst = odb.rootAssembly.instances[name]
        coords = all_coords[name]
        area = {}
        for element in inst.elements:
            corners = shell_corners(element)
            if not corners:
                continue
            points = [coords[label] for label in corners]
            a = sum(.5*np.linalg.norm(np.cross(points[i]-points[0],points[i+1]-points[0]))
                    for i in range(1,len(points)-1))
            for label in element.connectivity:
                area[label] = area.get(label,0.)+a/len(element.connectivity)
        if not area:
            raise ValueError('No supported shell elements in '+name)
        low = min(coords[label][axis] for label in area)
        high = max(coords[label][axis] for label in area)
        shell_labels = list(area)
        ids = [cluster_ids([coords[label][i] for label in shell_labels],tolerance) for i in transverse]
        grouped = {}
        for k,label in enumerate(shell_labels):
            grouped.setdefault((ids[0][k],ids[1][k]),[]).append(label)
        count, covered = 0, 0
        for labels in grouped.values():
            labels.sort(key=lambda label: coords[label][axis])
            z = np.array([coords[label][axis] for label in labels])
            if (len(z)<5 or abs(z[0]-low)>tolerance or abs(z[-1]-high)>tolerance or
                    np.any(np.diff(z)<=tolerance)):
                continue
            indices = list(range(len(node_keys),len(node_keys)+len(labels)))
            node_keys.extend((name,label) for label in labels)
            node_areas.extend(area[label] for label in labels)
            tracks.append(dict(instance=name,z=z,indices=indices,
                               weight=sum(area[label] for label in labels)/(high-low)))
            count += 1
            covered += len(labels)
        coverage = covered/float(len(area))
        if count < 3 or coverage < .8:
            raise ValueError('%s: only %d full-length tracks, %.1f%% node coverage. '
                'This mesh lacks longitudinal tracks; check --axis/--coord-tol. '
                'Unstructured meshes need spatial interpolation, which is not inferred here.' %
                (name,count,100*coverage))
        metadata.append(dict(instance=name,start=float(low),end=float(high),tracks=count,
                             shell_nodes=len(area),covered_nodes=covered,coverage=coverage))
    origin = min(m['start'] for m in metadata)
    end = max(m['end'] for m in metadata)
    if end-origin <= tolerance or any(abs(m['start']-origin)>tolerance or abs(m['end']-end)>tolerance
                                     for m in metadata):
        raise ValueError('The four pieces must have common start and end coordinates')
    return tracks, metadata, node_keys, np.asarray(node_areas), origin, end-origin, tolerance


def build_groups(tracks, origin, length, requested_n):
    grouped = {}
    for track in tracks:
        # Exact station equality avoids silently treating distinct grids as identical.
        key = (track['instance'],tuple(track['z']))
        grouped.setdefault(key,[]).append(track)
    cap = min(SineProjector(items[0]['z'],origin,length,requested_n).max_n
              for items in grouped.values())
    result = []
    for (instance,z),items in grouped.items():
        result.append(dict(instance=instance,fit=SineProjector(z,origin,length,cap),
            indices=np.array([t['indices'] for t in items],dtype=int).T,
            weights=np.sqrt([t['weight'] for t in items])))
    return result, cap


def label_tables(keys):
    """Per-instance node label -> row index arrays for vectorized bulk reads."""
    items = {}
    for index,(name,label) in enumerate(keys):
        items.setdefault(name,[]).append((label,index))
    tables = {}
    for name,pairs in items.items():
        table = np.full(max(label for label,_ in pairs)+1,-1,dtype=int)
        for label,index in pairs:
            table[label] = index
        tables[name] = table
    return tables


def read_displacements(frame, lookup, count, tables=None):
    if 'U' not in frame.fieldOutputs:
        raise ValueError('Frame is missing U field output')
    field = frame.fieldOutputs['U']
    if tables:
        # Fast path: NumPy bulk blocks; any inconsistency falls back to the value loop.
        try:
            data = np.full((count,3),np.nan)
            for block in field.bulkDataBlocks:
                if block.instance is None or block.instance.name not in tables:
                    continue
                table = tables[block.instance.name]
                for accessor in ('localCoordSystemDouble', 'localCoordSystem'):
                    system = getattr(block, accessor, None)
                    if system is not None and np.size(system):
                        raise ValueError('Bulk U is not in global coordinates; inspect individual values')
                labels = np.asarray(block.nodeLabels,dtype=int).ravel()
                values = None
                for attr in ('dataDouble','data'):
                    try:
                        candidate = getattr(block,attr,None)
                    except Exception:
                        continue  # Abaqus raises for the accessor of the other precision.
                    if candidate is not None and np.size(candidate):
                        values = np.asarray(candidate,dtype=float).reshape(len(labels),-1)
                        break
                if values is None or values.shape[1] < 3:
                    raise ValueError('Unsupported bulk U block')
                keep = (labels >= 0) & (labels < len(table))
                index = table[labels[keep]]
                valid = index >= 0
                rows = index[valid]
                if len(np.unique(rows)) != len(rows) or np.any(np.isfinite(data[rows, 0])):
                    raise ValueError('Duplicate bulk nodal U values')
                data[rows] = values[keep][valid,:3]
            if np.all(np.isfinite(data)):
                return data
        except Exception:
            pass
    data = np.full((count,3),np.nan)
    for value in field.values:
        if value.instance is None:
            continue
        index = lookup.get((value.instance.name,value.nodeLabel))
        if index is not None:
            if np.isfinite(data[index, 0]):
                raise ValueError('Duplicate nodal U value')
            system = getattr(value, 'localCoordSystemDouble' if str(value.precision)=='DOUBLE_PRECISION'
                             else 'localCoordSystem', None)
            if system is not None and np.size(system):
                raise ValueError('Local-coordinate U is unsupported; global nodal output is required')
            values = value.dataDouble if str(value.precision)=='DOUBLE_PRECISION' else value.data
            data[index] = values[:3]
    if not np.all(np.isfinite(data)):
        raise ValueError('U output does not cover all selected shell nodes')
    return data


def write_csv(path, rows, fields):
    with open(path,'w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def plot_points(path, rows, length, stress_available):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    ykey = 'critical_stress_MPa' if stress_available else 'eigenvalue'
    for reliable,marker,label in ((True,'o','Dominant, well-fitted half-wave'),
                                   (False,'x','Mixed / limited fit')):
        subset = [r for r in rows if r['half_wavelength_mm'] and r[ykey]>0 and
                  (r['status']=='dominant')==reliable]
        if subset:
            ax.scatter([r['half_wavelength_mm'] for r in subset], [r[ykey] for r in subset],
                       marker=marker,label=label)
            for r in (subset if len(rows)<=10 else []):
                ax.annotate(str(r['mode']),(r['half_wavelength_mm'],r[ykey]),
                            xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Dominant half-wavelength (mm)')
    if len(rows)>10:
        ax.set_xlabel('Dominant half-wavelength (mm); mode IDs in CSV')
    ax.set_ylabel('Critical stress (MPa)' if stress_available else 'Eigenvalue (load multiplier)')
    ax.set_title('Modal points at fixed L = %g mm\nNot a classical signature curve' % length)
    ax.grid(True,which='both',alpha=.25)
    if ax.collections:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def process(args):
    from odbAccess import openOdb
    odb_path = os.path.abspath(args.odb)
    odb = openOdb(path=odb_path,readOnly=True)
    warnings = []
    try:
        candidates = {}
        for name,step in odb.steps.items():
            frames = [(frame_eigen(f),f) for f in step.frames]
            modes = [(item,f) for item,f in frames if item is not None]
            if modes:
                candidates[name] = modes
        if args.step:
            selected = [name for name in candidates if name.lower()==args.step.lower()]
        else:
            selected = list(candidates)
        if len(selected)!=1:
            raise ValueError('Choose one buckling step with --step. Available: '+str(list(candidates)))
        step_name = selected[0]
        frames = sorted(candidates[step_name],key=lambda item:item[0][0])
        if len(set(item[0][0] for item in frames)) != len(frames):
            raise ValueError('Duplicate mode numbers in the selected step; select an unambiguous ODB')
        available = len(frames)
        if args.modes:
            if args.modes>available:
                warnings.append('Requested %d modes, but ODB contains only %d.' % (args.modes,available))
            frames = frames[:args.modes]
        names = args.instances or [name for name,inst in odb.rootAssembly.instances.items()
                                   if any(shell_corners(e) for e in inst.elements)]
        if len(names)!=4 or len(set(names))!=4 or any(name not in odb.rootAssembly.instances for name in names):
            raise ValueError('Select exactly four shell instances using --instances NAME1 NAME2 NAME3 NAME4')
        axis = 'xyz'.index(args.axis)
        transverse = [i for i in range(3) if i!=axis]
        tracks,meta,keys,areas,origin,length,coord_tol = collect_tracks(odb,names,axis,args.coord_tol)
        groups,cap = build_groups(tracks,origin,length,args.max_halfwaves)
        lookup = {key:i for i,key in enumerate(keys)}
        tables = label_tables(keys)
        rows,spectra = [],[]
        for (mode,eigenvalue),frame in frames:
            u = read_displacements(frame,lookup,len(keys),tables)
            total_u = float(np.sum(areas[:,None]*u**2))
            transverse_u = float(np.sum(areas[:,None]*u[:,transverse]**2))
            trans_share = transverse_u/total_u if total_u else 0.
            power = np.zeros(cap)
            error2,norm2 = 0.,0.
            per_piece = {name:np.zeros(cap) for name in names}
            endpoint2 = endnorm2 = 0.
            for group in groups:
                y = u[group['indices']][:,:,transverse]*group['weights'][None,:,None]
                y = y.reshape(y.shape[0],-1)
                p,e,norm = group['fit'].evaluate(y)
                power += p
                per_piece[group['instance']] += p
                error2 += e
                norm2 += norm
                endpoint2 += float(np.sum(y[[0,-1]]**2))
                endnorm2 += 2*float(np.sum(np.max(y**2,axis=0)))
            summary = spectral_summary(power,error2,norm2,length)
            end_ratio = math.sqrt(endpoint2/endnorm2) if endnorm2 else 0.
            flags = []
            if trans_share < 1e-8 or summary['dominant_n'] is None:
                flags.append('no_transverse_signal')
                summary.update(dominant_n=None,half_wavelength=None,full_wavelength=None)
            else:
                if summary['dominant_share'] < args.min_share:
                    flags.append('mixed')
                if summary['relative_error'] > args.max_error:
                    flags.append('poor_fit')
                if end_ratio > .05:
                    flags.append('end_conditions_not_sine_compatible')
                if summary['dominant_n']==cap:
                    flags.append('spectral_limit')
            if eigenvalue <= 0:
                flags.append('nonpositive_eigenvalue')
            piece_modes = {name:(int(np.argmax(p))+1 if p.sum()>power.sum()*1e-8 else None)
                           for name,p in per_piece.items()}
            row = dict(mode=mode,eigenvalue=eigenvalue,
                critical_stress_MPa=eigenvalue*args.sigma_ref if args.sigma_ref is not None else None,
                dominant_halfwaves=summary['dominant_n'], half_wavelength_mm=summary['half_wavelength'],
                full_wavelength_mm=summary['full_wavelength'],dominant_share=summary['dominant_share'],
                second_halfwaves=summary['second_n'],second_share=summary['second_share'],
                relative_fit_error=summary['relative_error'],transverse_displacement_share=trans_share,
                end_displacement_ratio=end_ratio,max_resolved_halfwaves=cap,
                status=';'.join(flags) if flags else 'dominant',piece_halfwaves=json.dumps(piece_modes,sort_keys=True))
            rows.append(row)
            for i,share in enumerate(summary['shares'],1):
                spectra.append(dict(mode=mode,halfwaves=i,half_wavelength_mm=length/i,
                                    full_wavelength_mm=2*length/i,displacement_coefficient_share=share))
            print('Mode %d: eigenvalue=%g, n=%s, half-wave=%s mm, share=%.3f, %s' %
                  (mode,eigenvalue,row['dominant_halfwaves'],row['half_wavelength_mm'],
                   row['dominant_share'],row['status']))
        prefix = os.path.abspath(args.output or os.path.splitext(odb_path)[0]+'_modal_wavelengths')
        if not os.path.isdir(os.path.dirname(prefix)):
            os.makedirs(os.path.dirname(prefix))
        fields = ['mode','eigenvalue','critical_stress_MPa','dominant_halfwaves','half_wavelength_mm',
                  'full_wavelength_mm','dominant_share','second_halfwaves','second_share','relative_fit_error',
                  'transverse_displacement_share','end_displacement_ratio','max_resolved_halfwaves',
                  'status','piece_halfwaves']
        write_csv(prefix+'.csv',rows,fields)
        write_csv(prefix+'_spectra.csv',spectra,['mode','halfwaves','half_wavelength_mm','full_wavelength_mm',
                                               'displacement_coefficient_share'])
        try:
            plot_points(prefix+'.png',rows,length,args.sigma_ref is not None)
        except ImportError as error:
            warnings.append('Plot unavailable: '+str(error))
        # The ODB may contain usable frames even if the overall job was aborted.
        sta = os.path.splitext(odb_path)[0]+'.sta'
        completion = 'unknown'
        if os.path.isfile(sta):
            with open(sta) as f:
                completion = 'completed' if 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in f.read().upper() else 'not_confirmed'
        if completion!='completed':
            warnings.append('Successful job completion was not confirmed; check the analysis logs.')
        report = dict(odb=odb_path,step=step_name,axis=args.axis,length_mm=length,origin=origin,
            processed_modes=len(rows),available_modes=available,instances=meta,max_resolved_halfwaves=cap,
            reference_stress_MPa=args.sigma_ref,job_completion=completion,warnings=warnings,
            interpretation='Modal samples of one finite column, NOT a classical signature curve.',
            method='Surface-weighted transverse displacement, trapezoidal longitudinal weights, '
                   'simultaneous sine least-squares; at least 4 intervals per half-wave.',
            limits='Straight common-ended four-piece shell columns with full longitudinal node tracks. '
                   'No cross-sectional local/distortional/global classification. Shares are not strain energy. '
                   'Stress conversion assumes no preload and the user-specified reference compression.',
            settings=dict(min_share=args.min_share,max_relative_error=args.max_error,coord_tolerance=args.coord_tol,
                          effective_coord_tolerance=coord_tol))
        with open(prefix+'_report.json','w') as f:
            json.dump(report,f,indent=2)
        for warning in warnings:
            print('WARNING: '+warning)
        print('Written: '+prefix+' [.csv, _spectra.csv, .png, _report.json]')
        return report
    finally:
        odb.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('odb')
    parser.add_argument('--modes',type=int,default=0,help='First N available modes; 0 reads all')
    parser.add_argument('--step',help='Buckling step; auto-select if only one is present')
    parser.add_argument('--instances',nargs=4,help='Four shell instance names; auto-detected by default')
    parser.add_argument('--axis',choices=('x','y','z'),default='z')
    parser.add_argument('--sigma-ref',type=float,help='Reference compression in MPa, no preload; omit to retain eigenvalues only')
    parser.add_argument('--max-halfwaves',type=int,default=100)
    parser.add_argument('--coord-tol',type=float,default=1e-4,help='Track matching tolerance, mm; raised to 1e-6 x model size for float32 ODB coordinates')
    parser.add_argument('--min-share',type=float,default=.7,help='Heuristic dominant-mode share threshold')
    parser.add_argument('--max-error',type=float,default=.2,help='Heuristic relative fit-error threshold')
    parser.add_argument('--output',help='Output path prefix; default: ODB name + _modal_wavelengths')
    args = parser.parse_args(argv)
    if args.modes<0 or args.max_halfwaves<1 or not args.coord_tol>0 or not math.isfinite(args.coord_tol):
        parser.error('Mode limits and coordinate tolerance must be positive (modes=0 means all)')
    if not 0<args.min_share<=1 or not 0<args.max_error<=1:
        parser.error('Share and error thresholds must be in (0,1]')
    if args.sigma_ref is not None and (not args.sigma_ref>0 or not math.isfinite(args.sigma_ref)):
        parser.error('--sigma-ref must be positive and finite')
    return process(args)


if __name__=='__main__':
    main()
