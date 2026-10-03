# -*- coding: utf-8 -*-
"""Enhanced modal reports AFTER abaqus_modal_wavelengths.py; never submit a job.

abaqus python abaqus_modal_report.py "RUN/model_modal_wavelengths_report.json"

Classification is a geometric displacement proxy, NOT cFSM/GBT identification.
All wavelength and eigenvalue data come from the existing CSV/report; the ODB
is read once for cross-sectional deformation diagnostics. Requires Abaqus Python
3, NumPy and Matplotlib. Output: 3 PNGs, interactive offline HTML, CSVs and JSON.
"""
import argparse
import base64
import csv
import html
import importlib.util
import json
import math
import os
import sys
from collections import Counter
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(
    globals().get('__file__', sys._getframe().f_code.co_filename)))
COLORS = {'Global-like': '#2369b0', 'Distortional-like': '#e58a16',
          'Local-like': '#29946b', 'Mixed': '#8b58a0', 'Unresolved': '#777777'}
LIMITATION = ('Geometric displacement classification only; not cFSM/GBT or strain-energy '
    'participation. Global = rigid motion of the entire cross-section; distortional proxy = '
    'coarse section deformation and relative component motion; local proxy = residual '
    'deformation within detected panels. Membrane stretching can enter the distortional '
    'proxy. Curved panels weaken the physical interpretation and are flagged; unsupported '
    'topology or poorly fitted modes remain unresolved. Inspect the displayed shapes.')


def orth(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    matrix = matrix[:, norms > 1e-12]
    if not matrix.shape[1]:
        return np.empty((matrix.shape[0], 0))
    matrix = matrix / np.linalg.norm(matrix, axis=0)
    u, singular, unused = np.linalg.svd(matrix, full_matrices=False)
    return u[:, singular > singular[0]*1e-10]


class SectionProjector:
    """Nested, weighted subspaces; contributions sum to one, but are not energies."""
    def __init__(self, xy, edges, pieces, weights, corner_angle=15.):
        self.xy = np.asarray(xy, dtype=float)
        count = len(xy)
        self.sqrtw = np.repeat(np.sqrt(weights), 2)
        centered = self.xy-np.average(self.xy, axis=0, weights=weights)
        rigid = np.zeros((count, 2, 3))
        rigid[:, 0, 0] = 1.
        rigid[:, 1, 1] = 1.
        rigid[:, 0, 2] = -centered[:, 1]
        rigid[:, 1, 2] = centered[:, 0]
        self.qglobal = orth(rigid.reshape(-1, 3)*self.sqrtw[:, None])
        adjacency = [set() for _ in range(count)]
        for a, b in edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        anchors = set()
        for i, neighbors in enumerate(adjacency):
            if len(neighbors) != 2:
                anchors.add(i)
            else:
                a, b = list(neighbors)
                v, w = self.xy[i]-self.xy[a], self.xy[b]-self.xy[i]
                angle = math.degrees(math.acos(float(np.clip(np.dot(v, w)/
                    max(np.linalg.norm(v)*np.linalg.norm(w), 1e-30), -1., 1.))))
                if angle >= corner_angle:
                    anchors.add(i)
        anchors = sorted(anchors)
        anchor_ids = {node: i for i, node in enumerate(anchors)}
        shape = np.zeros((count, len(anchors)))
        for node, i in anchor_ids.items():
            shape[node, i] = 1.
        visited, paths = set(), []
        for first in anchors:
            for neighbor in adjacency[first]:
                if tuple(sorted((first, neighbor))) in visited:
                    continue
                path, previous, current = [first], first, neighbor
                while True:
                    visited.add(tuple(sorted((previous, current))))
                    path.append(current)
                    if current in anchor_ids:
                        break
                    next_nodes = adjacency[current]-{previous}
                    if len(next_nodes) != 1 or len(path) > count:
                        break
                    previous, current = current, next(iter(next_nodes))
                if path[-1] not in anchor_ids or path[-1] == first:
                    continue
                paths.append(path)
                distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.xy[path], axis=0), axis=1))]
                t = distance/max(distance[-1], 1e-30)
                shape[path, anchor_ids[first]] = 1.-t
                shape[path, anchor_ids[path[-1]]] = t
        total_length = sum(np.linalg.norm(self.xy[b]-self.xy[a]) for a, b in edges)
        flat_length = 0.
        for path in paths:
            points = self.xy[path]
            chord = points[-1]-points[0]
            length = np.linalg.norm(chord)
            if length:
                deviation = np.abs(np.cross(points-points[0], chord))/length
                if np.max(deviation) <= .02*length:
                    flat_length += np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        coarse = np.kron(shape, np.eye(2))
        # Independent rigid motions of the pieces must not be called plate-local bending.
        piece_columns = []
        pieces = np.asarray(pieces)
        for name in sorted(set(pieces)):
            mask = np.repeat(pieces == name, 2)
            piece_columns.append(rigid.reshape(-1, 3)*mask[:, None])
        coarse = np.column_stack([coarse]+piece_columns)*self.sqrtw[:, None]
        coarse -= self.qglobal.dot(self.qglobal.T.dot(coarse))
        self.qdist = orth(coarse)
        self.flat_fraction = float(flat_length/max(total_length, 1e-30))
        self.supported = bool(
            count*2-self.qglobal.shape[1]-self.qdist.shape[1] >= 2 and
            len(visited) == len(edges) and all(adjacency))
        self.metadata = dict(anchor_nodes=anchors, corner_angle_deg=corner_angle,
            flat_panel_length_fraction=self.flat_fraction, geometry_supported=self.supported,
            curved_panel_proxy=self.flat_fraction < .6,
            global_rank=self.qglobal.shape[1], coarse_deformation_rank=self.qdist.shape[1],
            residual_rank=count*2-self.qglobal.shape[1]-self.qdist.shape[1])

    def shares(self, coefficients):
        y = np.asarray(coefficients).reshape(len(coefficients), -1)*self.sqrtw
        total = float(np.sum(y*y))
        if total <= 1e-300:
            return [0., 0., 0.]
        g = y.dot(self.qglobal)
        d = y.dot(self.qdist)
        remainder = y-g.dot(self.qglobal.T)-d.dot(self.qdist.T)
        values = np.array([np.sum(g*g), np.sum(d*d), np.sum(remainder*remainder)])
        return (values/values.sum()).tolist()


def family_label(shares, supported, threshold=.65, poor_fit=False):
    if sum(shares) <= 0 or poor_fit:
        return 'Unresolved'
    if shares[0] >= threshold:
        return 'Global-like'
    if not supported:
        return 'Unresolved'
    if shares[1] >= threshold:
        return 'Distortional-like'
    if shares[2] >= threshold:
        return 'Local-like'
    return 'Mixed'


def sample_envelope(rows, length, value_key):
    grouped = {}
    for row in rows:
        if row.get('dominant_halfwaves') and row.get(value_key) and row[value_key] > 0:
            grouped.setdefault(row['dominant_halfwaves'], []).append(row)
    result = []
    for n, items in sorted(grouped.items()):
        best = min(items, key=lambda r: r[value_key])
        filtered = [r for r in items if r['quality_ok']]
        clean = min(filtered, key=lambda r: r[value_key]) if filtered else None
        result.append(dict(halfwaves=n, half_wavelength_mm=length/n, sample_count=len(items),
            minimum_value=best[value_key], minimum_mode=best['mode'],
            filtered_minimum_value=clean[value_key] if clean else None,
            filtered_minimum_mode=clean['mode'] if clean else None))
    return result


def envelope_line(rows, key):
    x, y, previous = [], [], None
    for row in rows:
        if previous is not None and row['halfwaves'] != previous+1:
            x.append(float('nan'))
            y.append(float('nan'))
        x.append(row['half_wavelength_mm'])
        y.append(row[key] if row[key] is not None else float('nan'))
        previous = row['halfwaves']
    return x, y


def load_base(module_path):
    spec = importlib.util.spec_from_file_location('enhanced_base_processor', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_results(report_path):
    report_path = os.path.abspath(report_path)
    if not report_path.endswith('_report.json'):
        raise ValueError('Input must be the existing *_modal_wavelengths_report.json')
    prefix = report_path[:-len('_report.json')]
    with open(report_path) as stream:
        metadata = json.load(stream)
    with open(prefix+'.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    integer = ('mode', 'dominant_halfwaves', 'max_resolved_halfwaves')
    floats = ('eigenvalue', 'critical_stress_MPa', 'half_wavelength_mm', 'dominant_share',
              'relative_fit_error', 'end_displacement_ratio', 'transverse_displacement_share')
    for row in rows:
        for key in integer:
            row[key] = int(row[key]) if row[key] else None
        for key in floats:
            row[key] = float(row[key]) if row[key] else None
    rows.sort(key=lambda r: r['mode'])
    modes = [r['mode'] for r in rows]
    if len(set(modes)) != len(modes) or len(rows) != metadata['processed_modes']:
        raise ValueError('Mode CSV and report disagree, or contain duplicate modes')
    spectra = np.full((len(rows), metadata['max_resolved_halfwaves']), np.nan)
    index = {mode: i for i, mode in enumerate(modes)}
    with open(prefix+'_spectra.csv', newline='') as stream:
        for row in csv.DictReader(stream):
            i, j = index[int(row['mode'])], int(row['halfwaves'])-1
            if not math.isnan(spectra[i, j]):
                raise ValueError('Duplicate spectrum entry')
            spectra[i, j] = float(row['displacement_coefficient_share'])
    if not np.all(np.isfinite(spectra)) or np.any(spectra < 0):
        raise ValueError('Incomplete or invalid spectrum CSV')
    for i, row in enumerate(rows):
        if row['dominant_halfwaves']:
            if not np.isclose(spectra[i].sum(), 1., atol=1e-6):
                raise ValueError('Spectrum shares do not sum to one')
            if not np.isclose(spectra[i, row['dominant_halfwaves']-1], row['dominant_share']):
                raise ValueError('Spectrum and mode CSV disagree')
    return metadata, rows, spectra, prefix


def section_diagnostics(base, metadata, rows, spectra, report_dir, corner_angle, family_threshold):
    from odbAccess import openOdb
    odb_path = os.path.join(report_dir, os.path.basename(metadata['odb']))
    if not os.path.isfile(odb_path):
        odb_path = metadata['odb']
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        axis = 'xyz'.index(metadata['axis'])
        transverse = [i for i in range(3) if i != axis]
        names = [item['instance'] for item in metadata['instances']]
        tracks, meshmeta, keys, areas, origin, length, tolerance = base.collect_tracks(
            odb, names, axis, metadata['settings']['coord_tolerance'])
        if not np.isclose(length, metadata['length_mm']):
            raise ValueError('ODB length differs from the previous report')
        groups, cap = base.build_groups(tracks, origin, length, metadata['max_resolved_halfwaves'])
        if cap != spectra.shape[1]:
            raise ValueError('ODB mesh/spectrum resolution mismatch; regenerate base outputs')
        lookup = {key: i for i, key in enumerate(keys)}
        tables = base.label_tables(keys)
        coordinates = {(name, n.label): np.asarray(n.coordinates, dtype=float)
                       for name in names for n in odb.rootAssembly.instances[name].nodes}
        track_index = {}
        for i, track in enumerate(tracks):
            for index in track['indices']:
                track_index[index] = i
        xy = np.array([coordinates[keys[t['indices'][0]]][transverse] for t in tracks])
        edges = set()
        for name in names:
            for element in odb.rootAssembly.instances[name].elements:
                nodes = base.shell_corners(element)
                for a, b in zip(nodes, nodes[1:]+nodes[:1]):
                    ka, kb = (name, a), (name, b)
                    if ka not in lookup or kb not in lookup:
                        continue
                    ia, ib = track_index[lookup[ka]], track_index[lookup[kb]]
                    if ia != ib and abs(coordinates[ka][axis]-coordinates[kb][axis]) <= tolerance:
                        edges.add(tuple(sorted((ia, ib))))
        edges = sorted(edges)
        fit = SectionProjector(xy, edges, [t['instance'] for t in tracks],
                               [t['weight'] for t in tracks], corner_angle)
        group_tracks = [[track_index[int(index)] for index in group['indices'][0]] for group in groups]
        frames = {}
        for frame in odb.steps[metadata['step']].frames:
            item = base.frame_eigen(frame)
            if item:
                if item[0] in frames:
                    raise ValueError('Duplicate ODB mode numbers')
                frames[item[0]] = (item[1], frame)
        shapes = []
        for i, row in enumerate(rows):
            eigenvalue, frame = frames[row['mode']]
            if not np.isclose(eigenvalue, row['eigenvalue'], rtol=1e-7, atol=1e-8):
                raise ValueError('ODB eigenvalue differs from CSV at mode %d' % row['mode'])
            displacement = base.read_displacements(frame, lookup, len(keys), tables)
            coefficients = np.zeros((cap, len(tracks), 2))
            for group, indices in zip(groups, group_tracks):
                values = displacement[group['indices']][:, :, transverse]
                projected = group['fit'].project.dot(values[group['fit'].order].reshape(len(values), -1))
                coefficients[:, indices, :] = projected.reshape(cap, len(indices), 2)
            # Match original spectra, not just mode count, before combining old/new data.
            power = np.sum(coefficients**2*np.array([t['weight'] for t in tracks])[None, :, None], axis=(1, 2))
            if power.sum() > 1e-300 and not np.allclose(power/power.sum(), spectra[i], atol=1e-6, rtol=1e-5):
                raise ValueError('ODB mode shape differs from original spectrum at mode %d' % row['mode'])
            shares = fit.shares(coefficients)
            bad = (row['relative_fit_error'] is None or
                   row['relative_fit_error'] > metadata['settings']['max_relative_error'] or
                   row['end_displacement_ratio'] > .05 or
                   row['transverse_displacement_share'] < 1e-8)
            row.update(global_proxy_share=shares[0], distortional_proxy_share=shares[1],
                       local_proxy_share=shares[2],
                       family=family_label(shares, fit.supported, family_threshold, bad),
                       classification_basis='heuristic_transverse_displacement',
                       geometry_proxy_supported=fit.supported)
            n = row['dominant_halfwaves']
            shapes.append(coefficients[n-1].tolist() if n else np.zeros((len(tracks), 2)).tolist())
            if (i+1) % 50 == 0 or i == len(rows)-1:
                print('Enhanced section diagnostics: %d/%d modes' % (i+1, len(rows)))
                sys.stdout.flush()
        return dict(xy=xy.tolist(), edges=edges, pieces=[t['instance'] for t in tracks],
                    dominant_harmonic_shapes=shapes, diagnostics=fit.metadata, odb=odb_path)
    finally:
        odb.close()


def make_plots(prefix, metadata, rows, spectra, envelope, value_key, near_fraction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    length, cap = metadata['length_mm'], spectra.shape[1]
    limit, caution = length/cap, length/(cap*near_fraction)
    ylabel = 'Critical stress (MPa)' if value_key == 'critical_stress_MPa' else 'Eigenvalue (load multiplier)'
    def wavelength_axis(ax):
        ax.set_xscale('log')
        ax.set_xlim(limit*.65, length*1.15)
        ax.axvspan(limit*.65, limit, color='#dddddd', hatch='///', alpha=.5)
        ax.axvspan(limit, caution, color='#e6ad30', alpha=.12)
        ax.axvline(limit, color='#b13333', linestyle=':', linewidth=1.)
        ax.set_xlabel('Half-wavelength (mm); hatched = outside resolved range')
        ax.grid(True, which='both', alpha=.2)
    fig, ax = plt.subplots(figsize=(12, 7))
    for family, color in COLORS.items():
        subset = [r for r in rows if r['family'] == family and r['half_wavelength_mm'] and r[value_key] > 0]
        if subset:
            ax.scatter([r['half_wavelength_mm'] for r in subset], [r[value_key] for r in subset],
                s=[25+125*r['dominant_share'] for r in subset], c=color, alpha=.8,
                label='%s (%d)' % (family, len(subset)), edgecolors='white', linewidths=.4)
    limited = [r for r in rows if r['half_wavelength_mm'] and r[value_key] > 0 and not r['quality_ok']]
    if limited:
        ax.scatter([r['half_wavelength_mm'] for r in limited], [r[value_key] for r in limited],
                   marker='x', s=20, color='#303030', linewidths=.65, label='Spectral/fit/resolution flag')
    for family in COLORS:
        selected = [r for r in rows if r['family'] == family and r['half_wavelength_mm'] and r[value_key] > 0]
        if selected:
            r = min(selected, key=lambda item: item[value_key])
            ax.annotate('m%d' % r['mode'], (r['half_wavelength_mm'], r[value_key]),
                        xytext=(5, -12), textcoords='offset points', fontsize=8)
    wavelength_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title('Fixed-length modal samples | L = %g mm\nHeuristic section families; marker size = dominant harmonic share' % length)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.text(.06, .015, 'Family shares are displacement proxies, not strain energies. No mesh-convergence claim. Not a classical signature curve.', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(prefix+'_modes.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    half = length/np.arange(cap, 0, -1)
    if cap > 1:
        centers = np.sqrt(half[:-1]*half[1:])
        bounds = np.r_[half[0]**2/centers[0], centers, half[-1]**2/centers[-1]]
    else:
        bounds = np.array([half[0]*.9, half[0]*1.1])
    picture = ax.pcolormesh(bounds, np.arange(len(rows)+1)-.5, spectra[:, ::-1],
                           shading='flat', cmap='viridis', norm=PowerNorm(gamma=.45, vmin=0, vmax=1), rasterized=True)
    ax.scatter([r['half_wavelength_mm'] or np.nan for r in rows], np.arange(len(rows)),
               s=9, facecolors='none', edgecolors='white', linewidths=.5)
    wavelength_axis(ax)
    ax.set_ylim(len(rows)-.5, -.5)
    ticks = np.unique(np.linspace(0, len(rows)-1, min(12, len(rows))).astype(int))
    ax.set_yticks(ticks)
    ax.set_yticklabels([rows[i]['mode'] for i in ticks])
    ax.set_ylabel('Mode number')
    ax.set_title('Longitudinal spectrum of every mode\nWhite circles: dominant harmonic; colors: squared-displacement coefficient shares')
    fig.colorbar(picture, ax=ax, label='Harmonic share', ticks=[0, .05, .1, .25, .5, .75, 1.])
    fig.tight_layout()
    fig.savefig(prefix+'_spectra.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for key, style, label in [('minimum_value', 'o--', 'Minimum of available dominant-half-wave samples'),
            ('filtered_minimum_value', 's-', 'Minimum after spectral/fit/resolution filtering')]:
        x, y = envelope_line(envelope, key)
        ax.plot(x, y, style, markersize=4, linewidth=1., label=label)
    wavelength_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title('Discrete sample lower envelope | NOT a classical signature curve\nGaps remain where no dominant-half-wave sample is available')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(prefix+'_envelope.png', dpi=180)
    plt.close(fig)


def write_html(prefix, metadata, rows, spectra, geometry, value_key):
    images = {}
    for name in ('modes', 'spectra', 'envelope'):
        with open(prefix+'_'+name+'.png', 'rb') as stream:
            images[name] = base64.b64encode(stream.read()).decode('ascii')
    payload = json.dumps(dict(rows=rows, spectra=spectra.tolist(), geometry=geometry,
        length=metadata['length_mm'], valueKey=value_key, colors=COLORS), ensure_ascii=True, allow_nan=False).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Enhanced buckling mode report</title>
<style>body{font:15px system-ui;margin:25px auto;max-width:1200px;padding:0 20px;color:#203044;background:#f6f8fb}h1,h2{font-weight:650}section{background:white;padding:20px;margin:20px 0;border:1px solid #dde2ea;border-radius:9px}img{width:100%}button,select,input{font:inherit;padding:6px;margin:4px}svg{width:100%;height:auto;background:#fff}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}tr[data-index]{cursor:pointer}tr[data-index]:hover{background:#edf3ff}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.note{color:#5b6576;font-size:13px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>Fixed-length buckling modes</h1><p>Geometric family labels are <b>heuristic</b>. This report is not a classical signature curve.</p>
<p class="note">__LIMITATION__</p>
<section><h2>1. Modal samples</h2><p>Colors identify the proposed section family. Size measures the dominant harmonic share, not statistical confidence. Crosses identify flagged wavelength estimates. Mode numbers shown on the plot are selected family minima.</p><img src="data:image/png;base64,__MODES__"></section>
<section><h2>2. Inspect a mode and its main harmonics</h2><button id="prev">Previous</button><select id="mode"></select><button id="next">Next</button><label>Drawing amplification <input id="amplitude" type="range" min="0" max="30" value="10"></label><div id="details"></div>
<div class="grid"><div><h3>Dominant-harmonic section shape</h3><svg id="shape" viewBox="0 0 500 450"></svg><p class="note">Gray dashed: original section; colored: dominant-harmonic transverse coefficients at arbitrary scale. This is not a physical displacement amplitude or a full multimodal deformation snapshot.</p></div><div><h3>Longitudinal harmonic shares</h3><svg id="spectrum" viewBox="0 0 500 450"></svg><p class="note">All harmonics are shown; the table lists the most important ones. Multiple half-wavelengths belong to the same eigenvalue, not independent buckling solutions.</p></div></div><div id="components"></div></section>
<section><h2>3. Spectrum map</h2><img src="data:image/png;base64,__SPECTRA__"></section>
<section><h2>4. Lower envelope of available samples</h2><p>Only positive-eigenvalue dominant-half-wave samples are used. Secondary components are not treated as independent solutions. Lines do not bridge missing harmonics. Filtering is not mesh-convergence verification.</p><img src="data:image/png;base64,__ENVELOPE__"></section>
<section><h2>5. Mode table</h2><label>Family <select id="family"><option>All</option></select></label><div style="max-height:600px;overflow:auto"><table><thead><tr><th>Mode</th><th>Eigenvalue</th><th>Half-wave mm</th><th>Dominant %</th><th>Family (heuristic)</th><th>G / D / L %</th><th>Flags</th></tr></thead><tbody id="table"></tbody></table></div></section>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
function svg(tag,attrs,parent,text){let e=document.createElementNS(NS,tag);for(let k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;parent.appendChild(e);return e}
function fmt(x,n=2){return x===null?'--':Number(x).toFixed(n)}
D.rows.forEach((r,i)=>{let o=new Option('Mode '+r.mode+' | '+r.family,i);$('mode').add(o)});
Object.keys(D.colors).forEach(f=>$('family').add(new Option(f,f)));
function render(){let i=+$('mode').value,r=D.rows[i], color=D.colors[r.family], shares=[r.global_proxy_share,r.distortional_proxy_share,r.local_proxy_share];
$('details').textContent='Eigenvalue '+fmt(r.eigenvalue)+' | Half-wave '+fmt(r.half_wavelength_mm)+' mm | '+r.family+' | G/D/L proxies '+shares.map(v=>fmt(100*v,1)+'%').join(' / ')+' | Flags: '+r.enhanced_flags;
let xy=D.geometry.xy, u=D.geometry.dominant_harmonic_shapes[i], mins=[0,1].map(k=>Math.min(...xy.map(p=>p[k]))), maxs=[0,1].map(k=>Math.max(...xy.map(p=>p[k]))), span=Math.max(maxs[0]-mins[0],maxs[1]-mins[1],1), center=mins.map((v,k)=>(v+maxs[k])/2), umax=Math.max(...u.map(v=>Math.hypot(...v)),1e-30), gain=span*(+$('amplitude').value/100)/umax;
let map=p=>[250+(p[0]-center[0])*330/span,225-(p[1]-center[1])*330/span], deformed=xy.map((p,j)=>p.map((v,k)=>v+gain*u[j][k]));$('shape').replaceChildren();
for(let [a,b] of D.geometry.edges){for(let [points,stroke,dash] of [[xy,'#9ca7b2','4 3'],[deformed,color,'']]){let p=map(points[a]),q=map(points[b]);svg('line',{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:stroke,'stroke-width':2,'stroke-dasharray':dash},$('shape'))}}
for(let j of D.geometry.diagnostics.anchor_nodes){let p=map(xy[j]);svg('circle',{cx:p[0],cy:p[1],r:2,fill:'#45556a'},$('shape'))}
$('spectrum').replaceChildren();let s=D.spectra[i],w=420/s.length;
svg('line',{x1:50,y1:390,x2:475,y2:390,stroke:'#333'},$('spectrum'));
s.forEach((v,j)=>{let e=svg('rect',{x:50+j*w,y:390-v*340,width:Math.max(w-.7,.3),height:v*340,fill:color},$('spectrum'));svg('title',{},e,'n='+(j+1)+'; half-wave='+fmt(D.length/(j+1))+' mm; share='+fmt(v*100)+'%');if(j===0||(j+1)%5===0)svg('text',{x:50+(j+.5)*w,y:410,'text-anchor':'middle','font-size':11},$('spectrum'),j+1)});
for(let v of [0,.25,.5,.75,1])svg('text',{x:43,y:394-v*340,'text-anchor':'end','font-size':11},$('spectrum'),Math.round(v*100)+'%');
svg('text',{x:250,y:439,'text-anchor':'middle','font-size':12},$('spectrum'),'Number of longitudinal half-waves n');
$('components').replaceChildren();let p=document.createElement('p');p.textContent='Top components: '+r.top_components.map(c=>'n='+c.halfwaves+', h='+fmt(c.half_wavelength_mm)+' mm, '+fmt(c.share*100,1)+'%').join(' | ')+'; retained share '+fmt(r.top_components_share*100,1)+'%';$('components').appendChild(p)}
function table(){let target=$('table');target.replaceChildren();D.rows.forEach((r,i)=>{if($('family').value!=='All'&&r.family!==$('family').value)return;let tr=document.createElement('tr');tr.dataset.index=i;for(let v of [r.mode,fmt(r.eigenvalue),fmt(r.half_wavelength_mm),fmt(100*r.dominant_share,1),r.family,[r.global_proxy_share,r.distortional_proxy_share,r.local_proxy_share].map(x=>fmt(100*x,1)).join(' / '),r.enhanced_flags]){let td=document.createElement('td');td.textContent=v;tr.appendChild(td)}tr.onclick=()=>{$('mode').value=i;render();$('mode').scrollIntoView({behavior:'smooth'})};target.appendChild(tr)})}
$('mode').onchange=render;$('amplitude').oninput=render;$('family').onchange=table;
$('prev').onclick=()=>{$('mode').value=Math.max(0,+$('mode').value-1);render()};$('next').onclick=()=>{$('mode').value=Math.min(D.rows.length-1,+$('mode').value+1);render()};render();table();
</script></html>'''
    for key, value in {'__LIMITATION__': html.escape(LIMITATION), '__DATA__': payload,
                       '__MODES__': images['modes'], '__SPECTRA__': images['spectra'],
                       '__ENVELOPE__': images['envelope']}.items():
        page = page.replace(key, value)
    with open(prefix+'.html', 'w', encoding='utf-8') as stream:
        stream.write(page)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modal_report')
    parser.add_argument('--output', help='Output prefix; default: base prefix + _enhanced')
    parser.add_argument('--top-components', type=int, default=3)
    parser.add_argument('--corner-angle', type=float, default=15.)
    parser.add_argument('--family-threshold', type=float, default=.65)
    parser.add_argument('--near-limit-fraction', type=float, default=.8)
    args = parser.parse_args(argv)
    if args.top_components < 1 or not 0 < args.corner_angle < 180 or not .5 < args.family_threshold <= 1 or not 0 < args.near_limit_fraction <= 1:
        parser.error('Invalid component count, angle or threshold')
    metadata, rows, spectra, base_prefix = load_results(args.modal_report)
    prefix = os.path.abspath(args.output or base_prefix+'_enhanced')
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    base = load_base(os.path.join(SCRIPT_DIR, 'abaqus_modal_wavelengths.py'))
    geometry = section_diagnostics(base, metadata, rows, spectra,
        os.path.dirname(os.path.abspath(args.modal_report)), args.corner_angle, args.family_threshold)
    components = []
    for i, row in enumerate(rows):
        order = np.argsort(-spectra[i], kind='stable')[:args.top_components]
        top = [dict(halfwaves=int(j)+1, half_wavelength_mm=metadata['length_mm']/(j+1),
                    share=float(spectra[i, j])) for j in order if spectra[i, j] > 1e-12]
        row['top_components'] = top
        row['top_components_share'] = sum(c['share'] for c in top)
        n = row['dominant_halfwaves']
        row['near_resolution_limit'] = bool(n and n >= args.near_limit_fraction*spectra.shape[1])
        flags = [] if row['status'] == 'dominant' else row['status'].split(';')
        if row['near_resolution_limit']:
            flags.append('near_resolution_limit')
        if row['family'] == 'Unresolved':
            flags.append('section_family_unresolved')
        if geometry['diagnostics']['curved_panel_proxy']:
            flags.append('curved_panel_proxy')
        row['enhanced_flags'] = ';'.join(flags) if flags else 'none'
        row['quality_ok'] = bool(n and row['status'] == 'dominant' and not row['near_resolution_limit'])
        for rank, component in enumerate(top, 1):
            components.append(dict(mode=row['mode'], rank=rank, eigenvalue=row['eigenvalue'],
                critical_stress_MPa=row['critical_stress_MPa'], family=row['family'], **component))
    value_key = 'critical_stress_MPa' if metadata['reference_stress_MPa'] is not None else 'eigenvalue'
    envelope = sample_envelope(rows, metadata['length_mm'], value_key)
    base.write_csv(prefix+'_modes.csv', rows, [key for key in rows[0] if key != 'top_components'])
    base.write_csv(prefix+'_components.csv', components,
        ['mode', 'rank', 'eigenvalue', 'critical_stress_MPa', 'family', 'halfwaves', 'half_wavelength_mm', 'share'])
    base.write_csv(prefix+'_envelope.csv', envelope, ['halfwaves', 'half_wavelength_mm', 'sample_count',
        'minimum_value', 'minimum_mode', 'filtered_minimum_value', 'filtered_minimum_mode'])
    make_plots(prefix, metadata, rows, spectra, envelope, value_key, args.near_limit_fraction)
    write_html(prefix, metadata, rows, spectra, geometry, value_key)
    result = dict(processed_modes=len(rows), source_report=os.path.abspath(args.modal_report),
        output_prefix=prefix, family_counts={family: sum(r['family']==family for r in rows) for family in COLORS},
        classification_limitations=LIMITATION, settings=vars(args), geometry=geometry['diagnostics'],
        minimum_resolved_half_wavelength_mm=metadata['length_mm']/spectra.shape[1],
        caution_half_wavelength_mm=metadata['length_mm']/(args.near_limit_fraction*spectra.shape[1]),
        resolution_note='Four longitudinal intervals per half-wave safeguard; NOT a convergence study.',
        envelope_note='Minima of existing positive dominant-half-wave samples only; not a classical signature curve.',
        section_shape_note='Dominant harmonic coefficients, arbitrarily amplified; not physical deformation amplitude.',
        files=[prefix+suffix for suffix in ('.html', '_modes.png', '_spectra.png', '_envelope.png',
                '_modes.csv', '_components.csv', '_envelope.csv', '_report.json')])
    with open(prefix+'_report.json', 'w') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('ENHANCED COMPLETE: %d modes; %s.html' % (len(rows), prefix))
    return result


if __name__ == '__main__':
    main()
