# -*- coding: utf-8 -*-
"""Direct shape diagnostics and portable visual reports. No Abaqus solver calls.

L/D/G remain geometric proxies unless a separately validated mechanical basis
is supplied. Plot sampling never enters the numerical family calculation.
"""
import base64
import json
import math
import os
import numpy as np

COLORS = {'L': '#18a477', 'D': '#e89b32', 'G': '#4489e8', 'Assembly': '#b060c8',
          'Other': '#9b6b43', 'Mixed': '#a36ede', 'Unresolved': '#8492a5'}


def common_grid(tracks, tolerance):
    ordered = [np.argsort(t['z']) for t in tracks]
    z = np.asarray(tracks[0]['z'])[ordered[0]]
    if len(z) < 2 or np.any(np.diff(z) <= 0):
        return None
    for t, order in zip(tracks, ordered):
        zz = np.asarray(t['z'])[order]
        if len(zz) != len(z) or not np.allclose(zz, z, rtol=0, atol=tolerance):
            return None
    weights = np.zeros(len(z))
    weights[1:] += .5*np.diff(z); weights[:-1] += .5*np.diff(z)
    return dict(z=z, weights=weights/(z[-1]-z[0]), indices=np.column_stack(
        [np.asarray(t['indices'])[order] for t, order in zip(tracks, ordered)]))


def weighted_raw(values, longitudinal_weights, sqrt_section_weights):
    return (np.asarray(values).reshape(len(values), -1)*sqrt_section_weights*
            np.sqrt(longitudinal_weights)[:, None]).ravel()


def section_percentages(vector, projector):
    """Return audit order L/D/G using the physical-wall curvature-aware projector."""
    return projector.audit_shares(np.asarray(vector))


def relative_piece_basis(projector, pieces):
    # Compatibility helper.  Assembly motion is now an explicit operator/range
    # rather than part of qdist.
    return getattr(projector, 'qassembly', np.empty((len(projector.sqrtw), 0)))


def rigid_shares(vector, projector, qrelative=None):
    d = projector.component_diagnostics_weighted(np.asarray(vector))
    # Preserve the historical three diagnostic names while making their
    # semantics explicit: internal = true fold-line distortion + local plate
    # deformation, not piece-rigid assembly motion.
    g = d['global_percent']
    a = d['assembly_percent']
    # Re-normalize these three diagnostics to a common G/A/internal total.
    parts = projector._weighted_components(np.asarray(vector))
    norms = {k: float(np.sum(v*v)) for k, v in parts.items()}
    total = max(sum(norms.values()), 1e-250)
    return dict(
        whole_section_rigid_percent=100*norms['G']/total,
        relative_piece_rigid_percent=100*norms['A']/total,
        within_piece_deformation_percent=100*(norms['D']+norms['L']+norms.get('O', 0.))/total,
        local_within_ldg_percent=d['local_percent'],
        distortional_within_ldg_percent=d['distortional_percent'],
        assembly_percent=d['assembly_percent'], other_percent=d.get('other_percent', 0.),
        wall_curvature_index=d.get('wall_curvature_index'),
        component_norm_sum_over_input=d['component_norm_sum_over_input'],
        reconstruction_relative_error=d['reconstruction_relative_error'])


def sensitivity(shares, dominance):
    a = np.asarray(shares)
    labels = ['LDG'[int(np.argmax(row))] if max(row) >= 100*dominance else 'Mixed' for row in a]
    return dict(min_percent=a.min(axis=0).tolist(), max_percent=a.max(axis=0).tolist(),
                max_range_pp=float(np.max(np.ptp(a, axis=0))), labels=labels)


def mode_preview(u, tracks, transverse, origin, length):
    # A real ODB station at the largest transverse nodal displacement is included
    # even if it is not among the 25 evenly spaced visual slices.
    best_track = max(tracks, key=lambda t: float(np.max(np.linalg.norm(u[t['indices']][:, transverse], axis=1))))
    best_values = u[best_track['indices']][:, transverse]
    peak_z = float(best_track['z'][int(np.argmax(np.linalg.norm(best_values, axis=1)))])
    z = np.unique(np.r_[np.linspace(origin, origin+length, 25), peak_z])
    sections = np.empty((len(z), len(tracks), 2))
    for j, t in enumerate(tracks):
        order = np.argsort(t['z']); values = u[t['indices']][:, transverse][order]
        for k in range(2):
            sections[:, j, k] = np.interp(z, np.asarray(t['z'])[order], values[:, k])
    order = np.argsort(best_track['z'])
    return dict(z_mm=z.tolist(), peak_z_mm=peak_z, peak_index=int(np.argmin(abs(z-peak_z))),
                sections=sections, track_z_mm=np.asarray(best_track['z'])[order].tolist(),
                track_u=best_values[order].tolist(),
                convention='25 visual slices + peak station; linear interpolation on each ODB track. Numeric shares use all eligible nodes.')


def write_critical_stress_wavelength(output_dir, summary):
    """Publication-style fixed-length modal scatter/envelopes.

    This is intentionally NOT called a classical signature curve.  Each point
    is an Abaqus eigenmode sample at the member's fixed length; the x coordinate
    is the dominant longitudinal half-wavelength extracted from that mode.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in summary['modes'] if r.get('half_wavelength_mm') and
            r['half_wavelength_mm'] > 0 and
            (r.get('stress_MPa') if summary.get('sigma_ref_MPa') else r.get('eigenvalue')) is not None]
    if not rows:
        return []
    value = 'stress_MPa' if summary.get('sigma_ref_MPa') else 'eigenvalue'
    ylabel = 'Critical stress (MPa)' if value == 'stress_MPa' else 'Eigenvalue multiplier'

    fig, ax = plt.subplots(figsize=(15, 7.5))
    markers = {'L': 'o', 'D': '^', 'G': 's', 'Assembly': 'D', 'Other': 'P', 'Mixed': 'v', 'Unresolved': 'x'}
    labels = {'L': 'Abaqus modes: Local', 'D': 'Abaqus modes: Distortional',
              'G': 'Abaqus modes: Global', 'Assembly': 'Abaqus modes: Assembly-like',
              'Other': 'Abaqus modes: Other/extension-like',
              'Mixed': 'Abaqus modes: Mixed', 'Unresolved': 'Abaqus modes: Unresolved'}
    for family in ('L', 'D', 'G', 'Assembly', 'Other', 'Mixed', 'Unresolved'):
        subset = [r for r in rows if r['family'] == family]
        if subset:
            ax.scatter([r['half_wavelength_mm'] for r in subset], [r[value] for r in subset],
                       s=30, marker=markers[family], facecolors='none' if family != 'Unresolved' else COLORS[family],
                       edgecolors=COLORS[family], linewidths=.9, alpha=.9, label=labels[family])

    grouped = {}
    for r in rows:
        key = r.get('dominant_halfwaves')
        if key is None:
            key = round(math.log(max(r['half_wavelength_mm'], 1e-30)), 6)
        grouped.setdefault(key, []).append(r)
    envelope = [min(group, key=lambda r: r[value]) for unused, group in
                sorted(grouped.items(), key=lambda kv: min(x['half_wavelength_mm'] for x in kv[1]), reverse=True)]
    envelope.sort(key=lambda r: r['half_wavelength_mm'])
    ax.plot([r['half_wavelength_mm'] for r in envelope], [r[value] for r in envelope],
            color='#e87500', lw=1.8, label='Abaqus: lowest computed mode per half-wave')

    if value == 'stress_MPa':
        for ref in summary.get('curve_reference', []):
            points = ref.get('points', [])
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], lw=1.4,
                        label=ref.get('label', 'reference curve'))

    # Family minima are shown without implying that a geometric proxy is a
    # mechanically validated DSM Fcr.
    for family in ('L', 'D', 'G'):
        subset = [r for r in rows if r['family'] == family]
        if subset:
            r = min(subset, key=lambda x: x[value])
            ax.scatter([r['half_wavelength_mm']], [r[value]], s=55, color=COLORS[family], zorder=5)
            ax.annotate('%s min %.1f @ %.0f' % (family, r[value], r['half_wavelength_mm']),
                        (r['half_wavelength_mm'], r[value]), xytext=(6, -14),
                        textcoords='offset points', fontsize=8, color=COLORS[family])

    panel = (summary.get('proxy_geometry') or {}).get('maximum_panel_arc_length')
    if panel and panel > 0:
        ax.axvline(panel, color='#777777', ls=':', lw=.9,
                   label='max detected panel arc %.0f mm' % panel)

    ax.set_xscale('log')
    ax.set_xlabel('Dominant half-wavelength (mm)')
    ax.set_ylabel(ylabel)
    ax.set_title('Critical stress vs dominant half-wavelength | fixed-length Abaqus modal samples')
    ax.grid(True, which='both', alpha=.22)
    ax.legend(loc='best', fontsize=8)
    fig.text(.06, .015,
             'L/D/G are physical-wall curvature/fold-motion screening labels unless a validated mechanical basis is supplied. '
             'Assembly-like and extension-like Other motion are not forced into Distortional. This is not a classical CUFSM signature curve.',
             fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    paths = []
    for ext in ('png', 'pdf'):
        path = os.path.join(output_dir, 'critical_stress_vs_half_wavelength.'+ext)
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)

    # Second view: one lower envelope per family, visually analogous to an L/D/G
    # signature plot but explicitly based on available fixed-length modes.
    fig, ax = plt.subplots(figsize=(14, 7.5))
    for family in ('L', 'D', 'G'):
        subset = [r for r in rows if r['family'] == family]
        if not subset:
            continue
        fam_groups = {}
        for r in subset:
            key = r.get('dominant_halfwaves')
            if key is None:
                key = round(math.log(max(r['half_wavelength_mm'], 1e-30)), 6)
            fam_groups.setdefault(key, []).append(r)
        env = [min(g, key=lambda r: r[value]) for g in fam_groups.values()]
        env.sort(key=lambda r: r['half_wavelength_mm'])
        ax.plot([r['half_wavelength_mm'] for r in env], [r[value] for r in env],
                marker='o', ms=3, lw=1.4, color=COLORS[family],
                label={'L':'L (local proxy)','D':'D (distortional proxy)','G':'G (global proxy)'}[family])
    if value == 'stress_MPa':
        for ref in summary.get('curve_reference', []):
            points = ref.get('points', [])
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], ls='--', lw=1.1,
                        label=ref.get('label', 'reference curve'))
    ax.set_xscale('log')
    if all(r[value] > 0 for r in rows):
        ax.set_yscale('log')
    ax.set_xlabel('Dominant half-wavelength (mm)')
    ax.set_ylabel(ylabel)
    ax.set_title('Family lower envelopes from available Abaqus modes (not a classical signature curve)')
    ax.grid(True, which='both', alpha=.22)
    ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        path = os.path.join(output_dir, 'family_half_wavelength_envelopes.'+ext)
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths



def write_visuals(output_dir, summary, previews, geometry, spectra):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summary['modes']; x = [r['mode'] for r in rows]
    write_critical_stress_wavelength(output_dir, summary)
    proxy = not bool(summary['basis_metadata'])
    title = 'Geometric screening proxies (not validated L/D/G or energy shares)' if proxy else 'Supplied mechanical basis | see validation gates'
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    a = np.array([r['percentages'] for r in rows]); bottom = np.zeros(len(rows))
    for j, f in enumerate('LDG'):
        axes[0, 0].bar(x, a[:, j], bottom=bottom, width=1, color=COLORS[f], label=f)
        bottom += a[:, j]
    axes[0, 0].set(xlabel='Mode', ylabel='Norm share (%)', ylim=(0, 100), title='Mode composition'); axes[0, 0].legend(ncol=3)
    for f in COLORS:
        selected = [r for r in rows if r['family'] == f]
        axes[0, 1].scatter([r['half_wavelength_mm'] for r in selected],
            [r['stress_MPa'] if r['stress_MPa'] is not None else r['eigenvalue'] for r in selected],
            label=f, color=COLORS[f], s=17, alpha=.8)
    axes[0, 1].set(xscale='log', xlabel='Dominant half-wavelength (mm)',
        ylabel='Critical stress (MPa)' if summary['sigma_ref_MPa'] else 'Eigenvalue multiplier',
        title='Fixed-length modal samples, not a classical signature curve'); axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(x, [100*r['spectral_fit_error'] for r in rows], color='#b65157', label='Sine fit error (%)')
    axes[1, 0].plot(x, [(r.get('sensitivity') or {}).get('max_range_pp', np.nan) for r in rows], color='#9470c2', label='Wall-definition sensitivity (pp)')
    if any(r.get('raw_vs_fitted_max_pp') is not None for r in rows):
        axes[1, 0].plot(x, [r.get('raw_vs_fitted_max_pp', np.nan) for r in rows], color='#267e98', label='Raw vs fitted shares (pp)')
    axes[1, 0].plot(x, [(r.get('rigid_diagnostics') or {}).get('assembly_percent', np.nan) for r in rows],
                    color='#b060c8', alpha=.75, label='Assembly share (%)')
    axes[1, 0].plot(x, [(r.get('rigid_diagnostics') or {}).get('other_percent', np.nan) for r in rows],
                    color='#9b6b43', alpha=.75, label='Other/extension share (%)')
    axes[1, 0].set(xlabel='Mode', title='Quality + excluded non-DSM kinematic shares'); axes[1, 0].legend(fontsize=8)
    im = axes[1, 1].imshow(np.asarray(spectra).T, aspect='auto', origin='lower',
        extent=[.5, len(rows)+.5, .5, len(spectra[0])+.5], cmap='magma', vmin=0, vmax=1)
    axes[1, 1].set(xlabel='Mode index in report', ylabel='Number of half-waves', title='Resolved sine-spectrum share')
    fig.colorbar(im, ax=axes[1, 1], label='Coefficient norm share')
    fig.suptitle(title, fontsize=13); fig.tight_layout(rect=[0, 0, 1, .97])
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, 'modal_dashboard.'+ext), dpi=170)
    plt.close(fig)
    # Six actual mode-section views, selected for variation rather than inferred purity.
    chosen = list(dict.fromkeys([0, min(1, len(rows)-1)]+[
        next((i for i, r in enumerate(rows) if r['family'] == f), 0) for f in ('L', 'D', 'G', 'Assembly', 'Other', 'Mixed', 'Unresolved')]))
    for i in np.argsort([(r.get('raw_vs_fitted_max_pp') or 0) for r in rows])[::-1]:
        if len(chosen) >= min(6, len(rows)): break
        if int(i) not in chosen: chosen.append(int(i))
    fig, axs = plt.subplots(2, 3, figsize=(15, 9)); xy = np.asarray(geometry['xy']); edges = geometry['edges']
    width = max(np.ptp(xy, axis=0)); piece_names = list(dict.fromkeys(geometry['pieces']))
    palette = ['#237ec0', '#dc7757', '#42a281', '#9b75bd']
    for ax, i in zip(axs.flat, chosen[:6]):
        p = previews[i]; deformation = p['sections'][p['peak_index']]
        factor = .18*width/max(np.max(np.linalg.norm(deformation, axis=1)), 1e-30)
        moved = xy+factor*deformation
        for n, m in edges:
            ax.plot(xy[[n, m], 0], xy[[n, m], 1], color='#c9cdd3', lw=1)
            ax.plot(moved[[n, m], 0], moved[[n, m], 1], color=palette[piece_names.index(geometry['pieces'][n]) % 4], lw=1.5)
        ax.set_aspect('equal'); ax.set_title('Mode %s | %s | z=%.1f mm\nL/D/G=%.1f / %.1f / %.1f %%' %
            (rows[i]['mode'], rows[i]['family'], p['peak_z_mm'], *rows[i]['percentages']), fontsize=10)
        ax.set_xlabel('Section coordinate 1 (mm)'); ax.set_ylabel('Section coordinate 2 (mm)')
    for ax in list(axs.flat)[len(chosen):]: ax.set_visible(False)
    fig.suptitle('Peak-displacement sections | gray: original | colors: pieces | exaggerated independently')
    fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(os.path.join(output_dir, 'mode_sections.png'), dpi=170); plt.close(fig)
    payload = dict(title=title, geometry=geometry, modes=rows, spectra=np.asarray(spectra).tolist(), previews=[])
    for p in previews:
        packed = dict(p)
        packed['shape'] = list(p['sections'].shape)
        packed['sections'] = base64.b64encode(np.asarray(p['sections'], dtype='<f4').tobytes()).decode('ascii')
        payload['previews'].append(packed)
    content = HTML.replace('__DATA__', json.dumps(payload, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c'))
    with open(os.path.join(output_dir, 'modal_explorer.html'), 'w', encoding='utf-8') as stream:
        stream.write(content)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Abaqus modal explorer</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0c1421;color:#e6edf7;font:15px system-ui}header,main{max-width:1480px;margin:auto;padding:24px}h1{margin:0;font-size:28px}p{color:#a8b8ce;line-height:1.5}.tag{color:#efb460} .tools{display:flex;gap:16px;flex-wrap:wrap;align-items:center;position:sticky;top:0;background:#152137;padding:14px;z-index:2;border-radius:12px}select,input,button{font:inherit}button,select{background:#243653;color:#fff;border:1px solid #455574;padding:7px;border-radius:6px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}.card{background:#131f32;border:1px solid #293951;border-radius:12px;padding:18px;min-width:0}h2{font-size:17px;margin:0 0 12px}canvas{display:block;width:100%;height:330px}.bar{display:flex;height:32px;border-radius:5px;overflow:hidden;margin:10px 0}.bar span{text-align:center;line-height:32px;color:#091323;font-weight:700}pre{white-space:pre-wrap;font:12px/1.6 monospace;color:#b8c9df}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:8px;border-bottom:1px solid #2b3b51}tr{cursor:pointer}tr:hover{background:#23344e}.scroll{max-height:330px;overflow:auto}a{color:#81b9ff}@media(max-width:850px){.grid{grid-template-columns:1fr}header,main{padding:12px}}
</style><header><h1>Abaqus · Modal explorer</h1><p id="subtitle"></p><p class="tag">L / D / G are basis-dependent norm shares, not percentages of buckling load or energy. Shape amplitudes are arbitrary eigenvector normalization.</p></header>
<main><div class="tools"><label>Mode <select id="mode"></select></label><button id="prev">←</button><button id="next">→</button><button id="play">Animate shape</button><label>Visual slice <input id="slice" type="range" min="0" max="25" value="0"></label><button id="peak">Peak section</button><label>Display amplitude <input id="amp" type="range" min="0" max="40" value="18"></label></div>
<div class="grid"><section class="card"><h2 id="heading"></h2><div id="shares" class="bar"></div><p id="metrics"></p><pre id="quality"></pre></section><section class="card"><h2 id="sectionTitle">Cross-section</h2><canvas id="section"></canvas><p>Gray: undeformed. Colors: individual pieces. Visual slices interpolate nodal U along each track; the peak station is always included. Classification uses full numerical data.</p></section>
<section class="card"><h2>Sampled buckling map · click to select a mode</h2><canvas id="curve"></canvas><p>Fixed member length; points are modal samples, not a continuous classical signature curve. No family absence is inferred.</p></section><section class="card"><h2>Mode composition triangle · click to select</h2><canvas id="triangle"></canvas><p>Vertices: L, D, G. Nearby eigenvalues may rotate individual eigenvectors: consult cluster results in modal_audit.json.</p></section>
<section class="card"><h2>Longitudinal displacement · strongest track</h2><canvas id="profile"></canvas><p>Both transverse components at original ODB stations; arbitrary modal amplitude.</p></section><section class="card"><h2>Resolved half-wave spectrum</h2><canvas id="spectrum"></canvas><p>Number of half-waves on the horizontal axis. Spectral fit error is independent of the direct section classification.</p></section>
<section class="card" style="grid-column:1/-1"><h2>All modes · select a row</h2><div class="scroll"><table><thead><tr><th>Mode</th><th>Family</th><th>Eigenvalue</th><th>Half-wave mm</th><th>L / D / G %</th><th>Fit error %</th><th>Sensitivity pp</th></tr></thead><tbody id="rows"></tbody></table></div></section></div><p><a href="modal_dashboard.pdf">Printable dashboard PDF</a> · <a href="critical_stress_vs_half_wavelength.pdf">Stress–half-wavelength PDF</a> · <a href="family_half_wavelength_envelopes.pdf">Family envelopes PDF</a> · <a href="modal_percentages.csv">Full CSV</a> · <a href="modal_audit.json">Audit + clusters + DSM evidence</a> · <a href="eigenspace_validation.html">Eigenspace bounds and validation</a></p></main>
<script type="application/json" id="data">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),colors=['#18a477','#e89b32','#4489e8'],pieceColors=['#55a9ff','#f1976a','#65d2ad','#be98ef'];let idx=0,phase=1,playing=false,start=0;const hit={};
const fmt=(n,d=2)=>n===null||n===undefined?'unavailable':Number(n).toFixed(d);
$('subtitle').textContent=D.title+' · '+D.modes.length+' modes · offline / no external libraries';
D.modes.forEach((r,i)=>{const opt=document.createElement('option');opt.value=i;opt.textContent=`${r.mode} · ${r.family}`;$('mode').append(opt);const tr=document.createElement('tr');[r.mode,r.family,fmt(r.eigenvalue,5),fmt(r.half_wavelength_mm),r.percentages.map(x=>fmt(x,1)).join(' / '),fmt(100*r.spectral_fit_error),fmt(r.sensitivity?.max_range_pp)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});tr.onclick=()=>select(i);$('rows').append(tr)});
function setup(id){const canvas=$(id),w=canvas.clientWidth,h=330,dpr=window.devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.font='12px system-ui';c.fillStyle='#b8c9df';return{c,w,h}}
function line(c,points,color,width=1){c.strokeStyle=color;c.lineWidth=width;c.beginPath();points.forEach((p,i)=>i?c.lineTo(...p):c.moveTo(...p));c.stroke()}
function decode(p){if(p.decoded)return p.decoded;const s=atob(p.sections),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);p.decoded=new Float32Array(b.buffer);return p.decoded}
function section(){const{c,w,h}=setup('section'),g=D.geometry,p=D.previews[idx],n=g.xy.length,k=+$('slice').value,u=decode(p),base=k*n*2;let xmax=-Infinity,xmin=Infinity,ymax=-Infinity,ymin=Infinity,umax=0;g.xy.forEach(([x,y],i)=>{xmax=Math.max(xmax,x);xmin=Math.min(xmin,x);ymax=Math.max(ymax,y);ymin=Math.min(ymin,y);umax=Math.max(umax,Math.hypot(u[base+2*i],u[base+2*i+1]))});const width=Math.max(xmax-xmin,ymax-ymin),a=width*(+$('amp').value/100)/Math.max(umax,1e-30)*phase,scale=Math.min((w-60)/(xmax-xmin+width*.8),(h-45)/(ymax-ymin+width*.8)),cx=(xmax+xmin)/2,cy=(ymax+ymin)/2,point=(i,deformed)=>[(g.xy[i][0]+(deformed?a*u[base+2*i]:0)-cx)*scale+w/2,h/2-(g.xy[i][1]+(deformed?a*u[base+2*i+1]:0)-cy)*scale];g.edges.forEach(([i,j])=>{line(c,[point(i,false),point(j,false)],'#536073');line(c,[point(i,true),point(j,true)],pieceColors[[...new Set(g.pieces)].indexOf(g.pieces[i])%4],2)});$('sectionTitle').textContent=`Mode ${D.modes[idx].mode} · section at z=${fmt(p.z_mm[k],1)} mm`}
function scatter(id,tri){const{c,w,h}=setup(id);let pts=[];if(tri){const vertices=[[45,h-38],[w-45,h-38],[w/2,32]];line(c,[...vertices,vertices[0]],'#516984');['L','D','G'].forEach((v,i)=>{c.fillStyle=colors[i];c.fillText(v,vertices[i][0]-4,vertices[i][1]+(i===2?-12:24))});pts=D.modes.map(r=>[0,1].map(j=>r.percentages.reduce((a,v,i)=>a+v/100*vertices[i][j],0)))}else{let xx=D.modes.map(r=>Math.log10(r.half_wavelength_mm||1)),yy=D.modes.map(r=>r.stress_MPa??r.eigenvalue),xmin=Math.min(...xx),xmax=Math.max(...xx),ymin=Math.min(...yy),ymax=Math.max(...yy);pts=xx.map((x,i)=>[50+(x-xmin)/Math.max(xmax-xmin,1e-9)*(w-75),h-40-(yy[i]-ymin)/Math.max(ymax-ymin,1e-9)*(h-75)]);line(c,[[50,30],[50,h-40],[w-25,h-40]],'#516984');c.fillStyle='#b8c9df';c.fillText('log half-wavelength (mm)',w/2-60,h-6);c.fillText(`${fmt(10**xmin,0)} → ${fmt(10**xmax,0)}`,50,h-20);c.fillText((D.modes[0].stress_MPa===null?'Eigenvalue ':'MPa ')+fmt(ymin)+' – '+fmt(ymax),55,18)}pts.forEach((p,i)=>{const r=D.modes[i];c.fillStyle=colors[r.percentages.indexOf(Math.max(...r.percentages))];c.globalAlpha=i===idx?1:.55;c.beginPath();c.arc(...p,i===idx?6:3,0,2*Math.PI);c.fill();if(i===idx){c.strokeStyle='#fff';c.lineWidth=2;c.stroke()}});c.globalAlpha=1;hit[id]=pts}
function traces(id,xx,series){const{c,w,h}=setup(id);let vals=series.flat(),lo=Math.min(0,...vals),hi=Math.max(...vals),xmin=Math.min(...xx),xmax=Math.max(...xx);const point=(x,y)=>[48+(x-xmin)/Math.max(xmax-xmin,1e-30)*(w-70),h-38-(y-lo)/Math.max(hi-lo,1e-30)*(h-65)];line(c,[[48,20],[48,h-38],[w-20,h-38]],'#516984');series.forEach((s,k)=>line(c,s.map((v,i)=>point(xx[i],v)),pieceColors[k],1.7));c.fillText(fmt(xmin,1),48,h-15);c.fillText(fmt(xmax,1),w-75,h-15);c.fillText(fmt(hi,3),1,28);c.fillText(fmt(lo,3),1,h-36)}
function select(i){idx=(i+D.modes.length)%D.modes.length;$('mode').value=idx;const r=D.modes[idx],p=D.previews[idx];$('slice').max=p.z_mm.length-1;$('slice').value=p.peak_index;$('heading').textContent=`Mode ${r.mode} · ${r.family} · cluster ${r.cluster_id}`;$('shares').replaceChildren();r.percentages.forEach((v,j)=>{const s=document.createElement('span');s.style.width=v+'%';s.style.background=colors[j];s.textContent=v>8?'LDG'[j]+' '+fmt(v,1)+'%':'';$('shares').append(s)});$('metrics').textContent=`Eigenvalue ${fmt(r.eigenvalue,6)} · stress ${fmt(r.stress_MPa)} MPa · half-wave ${fmt(r.half_wavelength_mm)} mm`;$('quality').textContent=`Metric: ${r.percentage_kind}\nFinal eigenspace family: ${r.eigenspace_stable_family??"not evaluated"}\nSine fit error: ${fmt(100*r.spectral_fit_error)}% · wall-definition sensitivity: ${fmt(r.sensitivity?.max_range_pp)} pp\nRelative piece rigid motion: ${fmt(r.rigid_diagnostics?.relative_piece_rigid_percent)}% · Other: ${fmt(r.rigid_diagnostics?.other_percent)}% · Wall curvature index: ${fmt(r.rigid_diagnostics?.wall_curvature_index,3)}\nL/D/G energy: ${r.energy_status}\n${r.flags.join('\n')}`;draw()}
function draw(){section();scatter('curve',false);scatter('triangle',true);const p=D.previews[idx];traces('profile',p.track_z_mm,[p.track_u.map(u=>u[0]),p.track_u.map(u=>u[1])]);traces('spectrum',D.spectra[idx].map((_,i)=>i+1),[D.spectra[idx]])}
$('mode').onchange=()=>select(+$('mode').value);$('prev').onclick=()=>select(idx-1);$('next').onclick=()=>select(idx+1);$('slice').oninput=section;$('amp').oninput=section;$('peak').onclick=()=>{$('slice').value=D.previews[idx].peak_index;section()};['curve','triangle'].forEach(id=>$(id).onclick=e=>{const r=$(id).getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=0,d=Infinity;hit[id].forEach((p,i)=>{let dd=(p[0]-x)**2+(p[1]-y)**2;if(dd<d){d=dd;best=i}});select(best)});$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Animate shape';if(playing){start=performance.now();requestAnimationFrame(tick)}else{phase=1;section()}};function tick(t){if(!playing)return;phase=Math.cos((t-start)/650);section();requestAnimationFrame(tick)}window.addEventListener('resize',draw);select(0);
</script></html>'''
