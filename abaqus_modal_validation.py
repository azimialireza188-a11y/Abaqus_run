# -*- coding: utf-8 -*-
"""Eigenspace bounds, force-based algebra, and mesh-to-mesh shape validation.

No solver is launched. Geometric L/D/G bounds remain geometric screening.
Mesh comparison uses global U1/U2/U3 on common surface quadrature, never mode
numbers, display samples, or interpolated eigenvalues. It does not certify DSM.
"""
import argparse
import base64
import csv
import html
import hashlib
import json
import os
import numpy as np

FAMILIES = ('L', 'D', 'G')


def spectral_isolation(groups, tolerance):
    """A bounded-width group is not isolated if its neighbor is still close.

    Keep bounded-width grouping (avoids long chains spanning distinct physics)
    but never certify across an unresolved group boundary.
    """
    result = []
    for i, group in enumerate(groups):
        lo, hi = group[0]['eigenvalue'], group[-1]['eigenvalue']
        lower = ((lo-groups[i-1][-1]['eigenvalue'])/max(abs(lo), 1e-30)) if i else None
        upper = ((groups[i+1][0]['eigenvalue']-hi)/max(abs(hi), 1e-30)) if i+1 < len(groups) else None
        unresolved = (lower is not None and lower <= tolerance) or (upper is not None and upper <= tolerance)
        result.append(dict(lower_relative_gap=lower, upper_relative_gap=upper,
            unresolved_neighbor=bool(unresolved), upper_boundary_open=upper is None,
            isolated=bool(not unresolved and upper is not None)))
    return result


def gram_whitener(gram):
    diag = np.diag(gram)
    if not np.all(np.isfinite(gram)) or np.any(diag <= 0):
        raise ValueError('Zero/nonfinite cached mode norm')
    scales = np.sqrt(diag)
    correlation = gram/scales[:, None]/scales[None, :]
    eig, q = np.linalg.eigh(.5*(correlation+correlation.T))
    if eig[0] <= 1e-10*eig[-1]: raise ValueError('Rank-deficient cached eigenspace')
    return q/np.sqrt(eig)/scales[:, None]


def physics_keyword_signature(inp_path):
    """Extra check beyond geometry hashes; not a node-set equivalence proof."""
    names = {'BOUNDARY', 'MPC', 'EQUATION', 'COUPLING', 'KINEMATIC', 'DISTRIBUTING',
        'RIGID BODY', 'CONNECTOR SECTION', 'CONNECTOR BEHAVIOR', 'MATERIAL', 'ELASTIC',
        'PLASTIC', 'SHELL SECTION', 'SHELL GENERAL SECTION', 'ORIENTATION', 'TRANSFORM',
        'FRICTION', 'SURFACE INTERACTION', 'SURFACE BEHAVIOR', 'INITIAL CONDITIONS',
        'CLOAD', 'DLOAD', 'DSLOAD', 'TEMPERATURE', 'TIE', 'SPRING', 'DASHPOT'}
    selected, active = [], False
    with open(inp_path) as stream:
        for line in stream:
            line = line.strip().upper()
            if not line or line.startswith('**'): continue
            if line.startswith('*'):
                keyword = line[1:].split(',')[0].strip()
                if keyword in ('INCLUDE', 'PARAMETER'):
                    raise ValueError('External/parameterized INP requires explicit expanded input for comparison')
                active = keyword in names or keyword.startswith('CONTACT') or keyword.startswith('CONNECTOR ')
            if active: selected.append(','.join(x.strip() for x in line.split(',')))
    if not selected: raise ValueError('No physical INP keyword evidence')
    return hashlib.sha256('\n'.join(selected).encode('utf-8')).hexdigest()


def component_bounds(parts, dominance=.9):
    """Exact extrema of self-norm shares over ALL combinations in a subspace.

    Each part is (component DOFs, observed subspace dimension). Individual
    extrema are independently attainable; their maxima do not sum to 100%.
    The denominator is the sum of component self norms (also for nonorthogonal
    mechanical bases); it is NOT elastic energy or necessarily the input norm.
    """
    grams = [np.asarray(a).T @ np.asarray(a) for a in parts]
    h = sum(grams)
    h = .5*(h+h.T)
    ev = np.linalg.eigvalsh(h)
    if not len(ev) or ev[-1] <= 0 or ev[0] <= 1e-12*ev[-1]:
        raise ValueError('Rank-deficient observed component subspace')
    chol = np.linalg.cholesky(h)
    eigen = []
    for g in grams:
        left = np.linalg.solve(chol, g)
        whitened = np.linalg.solve(chol, left.T).T
        eigen.append(np.clip(np.linalg.eigvalsh(.5*(whitened+whitened.T)), 0., 1.))
    low = [float(100*x[0]) for x in eigen]
    high = [float(100*x[-1]) for x in eigen]
    means = [float(100*np.mean(x)) for x in eigen]
    stable = next((f for f, x in zip(FAMILIES, low) if x >= 100*dominance-1e-9), 'Mixed')
    return dict(min_percent=low, max_percent=high, mean_percent=means,
                stable_family=stable, dimension=len(ev))


class ForceProjector:
    """Exact fcFSM-style force split on a SUPPLIED restrained elastic space.

    L = ker(J.T), GD = range(K^-1 J), D = K^-1 J ker(E), G = GD minus D
    in the K inner product. The caller must map J, E, K through the SAME
    constraints and document wall/force semantics. This class does not infer
    physical wall definitions, contact state, or a valid basis from geometry.
    Dense/reduced spaces only; explicitly limited to prevent dense FE matrices.
    """
    def __init__(self, stiffness, wall_loads, equilibrium, weights=None):
        if stiffness.shape[0] > 5000:
            raise ValueError('Dense force projector limited to 5000 retained DOFs; supply a validated reduced space')
        k = np.asarray(stiffness, dtype=float)
        j = np.asarray(wall_loads, dtype=float)
        e = np.asarray(equilibrium, dtype=float)
        if (k.ndim != 2 or k.shape[0] != k.shape[1] or j.ndim != 2 or j.shape[0] != len(k)
                or e.ndim != 2 or e.shape[1] != j.shape[1] or not j.shape[1]
                or not all(np.all(np.isfinite(x)) for x in (k, j, e))):
            raise ValueError('Invalid K/J/E dimensions or nonfinite entries')
        if not np.allclose(k, k.T, rtol=1e-10, atol=np.max(abs(k))*1e-12):
            raise ValueError('Elastic K must be symmetric')
        self.k = .5*(k+k.T)
        try:
            np.linalg.cholesky(self.k)
        except np.linalg.LinAlgError as exc:
            raise ValueError('Elastic K must be positive definite after constraints') from exc
        scales = np.linalg.norm(j, axis=0)
        if np.any(scales == 0):
            raise ValueError('Zero wall-load direction after constraints')
        self.j = j/scales
        e = e/scales
        self.b = np.linalg.solve(self.k, self.j)
        self.s = self.j.T @ self.b
        try:
            np.linalg.cholesky(.5*(self.s+self.s.T))
        except np.linalg.LinAlgError as exc:
            raise ValueError('Dependent wall loads after constraints') from exc
        # Row scaling preserves equilibrium while avoiding force/moment units.
        norms = np.linalg.norm(e, axis=1)
        e = e[norms > 0]/norms[norms > 0, None]
        if len(e):
            unused, singular, vt = np.linalg.svd(e, full_matrices=True)
            rank = int(np.sum(singular > 1e-10*singular[0]))
            self.nd = vt[rank:].T
        else:
            self.nd = np.eye(self.j.shape[1])
        self.df = self.nd.T @ self.s @ self.nd if self.nd.shape[1] else None
        self.weights = np.ones(len(k)) if weights is None else np.asarray(weights, dtype=float)
        if self.weights.shape != (len(k),) or np.any(self.weights <= 0) or not np.all(np.isfinite(self.weights)):
            raise ValueError('Positive metric weights required')

    def project(self, displacement, subspace=False):
        x = np.asarray(displacement, dtype=float).reshape(len(self.k), -1)
        if subspace:
            from abaqus_dsm_modal_audit import orth
            x = orth(x*np.sqrt(self.weights[:, None]))/np.sqrt(self.weights[:, None])
        rhs = self.j.T @ x
        a = np.linalg.solve(self.s, rhs)
        gd = self.b @ a
        d = self.b @ self.nd @ np.linalg.solve(self.df, self.nd.T @ rhs) if self.df is not None else np.zeros_like(x)
        parts = dict(L=x-gd, D=d, G=gd-d)
        energies = np.array([.5*np.sum(p*(self.k@p)) for p in parts.values()])
        total = .5*float(np.sum(x*(self.k@x)))
        if not np.isfinite(total) or total <= 0:
            raise ValueError('Nonpositive modal elastic energy')
        norms = np.array([np.sum(self.weights[:, None]*p*p) for p in parts.values()])
        return dict(components=parts, residual=np.zeros_like(x), relative_residual=0.,
            percentages=(100*norms/norms.sum()).tolist(), energy_percent=(100*energies/total).tolist(),
            cross_relative=float(abs(energies.sum()-total)/total),
            displacement_cross_percent=float(100*(np.sum(self.weights[:, None]*x*x)-norms.sum())/
                                               np.sum(self.weights[:, None]*x*x)),
            component_norm_sum_over_input=float(norms.sum()/np.sum(self.weights[:, None]*x*x)),
            condition=float(np.linalg.cond(self.s)))


def proxy_cluster(vectors, projector, dominance=.9, max_assembly_percent=25., max_other_percent=25.):
    """Bounds for L/D/G plus explicit Assembly and Other kinematic gates."""
    from abaqus_dsm_modal_audit import orth
    q = orth(vectors)
    if q.shape[1] != vectors.shape[1]:
        raise ValueError('Observed transverse eigenvectors are rank deficient')
    component_columns = [[], [], [], [], []]  # L, D, G, A, O
    for j in range(q.shape[1]):
        parts = projector.audit_components(q[:, j])
        for k, part in enumerate(parts):
            component_columns[k].append(np.asarray(part).reshape(-1))
    parts = [np.column_stack(cols) for cols in component_columns]
    grams = [p.T @ p for p in parts]

    hall0 = sum(grams)
    hall = .5*(hall0+hall0.T)
    ev = np.linalg.eigvalsh(hall)
    if not len(ev) or ev[-1] <= 0 or ev[0] <= 1e-12*ev[-1]:
        raise ValueError('Rank-deficient observed proxy component space')
    chol = np.linalg.cholesky(hall)

    def share_eigs(index):
        left = np.linalg.solve(chol, grams[index])
        whitened = np.linalg.solve(chol, left.T).T
        eig = np.clip(np.linalg.eigvalsh(.5*(whitened+whitened.T)), 0., 1.)
        return float(100*eig[0]), float(100*eig[-1]), float(100*np.mean(eig))

    amin, amax, amean = share_eigs(3)
    omin, omax, omean = share_eigs(4)

    # Pure non-DSM spaces have no useful L/D/G denominator. Report them
    # directly rather than allowing numerical noise to create a false family.
    if amin >= max_assembly_percent or omin >= max_other_percent:
        z = [0.0, 0.0, 0.0]
        stable = 'Assembly' if amin >= max_assembly_percent else 'Other'
        return dict(min_percent=z[:], max_percent=z[:], mean_percent=z[:],
                    stable_family=stable, dimension=q.shape[1],
                    assembly_min_percent=amin, assembly_max_percent=amax,
                    assembly_mean_percent=amean, other_min_percent=omin,
                    other_max_percent=omax, other_mean_percent=omean)

    result = component_bounds(parts[:3], dominance)
    result.update(assembly_min_percent=amin, assembly_max_percent=amax,
                  assembly_mean_percent=amean, other_min_percent=omin,
                  other_max_percent=omax, other_mean_percent=omean)
    if amax >= max_assembly_percent or omax >= max_other_percent:
        result['stable_family'] = 'Mixed'
    return result


def ordered_path(xy, edges):
    adjacency = {i: [] for i in range(len(xy))}
    for a, b in edges:
        adjacency[a].append(b); adjacency[b].append(a)
    ends = [i for i, neighbors in adjacency.items() if len(neighbors) == 1]
    if len(ends) != 2 or any(len(n) not in (1, 2) for n in adjacency.values()):
        raise ValueError('Mesh matching requires an unbranched open section per piece')
    current = min(ends, key=lambda i: tuple(xy[i]))
    order = [current]
    while True:
        remaining = [i for i in adjacency[current] if len(order) < 2 or i != order[-2]]
        if not remaining: break
        current = remaining[0]
        if current in order: raise ValueError('Cycle in section path')
        order.append(current)
    if len(order) != len(xy): raise ValueError('Disconnected section path')
    return order


def reference_paths(source_directory):
    path = os.path.join(source_directory, 'builtup_segments.csv')
    records = np.loadtxt(path, delimiter=',', ndmin=2)
    result = {}
    for piece in np.unique(records[:, 0]):
        points, lookup, edges = [], {}, []
        for row in records[records[:, 0] == piece]:
            edge = []
            for xy in (row[1:3], row[3:5]):
                key = tuple(xy)
                if key not in lookup:
                    lookup[key] = len(points); points.append(xy)
                edge.append(lookup[key])
            edges.append(edge)
        points = np.asarray(points)
        points = points[ordered_path(points, edges)]
        result['P%d' % int(piece)] = points
    return result


def reference_coordinates(xy, reference):
    """Monotone arc-length pullback to the SAME source section, not chord length.

    Curved walls have different chord lengths on coarse/fine meshes. Nodes
    must lie on the reference polyline, within ODB coordinate precision.
    """
    delta = np.diff(reference, axis=0); lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 0): raise ValueError('Zero reference segment')
    cumulative = np.r_[0., np.cumsum(lengths)]
    offsets = np.asarray(xy)[:, None, :]-reference[None, :-1, :]
    fractions = np.clip(np.sum(offsets*delta, axis=2)/(lengths**2), 0, 1)
    projection = reference[None, :-1, :]+fractions[:, :, None]*delta
    distances = np.linalg.norm(np.asarray(xy)[:, None, :]-projection, axis=2)
    nearest = distances.argmin(axis=1); index = np.arange(len(xy))
    tolerance = 1e-6*max(1., cumulative[-1], float(np.max(abs(reference))))
    if np.max(distances[index, nearest]) > tolerance:
        raise ValueError('Mesh nodes do not lie on the supplied canonical section')
    s = cumulative[nearest]+fractions[index, nearest]*lengths[nearest]
    if abs(s[0]) > tolerance or abs(s[-1]-cumulative[-1]) > tolerance:
        raise ValueError('Section mesh omits a reference endpoint')
    s[0], s[-1] = 0., cumulative[-1]
    if np.any(np.diff(s) <= tolerance*.01): raise ValueError('Nonmonotone/ambiguous section-to-reference mapping')
    return s, cumulative


def export_shapes(output_dir, geo, grid, displacements, summary, build):
    """Archive all real nodal U, not the reduced display preview."""
    if grid is None:
        return dict(status='UNAVAILABLE_NONALIGNED_STATIONS')
    names = sorted(set(t['instance'] for t in geo['tracks']))
    arrays = {}
    reference = reference_paths(build['source_inputs']['source_directory']) if geo['axis'] == 2 else {}
    for i, name in enumerate(names):
        idx = [j for j, t in enumerate(geo['tracks']) if t['instance'] == name]
        reverse = {old: new for new, old in enumerate(idx)}
        edges = [(reverse[a], reverse[b]) for a, b in geo['edges'] if a in reverse and b in reverse]
        order = np.array(idx)[ordered_path(geo['xy'][idx], edges)]
        xy = geo['xy'][order]
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        if name in reference:
            s, rs = reference_coordinates(xy, reference[name])
            arrays['p%d_reference_s' % i] = rs
            arrays['p%d_reference_xy' % i] = reference[name]
        arrays['p%d_s' % i] = s
        arrays['p%d_xy' % i] = xy
        arrays['p%d_u' % i] = np.asarray(displacements)[:, grid['indices'][:, order], :]
    provenance = dict(schema_version=1, names=names, source_odb_sha256=summary['source_odb_sha256'],
        model_signature=summary['model_signature'], mesh_nodes=summary['mesh_nodes'],
        axis=geo['axis'], transverse=list(geo['transverse']),
        sigma_ref_MPa=summary['sigma_ref_MPa'], contact=build.get('contact'),
        constraint_equivalence='Model signature + physical surface check; detailed BC/MPC equivalence still requires review',
        metric='Integral of global U1/U2/U3 squared over shell midsurface; no rotations/energy',
        source_audit=os.path.join(output_dir, 'modal_audit.json'))
    provenance['physics_keywords_signature'] = physics_keyword_signature(os.path.splitext(summary['source_odb'])[0]+'.inp')
    provenance['section_mapping'] = ('Canonical source-polyline arclength; linear nodal-U pullback from faceted shells'
                                     if reference else 'Observed straight shell segments; identical surfaces required')
    arrays.update(metadata=json.dumps(provenance), z=grid['z'],
        modes=np.array([r['mode'] for r in summary['modes']]), eigenvalues=np.array([r['eigenvalue'] for r in summary['modes']]))
    path = os.path.join(output_dir, 'modal_shapes.npz')
    np.savez_compressed(path, **arrays)
    return dict(status='AVAILABLE', path=path, metric=provenance['metric'])


def gauss_union(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if min(len(a), len(b)) < 2 or any(np.any(np.diff(x) <= 0) for x in (a, b)):
        raise ValueError('Strictly increasing physical grids required')
    if not np.allclose(a[[0, -1]], b[[0, -1]], rtol=0, atol=1e-7*max(1., np.ptp(a))):
        raise ValueError('Meshes do not cover the same physical interval')
    # Snap endpoints only within the established tolerance, never extrapolate.
    b = b.copy(); b[0], b[-1] = a[0], a[-1]
    breaks = np.unique(np.r_[a, b])
    half = .5*np.diff(breaks); mid = .5*(breaks[1:]+breaks[:-1])
    return (mid[:, None]+half[:, None]*np.array([-1., 1.])/np.sqrt(3.)).ravel(), np.repeat(half, 2)


def sample_piece(u, s, z, qs, qz):
    s, z, qs, qz = [np.asarray(a, dtype=float) for a in (s, z, qs, qz)]
    if (min(qs) < s[0]-1e-7 or max(qs) > s[-1]+1e-7 or
            min(qz) < z[0]-1e-7 or max(qz) > z[-1]+1e-7):
        raise ValueError('Extrapolation beyond the observed mesh is forbidden')
    ii = np.clip(np.searchsorted(s, qs)-1, 0, len(s)-2)
    jj = np.clip(np.searchsorted(z, qz)-1, 0, len(z)-2)
    fs = (qs-s[ii])/(s[ii+1]-s[ii]); fz = (qz-z[jj])/(z[jj+1]-z[jj])
    along = u[:, :, ii, :]*(1-fs)[None, None, :, None]+u[:, :, ii+1, :]*fs[None, None, :, None]
    return along[:, jj, :, :]*(1-fz)[None, :, None, None]+along[:, jj+1, :, :]*fz[None, :, None, None]


def cross_mass(a, b):
    """Exact 1-D consistent cross-mass for two piecewise-linear grids."""
    from scipy.sparse import csr_matrix
    q, w = gauss_union(a, b)
    def interpolation(grid):
        grid = np.asarray(grid)
        indices = np.clip(np.searchsorted(grid, q)-1, 0, len(grid)-2)
        fraction = (q-grid[indices])/(grid[indices+1]-grid[indices])
        return csr_matrix((np.column_stack((1-fraction, fraction)).ravel(),
            (np.repeat(np.arange(len(q)), 2), np.column_stack((indices, indices+1)).ravel())), shape=(len(q), len(grid)))
    ia, ib = interpolation(a), interpolation(b)
    return (ia.T @ ib.multiply(w[:, None])).tocsr()


def surface_gram(ua, ub, sa, sb, za, zb):
    # Kronecker-separable integration equals full union Gauss sampling but
    # avoids materializing a quadrature grid for every mode/chunk.
    ms, mz = cross_mass(sa, sb), cross_mass(za, zb)
    ns, nz, nb = len(sb), len(zb), len(ub)
    along_s = (ms @ ub.transpose(2, 0, 1, 3).reshape(ns, -1)).reshape(len(sa), nb, nz, 3).transpose(2, 1, 0, 3)
    mapped = (mz @ along_s.reshape(nz, -1)).reshape(len(za), nb, len(sa), 3).transpose(1, 0, 2, 3)
    return ua.reshape(len(ua), -1) @ mapped.reshape(nb, -1).T


def subspace_match(a, b):
    from abaqus_dsm_modal_audit import orth
    qa, qb = orth(a), orth(b)
    if qa.shape[1] != a.shape[1] or qb.shape[1] != b.shape[1]:
        raise ValueError('Rank-deficient shape observations')
    c = np.clip(np.linalg.svd(qa.T@qb, compute_uv=False), 0., 1.)**2
    return dict(dimension_a=qa.shape[1], dimension_b=qb.shape[1], same_dimension=qa.shape[1] == qb.shape[1],
        minimum_cosine_squared=float(c.min()), mean_cosine_squared=float(c.mean()),
        principal_angles_deg=np.degrees(np.arccos(np.sqrt(c))).tolist(),
        coverage_a=float(c.sum()/qa.shape[1]), coverage_b=float(c.sum()/qb.shape[1]))


def compare_shapes(a_path, b_path, output_dir, cluster_tolerance=.001, eigen_tolerance=.05, shape_threshold=.95):
    from abaqus_dsm_modal_audit import close_clusters
    from scipy.optimize import linear_sum_assignment
    with np.load(a_path, allow_pickle=False) as fa, np.load(b_path, allow_pickle=False) as fb:
        a = {k: fa[k] for k in fa.files}; b = {k: fb[k] for k in fb.files}
    ma, mb = [json.loads(str(d['metadata'].item())) for d in (a, b)]
    if (not ma.get('model_signature') or ma['model_signature'] != mb.get('model_signature') or
            ma['names'] != mb['names'] or ma['axis'] != mb['axis'] or ma['transverse'] != mb['transverse'] or
            ma['sigma_ref_MPa'] != mb['sigma_ref_MPa'] or not ma.get('physics_keywords_signature') or
            ma.get('physics_keywords_signature') != mb.get('physics_keywords_signature')):
        raise ValueError('Different physical model signatures, axes, pieces or reference stress')
    gauss_union(a['z'], b['z'])  # Verify identical coverage before integration.
    na, nb = len(a['modes']), len(b['modes'])
    gaa, gbb, gab = np.zeros((na, na)), np.zeros((nb, nb)), np.zeros((na, nb))
    max_geometry_error = 0.
    for i, name in enumerate(ma['names']):
        key = 'p%d_' % i
        sa, sb = a[key+'s'], b[key+'s']
        qs, ws = gauss_union(sa, sb)
        breaks = np.unique(np.r_[sa, sb])
        xya = np.column_stack([np.interp(breaks, sa, a[key+'xy'][:, j]) for j in range(2)])
        xyb = np.column_stack([np.interp(breaks, sb, b[key+'xy'][:, j]) for j in range(2)])
        error = float(np.max(np.linalg.norm(xya-xyb, axis=1)))
        max_geometry_error = max(max_geometry_error, error)
        reference_key = key+'reference_xy'
        if reference_key in a and reference_key in b:
            if a[reference_key].shape != b[reference_key].shape or not np.array_equal(a[reference_key], b[reference_key]):
                raise ValueError('Different canonical section geometry on piece '+name)
            reference_coordinates(a[key+'xy'], a[reference_key])
            reference_coordinates(b[key+'xy'], b[reference_key])
        elif error > 1e-6*max(1., sa[-1]):
            raise ValueError('Different midsurface geometry without a common reference on piece '+name)
        ua, ub = a[key+'u'], b[key+'u']
        gaa += surface_gram(ua, ua, sa, sa, a['z'], a['z'])
        gbb += surface_gram(ub, ub, sb, sb, b['z'], b['z'])
        gab += surface_gram(ua, ub, sa, sb, a['z'], b['z'])
        print('Mesh comparison: integrated piece %s (%d/%d)' % (name, i+1, len(ma['names'])), flush=True)
    groups = [close_clusters([dict(mode=int(m), eigenvalue=float(e), index=i) for i, (m, e) in
              enumerate(zip(d['modes'], d['eigenvalues']))], cluster_tolerance) for d in (a, b)]
    isolation = [spectral_isolation(g, cluster_tolerance) for g in groups]
    wa = [gram_whitener(gaa[np.ix_([r['index'] for r in group], [r['index'] for r in group])]) for group in groups[0]]
    wb = [gram_whitener(gbb[np.ix_([r['index'] for r in group], [r['index'] for r in group])]) for group in groups[1]]
    scores = np.zeros((len(wa), len(wb))); pairs = {}
    for i, ga in enumerate(groups[0]):
        for j, gb in enumerate(groups[1]):
            cross = wa[i].T@gab[np.ix_([r['index'] for r in ga], [r['index'] for r in gb])]@wb[j]
            c = np.clip(np.linalg.svd(cross, compute_uv=False), 0., 1.)**2
            score = float(c.sum()/max(len(ga), len(gb)))
            scores[i, j] = score
            pairs[i, j] = c
    ai, bi = linear_sum_assignment(-scores)
    matches = []
    for i, j in zip(ai, bi):
        ga, gb = groups[0][i], groups[1][j]; c = pairs[i, j]
        equal = len(ga) == len(gb)
        ea, eb = [np.sort([r['eigenvalue'] for r in g]) for g in (ga, gb)]
        difference = float(np.max(abs(ea-eb)/np.maximum(abs(eb), 1e-30))) if equal else None
        tail = any(r['index'] == len(d['modes'])-1 for g, d in ((ga, a), (gb, b)) for r in g)
        reasons = []
        if not equal: reasons.append('SUBSPACE_DIMENSION_MISMATCH')
        if float(c.min()) < shape_threshold: reasons.append('SHAPE_MATCH_BELOW_THRESHOLD')
        if difference is None or difference > eigen_tolerance: reasons.append('EIGENVALUES_NOT_CONVERGED')
        if tail: reasons.append('UPPER_SPECTRUM_BOUNDARY_NOT_CLOSED')
        if isolation[0][i]['unresolved_neighbor'] or isolation[1][j]['unresolved_neighbor']:
            reasons.append('ADJACENT_CLUSTER_GAP_NOT_RESOLVED')
        if ma['mesh_nodes'] == mb['mesh_nodes']: reasons.append('NOT_A_DIFFERENT_MESH_RESOLUTION')
        matches.append(dict(modes_a=[r['mode'] for r in ga], modes_b=[r['mode'] for r in gb],
            minimum_cosine_squared=float(c.min()), principal_angles_deg=np.degrees(np.arccos(np.sqrt(c))).tolist(),
            coverage_a=float(c.sum()/len(ga)), coverage_b=float(c.sum()/len(gb)),
            relative_eigenvalue_difference=difference, numerical_checks_pass=not reasons, reasons=reasons))
    report = dict(schema_version=1, source_a=ma, source_b=mb, matches=matches,
        unmatched_a=[[r['mode'] for r in g] for i, g in enumerate(groups[0]) if i not in ai],
        unmatched_b=[[r['mode'] for r in g] for j, g in enumerate(groups[1]) if j not in bi],
        max_surface_difference_mm=max_geometry_error, model_signature=ma['model_signature'],
        numerical_pass_count=sum(r['numerical_checks_pass'] for r in matches),
        tolerances=dict(cluster_relative=cluster_tolerance, eigen_relative=eigen_tolerance, minimum_cosine_squared=shape_threshold),
        convention='Two-point Gauss integration of bilinear nodal-U pullbacks on union reference-arclength/longitudinal grids',
        limitations=['Translation-only shape convergence; UR/stress/energy convergence not tested.',
            'Family identification, BC/MPC/contact equivalence and DSM applicability still require independent evidence.',
            'Two meshes establish one comparison, not asymptotic convergence; use successive refinements.',
            'On curved sections the metric uses canonical source arclength. Faceted geometry differences are reported separately.',
            'Different cluster dimensions are not silently merged. Supply a documented cluster tolerance if necessary.'])
    os.makedirs(output_dir, exist_ok=False)
    np.savez_compressed(os.path.join(output_dir, 'modal_grams.npz'), aa=gaa, bb=gbb, ab=gab,
                        source_a=json.dumps(ma), source_b=json.dumps(mb))
    with open(os.path.join(output_dir, 'mesh_comparison.json'), 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    write_mesh_plot(output_dir, scores, report)
    write_mesh_html(output_dir, report)
    return report


def write_mesh_html(output_dir, report):
    with open(os.path.join(output_dir, 'mesh_comparison.png'), 'rb') as stream:
        encoded = base64.b64encode(stream.read()).decode('ascii')
    rows = ''.join('<tr><td>%s</td><td>%s</td><td>%.4f</td><td>%s</td><td>%s</td></tr>' %
        (', '.join(map(str, r['modes_a'])), ', '.join(map(str, r['modes_b'])), r['minimum_cosine_squared'],
         ('%.2f%%' % (100*r['relative_eigenvalue_difference'])) if r['relative_eigenvalue_difference'] is not None else 'different dimensions',
         html.escape('; '.join(r['reasons'])) or 'numerical checks passed') for r in report['matches'])
    text = '''<!doctype html><meta charset="utf-8"><title>Mesh shape comparison</title>
<style>body{font:15px/1.6 system-ui;max-width:1400px;margin:25px auto;padding:20px;background:#101b2c;color:#e8eef8}img{width:100%}td,th{padding:8px;border-bottom:1px solid #435169;text-align:left}a{color:#83c9ff}</style>
<h1>Mesh comparison: eigenvalues AND mode subspaces</h1>
<p>Candidate assignments: __COUNT__. Passing all numerical gates: <b>__PASS__</b>. A candidate assignment with failed shape gates is NOT a matched physical mode. Differences in cluster dimensions and unresolved neighboring gaps remain explicit.</p>
<p>Maximum difference between faceted section curves on the common source-arclength map: __GEOMETRY__ mm. The comparison uses all three global translations. It does not certify mechanical L/D/G families or DSM inputs.</p>
<img src="data:image/png;base64,__IMAGE__"><p><a href="mesh_comparison.json">Full evidence and limitations</a> · <a href="../eigenspace_validation.html">Eigenspace family bounds</a></p>
<table><tr><th>Coarse modes</th><th>Fine candidate modes</th><th>Min cos² angle</th><th>Eigenvalue difference</th><th>Decision reasons</th></tr>__ROWS__</table>'''
    for key, value in dict(COUNT=len(report['matches']), PASS=report['numerical_pass_count'],
                           GEOMETRY='%.6g' % report['max_surface_difference_mm'], IMAGE=encoded, ROWS=rows).items():
        text = text.replace('__'+key+'__', str(value))
    with open(os.path.join(output_dir, 'mesh_comparison.html'), 'w', encoding='utf-8') as stream: stream.write(text)


def write_mesh_plot(output_dir, scores, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(scores, origin='lower', aspect='auto', vmin=0, vmax=1, cmap='viridis')
    axes[0].set(xlabel='Mesh B cluster index', ylabel='Mesh A cluster index', title='Rotation-invariant subspace overlap')
    fig.colorbar(im, ax=axes[0], label='Overlap / larger dimension')
    for r in report['matches']:
        if r['relative_eigenvalue_difference'] is not None:
            axes[1].scatter(100*r['relative_eigenvalue_difference'], r['minimum_cosine_squared'],
                c='#169c78' if r['numerical_checks_pass'] else '#cc6655', s=22)
    axes[1].axhline(report['tolerances']['minimum_cosine_squared'], color='gray', ls='--')
    axes[1].axvline(100*report['tolerances']['eigen_relative'], color='gray', ls='--')
    axes[1].set(xlabel='Maximum cluster eigenvalue difference (%)', ylabel='Minimum cos²(principal angle)',
                title='Shape + eigenvalue convergence (not DSM certification)', ylim=(0, 1.02))
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, 'mesh_comparison.png'), dpi=160); plt.close(fig)


def write_cluster_report(output_dir, summary):
    clusters = summary.get('clusters', [])
    usable = [r for r in clusters if r.get('bounds')]
    path = os.path.join(output_dir, 'eigenspace_bounds.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cluster', 'modes', 'eigen_min', 'eigen_max', 'stable_family', 'spectrum_boundary',
                         'L_min', 'L_max', 'D_min', 'D_max', 'G_min', 'G_max', 'A_min', 'A_max', 'O_min', 'O_max', 'angle_sensitive',
                         'spectral_isolated', 'unresolved_neighbor', 'bound_error', 'mathematical_bound_family'])
        for r in usable:
            b = r['bounds']
            writer.writerow([r['cluster_id'], ';'.join(map(str, r['modes'])), *r['eigenvalue_range'],
                r['family'], r.get('spectrum_boundary', False),
                *[x for i in range(3) for x in (b['min_percent'][i], b['max_percent'][i])],
                b.get('assembly_min_percent'), b.get('assembly_max_percent'),
                b.get('other_min_percent'), b.get('other_max_percent'), r.get('angle_sensitive'),
                r.get('spectral_isolation', {}).get('isolated'),
                r.get('spectral_isolation', {}).get('unresolved_neighbor'), r.get('bound_error'), b['stable_family']])
    if not usable: return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_aux = any('assembly_min_percent' in r['bounds'] or 'other_min_percent' in r['bounds'] for r in usable)
    count = 5 if has_aux else 3
    fig, axes = plt.subplots(count, 1, figsize=(14, 14 if has_aux else 10), sharex=True)
    x = [r['modes'][0] for r in usable]
    for i, (ax, color) in enumerate(zip(axes[:3], ('#18a477', '#e89b32', '#4489e8'))):
        low = [r['bounds']['min_percent'][i] for r in usable]
        high = [r['bounds']['max_percent'][i] for r in usable]
        ax.vlines(x, low, high, color=color, lw=2)
        ax.scatter(x, [r['bounds']['mean_percent'][i] for r in usable], c=color, s=10)
        ax.axhline(90, color='#777777', ls='--', lw=.7)
        ax.set(ylabel=FAMILIES[i]+' norm share (%)', ylim=(-2, 102))
    if has_aux:
        for ax, prefix, color, ylabel, threshold in (
                (axes[3], 'assembly', '#b060c8', 'A assembly share (%)', summary['settings'].get('max_assembly_percent', 25.)),
                (axes[4], 'other', '#9b6b43', 'O extension share (%)', summary['settings'].get('max_other_percent', 25.))):
            low = [r['bounds'].get(prefix+'_min_percent', np.nan) for r in usable]
            high = [r['bounds'].get(prefix+'_max_percent', np.nan) for r in usable]
            mean = [r['bounds'].get(prefix+'_mean_percent', np.nan) for r in usable]
            ax.vlines(x, low, high, color=color, lw=2)
            ax.scatter(x, mean, c=color, s=10)
            ax.axhline(threshold, color='#777777', ls='--', lw=.7)
            ax.set(ylabel=ylabel, ylim=(-2, 102))
    axes[-1].set_xlabel('First mode of eigenvalue cluster; bars: attainable min/max, dots: mean')
    fig.suptitle('Eigenspace participation bounds | geometric screening unless validated mechanical basis supplied\n'
                 'All combinations in each cluster; near-distinct combinations are not individual eigenmodes')
    fig.tight_layout(rect=(0, 0, 1, .95))
    png = os.path.join(output_dir, 'eigenspace_bounds.png'); fig.savefig(png, dpi=160); plt.close(fig)
    with open(png, 'rb') as stream: encoded = base64.b64encode(stream.read()).decode('ascii')
    table = ''.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' %
        (html.escape(', '.join(map(str, r['modes']))), r['family'],
         ' / '.join('%.2f–%.2f' % p for p in zip(r['bounds']['min_percent'], r['bounds']['max_percent']))+
         ((' / A %.2f–%.2f' % (r['bounds']['assembly_min_percent'], r['bounds']['assembly_max_percent']))
          if 'assembly_min_percent' in r['bounds'] else '')+
         ((' / O %.2f–%.2f' % (r['bounds']['other_min_percent'], r['bounds']['other_max_percent']))
          if 'other_min_percent' in r['bounds'] else ''),
         'YES' if r.get('spectrum_boundary') else 'no',
         html.escape('; '.join(s for s in (
             'unresolved neighboring spectral gap' if r.get('spectral_isolation', {}).get('unresolved_neighbor') else '',
             'wall-definition sensitive' if r.get('angle_sensitive') else '', r.get('bound_error') or '',
             'residual limit exceeded' if r['bounds'].get('maximum_relative_residual', 0) > summary['settings']['max_residual'] else '') if s))) for r in usable)
    text = '''<!doctype html><meta charset="utf-8"><title>Eigenspace validation</title>
<style>body{font:16px/1.6 system-ui;max-width:1250px;margin:30px auto;padding:20px;background:#101b2c;color:#e8eef8}img{width:100%}td,th{padding:8px;border-bottom:1px solid #435169;text-align:left}a{color:#83c9ff}table{width:100%}</style>
<h1>Modal eigenspace validation</h1><p>Intervals replace unjustified certainty from one arbitrary eigenvector. A family is stable only if its minimum share meets the dominance threshold. Extrema are independently attainable, not additive. Near-distinct modes describe a subspace, not a new exact eigenmode.</p>
<p>These are geometric bounds unless a documented mechanical basis was supplied. They do not establish Fcrl, Fcrd or Fcre. The highest extracted cluster has no observed upper spectral gap and remains open.</p>
<img src="data:image/png;base64,__IMAGE__"><p><a href="eigenspace_bounds.csv">CSV intervals</a> · <a href="modal_audit.json">Full audit</a> · <a href="modal_explorer.html">Shape explorer</a></p>
<table><tr><th>Modes</th><th>Final family</th><th>L / D / G / A / O ranges %</th><th>Open upper boundary</th><th>Additional limits</th></tr>__TABLE__</table>'''.replace('__IMAGE__', encoded).replace('__TABLE__', table)
    with open(os.path.join(output_dir, 'eigenspace_validation.html'), 'w', encoding='utf-8') as stream: stream.write(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh-a', required=True, help='modal_shapes.npz from first audit')
    p.add_argument('--mesh-b', required=True, help='modal_shapes.npz from another mesh')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--cluster-tolerance', type=float, default=.001)
    p.add_argument('--eigen-tolerance', type=float, default=.05)
    p.add_argument('--shape-threshold', type=float, default=.95)
    a = p.parse_args()
    if not 0 < a.shape_threshold <= 1 or a.cluster_tolerance < 0 or a.eigen_tolerance <= 0:
        p.error('Invalid comparison tolerances')
    r = compare_shapes(a.mesh_a, a.mesh_b, a.output_dir, a.cluster_tolerance, a.eigen_tolerance, a.shape_threshold)
    print('Matched clusters: %d; numerical checks passed: %d. See limitations in mesh_comparison.json.' %
          (len(r['matches']), r['numerical_pass_count']))


if __name__ == '__main__':
    main()
