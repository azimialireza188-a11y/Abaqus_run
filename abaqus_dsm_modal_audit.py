# -*- coding: utf-8 -*-
"""Read-only audit of Abaqus buckling families; mixed-mode percentages and DSM gates.

abaqus python abaqus_dsm_modal_audit.py --run-dir "completed step-3 run"

Default: recompute the existing geometric projection, explicitly a SCREENING
PROXY. No cFSM/GBT basis or elastic stiffness is manufactured from those labels.
--basis accepts a mapped mechanical basis NPZ (see README_dsm_modal_audit.md).
An optional compatible elastic K in that pack enables energy partition with all
signed cross terms. Without K, energy is unavailable, never inferred from U.
--compare/--reference/--review provide separate convergence/benchmark/review
evidence. Accepted Fcr values remain null when any required evidence is missing.
Every available mode is read; no job/model/input is created or submitted.
"""
import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import sys
import numpy as np

FAMILIES = ('L', 'D', 'G')
HERE = os.path.dirname(os.path.abspath(globals().get('__file__', sys._getframe().f_code.co_filename)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # The CAE pipeline changes cwd to the run directory.
import abaqus_modal_visuals as visuals
import abaqus_modal_validation as validation


def orth(a, tol=1e-10):
    a = np.asarray(a, dtype=float)
    norms = np.linalg.norm(a, axis=0)
    a = a[:, norms > 0]
    if not a.shape[1]:
        return np.empty((a.shape[0], 0))
    u, s, unused = np.linalg.svd(a/np.linalg.norm(a, axis=0), full_matrices=False)
    return u[:, s > tol*s[0]]


class MechanicalProjector:
    """Joint least squares; no order-dependent cross-family orthogonalization.

    Percentages normalize the separate reconstructed component squared norms.
    Nonorthogonal cross terms are reported, so this is not an energy partition.
    Each family's internal representation may be rescaled or changed freely.
    """
    def __init__(self, bases, weights, max_condition=1e6):
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError('Basis metric weights must be positive finite scalars')
        self.sqrtw = np.sqrt(w)
        matrices, self.slices, start = [], {}, 0
        for family in FAMILIES:
            a = np.asarray(bases[family], dtype=float)
            if a.ndim != 2 or a.shape[0] != len(w) or not np.all(np.isfinite(a)):
                raise ValueError('Invalid basis matrix '+family)
            q = orth(a*self.sqrtw[:, None])
            if q.shape[1] == 0:
                raise ValueError('Empty family basis: '+family)
            matrices.append(q)
            self.slices[family] = slice(start, start+q.shape[1]); start += q.shape[1]
        self.b = np.column_stack(matrices)
        u, s, vt = np.linalg.svd(self.b, full_matrices=False)
        if len(s) < self.b.shape[1] or s[-1] <= s[0]/max_condition:
            raise ValueError('Family subspaces overlap or are ill-conditioned; percentages are not identifiable')
        self.condition = float(s[0]/s[-1])
        self.inverse = (vt.T/s) @ u.T

    def project(self, displacement, subspace=False):
        x = np.asarray(displacement, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] != len(self.sqrtw) or not np.all(np.isfinite(x)):
            raise ValueError('Invalid mode vector dimensions/values')
        y = x*self.sqrtw[:, None]
        if subspace:
            y = orth(y)
        total = float(np.sum(y*y))
        if total <= 1e-250:
            raise ValueError('Zero mode in the observed degrees of freedom')
        coefficients = self.inverse @ y
        pieces = {f: self.b[:, self.slices[f]] @ coefficients[self.slices[f]] for f in FAMILIES}
        reconstruction = sum(pieces.values())
        residual = y-reconstruction
        norms = np.array([np.sum(pieces[f]**2) for f in FAMILIES])
        denominator = float(norms.sum())
        percentages = (100*norms/denominator).tolist() if denominator > 1e-250 else [0., 0., 0.]
        return dict(percentages=percentages, relative_residual=float(np.linalg.norm(residual)/math.sqrt(total)),
            displacement_cross_percent=float(100*(np.sum(reconstruction**2)-norms.sum())/total),
            component_norm_sum_over_input=float(denominator/total), condition=self.condition,
            components={f: pieces[f]/self.sqrtw[:, None] for f in FAMILIES},
            residual=residual/self.sqrtw[:, None])


def energy_partition(parts, stiffness):
    """E_f=1/2 uf'Kuf; E_fg=uf'Kug. Cross terms can be NEGATIVE.

    PSD is checked on the observed component span, not claimed for all of K.
    Percentages are independent of the common eigenvector normalization.
    """
    labels = list(parts)
    vectors = [np.asarray(parts[name], dtype=float).reshape(stiffness.shape[0], -1) for name in labels]
    block = np.column_stack(vectors)
    gram = block.T @ (stiffness @ block)
    scale = max(float(np.max(np.abs(gram))), 1e-250)
    if not np.all(np.isfinite(gram)) or not np.allclose(gram, gram.T, rtol=1e-8, atol=scale*1e-10):
        raise ValueError('Stiffness is not symmetric on the observed component subspace')
    if np.linalg.eigvalsh(.5*(gram+gram.T))[0] < -1e-8*scale:
        raise ValueError('Elastic energy is indefinite on the observed component subspace')
    total_vector = sum(vectors)
    total = .5*float(np.sum(total_vector*(stiffness @ total_vector)))
    if not math.isfinite(total) or total <= 1e-14*scale:
        raise ValueError('Zero/nonpositive total elastic modal energy')
    diagonal = {name: .5*float(np.sum(v*(stiffness @ v))) for name, v in zip(labels, vectors)}
    cross = {labels[i]+':'+labels[j]: float(np.sum(vectors[i]*(stiffness @ vectors[j])))
             for i, j in itertools.combinations(range(len(labels)), 2)}
    closure = abs(sum(diagonal.values())+sum(cross.values())-total)/total
    return dict(total_normalized_mode_energy=total,
        diagonal_percent={k: 100*v/total for k, v in diagonal.items()},
        cross_percent={k: 100*v/total for k, v in cross.items()}, relative_closure_error=closure,
        convention='Elastic strain energy; includes residual R and signed cross terms; not absolute excitation energy')


def classify(result, dominance, max_error):
    if result['relative_residual'] > max_error or sum(result['percentages']) < 99.:
        return 'Unresolved'
    k = int(np.argmax(result['percentages']))
    return FAMILIES[k] if result['percentages'][k] >= 100*dominance else 'Mixed'


def candidate_for(family, rows):
    candidates = [r for r in rows if r['family'] == family and r.get('mechanical_eligible')
                  and r.get('stress_MPa') is not None and r['stress_MPa'] > 0]
    if not candidates:
        return None
    row = min(candidates, key=lambda r: r['stress_MPa'])
    return {key: row[key] for key in ('mode', 'eigenvalue', 'stress_MPa', 'family') if key in row}


def assess_family(candidate, comparison, reference, review, signature, setup_ok, tolerance, shape_match=None):
    reasons = []
    if not candidate:
        reasons.append('no_resolved_mechanical_candidate')
    if not signature:
        reasons.append('physical_model_signature_unavailable')
    if not setup_ok:
        reasons.append('elastic_buckle_setup_or_stress_scale_unverified')
    if not shape_match:
        reasons.append('quantitative_mesh_eigenspace_match_missing_or_failed')
    mesh_error = reference_error = None
    if not comparison:
        reasons.append('mesh_comparison_missing')
    elif (not comparison.get('compatible') or not comparison.get('candidate') or
          not positive_number(comparison['candidate'].get('stress_MPa'))):
        reasons.append('mesh_comparison_incompatible_or_family_missing')
    elif candidate:
        value = comparison['candidate']['stress_MPa']
        mesh_error = abs(candidate['stress_MPa']-value)/value
        if mesh_error > tolerance:
            reasons.append('mesh_difference_exceeds_tolerance')
    if not reference:
        reasons.append('independent_reference_missing')
    elif (reference.get('model_signature') != signature or reference.get('conditions_equivalent') is not True
          or not reference.get('source') or not positive_number(reference.get('stress_MPa'))):
        reasons.append('independent_reference_not_documented_or_incompatible')
    elif candidate:
        reference_error = abs(candidate['stress_MPa']-reference['stress_MPa'])/reference['stress_MPa']
        if reference_error > tolerance:
            reasons.append('reference_difference_exceeds_tolerance')
    if (not review or not candidate or review.get('mode') != candidate['mode'] or
            not all(str(review.get(k, '')).strip() for k in ('shape_review', 'coverage_review', 'mesh_shape_review'))):
        reasons.append('family_shape_coverage_and_mesh_shape_reviews_required')
    return dict(candidate=candidate, accepted_stress_MPa=candidate['stress_MPa'] if not reasons else None,
        status='SUPPORTED_BY_SUPPLIED_EVIDENCE' if not reasons else 'UNCONFIRMED', reasons=reasons,
        mesh_relative_difference=mesh_error, reference_relative_difference=reference_error)


def positive_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def read_json(path):
    with open(path, encoding='utf-8-sig') as stream:
        return json.load(stream)

def read_curve_reference(path):
    """Read optional external half-wavelength/critical-stress curves for plotting only.

    Required CSV columns: label, half_wavelength_mm, critical_stress_MPa.
    These data never participate in family classification or DSM acceptance.
    """
    grouped = {}
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'label', 'half_wavelength_mm', 'critical_stress_MPa'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Curve reference CSV requires: label, half_wavelength_mm, critical_stress_MPa')
        for row in reader:
            label = str(row['label']).strip()
            try:
                x = float(row['half_wavelength_mm']); y = float(row['critical_stress_MPa'])
            except (TypeError, ValueError):
                raise ValueError('Invalid numeric value in curve reference CSV')
            if not label or not math.isfinite(x) or not math.isfinite(y) or x <= 0 or y <= 0:
                raise ValueError('Curve reference values must have a label and positive finite coordinates')
            grouped.setdefault(label, []).append((x, y))
    return [dict(label=label, points=[list(p) for p in sorted(points)])
            for label, points in sorted(grouped.items())]



def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def discover(run_dir, suffix):
    paths = [os.path.join(run_dir, n) for n in os.listdir(run_dir) if n.endswith(suffix)]
    if len(paths) != 1:
        raise ValueError('Expected one %s in %s; found %d' % (suffix, run_dir, len(paths)))
    return paths[0]


def physical_signature(build):
    source = dict(build.get('source_inputs', {}))
    directory = source.pop('source_directory', None)
    if not directory or not os.path.isdir(directory):
        return None
    paths = [os.path.join(directory, n) for n in ('builtup_segments.csv', 'builtup_seams.csv', 'builtup_bolts_y.csv')]
    if not all(os.path.isfile(p) for p in paths):
        return None
    record = dict(inputs=source, geometry_hashes=[sha_file(p) for p in paths],
                  contact=build.get('contact'), bc_count=build.get('boundary_conditions'))
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode('utf-8')).hexdigest()


def load_basis(path, odb, odb_hash, signature):
    from scipy.sparse import csr_matrix
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata'].item()))
        if metadata.get('source_odb_sha256') != odb_hash or metadata.get('model_signature') != signature:
            raise ValueError('Mechanical basis provenance does not match this ODB/model')
        if not all(metadata.get(k) for k in ('method', 'family_definition_id', 'validation_notes',
                                            'metric_definition', 'coordinate_space_review')):
            raise ValueError('Mechanical basis lacks documented family definitions, validation or coordinate/metric convention')
        keys = [(str(n), int(label), int(dof)) for n, label, dof in
                zip(data['instances'], data['labels'], data['dofs'])]
        if len({len(data[k]) for k in ('instances', 'labels', 'dofs', 'coordinates', 'weights')}) != 1:
            raise ValueError('Mapped DOF arrays have different lengths')
        if not keys or len(set(keys)) != len(keys) or any(dof not in range(1, 7) for _, _, dof in keys):
            raise ValueError('Invalid/duplicate mapped basis DOFs')
        coordinates = np.asarray(data['coordinates'], dtype=float)
        if coordinates.shape != (len(keys), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError('Invalid basis reference coordinates')
        cache = {name: {node.label: node.coordinates for node in odb.rootAssembly.instances[name].nodes}
                 for name in set(k[0] for k in keys)}
        actual = np.asarray([cache[n][label] for n, label, unused in keys])
        tolerance = max(1e-5, 1e-6*float(np.max(np.abs(actual))))
        if np.max(np.abs(actual-coordinates)) > tolerance:
            raise ValueError('Mapped basis node coordinates differ from the ODB')
        stiffness = None
        if 'K_data' in data:
            if metadata.get('stiffness_kind') != 'elastic' or not metadata.get('stiffness_source'):
                raise ValueError('Energy requires a documented elastic K, not geometric or total tangent stiffness')
            stiffness = csr_matrix((data['K_data'], data['K_indices'], data['K_indptr']),
                                   shape=(len(keys), len(keys)))
            stiffness.check_format(full_check=True)
            scale = max(float(np.max(np.abs(stiffness.data))) if stiffness.nnz else 0., 1e-250)
            delta = stiffness-stiffness.T
            if (not np.all(np.isfinite(stiffness.data)) or
                    (delta.nnz and np.max(np.abs(delta.data)) > scale*1e-9)):
                raise ValueError('K must be finite, fully assembled and symmetric in exactly the mapped DOF order')
        if metadata.get('format') == 'force_based_KJE':
            if stiffness is None or not all(metadata.get(k) for k in (
                    'wall_definition', 'equilibrium_definition', 'constraint_mapping_review', 'contact_state_review')):
                raise ValueError('Force-based split requires elastic K and documented walls/equilibrium/constraints/contact')
            if len(keys) > 5000:
                raise ValueError('Dense force-based split is limited to 5000 mapped restrained DOFs')
            projector = validation.ForceProjector(stiffness.toarray(), data['J'], data['E'], data['weights'])
            projector.sqrtw = np.sqrt(data['weights'])
        else:
            projector = MechanicalProjector({f: data[f] for f in FAMILIES}, data['weights'])
        if len(projector.sqrtw) != len(keys):
            raise ValueError('DOF map and mechanical basis sizes differ')
        return projector, keys, stiffness, metadata


def read_mapped_mode(frame, keys):
    """Read U and UR globally; no invented rotations or missing-DOF zero fill."""
    needed = {field: {} for field in ('U', 'UR')}
    for i, (name, label, dof) in enumerate(keys):
        field = 'U' if dof <= 3 else 'UR'
        needed[field].setdefault((name, label), []).append((i, (dof-1) % 3))
    result = np.full(len(keys), np.nan)
    for field, lookup in needed.items():
        if not lookup:
            continue
        if field not in frame.fieldOutputs:
            raise ValueError('Required field missing from ODB: '+field)
        for value in frame.fieldOutputs[field].values:
            if value.instance is None:
                continue
            selected = lookup.get((value.instance.name, value.nodeLabel))
            if selected:
                double = str(value.precision) == 'DOUBLE_PRECISION'
                system = value.localCoordSystemDouble if double else value.localCoordSystem
                if system is not None and len(system):
                    raise ValueError('Local-coordinate nodal output is not supported; use global U/UR')
                vector = value.dataDouble if double else value.data
                for index, component in selected:
                    result[index] = vector[component]
    if not np.all(np.isfinite(result)):
        raise ValueError('Incomplete mapped U/UR data')
    return result


def proxy_geometry(base, enhanced, odb, metadata, build):
    axis = 'xyz'.index(metadata['axis'])
    transverse = [j for j in range(3) if j != axis]
    names = [m['instance'] for m in metadata['instances']]
    tracks, mesh, keys, areas, origin, length, tolerance = base.collect_tracks(
        odb, names, axis, metadata['settings']['coord_tolerance'])
    groups, cap = base.build_groups(tracks, origin, length, metadata['max_resolved_halfwaves'])
    if cap != metadata['max_resolved_halfwaves'] or not np.isclose(length, metadata['length_mm']):
        raise ValueError('ODB mesh/length differs from the previous report')
    lookup = {key: i for i, key in enumerate(keys)}
    node_to_track = {index: i for i, track in enumerate(tracks) for index in track['indices']}
    coords = {(name, n.label): np.asarray(n.coordinates) for name in names for n in odb.rootAssembly.instances[name].nodes}
    xy = np.array([coords[keys[t['indices'][0]]][transverse] for t in tracks])
    edges = set()
    for name in names:
        for element in odb.rootAssembly.instances[name].elements:
            nodes = base.shell_corners(element)
            for a, b in zip(nodes, nodes[1:]+nodes[:1]):
                ka, kb = (name, a), (name, b)
                if ka in lookup and kb in lookup and abs(coords[ka][axis]-coords[kb][axis]) <= tolerance:
                    ia, ib = node_to_track[lookup[ka]], node_to_track[lookup[kb]]
                    if ia != ib:
                        edges.add(tuple(sorted((ia, ib))))
    physical_segments=(build.get('source_inputs') or {}).get('section_segments')
    projector = enhanced.SectionProjector(xy, sorted(edges), [t['instance'] for t in tracks],
                                           [t['weight'] for t in tracks],
                                           physical_segments=physical_segments)
    group_tracks = [[node_to_track[int(i)] for i in group['indices'][0]] for group in groups]
    return dict(axis=axis, transverse=transverse, tracks=tracks, keys=keys, lookup=lookup,
                tables=base.label_tables(keys), groups=groups, group_tracks=group_tracks,
                cap=cap, projector=projector, mesh=mesh, coords=coords, xy=xy, edges=sorted(edges),
                origin=origin, length=length, tolerance=tolerance)


def proxy_percentages(weighted_coefficients, projector, cap, subspace=False):
    """Rotation-invariant trace shares for the anchor-driven L/D/G split."""
    y = np.asarray(weighted_coefficients, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if subspace:
        y = orth(y)
    if not y.size:
        return [0., 0., 0.]
    accum = np.zeros(3)
    for j in range(y.shape[1]):
        parts = projector.audit_components(y[:, j])
        for k in range(3):
            accum[k] += float(np.sum(parts[k]**2))
    return (100*accum/max(float(accum.sum()), 1e-250)).tolist()


def close_clusters(rows, tolerance):
    groups = []
    for row in sorted(rows, key=lambda r: r['eigenvalue']):
        if (not groups or abs(row['eigenvalue']-groups[-1][0]['eigenvalue']) >
                tolerance*max(abs(groups[-1][0]['eigenvalue']), 1e-30)):
            groups.append([])
        groups[-1].append(row)
    return groups


def process(args):
    from odbAccess import openOdb
    base = load_module('audit_base', 'abaqus_modal_wavelengths.py')
    enhanced = load_module('audit_enhanced', 'abaqus_modal_report.py')
    report_path = discover(args.run_dir, '_modal_wavelengths_report.json')
    metadata, rows, spectra, prefix = enhanced.load_results(report_path)
    if metadata['processed_modes'] != metadata['available_modes']:
        raise ValueError('Regenerate the base report without --modes: this audit requires all available modes')
    build_path = discover(args.run_dir, '_build.json')
    build = read_json(build_path)
    signature = physical_signature(build)
    sigma = metadata.get('reference_stress_MPa')
    if sigma is not None and (not math.isfinite(sigma) or sigma <= 0):
        raise ValueError('Invalid reference stress in the base report')
    odb_path = os.path.join(args.run_dir, os.path.basename(metadata['odb']))
    odb_hash = sha_file(odb_path)
    if os.path.exists(os.path.splitext(odb_path)[0]+'.lck'):
        raise ValueError('ODB is locked; wait for the analysis to finish')
    inp_path = os.path.splitext(odb_path)[0]+'.inp'
    with open(inp_path) as stream:
        inp = stream.read().upper()
    import re
    setup_ok = bool(sigma is not None and metadata.get('job_completion') == 'completed'
        and len(re.findall(r'^\*STEP\b', inp, re.M)) == 1 and '*BUCKLE' in inp
        and not re.search(r'^\*(PLASTIC|IMPERFECTION|INITIAL CONDITIONS)\b', inp, re.M)
        and np.isclose(sigma, build.get('reference_stress_MPa', float('nan'))))
    odb = openOdb(path=odb_path, readOnly=True)
    basis_meta, stiffness, mechanical, mapped_keys = {}, None, None, None
    mode_vectors, results, clusters, previews, full_shapes = {}, [], [], [], []
    try:
        geo = proxy_geometry(base, enhanced, odb, metadata, build)
        if args.basis:
            mechanical, mapped_keys, stiffness, basis_meta = load_basis(args.basis, odb, odb_hash, signature)
        frames = {base.frame_eigen(f)[0]: f for f in odb.steps[metadata['step']].frames if base.frame_eigen(f)}
        if len(frames) != len(rows):
            raise ValueError('ODB and base report mode counts differ')
        common_fields = sorted(set.intersection(*(set(f.fieldOutputs.keys()) for f in frames.values())))
        proxy = geo['projector']
        pieces = [t['instance'] for t in geo['tracks']]
        grid = visuals.common_grid(geo['tracks'], geo['tolerance'])
        physical_segments=(build.get('source_inputs') or {}).get('section_segments')
        variant_options = ([dict(bend_radius_fraction=f) for f in (.03, .05)]
                           if physical_segments else [dict(wall_angle_deg=a) for a in (2., 5.)])
        variants = [enhanced.SectionProjector(geo['xy'], geo['edges'], pieces,
                    [t['weight'] for t in geo['tracks']],
                    physical_segments=physical_segments, **options)
                    for options in variant_options]
        qrelative = visuals.relative_piece_basis(proxy, pieces)
        coverage = min(m['coverage'] for m in geo['mesh'])
        proxy_layers = len(grid['z']) if grid is not None else geo['cap']
        for i, row in enumerate(rows):
            frame = frames[row['mode']]
            eigenvalue = base.frame_eigen(frame)[1]
            if not np.isclose(eigenvalue, row['eigenvalue'], rtol=1e-7, atol=1e-8):
                raise ValueError('ODB and report eigenvalues differ')
            u = base.read_displacements(frame, geo['lookup'], len(geo['keys']), geo['tables'])
            full_shapes.append(u)
            previews.append(visuals.mode_preview(u, geo['tracks'], geo['transverse'], geo['origin'], geo['length']))
            raw_vector = None
            if grid is not None:
                raw_vector = visuals.weighted_raw(u[grid['indices']][:, :, geo['transverse']], grid['weights'], proxy.sqrtw)
            direct_diagnostics = None
            sensitivity = None; raw_delta = None; fitted_percentages = None
            if mechanical:
                vector = read_mapped_mode(frame, mapped_keys)
                projected = mechanical.project(vector)
                energy_parts = dict(projected['components'], R=projected['residual'])
                energy = energy_partition(energy_parts, stiffness) if stiffness is not None else None
            else:
                coefficients = np.zeros((geo['cap'], len(geo['tracks']), 2))
                for group, indices in zip(geo['groups'], geo['group_tracks']):
                    values = u[group['indices']][:, :, geo['transverse']]
                    fitted = group['fit'].project @ values[group['fit'].order].reshape(len(values), -1)
                    coefficients[:, indices, :] = fitted.reshape(geo['cap'], len(indices), 2)
                weighted = coefficients.reshape(geo['cap'], -1)*proxy.sqrtw
                power = np.sum(weighted**2, axis=1)
                if power.sum() and not np.allclose(power/power.sum(), spectra[i], rtol=1e-5, atol=1e-6):
                    raise ValueError('ODB mode shape no longer matches saved spectrum; regenerate base report')
                fitted_percentages = proxy_percentages(weighted.ravel(), proxy, geo['cap'])
                vector = raw_vector if raw_vector is not None else weighted.ravel()
                percentages = visuals.section_percentages(vector, proxy)
                sensitivity = visuals.sensitivity([visuals.section_percentages(vector, p) for p in [variants[0], proxy, variants[1]]], args.dominance)
                raw_delta = float(np.max(np.abs(np.asarray(percentages)-fitted_percentages))) if raw_vector is not None else None
                projected = dict(percentages=percentages,
                    relative_residual=0. if raw_vector is not None else (row['relative_fit_error'] if row['relative_fit_error'] is not None else 1.),
                    displacement_cross_percent=0., component_norm_sum_over_input=1., condition=None)
                energy = None
                direct_diagnostics = visuals.rigid_shares(vector, proxy, qrelative)
                projected['component_norm_sum_over_input'] = direct_diagnostics['component_norm_sum_over_input']
                projected['displacement_cross_percent'] = 100.*(1.-projected['component_norm_sum_over_input'])
            label = classify(projected, args.dominance, args.max_residual)
            if not mechanical and direct_diagnostics:
                if direct_diagnostics.get('assembly_percent', 0.) >= args.max_assembly_percent:
                    label = 'Assembly'
                elif direct_diagnostics.get('other_percent', 0.) >= args.max_other_percent:
                    label = 'Other'
            flags = []
            if mechanical and projected['condition'] > 1e3:
                flags.append('ILL_CONDITIONED_COMPONENT_FIT')
            if not mechanical:
                flags.append('GEOMETRIC_PROXY_NOT_MECHANICAL_IDENTIFICATION')
                if proxy.metadata['curved_panel_proxy']:
                    flags.append('CURVED_PANEL_PROXY')
                if not proxy.supported:
                    label = 'Unresolved'; flags.append('UNSUPPORTED_PROXY_TOPOLOGY')
                if sensitivity['max_range_pp'] > args.max_sensitivity_pp or len(set(sensitivity['labels'])) > 1:
                    label = 'Unresolved'; flags.append('PANEL_DEFINITION_SENSITIVE')
                if direct_diagnostics and direct_diagnostics.get('assembly_percent', 0.) >= args.max_assembly_percent:
                    label = 'Assembly'; flags.append('ASSEMBLY_RIGID_MOTION_DOMINANT')
                elif direct_diagnostics and direct_diagnostics.get('other_percent', 0.) >= args.max_other_percent:
                    label = 'Other'; flags.append('OTHER_TRANSVERSE_EXTENSION_DOMINANT')
                if raw_vector is None:
                    flags.append('NONALIGNED_STATIONS_FITTED_METRIC_FALLBACK')
                if coverage < .99:
                    label = 'Unresolved'; flags.append('INCOMPLETE_TRACK_COVERAGE')
                if row['transverse_displacement_share'] < .05:
                    label = 'Unresolved'; flags.append('TRANSVERSE_PROJECTION_HAS_LOW_PARTICIPATION')
            if row['relative_fit_error'] > args.max_residual:
                flags.append('POOR_SINE_FIT_WAVELENGTH_UNCERTAIN')
            if row['dominant_halfwaves'] and row['dominant_halfwaves'] >= .8*row['max_resolved_halfwaves']:
                flags.append('WAVELENGTH_NEAR_MESH_RESOLUTION_LIMIT')
            mode_vectors[row['mode']] = vector
            result = dict(mode=row['mode'], eigenvalue=eigenvalue, stress_MPa=eigenvalue*sigma if sigma else None,
                family=label, percentages=projected['percentages'], dominant_halfwaves=row.get('dominant_halfwaves'),
                relative_residual=projected['relative_residual'], displacement_cross_percent=projected['displacement_cross_percent'],
                condition=projected['condition'], half_wavelength_mm=row['half_wavelength_mm'], flags=flags,
                spectral_fit_error=row['relative_fit_error'], dominant_spectral_share=row['dominant_share'],
                transverse_share=row['transverse_displacement_share'], fitted_percentages=fitted_percentages,
                sensitivity=sensitivity, raw_vs_fitted_max_pp=raw_delta, rigid_diagnostics=direct_diagnostics,
                mechanical_eligible=bool(mechanical and label in FAMILIES and not flags),
                energy=energy, energy_status='AVAILABLE_WITH_CROSS_TERMS' if energy else 'UNAVAILABLE_NO_COMPATIBLE_ELASTIC_K',
                percentage_kind='MAPPED_MECHANICAL_BASIS_NORM' if mechanical else (
                    'GEOMETRIC_PROXY_DIRECT_NODAL_NORM' if raw_vector is not None else 'GEOMETRIC_PROXY_COEFFICIENT_NORM'))
            results.append(result)
            if (i+1) % 50 == 0 or i+1 == len(rows):
                print('Audit: %d/%d modes' % (i+1, len(rows))); sys.stdout.flush()
        grouped = close_clusters(results, args.cluster_tolerance)
        isolation = validation.spectral_isolation(grouped, args.cluster_tolerance)
        for cluster_id, members in enumerate(grouped, 1):
            vectors = np.column_stack([mode_vectors[r['mode']] for r in members])
            if mechanical:
                projection = mechanical.project(vectors, subspace=True)
                percentages = projection['percentages']
                label = classify(projection, args.dominance, args.max_residual)
                energy = energy_partition(dict(projection['components'], R=projection['residual']), stiffness) if stiffness is not None else None
            else:
                percentages = proxy_percentages(vectors, proxy, proxy_layers, subspace=True)
                label = classify(dict(percentages=percentages, relative_residual=0.), args.dominance, args.max_residual)
                energy = None
            bounds = None; angle_sensitive = False; bound_error = None
            try:
                if mechanical:
                    components = [projection['components'][f]*mechanical.sqrtw[:, None] for f in FAMILIES]
                    if components[0].shape[1] != len(members):
                        raise ValueError('Rank-deficient observed mechanical eigenspace')
                    bounds = validation.component_bounds(components, args.dominance)
                    residual = projection['residual']*mechanical.sqrtw[:, None]
                    bounds['maximum_relative_residual'] = float(np.sqrt(max(0., np.linalg.eigvalsh(residual.T@residual)[-1])))
                else:
                    bounds = validation.proxy_cluster(vectors, proxy, args.dominance, args.max_assembly_percent, args.max_other_percent)
                    angle_bounds = [validation.proxy_cluster(vectors, p, args.dominance, args.max_assembly_percent, args.max_other_percent) for p in variants]
                    angle_sensitive = (len(set(b['stable_family'] for b in angle_bounds+[bounds])) > 1 or
                        any(np.max(np.ptp([b[k] for b in angle_bounds+[bounds]], axis=0)) > args.max_sensitivity_pp
                            for k in ('min_percent', 'max_percent')))
                label = bounds['stable_family']
                if angle_sensitive: label = 'Unresolved'
                if bounds.get('maximum_relative_residual', 0.) > args.max_residual: label = 'Unresolved'
            except ValueError as exc:
                label = 'Unresolved'; bound_error = str(exc)
            spectrum_boundary = any(r['mode'] == rows[-1]['mode'] for r in members)
            gap = isolation[cluster_id-1]
            if not gap['isolated']: label = 'Unresolved'
            clusters.append(dict(cluster_id=cluster_id, modes=[r['mode'] for r in members],
                eigenvalue_range=[min(r['eigenvalue'] for r in members), max(r['eigenvalue'] for r in members)],
                percentages=percentages, family=label, energy=energy, bounds=bounds,
                angle_sensitive=angle_sensitive, bound_error=bound_error, spectrum_boundary=spectrum_boundary,
                spectral_isolation=gap,
                convention='Trace over a weighted-orthonormal observed eigenspace; invariant to basis rotations'))
            for row in members:
                row['cluster_id'] = cluster_id
                row['eigenspace_stable_family'] = label
                row['eigenspace_bounds'] = bounds
                if len(members) > 1:
                    row['flags'].append('NEAR_REPEATED_EIGENSPACE_SEE_CLUSTER_PERCENTAGES')
                if spectrum_boundary:
                    row['flags'].append('UPPER_SPECTRUM_BOUNDARY_NOT_CLOSED')
                if gap['unresolved_neighbor']:
                    row['flags'].append('ADJACENT_CLUSTER_GAP_NOT_RESOLVED')
                if label not in FAMILIES or row['family'] != label or spectrum_boundary:
                    row['mechanical_eligible'] = False
    finally:
        odb.close()
    candidates = {f: candidate_for(f, results) for f in FAMILIES}
    comparison = read_json(args.compare) if args.compare else None
    shape_comparison = read_json(args.shape_comparison) if getattr(args, 'shape_comparison', None) else None
    reference = read_json(args.reference) if args.reference else {}
    review = read_json(args.review) if args.review else {}
    mesh_nodes = sum(x['shell_nodes'] for x in metadata['instances'])
    family_assessment = {}
    for family in FAMILIES:
        compared = None
        if comparison:
            compared = dict(candidate=comparison.get('candidates', {}).get(family), compatible=bool(
                signature and comparison.get('model_signature') == signature and
                comparison.get('mesh_nodes') != mesh_nodes and
                comparison.get('basis_definition_id') == basis_meta.get('family_definition_id') and
                comparison.get('percentage_kind') == 'MAPPED_MECHANICAL_BASIS_NORM'))
        ref = dict(reference.get('families', {}).get(family, {})) or None
        if ref:
            ref['model_signature'] = reference.get('model_signature')
        reviewed = review.get('families', {}).get(family) if review.get('model_signature') == signature else None
        shape_match = False
        if shape_comparison and comparison and candidates[family] and (compared or {}).get('candidate'):
            sa, sb = shape_comparison.get('source_a', {}), shape_comparison.get('source_b', {})
            direct = sa.get('source_odb_sha256') == odb_hash and sb.get('source_odb_sha256') == comparison.get('source_odb_sha256')
            reverse = sb.get('source_odb_sha256') == odb_hash and sa.get('source_odb_sha256') == comparison.get('source_odb_sha256')
            if (direct or reverse) and shape_comparison.get('model_signature') == signature:
                current, other = ('modes_a', 'modes_b') if direct else ('modes_b', 'modes_a')
                shape_match = any(m.get('numerical_checks_pass') and candidates[family]['mode'] in m[current]
                    and compared['candidate']['mode'] in m[other]
                    and len(m[current]) == len(m[other])
                    and m.get('minimum_cosine_squared', 0.) >= args.mesh_shape_threshold
                    and m.get('relative_eigenvalue_difference') is not None
                    and m['relative_eigenvalue_difference'] <= args.validation_tolerance
                    for m in shape_comparison.get('matches', []))
        family_assessment[family] = assess_family(candidates[family], compared, ref, reviewed, signature, setup_ok,
                                                 args.validation_tolerance, shape_match)
    curve_reference = read_curve_reference(args.curve_reference) if args.curve_reference else []
    summary = dict(schema_version=3, source_odb=odb_path, source_odb_sha256=odb_hash, model_signature=signature,
        modes_processed=len(results), mesh_nodes=mesh_nodes, sigma_ref_MPa=sigma, setup_checks_pass=setup_ok,
        percentage_kind=results[0]['percentage_kind'], basis_definition_id=basis_meta.get('family_definition_id'),
        basis_metadata=basis_meta, proxy_geometry=proxy.metadata, settings=vars(args),
        curve_reference=curve_reference,
        candidates=candidates, families=family_assessment,
        dsm_inputs_MPa={name: family_assessment[f]['accepted_stress_MPa'] for name, f in
                        (('Fcrl', 'L'), ('Fcrd', 'D'), ('Fcre', 'G'))},
        clusters=clusters, modes=results, direct_grid_available=grid is not None, minimum_track_coverage=coverage,
        sensitivity_angles_deg=None if physical_segments else [2., 3., 5.],
        sensitivity_radius_fractions=[.03, .04, .05] if physical_segments else None,
        fields_available_in_all_modes=common_fields,
        requested_fields_missing_from_some_modes=sorted(set(build.get('modal_output', {}).get('fields', []))-set(common_fields)),
        limitations=['Not a classical signature curve or an automatic cFSM/GBT basis generator.',
            'Family percentages are metric/basis dependent, not portions of critical load.',
            'Without a mapped validated basis all L/D/G labels and percentages are geometric screening proxies.',
            'Independent rigid motion of built-up pieces is reported as Assembly and is not counted as Distortional.',
            'Anchor transverse extension is reported as Other and is not counted as Distortional.',
            'Local is measured first relative to endpoint-driven curved-wall motion that reproduces rigid rotation; D uses the remaining inextensional fold/coarse motion.',
            'L/D/G family energy requires compatible full elastic K and all retained DOFs; signed cross terms must not be discarded.',
            'SUPPLIED review/reference evidence is recorded, not independently certified by this program.',
            'No conclusion of family absence or DSM applicability follows from missing candidates.'])
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        summary['mesh_shape_archive'] = validation.export_shapes(args.output_dir, geo, grid, full_shapes, summary, build)
    except ValueError as exc:
        summary['mesh_shape_archive'] = dict(status='UNAVAILABLE_UNSUPPORTED_TOPOLOGY', reason=str(exc))
    write_outputs(args.output_dir, summary)
    validation.write_cluster_report(args.output_dir, summary)
    visuals.write_visuals(args.output_dir, summary, previews,
        dict(xy=geo['xy'].tolist(), edges=geo['edges'], pieces=pieces), spectra)
    with open(os.path.join(args.output_dir, 'odb_nodes.csv'), 'w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['instance', 'label', 'x_mm', 'y_mm', 'z_mm'])
        for (name, label), xyz in sorted(geo['coords'].items()):
            writer.writerow([name, label]+list(xyz))
    return summary


def write_outputs(output_dir, summary):
    serialized = json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'modal_audit.json'), 'w', encoding='utf-8') as stream:
        stream.write(serialized)
    fields = ['mode', 'eigenvalue', 'stress_MPa', 'half_wavelength_mm', 'dominant_halfwaves', 'family',
        'L_percent', 'D_percent', 'G_percent', 'percentage_kind', 'relative_residual',
        'displacement_cross_percent', 'condition', 'cluster_id', 'mechanical_eligible',
        'energy_status', 'energy_L_percent', 'energy_D_percent', 'energy_G_percent', 'energy_R_percent',
        'energy_cross_terms_percent', 'spectral_fit_error', 'dominant_spectral_share', 'transverse_share',
        'raw_vs_fitted_max_pp', 'sensitivity_range_pp', 'relative_piece_rigid_percent', 'assembly_percent', 'other_percent', 'wall_curvature_index', 'flags',
        'eigenspace_stable_family', 'eigenspace_L_min', 'eigenspace_L_max', 'eigenspace_D_min',
        'eigenspace_D_max', 'eigenspace_G_min', 'eigenspace_G_max']
    with open(os.path.join(output_dir, 'modal_percentages.csv'), 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for row in summary['modes']:
            flat = dict(row)
            flat.update({family+'_percent': value for family, value in zip(FAMILIES, row['percentages'])})
            flat['flags'] = ';'.join(row['flags'])
            flat['sensitivity_range_pp'] = (row.get('sensitivity') or {}).get('max_range_pp')
            flat['relative_piece_rigid_percent'] = (row.get('rigid_diagnostics') or {}).get('relative_piece_rigid_percent')
            flat['assembly_percent'] = (row.get('rigid_diagnostics') or {}).get('assembly_percent')
            flat['other_percent'] = (row.get('rigid_diagnostics') or {}).get('other_percent')
            flat['wall_curvature_index'] = (row.get('rigid_diagnostics') or {}).get('wall_curvature_index')
            if row.get('eigenspace_bounds'):
                for j, f in enumerate(FAMILIES):
                    for bound in ('min', 'max'):
                        flat['eigenspace_'+f+'_'+bound] = row['eigenspace_bounds'][bound+'_percent'][j]
            if row['energy']:
                flat.update({'energy_'+f+'_percent': value for f, value in row['energy']['diagonal_percent'].items()})
                flat['energy_cross_terms_percent'] = json.dumps(row['energy']['cross_percent'], sort_keys=True)
            writer.writerow(flat)
    with open(os.path.join(output_dir, 'dsm_inputs.csv'), 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['family', 'candidate_mode', 'candidate_stress_MPa', 'accepted_stress_MPa', 'status', 'reasons'])
        for family, item in summary['families'].items():
            candidate = item['candidate'] or {}
            writer.writerow([family, candidate.get('mode'), candidate.get('stress_MPa'),
                             item['accepted_stress_MPa'], item['status'], ';'.join(item['reasons'])])
    template = dict(model_signature=summary['model_signature'], families={f: dict(
        mode=(summary['candidates'][f] or {}).get('mode'), shape_review='', coverage_review='', mesh_shape_review='') for f in FAMILIES})
    with open(os.path.join(output_dir, 'review_template.json'), 'w') as stream:
        json.dump(template, stream, indent=2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summary['modes']
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    bottom = np.zeros(len(rows))
    colors = {'L': '#289b69', 'D': '#e49a28', 'G': '#397dcc', 'R': '#888888'}
    for j, family in enumerate(FAMILIES):
        values = np.array([r['percentages'][j] for r in rows])
        axes[0].bar(x, values, bottom=bottom, width=1., color=colors[family], label=family)
        bottom += values
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Component norm percentage (%)')
    axes[0].set_title(summary['percentage_kind']+' | NOT portions of critical load'+
        ('\nScreening proxies; NOT validated mechanical L/D/G participation' if not summary['basis_metadata'] else ''))
    axes[0].legend(ncol=3)
    if any(r['energy'] for r in rows):
        for family in ('L', 'D', 'G', 'R'):
            axes[1].plot(x, [r['energy']['diagonal_percent'][family] if r['energy'] else np.nan for r in rows],
                         color=colors[family], label=family+' self-energy')
        axes[1].plot(x, [sum(r['energy']['cross_percent'].values()) if r['energy'] else np.nan for r in rows],
                     color='#ab438f', label='Signed cross-term sum')
        axes[1].axhline(0, color='#666666', linewidth=.6)
        axes[1].legend(ncol=3, fontsize=8)
        axes[1].set_ylabel('Percent of total modal elastic energy')
        axes[1].set_title('Self terms + ALL signed cross terms + residual = 100% (see JSON for individual cross terms)')
    else:
        axes[1].text(.5, .5, 'L/D/G ENERGY UNAVAILABLE\nNo compatible mechanical basis + elastic K supplied.\nDisplacement shares are NOT energy shares.',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_yticks([])
    ticks = np.unique(np.linspace(0, len(rows)-1, min(16, len(rows))).astype(int))
    axes[1].set_xticks(ticks); axes[1].set_xticklabels([rows[i]['mode'] for i in ticks])
    axes[1].set_xlabel('Mode number; near-repeated eigenspace percentages are in modal_audit.json')
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, 'modal_percentages.png'), dpi=160); plt.close(fig)
    lines = ['Modal audit: %d modes' % len(rows), 'Percentage kind: '+summary['percentage_kind'],
             'Model signature: '+str(summary['model_signature']), '']
    for family, item in summary['families'].items():
        lines.append('%s: %s | accepted MPa=%s | %s' %
                     (family, item['status'], item['accepted_stress_MPa'], ', '.join(item['reasons'])))
    lines.extend(['']+summary['limitations'])
    with open(os.path.join(output_dir, 'SUMMARY.txt'), 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[:7])); print('SAVED: '+output_dir)


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output-dir')
    p.add_argument('--basis', help='Mapped cFSM/GBT/mechanical basis NPZ; optionally contains compatible elastic K')
    p.add_argument('--compare', help='modal_audit.json from a different mesh of the same physical model')
    p.add_argument('--shape-comparison', help='mesh_comparison.json with quantitative matched eigenspaces of both ODBs')
    p.add_argument('--reference', help='JSON with independent family buckling reference values and provenance')
    p.add_argument('--review', help='Completed review_template.json, with documented shape/coverage reviews')
    p.add_argument('--curve-reference', help='Optional plotting-only CSV: label, half_wavelength_mm, critical_stress_MPa')
    p.add_argument('--dominance', type=float, default=.9)
    p.add_argument('--max-residual', type=float, default=.05)
    p.add_argument('--cluster-tolerance', type=float, default=.001)
    p.add_argument('--validation-tolerance', type=float, default=.05)
    p.add_argument('--mesh-shape-threshold', type=float, default=.95, help='Minimum cos squared principal angle for DSM mesh evidence')
    p.add_argument('--max-sensitivity-pp', type=float, default=10., help='Panel-angle sensitivity limit in percentage points, not a probability')
    p.add_argument('--max-assembly-percent', type=float, default=25.,
                   help='Above this piece-rigid self-norm share, do not force the mode into DSM L/D/G')
    p.add_argument('--max-other-percent', type=float, default=25.,
                   help='Above this transverse-extension self-norm share, do not force the mode into DSM L/D/G')
    args = p.parse_args(argv)
    args.run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    if args.curve_reference:
        args.curve_reference = os.path.abspath(os.path.expanduser(args.curve_reference))
    args.output_dir = os.path.abspath(args.output_dir or os.path.join(args.run_dir, 'modal_dsm_audit'))
    for name in ('dominance', 'max_residual', 'cluster_tolerance', 'validation_tolerance', 'mesh_shape_threshold'):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value < 1:
            p.error(name+' must be between 0 and 1')
    if args.dominance <= .5:
        p.error('dominance must exceed 0.5 for a unique dominant family')
    if not math.isfinite(args.max_sensitivity_pp) or not 0 < args.max_sensitivity_pp <= 100:
        p.error('--max-sensitivity-pp must be in (0, 100]')
    if not math.isfinite(args.max_assembly_percent) or not 0 < args.max_assembly_percent < 100:
        p.error('--max-assembly-percent must be in (0, 100)')
    if not math.isfinite(args.max_other_percent) or not 0 < args.max_other_percent < 100:
        p.error('--max-other-percent must be in (0, 100)')
    if os.path.exists(args.output_dir) and (not os.path.isdir(args.output_dir) or os.listdir(args.output_dir)):
        p.error('Output directory must be new or empty; choose --output-dir for another audit')
    return args


if __name__ == '__main__':
    process(parse_arguments())
