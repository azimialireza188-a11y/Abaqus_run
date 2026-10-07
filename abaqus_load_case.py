# -*- coding: ascii -*-
"""Shared load-case definitions (compression / bending) for the Abaqus_run pipeline.

Standard library only: importable by normal Python, Abaqus Python and CAE noGUI.
Units: N, mm, MPa. Member axis = global Z; the section lies in the XY plane.

COMPRESSION (unchanged production convention)
  Uniform edge stress sigma_ref = 1 MPa on both end sections, so the buckling
  eigenvalue lambda is the critical stress in MPa (P_cr = lambda * A).

BENDING (CUFSM-consistent reference stress)
  Linear stress on both end sections, compression positive,
      sigma(x, y) = sigma_ref * eta / c,      eta = n . (r - r_c),
      n = (-sin(theta), cos(theta)),
  theta  = angle of the NEUTRAL AXIS from global X (theta = 0: neutral axis
           along X, compression on the +Y side; the moment vector is parallel
           to the neutral axis whenever I_nt = 0, e.g. any C4 section),
  r_c    = centroid of the CUFSM centre-line section,
  c      = max |eta| over the CUFSM centre-line section nodes (extreme fibre).
  sigma_ref = 1 MPa, so the extreme-fibre stress is exactly 1 MPa: lambda is the
  critical EXTREME-FIBRE stress in MPa, i.e. the CUFSM signature-curve load
  factor for a unit extreme-fibre reference stress, and
      M_cr = lambda * M_ref,  M_ref = I_nn / c  (N mm per MPa),
      M_cr / M_y = lambda / f_y   (DSM flexure, M_y = S f_y, S = I_nn / c).
  The stress is applied as CONSISTENT NODAL FORCES of the linear edge traction
  on the meshed end edges (exact for linear S4R edges, identical to a linearly
  varying shell edge traction), so the reference load is fully under control.
"""
import math

LOAD_CASES = ('compression', 'bending')
REFERENCE_STRESS_MPA = 1.0


def normalize_angle(theta_deg):
    """Neutral-axis angle in [0, 360) degrees, rounded to remove float noise."""
    value = float(theta_deg)
    if not math.isfinite(value):
        raise ValueError('Bending axis angle must be finite')
    value = round(value % 360.0, 9) % 360.0
    return 0.0 if value == 0 else value


def angle_token(theta_deg):
    """22.5 -> '022p5', 0 -> '000', 45 -> '045' (file-name safe, sortable)."""
    value = normalize_angle(theta_deg)
    whole, frac = ('%.9f' % value).split('.')
    text = '%03d' % int(whole)
    frac = frac.rstrip('0')
    if frac:
        text += 'p' + frac
    return text


def load_case_tag(load_case, theta_deg=0.0):
    """Suffix appended to run-folder and job names: '' for compression (unchanged)."""
    if load_case == 'compression':
        return ''
    if load_case == 'bending':
        return '_BEND' + angle_token(theta_deg)
    raise ValueError('Unknown load case: %r' % (load_case,))


def _segments(section):
    """Segments (x1, y1, x2, y2) from {piece: [[x1, y1, x2, y2], ...]} or a list."""
    if isinstance(section, dict):
        items = [section[k] for k in sorted(section, key=str)]
        return [tuple(float(v) for v in seg[:4]) for piece in items for seg in piece]
    return [tuple(float(v) for v in seg[:4]) for seg in section]


def _points(segments):
    out = []
    for x1, y1, x2, y2 in segments:
        out.append((x1, y1))
        out.append((x2, y2))
    return out


def section_properties(section, thickness):
    """Thin-walled centre-line properties (CUFSM grosprop convention: t * ds line integrals)."""
    segments = _segments(section)
    t = float(thickness)
    if not segments or not math.isfinite(t) or not t > 0:
        raise ValueError('Section segments and a positive thickness are required')
    if any(len(seg) != 4 or not all(math.isfinite(v) for v in seg) for seg in segments):
        raise ValueError('Section segments must contain four finite coordinates')
    if any(math.hypot(x2-x1, y2-y1) <= 0 for x1, y1, x2, y2 in segments):
        raise ValueError('Section segments must have positive length')
    # Integrate near the section, then about its centroid. Subtracting A*y_c^2
    # from a global second moment loses small-section inertia at large offsets.
    ox, oy = segments[0][:2]
    local = [(x1-ox, y1-oy, x2-ox, y2-oy) for x1, y1, x2, y2 in segments]
    area = sx = sy = ixx = iyy = ixy = 0.0
    for x1, y1, x2, y2 in local:
        length = math.hypot(x2 - x1, y2 - y1)
        area += t * length
        sx += t * length * (x1 + x2) / 2.0
        sy += t * length * (y1 + y2) / 2.0
    xc, yc = sx / area, sy / area
    for x1, y1, x2, y2 in local:
        length = math.hypot(x2 - x1, y2 - y1)
        x1, y1, x2, y2 = x1-xc, y1-yc, x2-xc, y2-yc
        ixx += t * length * (y1 * y1 + y1 * y2 + y2 * y2) / 3.0
        iyy += t * length * (x1 * x1 + x1 * x2 + x2 * x2) / 3.0
        ixy += t * length * (2 * x1 * y1 + x1 * y2 + x2 * y1 + 2 * x2 * y2) / 6.0
    return dict(area_mm2=area, centroid_mm=[xc+ox, yc+oy], Ixx_mm4=ixx, Iyy_mm4=iyy, Ixy_mm4=ixy,
                definition='centre-line thin-walled section (t*ds), axes through the centroid')


def _direction(theta_deg):
    theta = math.radians(normalize_angle(theta_deg))
    return (math.cos(theta), math.sin(theta)), (-math.sin(theta), math.cos(theta))


def _plastic(segments, t, eta_of):
    """Plastic neutral-axis offset d (equal areas) and plastic modulus t*int|eta - d| ds."""
    pieces = [(eta_of(x1, y1), eta_of(x2, y2), math.hypot(x2 - x1, y2 - y1)) for x1, y1, x2, y2 in segments]

    def area_above(d):
        total = 0.0
        for a, b, length in pieces:
            lo, hi = min(a, b), max(a, b)
            if lo >= d:
                total += length
            elif hi > d:
                total += length * (hi - d) / (hi - lo)
        return t * total

    total = t * sum(p[2] for p in pieces)
    lo = min(min(a, b) for a, b, _ in pieces)
    hi = max(max(a, b) for a, b, _ in pieces)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if area_above(mid) > 0.5 * total:
            lo = mid
        else:
            hi = mid
    d = 0.5 * (lo + hi)
    z = 0.0
    for a, b, length in pieces:
        a, b = a - d, b - d
        if a * b >= 0:
            z += length * abs(a + b) / 2.0
        else:
            z += length * (a * a + b * b) / (2.0 * (abs(a) + abs(b)))
    return d, t * z


def bending_reference(section, thickness, theta_deg):
    """Reference definition of a bending load case (see module docstring)."""
    segments = _segments(section)
    t = float(thickness)
    props = section_properties(segments, t)
    xc, yc = props['centroid_mm']
    (tx, ty), (nx, ny) = _direction(theta_deg)

    def eta_of(x, y):
        return nx * (x - xc) + ny * (y - yc)

    def xi_of(x, y):
        return tx * (x - xc) + ty * (y - yc)

    etas = [eta_of(x, y) for x, y in _points(segments)]
    c_comp, c_tens = max(etas), -min(etas)
    c = max(c_comp, c_tens)
    if not c > 0:
        raise ValueError('Degenerate section for bending')
    inn = int_ = 0.0
    for x1, y1, x2, y2 in segments:
        length = math.hypot(x2 - x1, y2 - y1)
        e1, e2 = eta_of(x1, y1), eta_of(x2, y2)
        s1, s2 = xi_of(x1, y1), xi_of(x2, y2)
        inn += t * length * (e1 * e1 + e1 * e2 + e2 * e2) / 3.0
        int_ += t * length * (2 * e1 * s1 + e1 * s2 + e2 * s1 + 2 * e2 * s2) / 6.0
    d_pna, zpl = _plastic(segments, t, eta_of)
    # Moment of the unit stress field about the centroid on the z = L face (outward normal +Z):
    # traction -sigma*e_z (compression positive), M = int r x (-sigma e_z) dA
    #   = (1/c) int eta (xi n - eta t) dA = (-I_nn t + I_nt n) / c.
    mx = (-inn * tx + int_ * nx) / c
    my = (-inn * ty + int_ * ny) / c
    moment_angle = math.degrees(math.atan2(my, mx))
    return dict(
        type='bending', neutral_axis_angle_deg=normalize_angle(theta_deg),
        neutral_axis_direction=[tx, ty], compression_normal=[nx, ny],
        centroid_mm=[xc, yc], c_extreme_mm=c, c_compression_mm=c_comp, c_tension_mm=c_tens,
        I_nn_mm4=inn, I_nt_mm4=int_, area_mm2=props['area_mm2'],
        reference_stress_MPa=REFERENCE_STRESS_MPA,
        reference_moment_Nmm_per_MPa=inn / c,
        moment_vector_angle_deg_on_zL_face=normalize_angle(moment_angle),
        section_modulus_mm3=inn / c,
        section_modulus_outer_fibre_mm3=inn / (c + 0.5 * t),
        plastic_modulus_mm3=zpl, plastic_neutral_axis_offset_mm=d_pna,
        stress_definition=('sigma = 1 MPa * eta / c, eta = n.(r - r_c), compression positive on +n; '
                           'c = max |eta| over the CUFSM centre-line nodes'),
        eigenvalue_meaning='lambda = critical extreme-fibre (centre-line) stress in MPa; M_cr = lambda * M_ref')


def compression_reference(section, thickness):
    props = section_properties(section, thickness)
    return dict(type='compression', reference_stress_MPa=REFERENCE_STRESS_MPA,
                area_mm2=props['area_mm2'], centroid_mm=props['centroid_mm'],
                stress_definition='uniform 1 MPa on both end sections',
                eigenvalue_meaning='lambda = critical stress in MPa; P_cr = lambda * A')


def unit_stress(reference, x, y):
    """sigma/sigma_ref at (x, y): 1 for compression, eta/c for bending."""
    if reference['type'] == 'compression':
        return 1.0
    nx, ny = reference['compression_normal']
    xc, yc = reference['centroid_mm']
    return (nx * (x - xc) + ny * (y - yc)) / reference['c_extreme_mm']


def consistent_edge_forces(coordinates, edges, thickness, reference):
    """Consistent nodal forces (N per MPa of sigma_ref) of the edge traction t*sigma on the
    straight end edges (linear shape functions): f_a += t*L*(2 s_a + s_b)/6. Exact for the
    linear bending stress; for compression it equals t*L/2 per edge end (tributary area)."""
    t = float(thickness)
    forces = {}
    for a, b in edges:
        pa, pb = coordinates[a], coordinates[b]
        length = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        sa, sb = unit_stress(reference, pa[0], pa[1]), unit_stress(reference, pb[0], pb[1])
        forces[a] = forces.get(a, 0.0) + t * length * (2.0 * sa + sb) / 6.0
        forces[b] = forces.get(b, 0.0) + t * length * (sa + 2.0 * sb) / 6.0
    return forces


# --------------------------------------------------------------------------
# Section symmetry and the bending axes needed for P-M interaction
# --------------------------------------------------------------------------
def _key(p, tol):
    return (int(round(p[0] / tol)), int(round(p[1] / tol)))


def _maps_onto(segments, endpoint_index, transform, tol):
    """Match entire walls, in either direction, with multiplicity preserved."""
    used = set()
    for a, b in segments:
        a, b = transform(a), transform(b)
        k = _key(a, tol)
        match = None
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for index, p, q in endpoint_index.get((k[0]+i, k[1]+j), ()):
                    if (index not in used and math.hypot(a[0]-p[0], a[1]-p[1]) <= tol
                            and math.hypot(b[0]-q[0], b[1]-q[1]) <= tol):
                        match = index
                        break
                if match is not None:
                    break
            if match is not None:
                break
        if match is None:
            return False
        used.add(match)
    return True


def section_symmetry(section, tol=1e-4):
    """Rotations (multiples of 90 deg) and mirror lines (deg, in [0, 180)) about the centroid
    that map all centre-line walls onto themselves (tolerance tol mm)."""
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError('Symmetry tolerance must be positive and finite')
    segments = _segments(section)
    xc, yc = section_properties(segments, 1.0)['centroid_mm']
    pts = [(x - xc, y - yc) for x, y in _points(segments)]
    walls = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
    endpoint_index = {}
    for index, (a, b) in enumerate(walls):
        endpoint_index.setdefault(_key(a, tol), []).append((index, a, b))
        endpoint_index.setdefault(_key(b, tol), []).append((index, b, a))
    rotations = []
    for k in range(4):
        a = math.radians(90.0 * k)
        ca, sa = round(math.cos(a)), round(math.sin(a))
        if _maps_onto(walls, endpoint_index, lambda p, ca=ca, sa=sa: (ca * p[0] - sa * p[1], sa * p[0] + ca * p[1]), tol):
            rotations.append(90 * k)
    # mirror-line candidates: bisectors between the farthest point and every point of equal radius
    p0 = max(pts, key=lambda p: (round(math.hypot(*p) / tol), p))
    r0 = math.hypot(*p0)
    a0 = math.atan2(p0[1], p0[0])
    candidates = set()
    for q in pts:
        if abs(math.hypot(*q) - r0) <= 10 * tol:
            alpha = 0.5 * (a0 + math.atan2(q[1], q[0]))
            for extra in (0.0, 0.5 * math.pi):
                candidates.add(round(math.degrees(alpha + extra) % 180.0, 6) % 180.0)
    mirrors = []
    for alpha in sorted(candidates):
        c2, s2 = math.cos(math.radians(2 * alpha)), math.sin(math.radians(2 * alpha))
        if _maps_onto(walls, endpoint_index, lambda p, c2=c2, s2=s2: (c2 * p[0] + s2 * p[1], s2 * p[0] - c2 * p[1]), tol):
            if not any(abs(alpha - m) < 1e-4 or abs(abs(alpha - m) - 180) < 1e-4 for m in mirrors):
                mirrors.append(alpha)
    return dict(rotations_deg=rotations, mirror_lines_deg=mirrors, centroid_mm=[xc, yc], tolerance_mm=tol)


def _orbit(theta, symmetry):
    """All neutral-axis angles equivalent to theta: a rotation phi maps theta -> theta + phi,
    a mirror about line alpha maps theta -> 2 alpha - theta + 180 (the compressed side follows)."""
    out = set()
    for rot in symmetry['rotations_deg'] or [0]:
        out.add(normalize_angle(theta + rot))
        for alpha in symmetry['mirror_lines_deg']:
            out.add(normalize_angle(2 * alpha - (theta + rot) + 180.0))
    return sorted(out)


def canonical_axis(theta, symmetry):
    """Smallest equivalent neutral-axis angle (the representative of theta's symmetry class)."""
    return min(_orbit(normalize_angle(theta), symmetry))


def required_bending_axes(symmetry):
    """Distinct bending cases needed for P-M (one per symmetry class) and the fundamental range.

    group order |G| = (#rotations) * (2 if mirrored else 1); fundamental range = 360/|G| deg.
    With mirror lines (e.g. D4): the axes along mirror lines bound the range and are the
    symmetric cases - they are REQUIRED; intermediate angles give the biaxial surface between them.
    Without mirrors (C4 only): no axis is privileged; sample the range."""
    order = max(1, len(symmetry['rotations_deg'])) * (2 if symmetry['mirror_lines_deg'] else 1)
    span = 360.0 / order
    if symmetry['mirror_lines_deg']:
        required = sorted(set(canonical_axis(a, symmetry) for m in symmetry['mirror_lines_deg']
                              for a in (m, m + 90.0, m + 180.0, m + 270.0)))
        optional = ([canonical_axis(0.5 * (required[0] + required[1]), symmetry)]
                    if len(required) == 2 else [])
    else:
        required = [0.0]
        optional = [round(span * k / 4.0, 6) for k in (1, 2, 3)]
    return dict(group_order=order, fundamental_range_width_deg=span, required_axes_deg=required,
                optional_intermediate_axes_deg=optional,
                moment_sign_reversal_equivalent=180 in symmetry['rotations_deg'])
