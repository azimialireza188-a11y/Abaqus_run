"""Source-polyline walls and linearized rigid-exact curved-wall kinematics.

This geometry-only detector is a reviewable screening assumption, not a cFSM
definition. Compact high-curvature bends separate walls; gentle waviness does
not. Inputs/thresholds and the resulting endpoints are retained for review.
"""
import math
import numpy as np


def physical_wall_layout(points, radius_fraction=.04, min_bend_angle_deg=30.):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError('An open source polyline with at least two points is required')
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(points)) or np.any(lengths <= 0):
        raise ValueError('Source geometry must be finite with nonzero segments')
    if not 0 < radius_fraction < .25 or not 0 < min_bend_angle_deg < 180:
        raise ValueError('Invalid bend detection thresholds')
    s = np.r_[0., np.cumsum(lengths)]
    turn = np.arctan2(vectors[:-1, 0]*vectors[1:, 1]-vectors[:-1, 1]*vectors[1:, 0],
                     np.sum(vectors[:-1]*vectors[1:], axis=1))
    curvature = turn/(.5*(lengths[:-1]+lengths[1:]))
    cutoff = 1./(radius_fraction*s[-1])
    minimum_turn = math.radians(min_bend_angle_deg)
    bends = [(i+1, i+1) for i, angle in enumerate(turn) if abs(angle) >= minimum_turn]
    runs = []
    for j, value in enumerate(curvature):
        if abs(value) < cutoff:
            continue
        if not runs or runs[-1][-1] != j-1 or turn[runs[-1][-1]]*turn[j] <= 0:
            runs.append([])
        runs[-1].append(j)
    for run in runs:
        if abs(sum(turn[run])) < minimum_turn:
            continue
        lo, hi = run[0]+1, run[-1]+1
        # Include transition vertices smeared by unequal straight/arc segments.
        if run[0] > 0 and 1e-8 < abs(turn[run[0]-1]) < minimum_turn:
            lo -= 1
        if run[-1]+1 < len(turn) and 1e-8 < abs(turn[run[-1]+1]) < minimum_turn:
            hi += 1
        if s[hi]-s[lo] <= 2.*radius_fraction*s[-1]:
            bends.append((lo, hi))
    merged = []
    for lo, hi in sorted(bends):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    walls = []; start = 0
    for lo, hi in merged:
        if lo > start:
            walls.append((start, lo))
        start = hi
    if start < len(points)-1:
        walls.append((start, len(points)-1))
    return dict(walls=walls, bends=merged, arclength=s,
                curvature_threshold_per_mm=cutoff, radius_fraction=radius_fraction,
                minimum_bend_angle_deg=min_bend_angle_deg)


def wall_kinematics(points):
    """Return endpoint interpolation and normals reproducing u=a+omega*J*x.

    A straight-chord interpolation alone falsely assigns rigid rotation of an
    initially curved wall to Local. The extra rotation of the reference curve
    removes that artifact. This is a linearized displacement operator.
    """
    points = np.asarray(points, float)
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    dist = np.r_[0., np.cumsum(ds)]
    chord = points[-1]-points[0]
    chord2 = float(chord@chord)
    if chord2 <= 1e-24*max(dist[-1]**2, 1e-250) or np.any(ds <= 0):
        raise ValueError('Wall endpoints must be distinct and wall segments nonzero')
    t = dist/dist[-1]
    offset = points-(points[0]+t[:, None]*chord)
    joffset = np.column_stack((-offset[:, 1], offset[:, 0]))
    jchord = np.array([-chord[1], chord[0]])
    correction = joffset[:, :, None]*jchord[None, None, :]/chord2
    a = (1.-t)[:, None, None]*np.eye(2)-correction
    b = t[:, None, None]*np.eye(2)+correction
    tangent = np.gradient(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return a, b, normal, dist
