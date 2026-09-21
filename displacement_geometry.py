"""Document 77: conforming adaptive mesh hierarchy and residual displacement.

Input is the existing FULL normal evidence, not brightness used as height.
Lengths here are millimetres (possibly estimated). Subdivision is linear
red/green triangle subdivision, not Catmull-Clark.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-15)


@dataclass
class Surface:
    rest: np.ndarray
    directions: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    materials: np.ndarray
    origins: np.ndarray
    height: np.ndarray
    generation: np.ndarray
    source_rest: np.ndarray | None = None
    source_directions: np.ndarray | None = None

    @property
    def positions(self):
        return self.rest + self.height[:, None] * self.directions


@dataclass
class Evidence:
    face: np.ndarray
    bary: np.ndarray
    gradient: np.ndarray
    weight: np.ndarray
    chain: np.ndarray
    domain: np.ndarray


BARY_NODES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [.5, .5, 0], [0, .5, .5], [.5, 0, .5]])
TEMPLATES = {
    0: [(0, 1, 2)],
    1: [(0, 3, 2), (3, 1, 2)],
    2: [(1, 4, 0), (4, 2, 0)],
    4: [(2, 5, 1), (5, 0, 1)],
    3: [(1, 4, 3), (0, 3, 2), (3, 4, 2)],
    6: [(2, 5, 4), (1, 4, 0), (4, 5, 0)],
    5: [(0, 3, 5), (2, 5, 1), (5, 3, 1)],
    7: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)],
}


def edges_of(faces):
    pairs = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    edges, inverse, counts = np.unique(pairs.reshape(-1, 2), axis=0,
                                       return_inverse=True, return_counts=True)
    return edges, inverse.reshape(-1, 3), counts


def split_surface(surface: Surface, evidence: Evidence, selected, level):
    """Share every new edge vertex and transport observations without resampling.

    Selected triangles become four children; transition neighbors become two
    or three, so the operation introduces no hanging vertices/T-junctions.
    Per-corner UV interpolation preserves pre-existing UV seams.
    """
    selected = np.asarray(selected, bool)
    if selected.shape != (len(surface.faces),):
        raise ValueError('One subdivision decision is required per face.')
    edges, inverse, counts = edges_of(surface.faces)
    marked = np.zeros(len(edges), bool)
    marked[inverse[selected].ravel()] = True
    marked[counts > 2] = False
    mids = np.full(len(edges), -1, np.int64)
    added = edges[marked]
    old_n = len(surface.rest)
    mids[marked] = np.arange(old_n, old_n + len(added))
    mask = np.sum(marked[inverse] * [1, 2, 4], axis=1).astype(np.int8)
    sizes = np.array([len(TEMPLATES[int(m)]) for m in mask])
    starts = np.r_[0, np.cumsum(sizes[:-1])]
    new_faces = np.empty((int(sizes.sum()), 3), np.int32)
    new_uv = np.empty((len(new_faces), 3, 2), np.float64)
    parent = np.empty(len(new_faces), np.int32)
    child_bary = np.empty((len(new_faces), 3, 3), np.float32)
    nodes = np.column_stack([surface.faces, mids[inverse]])
    sample_faces = np.full(len(evidence.face), -1, np.int32)
    sample_bary = np.zeros_like(evidence.bary)
    sample_mask = mask[evidence.face]
    for code, template in TEMPLATES.items():
        ids = np.flatnonzero(mask == code)
        sids = np.flatnonzero(sample_mask == code)
        if not len(ids):
            continue
        for j, triple in enumerate(template):
            dest = starts[ids] + j
            transform = BARY_NODES[list(triple)]
            new_faces[dest] = nodes[ids][:, triple]
            new_uv[dest] = np.einsum('ij,fjk->fik', transform, surface.uv[ids])
            parent[dest] = ids
            child_bary[dest] = transform
            if len(sids):
                local = evidence.bary[sids] @ np.linalg.inv(transform)
                keep = (local.min(axis=1) >= -2e-6) & (sample_faces[sids] < 0)
                picked = sids[keep]
                sample_faces[picked] = starts[evidence.face[picked]] + j
                local = np.maximum(local[keep], 0)
                sample_bary[picked] = local / local.sum(axis=1, keepdims=True)
    if np.any(sample_faces < 0) or np.any(new_faces < 0):
        raise RuntimeError('Subdivision lost a sample or introduced a missing edge.')
    rest = np.vstack([surface.rest, surface.rest[added].mean(axis=1)])
    direction = np.vstack([surface.directions, unit(surface.directions[added].mean(axis=1))])
    height = np.r_[surface.height, surface.height[added].mean(axis=1)]
    result = Surface(rest, direction, new_faces, new_uv, surface.materials[parent],
                     surface.origins[parent], height,
                     np.r_[surface.generation, np.full(len(added), level, np.int16)])
    if surface.source_rest is not None:
        result.source_rest = np.vstack([surface.source_rest, surface.source_rest[added].mean(axis=1)])
        result.source_directions = np.vstack([surface.source_directions,
            unit(surface.source_directions[added].mean(axis=1))])
    ev = Evidence(sample_faces, sample_bary, evidence.gradient, evidence.weight,
                  evidence.chain, evidence.domain)
    return result, ev, dict(parent_face=parent, child_bary=child_bary,
        new_edges=added, selected_faces=int(selected.sum()), four_way_faces=int((mask == 7).sum()),
        transition_faces=int(((mask > 0) & (mask != 7)).sum()),
        new_vertices=len(added), previous_vertices=old_n)


def differential(surface):
    points = surface.rest[surface.faces]
    a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    aa = np.einsum('ij,ij->i', a, a)
    ab = np.einsum('ij,ij->i', a, b)
    bb = np.einsum('ij,ij->i', b, b)
    determinant = aa * bb - ab * ab
    safe = np.maximum(determinant, 1e-24)
    g1 = (bb[:, None] * a - ab[:, None] * b) / safe[:, None]
    g2 = (aa[:, None] * b - ab[:, None] * a) / safe[:, None]
    grad = np.stack([-g1 - g2, g1, g2], axis=1)
    valid = determinant > 1e-18
    grad[~valid] = 0
    area = .5 * np.sqrt(np.maximum(determinant, 0))
    return grad, area, valid


def aggregate(surface, evidence):
    count = len(surface.faces)
    w = np.clip(evidence.weight, 0, 1).astype(float)
    sums = np.bincount(evidence.face, weights=w, minlength=count)
    hits = np.bincount(evidence.face, minlength=count)
    denominator = np.maximum(sums, 1e-15)
    gradient = np.column_stack([np.bincount(evidence.face, weights=w * evidence.gradient[:, k],
                                           minlength=count) / denominator for k in range(3)])
    square = np.bincount(evidence.face,
                        weights=w * np.sum(evidence.gradient**2, axis=1), minlength=count) / denominator
    chain = np.bincount(evidence.face, weights=w * evidence.chain,
                        minlength=count) / denominator
    skin = np.bincount(evidence.face, weights=(evidence.domain == 1), minlength=count)
    lips = np.bincount(evidence.face, weights=(evidence.domain == 2), minlength=count)
    return dict(gradient=gradient, square=square, confidence=sums / np.maximum(hits, 1),
                samples=hits, weight=sums, chain=chain, domain=np.where(lips > skin, 2, 1),
                mixed=(skin > 0) & (lips > 0))


def free_vertices(surface, stats, level, cfg):
    eligible = ((stats['samples'] > 0) & (stats['confidence'] >= cfg['minimum_confidence'])
                & ~stats['mixed'])
    touches = np.zeros(len(surface.rest), bool)
    forbidden = np.zeros_like(touches)
    touches[surface.faces[eligible].ravel()] = True
    forbidden[surface.faces[~eligible].ravel()] = True
    skin = np.zeros_like(touches); lips = np.zeros_like(touches)
    skin[surface.faces[eligible & (stats['domain'] == 1)].ravel()] = True
    lips[surface.faces[eligible & (stats['domain'] == 2)].ravel()] = True
    free = touches & ~forbidden & ~(skin & lips)
    if level > 0 and cfg['freeze_parent_vertices']:
        free &= surface.generation == level
    return free, eligible


def face_gradient(surface, grad):
    return np.einsum('fi,fij->fj', surface.height[surface.faces], grad)


def face_error(stats, prediction):
    return np.sqrt(np.maximum(stats['square'] - 2 * np.sum(prediction * stats['gradient'], axis=1)
                              + np.sum(prediction**2, axis=1), 0))


def solve_level(surface: Surface, evidence: Evidence, level, cfg):
    """Integrate only the unexplained gradient over actual mesh triangles."""
    stats = aggregate(surface, evidence)
    grad, area, nondegenerate = differential(surface)
    free, eligible = free_vertices(surface, stats, level, cfg)
    eligible &= nondegenerate
    weights = area * stats['confidence'] * eligible
    n = len(surface.rest)
    local = weights[:, None, None] * np.einsum('fik,fjk->fij', grad, grad)
    rows = np.broadcast_to(surface.faces[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(surface.faces[:, None, :], local.shape).ravel()
    stiffness = sparse.coo_matrix((local.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    mass = np.bincount(surface.faces.ravel(), weights=np.repeat(area * eligible / 3, 3), minlength=n)
    regularizer = mass / cfg['screening_length_mm']**2
    matrix = stiffness + sparse.diags(regularizer + 1e-14)
    target_local = weights[:, None] * np.einsum('fik,fk->fi', grad, stats['gradient'])
    target_rhs = np.bincount(surface.faces.ravel(), weights=target_local.ravel(), minlength=n)
    inherited = surface.height.copy()
    before = face_gradient(surface, grad)
    # Debit inherited geometry once. Do not add the full field at every level.
    rhs = target_rhs - matrix @ inherited
    ids = np.flatnonzero(free)
    delta = np.zeros(n)
    iterations = 0
    if len(ids) and np.linalg.norm(rhs[ids]) > 1e-14:
        a = matrix[ids][:, ids].tocsr()
        diagonal = np.maximum(a.diagonal(), 1e-14)
        preconditioner = LinearOperator(a.shape, matvec=lambda x: x / diagonal)
        def callback(_):
            nonlocal iterations
            iterations += 1
        solution, status = cg(a, rhs[ids], M=preconditioner, rtol=cfg['cg_rtol'],
                              atol=1e-12, maxiter=cfg['cg_maxiter'], callback=callback)
        if status != 0 or not np.isfinite(solution).all():
            raise RuntimeError(f'Level {level} displacement solve did not converge: {status}')
        delta[ids] = solution
    step_limit = cfg['step_limits_mm'][min(level, len(cfg['step_limits_mm']) - 1)]
    delta = np.clip(delta, -step_limit, step_limit)
    proposed = np.clip(inherited + delta, -cfg['maximum_displacement_mm'], cfg['maximum_displacement_mm'])
    delta = proposed - inherited
    delta[~free] = 0
    p = surface.positions[surface.faces]
    old_cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    old_area2 = np.linalg.norm(old_cross, axis=1)
    check = old_area2 > 1e-10

    def energy(h):
        pred = np.einsum('fi,fij->fj', h[surface.faces], grad)
        return float(np.sum(weights * np.sum((pred - stats['gradient'])**2, axis=1))
                     + np.sum(regularizer * h * h))

    initial_energy = energy(inherited)
    fraction = 1.
    accepted = False
    for _ in range(16):
        h = inherited + fraction * delta
        q = (surface.rest + h[:, None] * surface.directions)[surface.faces]
        cross = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        ratio = np.linalg.norm(cross, axis=1) / np.maximum(old_area2, 1e-30)
        safe = np.all(np.einsum('ij,ij->i', old_cross[check], cross[check]) > 0)
        safe &= np.all(ratio[check] >= .2)
        if safe and energy(h) <= initial_energy + max(1e-12, initial_energy * 1e-9):
            accepted = True
            break
        fraction *= .5
    if not accepted:
        fraction = 0.; h = inherited
    surface.height = h
    after = face_gradient(surface, grad)
    before_error, after_error = face_error(stats, before), face_error(stats, after)
    denominator = max(float(weights.sum()), 1e-15)
    report = dict(level=level, vertices=n, triangles=len(surface.faces),
        active_faces=int(eligible.sum()), free_vertices=len(ids), cg_iterations=iterations,
        inherited_height_subtracted=True, parent_vertices_frozen=bool(level and cfg['freeze_parent_vertices']),
        residual_rms_before=float(np.sqrt(np.sum(weights * before_error**2) / denominator)),
        residual_rms_after=float(np.sqrt(np.sum(weights * after_error**2) / denominator)),
        energy_before=initial_energy, energy_after=energy(h), accepted_fraction=fraction,
        max_delta_mm=float(np.max(np.abs(h - inherited), initial=0)),
        max_total_mm=float(np.max(np.abs(h), initial=0)),
        changed_vertices=int(np.count_nonzero(np.abs(h - inherited) > 1e-9)),
        boundary_and_parent_max_delta_mm=float(np.max(np.abs((h-inherited)[~free]), initial=0)),
        mixed_domain_faces=int(stats['mixed'].sum()))
    return report, stats, after_error, h - inherited


def choose_refinement(surface, stats, residual, cfg, pixel_scale):
    if not np.any(stats['square'] > 1e-12):
        return np.zeros(len(surface.faces), bool)
    points = surface.rest[surface.faces]
    max_length = np.max(np.linalg.norm(points - np.roll(points, 1, axis=1), axis=2), axis=1)
    curvature = np.max(np.linalg.norm(surface.directions[surface.faces]
                         - np.roll(surface.directions[surface.faces], 1, axis=1), axis=2), axis=1)
    wanted = ((stats['samples'] >= cfg['minimum_samples_to_split'])
              & (stats['confidence'] >= cfg['minimum_confidence'])
              & (max_length * pixel_scale >= cfg['minimum_edge_pixels'])
              & ((residual > cfg['residual_threshold'])
                 | ((stats['chain'] > .2) & (curvature > .025))))
    score = residual * np.sqrt(np.maximum(stats['confidence'], 0))
    score *= (1 + cfg['chain_priority'] * stats['chain'] + np.minimum(curvature, .5))
    candidates = np.flatnonzero(wanted)
    budget = min((cfg['max_vertices'] - len(surface.rest)) // 3,
                 (cfg['max_triangles'] - len(surface.faces)) // 6)
    if len(candidates) > max(budget, 0):
        candidates = candidates[np.argsort(score[candidates])[::-1][:max(budget, 0)]]
    result = np.zeros(len(surface.faces), bool)
    result[candidates] = True
    return result


def smooth_normals(vertices, faces):
    p = vertices[faces]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(n, faces[:, k], cross)
    return unit(n)


def deformed_normals(surface, reference_directions=None):
    """Apply actual geometric normal rotation to original custom smoothing."""
    a = smooth_normals(surface.rest, surface.faces)
    b = smooth_normals(surface.positions, surface.faces)
    v = np.cross(a, b)
    c = np.sum(a * b, axis=1)
    original = surface.directions if reference_directions is None else reference_directions
    rotated = original + np.cross(v, original)
    rotated += np.cross(v, np.cross(v, original)) / np.maximum(1 + c[:, None], 1e-6)
    return unit(rotated)


def settings(overrides=None):
    cfg = dict(levels=4, maximum_displacement_mm=1.5,
        step_limits_mm=[1., .5, .25, .125, .0625, .03125], screening_length_mm=20.,
        minimum_confidence=.025, freeze_parent_vertices=True,
        minimum_samples_to_split=12, minimum_edge_pixels=2.,
        residual_threshold=.008, chain_priority=.6,
        max_vertices=1050000, max_triangles=2000000,
        cg_rtol=1e-7, cg_maxiter=2500, millimeters_per_unit=None,
        assumed_face_width_mm=140., evidence_gain=1.)
    if overrides:
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError('Unknown displacement settings: ' + ', '.join(sorted(unknown)))
        cfg.update(overrides)
    if type(cfg['levels']) is not int or not 0 <= cfg['levels'] <= 5:
        raise ValueError('Subdivision levels must be an integer in 0..5.')
    for key in ('maximum_displacement_mm', 'screening_length_mm', 'minimum_edge_pixels',
                'residual_threshold', 'assumed_face_width_mm', 'cg_rtol'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError('A positive finite value is required for ' + key)
    if not 0 <= cfg['evidence_gain'] <= 4 or not 0 <= cfg['minimum_confidence'] <= 1:
        raise ValueError('Evidence gain/confidence is outside the supported range.')
    if not cfg['step_limits_mm'] or any(not np.isfinite(x) or x <= 0 for x in cfg['step_limits_mm']):
        raise ValueError('Positive finite per-level step limits are required.')
    if cfg['millimeters_per_unit'] is not None and (not np.isfinite(cfg['millimeters_per_unit']) or cfg['millimeters_per_unit'] <= 0):
        raise ValueError('millimeters_per_unit must be positive or null.')
    for key in ('max_vertices', 'max_triangles', 'minimum_samples_to_split', 'cg_maxiter'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key + ' must be a positive integer.')
    if type(cfg['freeze_parent_vertices']) is not bool:
        raise ValueError('freeze_parent_vertices must be a boolean.')
    return cfg
