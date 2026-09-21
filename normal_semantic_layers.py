"""Document 66: scale-aware line proposals, witnesses, and integrable layers.

Fold/Wrinkle/Fine are operational width/length classes, not a face-parsing model.
Millimeters inherit the mesh metric's explicitly declared or estimated scale.
Auxiliary relights verify displaced structures; they are not extra true cameras.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from normal_geometry import unit
from normal_geometry_field import (SurfaceDiffusion, integrate_height, surface_support,
                                   height_edges, curl_rms)
from normal_layers import isolated_blur
from normal_reference_fusion import weighted_median
from normal_structure import ALLOWED, line_features


SEMANTIC_NAMES = ('fold', 'wrinkle', 'fine', 'lips')
SEMANTIC_GAINS = dict(fold=2.2, wrinkle=.75, fine=.4, lips=1.)
SEMANTIC_CAPS = dict(fold=7., wrinkle=3., fine=1.5, lips=6.)
DEFAULT_SEMANTIC = dict(enabled=False, primary_strong=.70, primary_medium=.40,
    fallback_gain=.20, max_nodes_per_view=6000,
    policies=dict(
        fold=dict(radius_mm=5., min_length_mm=8., min_width_mm=1.4,
                  max_width_mm=18., scales_mm=[.7, 1.4, 2.8],
                  residual_sigmas_mm=[.55, 8.], height_sigmas_mm=[.55, 12.],
                  link_mm=1.5, spread_mm=1.1, orientation=.75, width_ratio=.32),
        wrinkle=dict(radius_mm=2., min_length_mm=2.5, min_width_mm=.45,
                     max_width_mm=3.5, scales_mm=[.25, .5, .9],
                     residual_sigmas_mm=[.22, 1.8], height_sigmas_mm=[.25, 1.8],
                     link_mm=.9, spread_mm=.5, orientation=.84, width_ratio=.5),
        fine=dict(radius_mm=.65, min_length_mm=1.5, min_width_mm=.22,
                  max_width_mm=1.3, scales_mm=[.12, .22, .38],
                  residual_sigmas_mm=[.10, .5], height_sigmas_mm=[.12, .5],
                  link_mm=.65, spread_mm=.24, orientation=.90, width_ratio=.65)))


def semantic_settings(options=None):
    cfg = copy.deepcopy(DEFAULT_SEMANTIC)
    supplied = options or {}
    unknown = set(supplied)-set(cfg)
    if unknown:
        raise ValueError('Unknown semantic_layers options: '+', '.join(sorted(unknown)))
    cfg.update({k: v for k, v in supplied.items() if k != 'policies'})
    for kind, policy in supplied.get('policies', {}).items():
        if kind not in cfg['policies'] or set(policy)-set(cfg['policies'][kind]):
            raise ValueError('Unknown semantic layer policy: '+kind)
        cfg['policies'][kind].update(policy)
    if type(cfg['enabled']) is not bool:
        raise ValueError('semantic_layers.enabled must be boolean.')
    for key in ('primary_strong', 'primary_medium', 'fallback_gain'):
        if isinstance(cfg[key], bool) or not np.isfinite(cfg[key]) or not 0 <= cfg[key] <= 1:
            raise ValueError('Invalid semantic_layers.'+key)
    if not 0 < cfg['primary_medium'] < cfg['primary_strong'] <= 1:
        raise ValueError('Primary strength thresholds must increase.')
    count = cfg['max_nodes_per_view']
    if isinstance(count, bool) or not isinstance(count, int) or not 10 <= count <= 24000:
        raise ValueError('max_nodes_per_view must be an integer in 10..24000.')
    for kind, policy in cfg['policies'].items():
        for key, value in policy.items():
            a = np.asarray(value, float)
            if not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(f'Invalid positive semantic policy: {kind}.{key}')
            if key.endswith('_sigmas_mm') and (a.shape != (2,) or np.any(np.diff(a) <= 0)):
                raise ValueError('Each semantic band needs two increasing physical scales.')
            if key == 'scales_mm' and (a.shape != (3,) or np.any(np.diff(a) <= 0)):
                raise ValueError('Each detector needs three increasing physical scales.')
        if not policy['min_width_mm'] < policy['max_width_mm']:
            raise ValueError('Structure width limits must increase.')
        if policy['radius_mm'] > 8 or policy['orientation'] > 1 or policy['width_ratio'] > 1:
            raise ValueError('Semantic matching policy exceeds its bounded range.')
    return cfg


def support_policy(kind, primary_strength, auxiliary_scores, total_views, options=None):
    """A/B/C fold witnesses, two-view wrinkles, and >=3 / 75% fine witnesses."""
    cfg = semantic_settings(options)
    primary = np.asarray(primary_strength, float)
    auxiliary = np.asarray(auxiliary_scores, float)
    if auxiliary.ndim != primary.ndim+1 or auxiliary.shape[1:] != primary.shape:
        raise ValueError('Auxiliary scores must have a leading reference axis.')
    if total_views != auxiliary.shape[0]+1:
        raise ValueError('Reference count does not match witness scores.')
    count = (auxiliary >= .68).sum(0)
    if kind == 'fold':
        strong_aux = (auxiliary >= .75).sum(0)
        accepted = ((primary >= cfg['primary_strong']) & (strong_aux >= 1))
        accepted |= (primary >= cfg['primary_medium']) & (count >= 2)
        accepted |= count >= 3
        # Explicitly tagged fallback, never counted as corroborated geometry.
        fallback = (~accepted) & (primary >= cfg['primary_strong'])
    elif kind == 'wrinkle':
        accepted = count+(primary >= cfg['primary_medium']) >= 2
        fallback = np.zeros(primary.shape, bool)
    elif kind == 'fine':
        accepted = count+(primary >= cfg['primary_medium']) >= max(3, int(np.ceil(.75*total_views)))
        fallback = np.zeros(primary.shape, bool)
    else:
        raise ValueError('Unknown structure class: '+kind)
    return accepted, fallback


def _metric_distance(delta, xx, xy, yy):
    return np.sqrt(np.maximum(xx*delta[..., 0]**2+2*xy*delta[..., 0]*delta[..., 1]
                              +yy*delta[..., 1]**2, 0))


def _allowed_semantics():
    allowed = np.eye(16, dtype=bool)
    for a, b in ALLOWED:
        allowed[a, b] = allowed[b, a] = True
    return allowed


def _path_valid(start, end, valid, steps):
    good = np.ones(start.shape[:-1], bool)
    for t in np.linspace(0, 1, max(2, int(steps))):
        xy = np.rint(start*(1-t)+end*t).astype(int)
        good &= valid[xy[..., 1], xy[..., 0]]
    return good


def _link_nodes(points, tangent, widths, polarity, valid, metric, semantics, policy, pixel_mm,
                bridge_evidence=None):
    """Degree-two, acyclic, physically measured chains, including allowed boundaries."""
    n = len(points)
    lengths = np.zeros(n, np.float32)
    ids = np.arange(n, dtype=np.int32)
    if n < 2:
        return lengths, ids, []
    dist, near = cKDTree(points).query(points, k=min(9, n),
        distance_upper_bound=max(3., policy['link_mm']/pixel_mm*1.8))
    left = np.repeat(np.arange(n), near.shape[1]-1)
    right = near[:, 1:].ravel()
    take = (right < n) & (left < right)
    left, right = left[take], right[take]
    if not len(left):
        return lengths, ids, []
    delta = points[right]-points[left]
    ay, ax = points[left, 1].astype(int), points[left, 0].astype(int)
    by, bx = points[right, 1].astype(int), points[right, 0].astype(int)
    xx, xy, yy = [.5*(metric[k][ay, ax]+metric[k][by, bx]) for k in ('xx', 'xy', 'yy')]
    physical = _metric_distance(delta, xx, xy, yy)
    norm = np.maximum(np.linalg.norm(delta, axis=-1), 1e-9)
    direction = np.abs(np.sum(tangent[left]*tangent[right], -1))
    forward = np.minimum(np.abs(np.sum(tangent[left]*delta, -1)),
                         np.abs(np.sum(tangent[right]*delta, -1)))/norm
    width_ratio = np.minimum(widths[left], widths[right])/np.maximum(widths[left], widths[right])
    good = (physical <= policy['link_mm']) & (direction >= .78) & (forward >= .65)
    good &= (width_ratio >= .45) & (polarity[left] == polarity[right])
    if semantics is not None:
        good &= _allowed_semantics()[semantics[ay, ax], semantics[by, bx]]
    selected = np.flatnonzero(good)
    if len(selected):
        good[selected] &= _path_valid(points[left[selected]], points[right[selected]], valid,
                                       np.ceil(norm[selected].max()*2)+1)
    score = direction+forward-.15*physical/policy['link_mm']
    degree = np.zeros(n, np.int8)
    parent = np.arange(n)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = []
    good_indices = np.flatnonzero(good)
    for index in good_indices[np.argsort(-score[good])]:
        a, b = int(left[index]), int(right[index])
        ra, rb = root(a), root(b)
        if degree[a] >= 2 or degree[b] >= 2 or ra == rb:
            continue
        parent[ra] = rb
        degree[a] += 1; degree[b] += 1
        edges.append((a, b, float(physical[index])))
    # A broad line can cross a low-response patch or a scale-selection boundary.
    # Join existing chain ends, never branch them, using weaker continuous line
    # evidence along the entire gap. This is not blind morphological closing.
    if bridge_evidence is not None:
        endpoints = np.flatnonzero(degree == 1)
        gap_mm = policy['link_mm']*2.2
        if len(endpoints) > 1:
            end_points = points[endpoints]
            distances, neighbors = cKDTree(end_points).query(end_points, k=min(16, len(endpoints)),
                distance_upper_bound=gap_mm/pixel_mm*1.8)
            candidates = []
            for row in range(len(endpoints)):
                a = int(endpoints[row])
                for column in range(1, neighbors.shape[1]):
                    j = int(neighbors[row, column])
                    if j >= len(endpoints):
                        continue
                    b = int(endpoints[j])
                    if a >= b or polarity[a] != polarity[b] or root(a) == root(b):
                        continue
                    delta = points[b]-points[a]
                    size = max(float(np.linalg.norm(delta)), 1e-9)
                    ya, xa = points[a, 1].astype(int), points[a, 0].astype(int)
                    yb, xb = points[b, 1].astype(int), points[b, 0].astype(int)
                    xx, xy, yy = [.5*(metric[k][ya, xa]+metric[k][yb, xb]) for k in ('xx', 'xy', 'yy')]
                    length = float(_metric_distance(delta, xx, xy, yy))
                    if length > gap_mm or abs(float(tangent[a]@tangent[b])) < .80:
                        continue
                    if min(abs(float(tangent[a]@delta)), abs(float(tangent[b]@delta)))/size < .73:
                        continue
                    if min(widths[a], widths[b])/max(widths[a], widths[b]) < .5:
                        continue
                    if semantics is not None and not _allowed_semantics()[semantics[ya, xa], semantics[yb, xb]]:
                        continue
                    path = np.rint(np.linspace(points[a], points[b], int(np.ceil(size*2))+1)).astype(int)
                    px, py = path[:, 0], path[:, 1]
                    if not valid[py, px].all() or bridge_evidence[py, px].mean() < .65:
                        continue
                    candidates.append((length, a, b))
            for length, a, b in sorted(candidates):
                ra, rb = root(a), root(b)
                if degree[a] != 1 or degree[b] != 1 or ra == rb:
                    continue
                parent[ra] = rb; degree[a] += 1; degree[b] += 1
                edges.append((a, b, length))
    ids = np.array([root(i) for i in range(n)], np.int32)
    total = np.zeros(n, np.float32)
    for a, b, length in edges:
        total[ids[a]] += length
    lengths = total[ids]
    return lengths, ids, edges


def detect_scale_nodes(field, weight, valid, metric, kind, options=None, semantics=None):
    cfg = semantic_settings(options); policy = cfg['policies'][kind]
    pixel_mm = float(np.median(np.sqrt(metric['area'][valid]))) if valid.any() else 1.
    scales = tuple(max(.75, s/pixel_mm) for s in policy['scales_mm'])
    feature = line_features(field, scales)
    response, tangent = feature['response'], feature['tangent']
    y, x = np.indices(field.shape, dtype=np.float32)
    coordinates = ([y-tangent[..., 0], x+tangent[..., 1]],
                   [y+tangent[..., 0], x-tangent[..., 1]])
    good = valid & (weight > .001)
    threshold = np.full(field.shape, .00010 if kind == 'fold' else .00006, np.float32)
    groups = semantics if semantics is not None else np.ones(valid.shape, np.int16)
    for label in np.unique(groups[good]):
        region = good & (groups == label)
        threshold[region] = max(float(threshold[region][0]), float(np.quantile(response[region], .52)))
    peaks = good & (response > threshold)
    amplitude = np.abs(field)
    # Gaussian Hessian tails can extend a tiny isolated mark into a long line.
    # The centerline must also carry actual residual amplitude at that location.
    peaks &= amplitude > np.maximum(.00008, .5*threshold)
    if kind == 'fold':
        # Opposite-sign Hessian lobes beside a ridge are not separate grooves.
        # Keep the sign of the actual residual and reject mask-cut edge centers.
        peaks &= field*feature['polarity'] > np.maximum(.00008, .5*threshold)
        peaks &= ndi.distance_transform_edt(good) >= max(2., policy['min_width_mm']/(2*pixel_mm))
    # Broad structures need not peak at exactly the same raw texel as their
    # scale-space curvature. Fine wrinkles retain the stricter two-part NMS.
    for coords in coordinates:
        peaks &= response >= ndi.map_coordinates(response, coords, order=1, mode='nearest')
        if kind != 'fold':
            peaks &= amplitude >= .995*ndi.map_coordinates(amplitude, coords, order=1, mode='nearest')
    # Keep a thin centerline, with enough samples for physical chain lengths.
    yy, xx = np.nonzero(peaks)
    if len(xx) > cfg['max_nodes_per_view']:
        tile = (yy//3)*((field.shape[1]+2)//3)+xx//3
        order = np.argsort(-response[yy, xx], kind='stable')
        _, first = np.unique(tile[order], return_index=True)
        keep = order[first]
        keep = keep[np.argsort(-response[yy[keep], xx[keep]])[:cfg['max_nodes_per_view']]]
        yy, xx = yy[keep], xx[keep]
    points = np.column_stack([xx, yy]).astype(float)
    t = tangent[yy, xx]
    nx, ny = t[:, 1], -t[:, 0]
    gxx, gxy, gyy = [metric[k][yy, xx] for k in ('xx', 'xy', 'yy')]
    det = np.maximum(gxx*gyy-gxy*gxy, 1e-20)
    transverse_step = np.sqrt(det/np.maximum(gyy*nx*nx-2*gxy*nx*ny+gxx*ny*ny, 1e-20))
    widths = 2.355*feature['scale'][yy, xx]*transverse_step
    strength = np.clip(response[yy, xx]/np.maximum(2*threshold[yy, xx], 1e-8), 0, 1)
    lengths, chain_ids, edges = _link_nodes(points, t, widths, feature['polarity'][yy, xx],
        good, metric, semantics, policy, pixel_mm,
        bridge_evidence=(response > threshold*.3) & (amplitude > .00008) if kind == 'fold' else None)
    eligible = ((lengths >= policy['min_length_mm']) & (widths >= policy['min_width_mm'])
                & (widths <= policy['max_width_mm']))
    return dict(points=points, tangent=t, width_mm=widths, strength=strength,
                length_mm=lengths, chain_id=chain_ids, edges=edges, eligible=eligible,
                pixel_mm=pixel_mm, scales_pixels=list(scales))


def match_scale_structures(fields, weights, valid, metric, kind, options=None, semantics=None):
    cfg = semantic_settings(options); policy = cfg['policies'][kind]
    fields, weights = np.asarray(fields, np.float32), np.asarray(weights, np.float32)
    if fields.ndim != 3 or fields.shape != weights.shape or fields.shape[1:] != valid.shape:
        raise ValueError('Structure evidence shapes do not agree.')
    if not np.isfinite(fields).all() or not np.isfinite(weights).all():
        raise ValueError('Structure evidence must be finite.')
    m, h, w = fields.shape
    detected = [detect_scale_nodes(f, q, valid, metric, kind, cfg, semantics) for f, q in zip(fields, weights)]
    pixel_mm = detected[0]['pixel_mm']
    # Primary proposes first. Auxiliary proposals cover absent/weak primary lines;
    # those must still pass the 3-auxiliary rule to become a corroborated fold.
    anchors, origins, source_ids = [], [], []
    existing = np.empty((0, 2))
    for view, nodes in enumerate(detected):
        ids = np.flatnonzero(nodes['eligible'])
        points = nodes['points'][ids]
        if view and len(existing) and len(points):
            distance = cKDTree(existing).query(points, k=1)[0]
            keep = distance > max(2., policy['min_width_mm']/pixel_mm*.45)
            ids, points = ids[keep], points[keep]
        anchors.extend(points); origins.extend([view]*len(points)); source_ids.extend(ids)
        if len(points):
            existing = np.concatenate([existing, points])
    anchor = np.asarray(anchors, float).reshape(-1, 2)
    origin = np.asarray(origins, int); source_ids = np.asarray(source_ids, int)
    count = len(anchor)
    offsets = np.zeros((m, count, 2), np.float32)
    scores = np.zeros((m, count), np.float32)
    strengths = np.zeros_like(scores)
    tangent = np.asarray([detected[v]['tangent'][j] for v, j in zip(origin, source_ids)]).reshape(-1, 2)
    widths = np.asarray([detected[v]['width_mm'][j] for v, j in zip(origin, source_ids)], np.float32)
    lengths = np.asarray([detected[v]['length_mm'][j] for v, j in zip(origin, source_ids)], np.float32)
    ay, ax = anchor[:, 1].astype(int), anchor[:, 0].astype(int)
    radius_px = max(2., policy['radius_mm']/pixel_mm*1.8)
    allowed = _allowed_semantics()
    for view, nodes in enumerate(detected):
        own = origin == view
        scores[view, own] = 1.
        strengths[view, own] = nodes['strength'][source_ids[own]]
        # Witnesses may be shorter fragments of an already long proposal.
        # Only the proposal itself must supply the full physical chain length.
        witness = nodes['eligible']
        if kind == 'fold':
            witness = ((nodes['length_mm'] >= min(2.0, policy['min_length_mm']))
                & (nodes['width_mm'] >= policy['min_width_mm'])
                & (nodes['width_mm'] <= policy['max_width_mm']))
        ids = np.flatnonzero(witness)
        targets = nodes['points'][ids]
        if not count or not len(targets):
            continue
        distances, near = cKDTree(targets).query(anchor, k=min(48, len(targets)), distance_upper_bound=radius_px)
        if distances.ndim == 1:
            distances, near = distances[:, None], near[:, None]
        safe = np.minimum(near, len(targets)-1)
        candidate = targets[safe]; node_id = ids[safe]
        yy, xx = candidate[..., 1].astype(int), candidate[..., 0].astype(int)
        delta = candidate-anchor[:, None]
        gxx, gxy, gyy = [.5*(metric[k][ay, ax, None]+metric[k][yy, xx]) for k in ('xx', 'xy', 'yy')]
        mm = _metric_distance(delta, gxx, gxy, gyy)
        orientation = np.abs(np.sum(tangent[:, None]*nodes['tangent'][node_id], -1))
        ratio = np.minimum(widths[:, None], nodes['width_mm'][node_id])/np.maximum(widths[:, None], nodes['width_mm'][node_id])
        good = np.isfinite(distances) & (mm <= policy['radius_mm'])
        good &= (orientation >= policy['orientation']) & (ratio >= policy['width_ratio'])
        if semantics is not None:
            good &= allowed[semantics[ay, ax, None], semantics[yy, xx]]
        # Test only plausible correspondences, at <= half-pixel path spacing.
        rows, cols = np.nonzero(good)
        if len(rows):
            good[rows, cols] &= _path_valid(anchor[rows], candidate[rows, cols],
                valid & (weights[view] > .001), np.ceil(distances[rows, cols].max()*2)+1)
        score = .30*np.exp(-.5*(mm/policy['radius_mm'])**2)+.40*orientation+.30*ratio
        score = np.where(good, score, -1)
        best = np.argmax(score, axis=1); rows = np.arange(count)
        take = (score[rows, best] >= .68) & ~own
        scores[view, take] = score[rows, best][take]
        strengths[view, take] = nodes['strength'][node_id[rows, best][take]]
        offsets[view, take] = delta[rows, best][take]
    primary = strengths[0]*scores[0] if count else np.zeros(0)
    accepted, fallback = support_policy(kind, primary, scores[1:], m, cfg)
    # Missing auxiliary lines lower an unsupported proposal; observable nonmatches
    # are not relabeled as agreement. Single-image fallback remains explicit.
    if count:
        observable = np.stack([q[ay, ax] > .15 for q in weights[1:]]) if m > 1 else np.empty((0, count), bool)
        contradicted = ((scores[1:] < .68) & observable).sum(0) >= 2
        fallback &= ~contradicted
    vote = scores.copy()
    if count:
        vote[0] *= 1.5
        latent = np.column_stack([weighted_median(offsets[..., j], vote) for j in range(2)])
        center = anchor+latent
        safe = _path_valid(anchor, center, valid, np.ceil(radius_px*2)+1)
        latent[~safe] = 0
    else:
        latent = np.zeros((0, 2), np.float32)
    edges = []
    for view, nodes in enumerate(detected):
        lookup = {int(j): i for i, (v, j) in enumerate(zip(origin, source_ids)) if v == view}
        for a, b, length in nodes['edges']:
            if a in lookup and b in lookup:
                ia, ib = lookup[a], lookup[b]
                if accepted[ia] and accepted[ib]:
                    edges.append([ia, ib, length])
    report = dict(kind=kind, policy=policy, primary_proposals=int((origin == 0).sum()),
        auxiliary_proposals=int((origin != 0).sum()), anchor_nodes=count,
        consensus_nodes=int(accepted.sum()), fallback_nodes=int(fallback.sum()),
        rejected_nodes=int((~accepted & ~fallback).sum()),
        matched_nodes_per_view=[int((s >= .68).sum()) for s in scores],
        eligible_nodes_per_view=[int(n['eligible'].sum()) for n in detected],
        radius_mm=policy['radius_mm'], centerline_edges=edges,
        position_rule='Primary-first proposals; weighted-median latent centers; physical displacement and protected-path checks.',
        width_definition='2.355 * selected Hessian sigma, converted with the local inverse surface metric; an estimated width proxy.',
        nodes=[dict(x=float(p[0]+d[0]), y=float(p[1]+d[1]), width_mm=float(width),
                    chain_length_mm=float(length), primary_strength=float(ps), views=int((s >= .68).sum()),
                    accepted=bool(a), fallback=bool(f), origin=int(v))
               for p, d, width, length, ps, s, a, f, v in
               zip(anchor, latent, widths, lengths, primary, scores.T, accepted, fallback, origin)])
    return dict(anchor=anchor, offsets=offsets, scores=scores, strengths=strengths,
                latent=latent, tangent=tangent, accepted=accepted, fallback=fallback,
                report=report, pixel_mm=pixel_mm)


def spread_proposals(fields, observation_weights, valid, metric, matches, kind, options=None):
    """Continuous compact line support and signed, mask-safe local correspondence."""
    cfg = semantic_settings(options); policy = cfg['policies'][kind]
    m, h, w = fields.shape
    keep = matches['accepted'] | matches['fallback']
    empty = dict(fields=np.zeros_like(fields), weights=np.zeros_like(fields),
                 accepted=np.zeros((h, w), np.float32), fallback=np.zeros((h, w), np.float32),
                 tangent=np.zeros((h, w, 2), np.float32), max_shift_mm=0.)
    if not keep.any():
        return empty
    target = matches['anchor'][keep]+matches['latent'][keep]
    ty, tx = np.rint(target[:, 1]).astype(int), np.rint(target[:, 0]).astype(int)
    sigma = max(1., policy['spread_mm']/matches['pixel_mm'])
    seed = np.zeros((h, w), np.float32)
    np.add.at(seed, (ty, tx), 1.)
    denominator = ndi.gaussian_filter(seed, sigma, mode='constant', truncate=3.)
    density = np.clip(denominator*(2*np.pi*sigma*sigma), 0, 1)*valid
    def spread(values, source_weight=None):
        source_weight = np.ones(len(tx), np.float32) if source_weight is None else source_weight
        numerator = np.zeros((h, w), np.float32)
        np.add.at(numerator, (ty, tx), values*source_weight)
        num = ndi.gaussian_filter(numerator, sigma, mode='constant', truncate=3.)
        if np.all(source_weight == 1):
            den = denominator
        else:
            mass = np.zeros((h, w), np.float32)
            np.add.at(mass, (ty, tx), source_weight)
            den = ndi.gaussian_filter(mass, sigma, mode='constant', truncate=3.)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-10)*valid
    accepted = density*spread(matches['accepted'][keep].astype(np.float32))
    fallback = density*spread(matches['fallback'][keep].astype(np.float32))*(1-accepted)
    tangent = matches['tangent'][keep]
    cos2 = spread(tangent[:, 0]**2-tangent[:, 1]**2)
    sin2 = spread(2*tangent[:, 0]*tangent[:, 1])
    phi = .5*np.arctan2(sin2, cos2)
    dense_tangent = np.stack([np.cos(phi), np.sin(phi)], -1).astype(np.float32)
    yy, xx = np.indices((h, w), dtype=np.float32)
    warped = np.zeros_like(fields); weights = np.zeros_like(fields)
    maximum_shift = 0.
    for view in range(m):
        score = matches['scores'][view, keep]
        if not np.any(score > 0):
            continue
        delta = matches['offsets'][view, keep]-matches['latent'][keep]
        dx, dy = spread(delta[:, 0], score), spread(delta[:, 1], score)
        mm = _metric_distance(np.stack([dx, dy], -1), metric['xx'], metric['xy'], metric['yy'])
        factor = np.minimum(1., policy['radius_mm']/np.maximum(mm, 1e-8))
        dx *= factor; dy *= factor
        support = density*spread(score)
        path = valid.copy()
        steps = max(2, int(np.ceil(max(float(np.abs(dx).max()), float(np.abs(dy).max()))*2))+1)
        for t in np.linspace(0, 1, steps):
            path &= ndi.map_coordinates(valid.astype(np.float32), [yy+t*dy, xx+t*dx],
                order=1, mode='constant', cval=0) >= 1-1e-6
        warped[view] = ndi.map_coordinates(fields[view], [yy+dy, xx+dx], order=1, mode='constant')
        weights[view] = ndi.map_coordinates(observation_weights[view], [yy+dy, xx+dx],
            order=1, mode='constant')*support*path
        warped[view, ~path] = 0
        if np.any(weights[view] > .001):
            maximum_shift = max(maximum_shift, float((mm*factor)[weights[view] > .001].max()))
    return dict(fields=warped, weights=weights, accepted=accepted, fallback=fallback,
                tangent=dense_tangent, max_shift_mm=maximum_shift)


def fit_transverse_slope(base, direction, residuals, weights, lights, ambients, cap_degrees):
    """Fit one transverse slope plus an albedo nuisance term using signed lights.

    Two independent lights can constrain these two variables. Rank-deficient
    observations have zero fit quality, regardless of line-matching agreement.
    This directional small-slope model does not infer a full 2D normal from 2 images.
    """
    residuals, weights = np.asarray(residuals, np.float32), np.asarray(weights, np.float32)
    if residuals.shape != weights.shape or residuals.shape[1:] != base.shape[:2]:
        raise ValueError('Directional photometric evidence shapes do not agree.')
    a = np.stack([direction@light for light in lights])
    b = np.stack([ambient+np.maximum(base@light, 0) for ambient, light in zip(ambients, lights)])
    robust = weights.copy()
    slope = np.zeros(base.shape[:2], np.float32); albedo = np.zeros_like(slope)
    for _ in range(4):
        aa, ab, bb = (np.sum(robust*t, axis=0) for t in (a*a, a*b, b*b))
        ar, br = (np.sum(robust*t*residuals, axis=0) for t in (a, b))
        determinant = (aa+.0002)*(bb+1e-8)-ab*ab
        slope = (ar*(bb+1e-8)-br*ab)/np.maximum(determinant, 1e-12)
        albedo = (br*(aa+.0002)-ar*ab)/np.maximum(determinant, 1e-12)
        error = residuals-a*slope-b*albedo
        robust = weights/np.sqrt(1+(error/.025)**2)
    condition = np.clip((aa*bb-ab*ab)/np.maximum(aa*bb, 1e-12), 0, 1)
    rms = np.sqrt(np.sum(robust*error*error, axis=0)/np.maximum(robust.sum(0), 1e-10))
    independent = ((weights > .001).sum(0) >= 2) & (condition > .025)
    quality = np.clip((condition-.025)/.20, 0, 1)*np.exp(-(rms/.04)**2)*independent
    cap = np.tan(np.deg2rad(cap_degrees))
    slope = cap*np.tanh(slope/cap)
    slope[~independent] = 0
    # The primary supplies a bounded proposal profile. If it is unlit/missing,
    # use the best observed directional view, still gated by the witness policy.
    best = np.argmax(weights*np.abs(a), axis=0)
    primary_usable = (weights[0] > .001) & (np.abs(a[0]) > .04)
    best = np.where(primary_usable, 0, best)
    aa_primary = np.take_along_axis(a, best[None], 0)[0]
    rr_primary = np.take_along_axis(residuals, best[None], 0)[0]
    prior = rr_primary*aa_primary/(aa_primary*aa_primary+.003)
    prior = cap*np.tanh(prior/cap)
    observed = weights.sum(0) > .001
    prior[~observed] = 0
    report = dict(method='Signed transverse slope + shared relative-albedo nuisance; robust 2-variable fit',
        independent_fit_samples=int(independent.sum()),
        mean_fit_quality=float(quality[observed].mean()) if observed.any() else 0.,
        mean_fit_rms=float(rms[observed].mean()) if observed.any() else 0.,
        maximum_prior_angle_degrees=float(np.degrees(np.arctan(np.abs(prior))).max()),
        minimum_independent_lights=2, full_2d_normal_inferred=False)
    return slope.astype(np.float32), prior.astype(np.float32), quality.astype(np.float32), report


def build_semantic_heights(base, arrays, domain, metric, geometry_options, options=None, caps=None):
    cfg = semantic_settings(options); caps = dict(SEMANTIC_CAPS, **(caps or {}))
    observations = arrays['geometry_observations']
    observation_weights = arrays['geometry_confidences']
    lights, ambients = arrays['geometry_lights'], arrays['geometry_ambients']
    valid = surface_support(domain & metric['valid'] & (observation_weights > .001).any(0))
    pixel_mm = float(np.median(np.sqrt(metric['area'][valid]))) if valid.any() else 1.
    raw = np.stack([image-(a+np.maximum(base@light, 0))
                    for image, light, a in zip(observations, lights, ambients)])
    raw[:, ~valid] = 0
    semantics = arrays.get('fusion_semantics')
    operator = SurfaceDiffusion(valid, metric)
    bands, confidences, diagnostics, reports = {}, {}, {}, {}
    height_sum = np.zeros(domain.shape, float)
    for kind in ('fold', 'wrinkle', 'fine'):
        print('  primary proposals / displaced witnesses:', kind, flush=True)
        policy = cfg['policies'][kind]
        lo, hi = [max(.55, s/pixel_mm) for s in policy['residual_sigmas_mm']]
        fields = np.stack([isolated_blur(image, valid & (w > .001), lo)
                           -isolated_blur(image, valid & (w > .001), hi)
                           for image, w in zip(raw, observation_weights)])
        matches = match_scale_structures(fields, observation_weights, valid, metric, kind, cfg, semantics)
        dense = spread_proposals(fields, observation_weights, valid, metric, matches, kind, cfg)
        tangent = dense['tangent']
        line = metric['jx']*tangent[..., :1]+metric['jy']*tangent[..., 1:]
        direction = unit(np.cross(base, line)).astype(np.float32)
        fitted, primary, quality, photo_report = fit_transverse_slope(base, direction,
            dense['fields'], dense['weights'], lights, ambients, caps[kind])
        accepted, fallback = dense['accepted'], dense['fallback']
        structure_weight = accepted*quality
        primary_weight = accepted*(1-quality)+cfg['fallback_gain']*fallback
        support = structure_weight+primary_weight
        scalar = structure_weight*fitted+primary_weight*primary
        slope = direction*scalar[..., None]
        gx, gy = -np.sum(slope*metric['jx'], -1), -np.sum(slope*metric['jy'], -1)
        gx[~valid] = 0; gy[~valid] = 0
        if np.any(support > .001):
            # Confidence controls projection weights; support has already gated
            # the proposed slope. Do not multiply the final height by it again.
            weight = (.05+.95*support)*valid
            height, projection = integrate_height(gx, gy, weight, valid, metric, geometry_options)
            small, small_report = operator.smooth(height, policy['height_sigmas_mm'][0], geometry_options)
            large, large_report = operator.smooth(height, policy['height_sigmas_mm'][1], geometry_options)
            height = small-large
            confidence = (.05+.9*support)*valid
        else:
            height = np.zeros(domain.shape, float)
            confidence = np.zeros(domain.shape, np.float32)
            projection = dict(converged=True, active_samples=0, curl_before=0., curl_after=0.)
            small_report = large_report = dict(converged=True, iterations=0, relative_residual=0.)
        height[~valid] = 0
        bands[kind] = height; confidences[kind] = confidence.astype(np.float32)
        height_sum += height
        active = support > .01
        diagnostics[kind+'_proposal_support'] = np.clip(accepted+fallback, 0, 1)
        diagnostics[kind+'_confirmed_support'] = accepted
        diagnostics[kind+'_fallback_support'] = fallback
        diagnostics[kind+'_weight_structure'] = structure_weight
        diagnostics[kind+'_weight_primary'] = primary_weight
        diagnostics[kind+'_photo_quality'] = quality*valid
        reports[kind] = dict(matching=matches['report'], photometric=photo_report,
            integrability=projection, physical_filter=dict(sigmas_mm=policy['height_sigmas_mm'],
                small=small_report, large=large_report), max_warp_mm=dense['max_shift_mm'],
            supported_samples=int(active.sum()), confirmed_samples=int((accepted > .1).sum()),
            structure_dominant_samples=int(((structure_weight > primary_weight) & active).sum()),
            mean_primary_fraction=float((primary_weight[active]/np.maximum(support[active], 1e-8)).mean()) if active.any() else 0.,
            mean_height_abs_mm=float(np.mean(np.abs(height[valid]))) if valid.any() else 0.,
            curl_after=curl_rms(*height_edges(height), valid))
        print('    accepted / fallback / rejected:', matches['report']['consensus_nodes'],
            matches['report']['fallback_nodes'], matches['report']['rejected_nodes'], flush=True)
    combined_confidence = np.maximum.reduce(list(confidences.values()))
    report = dict(enabled=True, mode='scale-aware-primary-proposals', settings=cfg,
        matching={k: r['matching'] for k, r in reports.items()}, layers=reports,
        primary_reference_index=0, pore_reconstruction=False, primary_dominant_samples=0,
        mean_primary_weight=reports['fold']['mean_primary_fraction'],
        multilight_dominant_samples=reports['fold']['structure_dominant_samples'],
        support_note='Support/fit quality are heuristic evidence scores, not calibrated probabilities.',
        continuity_note='Physical chain lengths and widths; relaxed displaced witnesses for folds; independent screened scalar heights.',
        limitations=['Fold/Wrinkle/Fine are width/chain classes, not guaranteed anatomical labels.',
            'Detection scales use the median chart scale; width, length, matching and height filtering use the surface metric.',
            'Primary-only fold profiles remain bounded shading priors, not recovered true geometry.',
            'Directional two-light fits estimate one transverse slope and albedo, not a complete independent normal.',
            'Local linked latent centers are exported; no global spline or unified cross-section optimization.'])
    return dict(bands=bands, layer_confidence=confidences, confidence=combined_confidence,
                valid=valid, height=height_sum, diagnostic=diagnostics, report=report)
