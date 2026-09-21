"""Document 55: bounded structure matching and an explicit primary-image prior.

The primary prior is a regularized shading interpretation, not measured shape.
Only mid-band evidence is locally warped; raw photometry and the mesh stay fixed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from normal_geometry import unit
from normal_geometry_field import (integrate_height, solve_geometry, surface_support,
                                   curl_rms, height_edges)
from normal_layers import isolated_blur, slope_from_base
from normal_structure import ALLOWED, line_features


DEFAULT_FUSION = dict(enabled=True, match_radius_pixels=5., blend_sigma_pixels=3.,
    confidence_low=.35, confidence_high=.75, primary_max_angle_degrees=5.,
    primary_low_fraction=.25, structure_fraction=.35, brightness_bins=4)


def fusion_settings(options=None):
    cfg = dict(DEFAULT_FUSION, **(options or {}))
    if type(cfg['enabled']) is not bool:
        raise ValueError('reference_fusion.enabled must be boolean.')
    for key in DEFAULT_FUSION.keys()-{'enabled', 'brightness_bins'}:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError('Invalid reference_fusion.'+key)
    if not 0 < cfg['match_radius_pixels'] <= 12 or not 0 < cfg['blend_sigma_pixels'] <= 20:
        raise ValueError('Fusion displacement/smoothing exceeds its bounded range.')
    if not 0 <= cfg['confidence_low'] < cfg['confidence_high'] <= 1:
        raise ValueError('Fusion confidence thresholds must increase within [0, 1].')
    if not 0 < cfg['primary_max_angle_degrees'] <= 12:
        raise ValueError('Primary angular cap must be in (0, 12].')
    if not 0 <= cfg['primary_low_fraction'] <= 1 or not 0 <= cfg['structure_fraction'] <= .5:
        raise ValueError('Invalid primary/structure prior weight.')
    bins = cfg['brightness_bins']
    if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 8:
        raise ValueError('brightness_bins must be an integer in 1..8.')
    return cfg


def weighted_median(values, weights):
    """Column-wise robust position estimator; zero-weight views have no vote."""
    values, weights = np.asarray(values), np.asarray(weights)
    if values.shape != weights.shape or values.ndim != 2 or np.any(weights < 0):
        raise ValueError('Matching values and nonnegative weights are required.')
    order = np.argsort(values, axis=0, kind='stable')
    v, w = np.take_along_axis(values, order, 0), np.take_along_axis(weights, order, 0)
    index = np.argmax(np.cumsum(w, axis=0) >= .5*w.sum(0), axis=0)
    return np.take_along_axis(v, index[None], 0)[0]


def _ridge_nodes(field, confidence, valid):
    feature = line_features(field, (2., 4., 8.))
    response, tangent = feature['response'], feature['tangent']
    y, x = np.indices(field.shape, dtype=np.float32)
    # Suppress across, not along, the line. Keep the original native pixel scale.
    ahead = ndi.map_coordinates(response, [y-tangent[..., 0], x+tangent[..., 1]], order=1)
    behind = ndi.map_coordinates(response, [y+tangent[..., 0], x-tangent[..., 1]], order=1)
    good = valid & (confidence > .015)
    threshold = max(.00015, float(np.quantile(response[good], .6))) if good.any() else np.inf
    peaks = good & (response > threshold) & (response >= ahead) & (response >= behind)
    # A Hessian has strong opposite-sign sidelobes beside an actual ridge.
    # Require a residual-amplitude center as well, otherwise a nearby sidelobe
    # can impersonate a shifted wrinkle even when the real center is occluded.
    amplitude = np.abs(field)
    aa = ndi.map_coordinates(amplitude, [y-tangent[..., 0], x+tangent[..., 1]], order=1)
    ab = ndi.map_coordinates(amplitude, [y+tangent[..., 0], x-tangent[..., 1]], order=1)
    peaks &= (amplitude >= aa*.999) & (amplitude >= ab*.999)
    peaks &= ((x.astype(int) % 3 == 0) | (y.astype(int) % 3 == 0))
    yy, xx = np.nonzero(peaks)
    if len(xx) > 24000:
        selected = np.argsort(response[yy, xx])[-24000:]
        yy, xx = yy[selected], xx[selected]
    return np.column_stack([xx, yy]).astype(float), feature


def match_mid_structures(residuals, weights, valid, options=None, semantics=None):
    """Match nearby line centers; opposite-light intensity signs may differ.

    At least three views must support a latent center. Position, orientation,
    width and scale stability contribute separately. No match crosses a hole.
    """
    cfg = fusion_settings(options)
    fields, weights = np.asarray(residuals, np.float32), np.asarray(weights, np.float32)
    if fields.ndim != 3 or fields.shape != weights.shape or fields.shape[1:] != valid.shape:
        raise ValueError('Structure evidence shapes do not match.')
    if not np.isfinite(fields).all() or not np.isfinite(weights).all():
        raise ValueError('Nonfinite structure evidence.')
    m, h, w = fields.shape
    anchor, first = _ridge_nodes(fields[0], weights[0], valid)
    count = len(anchor)
    offsets = np.zeros((m, count, 2), np.float32)
    scores = np.zeros((m, count), np.float32)
    scores[0] = 1
    widths = np.ones((m, count), np.float32)
    ay, ax = anchor[:, 1].astype(int), anchor[:, 0].astype(int)
    widths[0] = first['scale'][ay, ax]
    radius = cfg['match_radius_pixels']
    parts = ndi.label(valid)[0]
    allowed = np.eye(16, dtype=bool)
    for a, b in ALLOWED:
        allowed[a, b] = allowed[b, a] = True
    for i in range(1, m):
        nodes, feature = _ridge_nodes(fields[i], weights[i], valid)
        if not count or not len(nodes):
            continue
        distances, near = cKDTree(nodes).query(anchor, k=min(8, len(nodes)), distance_upper_bound=radius)
        if distances.ndim == 1:
            distances, near = distances[:, None], near[:, None]
        safe = np.minimum(near, len(nodes)-1)
        targets = nodes[safe]
        yy, xx = targets[..., 1].astype(int), targets[..., 0].astype(int)
        orientation = np.abs(np.sum(first['tangent'][ay, ax, None]*feature['tangent'][yy, xx], -1))
        scale_ratio = feature['scale'][yy, xx]/widths[0, :, None]
        width = np.exp(-np.abs(np.log(np.maximum(scale_ratio, 1e-8))))
        continuity = np.minimum(first['stability'][ay, ax, None], feature['stability'][yy, xx])
        position = np.exp(-.5*(distances/radius)**2)
        score = .25*position+.30*orientation+.20*width+.25*continuity
        good = np.isfinite(distances) & (orientation >= .80) & (width >= .4)
        good &= parts[ay, ax, None] == parts[yy, xx]
        if semantics is not None:
            good &= allowed[semantics[ay, ax, None], semantics[yy, xx]]
        # Check the whole short correspondence path, including one-pixel holes.
        for t in np.linspace(0, 1, int(np.ceil(radius*2))+1):
            xy = anchor[:, None]*(1-t)+targets*t
            py, px = np.rint(xy[..., 1]).astype(int), np.rint(xy[..., 0]).astype(int)
            good &= valid[py, px] & (weights[i, py, px] > 0)
        score = np.where(good, score, -1)
        best = np.argmax(score, axis=1)
        rows = np.arange(count)
        strength = score[rows, best]
        take = strength >= .68
        offsets[i, take] = targets[rows, best][take]-anchor[take]
        scores[i, take] = strength[take]
        widths[i, take] = feature['scale'][yy[rows, best][take], xx[rows, best][take]]
    consensus = (scores > 0).sum(0) >= 3
    latent = np.column_stack([weighted_median(offsets[..., j], scores) for j in range(2)])
    latent[~consensus] = 0
    latent_width = weighted_median(widths, scores)
    report = dict(anchor_nodes=count, consensus_nodes=int(consensus.sum()),
        matched_nodes_per_view=[int((a > 0).sum()) for a in scores],
        radius_native_pixels=radius, minimum_views=3,
        components=['position', 'orientation', 'width', 'continuity', 'registration/visibility'],
        component_weights=[.25, .30, .20, .25],
        latent_position='weighted median; primary stays unwarped in the fallback branch',
        latent_width='weighted median of detected native-pixel scales',
        polarity='Not compared across lights; signed photometric refit follows matching',
        nodes=[dict(x=float(p[0]+d[0]), y=float(p[1]+d[1]), width=float(s),
                    views=int((v > 0).sum()))
               for p, d, s, v in zip(anchor[consensus], latent[consensus], latent_width[consensus], scores[:, consensus].T)])
    return dict(anchor=anchor, offsets=offsets, scores=scores, latent=latent,
                consensus=consensus, report=report)


def align_mid_evidence(fields, observation_weights, valid, matches, radius):
    """Compact local warps from matched centers, with full footprint/path checks."""
    m, h, w = fields.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    out = np.zeros_like(fields); weights = np.zeros_like(fields)
    keep = matches['consensus']
    if not keep.any():
        return out, weights
    anchor, latent = matches['anchor'][keep], matches['latent'][keep]
    targets = anchor+latent
    ty = np.clip(np.rint(targets[:, 1]).astype(int), 0, h-1)
    tx = np.clip(np.rint(targets[:, 0]).astype(int), 0, w-1)
    for i in range(m):
        score = matches['scores'][i, keep]
        offset = matches['offsets'][i, keep]-latent
        offset *= np.minimum(1., radius/np.maximum(np.linalg.norm(offset, axis=-1), 1e-8))[:, None]
        seed = np.zeros((h, w), np.float32)
        np.add.at(seed, (ty, tx), score)
        denominator = ndi.gaussian_filter(seed, 3., mode='constant')
        delta = []
        for j in range(2):
            numerator = np.zeros_like(seed)
            np.add.at(numerator, (ty, tx), score*offset[:, j])
            delta.append(ndi.gaussian_filter(numerator, 3., mode='constant')/np.maximum(denominator, 1e-9))
        dx, dy = delta
        support = np.clip(denominator*45., 0, 1)*valid
        path = valid.copy()
        for t in np.linspace(0, 1, int(np.ceil(radius*2))+1):
            path &= ndi.map_coordinates(valid.astype(np.float32), [yy+t*dy, xx+t*dx],
                                        order=1, mode='constant', cval=0) >= 1-1e-6
        out[i] = ndi.map_coordinates(fields[i], [yy+dy, xx+dx], order=1, mode='constant')
        weights[i] = ndi.map_coordinates(observation_weights[i], [yy+dy, xx+dx],
            order=1, mode='constant')*support*path
        out[i, ~path] = 0
    return out, weights


def primary_slope(base, residual, light, direction, maximum_angle):
    """One-light directional prior with bounded smooth saturation, not a 2D fit."""
    direction = unit(direction-base*np.sum(base*direction, -1, keepdims=True))
    perpendicular = unit(np.cross(base, direction))
    a, b = direction@light, perpendicular@light
    prior, directional_prior = .006, .08
    determinant = (a*a+prior)*(b*b+prior+directional_prior)-(a*b)**2
    along = residual*a*(prior+directional_prior)/np.maximum(determinant, 1e-10)
    across = residual*b*prior/np.maximum(determinant, 1e-10)
    slope = direction*along[..., None]+perpendicular*across[..., None]
    cap = np.tan(np.deg2rad(maximum_angle))
    magnitude = np.linalg.norm(slope, axis=-1)
    slope *= (cap*np.tanh(magnitude/cap)/np.maximum(magnitude, 1e-12))[..., None]
    return slope.astype(np.float32)


def blend_weights(photo_confidence, structure_confidence, valid, options=None):
    cfg = fusion_settings(options)
    c = np.clip((photo_confidence-cfg['confidence_low'])/
                (cfg['confidence_high']-cfg['confidence_low']), 0, 1)
    multi = isolated_blur(c*c*(3-2*c), valid, cfg['blend_sigma_pixels'])
    structure = (1-multi)*cfg['structure_fraction']*isolated_blur(
        np.clip(structure_confidence, 0, 1), valid, cfg['blend_sigma_pixels'])
    primary = (1-multi-structure)*valid
    return multi.astype(np.float32), primary.astype(np.float32), structure.astype(np.float32)


def project_candidate_height(slope, observation_weight, prior_slope, valid, metric,
                             geometry_options, reference_height=None):
    """Complete a candidate within the shared domain without zero-height holes.

    The weak prior is a continuity regularizer, not another image observation.
    Fully supported measurements retain their full data term. Where this source
    is missing, use the primary directional prior and connected height edges.
    """
    observed = np.clip(observation_weight, 0, 1)
    prior_weight = .05*(1-observed)
    weight = observed+prior_weight
    field = (slope*observed[..., None]+prior_slope*prior_weight[..., None])/weight[..., None]
    gx, gy = -np.sum(field*metric['jx'], -1), -np.sum(field*metric['jy'], -1)
    height, report = integrate_height(gx, gy, weight*valid, valid, metric,
                                      geometry_options, reference_height=reference_height)
    report['completed_samples'] = int((valid & (observed <= .001)).sum())
    report['completion_prior_weight'] = .05
    return height, report


def fuse_reference_height(base, candidate, extra, arrays, domain, metric, geometry_options,
                          options=None):
    cfg = fusion_settings(options)
    photo = extra['photo_confidence']
    primary_observed = arrays['geometry_confidences'][0]
    primary_valid = domain & metric['valid'] & (primary_observed > .001)
    valid = surface_support(domain & metric['valid'] & (primary_valid | (photo > 0)))
    primary_field = arrays['mid_residuals'][0]+cfg['primary_low_fraction']*arrays['low_residuals'][0]
    p_slope = primary_slope(base, primary_field, arrays['mid_lights'][0],
        arrays['mid_anchor_direction'], cfg['primary_max_angle_degrees'])
    p_slope[~primary_valid] = 0
    def gradients(slope):
        return -np.sum(slope*metric['jx'], -1), -np.sum(slope*metric['jy'], -1)
    def project(slope, confidence, prior=None):
        return project_candidate_height(slope, confidence, p_slope, valid, metric,
                                        geometry_options, reference_height=prior)
    height_primary, p_report = project(p_slope, primary_observed)
    raw_slope = slope_from_base(base, candidate)
    height_multi, m_report = project(raw_slope, photo, height_primary)
    print('Matching nearby mid-band structures before signed refitting...', flush=True)
    matches = match_mid_structures(arrays['mid_residuals'], arrays['geometry_confidences'],
        domain & metric['valid'], cfg, arrays.get('fusion_semantics'))
    warped, warped_weights = align_mid_evidence(arrays['mid_residuals'],
        arrays['geometry_confidences'], valid, matches, cfg['match_radius_pixels'])
    height_structure = height_primary.copy()
    structure_conf = np.zeros(valid.shape, np.float32)
    s_slope = p_slope
    s_report = dict(converged=True, active_samples=0)
    if np.any((warped_weights > .025).sum(0) >= 3):
        lights, ambient = arrays['mid_lights'], arrays['mid_ambients']
        # Warp only band residuals: the base lighting term is evaluated at the
        # destination. Moving raw intensities would manufacture base geometry.
        predicted = np.stack([a+np.maximum(base@l, 0) for a, l in zip(ambient, lights)])
        opts = dict(geometry_options, confidence_as_blend_weight=True,
                    robust_scale=geometry_options['robust_scale']*1.4)
        normal, _, detail = solve_geometry(base, predicted+warped, warped_weights, lights, ambient, opts)
        structure_conf = detail['photo_confidence']
        s_slope = slope_from_base(base, normal)
        height_structure, s_report = project(s_slope, structure_conf, height_primary)
        matches['report']['signed_fit'] = detail['stats']
    wm, wp, ws = blend_weights(photo, structure_conf, valid, cfg)
    # Fade the availability term as well: a hard source switch can turn harmless
    # differences in the height datum into a large artificial boundary slope.
    available = isolated_blur(primary_valid.astype(np.float32), valid, cfg['blend_sigma_pixels'])
    wm = (wm+(1-wm)*(1-available))*valid
    wp *= available
    ws *= available
    height = wm*height_multi+wp*height_primary+ws*height_structure
    height[~valid] = 0
    slope = raw_slope*wm[..., None]+p_slope*wp[..., None]+s_slope*ws[..., None]
    gx, gy = gradients(slope)
    gx[~valid] = 0; gy[~valid] = 0
    # Baking support is explicitly separate from the photo/source-selection
    # confidence. Do not multiply the already blended height by confidence again.
    confidence = np.where(valid, .05+.9*np.maximum(primary_observed, photo), 0).astype(np.float32)
    projection = dict(converged=all(r['converged'] for r in (p_report, m_report, s_report)),
        active_samples=int(valid.sum()), primary=p_report, multi=m_report, structure=s_report,
        curl_before=curl_rms((gx[:, :-1]+gx[:, 1:])*.5, (gy[:-1]+gy[1:])*.5, valid),
        curl_after=curl_rms(*height_edges(height), valid),
        method='Three screened height projections followed by a continuous partition-of-unity height blend')
    diagnostic = dict(height_primary=height_primary, height_multi=height_multi,
        height_structure=height_structure, weight_multi=wm, weight_primary=wp, weight_structure=ws,
        photo_confidence=photo*valid, primary_observation_support=primary_observed*valid,
        structure_confidence=structure_conf*valid)
    report = dict(enabled=True, primary_reference_index=0, settings=cfg,
        primary_dominant_samples=int(((wp > .5) & valid).sum()),
        multilight_dominant_samples=int(((wm > .5) & valid).sum()),
        recovered_from_no_multilight_samples=int((valid & (photo <= .015)).sum()),
        mean_primary_weight=float(wp[valid].mean()) if valid.any() else 0.,
        weight_sum_max_error=float(np.max(np.abs((wm+wp+ws)[valid]-1))) if valid.any() else 0.,
        matching=matches['report'],
        support_note='Export confidence encodes bake support, not measured geometric certainty.',
        continuity_note='Candidate heights use weak connected primary-gradient completion; missing source samples are never isolated zero-height anchors. Source availability also fades continuously.',
        limitations=['Primary shading prior can include albedo/shadow artifacts; it is bounded, not ground truth.',
                    'Latent line centers use local orientation/width matching; no full curve or semantic wrinkle reconstruction.',
                    'Raw multilight rank/occlusion protections remain; low trust selects primary instead of zero.'])
    return dict(height=height, confidence=confidence, valid=valid, gx=gx, gy=gy,
                diagnostic=diagnostic, report=report, projection=projection)
