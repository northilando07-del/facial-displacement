"""33.doc: independent lip observations and bounded multiscale normal residuals.

The frequency names describe image-space bands, not a semantic recognition of
pores or nasolabial folds. Every output is a base-relative normal perturbation.
No height field, displacement or new geometry is inferred here.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from normal_geometry import unit
from normal_multilight import (MultiSettings, angular, constrain_normals,
                               fit_metrics, solve_multilight, tangent_frame)
from normal_shading import Settings, solve, srgb_to_linear, tensor_direction
from normal_structure import (connect_structures, direction_weights,
                              line_features, smooth_along)


LAYER_NAMES = ('low', 'mid', 'high', 'lips')
DEFAULT_GAINS = dict(low=.20, mid=1.50, high=.60, lips=1.0)
DEFAULT_CAPS = dict(low=1.5, mid=8., high=4., lips=6.)


def isolated_blur(field, valid, sigma, components=None):
    """Normalized convolution separately in connected regions; never mix lips.

    Invalid pixel values cannot contribute. In particular, upper and lower lip
    components on opposite sides of a protected mouth opening stay independent.
    """
    a = np.asarray(field, np.float32)
    valid = np.asarray(valid, bool)
    if a.shape != valid.shape or a.ndim != 2 or sigma <= 0:
        raise ValueError('Expected a scalar field, matching mask and positive sigma.')
    if not np.isfinite(a[valid]).all():
        raise ValueError('Valid residual values must be finite.')
    components = ndi.label(valid)[0] if components is None else components
    out = np.zeros_like(a)
    for label, sl in enumerate(ndi.find_objects(components), 1):
        if sl is None:
            continue
        mask = (components[sl] == label) & valid[sl]
        weight = mask.astype(np.float32)
        # Explicit zero padding avoids extending a region's boundary values.
        numerator = ndi.gaussian_filter(np.where(mask, a[sl], 0), sigma,
                                        mode='constant', cval=0)
        denominator = ndi.gaussian_filter(weight, sigma, mode='constant', cval=0)
        values = numerator / np.maximum(denominator, 1e-12)
        out[sl][mask] = values[mask]
    return out


def split_residual(raw, valid, sigmas=(.8, 3., 18., 64.)):
    """A telescoping Gaussian pyramid including explicit discarded components.

    raw = noise_remainder + high + mid + low + illumination_remainder.
    This identity is tested before any denoising, confidence gate or enhancement.
    """
    sigmas = np.asarray(sigmas, float)
    if sigmas.shape != (4,) or not np.isfinite(sigmas).all() or np.any(sigmas <= 0) or np.any(np.diff(sigmas) <= 0):
        raise ValueError('Provide four finite, positive, strictly increasing scales.')
    raw = np.asarray(raw, np.float32)
    components = ndi.label(valid)[0]
    smooth = [isolated_blur(raw, valid, float(s), components) for s in sigmas]
    result = dict(high=smooth[0]-smooth[1], mid=smooth[1]-smooth[2],
                  low=smooth[2]-smooth[3],
                  noise_remainder=np.where(valid, raw-smooth[0], 0),
                  illumination_remainder=smooth[3])
    reconstruction = sum(result.values())
    error = float(np.max(np.abs(reconstruction[valid]-raw[valid]))) if np.any(valid) else 0.
    result['reconstruction_max_error'] = error
    result['sigmas_pixels'] = sigmas.tolist()
    return result


def lip_masks(shape, landmarks, geom, normals, slit_radius=None):
    lm = np.asarray(landmarks, float)
    if lm.shape != (68, 2) or not np.isfinite(lm).all():
        raise ValueError('Lip extraction needs 68 finite face landmarks.')
    h, w = shape
    if geom.shape != shape or normals.shape != (*shape, 3):
        raise ValueError('Lip mask geometry shape mismatch.')
    outer = Image.new('L', (w, h), 0)
    ImageDraw.Draw(outer).polygon([tuple(p) for p in lm[48:60]], fill=255)
    inner = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(inner)
    d.polygon([tuple(p) for p in lm[60:68]], fill=255)
    # A closed mouth may have an inner polygon of zero area. Keep its slit out.
    d.line([tuple(p) for p in lm[60:65]], fill=255, width=1)
    d.line([tuple(lm[i]) for i in (64, 65, 66, 67, 60)], fill=255, width=1)
    radius = max(1, round(h/1374*2)) if slit_radius is None else int(slit_radius)
    if radius < 1:
        raise ValueError('The protected mouth slit needs at least one pixel.')
    opening = ndi.binary_dilation(np.asarray(inner) > 0, iterations=radius)
    outside = np.asarray(outer) > 0
    domain = outside & ~opening & geom & (normals[..., 2] > .25)
    return dict(domain=domain, outer=outside, mouth_opening=opening,
                feather=np.clip(ndi.distance_transform_edt(domain)/max(3., h/343.5), 0, 1).astype(np.float32))


def lip_observation(rgb, landmarks, geom, normals, anchor_domain):
    masks = lip_masks(rgb.shape[:2], landmarks, geom, normals)
    domain = masks['domain'] & anchor_domain
    lum = (srgb_to_linear(rgb) @ np.array([.2126, .7152, .0722], np.float32)).astype(np.float32)
    valid = domain & (lum > .012) & (lum < .85) & (np.max(rgb, axis=-1) < .985)
    if int(valid.sum()) < 32:
        return valid & False, dict(masks, appearance_samples=0)
    chroma = rgb/np.maximum(rgb.sum(-1, keepdims=True), .05)
    median = np.median(chroma[valid], axis=0)
    distance = np.linalg.norm(chroma-median, axis=-1)
    med = float(np.median(distance[valid]))
    mad = float(np.median(np.abs(distance[valid]-med)))*1.4826
    cutoff = max(.065, med+4*mad)
    valid &= distance < cutoff
    edge = np.zeros_like(lum)
    for c in range(3):
        gy, gx = np.gradient(ndi.gaussian_filter(chroma[..., c], .8))
        edge += gx*gx+gy*gy
    valid &= np.sqrt(edge) < .045
    local = lum-isolated_blur(lum, valid, 2.0)
    # Bright narrow spots are more likely specular than stable diffuse evidence.
    spread = max(.006, float(np.median(np.abs(local[valid])))*1.4826) if valid.any() else .006
    valid &= local < max(.055, 5*spread)
    valid &= ndi.binary_erosion(valid, iterations=1)
    return valid, dict(masks, appearance_samples=int(valid.sum()),
                       lip_chromaticity=median.tolist(), chromaticity_threshold=cutoff)


def fit_lip_appearance(normals, observed, valid, skin_light):
    """Keep the shared light direction, fit a separate lip reflectance scale.

    Lip colors never enter the skin appearance statistics. The two-parameter fit
    absorbs only a constant appearance difference, not fine lip texture.
    """
    if valid.sum() < 32:
        return 0., np.asarray(skin_light), dict(samples=0, skipped=True)
    x = np.maximum(normals[valid] @ skin_light, 0).astype(float)
    y = observed[valid].astype(float)
    a = np.column_stack([np.ones_like(x), x])
    initial = np.linalg.lstsq(a, y, rcond=None)[0]
    initial = np.clip(initial, [0., .15], [.7, 2.5])
    fit = least_squares(lambda c: a@c-y, initial, bounds=([0., .15], [.7, 2.5]),
                        loss='soft_l1', f_scale=.025, max_nfev=80)
    ambient, scale = map(float, fit.x)
    return ambient, np.asarray(skin_light)*scale, dict(
        ambient=ambient, relative_lip_scale=scale, samples=int(valid.sum()),
        rmse=float(np.sqrt(np.mean((a@fit.x-y)**2))),
        note='Skin light direction shared; lip reflectance/ambient fitted independently.')


def analyze_bands(normals, observed, valid, labels, semantics, ambient, light,
                  sigmas=(.8, 3., 18., 64.), lips=False, continuity=True,
                  noise_floor=.0012, selected_names=None, brightness_bins=0):
    raw = observed-(ambient+np.maximum(normals@light, 0))
    pyramid = split_residual(raw, valid, sigmas)
    noise = max(noise_floor, float(np.median(np.abs(pyramid['noise_remainder'][valid])))/.67449) if valid.any() else noise_floor
    feather = np.clip(ndi.distance_transform_edt(valid)/(3. if lips else 5.), 0, 1).astype(np.float32)
    facing = np.clip((normals[..., 2]-.20)/.40, 0, 1)
    lit = np.clip((normals@light-.003)/.040, 0, 1)
    local_outlier = np.exp(-np.maximum(np.abs(raw-pyramid['illumination_remainder'])-.22, 0)**2/.09**2)
    names = ('lips',) if lips else ('high', 'mid', 'low')
    if selected_names is not None:
        if not set(selected_names).issubset(names):
            raise ValueError('Invalid selected intensity bands.')
        names = tuple(selected_names)
    result = {}
    for name in names:
        field = (.55*pyramid['high']+pyramid['mid']) if lips else pyramid[name].copy()
        if name == 'high':
            field = np.sign(field)*np.maximum(np.abs(field)-.45*noise, 0)
            field = isolated_blur(field, valid, .65)
        tensor_sigma = dict(high=1.4, mid=3.2, low=8., lips=1.5)[name]
        direction, coherence, _ = tensor_direction(field, normals, tensor_sigma)
        band_noise = noise*dict(high=.7, mid=.32, low=.13, lips=.4)[name]
        signal = isolated_blur(np.abs(field), valid, tensor_sigma)
        snr = np.clip((signal-.4*band_noise)/max(2.3*band_noise, 1e-6), 0, 1)
        structure = np.clip((coherence-.08)/.72, 0, 1)
        structure = (.1+.9*structure) if name == 'high' else (.25+.75*structure)
        confidence = (valid*feather*facing*lit*structure*snr*local_outlier).astype(np.float32)
        support = np.zeros_like(field)
        feature = line_features(field, scales=(.8, 1.5, 2.8) if name in ('high', 'lips') else (2., 4., 8.))
        tangent = feature['tangent']
        graph = dict(nodes=[], edges=[], chains=[], stats=dict(nodes=0, edges=0, chains=0, cross_region_edges=0))
        if continuity and name in ('mid', 'lips'):
            link_labels = labels
            if name == 'mid' and brightness_bins > 1 and valid.any():
                illumination = isolated_blur(observed, valid, 8.)
                cuts = np.unique(np.quantile(illumination[valid],
                    np.linspace(0, 1, int(brightness_bins)+1)[1:-1]))
                brightness = np.searchsorted(cuts, illumination).astype(np.int32)
                link_labels = labels.astype(np.int32)*int(brightness_bins)+brightness
            linked = connect_structures(field, confidence, valid, link_labels, semantics,
                band_noise, normals=normals, step=5 if lips else 12,
                enhancement=.10 if lips else .18,
                line_scales=(.8, 1.5, 2.8) if lips else (2., 4., 8.),
                support_radius=2. if lips else 5.)
            field, confidence = linked['residual'], linked['confidence']
            support, tangent, graph = linked['support'], linked['tangent'], linked['graph']
            if name == 'mid' and brightness_bins > 1:
                graph['stats']['cross_brightness_edges'] = sum(
                    graph['nodes'][e['a']]['region_id'] % brightness_bins !=
                    graph['nodes'][e['b']]['region_id'] % brightness_bins
                    for e in graph['edges'])
                graph['brightness_bins'] = int(brightness_bins)
        elif name == 'high':
            support = structure*valid
            wh, wv = direction_weights(tangent, support, valid, normals)
            field = smooth_along(field, wh, wv, iterations=6, amount=1.1)
            confidence *= .75
        field = np.clip(field, -.18, .18).astype(np.float32)
        field[~valid] = 0
        confidence[~valid] = 0
        result[name] = dict(residual=field, confidence=confidence, direction=direction.astype(np.float32),
                            tangent=tangent, support=support, graph=graph, noise=band_noise)
    return result, pyramid


def slope_from_base(base, normal):
    cosine = np.sum(base*normal, -1, keepdims=True)
    slope = normal/np.maximum(cosine, .1)-base
    slope -= base*np.sum(base*slope, -1, keepdims=True)
    # An unchanged float32 base is only approximately unit length. Explicitly
    # preserve its identity instead of creating a sub-ULP tangent perturbation.
    slope[np.all(normal == base, axis=-1)] = 0
    return slope.astype(np.float32)


def compose_normals(base, layers, gains=None, maximum_angle_degrees=12.):
    """Combine common-base tangent slopes, not RGB averages or normal sums."""
    gains = DEFAULT_GAINS if gains is None else gains
    if not 0 < maximum_angle_degrees < 85:
        raise ValueError('Normal angle limit must be between zero and 85 degrees.')
    delta = np.zeros_like(base, dtype=np.float32)
    for name, normal in layers.items():
        gain = float(gains.get(name, 0.))
        if not np.isfinite(gain) or gain < 0:
            raise ValueError('Layer gains must be finite and nonnegative.')
        if normal.shape != base.shape or not np.isfinite(normal).all():
            raise ValueError('Layer normal shape or finite-value check failed.')
        if gain:
            delta += gain*slope_from_base(base, normal)
    return normal_from_slope(base, delta, maximum_angle_degrees)


def normal_from_slope(base, delta, maximum_angle_degrees=12.):
    delta = np.asarray(delta, np.float32).copy()
    delta -= base*np.sum(base*delta, -1, keepdims=True)
    length = np.linalg.norm(delta, axis=-1)
    cap = np.tan(np.deg2rad(maximum_angle_degrees))
    delta *= np.minimum(1, cap/np.maximum(length, 1e-12))[..., None]
    result = unit(base+delta).astype(np.float32)
    result[length < 1e-9] = base[length < 1e-9]
    return result


def consensus_confidence(base, residuals, confidences, lights, settings):
    """Use redundant views to initialize robust weights before the local solve.

    Four observations can contain one damaged image. Compare every leave-one-out
    fit on the same trimmed, prior-regularized objective instead of initializing
    IRLS from a fit already pulled toward that damaged image. Pixels with fewer
    than four observations retain their original confidence and rank fallback.
    """
    r = np.asarray(residuals, np.float32)
    conf = np.asarray(confidences, np.float32)
    if r.shape != conf.shape or r.ndim != 3 or r.shape[1:] != base.shape[:2]:
        raise ValueError('Consensus residual and confidence shapes must match the base.')
    if not np.isfinite(r).all() or not np.isfinite(conf).all():
        raise ValueError('Consensus observations must be finite.')
    m = len(r)
    result = conf.copy()
    if m < 4:
        return result, dict(eligible_pixels=0, strongly_downweighted_observations=0)
    rf, cf, adjusted = r.reshape(m, -1), conf.reshape(m, -1), result.reshape(m, -1)
    active = np.flatnonzero((cf > .025).sum(0) >= 4)
    flat = base.reshape(-1, 3)
    regularizer = np.diag([settings.prior, settings.prior, settings.albedo_prior])
    rejected = 0
    for start in range(0, len(active), settings.chunk_size):
        ids = active[start:start+settings.chunk_size]
        n = flat[ids].astype(np.float64)
        t, b = tangent_frame(n)
        light = np.asarray(lights, np.float64)
        design = np.stack([t@light.T, b@light.T, np.maximum(n@light.T, 0)], -1)
        target = rf[:, ids].T.astype(np.float64)
        weights = cf[:, ids].T.astype(np.float64)
        best_score = np.full(len(ids), np.inf)
        best_error = np.zeros_like(target)
        count = np.maximum((weights > .025).sum(1)-1, 3)
        for omit in range(-1, m):
            fitting = weights.copy()
            if omit >= 0:
                fitting[:, omit] = 0
            lhs = np.einsum('nmi,nm,nmj->nij', design, fitting, design)+regularizer
            rhs = np.einsum('nmi,nm,nm->ni', design, fitting, target)
            solution = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            solution[:, 2] = np.clip(solution[:, 2], -.18, .18)
            error = np.einsum('nmi,ni->nm', design, solution)-target
            losses = weights*error**2
            trimmed = losses.sum(1)-losses.max(1)
            penalty = settings.prior*np.sum(solution[:, :2]**2, axis=1)+settings.albedo_prior*solution[:, 2]**2
            score = (trimmed+penalty)/count
            take = score < best_score
            best_score[take] = score[take]
            best_error[take] = error[take]
        # A consensus disagreement is an additional independent reliability gate,
        # before the solver's own robust iteration. Keep it conservative: a
        # heavily disputed reference must not regain influence through a band
        # edge created by Gaussian decomposition of its original bright patch.
        factors = 1/(1+(best_error/settings.robust_scale)**2)**2
        adjusted[:, ids] *= factors.T.astype(np.float32)
        rejected += int(((factors < .25) & (weights > .025)).sum())
    return result, dict(eligible_pixels=int(len(active)),
                       strongly_downweighted_observations=rejected,
                       method='Leave-one-out initialization, trimmed weighted error and base/albedo priors.')


def solve_band(base, residuals, confidences, lights, direction, settings=None,
               name='mid', allow_anchor_fallback=False):
    settings = settings or MultiSettings(maximum_angle_degrees=DEFAULT_CAPS[name])
    consensus, consensus_stats = consensus_confidence(base, residuals, confidences, lights, settings)
    refined, confidence, extra = solve_multilight(base, residuals, consensus, lights, settings)
    fallback = np.zeros(confidence.shape, bool)
    if allow_anchor_fallback or len(lights) == 1:
        # Only missing observations can fall back. Rank-deficient MULTI-view
        # data still returns the base exactly, retaining the existing safeguard.
        counts = (confidences > .015).sum(0)
        fallback = (counts < 2) & (confidences[0] > .12)
        if fallback.any():
            y, x = np.nonzero(fallback)
            sl = np.s_[max(0, y.min()-2):min(base.shape[0], y.max()+3),
                       max(0, x.min()-2):min(base.shape[1], x.max()+3)]
            fallback_conf = (confidences[0]*.35*fallback).astype(np.float32)
            single_settings = Settings(maximum_angle_degrees=min(3., settings.maximum_angle_degrees),
                prior=max(.01, settings.prior), iterations=45,
                smoothness=.08, direction_prior=.06)
            single, _ = solve(base[sl], residuals[0][sl], fallback_conf[sl], direction[sl],
                               lights[0], single_settings)
            take = fallback[sl]
            refined[sl][take] = single[take]
            confidence[sl][take] = fallback_conf[sl][take]
    refined = constrain_normals(base, refined, confidence, settings.maximum_angle_degrees)
    active = confidence > .015
    if not np.array_equal(refined[~active], base[~active]):
        raise RuntimeError('A protected pixel was changed.')
    angles = angular(base, refined)
    stats = dict(extra['stats'])
    stats.update(fit_metrics(base, refined, residuals, confidences, lights))
    stats.update(anchor_only_prior_samples=int(fallback.sum()),
                 active_samples=int(active.sum()), changed_samples=int((angles > .001).sum()),
                 maximum_angle_degrees=float(angles.max()),
                 mean_active_angle_degrees=float(angles[active].mean()) if active.any() else 0.,
                 p95_active_angle_degrees=float(np.quantile(angles[active], .95)) if active.any() else 0.)
    stats['consensus_initialization'] = consensus_stats
    return refined, confidence, dict(extra, stats=stats)
