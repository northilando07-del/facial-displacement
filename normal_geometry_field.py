"""Document 44: joint photometry -> integrable height -> surface-metric bands.

The height is a small base-normal offset approximation, not measured geometry.
One image supplies ONE directional constraint; gradients are solved jointly.
No image-polarity voting or brightness-band line enhancement is used here.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator

from normal_geometry import unit
from normal_multilight import tangent_frame, angular, blur, sample_image
from normal_shading import fit_light


DEFAULT_GEOMETRY = dict(enabled=True, millimeters_per_unit=None,
    assumed_face_width_mm=140., sigmas_mm=[.5, 1.5, 4., 12.],
    maximum_angle_degrees=8., prior=.003, albedo_prior=.002,
    robust_scale=.018, iterations=7, chunk_size=32768,
    min_condition_ratio=.015, registration_tolerance_pixels=8.,
    screening_length_mm=25., cg_rtol=2e-4, cg_maxiter=1800)


def geometry_settings(options=None):
    cfg = dict(DEFAULT_GEOMETRY, **(options or {}))
    if type(cfg['enabled']) is not bool:
        raise ValueError('geometry.enabled must be a boolean.')
    for name in ('assumed_face_width_mm', 'maximum_angle_degrees', 'prior',
                 'albedo_prior', 'robust_scale', 'registration_tolerance_pixels',
                 'screening_length_mm', 'cg_rtol', 'min_condition_ratio'):
        value = cfg[name]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError('geometry.'+name+' must be finite and positive.')
    if cfg['maximum_angle_degrees'] >= 60 or cfg['cg_rtol'] >= .1:
        raise ValueError('Invalid geometry angle limit or solver tolerance.')
    for name in ('iterations', 'chunk_size', 'cg_maxiter'):
        if isinstance(cfg[name], bool) or int(cfg[name]) != cfg[name] or cfg[name] < 1:
            raise ValueError('geometry.'+name+' must be a positive integer.')
        cfg[name] = int(cfg[name])
    scale = cfg['millimeters_per_unit']
    if scale is not None and (isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0):
        raise ValueError('millimeters_per_unit must be positive or null (estimated).')
    sigmas = np.asarray(cfg['sigmas_mm'], float)
    if sigmas.shape != (4,) or not np.isfinite(sigmas).all() or np.any(sigmas <= 0) or np.any(np.diff(sigmas) <= 0):
        raise ValueError('geometry.sigmas_mm needs four increasing positive scales.')
    return cfg


def geometry_observation(base, observed, valid, semantics, registration):
    """Estimate effective light in smooth facial regions; preserve raw intensities."""
    smooth = valid & np.isin(semantics, [1, 11, 12]) & (base[..., 2] > .45)
    curvature = sum(np.sum(a*a, axis=-1) for a in np.gradient(base, axis=(0, 1)))
    if smooth.sum() >= 500:
        smooth &= curvature <= np.quantile(curvature[smooth], .8)
    fallback = smooth.sum() < 300
    fitting = valid if fallback else smooth
    ambient, light, report = fit_light(base, blur(observed, valid, 2.), fitting)
    feather = np.clip(ndi.distance_transform_edt(valid)/5., 0, 1)
    lit = np.clip((base@light-.004)/.045, 0, 1)
    facing = np.clip((base[..., 2]-.25)/.4, 0, 1)
    quality = 1./(1.+(report['linear_rmse']/.12)**2)
    weight = valid*feather*lit*facing*registration*quality
    # Clipped highlights/dark pixels cannot constrain a Lambertian derivative.
    weight *= (observed > .008) & (observed < .95)
    report.update(smooth_region_samples=int(fitting.sum()),
        smooth_region_fallback=bool(fallback), fit_confidence=float(quality),
        regions='landmark forehead/cheeks, lower base-normal variation',
        observation='Unfiltered linear luminance. No brightness band-pass or polarity test.')
    return float(ambient), np.asarray(light), weight.astype(np.float32), report


def solve_geometry(base, observations, weights, lights, ambients, options=None):
    """Bounded robust joint slope/albedo fit, requiring three independent views.

The albedo column includes the ambient term. Conditioning is tested AFTER
eliminating that column, so reflectance cannot masquerade as a second slope.
"""
    cfg = geometry_settings(options)
    base = np.asarray(base, np.float32)
    obs, conf = np.asarray(observations, np.float32), np.asarray(weights, np.float32)
    light, ambient = np.asarray(lights, float), np.asarray(ambients, float)
    if obs.ndim != 3 or obs.shape != conf.shape or obs.shape[1:] != base.shape[:2] or light.shape != (len(obs), 3) or ambient.shape != (len(obs),):
        raise ValueError('Joint geometry observation shapes do not match.')
    if not all(np.isfinite(a).all() for a in (base, obs, conf, light, ambient)):
        raise ValueError('Geometry observations contain nonfinite values.')
    m, h, w = obs.shape
    flat = base.reshape(-1, 3)
    observed, input_weights = obs.reshape(m, -1), np.clip(conf.reshape(m, -1), 0, 1)
    result = flat.copy()
    confidence = np.zeros(h*w, np.float32)
    photo_confidence = np.zeros(h*w, np.float32)
    albedo = np.zeros(h*w, np.float32)
    condition = np.zeros(h*w, np.float32)
    fit_error = np.zeros(h*w, np.float32)
    final_weights = np.zeros_like(input_weights)
    ids_all = np.flatnonzero((input_weights > .025).sum(0) >= 3)
    rejected = 0
    before_sum = after_sum = weight_sum = 0.
    reg = np.diag([cfg['prior'], cfg['prior'], cfg['albedo_prior']])
    for start in range(0, len(ids_all), cfg['chunk_size']):
        ids = ids_all[start:start+cfg['chunk_size']]
        n = flat[ids].astype(float)
        t, b = tangent_frame(n)
        predicted = ambient[None, :]+np.maximum(n@light.T, 0)
        design = np.stack([t@light.T, b@light.T, predicted], -1)
        target = observed[:, ids].T-predicted
        weight = input_weights[:, ids].T.astype(float)

        def fitting(ww):
            lhs = np.einsum('nmi,nm,nmj->nij', design, ww, design)+reg
            rhs = np.einsum('nmi,nm,nm->ni', design, ww, target)
            q = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            q[:, 2] = np.clip(q[:, 2], -.6, .6)
            return q

        # Redundant observations initialize IRLS without letting a single
        # relighting artifact pull every image toward its invented feature.
        q = fitting(weight)
        error = np.einsum('nmi,ni->nm', design, q)-target
        if m >= 4:
            best = np.full(len(ids), np.inf)
            for omit in range(-1, m):
                ww = weight.copy()
                if omit >= 0:
                    ww[:, omit] = 0
                trial = fitting(ww)
                err = np.einsum('nmi,ni->nm', design, trial)-target
                losses = weight*err**2
                penalty = np.einsum('ni,ij,nj->n', trial, reg, trial)
                score = losses.sum(1)-losses.max(1)+penalty
                score[(ww > .025).sum(1) < 3] = np.inf
                take = score < best
                error[take] = err[take]
                best[take] = score[take]
        robust = 1./(1.+(error/cfg['robust_scale'])**2)
        for _ in range(cfg['iterations']):
            q = fitting(weight*robust)
            error = np.einsum('nmi,ni->nm', design, q)-target
            robust = 1./(1.+(error/cfg['robust_scale'])**2)
        ww = weight*robust
        fisher = np.einsum('nmi,nm,nmj->nij', design, ww, design)
        marginalized = fisher[:, :2, :2]-fisher[:, :2, 2:]*fisher[:, 2:, :2]/np.maximum(fisher[:, 2:, 2:], 1e-12)
        eig = np.linalg.eigvalsh(marginalized)
        ratio = np.maximum(eig[:, 0], 0)/np.maximum(eig[:, 1], 1e-12)
        supported = ((ww > .025).sum(1) >= 3) & (ratio >= cfg['min_condition_ratio']) & (eig[:, 0] > 1e-7)
        reliability = np.clip(ww.sum(1)/3., 0, 1)*np.clip(ratio/.12, 0, 1)*supported
        # Source selection must not mistake an accurately flat/albedo-only fit
        # for an unreliable fit merely because its geometric signal is small.
        photo_confidence[ids] = reliability
        delta = (t*q[:, :1]+b*q[:, 1:2])/np.maximum(1.+q[:, 2:], .4)
        response = np.sqrt(np.mean((design[..., :2]*q[:, None, :2]).sum(-1)**2, axis=1))
        snr = response/(response+cfg['robust_scale']*.2)
        reliability *= .15+.85*snr
        cap = np.tan(np.deg2rad(cfg['maximum_angle_degrees']))
        if not cfg.get('confidence_as_blend_weight', False):
            cap = cap*np.sqrt(reliability)
        delta *= np.minimum(1., cap/np.maximum(np.linalg.norm(delta, axis=-1), 1e-12))[:, None]
        candidate = unit(n+delta).astype(np.float32)
        rejected_fit = ~supported if cfg.get('confidence_as_blend_weight', False) else reliability <= .015
        candidate[rejected_fit] = flat[ids][rejected_fit]
        reconstructed = (1.+q[:, 2:])*(ambient[None, :]+np.maximum(candidate@light.T, 0))
        result[ids] = candidate
        confidence[ids] = reliability
        albedo[ids] = q[:, 2]*supported
        condition[ids] = ratio
        fit_error[ids] = np.sqrt((ww*(reconstructed-observed[:, ids].T)**2).sum(1)/np.maximum(ww.sum(1), 1e-12))
        final_weights[:, ids] = ww.T
        rejected += int(((robust < .25) & (weight > .025)).sum())
        before_sum += float((weight*target**2).sum())
        after_sum += float((weight*(reconstructed-observed[:, ids].T)**2).sum())
        weight_sum += float(weight.sum())
    active = confidence > .015
    stats = dict(method='Joint unfiltered photometry; two tangent slopes plus shared relative albedo; robust IRLS',
        minimum_independent_views=3, active_samples=int(active.sum()),
        insufficient_or_degenerate_samples=int(((input_weights > .025).any(0) & ~active).sum()),
        rejected_observations=rejected,
        marginalized_condition_median=float(np.median(condition[active])) if active.any() else 0.,
        raw_photometric_rmse_before=float(np.sqrt(before_sum/max(weight_sum, 1e-12))),
        raw_photometric_rmse_after=float(np.sqrt(after_sum/max(weight_sum, 1e-12))),
        raw_photometric_metric='Same input weights; recovered normal AND relative albedo. Not normal ground-truth accuracy.',
        polarity_voting=False, brightness_bandpass_before_geometry=False,
        confidence_factors='Light fit, registration, visibility, robust geometric-response agreement, marginalized rank and response SNR')
    return result.reshape(h, w, 3), confidence.reshape(h, w), dict(stats=stats,
        relative_albedo=albedo.reshape(h, w), condition=condition.reshape(h, w),
        fit_error=fit_error.reshape(h, w), observation_weights=final_weights.reshape(obs.shape),
        photo_confidence=photo_confidence.reshape(h, w))


def surface_metric(mesh, camera, raster, options=None, landmark_points=None):
    """dP/du,dP/dv -> screen chart Jacobian -> full surface metric in mm.

OBJ is unitless. Without an explicit conversion, the landmark face width is
assigned an explicitly reported nominal length; millimeters are then estimated.
"""
    cfg = geometry_settings(options)
    scale = cfg['millimeters_per_unit']
    if scale is None:
        if landmark_points is None:
            raise ValueError('Unitless OBJ needs landmark points or millimeters_per_unit.')
        width = float(np.linalg.norm(landmark_points[0]-landmark_points[16]))
        if width <= 1e-9:
            raise ValueError('Face-width scale is degenerate.')
        scale = cfg['assumed_face_width_mm']/width
        provenance = 'Estimated from landmark 0-to-16 width assigned '+str(cfg['assumed_face_width_mm'])+' mm; not measured millimeters.'
    else:
        provenance = 'User-configured millimeters per OBJ unit.'
    p = mesh.vertices[mesh.triangles]@camera.rotation.T*scale
    uv = mesh.uv[np.maximum(mesh.triangle_uv, 0)]
    screen = raster['screen_vertices'][mesh.triangles]
    ep = np.stack([p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]], -1)
    euv = np.stack([uv[:, 1]-uv[:, 0], uv[:, 2]-uv[:, 0]], -1)
    es = np.stack([screen[:, 1]-screen[:, 0], screen[:, 2]-screen[:, 0]], -1)
    good = (mesh.triangle_uv >= 0).all(1) & (np.abs(np.linalg.det(euv)) > 1e-14) & (np.abs(np.linalg.det(es)) > 1e-10)
    dpduv = np.zeros_like(ep)
    jac = np.zeros_like(ep)
    dpduv[good] = ep[good]@np.linalg.inv(euv[good])
    jac[good] = dpduv[good]@(euv[good]@np.linalg.inv(es[good]))
    tid = raster['triangle']
    n = raster['normal']
    local = jac[np.maximum(tid, 0)].astype(np.float32)
    # Use the mesh differential projected into its smooth shading tangent plane.
    local -= n[..., :, None]*np.sum(n[..., :, None]*local, axis=-2, keepdims=True)
    jx, jy = local[..., 0], local[..., 1]
    xx, xy, yy = np.sum(jx*jx, -1), np.sum(jx*jy, -1), np.sum(jy*jy, -1)
    determinant = xx*yy-xy*xy
    valid = (tid >= 0) & good[np.maximum(tid, 0)] & (determinant > 1e-8*np.maximum(xx*yy, 1e-15)) & (n[..., 2] > .25)
    metric = dict(jx=jx, jy=jy, xx=xx, xy=xy, yy=yy, valid=valid,
        area=np.sqrt(np.maximum(determinant, 1e-15)).astype(np.float32))
    report = dict(millimeters_per_unit=float(scale), scale_provenance=provenance,
        invalid_uv_or_projected_triangles=int((~good).sum()), valid_metric_pixels=int(valid.sum()),
        method='Full dP/du,dP/dv chained to image chart; smooth-tangent projection; mixed metric term retained.',
        mm_per_pixel_median=float(np.median(np.sqrt(metric['area'][valid]))) if valid.any() else 0.)
    return metric, report


def _cg_checked(matrix, rhs, initial, cfg, label):
    diagonal = np.maximum(matrix.diagonal(), 1e-15)
    preconditioner = LinearOperator(matrix.shape, matvec=lambda a: a/diagonal, dtype=np.float64)
    count = [0]
    def count_step(_): count[0] += 1
    solution, info = cg(matrix, rhs, x0=initial, M=preconditioner,
        rtol=cfg['cg_rtol'], atol=1e-11, maxiter=cfg['cg_maxiter'], callback=count_step)
    residual = float(np.linalg.norm(matrix@solution-rhs)/max(np.linalg.norm(rhs), 1e-15))
    if info != 0 or not np.isfinite(solution).all():
        raise RuntimeError(f'{label} did not converge: info={info}, relative residual={residual:g}')
    return solution, dict(iterations=count[0], relative_residual=residual, converged=True)


def height_edges(height):
    return np.diff(height, axis=1), np.diff(height, axis=0)


def chart_topology(valid):
    """One triangle domain for integration, diffusion and differentiation.

    A valid pixel alone, or a one-pixel bridge without an incident triangle,
    cannot support a two-dimensional surface derivative. In particular, do not
    differentiate across edges omitted by the surface diffusion operator.
    """
    valid = np.asarray(valid, bool)
    first = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]
    second = valid[1:, 1:] & valid[1:, :-1] & valid[:-1, 1:]
    supported = np.zeros_like(valid)
    supported[:-1, :-1] |= first
    supported[:-1, 1:] |= first | second
    supported[1:, :-1] |= first | second
    supported[1:, 1:] |= second
    horizontal = np.zeros((valid.shape[0], max(valid.shape[1]-1, 0)), bool)
    vertical = np.zeros((max(valid.shape[0]-1, 0), valid.shape[1]), bool)
    horizontal[:-1] |= first
    horizontal[1:] |= second
    vertical[:, :-1] |= first
    vertical[:, 1:] |= second
    return supported, horizontal, vertical, first, second


def surface_support(valid):
    return chart_topology(valid)[0]


def resample_height(height, valid, shape):
    """Resample heights without treating missing observations as zero height.

    Require a fully supported interpolation footprint; normalize its weights
    to remove even subpixel contamination at holes. Never fill across a hole.
    """
    valid = surface_support(valid)
    weights = sample_image(valid.astype(np.float32), shape)
    numerator = sample_image(np.where(valid, height, 0).astype(np.float32), shape)
    supported = surface_support(weights >= 1.-1e-6)
    result = np.zeros(shape, np.float32)
    np.divide(numerator, weights, out=result, where=supported)
    return result, supported


def curl_rms(gx_edges, gy_edges, valid):
    quads = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    curl = gx_edges[:-1]+gy_edges[:, 1:]-gx_edges[1:]-gy_edges[:, :-1]
    return float(np.sqrt(np.mean(curl[quads]**2))) if quads.any() else 0.


def integrate_height(gx, gy, confidence, valid, metric, options=None, reference_height=None):
    """Masked confidence-weighted screened Poisson with no edges across holes."""
    cfg = geometry_settings(options)
    input_valid = np.asarray(valid, bool)
    if reference_height is not None:
        reference_height = np.asarray(reference_height, float)
        if reference_height.shape != input_valid.shape or not np.isfinite(reference_height).all():
            raise ValueError('The screening reference must be a finite matching height field.')
    valid, eh, ev, _, _ = chart_topology(input_valid & metric['valid'])
    height = np.zeros(valid.shape, np.float64)
    if not valid.any():
        return height, dict(converged=True, active_samples=0, curl_before=0., curl_after=0.)
    ids = np.full(valid.shape, -1, np.int32)
    ids[valid] = np.arange(valid.sum(), dtype=np.int32)
    a = np.concatenate([ids[:, :-1][eh], ids[:-1][ev]])
    b = np.concatenate([ids[:, 1:][eh], ids[1:][ev]])
    area = metric['area']
    wh = np.sqrt(confidence[:, :-1]*confidence[:, 1:])*(area[:, :-1]+area[:, 1:])/np.maximum(metric['xx'][:, :-1]+metric['xx'][:, 1:], 1e-15)
    wv = np.sqrt(confidence[:-1]*confidence[1:])*(area[:-1]+area[1:])/np.maximum(metric['yy'][:-1]+metric['yy'][1:], 1e-15)
    weight = np.concatenate([wh[eh], wv[ev]]).astype(float)
    ex, ey = (gx[:, :-1]+gx[:, 1:])*.5, (gy[:-1]+gy[1:])*.5
    target = np.concatenate([ex[eh], ey[ev]])
    count = int(valid.sum())
    anchor = area[valid].astype(float)/cfg['screening_length_mm']**2
    diagonal = np.bincount(a, weight, minlength=count)+np.bincount(b, weight, minlength=count)+anchor
    matrix = sparse.coo_matrix((np.concatenate([-weight, -weight, diagonal]),
        (np.concatenate([a, b, np.arange(count)]), np.concatenate([b, a, np.arange(count)]))), shape=(count, count)).tocsr()
    rhs = np.bincount(b, weight*target, minlength=count)-np.bincount(a, weight*target, minlength=count)
    if reference_height is not None:
        rhs += anchor*reference_height[valid]
    solved, stats = _cg_checked(matrix, rhs, None, cfg, 'Geometry Poisson projection')
    height[valid] = solved
    hx, hy = height_edges(height)
    actual = np.concatenate([hx[eh], hy[ev]])
    stats.update(active_samples=count, curl_before=curl_rms(ex, ey, valid),
        unsupported_surface_samples=int((input_valid & ~valid).sum()),
        curl_after=curl_rms(hx, hy, valid),
        weighted_gradient_correction_rms=float(np.sqrt(np.sum(weight*(actual-target)**2)/max(weight.sum(), 1e-15))),
        screening_length_mm=cfg['screening_length_mm'],
        method='Weighted screened Poisson on the shared triangle chart; free hole boundaries; no unsupported bridge edges',
        curl_units='mm per oriented pixel-cell circulation; computed on identical valid quads')
    return height, stats


class SurfaceDiffusion:
    """Linear triangle FEM retaining the mixed surface-metric coefficient.

The screen grid is just a chart. M and K are assembled from physical surface
area and inverse metric. Each scale solves (M + sigma_mm^2/2 K) h = M h0.
This is implicit diffusion, not a fixed pixel Gaussian or an exact heat kernel.
"""
    def __init__(self, valid, metric):
        self.valid = valid = surface_support(np.asarray(valid, bool) & metric['valid'])
        ids = np.full(valid.shape, -1, np.int32)
        ids[valid] = np.arange(valid.sum(), dtype=np.int32)
        self.count = int(valid.sum())
        triangles = np.concatenate([
            np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1]], -1).reshape(-1, 3),
            np.stack([ids[1:, 1:], ids[1:, :-1], ids[:-1, 1:]], -1).reshape(-1, 3)])
        triangles = triangles[(triangles >= 0).all(1)]
        xx, xy, yy = [np.asarray(metric[k][valid], float) for k in ('xx', 'xy', 'yy')]
        tx, tc, ty = [a[triangles].mean(1) for a in (xx, xy, yy)]
        area2 = np.sqrt(np.maximum(tx*ty-tc*tc, 1e-20))
        edge_a = np.concatenate([triangles[:, 0], triangles[:, 0], triangles[:, 1]])
        edge_b = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 2]])
        weights = np.concatenate([.5*(ty-tc)/area2, .5*(tx-tc)/area2, .5*tc/area2])
        diagonal = np.bincount(edge_a, weights, minlength=self.count)+np.bincount(edge_b, weights, minlength=self.count)
        self.stiffness = sparse.coo_matrix((np.concatenate([-weights, -weights, diagonal]),
            (np.concatenate([edge_a, edge_b, np.arange(self.count)]),
             np.concatenate([edge_b, edge_a, np.arange(self.count)]))), shape=(self.count, self.count)).tocsr()
        self.mass = np.zeros(self.count, float)
        for corner in range(3):
            self.mass += np.bincount(triangles[:, corner], area2/6., minlength=self.count)
        isolated = self.mass <= 1e-15
        self.mass[isolated] = np.maximum(metric['area'][valid][isolated], 1e-12)

    def smooth(self, height, sigma_mm, cfg):
        output = np.zeros(self.valid.shape, np.float64)
        if not self.count:
            return output, dict(converged=True, iterations=0, relative_residual=0.)
        source = np.asarray(height[self.valid], float)
        matrix = sparse.diags(self.mass, format='csr')+.5*sigma_mm**2*self.stiffness
        result, stats = _cg_checked(matrix, self.mass*source, source, cfg, 'Metric diffusion')
        output[self.valid] = result
        return output, stats


def split_height(height, valid, metric, options=None):
    cfg = geometry_settings(options)
    operator = SurfaceDiffusion(valid, metric)
    smooth, reports = [], []
    for sigma in cfg['sigmas_mm']:
        field, report = operator.smooth(height, sigma, cfg)
        smooth.append(field)
        reports.append(dict(sigma_mm=sigma, **report))
    valid = operator.valid
    height = np.where(valid, height, 0)
    bands = dict(mid_fine=smooth[0]-smooth[1], mid_large=smooth[1]-smooth[2],
        low=smooth[2]-smooth[3], illumination_remainder=smooth[3],
        fine_remainder=np.where(valid, height-smooth[0], 0))
    reconstructed = sum(bands.values())
    error = float(np.max(np.abs(reconstructed-height)))
    bands['mid'] = bands['mid_fine']+bands['mid_large']
    return bands, dict(method='Full-metric surface FEM implicit diffusion; physical scale before frequency differences',
        scales=reports, reconstruction_max_error_mm=error,
        mid_definition='mid_fine + mid_large; both saved independently as height diagnostics')


def limit_height_edges(height, valid, metric, maximum_angle_degrees,
                       maxiter=32, tolerance=.001):
    """Local convex height constraints, preserving a scalar integrable field.

    Bound both triangle derivatives using the full metric. For correlation r,
    |hx| <= cap*sqrt(gxx)*sqrt((1-|r|)/2), and analogously for hy, imply the
    physical gradient norm is at most cap. Each incident triangle uses the
    minimum bound of its three vertices, so area averaging preserves the bound.

    A short cyclic projection removes isolated spikes. If it has not converged,
    shortest-path lower/upper Lipschitz envelopes finish the constraints exactly.
    Their midpoint is feasible on every edge, including disconnected islands.
    No whole-layer gain is applied and already-feasible fields stay unchanged.
    """
    if not 0 < maximum_angle_degrees < 85 or maxiter < 1 or tolerance <= 0:
        raise ValueError('Invalid height constraint angle, iteration count or tolerance.')
    valid, eh, ev, first, second = chart_topology(np.asarray(valid, bool) & metric['valid'])
    original = np.where(valid, np.asarray(height, float), 0)
    if not np.isfinite(original).all():
        raise ValueError('Height constraints require finite supported heights.')
    result = original.copy()
    if not valid.any():
        return result, dict(converged=True, iterations=0, changed_samples=0)
    xx, xy, yy = metric['xx'], metric['xy'], metric['yy']
    rho = np.abs(xy)/np.maximum(np.sqrt(xx*yy), 1e-15)
    factor = np.sqrt(np.maximum((1-rho)*.5, 0))
    cap = .99*np.tan(np.deg2rad(maximum_angle_degrees))
    bx, by = cap*np.sqrt(xx)*factor, cap*np.sqrt(yy)*factor
    lh = np.full(eh.shape, np.inf); lv = np.full(ev.shape, np.inf)
    a, b, c, d = np.s_[:-1, :-1], np.s_[:-1, 1:], np.s_[1:, :-1], np.s_[1:, 1:]
    for mask, corners, hs, vs in (
            (first, (a, b, c), np.s_[:-1, :], np.s_[:, :-1]),
            (second, (d, c, b), np.s_[1:, :], np.s_[:, 1:])):
        lx = np.minimum.reduce([bx[s] for s in corners])
        ly = np.minimum.reduce([by[s] for s in corners])
        lh[hs] = np.minimum(lh[hs], np.where(mask, lx, np.inf))
        lv[vs] = np.minimum(lv[vs], np.where(mask, ly, np.inf))
    ids = np.arange(valid.size, dtype=np.int32).reshape(valid.shape)
    groups = []
    for parity in range(2):
        horizontal = eh & (np.arange(eh.shape[1])[None, :] % 2 == parity)
        vertical = ev & (np.arange(ev.shape[0])[:, None] % 2 == parity)
        groups.append((ids[:, :-1][horizontal], ids[:, 1:][horizontal], lh[horizontal]))
        groups.append((ids[:-1][vertical], ids[1:][vertical], lv[vertical]))
    flat = result.ravel()
    def violation():
        return max((float(np.max(np.maximum(np.abs(flat[right]-flat[left])-bound, 0)
                     /np.maximum(bound, 1e-12))) for left, right, bound in groups if len(left)), default=0.)
    worst = violation()
    envelope_used = False
    for iteration in range(maxiter):
        if worst <= tolerance:
            break
        for left, right, bound in groups:
            if not len(left):
                continue
            difference = flat[right]-flat[left]
            excess = np.maximum(np.abs(difference)-bound, 0)
            correction = .5*np.sign(difference)*excess
            flat[left] += correction
            flat[right] -= correction
        worst = violation()
    warmstart_violation = worst
    if worst > tolerance:
        from scipy.sparse.csgraph import dijkstra
        active = np.flatnonzero(valid.ravel())
        count = len(active)
        dense = np.full(valid.size, -1, np.int32)
        dense[active] = np.arange(count, dtype=np.int32)
        left = np.concatenate([dense[a] for a, _, _ in groups])
        right = np.concatenate([dense[b] for _, b, _ in groups])
        bound = np.concatenate([b for _, _, b in groups])
        # A super-source with initial heights computes min_j(h_j + d(i,j)).
        # All graph lengths and source edges are nonnegative for Dijkstra.
        rows = np.concatenate([left, right, np.full(count, count, np.int32)])
        cols = np.concatenate([right, left, np.arange(count, dtype=np.int32)])
        source = flat[active].copy()
        envelopes = []
        for sign in (1., -1.):
            initial = sign*source
            origin = float(initial.min())
            lengths = np.concatenate([bound, bound, initial-origin+1e-12])
            graph = sparse.coo_matrix((lengths, (rows, cols)), shape=(count+1, count+1)).tocsr()
            distance = dijkstra(graph, directed=True, indices=count, return_predecessors=False)
            envelopes.append(sign*(distance[:count]+origin-1e-12))
        flat[active] = .5*(envelopes[0]+envelopes[1])
        envelope_used = True
        worst = violation()
    if not np.isfinite(worst) or worst > tolerance:
        raise RuntimeError(f'Local height constraints did not converge: {worst:g}')
    changed = np.abs(result-original)
    return result, dict(converged=bool(worst <= tolerance), iterations=iteration+1,
        maximum_relative_edge_violation=worst, tolerance=tolerance,
        warmstart_relative_edge_violation=warmstart_violation,
        exact_envelope_completion=envelope_used,
        changed_samples=int((changed > 1e-8).sum()), maximum_height_change_mm=float(changed.max()),
        method='Full-metric local height bounds; cyclic warm start and feasible shortest-path Lipschitz envelopes; no global attenuation')


def height_to_normal(base, height, valid, metric, maximum_angle_degrees):
    """Differentiate a scalar height, then limit by ONE scalar to preserve curl.

The global layer attenuation avoids pointwise normal clipping, which would
generally destroy integrability. Mask crossings never enter a derivative.
"""
    input_valid = np.asarray(valid, bool) & metric['valid']
    valid, _, _, first, second = chart_topology(input_valid)
    height = np.where(valid, np.asarray(height, float), 0)
    # Differentiate the same linear triangles used by SurfaceDiffusion.
    # Each boundary vertex receives BOTH derivative components from its
    # incident triangles, including the diagonal edge of a masked cell.
    hx = np.zeros(valid.shape, float); hy = np.zeros_like(hx)
    mass = np.zeros_like(hx)
    area = metric['area']
    a = np.s_[:-1, :-1]; b = np.s_[:-1, 1:]
    c = np.s_[1:, :-1]; d = np.s_[1:, 1:]
    for mask, corners, dx, dy in (
            (first, (a, b, c), height[b]-height[a], height[c]-height[a]),
            (second, (d, c, b), height[d]-height[c], height[d]-height[b])):
        weight = sum(area[s] for s in corners)/3.*mask
        for s in corners:
            hx[s] += dx*weight
            hy[s] += dy*weight
            mass[s] += weight
    hx /= np.maximum(mass, 1e-20); hy /= np.maximum(mass, 1e-20)
    determinant = np.maximum(metric['xx']*metric['yy']-metric['xy']**2, 1e-20)
    a = (metric['yy']*hx-metric['xy']*hy)/determinant
    b = (metric['xx']*hy-metric['xy']*hx)/determinant
    gradient = metric['jx']*a[..., None]+metric['jy']*b[..., None]
    gradient[~valid] = 0
    peak = float(np.max(np.linalg.norm(gradient, axis=-1)))
    factor = min(1., np.tan(np.deg2rad(maximum_angle_degrees))/max(peak, 1e-15))
    refined = unit(base-gradient*factor).astype(np.float32)
    unchanged = (~valid) | (np.linalg.norm(gradient, axis=-1) < 1e-12)
    refined[unchanged] = base[unchanged]
    return refined, np.asarray(height)*factor, dict(height_cap_scale=factor,
        unsupported_surface_samples=int((input_valid & ~valid).sum()),
        derivative_method='Area-weighted derivatives of the shared surface triangles',
        maximum_angle_degrees=float(angular(base, refined).max()),
        cap_method='Uniform height/slope scaling per layer, preserving its integrability')
