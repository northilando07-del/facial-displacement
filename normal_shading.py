"""Local, prior-constrained shading refinement. NumPy/SciPy only; no ML."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage as ndi
from normal_geometry import unit


@dataclass
class Settings:
    detail_sigma: float = 1.1
    low_frequency_sigma: float = 13.
    tensor_sigma: float = 2.5
    maximum_angle_degrees: float = 8.
    iterations: int = 140
    prior: float = .008
    smoothness: float = .08
    direction_prior: float = .04
    min_coherence: float = .22
    noise_floor: float = .0012


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055)


def masked_blur(a: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    weight = ndi.gaussian_filter(mask.astype(np.float64), sigma, mode='nearest')
    if a.ndim == 3:
        return ndi.gaussian_filter(a * mask[..., None], (sigma, sigma, 0), mode='nearest') / np.maximum(weight[..., None], 1e-9)
    return ndi.gaussian_filter(a * mask, sigma, mode='nearest') / np.maximum(weight, 1e-9)


def fit_light(normals: np.ndarray, observed: np.ndarray, valid: np.ndarray) -> tuple[float, np.ndarray, dict]:
    """Robustly fit ambient plus a directional diffuse term in linear light.

    Albedo and source power remain coupled: the fitted vector absorbs both.
    This is an effective lighting approximation, not intrinsic decomposition.
    """
    n = normals[valid].astype(float); y = observed[valid].astype(float)
    if len(y) < 100:
        raise ValueError('Too few valid skin samples to estimate lighting.')
    stride = max(1, len(y) // 60000); n = n[::stride]; y = y[::stride]
    design = np.column_stack([np.ones(len(y)), n])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    weights = np.ones(len(y))
    for _ in range(12):
        err = y - design @ coef
        sigma = max(float(np.median(np.abs(err - np.median(err))) * 1.4826), .005)
        weights = np.minimum(1, 1.5 * sigma / np.maximum(np.abs(err), 1e-8))
        a = design.T @ (design * weights[:, None]) + np.diag([.0001, .01, .01, .01])
        coef = np.linalg.solve(a, design.T @ (weights * y))
    # An unconstrained ambient can become unphysical on small patches. Refit
    # with nonnegative ambient and camera-facing light bounds, transparently.
    from scipy.optimize import least_squares
    initial = coef.copy(); initial[0] = max(0, initial[0]); initial[3] = max(.005, initial[3])
    def residual(c):
        return (c[0] + np.maximum(n @ c[1:], 0) - y) * np.sqrt(weights)
    fit = least_squares(residual, initial, bounds=([0, -2, -2, .005], [1, 2, 2, 2]),
                        loss='soft_l1', f_scale=.025, max_nfev=100)
    c = fit.x
    pred = c[0] + np.maximum(n @ c[1:], 0)
    return float(c[0]), c[1:], dict(ambient=float(c[0]), effective_light_vector=c[1:].tolist(),
        direction=unit(c[1:]).tolist(), strength=float(np.linalg.norm(c[1:])),
        linear_rmse=float(np.sqrt(np.mean((pred - y) ** 2))),
        design_condition=float(np.linalg.cond(design)), samples=int(len(y)),
        note='Effective diffuse light; unknown albedo and light power are not separated.')


def tensor_direction(field: np.ndarray, normals: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy, gx = np.gradient(field)
    a = ndi.gaussian_filter(gx * gx, sigma)
    b = ndi.gaussian_filter(gx * gy, sigma)
    c = ndi.gaussian_filter(gy * gy, sigma)
    disc = np.sqrt(np.maximum((a - c) ** 2 + 4 * b * b, 0))
    coherence = disc / (a + c + 1e-12)
    phi = .5 * np.arctan2(2 * b, a - c)
    # Inverse orthographic projection differential into the surface tangent.
    # Image y is down; camera y is up. This is a tangent-direction PRIOR.
    dx, dy = np.cos(phi), -np.sin(phi)
    nz = np.maximum(normals[..., 2], .15)
    dz = -(normals[..., 0] * dx + normals[..., 1] * dy) / nz
    direction = unit(np.stack([dx, dy, dz], -1))
    return direction, coherence, np.sqrt(a + c)


def prepare_detail(normals: np.ndarray, observed: np.ndarray, valid: np.ndarray,
                   labels: np.ndarray, ambient: float, light: np.ndarray,
                   settings: Settings) -> dict:
    predicted = ambient + np.maximum(normals @ light, 0)
    raw = observed - predicted
    blurred = masked_blur(raw, valid, settings.detail_sigma)
    high = observed - masked_blur(observed, valid, .65)
    noise = max(settings.noise_floor, float(np.median(np.abs(high[valid])) / .67449))
    modal = np.zeros_like(raw)
    rows = []
    for label in np.unique(labels[valid]):
        if label < 0:
            continue
        inside = (labels == label) & valid
        values = raw[inside]
        if not len(values):
            continue
        lo, hi = np.quantile(values, [.05, .95])
        if hi - lo < 1e-5:
            mode = float(np.median(values))
        else:
            hist, edges = np.histogram(values, bins=max(8, min(40, int(np.sqrt(len(values))))), range=(lo, hi))
            peak = int(np.argmax(hist))
            members = values[(values >= edges[peak]) & (values <= edges[peak + 1])]
            # A tiny projected triangle can have just two pixels, both outside
            # the trimmed histogram range. Its modal fallback must stay finite.
            mode = float(np.median(members if len(members) else values))
        modal[inside] = mode
        rows.append(dict(patch=int(label), pixels=int(len(values)), modal_residual=mode,
                         residual_mad=float(np.median(np.abs(values - np.median(values))) * 1.4826)))
    modal = masked_blur(modal, valid, settings.low_frequency_sigma)
    centered = blurred - modal
    residual = centered - masked_blur(centered, valid, settings.low_frequency_sigma)
    residual[~valid] = 0
    direction, coherence, energy = tensor_direction(residual, normals, settings.tensor_sigma)
    regular = np.zeros_like(valid)
    for row in rows:
        inside = (labels == row['patch']) & valid
        signal = float(np.quantile(np.abs(residual[inside]), .8))
        coherent_fraction = float(np.mean(coherence[inside] > settings.min_coherence))
        enabled = row['pixels'] >= 12 and signal > noise * 1.3 and coherent_fraction > .12
        row.update(signal_p80=signal, coherent_fraction=coherent_fraction, refined=enabled)
        if enabled:
            regular[inside] = True
    border = np.clip(ndi.distance_transform_edt(valid) / 5., 0, 1)
    orientation = np.clip((normals[..., 2] - .2) / .4, 0, 1)
    structure = np.clip((coherence - settings.min_coherence) / .55, 0, 1)
    signal = ndi.gaussian_filter(np.abs(residual), settings.tensor_sigma)
    snr = np.clip((signal - .6 * noise) / max(3 * noise, 1e-5), 0, 1)
    confidence = valid * regular * border * orientation * structure * snr
    confidence = masked_blur(confidence, valid, .8) * valid * border
    # Large discrepancies may be cast shadows, reflectance changes, or misalignment.
    confidence *= np.exp(-np.maximum(np.abs(raw - modal) - .12, 0) ** 2 / .035 ** 2)
    residual = np.clip(residual, -.1, .1)
    return dict(predicted=predicted, raw=raw, modal=modal, residual=residual,
                direction=direction, coherence=coherence, confidence=confidence,
                noise=noise, patches=rows, energy=energy)


def solve(normals: np.ndarray, residual: np.ndarray, confidence: np.ndarray,
          direction: np.ndarray, light: np.ndarray, settings: Settings,
          depth: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Constrained nonlinear projected-gradient solve on normal residuals.

    Light sensitivity is light - (light.n0)n0, NOT its normalized version.
    The image-gradient direction is an optional regularizer, not a unique
    geometrical solution. Neighbor regularization acts on vector residuals.
    """
    if not all(np.isfinite(np.asarray(a)).all() for a in (normals, residual, confidence, direction, light)):
        raise ValueError('Normal solver inputs contain NaN or infinity.')
    n0 = normals.astype(np.float64)
    direction = unit(direction - n0 * np.sum(n0 * direction, -1, keepdims=True))
    active = confidence > .015
    weight = np.clip(confidence, 0, 1) * active
    light = np.asarray(light, dtype=float)
    baseline = np.maximum(n0 @ light, 0)
    target = baseline + residual
    tangent_light = light - n0 * (n0 @ light)[..., None]
    directional = np.sum(tangent_light * direction, -1)
    alpha = weight * residual * directional / (weight * directional ** 2 + settings.prior + .01)
    delta = alpha[..., None] * direction
    cap = np.tan(np.deg2rad(settings.maximum_angle_degrees)) * np.sqrt(weight)
    def constrain(d):
        d -= n0 * np.sum(d * n0, -1, keepdims=True)
        length = np.linalg.norm(d, axis=-1)
        d *= np.minimum(1, cap / np.maximum(length, 1e-12))[..., None]
        d[~active] = 0
        return d
    delta = constrain(delta)
    # Neighbor terms stop at invalid pixels, large normal changes, or depth breaks.
    wh = active[:, 1:] & active[:, :-1]
    wv = active[1:] & active[:-1]
    wh &= np.sum(n0[:, 1:] * n0[:, :-1], -1) > .9
    wv &= np.sum(n0[1:] * n0[:-1], -1) > .9
    if depth is not None:
        finite = np.where(np.isfinite(depth), depth, 0)
        dh = np.abs(finite[:, 1:] - finite[:, :-1]); dv = np.abs(finite[1:] - finite[:-1])
        typical = np.median(np.concatenate([dh[wh], dv[wv]])) if wh.any() and wv.any() else 0.
        wh &= dh < max(.05, 8 * typical); wv &= dv < max(.05, 8 * typical)
    perpendicular = unit(np.cross(n0, direction))
    step = .7 / (np.dot(light, light) * 1.5 + settings.prior + 8 * settings.smoothness + settings.direction_prior)
    def energy(d):
        n = unit(n0 + d); err = np.maximum(n @ light, 0) - target
        value = np.sum(weight * err ** 2) + settings.prior * np.sum(d * d)
        value += settings.smoothness * (np.sum((d[:, 1:] - d[:, :-1]) ** 2 * wh[..., None]) + np.sum((d[1:] - d[:-1]) ** 2 * wv[..., None]))
        value += settings.direction_prior * np.sum(np.sum(d * perpendicular, -1) ** 2 * weight)
        return float(value)
    initial_energy = energy(np.zeros_like(delta)); best_energy = energy(delta)
    if best_energy > initial_energy:
        delta[:] = 0; best_energy = initial_energy
    accepted = 0
    for iteration in range(settings.iterations):
        u = n0 + delta; length = np.linalg.norm(u, axis=-1, keepdims=True); n = u / length
        dot = n @ light
        err = np.maximum(dot, 0) - target
        jac = (light - n * dot[..., None]) / length
        grad = (weight * err * (dot > 0))[..., None] * jac + settings.prior * delta
        diffh = (delta[:, 1:] - delta[:, :-1]) * wh[..., None]
        diffv = (delta[1:] - delta[:-1]) * wv[..., None]
        grad[:, 1:] += settings.smoothness * diffh; grad[:, :-1] -= settings.smoothness * diffh
        grad[1:] += settings.smoothness * diffv; grad[:-1] -= settings.smoothness * diffv
        grad += settings.direction_prior * (weight * np.sum(delta * perpendicular, -1))[..., None] * perpendicular
        candidate = constrain(delta - step * grad)
        value = energy(candidate)
        if value <= best_energy + 1e-10:
            delta = candidate; best_energy = value; accepted += 1
        else:
            step *= .5
        if step < 1e-8:
            break
    result = unit(n0 + delta)
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(n0, result), axis=-1), np.sum(n0 * result, axis=-1)))
    wsum = max(float(weight.sum()), 1e-12)
    before = np.sqrt(np.sum(weight * residual ** 2) / wsum)
    after = np.sqrt(np.sum(weight * (np.maximum(result @ light, 0) - target) ** 2) / wsum)
    stats = dict(weighted_detail_rmse_before=float(before), weighted_detail_rmse_after=float(after),
                 detail_rmse_reduction_fraction=float(1 - after / before) if before > 1e-12 else 0.,
                 active_virtual_cells=int(active.sum()), normal_angle_max_degrees=float(angles.max()),
                 normal_angle_mean_active_degrees=float(angles[active].mean()) if active.any() else 0.,
                 normal_angle_p95_active_degrees=float(np.quantile(angles[active], .95)) if active.any() else 0.,
                 normal_unit_length_max_error=float(np.abs(np.linalg.norm(result, axis=-1) - 1).max()),
                 objective_before=initial_energy, objective_after=best_energy, accepted_iterations=accepted,
                 settings=asdict(settings),
                 metric_note='Fitting error against filtered target on a FIXED confidence mask; not ground-truth shape accuracy.')
    return result.astype(np.float32), stats
