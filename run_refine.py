"""Run the document's local shading-to-normal prototype on a registered OBJ."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
import struct
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from normal_geometry import (load_obj, calibrate_camera, rasterize, group_patches,
                             corner_tangents, triangle_pixels, adjacency, unit)
from normal_shading import (Settings, srgb_to_linear, linear_to_srgb, masked_blur,
                            fit_light, prepare_detail, solve)


def save_json(path: Path, data: dict | list):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def png16(path: Path, values: np.ndarray):
    """Write real 16-bit RGB PNG, without an sRGB/gamma tag (normal data)."""
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError('16-bit normal PNG must be H x W x 3.')
    arr = np.rint(np.clip(values, 0, 1) * 65535).astype('>u2')
    h, w = arr.shape[:2]
    compressor = zlib.compressobj(6)
    encoded = []
    for row in arr:
        encoded.append(compressor.compress(b'\x00' + row.tobytes()))
    encoded.append(compressor.flush())
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    with path.open('wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 16, 2, 0, 0, 0)))
        f.write(chunk(b'IDAT', b''.join(encoded)))
        f.write(chunk(b'IEND', b''))


def image8(a: np.ndarray) -> Image.Image:
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    return Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8))


def montage(path: Path, panels: list[tuple[str, Image.Image]], columns: int = 3):
    width = 360
    height = max(round(image.height * width / image.width) for _, image in panels)
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * width, rows * (height + 40)), (245, 245, 245))
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for i, (title, image) in enumerate(panels):
        x = (i % columns) * width; y = (i // columns) * (height + 40)
        draw.text((x + 10, y + 10), title, fill=(20, 20, 20), font=font)
        thumb = ImageOps.contain(image, (width, height))
        canvas.paste(thumb, (x + (width - thumb.width) // 2, y + 40))
    canvas.save(path)


def observation_points(cfg: dict, mesh, reference: Image.Image, out: Path):
    """Reuse explicit source-surface bindings and cached image detections.

    The supplied SOAP test assets have a provenance check against their saved
    vertex array. No new learned model is run in this prototype.
    """
    if cfg.get('calibration_observations'):
        data = json.loads(Path(cfg['calibration_observations']).read_text(encoding='utf-8'))
        if data['mesh_sha256'] != mesh.sha256:
            raise ValueError('Calibration is bound to a different mesh.')
        if data['reference_sha256'] != hashlib.sha256(Path(cfg['reference']).read_bytes()).hexdigest():
            raise ValueError('Calibration is bound to a different reference image.')
        ids = np.asarray(data['source_indices']); bary = np.asarray(data['source_barycentric'])
        target = np.asarray(data['target_original_pixels'])
        if ids.shape!=(68,3) or bary.shape!=(68,3) or target.shape!=(68,2) or ids.min()<0 or ids.max()>=len(mesh.vertices):
            raise ValueError('Invalid automatically generated surface bindings.')
        if not np.isfinite(bary).all() or not np.isfinite(target).all() or np.any(bary<0) or not np.allclose(bary.sum(1),1):
            raise ValueError('Invalid barycentric landmark weights.')
        save_json(out/'calibration_observations.json',data)
        return np.sum(mesh.vertices[ids] * bary[..., None], axis=1), target, data
    job = Path(cfg['soap_job'])
    source = np.load(job / 'source_landmarks.npz', allow_pickle=False)
    transfer = np.load(job / 'transfer.npz', allow_pickle=False)
    if transfer['original'].shape != mesh.vertices.shape:
        raise ValueError('SOAP vertex count differs from this OBJ.')
    error = float(np.max(np.abs(transfer['original'] + transfer['delta'] - mesh.vertices)))
    if error > 1e-5:
        raise ValueError(f'Stale SOAP binding: mesh differs by {error:g}.')
    ids = source['source_indices']; bary = source['source_barycentric'].astype(float)
    bary /= bary.sum(1, keepdims=True)
    if ids.min() < 0 or ids.max() >= len(mesh.vertices):
        raise ValueError('Source landmark vertex index outside OBJ.')
    records = json.loads(Path(cfg['reference_detections']).read_text(encoding='utf-8-sig'))
    suffix = Path(cfg['reference']).name.rsplit(' ', 1)[-1]
    candidates = [row for row in records if row['path'].endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError('Cannot uniquely match cached reference landmarks. Supply explicit calibration_observations.')
    record = candidates[0]
    recorded_size = np.asarray(record['size'])
    ratio = reference.width / reference.height
    if abs(recorded_size[0] / recorded_size[1] - ratio) > .005:
        raise ValueError('Cached reference detection aspect ratio differs from the input.')
    target = (np.asarray(record['landmarks']) + .5) * (np.array(reference.size) / recorded_size) - .5
    data = dict(mesh_sha256=mesh.sha256,
                reference_sha256=hashlib.sha256(Path(cfg['reference']).read_bytes()).hexdigest(),
                source_indices=ids.tolist(), source_barycentric=bary.tolist(),
                target_original_pixels=target.tolist(), original_image_size=list(reference.size),
                cached_detection_size=record['size'], cache_match='unique filename time suffix + aspect ratio',
                cache_note='Existing FAN detections are approximate observations, not measured ground truth; the old cache has no image checksum.',
                mesh_binding_max_error=error)
    save_json(out / 'calibration_observations.json', data)
    return np.sum(mesh.vertices[ids] * bary[..., None], axis=1), target, data


def semantic_mask(rgb: np.ndarray, landmarks: np.ndarray, geom: np.ndarray, normals: np.ndarray,
                  brow_width_scale: float = 1.0):
    """Conservative landmark exclusions and chromaticity gates, with diagnostics."""
    h, w = rgb.shape[:2]
    mask = Image.new('L', (w, h), 0); draw = ImageDraw.Draw(mask)
    face_height = float(landmarks[8, 1] - landmarks[27, 1])
    forehead = landmarks[17:27].copy(); forehead[:, 1] -= .72 * face_height
    boundary = np.concatenate([landmarks[:17], forehead])
    hull = boundary[ConvexHull(boundary).vertices]
    draw.polygon([tuple(p) for p in hull], fill=255)
    excluded = Image.new('L', (w, h), 0); ex = ImageDraw.Draw(excluded)
    for ids in (range(36, 42), range(42, 48), range(48, 60)):
        points = landmarks[list(ids)]
        ex.polygon([tuple(p) for p in points], fill=255)
    brows = Image.new('L', (w, h), 0); brow_draw = ImageDraw.Draw(brows)
    if not 0.1 <= brow_width_scale <= 2:
        raise ValueError('Brow width scale must be 0.1..2.')
    line_width = max(2, round(h / 80 * brow_width_scale))
    for ids in (range(17, 22), range(22, 27)):
        brow_draw.line([tuple(p) for p in landmarks[list(ids)]], fill=255, width=line_width)
    # Nose openings are unreliable AO/cast-shadow observations.
    for index in (31, 32, 34, 35):
        x, y = landmarks[index]; radius = max(3, h / 140)
        ex.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    exclusions = ndi.binary_dilation(np.asarray(excluded) > 0, iterations=max(3, round(h / 110)))
    brow_exclusions = ndi.binary_dilation(np.asarray(brows) > 0,
        iterations=max(1, round(max(3, h / 110)*brow_width_scale)))
    anatomical_without_brows = (np.asarray(mask) > 0) & ~exclusions & geom & (normals[..., 2] > .25)
    exclusions |= brow_exclusions
    anatomical = (np.asarray(mask) > 0) & ~exclusions & geom & (normals[..., 2] > .25)
    lum = srgb_to_linear(rgb) @ np.array([.2126, .7152, .0722])
    base = anatomical & (lum > .03) & (lum < .85)
    if base.sum() < 200:
        raise ValueError('Insufficient anatomical skin pixels after projection.')
    chroma = rgb / np.maximum(rgb.sum(-1, keepdims=True), .05)
    median = np.median(chroma[base], axis=0)
    distance = np.linalg.norm(chroma - median, axis=-1)
    med_d = float(np.median(distance[base]))
    mad = float(np.median(np.abs(distance[base] - med_d))) * 1.4826
    cutoff = max(.055, med_d + 3 * mad)
    color_valid = (distance < cutoff) & (lum > max(.025, np.quantile(lum[base], .025))) & (lum < .92)
    # Suppress strong color edges before filtering so they do not bleed inward.
    edge = np.zeros_like(lum)
    for c in range(3):
        gy, gx = np.gradient(ndi.gaussian_filter(chroma[..., c], .8))
        edge += gx * gx + gy * gy
    color_valid &= np.sqrt(edge) < .025
    valid = anatomical & ndi.binary_erosion(color_valid, iterations=2)
    return valid, dict(anatomical=anatomical, exclusions=exclusions, color_valid=color_valid,
                       brow_exclusions=brow_exclusions, anatomical_without_brows=anatomical_without_brows,
                       skin_chromaticity=median.tolist(), chromaticity_threshold=cutoff)


def bake_uv(mesh, camera, raster, refined, confidence, material_names, size, out):
    """Inverse UV rasterization, camera-depth visibility and surface-neighbor checks."""
    ts, bs = corner_tangents(mesh)
    graph = adjacency(mesh)
    n0screen = raster['normal']
    cos = np.sum(n0screen * refined, axis=-1, keepdims=True)
    delta = refined / np.maximum(cos, .2) - n0screen
    depth = np.where(np.isfinite(raster['depth']), raster['depth'], -1e8)
    h, w = confidence.shape
    summaries = []
    for name in material_names:
        material = mesh.materials.index(name)
        face_ids = np.flatnonzero(mesh.material_ids == material)
        if (mesh.triangle_uv[face_ids] < 0).any():
            raise ValueError(f'{name} contains faces with missing UV coordinates.')
        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
        directory = out / 'maps' / safe_name; directory.mkdir(parents=True, exist_ok=True)
        tangent = np.zeros((size, size, 3), np.float32); tangent[..., 2] = 1
        world = np.zeros_like(tangent)
        mask = np.zeros((size, size), bool); conf_map = np.zeros((size, size), np.float32)
        overlap = 0; degenerate = 0
        for face_id in face_ids:
            uv = mesh.uv[mesh.triangle_uv[face_id]]
            if uv.min() < -1e-5 or uv.max() > 1 + 1e-5:
                raise ValueError('Prototype expects a 0-1 UV tile per material. UDIM requires an explicit per-tile bake.')
            hit = triangle_pixels(uv * [size, -size] + [0, size], size, size)
            if hit is None:
                degenerate += 1; continue
            yy, xx, bary = hit
            overlap += int(mask[yy, xx].sum()); mask[yy, xx] = True
            n0 = unit(bary @ mesh.corner_normals[face_id])
            world[yy, xx] = n0
            face = mesh.triangles[face_id]
            screen = bary @ raster['screen_vertices'][face]
            zz = bary @ raster['vertex_depth'][face]
            coords = np.stack([screen[:, 1] - .5, screen[:, 0] - .5])
            inside = (screen[:, 0] >= .5) & (screen[:, 0] < w - .5) & (screen[:, 1] >= .5) & (screen[:, 1] < h - .5)
            sx = np.clip(np.floor(screen[:, 0]).astype(int), 0, w - 1)
            sy = np.clip(np.floor(screen[:, 1]).astype(int), 0, h - 1)
            nearest_id = raster['triangle'][sy, sx]
            surface_ok = np.isin(nearest_id, [face_id] + graph[face_id])
            seen_z = ndi.map_coordinates(depth, coords, order=1, mode='constant', cval=-1e8)
            sample_conf = ndi.map_coordinates(confidence, coords, order=1, mode='constant', cval=0)
            visible = inside & surface_ok & (np.abs(zz - seen_z) < 2.5 / camera.scale) & (sample_conf > .015)
            if not visible.any():
                continue
            yy, xx, bary, coords, n0 = yy[visible], xx[visible], bary[visible], coords[:, visible], n0[visible]
            d = np.column_stack([ndi.map_coordinates(delta[..., c], coords, order=1, mode='constant', cval=0) for c in range(3)])
            d = d @ camera.rotation
            d -= n0 * np.sum(d * n0, axis=-1, keepdims=True)
            n1 = unit(n0 + d)
            t = bary @ ts[face_id]; t = unit(t - n0 * np.sum(n0 * t, axis=-1, keepdims=True))
            b_raw = bary @ bs[face_id]
            b = unit(np.cross(n0, t)) * np.where(np.sum(np.cross(n0, t) * b_raw, axis=-1, keepdims=True) < 0, -1, 1)
            nt = unit(np.column_stack([np.sum(n1 * t, -1), np.sum(n1 * b, -1), np.sum(n1 * n0, -1)]))
            tangent[yy, xx] = nt; world[yy, xx] = n1; conf_map[yy, xx] = sample_conf[visible]
        distance, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
        pad = (~mask) & (distance <= 8)
        tangent[pad] = tangent[nearest[0][pad], nearest[1][pad]]
        gl = tangent * .5 + .5
        png16(directory / 'normal_opengl16.png', gl)
        dx = gl.copy(); dx[..., 1] = 1 - dx[..., 1]
        png16(directory / 'normal_directx16.png', dx)
        image8(gl).save(directory / 'normal_preview8.png')
        image8(conf_map).save(directory / 'confidence.png')
        image8(mask.astype(float)).save(directory / 'uv_coverage.png')
        image8(world * .5 + .5).save(directory / 'object_normal_diagnostic.png')
        np.savez_compressed(directory / 'normal_float32.npz', tangent_normal=tangent,
                            confidence=conf_map, uv_occupied=mask)
        quantized = unit(np.rint(gl * 65535) / 65535 * 2 - 1)
        quant_error = np.degrees(np.arctan2(np.linalg.norm(np.cross(tangent, quantized), axis=-1), np.sum(tangent * quantized, -1)))
        summaries.append(dict(material=name, directory=str(directory), resolution=size,
            uv_occupied_texels=int(mask.sum()), corrected_texels=int((conf_map > .015).sum()),
            overlapping_raster_samples=overlap, triangles_without_texel_centers=degenerate,
            png_bit_depth=16, max_quantization_error_degrees=float(quant_error.max()),
            tangent_basis='Area-weighted per-UV-corner tangent, Gram-Schmidt, mirrored-UV handedness; not certified MikkTSpace.',
            unseen_surface='neutral tangent normal; no inferred detail on invisible surfaces'))
    return summaries


def make_previews(out, image, raster, refined, detail, valid, labels, camera_report):
    rgb = np.asarray(image) / 255.
    mask = raster['triangle'] >= 0
    def background(a):
        x = a.copy(); x[~mask] = .93; return image8(x)
    n0 = raster['normal']; light = np.asarray(detail['light'])
    ambient = detail['ambient']
    before = ambient + np.maximum(n0 @ light, 0)
    after = ambient + np.maximum(refined @ light, 0)
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(n0, refined), axis=-1), np.sum(n0 * refined, -1)))
    strength = angle / max(detail['max_angle'], 1e-6)
    heat = np.stack([strength, .12 * strength, 1 - strength], -1)
    heat[~mask] = .93
    montage(out / 'comparison.png', [('Reference', image),
        ('Base normals / fitted diffuse light', background(linear_to_srgb(before))),
        ('Refined normals / same light', background(linear_to_srgb(after))),
        ('Base camera-space normals', background(n0 * .5 + .5)),
        ('Refined camera-space normals', background(refined * .5 + .5)),
        (f'Normal change: blue 0, red {detail["max_angle"]:g} deg', image8(heat))])
    relight = []
    for name, vector in [('Left', [-.65, .4, .64]), ('Right', [.65, .4, .64])]:
        l = unit(np.asarray(vector))
        for title, n in [('base', n0), ('refined', refined)]:
            shading = linear_to_srgb(.10 + .65 * np.maximum(n @ l, 0))
            relight.append((f'{name} light / {title}', background(shading)))
    montage(out / 'relighting.png', relight, columns=4)
    rng = np.random.default_rng(11)
    colors = rng.uniform(.15, .95, (max(int(labels.max()) + 1, 1), 3))
    patches = colors[np.maximum(labels, 0)]; patches[labels < 0] = .94
    signed = np.clip(detail['residual'] / .045, -1, 1)
    residual_image = np.stack([.5 + signed * .5, .5 - np.abs(signed) * .5, .5 - signed * .5], -1)
    montage(out / 'diagnostics.png', [('Eligible skin (white)', image8(valid.astype(float))),
        ('Connected normal-similar patches', image8(patches)),
        ('Confidence (white = high)', image8(detail['confidence'])),
        ('Band-pass shading residual', image8(residual_image)),
        ('Gradient coherence', image8(detail['coherence'])),
        ('Excluded areas dimmed', image8(rgb * (.15 + .85 * valid[..., None])))])
    overlay = image.copy(); draw = ImageDraw.Draw(overlay)
    projected = np.asarray(camera_report['projected']); target = np.asarray(camera_report['target'])
    for p, q in zip(projected, target):
        draw.line([tuple(p), tuple(q)], fill=(255, 230, 0), width=2)
        draw.ellipse((p[0]-2, p[1]-2, p[0]+2, p[1]+2), fill=(30, 220, 255))
        draw.ellipse((q[0]-2, q[1]-2, q[0]+2, q[1]+2), outline=(255, 50, 30))
    overlay.save(out / 'alignment_landmarks.png')
    projected_mesh = image.copy(); draw = ImageDraw.Draw(projected_mesh)
    # The actual mesh overlay is saved separately by main, using input faces.
    image8(detail['confidence']).save(out / 'confidence.png')
    image8(refined * .5 + .5).save(out / 'normal_camera_refined.png')


def run(cfg: dict) -> dict:
    start = time.perf_counter()
    out = Path(cfg['output']).resolve(); out.mkdir(parents=True, exist_ok=True)
    source_path = Path(cfg['mesh']); reference_path = Path(cfg['reference'])
    for path in (source_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    print('Loading mesh and validating source-surface landmark binding...', flush=True)
    mesh = load_obj(source_path)
    reference = ImageOps.exif_transpose(Image.open(reference_path)).convert('RGB')
    points, target_full, provenance = observation_points(cfg, mesh, reference, out)
    image = reference.copy(); image.thumbnail((cfg.get('screen_resolution', 768),) * 2, Image.Resampling.LANCZOS)
    target = (target_full + .5) * (np.asarray(image.size) / np.asarray(reference.size)) - .5
    camera, camera_report = calibrate_camera(points, target)
    save_json(out / 'camera.json', camera_report)
    print(f'Calibration mean {camera_report["mean_pixels"]:.3f}px, key mean {camera_report["key_mean_pixels"]:.3f}px at {image.size}. Rasterizing...', flush=True)
    skipped = [name for name in mesh.materials if any(k in name.lower() for k in ('eyebrow', 'eyelash', 'tear'))]
    raster = rasterize(mesh, camera, (image.height, image.width), skipped)
    material_names = cfg.get('skin_materials') or [name for name in mesh.materials if re.fullmatch(r'Genesis9SG\d+', name)]
    if not material_names or any(name not in mesh.materials for name in material_names):
        raise ValueError('Specify actual skin_materials from the OBJ in the config.')
    mids = [mesh.materials.index(name) for name in material_names]
    tid = raster['triangle']; geom = (tid >= 0) & np.isin(mesh.material_ids[np.maximum(tid, 0)], mids)
    rgb = np.asarray(image).astype(np.float64) / 255.
    valid, masks = semantic_mask(rgb, target, geom, raster['normal'])
    if valid.sum() < 1000:
        raise ValueError(f'Only {int(valid.sum())} valid skin pixels: inspect camera.json and inputs.')
    labels, regions = group_patches(mesh, raster, valid, radius_pixels=image.height * .0586)
    linear = srgb_to_linear(rgb)
    observed = linear @ np.array([.2126, .7152, .0722])
    ambient, light, light_report = fit_light(raster['normal'], masked_blur(observed, valid, 2), valid)
    allowed = {f.name for f in fields(Settings)}
    if set(cfg.get('solver', {})) - allowed:
        raise ValueError('Unknown solver settings: ' + str(set(cfg['solver']) - allowed))
    settings = Settings(**cfg.get('solver', {}))
    detail = prepare_detail(raster['normal'], observed, valid, labels, ambient, light, settings)
    # Solve only the valid bounding box; image-space metadata remains full size.
    yy, xx = np.nonzero(valid); y0,y1 = max(0, yy.min()-2), min(image.height, yy.max()+3); x0,x1 = max(0, xx.min()-2), min(image.width, xx.max()+3)
    sl = np.s_[y0:y1, x0:x1]
    print(f'{int(valid.sum())} skin pixels, {len(regions)} mesh patches, {sum(p["refined"] for p in detail["patches"])} selected. Solving normal residuals...', flush=True)
    local, stats = solve(raster['normal'][sl], detail['residual'][sl], detail['confidence'][sl],
                         detail['direction'][sl], light, settings, raster['depth'][sl])
    refined = raster['normal'].copy(); refined[sl] = local
    detail.update(light=light.tolist(), ambient=ambient, max_angle=settings.maximum_angle_degrees)
    print(f'Detail fitting RMSE: {stats["weighted_detail_rmse_before"]:.6f} -> {stats["weighted_detail_rmse_after"]:.6f}. Baking UV maps...', flush=True)
    maps = bake_uv(mesh, camera, raster, refined, detail['confidence'], material_names, cfg.get('texture_resolution', 2048), out)
    active = detail['confidence'] > .015
    ys,xs = np.nonzero(active); face_ids = tid[active]; bary = raster['barycentric'][active]
    uv_ids = mesh.triangle_uv[face_ids]
    uv = np.sum(mesh.uv[np.maximum(uv_ids, 0)] * bary[..., None], axis=1)
    np.savez_compressed(out / 'virtual_cells.npz', pixel_xy=np.column_stack([xs,ys]),
                        triangle_id=face_ids, polygon_id=mesh.polygon_ids[face_ids],
                        barycentric=bary, uv=uv, normal_base_camera=raster['normal'][active],
                        normal_refined_camera=refined[active], confidence=detail['confidence'][active],
                        patch_id=labels[active])
    np.savez_compressed(out / 'screen_fields.npz', normal_base=raster['normal'], normal_refined=refined,
                        residual=detail['residual'].astype(np.float32), confidence=detail['confidence'].astype(np.float32),
                        triangle=tid, labels=labels, valid=valid)
    save_json(out / 'patches.json', detail['patches'])
    make_previews(out, image, raster, refined, detail, valid, labels, camera_report)
    overlay = image.copy(); draw = ImageDraw.Draw(overlay)
    visible_ids = np.unique(tid[valid])
    for i in visible_ids:
        projected = raster['screen_vertices'][mesh.triangles[i]]
        draw.line([tuple(p) for p in np.vstack([projected, projected[0]])], fill=(50, 180, 240), width=1)
    overlay.save(out / 'mesh_projection.png')
    pixel_reprojection = np.sum(raster['screen_vertices'][mesh.triangles[face_ids]] * bary[...,None], 1)
    pixel_error = np.linalg.norm(pixel_reprojection - np.column_stack([xs+.5,ys+.5]), axis=1)
    original_unchanged = hashlib.sha256(source_path.read_bytes()).hexdigest() == mesh.sha256
    warnings = ['Single-view shading has no real normal ground truth; lower fitting error does not prove recovered geometry.',
                'Skin color, specular reflection, AO, and cast shadows cannot be completely separated by these heuristic masks.',
                'Existing approximate FAN landmarks calibrate the camera; the refinement itself uses no learned model.',
                'Directional gradient alignment is a regularization prior, not a unique consequence of image brightness.',
                'Only observed, trusted skin receives virtual normal details. No displacement, geometry subdivision, or backside reconstruction.',
                '16-bit normal maps use a documented UV tangent basis; verify tangent compatibility in the destination DCC.',
                'Numerical tests and output generation do not replace visual review; local image display was unavailable in this tool session.']
    if camera_report['key_mean_pixels'] > image.height / 100:
        warnings.append('Key landmark projection error is large relative to fine details; alignment may contaminate normal residuals.')
    if sum(m['overlapping_raster_samples'] for m in maps) > 100:
        warnings.append('UV raster overlaps detected. Inspect per-material UV coverage before using normal maps.')
    result = dict(version='local-shading-normal-prototype-1', input_mesh=str(source_path),
        input_reference=str(reference_path), mesh_sha256=mesh.sha256, source_mesh_unchanged=original_unchanged,
        input_vertices=len(mesh.vertices), input_polygons=mesh.polygon_count, computational_triangles=len(mesh.triangles),
        original_topology_and_uv_unchanged=original_unchanged, screen_size=list(image.size),
        valid_skin_pixels=int(valid.sum()), total_patches=len(regions), refined_patches=sum(p['refined'] for p in detail['patches']),
        virtual_cell_definition='Visible pixel-footprint samples with parent triangle, barycentric coordinates, UV and normal; no topological subdivision.',
        virtual_cells_reprojection_max_pixels=float(pixel_error.max()) if len(pixel_error) else 0.,
        camera=camera_report, lighting=light_report, solver=stats, maps=maps,
        estimated_noise_linear=float(detail['noise']), skin_chromaticity=masks['skin_chromaticity'],
        observations=provenance, elapsed_seconds=time.perf_counter()-start, warnings=warnings)
    if not original_unchanged:
        raise RuntimeError('Input mesh changed during run; refusing to certify topology preservation.')
    if not np.isfinite(refined).all() or stats['normal_angle_max_degrees'] > settings.maximum_angle_degrees + 1e-3:
        raise RuntimeError('Normal constraints failed.')
    save_json(out / 'report.json', result)
    save_json(out / 'run_config.json', cfg)
    print(json.dumps({k:result[k] for k in ('output','elapsed_seconds') if k in result} | {
        'output':str(out),'normal_angle_max_degrees':stats['normal_angle_max_degrees'],
        'active_virtual_cells':stats['active_virtual_cells'],'source_mesh_unchanged':original_unchanged}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('demo.json'))
    parser.add_argument('--screen-resolution', type=int)
    parser.add_argument('--texture-resolution', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    for key in ('screen_resolution','texture_resolution','output'):
        value = getattr(args,key)
        if value is not None:
            cfg[key] = str(value) if isinstance(value,Path) else value
    for key in ('screen_resolution','texture_resolution'):
        if not 64 <= cfg.get(key,768) <= 4096:
            parser.error(f'{key} must be between 64 and 4096.')
    run(cfg)


if __name__ == '__main__':
    main()
