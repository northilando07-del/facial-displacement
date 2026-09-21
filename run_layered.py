"""Integrable Fold/Wrinkle/Fine layers and independent lips; legacy mode retained."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from normal_geometry import calibrate_camera, group_patches, load_obj, rasterize, unit
from normal_layers import (DEFAULT_CAPS, DEFAULT_GAINS, LAYER_NAMES, analyze_bands,
    fit_lip_appearance, lip_masks, lip_observation, normal_from_slope, slope_from_base,
    solve_band)
from normal_multilight import (MultiSettings, angular, blur, constrain_normals,
    fit_metrics, sample_image, similarity_registration)
from normal_shading import fit_light, linear_to_srgb, srgb_to_linear
from normal_structure import direction_weights, semantic_regions, smooth_along
from run_refine import (bake_uv, image8, montage, observation_points, png16,
                        save_json, semantic_mask)
from run_upgrade import alignment_confidence, choose_work_size, digest
from normal_pores import fill_pores
from normal_geometry_field import (geometry_settings, geometry_observation,
    solve_geometry, surface_metric, integrate_height, split_height, height_to_normal,
    surface_support, resample_height, limit_height_edges)
from normal_reference_fusion import fusion_settings, fuse_reference_height
from normal_semantic_layers import (SEMANTIC_NAMES, SEMANTIC_GAINS, SEMANTIC_CAPS,
                                   semantic_settings, build_semantic_heights)


def merge_uv_layers(out, materials, gains, maximum_angle, layer_names=LAYER_NAMES):
    """Compose the EXPORTED layer normals, keeping the shader and combined PNG aligned."""
    summaries = []
    for material in materials:
        safe_material=re.sub(r'[^A-Za-z0-9_.-]', '_', material)
        slope = coverage = confidence = neutral = None
        for name in layer_names:
            path = out/'layers'/name/'maps'/safe_material/'normal_float32.npz'
            with np.load(path) as data:
                normal = data['tangent_normal']
                layer_confidence = data['confidence']
                if slope is None:
                    slope = np.zeros_like(normal)
                    neutral = np.zeros_like(normal); neutral[..., 2] = 1
                    coverage = data['uv_occupied']
                    confidence = np.zeros(normal.shape[:2], np.float32)
                slope[..., :2] += gains[name]*normal[..., :2]/np.maximum(normal[..., 2:], .1)
                if gains[name] > 0:
                    confidence = np.maximum(confidence, layer_confidence)
                del normal, layer_confidence
        tangent = normal_from_slope(neutral, slope, maximum_angle)
        directory = out/'maps'/safe_material; directory.mkdir(parents=True, exist_ok=True)
        gl = tangent*.5+.5
        png16(directory/'normal_opengl16.png', gl)
        gl[..., 1] = 1-gl[..., 1]
        png16(directory/'normal_directx16.png', gl)
        gl[..., 1] = 1-gl[..., 1]
        image8(gl).save(directory/'normal_preview8.png')
        image8(confidence).save(directory/'confidence.png')
        image8(coverage).save(directory/'uv_coverage.png')
        np.savez_compressed(directory/'normal_float32.npz', tangent_normal=tangent,
                            confidence=confidence, uv_occupied=coverage)
        summaries.append(dict(material=material, directory=str(directory), resolution=tangent.shape[0],
            png_bit_depth=16, corrected_texels=int((confidence > .015).sum()),
            uv_occupied_texels=int(coverage.sum()),
            composition='Weighted common-base tangent slopes; one normalization and an angular cap.',
            gains=gains, maximum_angle_degrees=maximum_angle,
            tangent_basis='Original per-UV-corner tangent basis; not certified MikkTSpace.'))
        del slope, neutral, tangent, confidence, coverage, gl
        gc.collect()
    return summaries


def native_evidence(cfg, records, mesh, native, target, geom, domain, lips_domain,
                    labels, semantics, out, anchor_size):
    geometry_cfg = geometry_settings(cfg.get('geometry'))
    geometry_enabled = geometry_cfg['enabled']
    fusion_cfg = fusion_settings(cfg.get('reference_fusion'))
    semantic_cfg = semantic_settings(cfg.get('semantic_layers'))
    semantic_enabled = geometry_enabled and semantic_cfg['enabled']
    fusion_enabled = geometry_enabled and fusion_cfg['enabled'] and not semantic_enabled
    cache_dir = Path(cfg.get('cache_directory', out/'cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_data = dict(mesh=mesh.sha256, records=records, settings=cfg.get('layers', {}),
        skin_materials=cfg['skin_materials'], continuity=cfg.get('continuity', True), geometry=geometry_cfg,
        reference_fusion=fusion_cfg, semantic_layers=semantic_cfg,
        target=target.tolist(), protection=cfg.get('protection', {}), camera_vertices_sha256=hashlib.sha256(native['screen_vertices'].tobytes()).hexdigest(),
        code={name: digest(Path(__file__).with_name(name)) for name in
              ('run_layered.py', 'normal_layers.py', 'normal_structure.py', 'normal_multilight.py', 'normal_geometry_field.py', 'normal_reference_fusion.py', 'normal_semantic_layers.py', 'run_refine.py')})
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    npz_path, json_path = cache_dir/(key+'.npz'), cache_dir/(key+'.json')
    if npz_path.exists() and json_path.exists():
        print('Reusing hash-bound full-resolution layer evidence.', flush=True)
        with np.load(npz_path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        meta = json.loads(json_path.read_text(encoding='utf-8'))
        for entry in meta['graphs']:
            save_json(out/entry['filename'], entry['graph'])
        return arrays, meta

    base = native['normal']; shape = base.shape[:2]
    arrays = {}
    if fusion_enabled or semantic_enabled:
        arrays['fusion_semantics'] = semantics
    if geometry_enabled:
        for suffix in ('observations', 'confidences', 'lights', 'ambients'):
            arrays['geometry_'+suffix] = []
    for name in LAYER_NAMES:
        arrays[name+'_residuals'] = []
        arrays[name+'_confidences'] = []
        arrays[name+'_lights'] = []
        arrays[name+'_ambients'] = []
        arrays[name+'_tensor'] = np.zeros((*shape, 3), np.float32)
        arrays[name+'_support'] = np.zeros(shape, np.float32)
    meta = dict(views=[], graphs=[], decomposition_max_error=0., cache_key=key)
    panels = []
    layer_cfg = cfg.get('layers', {})
    tolerances = dict(low=14., mid=8., high=4., lips=4.)
    tolerances.update(layer_cfg.get('registration_tolerances_pixels', {}))
    for index, record in enumerate(records):
        print(f'Original-resolution layer evidence {index+1}/{len(records)}', flush=True)
        image = ImageOps.exif_transpose(Image.open(record['path'])).convert('RGB')
        matrix, offset, registration = similarity_registration(target, record['landmarks'])
        if index == 0:
            matrix, offset = np.eye(2), np.zeros(2)
            registration.update(matrix=matrix.tolist(), offset=offset.tolist(),
                per_landmark_residual=[0.]*68, mean_landmark_residual=0., p95_landmark_residual=0.)
        if registration['p95_landmark_residual'] > anchor_size[1]*.06:
            raise ValueError('Reference expression/pose differs too much: '+record['path'])
        rgb = sample_image(np.asarray(image, np.float32)/255, shape, matrix, offset, anchor_size)
        observed = (srgb_to_linear(rgb) @ np.array([.2126, .7152, .0722], np.float32)).astype(np.float32)
        valid, _ = semantic_mask(rgb, target, geom, base,
            brow_width_scale=cfg.get('protection', {}).get('brow_width_scale', .55))
        valid &= domain
        ambient, light, light_report = fit_light(base, blur(observed, valid, 2), valid)
        geometry_light_report = None
        if geometry_enabled:
            registration_weight = alignment_confidence(shape, target,
                registration['per_landmark_residual'], domain,
                tolerance=geometry_cfg['registration_tolerance_pixels'])
            ga, gl, gw, geometry_light_report = geometry_observation(
                base, observed, valid, semantics, registration_weight)
            arrays['geometry_observations'].append(observed.copy())
            arrays['geometry_confidences'].append(gw)
            arrays['geometry_lights'].append(gl)
            arrays['geometry_ambients'].append(ga)
        bands, pyramid = analyze_bands(base, observed, valid, labels, semantics, ambient, light,
            sigmas=layer_cfg.get('skin_sigmas_pixels', [.8, 3., 18., 64.]),
            continuity=cfg.get('continuity', True),
            selected_names=() if semantic_enabled else ('high',) if geometry_enabled and not fusion_enabled else None,
            brightness_bins=fusion_cfg['brightness_bins'] if fusion_enabled else 0)
        meta['decomposition_max_error'] = max(meta['decomposition_max_error'], pyramid['reconstruction_max_error'])
        # Three scalar residuals can share RGB. This file is NOT a normal map.
        residual_scale = .18
        packed = np.stack([pyramid[k] for k in ('low', 'mid', 'high')], -1)
        png16(out/f'residual_layers_rgb16_{index:02d}.png', .5+.5*np.clip(packed/residual_scale, -1, 1))
        image8(.5+.5*np.clip(packed/residual_scale, -1, 1)).save(out/f'residual_layers_preview_{index:02d}.png')
        del pyramid, packed

        # Each reference's own lip landmarks are brought through only the same
        # global registration. Different mouth shapes are not locally warped.
        own_lm = (np.asarray(record['landmarks'])-offset) @ np.linalg.inv(matrix).T
        lip_valid, lip_report = lip_observation(rgb, own_lm, geom, base, lips_domain)
        lip_ambient, lip_light, appearance = fit_lip_appearance(base, observed, lip_valid, light)
        lip_semantics = np.full(shape, 14, np.int16)
        lip_bands, lip_pyramid = analyze_bands(base, observed, lip_valid, labels, lip_semantics,
            lip_ambient, lip_light, lips=True,
            sigmas=layer_cfg.get('lip_sigmas_pixels', [.65, 2.3, 9., 30.]),
            continuity=cfg.get('continuity', True))
        meta['decomposition_max_error'] = max(meta['decomposition_max_error'], lip_pyramid['reconstruction_max_error'])
        bands.update(lip_bands)
        image8(lip_valid).save(out/f'lip_observation_mask_{index:02d}.png')
        details = {}
        for name in bands:
            item = bands[name]
            errors = registration['per_landmark_residual']
            points, errors_for_region = (target[48:68], errors[48:68]) if name == 'lips' else (target, errors)
            reg_conf = alignment_confidence(shape, points, errors_for_region,
                lips_domain if name == 'lips' else domain, tolerance=tolerances[name])
            item['confidence'] *= reg_conf
            arrays[name+'_residuals'].append(item['residual'])
            arrays[name+'_confidences'].append(item['confidence'])
            arrays[name+'_lights'].append(lip_light if name == 'lips' else light)
            arrays[name+'_ambients'].append(lip_ambient if name == 'lips' else ambient)
            t = item['tangent']; weight = item['support']*item['confidence']
            arrays[name+'_tensor'][..., 0] += weight*t[..., 0]**2
            arrays[name+'_tensor'][..., 1] += weight*t[..., 0]*t[..., 1]
            arrays[name+'_tensor'][..., 2] += weight*t[..., 1]**2
            arrays[name+'_support'] = np.maximum(arrays[name+'_support'], item['support'])
            if index == 0:
                arrays[name+'_anchor_direction'] = item['direction']
            graph_name = f'structure_{name}_{index:02d}.json'
            save_json(out/graph_name, item['graph'])
            meta['graphs'].append(dict(filename=graph_name, graph=item['graph']))
            details[name] = dict(noise=item['noise'], observation_pixels=int((item['confidence'] > .015).sum()),
                                 structure=item['graph']['stats'])
            if name in ('mid', 'lips'):
                overlay = image8(rgb); draw = ImageDraw.Draw(overlay)
                for edge in item['graph']['edges']:
                    a, b = (item['graph']['nodes'][edge[k]] for k in ('a', 'b'))
                    draw.line([tuple(a['center']), tuple(b['center'])], width=2,
                              fill=(255, 100, 20) if edge['cross_region'] else (40, 230, 100))
                overlay.save(out/f'structure_links_{name}_{index:02d}.png')
            image8(item['confidence']).save(out/f'confidence_{name}_{index:02d}.png')
        panels.append((f'Reference {index+1}: registered', image8(rgb)))
        meta['views'].append(dict(path=record['path'], sha256=record['sha256'], original_size=list(image.size),
            registration=registration, lighting=light_report, geometry_lighting=geometry_light_report, lip_appearance=appearance,
            valid_skin_pixels=int(valid.sum()), valid_lip_pixels=int(lip_valid.sum()),
            lip_color_statistics={key: lip_report[key] for key in ('lip_chromaticity', 'chromaticity_threshold') if key in lip_report},
            layers=details))
        print('  skin/lip observed pixels:', int(valid.sum()), int(lip_valid.sum()),
              'band supports:', {k: v['observation_pixels'] for k, v in details.items()}, flush=True)
        del bands, lip_bands, lip_pyramid, rgb, observed, valid, lip_valid
    for name in LAYER_NAMES:
        for suffix in ('residuals', 'confidences', 'lights', 'ambients'):
            arrays[name+'_'+suffix] = np.asarray(arrays[name+'_'+suffix], np.float32)
    if geometry_enabled:
        for suffix in ('observations', 'confidences', 'lights', 'ambients'):
            arrays['geometry_'+suffix] = np.asarray(arrays['geometry_'+suffix], np.float32)
    np.savez(npz_path, **arrays)
    save_json(json_path, meta)
    montage(out/'registered_references.png', panels, 2)
    save_json(out/'residual_channels.json', dict(channels=dict(R='low', G='mid', B='high'),
        encoding='signed linear residual = (channel*2 - 1)*0.18', neutral=.5,
        coordinates='registered reference image pixels, not UV', is_normal_map=False,
        used_for_low_mid=(not geometry_enabled) or fusion_enabled,
        note='Document 55 uses residuals for a bounded primary/structure prior, then forms physical height bands. RGB cannot contain three complete XYZ normals.'))
    return arrays, meta


def reconstruct_geometry(cfg, arrays, mesh, camera, raster, points, domain, out):
    options = geometry_settings(cfg.get('geometry'))
    fusion_cfg = fusion_settings(cfg.get('reference_fusion'))
    semantic_cfg = semantic_settings(cfg.get('semantic_layers'))
    if semantic_cfg['enabled']:
        directory = out/'geometry'; directory.mkdir(parents=True, exist_ok=True)
        metric, metric_report = surface_metric(mesh, camera, raster, options, points)
        result = build_semantic_heights(raster['normal'], arrays, domain, metric, options,
            semantic_cfg, cfg.get('layers', {}).get('angle_caps_degrees'))
        np.savez_compressed(directory/'field.npz', height_mm=result['height'].astype(np.float32),
            valid=result['valid'], confidence=result['confidence'],
            metric_xx=metric['xx'], metric_xy=metric['xy'], metric_yy=metric['yy'])
        np.savez_compressed(directory/'height_bands.npz',
            **{k: v.astype(np.float32) for k, v in result['bands'].items()})
        np.savez_compressed(directory/'semantic_evidence.npz', **result['diagnostic'])
        for name, field in result['diagnostic'].items():
            image8(field).save(directory/(name+'.png'))
        previews = {}
        for name, field in result['bands'].items():
            scale = max(float(np.quantile(np.abs(field[result['valid']]), .995)), 1e-8) if result['valid'].any() else 1.
            image8(.5+.5*np.clip(field/scale, -1, 1)).save(directory/(name+'_preview.png'))
            previews[name] = dict(zero=.5, scale_mm=scale, display_clipped=True)
        image8(result['confidence']).save(directory/'confidence.png')
        save_json(directory/'structure_matching.json', result['report']['matching'])
        fusion_report = result['report']
        fusion_report['primary_reference'] = cfg.get('references', ['See input_reference'])[0]
        report = dict(enabled=True, metric=metric_report, height_previews=previews,
            reference_fusion=fusion_report, semantic_layers=True,
            integrability=dict(converged=all(v['integrability']['converged'] for v in fusion_report['layers'].values()),
                               layers={k: v['integrability'] for k, v in fusion_report['layers'].items()}),
            geometry_active_samples=int(result['valid'].sum()),
            limitations=fusion_report['limitations'])
        save_json(directory/'report.json', report)
        return dict(result, metric=metric, report=report)
    if fusion_cfg['enabled']:
        options['confidence_as_blend_weight'] = True
    directory = out/'geometry'; directory.mkdir(parents=True, exist_ok=True)
    print('Joint geometry from unfiltered multi-light observations...', flush=True)
    candidate, confidence, extra = solve_geometry(raster['normal'],
        arrays['geometry_observations'], arrays['geometry_confidences'],
        arrays['geometry_lights'], arrays['geometry_ambients'], options)
    metric, metric_report = surface_metric(mesh, camera, raster, options, points)
    fusion_report = dict(enabled=False)
    if fusion_cfg['enabled']:
        print('Building the bounded primary-image height and continuous source weights...', flush=True)
        fused = fuse_reference_height(raster['normal'], candidate, extra, arrays,
            domain, metric, options, fusion_cfg)
        height, confidence, valid = fused['height'], fused['confidence'], fused['valid']
        gx, gy, projection = fused['gx'], fused['gy'], fused['projection']
        fusion_report = fused['report']
        fusion_report['primary_reference'] = cfg.get('references', ['See report.input_reference'])[0]
        observation_valid = domain & metric['valid'] & (
            (arrays['geometry_confidences'][0] > .001) | (extra['photo_confidence'] > 0))
        np.savez_compressed(directory/'reference_fusion.npz',
            **{k: a.astype(np.float32) for k, a in fused['diagnostic'].items()})
        for key in ('weight_multi', 'weight_primary', 'weight_structure', 'photo_confidence', 'structure_confidence'):
            image8(fused['diagnostic'][key]).save(directory/(key+'.png'))
        save_json(directory/'structure_matching.json', fusion_report['matching'])
        del fused
    else:
        observation_valid = domain & metric['valid'] & (confidence > .015)
        valid = surface_support(observation_valid)
        confidence[~valid] = 0
        slope = slope_from_base(raster['normal'], candidate)
        gx = -np.sum(slope*metric['jx'], -1)
        gy = -np.sum(slope*metric['jy'], -1)
        gx[~valid] = 0; gy[~valid] = 0
        height, projection = integrate_height(gx, gy, confidence, valid, metric, options)
    print('Splitting physical surface scales:', options['sigmas_mm'], flush=True)
    bands, frequency = split_height(height, valid, metric, options)
    np.savez_compressed(directory/'field.npz', height_mm=height.astype(np.float32),
        raw_gx_mm_per_pixel=gx, raw_gy_mm_per_pixel=gy, valid=valid,
        confidence=confidence, normal_photometric=candidate,
        relative_albedo=extra['relative_albedo'], condition=extra['condition'],
        photometric_fit_error=extra['fit_error'],
        metric_xx=metric['xx'], metric_xy=metric['xy'], metric_yy=metric['yy'])
    np.savez_compressed(directory/'height_bands.npz',
        **{k: a.astype(np.float32) for k, a in bands.items()})
    image8(confidence).save(directory/'confidence.png')
    image8(np.clip(extra['condition']/.12, 0, 1)).save(directory/'light_condition.png')
    previews = {}
    for name, field in dict(height=height, **{k: bands[k] for k in ('low', 'mid', 'mid_fine', 'mid_large')}).items():
        scale = max(float(np.quantile(np.abs(field[valid]), .995)), 1e-8) if valid.any() else 1.
        encoded = .5+.5*np.clip(field/scale, -1, 1)
        image8(encoded).save(directory/(name+'_preview.png'))
        previews[name] = dict(zero=.5, scale_mm=scale, display_clipped=True)
    report = dict(enabled=True, joint_fit=extra['stats'], metric=metric_report,
        integrability=projection, frequency=frequency, height_previews=previews,
        reference_fusion=fusion_report,
        geometry_active_samples=int(valid.sum()),
        unsupported_surface_samples=int((observation_valid & ~valid).sum()),
        limitations=[
            'Estimated effective illumination and AI references do not identify measured true geometry.',
            'Height is a linearized small base-normal offset on the visible chart; it is not applied as displacement.',
            'Integrability applies to low/mid height fields; high/lips remain conservative legacy detail branches.',
            'UV interpolation, texture quantization and renderer tangent conventions are outside the height-field integrability guarantee.'
        ])
    save_json(directory/'report.json', report)
    return dict(bands=bands, confidence=confidence, valid=valid, metric=metric, report=report)


def run(cfg):
    start = time.perf_counter()
    out = Path(cfg['output']).resolve(); out.mkdir(parents=True, exist_ok=True)
    layer_cfg = cfg.get('layers', {})
    geometry_cfg = geometry_settings(cfg.get('geometry'))
    geometry_enabled = geometry_cfg['enabled']
    semantic_cfg = semantic_settings(cfg.get('semantic_layers'))
    semantic_enabled = semantic_cfg['enabled']
    if semantic_enabled and not geometry_enabled:
        raise ValueError('Semantic layers require the surface-metric geometry path.')
    layer_names = SEMANTIC_NAMES if semantic_enabled else LAYER_NAMES
    gain_defaults = SEMANTIC_GAINS if semantic_enabled else DEFAULT_GAINS
    cap_defaults = SEMANTIC_CAPS if semantic_enabled else DEFAULT_CAPS
    gains = {name: layer_cfg.get('gains', {}).get(name, gain_defaults[name]) for name in layer_names}
    caps = {name: layer_cfg.get('angle_caps_degrees', {}).get(name, cap_defaults[name]) for name in layer_names}
    total_cap = float(layer_cfg.get('combined_angle_cap_degrees', 12.))
    for name in layer_names:
        if not np.isfinite(gains[name]) or gains[name] < 0 or not 0 < caps[name] < 85:
            raise ValueError('Invalid layer gain or angular limit: '+name)
    if not 0 < total_cap < 85:
        raise ValueError('Invalid combined normal angle limit.')
    records = json.loads(Path(cfg['reference_detections']).read_text(encoding='utf-8-sig'))
    records = records[:cfg.get('max_references', len(records))]
    if not records:
        raise ValueError('At least one reference is required.')
    for record in records:
        if digest(record['path']) != record['sha256']:
            raise ValueError('Reference landmark cache is stale: '+record['path'])
    mesh = load_obj(Path(cfg['mesh']))
    materials = cfg['skin_materials']
    if any(name not in mesh.materials for name in materials):
        raise ValueError('A configured surface material is not in the source OBJ.')
    anchor = ImageOps.exif_transpose(Image.open(records[0]['path'])).convert('RGB')
    points, target, provenance = observation_points(dict(cfg, reference=records[0]['path']), mesh, anchor, out)
    native_camera, native_camera_report = calibrate_camera(points, target,cfg.get('source_rotation'))
    skipped = [n for n in mesh.materials if any(k in n.lower() for k in ('eyebrow', 'eyelash', 'tear'))]
    native = rasterize(mesh, native_camera, (anchor.height, anchor.width), skipped)
    tids = native['triangle']; base_native = native['normal']
    mids = [mesh.materials.index(n) for n in materials]
    geom = (tids >= 0) & np.isin(mesh.material_ids[np.maximum(tids, 0)], mids)
    protection=cfg.get('protection', {})
    _, skin_masks = semantic_mask(np.asarray(anchor, np.float32)/255, target, geom, base_native,
        brow_width_scale=protection.get('brow_width_scale', .55))
    domain = skin_masks['anatomical']
    lips = lip_masks(domain.shape, target, geom, base_native)
    lip_domain = lips['domain']
    if np.any(domain & lip_domain):
        raise RuntimeError('Skin and lips overlap in the independent analysis masks.')
    if int(lip_domain.sum()) < 32:
        raise ValueError('No credible lip surface after projection. Inspect source landmarks/materials.')
    all_domain = domain | lip_domain
    labels, regions = group_patches(mesh, native, all_domain, radius_pixels=anchor.height*.0586)
    semantics = semantic_regions(domain.shape, target, domain)
    for name, mask in [('skin_domain', domain), ('lip_domain', lip_domain), ('mouth_opening_protected', lips['mouth_opening'])]:
        image8(mask).save(out/(name+'.png'))
    overlay = np.asarray(anchor, np.float32)/255
    overlay[domain] = .55*overlay[domain]+[.12, .35, .10]
    overlay[lip_domain] = .55*overlay[lip_domain]+[.40, .06, .10]
    image8(overlay).save(out/'independent_regions_overlay.png')
    arrays, evidence_meta = native_evidence(cfg, records, mesh, native, target, geom,
        domain, lip_domain, labels, semantics, out, anchor.size)
    geometry_result = reconstruct_geometry(cfg, arrays, mesh, native_camera,
        native, points, domain, out) if geometry_enabled else None
    work_size = choose_work_size(anchor.size, int(all_domain.sum()), cfg.get('virtual_surface_samples', 2000000),
                                 cfg.get('max_analysis_side', 4096))
    work_shape = work_size[::-1]
    target_work = (target+.5)*(np.asarray(work_size)/anchor.size)-.5
    camera, camera_report = calibrate_camera(points, target_work,cfg.get('source_rotation'))
    save_json(out/'camera.json', camera_report)
    raster = native if work_size == anchor.size else rasterize(mesh, camera, work_shape, skipped)
    base = raster['normal']; tid = raster['triangle']
    if geometry_enabled:
        work_metric = geometry_result['metric'] if raster is native else surface_metric(
            mesh, camera, raster, geometry_cfg, points)[0]
    high_geom = (tid >= 0) & np.isin(mesh.material_ids[np.maximum(tid, 0)], mids)
    domains = dict(skin=(sample_image(domain.astype(np.float32), work_shape) > .99) & high_geom,
                   lips=(sample_image(lip_domain.astype(np.float32), work_shape) > .99) & high_geom)
    brow=sample_image(skin_masks['brow_exclusions'].astype(np.float32),work_shape)>.5
    allowed=(sample_image(skin_masks['anatomical_without_brows'].astype(np.float32),work_shape)>.99)&high_geom
    rows=np.arange(work_shape[0])[:,None]
    forehead=rows<float(np.min(target_work[17:27,1]))
    pore_region=allowed & (brow|forehead) & ~domains['lips']
    synthesis_report=dict(enabled=False if semantic_enabled else protection.get('fill_pores',True),
                          synthesized_pixels=0)
    image8(domains['lips']).save(out/'lip_domain_work.png')
    print('Virtual grid:', work_size, 'visible skin/lip samples:', int(domains['skin'].sum()),
          int(domains['lips'].sum()), flush=True)
    combined_slope = np.zeros_like(base)
    combined_confidence = np.zeros(work_shape, np.float32)
    layer_reports = {}
    layer_panels = [('Reference', anchor)]
    probe_light = unit(np.array([-.65, .3, .7]))
    before = image8(linear_to_srgb(.08+.65*np.maximum(base@probe_light, 0)))
    layer_panels.append(('Base normal', before))
    for name in layer_names:
        layer_start = time.perf_counter()
        layer_out = out/'layers'/name; layer_out.mkdir(parents=True, exist_ok=True)
        active_domain = domains['lips' if name == 'lips' else 'skin']
        # A band is a perturbation only. Do not insert a second resampled base
        # normal into every frequency layer.
        geometry_band = geometry_enabled and name in (('fold', 'wrinkle', 'fine') if semantic_enabled else ('low', 'mid'))
        options = dict(cfg.get('solver', {}))
        options['maximum_angle_degrees'] = caps[name]
        options['prior'] = max(options.get('prior', .003), .025 if name == 'low' else .006 if name == 'high' else .003)
        settings = MultiSettings(**options)
        print(f'Solving {name}: {caps[name]:g} degree cap, display gain {gains[name]:g}', flush=True)
        if geometry_band:
            height, supported = resample_height(geometry_result['bands'][name],
                geometry_result['valid'], work_shape)
            supported = surface_support(supported & active_domain & work_metric['valid'])
            source_confidence = geometry_result.get('layer_confidence', {}).get(name, geometry_result['confidence'])
            confidence = sample_image(source_confidence, work_shape)*supported
            height[~supported] = 0
            local_limit = None
            if geometry_result['report']['reference_fusion'].get('enabled'):
                print('  limiting local height outliers before differentiation', flush=True)
                height, local_limit = limit_height_edges(height, supported, work_metric, caps[name])
            refined, height, cap_report = height_to_normal(base, height, supported, work_metric, caps[name])
            if semantic_enabled and cap_report['height_cap_scale'] < 1-1e-5:
                raise RuntimeError('A semantic height field still required global attenuation: '+name)
            if local_limit is not None:
                cap_report['local_height_limit'] = local_limit
            np.savez_compressed(layer_out/'height_field.npz', height_mm=height.astype(np.float32), valid=supported)
            extra = dict(stats=dict(method='Differentiate a band of one integrable, surface-metric height field',
                **cap_report, primary_prior_native_samples=geometry_result['report']['reference_fusion'].get('primary_dominant_samples', 0)))
            residuals = confidences = direction = None
        else:
            residuals = np.stack([sample_image(a, work_shape) for a in arrays[name+'_residuals']])
            confidences = np.stack([sample_image(a, work_shape)*active_domain for a in arrays[name+'_confidences']])
            residuals[:, ~active_domain] = 0
            light_vectors = arrays[name+'_lights']
            direction = unit(sample_image(arrays[name+'_anchor_direction'], work_shape))
            refined, confidence, extra = solve_band(base, residuals, confidences, light_vectors, direction,
                settings=settings, name=name, allow_anchor_fallback=(name == 'lips' and layer_cfg.get('lip_anchor_fallback', True)))
        if not geometry_band and name in ('mid', 'lips') and cfg.get('continuity', True):
            tensor = arrays[name+'_tensor']
            phi = .5*np.arctan2(2*tensor[..., 1], tensor[..., 0]-tensor[..., 2])
            tangent = unit(sample_image(np.stack([np.cos(phi), np.sin(phi)], -1).astype(np.float32), work_shape))
            support = sample_image(arrays[name+'_support'], work_shape)*active_domain
            wh, wv = direction_weights(tangent, support, (confidence > .015) & active_domain, base)
            delta = smooth_along(slope_from_base(base, refined), wh, wv, iterations=4)
            refined = constrain_normals(base, unit(base+delta), confidence, caps[name])
            del tangent, support, wh, wv, delta
        if name=='high' and protection.get('fill_pores',True):
            refined,confidence,generated,synthesis_report=fill_pores(base,refined,confidence,
                pore_region,domains['skin']&~brow,
                pixel_scale=work_size[1]/anchor.height,
                strength=protection.get('pore_strength',.35),
                max_angle=min(caps[name],protection.get('pore_max_angle',1.2)),
                max_distance=protection.get('donor_distance',80.))
            active_domain=active_domain|(generated>0)
            domains['skin']|=generated>0
            image8(generated).save(out/'synthesized_pores_mask.png')
            save_json(out/'pore_synthesis.json',synthesis_report)
        angle = angular(base, refined)
        if np.any(~np.isfinite(refined)) or float(angle.max()) > caps[name]+.002:
            raise RuntimeError('Per-layer angular/finite constraint failed: '+name)
        if not np.array_equal(refined[~active_domain], base[~active_domain]):
            raise RuntimeError('An independent layer leaked outside its region: '+name)
        stats = extra['stats']
        if not geometry_band:
            stats.update(fit_metrics(base, refined, residuals, confidences, light_vectors))
        observed = confidence > .015
        stats.update(maximum_angle_degrees=float(angle.max()),
                     mean_active_angle_degrees=float(angle[observed].mean()) if observed.any() else 0.,
                     p95_active_angle_degrees=float(np.quantile(angle[observed], .95)) if observed.any() else 0.)
        print('  baking', name, 'active samples', int(observed.sum()), flush=True)
        maps = [] if cfg.get('evidence_only', False) else bake_uv(
            mesh, camera, raster, refined, confidence, materials,
            cfg.get('texture_resolution', 4096), layer_out)
        np.savez_compressed(layer_out/'screen_fields.npz', normal_refined=refined,
                            confidence=confidence, domain=active_domain)
        image8(confidence).save(layer_out/'confidence.png')
        save_json(layer_out/'report.json', dict(layer=name, gain=gains[name], cap_degrees=caps[name],
            solver=stats, settings=geometry_cfg if geometry_band else asdict(settings), maps=maps))
        layer_panels.append((name+' only (gain 1)', image8(linear_to_srgb(.08+.65*np.maximum(refined@probe_light, 0)))))
        if gains[name] > 0:
            combined_slope += gains[name]*slope_from_base(base, refined)
            combined_confidence = np.maximum(combined_confidence, confidence)
        layer_reports[name] = dict(gain=gains[name], cap_degrees=caps[name], solver=stats, maps=maps,
            eligible_samples=int(active_domain.sum()), active_samples=int(observed.sum()),
            outside_domain_max_angle_degrees=float(angle[~active_domain].max()),
            elapsed_seconds=time.perf_counter()-layer_start)
        print('  complete', name, 'max/mean degrees', stats['maximum_angle_degrees'],
              stats['mean_active_angle_degrees'], flush=True)
        del residuals, confidences, direction, refined, confidence, extra, angle
        gc.collect()
    refined = normal_from_slope(base, combined_slope, total_cap)
    del combined_slope
    maps = [] if cfg.get('evidence_only', False) else merge_uv_layers(out, materials, gains, total_cap, layer_names)
    angle = angular(base, refined); active = combined_confidence > .015
    if not active.any() and not cfg.get('evidence_only', False):
        raise RuntimeError('All layers returned neutral normals; inspect observations.')
    if not np.array_equal(refined[~(domains['skin'] | domains['lips'])], base[~(domains['skin'] | domains['lips'])]):
        raise RuntimeError('Combined normal modified a protected region.')
    ys, xs = np.nonzero(active); ids = tid[active]; bary = raster['barycentric'][active]
    uv = np.sum(mesh.uv[np.maximum(mesh.triangle_uv[ids], 0)]*bary[..., None], axis=1)
    np.savez_compressed(out/'virtual_cells.npz', pixel_xy=np.column_stack([xs, ys]),
        triangle_id=ids, polygon_id=mesh.polygon_ids[ids], barycentric=bary, uv=uv,
        normal_base_camera=base[active], normal_refined_camera=refined[active],
        confidence=combined_confidence[active])
    np.savez_compressed(out/'screen_fields.npz', normal_base=base, normal_refined=refined,
        confidence=combined_confidence, triangle=tid, skin_domain=domains['skin'], lip_domain=domains['lips'])
    layer_panels.append(('Combined / default gains', image8(linear_to_srgb(.08+.65*np.maximum(refined@probe_light, 0)))))
    if cfg.get('comparison_previews',False):
        montage(out/'layer_comparison.png', layer_panels, columns=3)
    image8(linear_to_srgb(.08+.65*np.maximum(refined@probe_light, 0))).save(out/'relit_full_resolution.png')
    source_unchanged = digest(cfg['mesh']) == mesh.sha256
    refs_unchanged = all(digest(r['path']) == r['sha256'] for r in records)
    if not source_unchanged or not refs_unchanged:
        raise RuntimeError('An input changed during execution.')
    version = ('document-55-reference-fusion-1' if fusion_settings(cfg.get('reference_fusion'))['enabled']
               else 'document-44-geometry-2') if geometry_enabled else 'document-33-layers-1'
    if semantic_enabled:
        version = 'document-66-scale-proposals-1'
    report = dict(version=version, input_mesh=cfg['mesh'], input_reference=records[0]['path'],
        mesh_sha256=mesh.sha256, source_mesh_unchanged=source_unchanged, references_unchanged=refs_unchanged,
        original_topology_and_uv_unchanged=source_unchanged, input_vertices=len(mesh.vertices),
        input_polygons=mesh.polygon_count, computational_triangles=len(mesh.triangles),
        screen_size=list(work_size), native_reference_size=list(anchor.size), native_analysis_downsampling=False,
        virtual_surface_samples=int((domains['skin'] | domains['lips']).sum()),
        skin_virtual_samples=int(domains['skin'].sum()), lip_virtual_samples=int(domains['lips'].sum()),
        camera=camera_report, camera_native=native_camera_report, observations=provenance,
        views=evidence_meta['views'], layers=layer_reports, maps=maps,
        decomposition_max_error=evidence_meta['decomposition_max_error'],
        combined_gains=gains, combined_angle_cap_degrees=total_cap, normal_strength=cfg.get('normal_strength',.8),
        pore_synthesis=synthesis_report,
        geometry=geometry_result['report'] if geometry_enabled else dict(enabled=False),
        combined_max_angle_degrees=float(angle.max()),
        combined_mean_active_angle_degrees=float(angle[active].mean()) if active.any() else 0.,
        combined_p95_active_angle_degrees=float(np.quantile(angle[active], .95)) if active.any() else 0.,
        elapsed_seconds=time.perf_counter()-start,
        limitations=[
            ('Fold/Wrinkle/Fine have separate proposal policies, scalar heights, caps and gains; no pore synthesis. Class names do not prove anatomical identity.' if semantic_enabled else
             'Low/mid use physical-scale height bands; document 55 adds a bounded primary-image shading prior and local structure matching. These do not prove true geometry or semantic wrinkle identity.'),
            'Lip boundary is landmark based; makeup, specular highlights and base/image mismatch remain limitations.',
            'Missing lip observations may use a low-confidence anchor-only directional prior, separately counted.',
            'AI-relit images have no measured normal ground truth; shading error is not geometric accuracy.',
            'Native evidence is never downsampled; virtual supersampling adds no source information.',
            'No displacement, topology modification or source OBJ rewrite; original UV tangent convention retained.'
        ])
    save_json(out/'report.json', report); save_json(out/'run_config.json', cfg)
    print('LAYERED_COMPLETE', json.dumps(dict(output=str(out), seconds=report['elapsed_seconds'],
        max_angle=report['combined_max_angle_degrees'], layers={k: v['active_samples'] for k, v in layer_reports.items()}),
        ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('layers.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--texture-resolution', type=int)
    parser.add_argument('--max-references', type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.output: cfg['output'] = str(args.output)
    if args.native_only: cfg['virtual_surface_samples'] = 0
    if args.texture_resolution is not None: cfg['texture_resolution'] = args.texture_resolution
    if args.max_references is not None: cfg['max_references'] = args.max_references
    if not 64 <= cfg.get('texture_resolution', 4096) <= 8192:
        parser.error('Texture resolution must be 64..8192.')
    if cfg.get('max_references', 1) < 1 or cfg.get('virtual_surface_samples', 0) < 0:
        parser.error('Reference count must be positive and virtual sample count nonnegative.')
    if not cfg.get('reference_detections') or not cfg.get('skin_materials'):
        import subprocess,sys,uuid
        folder=Path(cfg['output']).resolve()/'auto_inputs'/'requests'
        folder.mkdir(parents=True,exist_ok=True)
        request=folder/(uuid.uuid4().hex+'.json');save_json(request,cfg)
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('prepare_inputs.py')),
            '--config',str(request)],check=True)
        return
    run(cfg)


if __name__ == '__main__':
    main()
