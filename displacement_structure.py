"""Document 88: connected coarse bases, multi-light normals and SOAP constraints.

The two virtual grids pool 16 and 4 ORIGINAL polygons respectively. Their
coefficients are XYZ vectors, blended on shared mesh vertices before solving.
The fixed camera/landmark correspondences survive each geometry update.
"""
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import os

import numpy as np
from PIL import Image, ImageOps
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import lsmr

from normal_geometry import load_obj, calibrate_camera, rasterize, adjacency, unit
from normal_multilight import sample_image, similarity_registration
from normal_geometry_field import geometry_observation, solve_geometry
from normal_shading import srgb_to_linear
from normal_structure import semantic_regions
from run_upgrade import alignment_confidence
from run_refine import semantic_mask
from displacement_geometry import Surface, edges_of, smooth_normals


def structure_settings(overrides=None):
    cfg = dict(enabled=True, cells=[16, 4], iterations_per_stage=3,
        maximum_displacement_mm=4., step_limits_mm=[3., 1.5],
        normal_angle_degrees=25., landmark_weight=4., contour_weight=2.,
        smoothness=1., screening_length_mm=60., analysis_max_side=1024,
        minimum_confidence=.025, cg_maxiter=2000)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise ValueError('structure 必须是配置对象')
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise ValueError('未知大形参数: ' + ', '.join(sorted(unknown)))
        cfg.update(overrides)
    if type(cfg['enabled']) is not bool or cfg['cells'] != [16, 4]:
        raise ValueError('大形粗读取顺序必须为 16 格、4 格')
    for key in ('maximum_displacement_mm', 'normal_angle_degrees', 'landmark_weight',
                'contour_weight', 'smoothness', 'screening_length_mm', 'minimum_confidence'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError('大形参数必须是正数: ' + key)
    if cfg['normal_angle_degrees'] >= 60 or cfg['minimum_confidence'] > 1:
        raise ValueError('大形法线角度或置信度超出范围')
    if not isinstance(cfg['step_limits_mm'], list) or len(cfg['step_limits_mm']) != 2:
        raise ValueError('两个粗层各需要一个位移步长')
    for value in cfg['step_limits_mm']:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError('粗层位移步长必须是正数')
    for key, lo, hi in (('iterations_per_stage', 1, 10), ('analysis_max_side', 256, 2048), ('cg_maxiter', 1, 10000)):
        if type(cfg[key]) is not int or not lo <= cfg[key] <= hi:
            raise ValueError('大形整数参数超出范围: ' + key)
    return cfg


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def coarse_basis(mesh, allowed, cells):
    """Pool connected polygons, then diffuse weights across shared boundaries.

    This is a partition-of-unity finite-element basis; adjacent regions share
    vertex displacements, with Laplacian regularization on the resulting field.
    No merge, remesh, UV projection or independent per-patch seam is introduced.
    """
    graph = adjacency(mesh)
    polygons = mesh.polygon_ids
    pg = [set() for _ in range(mesh.polygon_count)]
    active = np.zeros(mesh.polygon_count, bool)
    active[polygons[allowed]] = True
    for i in np.flatnonzero(allowed):
        for j in graph[i]:
            if allowed[j] and polygons[i] != polygons[j]:
                pg[polygons[i]].add(int(polygons[j]))
    labels = np.full(mesh.polygon_count, -1, np.int32)
    count = 0
    for seed in np.flatnonzero(active):
        if labels[seed] >= 0:
            continue
        queue = deque([int(seed)]); labels[seed] = count; size = 1
        while queue:
            for j in sorted(pg[queue.popleft()]):
                if labels[j] < 0 and size < cells:
                    labels[j] = count; queue.append(j); size += 1
        count += 1
    if not count:
        raise ValueError('粗层未找到可变形的面部网格')
    faces = mesh.triangles[allowed]
    n = len(mesh.vertices)
    basis = sparse.coo_matrix((np.ones(faces.size),
        (faces.ravel(), np.repeat(labels[polygons[allowed]], 3))), shape=(n, count)).tocsr()
    basis = sparse.diags(1 / np.maximum(np.asarray(basis.sum(1)).ravel(), 1)) @ basis
    edges, _, _ = edges_of(faces)
    rows, cols = edges.T
    graph = sparse.coo_matrix((np.ones(2*len(edges)),
        (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()
    transition = sparse.diags(1 / np.maximum(np.asarray(graph.sum(1)).ravel(), 1)) @ graph
    for _ in range(int(np.sqrt(cells))):
        basis = (.5*basis + .5*transition@basis).tocsr()
    # Pin the outer boundary to zero without severing internal patch continuity.
    fixed = np.zeros(n, bool)
    fixed[mesh.triangles[~allowed].ravel()] = True
    basis = sparse.diags((~fixed).astype(float)) @ basis
    keep = np.asarray(abs(basis).sum(0)).ravel() > 1e-12
    basis = basis[:, keep].tocsr()
    laplacian = sparse.eye(n, format='csr') - transition
    return basis, laplacian, dict(requested_polygons_per_region=cells,
        regions=int(keep.sum()), original_polygons=int(active.sum()),
        continuity='Shared vertex weights and field Laplacian; no topology merge')


def aligned_references(records, shape):
    size = records[0]['size']
    target = np.asarray(records[0]['landmarks'], float)
    rgbs, registrations = [], []
    for i, record in enumerate(records):
        matrix, offset, stats = similarity_registration(target, record['landmarks'])
        if i == 0:
            matrix, offset = np.eye(2), np.zeros(2)
            stats['per_landmark_residual'] = [0.]*68
            stats['p95_landmark_residual'] = 0.
        if stats['p95_landmark_residual'] > size[1]*.06:
            raise ValueError('多光源参考图的视角/表情差异过大: ' + record['path'])
        with Image.open(record['path']) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            rgbs.append(sample_image(np.asarray(image, np.float32)/255,
                shape, matrix, offset, size))
        registrations.append(stats)
    return rgbs, registrations


def normal_constraints(mesh, camera, shape, target, rgbs, registrations, materials, sigma, cfg):
    """Reproject the UPDATED mesh, refit global lights, solve low-frequency normals."""
    skip = [s for s in mesh.materials if any(x in s.lower() for x in ('eyebrow', 'eyelash', 'tear'))]
    raster = rasterize(mesh, camera, shape, skip)
    tid = raster['triangle']; base = raster['normal']
    geom = (tid >= 0) & np.isin(mesh.material_ids[np.maximum(tid, 0)], materials)
    observations, weights, lights, ambients, light_reports = [], [], [], [], []
    for rgb, registration in zip(rgbs, registrations):
        valid, masks = semantic_mask(rgb, target, geom, base, brow_width_scale=.5)
        semantics = semantic_regions(shape, target, masks['anatomical'])
        errors = np.asarray(registration['per_landmark_residual'])
        # Registration errors have already been converted to working pixels.
        alignment = alignment_confidence(shape, target, errors, masks['anatomical'], tolerance=8.)
        observed = (srgb_to_linear(rgb) @ np.array([.2126, .7152, .0722])).astype(np.float32)
        ambient, light, weight, light_report = geometry_observation(
            base, observed, valid, semantics, alignment)
        observations.append(observed); weights.append(weight)
        lights.append(light); ambients.append(ambient); light_reports.append(light_report)
    result, confidence, extra = solve_geometry(base, observations, weights, lights, ambients,
        dict(maximum_angle_degrees=cfg['normal_angle_degrees']))
    # Smooth the inferred normal residual, not absolute image luminance.
    mass = gaussian_filter(confidence, sigma)
    delta = np.stack([gaussian_filter((result[..., k]-base[..., k])*confidence, sigma)
        / np.maximum(mass, 1e-8) for k in range(3)], -1)
    result = unit(base + delta)
    confidence = np.minimum(confidence, mass)
    valid = geom & (confidence >= cfg['minimum_confidence'])
    face_ids = tid[valid]
    conf = confidence[valid]
    total = np.bincount(face_ids, weights=conf, minlength=len(mesh.triangles))
    hits = np.bincount(face_ids, minlength=len(mesh.triangles))
    normals = result[valid] @ camera.rotation
    face_normals = unit(np.column_stack([np.bincount(face_ids, weights=normals[:, k]*conf,
        minlength=len(mesh.triangles)) for k in range(3)]))
    return face_normals, total/np.maximum(hits, 1), raster, dict(
        photometry=extra['stats'], lighting=light_reports, lowpass_sigma_pixels=float(sigma),
        supported_faces=int((total > 0).sum()), reprojected_current_geometry=True)


def solve_vectors(surface, basis, laplacian, target_normals, confidence,
                  indices, bary, target, camera, unit_scale, cfg, stage):
    """Linear normal/landmark constraints in the shared coarse XYZ basis."""
    p = surface.positions
    faces = surface.faces
    n = len(p)
    ids = np.flatnonzero(confidence >= cfg['minimum_confidence'])
    tri = faces[ids]
    normals = target_normals[ids]
    blocks, rhs = [], []
    # n_ref dot (P_b + D_b - P_a - D_a) = 0, for both surface tangents.
    for corner in (1, 2):
        edge = p[tri[:, corner]] - p[tri[:, 0]]
        factor = np.sqrt(confidence[ids])/np.maximum(np.linalg.norm(edge, axis=1), .1)
        vals = np.concatenate([-normals, normals], axis=1)*factor[:, None]
        cols = np.concatenate([tri[:, :1]*3+np.arange(3),
                               tri[:, corner:corner+1]*3+np.arange(3)], axis=1)
        block = sparse.coo_matrix((vals.ravel(),
            (np.repeat(np.arange(len(ids)), 6), cols.ravel())), shape=(len(ids), n*3)).tocsr()
        blocks.append(block); rhs.append(-np.sum(normals*edge, axis=1)*factor)
    points = np.sum(p[indices]*bary[..., None], axis=1)
    predicted, _ = camera.project(points/unit_scale)
    landmark_weight = np.full(68, cfg['landmark_weight'], dtype=np.float64)
    landmark_weight[:17] = cfg['contour_weight']
    landmark_weight[17:27] *= .3
    landmark_weight[60:] *= .25
    # Normalize by physical face scale, not arbitrary OBJ units.
    factor = np.sqrt(landmark_weight)/4.
    for axis, sign in ((0, 1.), (1, -1.)):
        vals = bary[..., None]*camera.rotation[axis]*factor[:, None, None]
        cols = indices[..., None]*3+np.arange(3)
        blocks.append(sparse.coo_matrix((vals.ravel(),
            (np.repeat(np.arange(68), 9), cols.ravel())), shape=(68, 3*n)).tocsr())
        rhs.append((target[:, axis]-predicted[:, axis])*sign*unit_scale/camera.scale*factor)
    inherited = (p-surface.rest).ravel()
    regularizer = sparse.kron(laplacian, sparse.eye(3), format='csr')*np.sqrt(cfg['smoothness'])
    screen = sparse.eye(3*n, format='csr')/cfg['screening_length_mm']
    blocks.extend([regularizer, screen])
    rhs.extend([-regularizer@inherited, -screen@inherited])
    full = sparse.vstack(blocks, format='csr')
    expansion = sparse.kron(basis, sparse.eye(3), format='csr')
    matrix = (full@expansion).tocsr()
    target_rhs = np.concatenate(rhs)
    fit = lsmr(matrix, target_rhs, atol=1e-6, btol=1e-6, maxiter=cfg['cg_maxiter'])
    if fit[1] not in (0, 1, 2, 4, 5) or not np.isfinite(fit[0]).all():
        raise RuntimeError(f'粗层连续向量场求解未收敛: {fit[1]}')
    delta = np.asarray(expansion@fit[0]).reshape(-1, 3)
    # One global attenuation preserves smoothness of the solved basis field.
    delta *= min(1., cfg['step_limits_mm'][stage]/max(float(np.linalg.norm(delta, axis=1).max()), 1e-12))
    old = p[faces]; orig = surface.rest[faces]
    old_cross = np.cross(old[:, 1]-old[:, 0], old[:, 2]-old[:, 0])
    orig_cross = np.cross(orig[:, 1]-orig[:, 0], orig[:, 2]-orig[:, 0])
    area = np.linalg.norm(old_cross, axis=1)
    check = area > 1e-10
    before = float(target_rhs@target_rhs)
    fraction = 1.
    for _ in range(20):
        q = p + delta*fraction
        tri = q[faces]
        cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        safe = np.all(np.sum(cross[check]*old_cross[check], axis=1) > 0)
        safe &= np.all(np.sum(cross[check]*orig_cross[check], axis=1) > 0)
        safe &= np.all(np.linalg.norm(cross[check], axis=1) >= .25*area[check])
        safe &= np.max(np.linalg.norm(q-surface.rest, axis=1)) <= cfg['maximum_displacement_mm']+1e-10
        error = full@(delta*fraction).ravel()-target_rhs
        energy = float(error@error)
        if safe and energy <= before+1e-10:
            break
        fraction *= .5
    else:
        fraction = 0.; energy = before
    # Surface.rest remains the original mesh for total-field regularization.
    final = p + delta*fraction
    return final, dict(solver_iterations=int(fit[2]), accepted_fraction=fraction,
        energy_before=before, energy_after=energy,
        max_increment_mm=float(np.linalg.norm(delta*fraction, axis=1).max()),
        max_total_mm=float(np.linalg.norm(final-surface.rest, axis=1).max()),
        landmark_rms_before_pixels=float(np.sqrt(np.mean((predicted-target)**2))),
        landmark_rms_after_pixels=float(np.sqrt(np.mean((camera.project(
            np.sum(final[indices]*bary[..., None], axis=1)/unit_scale)[0]-target)**2))))


def refine_structure(cfg, out):
    """Return a rebound config for detail extraction and the original/corrected mesh."""
    from run_displacement import digest, save_surface
    root = Path(__file__).resolve().parent
    opts = structure_settings(cfg.get('structure'))
    if len(cfg['references']) < 3:
        raise ValueError('88 大形精修需要至少三张同视角、不同光源方向的参考图')
    folder = out/'structure'; folder.mkdir(parents=True, exist_ok=True)
    inputs = folder/'inputs'; inputs.mkdir(exist_ok=True)
    request = dict(cfg, output=str(inputs))
    _write(inputs/'request.json', request)
    subprocess.run([sys.executable, '-u', str(root/'prepare_inputs.py'), '--config',
        str(inputs/'request.json'), '--bindings-only'], cwd=root, check=True,
        env=dict(os.environ, PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1'))
    prepared = _json(inputs/'prepared_config.json')
    obs = _json(prepared['calibration_observations'])
    records = _json(prepared['reference_detections'])
    mesh = load_obj(Path(cfg['mesh']))
    indices = np.asarray(obs['source_indices'], int); bary = np.asarray(obs['source_barycentric'], float)
    points = np.sum(mesh.vertices[indices]*bary[..., None], axis=1)
    disp = cfg['displacement']
    scale = disp['millimeters_per_unit']
    if scale is None:
        width = np.linalg.norm(points[16]-points[0])
        if width < 1e-9:
            raise ValueError('无法从脸宽估算模型单位')
        scale = disp['assumed_face_width_mm']/width
    size = np.asarray(records[0]['size'])
    work_size = np.maximum(1, np.rint(size*min(1., opts['analysis_max_side']/max(size))).astype(int))
    shape = tuple(work_size[::-1])
    target = (np.asarray(obs['target_original_pixels'])+.5)*work_size/size-.5
    camera, camera_report = calibrate_camera(points, target, prepared.get('source_rotation'))
    rgbs, registrations = aligned_references(records, shape)
    for reg in registrations:
        reg['per_landmark_residual'] = (np.asarray(reg['per_landmark_residual'])*work_size[1]/size[1]).tolist()
    directions = np.zeros_like(mesh.vertices)
    for k in range(3):
        np.add.at(directions, mesh.triangles[:, k], mesh.corner_normals[:, k])
    directions = unit(directions)
    initial = Surface(mesh.vertices*scale, directions, mesh.triangles.copy(),
        mesh.uv[np.maximum(mesh.triangle_uv, 0)], mesh.material_ids.copy(),
        np.arange(len(mesh.triangles), dtype=np.int32), np.zeros(len(mesh.vertices)),
        np.zeros(len(mesh.vertices), np.int16))
    save_surface(folder/'original', initial, scale, mesh.materials)
    positions = initial.rest.copy()
    materials = [mesh.materials.index(m) for m in prepared['skin_materials']]
    report = dict(enabled=True, cells=opts['cells'], stages=[], camera=camera_report,
        millimeters_per_unit=float(scale), correspondence='Original SOAP vertex IDs and barycentrics',
        contour_source='17 corresponding jaw-outline landmarks; no dense side-view silhouette supplied',
        photometry='Estimated global directional lights, robust shared-albedo normal fit; not measured normals')
    allowed = None
    for stage, cells in enumerate(opts['cells']):
        print(f'STRUCTURE_STAGE {cells} 格：脸型与五官连续向量场', flush=True)
        stage_start = positions.copy()
        current = replace(mesh, vertices=positions/scale)
        if allowed is None:
            raster = rasterize(current, camera, shape)
            visible = np.zeros(len(mesh.triangles), bool)
            visible[np.unique(raster['triangle'][raster['triangle'] >= 0])] = True
            projected = raster['screen_vertices'][mesh.triangles].mean(1)
            lo, hi = target.min(0), target.max(0)
            span = hi-lo
            allowed = visible & np.isin(mesh.material_ids, materials)
            allowed &= np.all((projected >= lo-span*[.2, .6]) & (projected <= hi+span*.2), axis=1)
            # Expand through actual adjacency for smooth boundary support.
            graph = adjacency(mesh)
            for _ in range(3):
                expanded = allowed.copy()
                for i in np.flatnonzero(allowed):
                    expanded[graph[i]] = True
                allowed = expanded & np.isin(mesh.material_ids, materials)
        basis, laplacian, stage_report = coarse_basis(mesh, allowed, cells)
        if not basis.shape[1]:
            raise ValueError('粗层可动区域过小，请检查模型对应点')
        triangle_area = np.linalg.norm(np.cross(initial.rest[mesh.triangles[:, 1]]-initial.rest[mesh.triangles[:, 0]],
            initial.rest[mesh.triangles[:, 2]]-initial.rest[mesh.triangles[:, 0]]), axis=1)*.5
        poly_area = np.bincount(mesh.polygon_ids, weights=triangle_area)
        cell_width = np.sqrt(np.median(poly_area[np.unique(mesh.polygon_ids[allowed])])*cells)
        sigma = max(1., .5*cell_width*camera.scale/scale)
        stage_report.update(iterations=[], support_width_mm=float(cell_width))
        for iteration in range(opts['iterations_per_stage']):
            # Rotate custom normals by actual geometric deformation.
            vector = positions-initial.rest
            length = np.linalg.norm(vector, axis=1)
            a = smooth_normals(initial.rest, initial.faces); b = smooth_normals(positions, initial.faces)
            cross = np.cross(a, b); dot = np.sum(a*b, axis=1)
            rotated = unit(directions + np.cross(cross, directions)
                + np.cross(cross, np.cross(cross, directions))/np.maximum(1+dot[:, None], 1e-6))
            current = replace(mesh, vertices=positions/scale, corner_normals=rotated[mesh.triangles])
            normals, confidence, raster, photo = normal_constraints(current, camera, shape,
                target, rgbs, registrations, materials, sigma, opts)
            # Represent the current vector deformation exactly for this solve.
            state = replace(initial, directions=unit(vector), height=length)
            positions, result = solve_vectors(state, basis, laplacian, normals, confidence,
                indices, bary, target, camera, scale, opts, stage)
            result.update(iteration=iteration+1, observations=photo)
            stage_report['iterations'].append(result)
            print('STRUCTURE_RESULT', cells, json.dumps(result, ensure_ascii=False), flush=True)
            if result['max_increment_mm'] < 1e-5:
                break
        vector = positions-initial.rest
        length = np.linalg.norm(vector, axis=1)
        export = replace(initial, directions=unit(vector), height=length,
                         source_rest=initial.rest, source_directions=directions)
        # save_surface uses original custom normals + true geometry rotation.
        export.directions[length < 1e-12] = directions[length < 1e-12]
        save_surface(folder/f'coarse_{cells:02d}', export, scale, mesh.materials,
            vector_delta=positions-stage_start, normal_reference=directions)
        report['stages'].append(stage_report)
        _write(folder/'report.json', report)
    corrected = folder/'coarse_04.obj'
    # Keep SOAP correspondences. Only the mesh hash changes after deformation.
    obs['mesh_sha256'] = digest(corrected)
    obs['provenance'] = str(obs.get('provenance', ''))+'; document88 preserved bindings after structure fitting'
    _write(folder/'calibration_observations.json', obs)
    prepared.update(mesh=str(corrected), calibration_observations=str(folder/'calibration_observations.json'),
        output=str(out/'evidence_full'), evidence_only=True,
        displacement=dict(disp, millimeters_per_unit=float(scale)))
    report.update(corrected_mesh=str(corrected), maximum_displacement_mm=float(np.linalg.norm(vector, axis=1).max()))
    _write(folder/'report.json', report)
    return prepared, report, mesh, initial.rest, directions
