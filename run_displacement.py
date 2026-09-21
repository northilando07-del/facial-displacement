"""FULL evidence -> real residual mesh hierarchy -> displacement and normal bake."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from normal_geometry import load_obj
from displacement_geometry import (Surface, Evidence, unit, settings, solve_level,
    split_surface, choose_refinement, deformed_normals, differential, face_gradient, edges_of)

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def evidence_matches(directory, cfg):
    directory = Path(directory)
    required = ('report.json', 'run_config.json', 'virtual_cells.npz', 'screen_fields.npz', 'camera.json')
    if not all((directory / name).is_file() for name in required):
        return False
    report = read_json(directory / 'report.json')
    old = read_json(directory / 'run_config.json')
    if report['mesh_sha256'] != digest(cfg['mesh']):
        return False
    if [digest(p) for p in cfg['references']] != [v['sha256'] for v in report['views']]:
        return False
    if report.get('geometry', {}).get('enabled'):
        return False
    for key in ('geometry', 'reference_fusion', 'semantic_layers'):
        if cfg.get(key, {}).get('enabled') or old.get(key, {}).get('enabled'):
            return False
    for key in ('layers', 'solver', 'continuity', 'protection', 'virtual_surface_samples',
                'max_analysis_side'):
        if cfg.get(key) != old.get(key):
            return False
    return True


def ensure_evidence(cfg, out, rebuild=False, prepared=False):
    for key in ('geometry', 'reference_fusion', 'semantic_layers'):
        if cfg.get(key, {}).get('enabled'):
            raise ValueError('Document 77 requires FULL evidence; disable ' + key + '.enabled.')
    candidates = [out / 'evidence_full']
    if cfg.get('reuse_full_evidence'):
        candidates.append(Path(cfg['reuse_full_evidence']))
    if not rebuild:
        for candidate in candidates:
            if evidence_matches(candidate, cfg):
                print('Reusing content-checked FULL evidence:', candidate, flush=True)
                return candidate, 'Cached FULL extraction; all new geometry and baking recomputed.'
    evidence = out / 'evidence_full'
    normal_cfg = dict(cfg)
    normal_cfg['output'] = str(evidence)
    normal_cfg['cache_directory'] = str(ROOT / 'cache' / 'full')
    normal_cfg['evidence_only'] = True
    if not prepared:
        for key in ('reference_detections', 'calibration_observations', 'source_rotation', 'skin_materials'):
            normal_cfg.pop(key, None)
    for key in ('geometry', 'semantic_layers', 'reference_fusion'):
        if normal_cfg.get(key, {}).get('enabled'):
            raise ValueError('Document 77 uses FULL extraction; disable ' + key + '.enabled.')
    path = out / 'full_input_config.json'
    write_json(path, normal_cfg)
    script = 'run_layered.py' if prepared else 'prepare_inputs.py'
    subprocess.run([sys.executable, '-u', str(ROOT / script), '--config', str(path)],
        cwd=ROOT, check=True, env=dict(os.environ, PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1'))
    if not evidence_matches(evidence, cfg):
        raise RuntimeError('New FULL evidence failed input validation.')
    return evidence, ('Fresh residual extraction on corrected geometry; preserved SOAP bindings.'
        if prepared else 'Fresh automatic bindings and FULL extraction.')


def chain_support(directory, report, xy):
    graph_path = directory / 'structure_mid_00.json'
    if not graph_path.exists():
        return np.zeros(len(xy), np.float32), dict(chains=0, cross_region_edges=0)
    graph = read_json(graph_path)
    width, height = report['screen_size']
    sx, sy = np.array([width, height]) / report['native_reference_size']
    canvas = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    linked = {i for group in graph['chains'] for i in group}
    for edge in graph['edges']:
        if edge['a'] in linked and edge['b'] in linked:
            points = []
            for index in (edge['a'], edge['b']):
                x, y = graph['nodes'][index]['center']
                points.append(((x + .5) * sx - .5, (y + .5) * sy - .5))
            draw.line(points, fill=255, width=max(2, round(2 * sx)))
    support = gaussian_filter(np.asarray(canvas, np.float32) / 255, max(1., sx * 2))
    support = np.clip(support * 3, 0, 1)
    return support[xy[:, 1], xy[:, 0]], dict(graph['stats'],
        use='Preserved FULL directional/polarity/semantic chains prioritize subdivision; no added height gain.')


def load_evidence(directory, cfg):
    report = read_json(directory / 'report.json')
    mesh = load_obj(Path(cfg['mesh']))
    disp = settings(cfg.get('displacement'))
    scale = disp['millimeters_per_unit']
    if scale is None:
        obs = report['observations']
        points = np.sum(mesh.vertices[np.asarray(obs['source_indices'])]
                        * np.asarray(obs['source_barycentric'])[..., None], axis=1)
        width = float(np.linalg.norm(points[16] - points[0]))
        if width <= 1e-9:
            raise ValueError('Degenerate face-width calibration.')
        scale = disp['assumed_face_width_mm'] / width
        provenance = 'Estimated units: landmark 0-to-16 width assigned %.6g mm; not measured.' % disp['assumed_face_width_mm']
    else:
        provenance = 'User-supplied millimeters_per_unit.'
    directions = np.zeros_like(mesh.vertices)
    for k in range(3):
        np.add.at(directions, mesh.triangles[:, k], mesh.corner_normals[:, k])
    directions = unit(directions)
    surface = Surface(mesh.vertices * scale, directions, mesh.triangles.copy(),
        mesh.uv[np.maximum(mesh.triangle_uv, 0)], mesh.material_ids.copy(),
        np.arange(len(mesh.triangles), dtype=np.int32), np.zeros(len(mesh.vertices)),
        np.zeros(len(mesh.vertices), np.int16))
    with np.load(directory / 'virtual_cells.npz') as data:
        xy = data['pixel_xy']
        n0, n1 = unit(data['normal_base_camera']), unit(data['normal_refined_camera'])
        dot = np.maximum(np.sum(n0 * n1, axis=1), .2)
        slope = n1 / dot[:, None] - n0
        gradient = -slope @ np.asarray(report['camera']['rotation'])
        gradient *= disp['evidence_gain']
        chain, chains_report = chain_support(directory, report, xy)
        with np.load(directory / 'screen_fields.npz') as screen:
            domain = np.where(screen['lip_domain'][xy[:, 1], xy[:, 0]], 2, 1).astype(np.int8)
        evidence = Evidence(data['triangle_id'].astype(np.int32), data['barycentric'].copy(),
            gradient.astype(np.float32), data['confidence'].copy(), chain, domain)
    return surface, evidence, mesh, report, float(scale), provenance, chains_report


def save_surface(path, surface, scale, material_names, delta=None, parent=None,
                 vector_delta=None, normal_reference=None):
    normals = deformed_normals(surface, normal_reference)
    source_rest = surface.rest if surface.source_rest is None else surface.source_rest
    source_normals = surface.directions if surface.source_directions is None else surface.source_directions
    vector = surface.positions-source_rest
    arrays = dict(rest=source_rest / scale, detail_rest=surface.rest / scale,
        vertices=surface.positions / scale,
        vector_displacement_mm=vector.astype(np.float32),
        structure_displacement_mm=(surface.rest-source_rest).astype(np.float32),
        normals=normals.astype(np.float32), base_normals=source_normals.astype(np.float32),
        detail_base_normals=surface.directions.astype(np.float32),
        faces=surface.faces, corner_uv=surface.uv.astype(np.float32),
        material_ids=surface.materials, material_names=np.asarray(material_names),
        original_triangles=surface.origins, height_mm=surface.height.astype(np.float32),
        generation=surface.generation, millimeters_per_unit=scale)
    if delta is not None:
        arrays['delta_mm'] = delta.astype(np.float32)
    if vector_delta is not None:
        arrays['vector_delta_mm'] = vector_delta.astype(np.float32)
    if normal_reference is not None:
        arrays['height_mm'] = np.zeros(len(surface.rest), np.float32)
        arrays['detail_rest'] = surface.positions/scale
        arrays['detail_base_normals'] = normals.astype(np.float32)
        arrays['structure_displacement_mm'] = vector.astype(np.float32)
    if parent is not None:
        arrays['parent_faces'] = parent['parent_face']
        arrays['child_parent_barycentric'] = parent['child_bary']
        arrays['new_vertex_parent_edges'] = parent['new_edges']
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    with path.with_suffix('.obj').open('w', encoding='utf-8', newline='\n') as stream:
        stream.write('# Real displaced geometry, original world units; preserved UVs.\n')
        np.savetxt(stream, surface.positions / scale, fmt='v %.9g %.9g %.9g')
        np.savetxt(stream, surface.uv.reshape(-1, 2), fmt='vt %.9g %.9g')
        np.savetxt(stream, normals, fmt='vn %.9g %.9g %.9g')
        stream.write('s 1\n')
        material = -1
        for i, face in enumerate(surface.faces):
            mid = int(surface.materials[i])
            if mid != material:
                stream.write('usemtl ' + material_names[mid] + '\n'); material = mid
            stream.write('f ' + ' '.join(f'{int(v)+1}/{i*3+k+1}/{int(v)+1}' for k, v in enumerate(face)) + '\n')


def run(cfg, geometry_only=False, rebuild=False):
    start = time.perf_counter()
    size = cfg.get('bake_resolution', 8192)
    if type(size) is not int or not 16 <= size <= 8192:
        raise ValueError('bake_resolution must be an integer in 16..8192.')
    cfg['bake_resolution'] = size
    out = Path(cfg['output']).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Document 77 outputs must be inside the new workflow directory.')
    out.mkdir(parents=True, exist_ok=True)
    cfg['displacement'] = settings(cfg.get('displacement'))
    from displacement_structure import structure_settings, refine_structure
    cfg['structure'] = structure_settings(cfg.get('structure'))
    if cfg['structure']['enabled'] and len(cfg['references']) < 3:
        raise ValueError('88 大形精修需要至少三张同视角、不同光源的参考图')
    # Preserve previous new-workflow results and avoid stale higher levels
    # appearing when a later run selects fewer subdivision levels.
    if (out / 'report.json').exists():
        generated=('structure','levels','maps','layer_maps','scene','head_displaced.obj','head_displaced.npz',
                   'report.json','run_config.json','verification.json','displacement_maps.json',
                   'vector_displacement_maps.json')
        inputs=[Path(cfg['mesh']).resolve(),*[Path(p).resolve() for p in cfg['references']]]
        if any(p.is_relative_to((out/name).resolve()) for p in inputs for name in generated):
            raise ValueError('An input is inside this output set. Choose a separate output directory.')
        history=out/'run_history'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        history.mkdir(parents=True)
        for name in generated:
            if (out/name).exists():
                shutil.move(str(out/name),str(history/name))
    write_json(out / 'run_config.json', cfg)
    working_cfg = cfg
    structure_report = dict(enabled=False)
    if cfg['structure']['enabled']:
        working_cfg, structure_report, original_mesh, original_rest, original_normals = refine_structure(cfg, out)
        print('STRUCTURE_COMPLETE; reprojecting corrected mesh for 1-cell residual evidence', flush=True)
    directory, reuse = ensure_evidence(working_cfg, out, rebuild, prepared=cfg['structure']['enabled'])
    surface, ev, mesh, source, scale, scale_note, chains = load_evidence(directory, working_cfg)
    if cfg['structure']['enabled']:
        surface.source_rest = original_rest
        surface.source_directions = original_normals
        material_map = np.array([original_mesh.materials.index(name) for name in mesh.materials])
        surface.materials = material_map[surface.materials]
        mesh = original_mesh
        if cfg['displacement']['millimeters_per_unit'] is None:
            scale_note = 'Estimated from original SOAP landmark face width; scale fixed through all stages.'
    disp = cfg['displacement']
    if len(surface.rest) > disp['max_vertices'] or len(surface.faces) > disp['max_triangles']:
        raise ValueError('Input mesh already exceeds the configured geometry budget.')
    report = dict(version='document88-coarse-vector-residual-1' if cfg['structure']['enabled'] else 'document77-real-mesh-1', status='geometry_running',
        input_mesh=cfg['mesh'], mesh_sha256=mesh.sha256,
        references=[dict(path=v['path'], sha256=v['sha256']) for v in source['views']],
        evidence_directory=str(directory), evidence_reuse=reuse,
        evidence_sha256=digest(directory / 'virtual_cells.npz'),
        input_vertices=len(mesh.vertices), input_polygons=mesh.polygon_count,
        input_triangles=len(mesh.triangles), camera=source['camera'],
        screen_size=source['screen_size'], native_reference_size=source['native_reference_size'],
        source_normal_mean_degrees=source['combined_mean_active_angle_degrees'],
        millimeters_per_unit=scale, unit_provenance=scale_note,
        structure_continuity=chains, structure=structure_report, levels=[],
        workflow=['coarse_16', 'coarse_4', 'base_1_cell'] + [f'detail_{i}' for i in range(1, cfg['displacement']['levels']+1)]
            if cfg['structure']['enabled'] else ['base_1_cell'] + [f'detail_{i}' for i in range(1, cfg['displacement']['levels']+1)],
        requested_subdivision_levels=disp['levels'], bake_resolution=size,
        assumptions=['FULL normals are candidate evidence, not geometric ground truth.',
            'Small normal displacements use a linear surface-gradient approximation.',
            'Linear adaptive triangle subdivision retains parent vertices; it is not Catmull-Clark.',
            'Local face-flip checks do not certify absence of all global self-intersections.'])
    level_dir = out / 'levels'; level_dir.mkdir(exist_ok=True)
    parent = None
    for level in range(disp['levels'] + 1):
        print('Solving mesh level', level, 'vertices', len(surface.rest), 'triangles', len(surface.faces), flush=True)
        result, stats, residual, delta = solve_level(surface, ev, level, disp)
        if parent is not None:
            result['subdivision'] = {k: v for k, v in parent.items() if not isinstance(v, np.ndarray)}
        save_surface(level_dir / f'level_{level:02d}', surface, scale, mesh.materials, delta, parent)
        report['levels'].append(result)
        print('LEVEL_RESULT', json.dumps(result), flush=True)
        write_json(out / 'report.json', report)
        if level == disp['levels']:
            break
        selected = choose_refinement(surface, stats, residual, disp, source['camera']['scale'] / scale)
        if not selected.any():
            report['adaptive_stop'] = 'No supported residual exceeds subdivision criteria/budget.'
            break
        surface, ev, parent = split_surface(surface, ev, selected, level + 1)
    save_surface(out / 'head_displaced', surface, scale, mesh.materials)
    edges, _, counts = edges_of(surface.faces)
    source_edges, _, source_counts = edges_of(mesh.triangles)
    changed = np.linalg.norm(surface.positions[:len(mesh.vertices)] / scale - mesh.vertices, axis=1)
    normals = deformed_normals(surface)
    original_normals = surface.directions if surface.source_directions is None else surface.source_directions
    original_rest = surface.rest if surface.source_rest is None else surface.source_rest
    magnitude = np.linalg.norm(surface.positions-original_rest, axis=1)
    moved = magnitude > 1e-8
    angle = np.degrees(np.arccos(np.clip(np.sum(normals * original_normals, axis=1), -1, 1)))
    report.update(status='geometry_complete', final_vertices=len(surface.rest),
        completed_subdivision_levels=report['levels'][-1]['level'],
        final_triangles=len(surface.faces), moved_base_vertices=int(np.count_nonzero(changed > 1e-8)),
        moved_final_vertices=int(moved.sum()), max_displacement_mm=float(np.max(magnitude)),
        max_detail_displacement_mm=float(np.max(np.abs(surface.height))),
        displacement_p95_mm=float(np.quantile(magnitude[moved], .95)) if moved.any() else 0.,
        geometric_normal_change_mean_degrees=float(angle[moved].mean()) if moved.any() else 0.,
        geometric_normal_change_max_degrees=float(angle[moved].max()) if moved.any() else 0.,
        original_nonmanifold_edges=int((source_counts > 2).sum()),
        final_nonmanifold_edges=int((counts > 2).sum()),
        source_mesh_unchanged=digest(cfg['mesh']) == mesh.sha256,
        references_unchanged=all(digest(v['path']) == v['sha256'] for v in source['views']),
        evidence_unchanged=digest(directory / 'virtual_cells.npz') == report['evidence_sha256'],
        geometry_seconds=time.perf_counter() - start,
        final_model=str(out / 'head_displaced.obj'))
    write_json(out / 'report.json', report)
    if not geometry_only:
        from export_displacement_maps import run as export_maps
        export_maps(out)
        blender = Path(cfg['blender'])
        if not blender.is_file():
            raise FileNotFoundError('Blender executable unavailable: ' + str(blender))
        subprocess.run([str(blender), '-b', '-t', '8', '--python-exit-code', '10',
            '--python', str(ROOT / 'render_displacement.py'), '--', '--output', str(out)], cwd=ROOT, check=True)
        report=read_json(out/'report.json')
        from run_refine import montage
        states = [('base', 'Original'), ('structure', 'Structure 16 / 4'),
                  ('geometry', 'Final detail geometry'), ('baked', 'Original + baked normal')]
        if not cfg['structure']['enabled']:
            states = [pair for pair in states if pair[0] != 'structure']
        for view in ('left', 'right', 'oblique'):
            panels = []
            for key, title in states:
                with Image.open(out/'scene'/f'{view}_{key}.png') as image:
                    panels.append((title, image.convert('RGB')))
            montage(out/'scene'/f'{view}_comparison.png', panels, columns=len(panels))
        report['status'] = 'complete'
        report['validation'] = 'Not automatically tested; inspect the models, maps and UI results.'
        write_json(out/'report.json', report)
    print('DISPLACEMENT_COMPLETE', str(out), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'displacement.json')
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--rebuild-evidence', action='store_true')
    args = parser.parse_args()
    run(read_json(args.config), args.geometry_only, args.rebuild_evidence)


if __name__ == '__main__':
    main()
