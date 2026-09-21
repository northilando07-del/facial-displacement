"""OBJ-preserving geometry, orthographic projection and CPU rasterization."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@dataclass
class Mesh:
    vertices: np.ndarray
    uv: np.ndarray
    triangles: np.ndarray
    triangle_uv: np.ndarray
    corner_normals: np.ndarray
    polygon_ids: np.ndarray
    material_ids: np.ndarray
    materials: list[str]
    polygon_count: int
    sha256: str


def load_obj(path: Path) -> Mesh:
    """Fan-triangulate for calculation only; never rewrite the source OBJ."""
    raw = path.read_bytes()
    v, uv, vn, faces, fuv, fn, polygons, mats = [], [], [], [], [], [], [], []
    names = ['default']
    material = 0
    poly = 0
    for line in raw.decode('utf-8-sig', errors='replace').splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v':
            v.append([float(x) for x in parts[1:4]])
        elif parts[0] == 'vt':
            uv.append([float(x) for x in parts[1:3]])
        elif parts[0] == 'vn':
            vn.append([float(x) for x in parts[1:4]])
        elif parts[0] == 'usemtl':
            name = ' '.join(parts[1:])
            if name not in names:
                names.append(name)
            material = names.index(name)
        elif parts[0] == 'f':
            corners = []
            for token in parts[1:]:
                values = token.split('/')
                ids = []
                for k, count in enumerate((len(v), len(uv), len(vn))):
                    a = int(values[k]) if k < len(values) and values[k] else 0
                    ids.append(a - 1 if a > 0 else count + a if a < 0 else -1)
                corners.append(ids)
            for i in range(1, len(corners) - 1):
                tri = np.array([corners[0], corners[i], corners[i + 1]])
                faces.append(tri[:, 0]); fuv.append(tri[:, 1]); fn.append(tri[:, 2])
                polygons.append(poly); mats.append(material)
            poly += 1
    vertices = np.asarray(v, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int32)
    if not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError('OBJ has no triangles or has nonfinite vertices.')
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError('Invalid OBJ vertex indices.')
    t = vertices[triangles]
    cross = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    normal = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normal, triangles[:, k], cross)
    cn = unit(normal)[triangles]
    nids = np.asarray(fn)
    if vn:
        supplied = unit(np.asarray(vn))
        valid = nids >= 0
        cn[valid] = supplied[nids[valid]]
    return Mesh(vertices, np.asarray(uv, dtype=float).reshape(-1, 2), triangles,
                np.asarray(fuv, dtype=np.int32), cn, np.asarray(polygons),
                np.asarray(mats), names, poly, hashlib.sha256(raw).hexdigest())


@dataclass
class Camera:
    center: np.ndarray
    rotation: np.ndarray
    scale: float
    translation: np.ndarray

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = (points - self.center) @ self.rotation.T
        return p[:, :2] * [self.scale, -self.scale] + self.translation, p[:, 2]


def calibrate_camera(points: np.ndarray, target: np.ndarray, source_rotation=None) -> tuple[Camera, dict]:
    """Fit one rigid pose and one isotropic pixel scale; no shape warping."""
    if source_rotation is not None:
        frame=np.asarray(source_rotation,dtype=float)
        camera,report=calibrate_camera(points@frame.T,target)
        camera.center=camera.center@frame
        camera.rotation=camera.rotation@frame
        report.update(center=camera.center.tolist(),rotation=camera.rotation.tolist(),source_rotation=frame.tolist())
        return camera,report
    if points.shape != (68, 3) or target.shape != (68, 2):
        raise ValueError('Calibration needs corresponding 68-point observations.')
    center = points.mean(0)
    p3 = points - center
    weights = np.ones(68)
    weights[:17] = .3
    weights[17:27] = .12
    weights[60:] = .12
    weights[[30, 33, 36, 39, 42, 45, 48, 54]] = 2
    best = None
    for yaw in (0., -.25, .25):
        rot = Rotation.from_euler('xyz', [0, yaw, 0]).as_matrix()
        xy = (p3 @ rot.T)[:, :2] * [1, -1]
        a = np.zeros((136, 3)); a[::2, 0] = xy[:, 0]; a[1::2, 0] = xy[:, 1]
        a[::2, 1] = 1; a[1::2, 2] = 1
        w = np.repeat(np.sqrt(weights), 2)
        s, tx, ty = np.linalg.lstsq(a * w[:, None], target.ravel() * w, rcond=None)[0]
        initial = np.array([0, yaw, 0, np.log(max(s, 1e-3)), tx, ty])
        def residual(q):
            r = Rotation.from_euler('xyz', q[:3]).as_matrix()
            pred = (p3 @ r.T)[:, :2] * [1, -1] * np.exp(q[3]) + q[4:6]
            return ((pred - target) * np.sqrt(weights[:, None])).ravel()
        result = least_squares(residual, initial, loss='soft_l1', f_scale=3,
                               max_nfev=250, bounds=([-.65, -1.2, -.5, -8, -10000, -10000],
                                                    [.65, 1.2, .5, 8, 10000, 10000]))
        if best is None or result.cost < best.cost:
            best = result
    q = best.x
    camera = Camera(center, Rotation.from_euler('xyz', q[:3]).as_matrix(),
                    float(np.exp(q[3])), q[4:6])
    projected, _ = camera.project(points)
    errors = np.linalg.norm(projected - target, axis=1)
    return camera, dict(mean_pixels=float(errors.mean()), p95_pixels=float(np.quantile(errors, .95)),
                        max_pixels=float(errors.max()), key_mean_pixels=float(errors[[30,36,39,42,45,48,54]].mean()),
                        errors_pixels=errors.tolist(), projected=projected.tolist(),
                        target=target.tolist(), center=center.tolist(), rotation=camera.rotation.tolist(),
                        scale=camera.scale, translation=camera.translation.tolist(),
                        optimizer_converged=bool(best.success), model='rigid orthographic, isotropic scale')


def triangle_pixels(p: np.ndarray, width: int, height: int):
    lo = np.maximum(np.floor(p.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(p.max(0)).astype(int), [width - 1, height - 1])
    if np.any(hi < lo):
        return None
    x, y = np.meshgrid(np.arange(lo[0], hi[0] + 1) + .5, np.arange(lo[1], hi[1] + 1) + .5)
    a, b, c = p
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(den) < 1e-10:
        return None
    w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / den
    w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / den
    w2 = 1 - w0 - w1
    inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
    yy, xx = np.nonzero(inside)
    if not len(xx):
        return None
    bary = np.column_stack([w0[inside], w1[inside], w2[inside]])
    return yy + lo[1], xx + lo[0], bary


def rasterize(mesh: Mesh, camera: Camera, shape: tuple[int, int], skip_materials=()) -> dict:
    h, w = shape
    screen, z = camera.project(mesh.vertices)
    depth = np.full((h, w), -np.inf, dtype=np.float32)
    tid = np.full((h, w), -1, dtype=np.int32)
    barycentric = np.zeros((h, w, 3), dtype=np.float32)
    normal = np.zeros((h, w, 3), dtype=np.float32)
    normal[..., 2] = 1
    cn = mesh.corner_normals @ camera.rotation.T
    for i, face in enumerate(mesh.triangles):
        if mesh.materials[mesh.material_ids[i]] in skip_materials:
            continue
        hit = triangle_pixels(screen[face], w, h)
        if hit is None:
            continue
        yy, xx, bary = hit
        zz = bary @ z[face]
        keep = zz > depth[yy, xx]
        yy, xx, bary, zz = yy[keep], xx[keep], bary[keep], zz[keep]
        depth[yy, xx] = zz
        tid[yy, xx] = i
        barycentric[yy, xx] = bary
        normal[yy, xx] = unit(bary @ cn[i])
    return dict(triangle=tid, barycentric=barycentric, normal=normal, depth=depth,
                screen_vertices=screen, vertex_depth=z)


def adjacency(mesh: Mesh) -> list[list[int]]:
    edges = defaultdict(list)
    result = [[] for _ in mesh.triangles]
    for i, f in enumerate(mesh.triangles):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            edges[(min(a, b), max(a, b))].append(i)
    for neighbors in edges.values():
        if len(neighbors) == 2:
            a, b = neighbors
            if mesh.material_ids[a] == mesh.material_ids[b]:
                result[a].append(b); result[b].append(a)
    return result


def group_patches(mesh: Mesh, raster: dict, eligible: np.ndarray,
                  max_angle: float = 18, radius_pixels: float = 45) -> tuple[np.ndarray, list[list[int]]]:
    """Connected mesh patches, constrained against the seed normal (no drift)."""
    graph = adjacency(mesh)
    tid = raster['triangle']
    counts = np.bincount(tid[eligible], minlength=len(mesh.triangles))
    normals = unit(mesh.corner_normals.mean(1))
    centers = raster['screen_vertices'][mesh.triangles].mean(1)
    labels = np.full(len(mesh.triangles), -1, dtype=np.int32)
    threshold = np.cos(np.deg2rad(max_angle))
    regions = []
    for seed in np.argsort(-counts):
        if not counts[seed] or labels[seed] >= 0:
            continue
        label = len(regions); labels[seed] = label
        queue = deque([int(seed)]); members = []
        while queue:
            i = queue.popleft(); members.append(i)
            for j in graph[i]:
                if labels[j] >= 0 or not counts[j] or len(members) + len(queue) >= 80:
                    continue
                if normals[j] @ normals[seed] < threshold:
                    continue
                if np.linalg.norm(centers[j] - centers[seed]) > radius_pixels:
                    continue
                labels[j] = label; queue.append(j)
        regions.append(members)
    image_labels = np.full_like(tid, -1)
    visible = tid >= 0
    image_labels[visible] = labels[tid[visible]]
    return image_labels, regions


def corner_tangents(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted UV seam-aware tangents. Not claimed to be MikkTSpace."""
    ids = mesh.triangle_uv
    valid = np.all(ids >= 0, axis=1)
    p = mesh.vertices[mesh.triangles]
    u = mesh.uv[np.maximum(ids, 0)]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    valid &= np.abs(det) > 1e-12
    denom = np.where(valid, det, 1)
    t = (e1 * d2[:, 1, None] - e2 * d1[:, 1, None]) / denom[:, None]
    b = (-e1 * d2[:, 0, None] + e2 * d1[:, 0, None]) / denom[:, None]
    area = np.linalg.norm(np.cross(e1, e2), axis=1)
    t *= (area * valid)[:, None]; b *= (area * valid)[:, None]
    # Normal values in the key preserve explicitly split OBJ normals.
    keys = np.column_stack([mesh.triangles.ravel(), ids.ravel(),
                            np.repeat(mesh.material_ids, 3),
                            np.round(mesh.corner_normals.reshape(-1, 3) * 1e6).astype(np.int64)])
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    ts = np.zeros((inv.max() + 1, 3)); bs = np.zeros_like(ts)
    np.add.at(ts, inv, np.repeat(t, 3, axis=0)); np.add.at(bs, inv, np.repeat(b, 3, axis=0))
    ts = ts[inv].reshape(-1, 3, 3); bs = bs[inv].reshape(-1, 3, 3)
    n = mesh.corner_normals
    ts = unit(ts - n * np.sum(n * ts, axis=-1, keepdims=True))
    bs = unit(np.cross(n, ts)) * np.where(np.sum(np.cross(n, ts) * bs, axis=-1, keepdims=True) < 0, -1, 1)
    return ts, bs
