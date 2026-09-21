"""Rasterize actual mesh displacement to preserved per-corner UVs."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from normal_geometry import triangle_pixels


def bake(source, output, size, limit, key='height_mm', material_names=None):
    with np.load(source) as data:
        faces=data['faces']; uv=data['corner_uv']; mid=data['material_ids']
        names=data['material_names'].tolist(); height=data[key]
        unit=float(data['millimeters_per_unit'])
    outputs=[]
    for material in material_names or names:
        if material not in names:
            continue
        ids=np.flatnonzero((mid==names.index(material)) & (np.max(np.abs(height[faces]),axis=1)>1e-9))
        field=np.zeros((size,size),np.float32)
        covered=np.zeros((size,size),bool)
        for i in ids:
            coords=uv[i]*[size,-size]+[0,size]
            hit=triangle_pixels(coords,size,size)
            if hit is None:continue
            yy,xx,bary=hit
            field[yy,xx]=bary@height[faces[i]]
            covered[yy,xx]=True
        directory=output/material;directory.mkdir(parents=True,exist_ok=True)
        np.save(directory/'displacement_signed_mm.npy',field)
        # Quantize in float64 to keep rounding within half a 16-bit step.
        encoded=np.rint(np.clip(.5+field.astype(np.float64)/(2*limit),0,1)*65535).astype(np.uint16)
        Image.fromarray(encoded).save(directory/'displacement_16.png')
        Image.fromarray(np.rint(np.clip(.5+field/(2*limit),0,1)*255).astype(np.uint8)).save(directory/'displacement_preview.png')
        Image.fromarray((covered*255).astype(np.uint8)).save(directory/'displaced_coverage.png')
        metadata=dict(material=material,source_mesh=str(source),field=key,
            resolution=size,encoded_range_mm=[-limit,limit],midlevel=.5,
            decode='height_mm = (pixel_normalized - 0.5) * (2 * range_mm)',
            range_mm=limit,millimeters_per_unit=unit,
            displacement_node_scale_original_units=2*limit/unit,
            observed_min_mm=float(field.min()),observed_max_mm=float(field.max()),
            changed_texels=int((np.abs(field)>1e-7).sum()),
            note='Signed DETAIL normal displacement relative to the corrected base, if structure is enabled. '
                 'Full XYZ deformation is in vector_displacement_mm.npy / vector_displacement_16.png. Use Non-Color.')
        (directory/'displacement_decode.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        outputs.append(metadata)
    return outputs


def bake_vector(source, output, size, limit, material_names, key='vector_displacement_mm'):
    """Bake signed object-space XYZ with bounded row buffers for 8K output.

    Both NPY and PNG use the original OBJ axes. RGB is not a tangent normal;
    zero is (0.5, 0.5, 0.5), with a symmetric signed range per component.
    """
    import struct
    import zlib
    with np.load(source) as data:
        faces, uv, mid = data['faces'], data['corner_uv'], data['material_ids']
        names, vectors = data['material_names'].tolist(), data[key]
        scale = float(data['millimeters_per_unit'])
    records = []
    for name in material_names:
        if name not in names:
            continue
        folder = output/name; folder.mkdir(parents=True, exist_ok=True)
        ids = np.flatnonzero((mid == names.index(name)) &
            (np.max(np.abs(vectors[faces]), axis=(1, 2)) > 1e-10))
        field = np.lib.format.open_memmap(folder/'vector_displacement_mm.npy', mode='w+',
            dtype=np.float32, shape=(size, size, 3))
        field[:] = 0
        for i in ids:
            hit = triangle_pixels(uv[i]*[size, -size]+[0, size], size, size)
            if hit is None:
                continue
            yy, xx, bary = hit
            field[yy, xx] = bary@vectors[faces[i]]
        field.flush()
        def chunk(kind, payload):
            return struct.pack('>I', len(payload))+kind+payload+struct.pack('>I', zlib.crc32(kind+payload)&0xffffffff)
        maximum = 0.
        compressor = zlib.compressobj(6)
        with (folder/'vector_displacement_16.png').open('wb') as stream:
            stream.write(b'\x89PNG\r\n\x1a\n')
            stream.write(chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 16, 2, 0, 0, 0)))
            for row in field:
                maximum = max(maximum, float(np.max(np.abs(row))))
                encoded = np.rint(np.clip(.5+row.astype(np.float64)/(2*limit), 0, 1)*65535).astype('>u2')
                payload = compressor.compress(b'\0'+encoded.tobytes())
                if payload:
                    stream.write(chunk(b'IDAT', payload))
            stream.write(chunk(b'IDAT', compressor.flush())); stream.write(chunk(b'IEND', b''))
        step = max(1, size//1024)
        preview = np.asarray(field[::step, ::step])
        Image.fromarray(np.rint(np.clip(.5+preview/(2*limit), 0, 1)*255).astype(np.uint8)).save(folder/'vector_displacement_preview.png')
        del preview, field
        record = dict(material=name, source_mesh=str(source), field=key, resolution=size,
            space='OBJECT: original OBJ axes, before any scene rotation', channels=['X', 'Y', 'Z'],
            units='millimeters', color_space='Non-Color / Raw', midlevel=[.5, .5, .5], range_mm=limit,
            decode='D_original_units = (RGB - 0.5) * (2 * range_mm) / millimeters_per_unit',
            millimeters_per_unit=scale, observed_max_component_mm=maximum,
            reconstruction='Original linearly subdivided mesh + decoded XYZ; use matching UVs. '
                           'Do not add the total vector map to an already corrected model.')
        (folder/'vector_displacement_decode.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        records.append(record)
    return records


def run(output):
    output=Path(output)
    cfg=json.loads((output/'run_config.json').read_text(encoding='utf-8-sig'))
    source=json.loads((Path(json.loads((output/'report.json').read_text(encoding='utf-8'))['evidence_directory'])/'run_config.json').read_text(encoding='utf-8-sig'))
    names=source['skin_materials'];size=int(cfg.get('bake_resolution',8192))
    limit=float(cfg['displacement']['maximum_displacement_mm'])
    records=bake(output/'head_displaced.npz',output/'maps',size,limit,material_names=names)
    print('DISPLACEMENT_MAP_COMPLETE total',size,flush=True)
    for path in sorted((output/'levels').glob('level_*.npz')):
        records.extend(bake(path,output/'layer_maps'/path.stem,size,limit,key='delta_mm',material_names=names))
        print('DISPLACEMENT_MAP_COMPLETE',path.stem,size,flush=True)
    (output/'displacement_maps.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    if cfg.get('structure', {}).get('enabled', False):
        vector_limit = limit + cfg['structure']['maximum_displacement_mm']
        vectors = bake_vector(output/'head_displaced.npz', output/'maps', size, vector_limit, names)
        print('VECTOR_DISPLACEMENT_MAP_COMPLETE total', size, flush=True)
        for path in sorted((output/'structure').glob('coarse_*.npz'), reverse=True):
            vectors.extend(bake_vector(path, output/'layer_maps'/path.stem, size,
                vector_limit, names, key='vector_delta_mm'))
            print('VECTOR_DISPLACEMENT_MAP_COMPLETE', path.stem, size, flush=True)
        (output/'vector_displacement_maps.json').write_text(json.dumps(vectors, indent=2), encoding='utf-8')
    print('DISPLACEMENT_MAPS',len(records),'actual-geometry scalar maps',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
