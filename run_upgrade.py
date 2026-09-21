"""22.docx upgrade: native evidence, connected structure, robust multi-light fusion."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from normal_geometry import load_obj,calibrate_camera,rasterize,group_patches,unit
from normal_shading import fit_light,srgb_to_linear,linear_to_srgb,Settings,solve
from normal_structure import connect_structures,semantic_regions,direction_weights,smooth_along
from normal_multilight import (MultiSettings,blur,detail_evidence,similarity_registration,
    sample_image,solve_multilight,constrain_normals,angular,fit_metrics)
from run_refine import (save_json,image8,montage,observation_points,semantic_mask,
                       bake_uv,make_previews)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_work_size(source_size,surface_pixels,target,max_side=4096):
    if surface_pixels<=0:raise ValueError('No surface samples available.')
    factor=max(1.,np.sqrt(max(target,0)/surface_pixels))
    if max(source_size)>max_side:
        raise ValueError('max_analysis_side is smaller than the original image; refusing to downsample.')
    factor=min(factor,max_side/max(source_size))
    size=tuple(max(original,int(np.ceil(original*factor))) for original in source_size)
    return size


def alignment_confidence(shape,landmarks,errors,valid,tolerance=6.):
    result=np.ones(shape,np.float32)
    ys,xs=np.nonzero(valid)
    if not len(xs):return result
    tree=cKDTree(landmarks)
    for start in range(0,len(xs),100000):
        xx,yy=xs[start:start+100000],ys[start:start+100000]
        distances,ids=tree.query(np.column_stack([xx,yy]),k=3)
        weights=1/np.maximum(distances,3)**2
        local=np.sum(weights*np.asarray(errors)[ids],axis=1)/weights.sum(1)
        result[yy,xx]=np.exp(-.5*(local/tolerance)**2)
    return result


def run(cfg):
    start=time.perf_counter()
    out=Path(cfg['output']).resolve();out.mkdir(parents=True,exist_ok=True)
    records=json.loads(Path(cfg['reference_detections']).read_text(encoding='utf-8-sig'))
    if cfg.get('references'):
        lookup={str(Path(r['path']).resolve()):r for r in records}
        records=[lookup[str(Path(p).resolve())] for p in cfg['references']]
    records=records[:cfg.get('max_references',len(records))]
    if not records:raise ValueError('At least one reference is required.')
    for record in records:
        if digest(record['path'])!=record['sha256']:
            raise ValueError('Stale image landmark cache: '+record['path'])
    source=Path(cfg['mesh']);mesh=load_obj(source)
    reference_path=Path(records[0]['path'])
    anchor=ImageOps.exif_transpose(Image.open(reference_path)).convert('RGB')
    if list(anchor.size)!=records[0]['size']:raise ValueError('Landmark/image size mismatch.')
    observation_cfg=dict(cfg,reference=str(reference_path))
    points,target,provenance=observation_points(observation_cfg,mesh,anchor,out)
    camera_native,camera_native_report=calibrate_camera(points,target)
    print('Native reference size',anchor.size,'camera key error',camera_native_report['key_mean_pixels'],flush=True)
    skipped=[name for name in mesh.materials if any(s in name.lower() for s in ('eyebrow','eyelash','tear'))]
    native=rasterize(mesh,camera_native,(anchor.height,anchor.width),skipped)
    materials=cfg.get('skin_materials',['Genesis9SG5'])
    mids=[mesh.materials.index(name) for name in materials]
    tid=native['triangle'];n0=native['normal']
    geom=(tid>=0)&np.isin(mesh.material_ids[np.maximum(tid,0)],mids)
    anchor_rgb=np.asarray(anchor,dtype=np.float32)/255
    anchor_valid,anchor_masks=semantic_mask(anchor_rgb,target,geom,n0)
    domain=anchor_masks['anatomical']
    labels,regions=group_patches(mesh,native,domain,radius_pixels=anchor.height*.0586)
    semantic=semantic_regions(domain.shape,target,domain)
    image8(semantic/max(int(semantic.max()),1)).save(out/'semantic_regions.png')
    residuals=[];confidences=[];lights=[];view_reports=[];graphs=[]
    support=np.zeros(domain.shape,np.float32)
    direction_tensor=np.zeros((*domain.shape,3),np.float32)
    reference_panels=[];raw_residuals=[];raw_confidences=[]
    detail_first=None
    cache_dir=Path(cfg.get('cache_directory',out/'cache'));cache_dir.mkdir(parents=True,exist_ok=True)
    cache_key_data=dict(hashes=[r['sha256'] for r in records],mesh=mesh.sha256,
        continuity=cfg.get('continuity',True),analysis=cfg.get('analysis',{}),
        code=[digest(Path(__file__).with_name(name)) for name in
              ['normal_structure.py','normal_multilight.py','run_upgrade.py']])
    cache_key=hashlib.sha256(json.dumps(cache_key_data,sort_keys=True).encode()).hexdigest()
    cache_npz=cache_dir/(cache_key+'.npz');cache_json=cache_dir/(cache_key+'.json')
    if cache_npz.exists() and cache_json.exists():
        print('Reusing verified native evidence cache.',flush=True)
        cached=np.load(cache_npz)
        residuals=cached['residuals'];confidences=cached['confidences'];lights=cached['lights']
        raw_residuals=cached['raw_residuals'];raw_confidences=cached['raw_confidences']
        support=cached['support'];direction_tensor=cached['direction_tensor']
        metadata=json.loads(cache_json.read_text(encoding='utf-8'))
        view_reports=metadata['views'];graphs=metadata['graphs']
        detail_first=dict(coherence=cached['coherence'],direction=cached['direction'],noise=metadata['noise'])
    else:
        for index,record in enumerate(records):
            print(f'Analyzing original pixels: reference {index+1}/{len(records)}...',flush=True)
            image=ImageOps.exif_transpose(Image.open(record['path'])).convert('RGB')
            if list(image.size)!=record['size']:raise ValueError('Reference size differs from landmark cache.')
            matrix,offset,registration=similarity_registration(target,record['landmarks'])
            if index==0:
                matrix=np.eye(2);offset=np.zeros(2)
                registration.update(matrix=matrix.tolist(),offset=offset.tolist(),
                                    mean_landmark_residual=0.,p95_landmark_residual=0.,
                                    per_landmark_residual=[0.]*68,scale=1.,rotation_degrees=0.)
            if registration['p95_landmark_residual']>anchor.height*.06:
                raise ValueError('Reference pose/expression differs too much for same-camera fusion: '+record['path'])
            rgb=sample_image(np.asarray(image,np.float32)/255,domain.shape,matrix,offset,anchor.size)
            valid,masks=semantic_mask(rgb,target,geom,n0)
            valid&=domain
            if valid.sum()<1000:raise ValueError('Insufficient visible skin in reference '+record['path'])
            registration_weight=alignment_confidence(domain.shape,target,
                registration['per_landmark_residual'],domain)
            observed=(srgb_to_linear(rgb)@np.array([.2126,.7152,.0722],np.float32)).astype(np.float32)
            ambient,light,light_report=fit_light(n0,blur(observed,valid,2),valid)
            detail=detail_evidence(n0,observed,valid,labels,ambient,light,**cfg.get('analysis',{}))
            detail['confidence']*=registration_weight
            raw_residuals.append(detail['residual'].copy())
            raw_confidences.append(detail['confidence'].copy())
            if cfg.get('continuity',True):
                structure=connect_structures(detail['residual'],detail['confidence'],valid,
                    labels,semantic,detail['noise'],normals=n0)
                graph=structure['graph']
                detail['residual']=structure['residual'];detail['confidence']=structure['confidence']
                support=np.maximum(support,structure['support'])
                t=structure['tangent'];weight=structure['support']*structure['confidence']
                direction_tensor[...,0]+=weight*t[...,0]**2
                direction_tensor[...,1]+=weight*t[...,0]*t[...,1]
                direction_tensor[...,2]+=weight*t[...,1]**2
            else:
                graph=dict(nodes=[],edges=[],chains=[],stats=dict(nodes=0,edges=0,chains=0,cross_region_edges=0))
            graphs.append(graph)
            save_json(out/f'structure_graph_{index:02d}.json',graph)
            graph_image=image8(rgb);draw=ImageDraw.Draw(graph_image)
            for edge in graph['edges']:
                a,b=graph['nodes'][edge['a']],graph['nodes'][edge['b']]
                draw.line([tuple(a['center']),tuple(b['center'])],
                    fill=(255,70,20) if edge['cross_region'] else (50,220,100),width=2)
            graph_image.save(out/f'structure_links_{index:02d}.png')
            image8(detail['confidence']).save(out/f'observation_confidence_{index:02d}.png')
            reference_panels.append((f'Light {index+1}: registered source',image8(rgb)))
            residuals.append(detail['residual']);confidences.append(detail['confidence']);lights.append(light)
            view_reports.append(dict(path=record['path'],sha256=record['sha256'],original_size=list(image.size),
                registration=registration,lighting=light_report,noise=detail['noise'],
                eligible_pixels=int(valid.sum()),structure=graph['stats']))
            if index==0:detail_first=detail
            print('  structure',graph['stats'],'light',light_report['direction'],flush=True)
        residuals=np.asarray(residuals,np.float32);confidences=np.asarray(confidences,np.float32)
        lights=np.asarray(lights)
        raw_residuals=np.asarray(raw_residuals,np.float32);raw_confidences=np.asarray(raw_confidences,np.float32)
        np.savez(cache_npz,residuals=residuals,confidences=confidences,lights=lights,
            raw_residuals=raw_residuals,raw_confidences=raw_confidences,support=support,
            direction_tensor=direction_tensor,coherence=detail_first['coherence'],direction=detail_first['direction'])
        save_json(cache_json,dict(views=view_reports,graphs=graphs,noise=float(detail_first['noise'])))
        montage(out/'registered_references.png',reference_panels,2)
    for index,graph in enumerate(graphs):save_json(out/f'structure_graph_{index:02d}.json',graph)
    # The evidence has been analyzed without shrinking any reference image.
    # Subpixel sampling increases surface-normal density, NOT source information.
    work_size=choose_work_size(anchor.size,int(domain.sum()),cfg.get('virtual_surface_samples',2000000),
                               cfg.get('max_analysis_side',4096))
    work_shape=(work_size[1],work_size[0])
    print('Virtual surface grid',work_size,'target trusted-domain samples',cfg.get('virtual_surface_samples'),flush=True)
    ratio=np.asarray(work_size)/anchor.size
    target_work=(target+.5)*ratio-.5
    camera,camera_report=calibrate_camera(points,target_work)
    save_json(out/'camera.json',camera_report)
    raster=rasterize(mesh,camera,work_shape,skipped) if work_size!=anchor.size else native
    base=raster['normal'];tid=raster['triangle']
    high_geom=(tid>=0)&np.isin(mesh.material_ids[np.maximum(tid,0)],mids)
    valid=(sample_image(domain.astype(np.float32),work_shape)>.99)&high_geom
    projected_region=np.full(len(mesh.triangles),-1,np.int32)
    for i,region in enumerate(regions):projected_region[region]=i
    high_labels=projected_region[np.maximum(tid,0)];high_labels[tid<0]=-1
    high_r=np.empty((len(records),*work_shape),np.float32)
    high_c=np.empty_like(high_r)
    for index,light in enumerate(lights):
        # Sample filtered observed shading, then subtract the freshly rasterized
        # base. This avoids confusing interpolation of base normals with detail.
        target_shading=np.maximum(n0@light,0)+residuals[index]
        high_r[index]=sample_image(target_shading.astype(np.float32),work_shape)-np.maximum(base@light,0)
        high_c[index]=sample_image(confidences[index],work_shape)*valid
        high_r[index][~valid]=0
    settings=MultiSettings(**cfg.get('solver',{}))
    if len(records)>1:
        print('Robust multi-light solve...',flush=True)
        refined,confidence,extra=solve_multilight(base,high_r,high_c,lights,settings)
        # Smooth the shared vector update only along evidence-supported chains.
        phi=.5*np.arctan2(2*direction_tensor[...,1],direction_tensor[...,0]-direction_tensor[...,2])
        tangent_native=np.stack([np.cos(phi),np.sin(phi)],-1).astype(np.float32)
        tangent=unit(sample_image(tangent_native,work_shape))
        high_support=sample_image(support,work_shape)*valid
        wh,wv=direction_weights(tangent,high_support,confidence>.015,base)
        delta=refined/np.maximum(np.sum(base*refined,-1,keepdims=True),.2)-base
        candidate=unit(base+smooth_along(delta,wh,wv,iterations=4))
        refined=constrain_normals(base,candidate,confidence,settings.maximum_angle_degrees)
        stats=extra['stats']
        stats.update(fit_metrics(base,refined,high_r,high_c,lights))
        image8(extra['condition_ratio']).save(out/'light_condition.png')
        for i in range(len(records)):
            image8(extra['observation_weights'][i]).save(out/f'robust_weight_{i:02d}.png')
        # Diagnostic nuisance estimates stay in the data archive, not the normal map.
        np.savez_compressed(out/'photometric_diagnostics.npz',condition_ratio=extra['condition_ratio'],
                            albedo_delta=extra['albedo_delta'],observation_weights=extra['observation_weights'])
        del extra,delta,candidate,tangent,high_support,wh,wv
    else:
        direction=unit(sample_image(detail_first['direction'],work_shape))
        ys,xs=np.nonzero(valid)
        sl=np.s_[ys.min():ys.max()+1,xs.min():xs.max()+1]
        refined=base.copy();confidence=high_c[0].copy()
        refined[sl],stats=solve(base[sl],high_r[0][sl],confidence[sl],direction[sl],lights[0],
            Settings(maximum_angle_degrees=settings.maximum_angle_degrees,prior=settings.prior,iterations=100),raster['depth'][sl])
    angles=angular(base,refined);active=confidence>.015
    stats.update(normal_angle_max_degrees=float(angles.max()),active_virtual_cells=int(active.sum()),
        normal_angle_mean_active_degrees=float(angles[active].mean()) if active.any() else 0.,
        normal_angle_p95_active_degrees=float(np.quantile(angles[active],.95)) if active.any() else 0.)
    if not np.isfinite(refined).all() or angles.max()>settings.maximum_angle_degrees+1e-3:
        raise RuntimeError('Normal constraints failed.')
    if not np.array_equal(refined[~active],base[~active]):
        raise RuntimeError('Protected/unobserved base normals were modified.')
    print('Baking original UVs at',cfg.get('texture_resolution',4096),'pixels...',flush=True)
    maps=bake_uv(mesh,camera,raster,refined,confidence,materials,cfg.get('texture_resolution',4096),out)
    ys,xs=np.nonzero(active);face_ids=tid[active];bary=raster['barycentric'][active]
    uv=np.sum(mesh.uv[np.maximum(mesh.triangle_uv[face_ids],0)]*bary[...,None],axis=1)
    np.savez_compressed(out/'virtual_cells.npz',pixel_xy=np.column_stack([xs,ys]),triangle_id=face_ids,
        polygon_id=mesh.polygon_ids[face_ids],barycentric=bary,uv=uv,normal_base_camera=base[active],
        normal_refined_camera=refined[active],confidence=confidence[active],patch_id=high_labels[active])
    np.savez_compressed(out/'screen_fields.npz',normal_base=base,normal_refined=refined,
        residual=high_r[0],confidence=confidence,triangle=tid,labels=high_labels,valid=valid)
    preview_detail=dict(light=lights[0].tolist(),ambient=view_reports[0]['lighting']['ambient'],
        max_angle=settings.maximum_angle_degrees,residual=high_r[0],confidence=confidence,
        coherence=sample_image(detail_first['coherence'],work_shape))
    image=anchor.resize(work_size,Image.Resampling.BICUBIC)
    make_previews(out,image,raster,refined,preview_detail,valid,high_labels,camera_report)
    image8(linear_to_srgb(.08+.65*np.maximum(refined@unit(np.array([-.7,.3,.65])),0))).save(out/'relit_full_resolution.png')
    projected=np.sum(raster['screen_vertices'][mesh.triangles[face_ids]]*bary[...,None],axis=1)
    reprojection=np.linalg.norm(projected-np.column_stack([xs+.5,ys+.5]),axis=1)
    unchanged=digest(source)==mesh.sha256
    references_unchanged=all(digest(r['path'])==r['sha256'] for r in records)
    if not unchanged or not references_unchanged:raise RuntimeError('An input changed during the run.')
    report=dict(version='document-22-upgrade-2',input_mesh=str(source),input_reference=str(reference_path),
        mesh_sha256=mesh.sha256,source_mesh_unchanged=unchanged,references_unchanged=references_unchanged,
        original_topology_and_uv_unchanged=unchanged,input_vertices=len(mesh.vertices),
        input_polygons=mesh.polygon_count,computational_triangles=len(mesh.triangles),
        screen_size=list(work_size),native_reference_size=list(anchor.size),
        source_pixels_per_reference=int(anchor.width*anchor.height),
        native_analysis_downsampling=False,requested_virtual_surface_samples=cfg.get('virtual_surface_samples'),
        virtual_surface_samples=int(valid.sum()),active_virtual_cells=int(active.sum()),
        virtual_cell_definition='Subpixel footprint samples on original triangles; geometry/UV/topology unchanged. Supersampling adds no new source information.',
        evidence_analysis='All shading/structure analysis uses original reference resolution; high-density normal solve samples this continuous evidence.',
        valid_skin_pixels=int(valid.sum()),total_patches=len(regions),camera=camera_report,
        camera_native=camera_native_report,observations=provenance,views=view_reports,
        solver=stats,solver_settings=asdict(settings),maps=maps,
        virtual_cells_reprojection_max_pixels=float(reprojection.max()) if reprojection.size else 0.,
        elapsed_seconds=time.perf_counter()-start,
        limitations=['No real normal ground truth for supplied images; fitting error is not geometric accuracy.',
            'Effective lights are estimated from the base mesh. AI relighting may violate shared geometry/albedo.',
            'Landmark similarity registration and semantic regions are approximations; local changes can remain.',
            '16-bit UV maps use the existing seam-aware tangent basis, not certified MikkTSpace.',
            'No displacement, real mesh subdivision, backside reconstruction or metric albedo recovery.'])
    save_json(out/'report.json',report);save_json(out/'run_config.json',cfg)
    print(json.dumps(dict(output=str(out),seconds=report['elapsed_seconds'],surface_samples=int(valid.sum()),
                         active_samples=int(active.sum()),rmse_before=stats['weighted_detail_rmse_before'],
                         rmse_after=stats['weighted_detail_rmse_after']),ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('upgrade.json'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--native-only',action='store_true')
    parser.add_argument('--without-continuity',action='store_true')
    parser.add_argument('--max-references',type=int)
    parser.add_argument('--texture-resolution',type=int)
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.output:cfg['output']=str(args.output)
    if args.native_only:cfg['virtual_surface_samples']=0
    if args.without_continuity:cfg['continuity']=False
    if args.max_references is not None:
        if args.max_references<1:parser.error('--max-references must be positive.')
        cfg['max_references']=args.max_references
    if args.texture_resolution:cfg['texture_resolution']=args.texture_resolution
    if not 64<=cfg.get('texture_resolution',4096)<=8192:parser.error('Texture resolution must be 64..8192.')
    if cfg.get('virtual_surface_samples',0)<0:parser.error('Virtual sample count must be nonnegative.')
    run(cfg)


if __name__=='__main__':
    main()
