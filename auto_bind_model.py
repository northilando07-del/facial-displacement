"""Detect a face in shaded source-mesh views, lift FAN68 onto visible triangles.

Run in the existing face_alignment Python environment. No source mesh edits.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

from normal_geometry import Camera,load_obj,rasterize,unit


def main():
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--soap-root',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.soap_root))
    from headlab_geometry import ObjModel,select_head
    import torch,face_alignment
    detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
        device='cuda' if torch.cuda.is_available() else 'cpu',flip_input=False,face_detector='sfd')
    original=load_obj(args.mesh);model=ObjModel(args.mesh)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    skip=[n for n in original.materials if any(k in n.lower() for k in ('brow','lash','tear','hair'))]
    frames=[np.eye(3),np.diag([-1,1,-1])]
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            frame=np.eye(3)[list(permutation)]*np.asarray(signs)[:,None]
            if np.linalg.det(frame)>.5 and not any(np.array_equal(frame,r) for r in frames):frames.append(frame)
    attempts=[];resolution=640
    for index,frame in enumerate(frames):
        print('Source face orientation',index+1,'/',len(frames),flush=True)
        model.v=original.vertices@frame.T
        try:selection=select_head(model)
        except ValueError as e:
            attempts.append(dict(view=index,error=str(e)));continue
        center=(selection['head_min']+selection['head_max'])/2
        span=max(selection['head_height']*1.25,(selection['head_max'][0]-selection['head_min'][0])*1.2)
        if span<=0:continue
        camera=Camera(center@frame,frame,resolution/span,np.array([resolution/2,resolution/2]))
        raster=rasterize(original,camera,(resolution,resolution),skip)
        normals=raster['normal'];visible=raster['triangle']>=0
        shade=.3+.65*np.maximum(normals@unit(np.array([-.35,.45,.85])),0)
        rgb=np.repeat(shade[...,None],3,axis=-1);rgb[~visible]=.92
        image=Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8))
        faces=detector.get_landmarks(np.asarray(image))
        if faces is None or len(faces)!=1:
            attempts.append(dict(view=index,face_count=0 if faces is None else len(faces)));continue
        xy=np.asarray(faces[0],float)
        if xy.shape!=(68,2) or not np.isfinite(xy).all() or not (xy[36,0]<xy[39,0]<xy[42,0]<xy[45,0] and xy[30,1]<xy[48:60,1].mean()<xy[8,1]):continue
        # Material selection follows actual visible facial samples, never old names.
        allowed=visible & (normals[...,2]>.05)
        for i,name in enumerate(original.materials):
            if any(s in name.lower() for s in ('eye','lash','brow','tear','teeth','tongue','hair','mouth')):
                allowed &= original.material_ids[np.maximum(raster['triangle'],0)]!=i
        yy,xx=np.nonzero(allowed)
        if not len(xx):continue
        distances,near=cKDTree(np.column_stack([xx,yy])).query(xy)
        if distances.max()>resolution*.04:
            attempts.append(dict(view=index,snap_max=float(distances.max())));continue
        ys,xs=yy[near],xx[near];ids=raster['triangle'][ys,xs];bary=raster['barycentric'][ys,xs].astype(float)
        bary=np.clip(bary,0,1);bary/=bary.sum(1,keepdims=True)
        # Include skin materials present around the detected face, including cheeks.
        x0,x1=np.clip([int(xy[:,0].min()),int(xy[:,0].max())+1],0,resolution)
        y0,y1=np.clip([int(xy[27,1]-(xy[8,1]-xy[27,1])*.65),int(xy[8,1])+1],0,resolution)
        mask=allowed.copy();mask[:y0]=False;mask[y1:]=False;mask[:,:x0]=False;mask[:,x1:]=False
        mids,counts=np.unique(original.material_ids[raster['triangle'][mask]],return_counts=True)
        selected=[int(i) for i,n in zip(mids,counts) if n>=max(20,mask.sum()*.005)]
        if not selected:continue
        face_mask=np.isin(original.material_ids,selected)
        if np.any(original.triangle_uv[face_mask]<0):raise ValueError('Selected facial surface has missing UVs. Supply an OBJ with UV coordinates.')
        uvs=original.uv[original.triangle_uv[face_mask]]
        if not np.isfinite(uvs).all() or uvs.min()<-1e-5 or uvs.max()>1.00001:
            raise ValueError('Current bake requires 0-1 UVs per material; UDIM needs conversion first.')
        image.save(out/'source_registration.png')
        data=dict(mesh_sha256=original.sha256,source_indices=original.triangles[ids].tolist(),
            source_barycentric=bary.tolist(),source_rotation=frame.tolist(),source_landmarks_pixels=xy.tolist(),
            source_registration_size=[resolution,resolution],pixel_snap_max=float(distances.max()),
            skin_materials=[original.materials[i] for i in selected],attempts=attempts,
            method='FAN68 on orthographic source render, visible triangle barycentric binding; automatic head selection and axis search.')
        (out/'source_binding.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        print('SOURCE_BINDING_READY',flush=True);return
    (out/'binding_failure.json').write_text(json.dumps(attempts,indent=2),encoding='utf-8')
    raise ValueError('Could not identify a reliable source face. Use a clearly modeled human head with correct normals and visible facial features; inspect model orientation/head range.')


if __name__=='__main__':main()
