"""Optional synthesized microdetail for brow/forehead gaps; never measured detail."""
import numpy as np
from scipy import ndimage as ndi
from normal_multilight import tangent_frame
from normal_layers import slope_from_base, normal_from_slope


def fill_pores(base, normal, confidence, target_region, donor_region, pixel_scale=1.,
               strength=.35, max_angle=1.2, max_distance=80., seed=33):
    if not 0 <= strength <= 1 or not 0 < max_angle <= 3 or max_distance <= 0:
        raise ValueError('Invalid pore synthesis parameters.')
    delta = slope_from_base(base, normal)
    tangent, bitangent = tangent_frame(base)
    uv = np.stack([np.sum(delta*tangent,-1), np.sum(delta*bitangent,-1)],-1)
    donors = donor_region & (confidence > .12)
    donors = ndi.binary_erosion(donors, iterations=max(1,round(3*pixel_scale)))
    weight = ndi.gaussian_filter(donors.astype(np.float32), 1.8*pixel_scale)
    micro = np.zeros_like(uv)
    for c in range(2):
        field = np.where(donors, uv[...,c],0)
        a=ndi.gaussian_filter(field,.6*pixel_scale)/np.maximum(ndi.gaussian_filter(donors.astype(np.float32),.6*pixel_scale),1e-8)
        b=ndi.gaussian_filter(field,1.8*pixel_scale)/np.maximum(weight,1e-8)
        micro[...,c]=a-b
    # Reject strongly oriented donor patches (hair-like lines / wrinkle edges).
    energy=np.sum(micro**2,-1)
    gx=ndi.sobel(energy,axis=1);gy=ndi.sobel(energy,axis=0)
    a=ndi.gaussian_filter(gx*gx,3*pixel_scale);b=ndi.gaussian_filter(gx*gy,3*pixel_scale);c=ndi.gaussian_filter(gy*gy,3*pixel_scale)
    coherence=np.sqrt((a-c)**2+4*b*b)/(a+c+1e-15)
    donors &= (coherence < .7) & (energy > 1e-10)
    output=normal.copy(); conf=confidence.copy(); generated=np.zeros(conf.shape,np.float32)
    if not donors.any() or strength==0:
        return output,conf,generated,dict(synthesized_pixels=0,reason='No reliable microtexture donors or zero strength.')
    distance,nearest=ndi.distance_transform_edt(~donors,return_indices=True)
    target=target_region & (confidence <= .015) & (distance <= max_distance*pixel_scale)
    if not target.any():
        return output,conf,generated,dict(synthesized_pixels=0,reason='No eligible gaps within donor reach.')
    # Overlapping donor patches preserve local fine texture without copying broad shading.
    size=max(8,round(20*pixel_scale));step=max(4,size//2);radius=size//2
    canvas=np.zeros_like(uv);weights=np.zeros(conf.shape,np.float32);rng=np.random.default_rng(seed)
    yy,xx=np.nonzero(target);h,w=target.shape
    for cy in range(int(yy.min()),int(yy.max())+step,step):
        for cx in range(int(xx.min()),int(xx.max())+step,step):
            y0,y1=max(0,cy-radius),min(h,cy+radius+1);x0,x1=max(0,cx-radius),min(w,cx+radius+1)
            if y0>=y1 or x0>=x1 or not target[y0:y1,x0:x1].any():continue
            py=int(np.clip(cy+rng.integers(-step,step+1),0,h-1));px=int(np.clip(cx+rng.integers(-step,step+1),0,w-1))
            dy,dx=nearest[:,py,px]
            sy=np.clip(np.arange(y0,y1)-cy+dy,0,h-1);sx=np.clip(np.arange(x0,x1)-cx+dx,0,w-1)
            patch=micro[sy[:,None],sx[None,:]];valid=donors[sy[:,None],sx[None,:]]
            window=np.exp(-2*((np.arange(y0,y1)-cy)/radius)**2)[:,None]*np.exp(-2*((np.arange(x0,x1)-cx)/radius)**2)[None,:]*valid
            canvas[y0:y1,x0:x1]+=patch*window[...,None];weights[y0:y1,x0:x1]+=window
    canvas/=np.maximum(weights[...,None],1e-8)
    for c in range(2):canvas[...,c]-=ndi.gaussian_filter(canvas[...,c],2*pixel_scale)
    feather=np.clip(ndi.distance_transform_edt(target)/(5*pixel_scale),0,1)
    generated=target*feather*(weights>1e-6)
    synthesized=(tangent*canvas[...,:1]+bitangent*canvas[...,1:2])*strength*generated[...,None]
    candidate=normal_from_slope(base,synthesized,max_angle)
    take=generated>0
    output[take]=candidate[take];conf[take]=.06*generated[take]+.016
    return output,conf,generated.astype(np.float32),dict(synthesized_pixels=int(take.sum()),
        donor_pixels=int(donors.sum()),strength=strength,angle_cap=max_angle,
        provenance='Synthesized overlapping nearby high-frequency normal patches; not recovered anatomy.')
