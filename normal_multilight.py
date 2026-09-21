"""Full-source-resolution evidence and robust base-normal-constrained light fusion."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from normal_geometry import unit
from normal_shading import tensor_direction


@dataclass
class MultiSettings:
    maximum_angle_degrees: float = 12.
    prior: float = .003
    albedo_prior: float = .08
    robust_scale: float = .018
    iterations: int = 7
    chunk_size: int = 65536
    min_condition_ratio: float = .015


def blur(field,mask,sigma):
    mask=np.asarray(mask,np.float32)
    denom=ndi.gaussian_filter(mask,sigma,mode='nearest')
    return ndi.gaussian_filter(np.asarray(field,np.float32)*mask,sigma,mode='nearest')/np.maximum(denom,1e-8)


def detail_evidence(normals,observed,valid,labels,ambient,light,
                    detail_sigma=.65,low_sigma=15.,noise_floor=.0012):
    """Local modal residuals at ORIGINAL image resolution; grouped ROI statistics."""
    predicted=(ambient+np.maximum(normals@light,0)).astype(np.float32)
    raw=observed-predicted
    high=observed-blur(observed,valid,.65)
    noise=max(noise_floor,float(np.median(np.abs(high[valid]))/.67449))
    count=max(int(labels.max())+1,1)
    modes=np.zeros(count,np.float32)
    rows=[]
    slices=ndi.find_objects(np.where(valid,labels+1,0))
    for label,sl in enumerate(slices):
        if sl is None: continue
        inside=(labels[sl]==label)&valid[sl]
        values=raw[sl][inside]
        if not values.size: continue
        lo,hi=np.quantile(values,[.05,.95])
        mode=float(np.median(values))
        if hi-lo>1e-5:
            hist,edges=np.histogram(values,bins=max(8,min(40,int(np.sqrt(values.size)))),range=(lo,hi))
            peak=int(hist.argmax());members=values[(values>=edges[peak])&(values<=edges[peak+1])]
            if members.size:mode=float(np.median(members))
        modes[label]=mode
        rows.append(dict(patch=label,pixels=int(values.size),modal_residual=mode))
    modal=blur(modes[np.maximum(labels,0)],valid,low_sigma)
    centered=blur(raw,valid,detail_sigma)-modal
    residual=centered-blur(centered,valid,low_sigma)
    residual[~valid]=0
    direction,coherence,energy=tensor_direction(residual,normals,2.2)
    enabled=np.zeros(count,bool)
    for row in rows:
        label=row['patch'];sl=slices[label];inside=(labels[sl]==label)&valid[sl]
        signal=float(np.quantile(np.abs(residual[sl][inside]),.8))
        fraction=float(np.mean(coherence[sl][inside]>.18))
        enabled[label]=row['pixels']>=12 and signal>noise*1.15 and fraction>.10
        row.update(signal_p80=signal,coherent_fraction=fraction,refined=bool(enabled[label]))
    border=np.clip(ndi.distance_transform_edt(valid)/5,0,1).astype(np.float32)
    facing=np.clip((normals[...,2]-.2)/.4,0,1)
    structure=np.clip((coherence-.15)/.55,0,1)
    signal=ndi.gaussian_filter(np.abs(residual),2.2)
    snr=np.clip((signal-.4*noise)/max(2.5*noise,1e-5),0,1)
    confidence=valid*enabled[np.maximum(labels,0)]*border*facing*structure*snr
    confidence=blur(confidence,valid,.7)*valid*border
    confidence*=np.exp(-np.maximum(np.abs(raw-modal)-.15,0)**2/.055**2)
    # Observations on the unlit side of the fitted diffuse lobe are unreliable.
    confidence*=np.clip((normals@light-.005)/.045,0,1)
    return dict(predicted=predicted,raw=raw,modal=modal,
        residual=np.clip(residual,-.15,.15).astype(np.float32),
        direction=direction.astype(np.float32),coherence=coherence.astype(np.float32),
        confidence=confidence.astype(np.float32),noise=noise,patches=rows,energy=energy)


def similarity_registration(anchor,source):
    """Robust global similarity, without nonrigidly forcing different faces to match."""
    anchor,source=np.asarray(anchor,float),np.asarray(source,float)
    if anchor.shape!=(68,2) or source.shape!=(68,2):
        raise ValueError('Registration requires 68 corresponding landmarks.')
    weights=np.ones(68);weights[:17]=.4;weights[17:27]=.2;weights[60:]=.3
    center=anchor.mean(0)
    p=anchor-center
    initial=[1,0,*source.mean(0)]
    def transform(q):
        a,b,tx,ty=q
        return p@np.array([[a,b],[-b,a]])+[tx,ty]
    fit=least_squares(lambda q:((transform(q)-source)*np.sqrt(weights[:,None])).ravel(),
                      initial,loss='soft_l1',f_scale=2.)
    a,b,tx,ty=fit.x
    matrix=np.array([[a,-b],[b,a]])
    offset=np.array([tx,ty])-matrix@center
    error=np.linalg.norm(anchor@matrix.T+offset-source,axis=1)
    return matrix,offset,dict(matrix=matrix.tolist(),offset=offset.tolist(),
        mean_landmark_residual=float(error.mean()),p95_landmark_residual=float(np.quantile(error,.95)),
        scale=float(np.hypot(a,b)),rotation_degrees=float(np.degrees(np.arctan2(b,a))),
        per_landmark_residual=error.tolist(),kind='global similarity; no local warping')


def sample_image(field,shape,matrix=None,offset=None,source_size=None,order=1):
    """Pixel-center-correct source-to-working-grid resampling, in row chunks."""
    h,w=shape
    source_h,source_w=field.shape[:2]
    ref_w,ref_h=source_size or (source_w,source_h)
    matrix=np.eye(2) if matrix is None else np.asarray(matrix)
    offset=np.zeros(2) if offset is None else np.asarray(offset)
    result=np.empty((h,w)+field.shape[2:],np.float32)
    for y0 in range(0,h,128):
        y1=min(y0+128,h)
        yy,xx=np.meshgrid((np.arange(y0,y1)+.5)*ref_h/h-.5,
                          (np.arange(w)+.5)*ref_w/w-.5,indexing='ij')
        sx=matrix[0,0]*xx+matrix[0,1]*yy+offset[0]
        sy=matrix[1,0]*xx+matrix[1,1]*yy+offset[1]
        coords=np.stack([sy,sx])
        if field.ndim==2:
            result[y0:y1]=ndi.map_coordinates(field,coords,order=order,mode='constant',cval=0,prefilter=order>1)
        else:
            for c in range(field.shape[2]):
                result[y0:y1,:,c]=ndi.map_coordinates(field[...,c],coords,order=order,mode='constant',cval=0,prefilter=order>1)
    return result


def tangent_frame(normals):
    axis=np.zeros_like(normals);axis[...,0]=1
    t=unit(axis-normals*normals[...,:1])
    bad=np.linalg.norm(t,axis=-1)<.1
    if np.any(bad):t[bad]=unit(np.cross(normals[bad],[0,1,0]))
    return t,unit(np.cross(normals,t))


def constrain_normals(base,result,confidence,angle):
    cosine=np.sum(base*result,-1,keepdims=True)
    delta=result/np.maximum(cosine,.1)-base
    delta-=base*np.sum(base*delta,-1,keepdims=True)
    cap=np.tan(np.deg2rad(angle))*np.sqrt(np.clip(confidence,0,1))
    delta*=np.minimum(1,cap/np.maximum(np.linalg.norm(delta,axis=-1),1e-9))[...,None]
    delta[confidence<=.015]=0
    result=unit(base+delta).astype(np.float32)
    result[confidence<=.015]=base[confidence<=.015]
    return result


def solve_multilight(normals,residuals,confidences,lights,settings=None):
    """Robust local normal update plus a shared relative-albedo nuisance term.

    Estimated light powers absorb the unknown average albedo. This is a
    prior-constrained residual model, not calibrated metric photometric stereo.
    The weighted tangent design must have rank two; degenerate pixels fall back
    exactly to the supplied base. Shadow/misalignment gates arrive per image.
    """
    settings=settings or MultiSettings()
    n0=np.asarray(normals,np.float32)
    r=np.asarray(residuals,np.float32);conf=np.asarray(confidences,np.float32)
    lights=np.asarray(lights,np.float64)
    if r.ndim!=3 or r.shape!=conf.shape or r.shape[1:]!=n0.shape[:2] or lights.shape!=(r.shape[0],3):
        raise ValueError('Incompatible multi-light shapes.')
    if not all(np.isfinite(a).all() for a in (n0,r,conf,lights)):
        raise ValueError('Multi-light inputs contain NaN or infinity.')
    if settings.prior<=0 or settings.albedo_prior<=0 or settings.robust_scale<=0:
        raise ValueError('Priors and robust scale must be positive.')
    h,w=n0.shape[:2];m=len(lights)
    flat=n0.reshape(-1,3)
    rf=r.reshape(m,-1);cf=np.clip(conf.reshape(m,-1),0,1)
    out=flat.copy();confidence=np.zeros(h*w,np.float32)
    relative_albedo=np.zeros(h*w,np.float32)
    conditions=np.zeros(h*w,np.float32)
    final_weights=np.zeros_like(cf)
    active_ids=np.flatnonzero((cf>.015).sum(0)>=2)
    for start in range(0,len(active_ids),settings.chunk_size):
        ids=active_ids[start:start+settings.chunk_size]
        base=flat[ids].astype(np.float64)
        t,b=tangent_frame(base)
        baseline=np.maximum(base@lights.T,0)
        design=np.stack([t@lights.T,b@lights.T,baseline],axis=-1)
        target=rf[:,ids].T.astype(np.float64)
        weights=cf[:,ids].T.astype(np.float64)
        # Signed residual spikes beyond the base-normal motion envelope are
        # suppressed before IRLS; this also makes the initial fit less fragile.
        envelope=np.tan(np.deg2rad(settings.maximum_angle_degrees))*np.linalg.norm(design[...,:2],axis=-1)+settings.robust_scale
        weights*=np.minimum(1,envelope/np.maximum(np.abs(target),1e-9))**2
        robust=np.ones_like(weights)
        solution=np.zeros((len(ids),3))
        regularizer=np.diag([settings.prior,settings.prior,settings.albedo_prior])
        for _ in range(settings.iterations):
            ww=weights*robust
            lhs=np.einsum('nmi,nm,nmj->nij',design,ww,design)+regularizer
            rhs=np.einsum('nmi,nm,nm->ni',design,ww,target)
            solution=np.linalg.solve(lhs,rhs[...,None])[...,0]
            solution[:,2]=np.clip(solution[:,2],-.18,.18)
            error=np.einsum('nmi,ni->nm',design,solution)-target
            robust=1/(1+(error/settings.robust_scale)**2)
        ww=weights*robust
        tangent_design=design[...,:2]
        fisher=np.einsum('nmi,nm,nmj->nij',tangent_design,ww,tangent_design)
        eigen=np.linalg.eigvalsh(fisher)
        ratio=eigen[:,0]/np.maximum(eigen[:,1],1e-10)
        supported=(ww>.025).sum(1)>=2
        supported&=ratio>=settings.min_condition_ratio
        reliability=np.clip(ww.sum(1)/max(min(m,3),1),0,1)
        reliability*=np.clip(ratio/.12,0,1)*supported
        candidate=unit(base+t*solution[:,:1]+b*solution[:,1:2])
        candidate=constrain_normals(base,candidate,reliability,settings.maximum_angle_degrees)
        out[ids]=candidate;confidence[ids]=reliability
        relative_albedo[ids]=solution[:,2]*supported
        conditions[ids]=ratio
        final_weights[:,ids]=ww.T
    result=out.reshape(h,w,3)
    stats=fit_metrics(n0,result,r,conf,lights)
    angles=angular(n0,result)
    active=confidence.reshape(h,w)>.015
    stats.update(active_virtual_cells=int(active.sum()),
        normal_angle_max_degrees=float(angles.max()),
        normal_angle_mean_active_degrees=float(angles[active].mean()) if active.any() else 0.,
        normal_angle_p95_active_degrees=float(np.quantile(angles[active],.95)) if active.any() else 0.,
        normal_unit_length_max_error=float(np.abs(np.linalg.norm(result,axis=-1)-1).max()),
        insufficient_or_degenerate_cells=int(((cf>.015).any(0)&(confidence<=.015)).sum()),
        globally_estimated_light_condition=float(np.linalg.cond(lights)) if np.linalg.matrix_rank(lights)==3 else None,
        robust_rejected_observations=int(((cf>.025)&(final_weights<cf*.25)).sum()),
        local_condition_ratio_median=float(np.median(conditions[confidence>.015])) if active.any() else 0.,
        solver='Tangent normal + shared relative-albedo nuisance, bounded IRLS; base fallback on rank loss',
        albedo_note='Relative nuisance estimate, not an absolute skin reflectance texture.')
    return result,confidence.reshape(h,w),dict(stats=stats,
        albedo_delta=relative_albedo.reshape(h,w),condition_ratio=conditions.reshape(h,w),
        observation_weights=final_weights.reshape(r.shape))


def angular(a,b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))


def fit_metrics(base,result,residuals,confidences,lights):
    rows=[];before_total=0.;after_total=0.;weight_total=0.
    for residual,confidence,light in zip(residuals,confidences,lights):
        target=np.maximum(base@light,0)+residual
        before=float(np.sum(confidence*residual**2,dtype=np.float64))
        after=float(np.sum(confidence*(np.maximum(result@light,0)-target)**2,dtype=np.float64))
        weight=float(confidence.sum(dtype=np.float64))
        before_total+=before;after_total+=after;weight_total+=weight
        rows.append(dict(rmse_before=float(np.sqrt(before/max(weight,1e-12))),
                         rmse_after=float(np.sqrt(after/max(weight,1e-12)))))
    before=float(np.sqrt(before_total/max(weight_total,1e-12)))
    after=float(np.sqrt(after_total/max(weight_total,1e-12)))
    return dict(weighted_detail_rmse_before=before,weighted_detail_rmse_after=after,
                detail_rmse_reduction_fraction=1-after/before if before>1e-12 else 0.,
                per_view=rows,metric_note='Fixed input confidence, filtered shading target; not normal ground-truth accuracy. Albedo nuisance excluded from this normal-only metric.')
