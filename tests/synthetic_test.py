"""Synthetic ground-truth evaluation; reports accuracy separately from fit."""
from pathlib import Path
import numpy as np
from PIL import Image

from normal_geometry import unit
from normal_shading import Settings, solve, fit_light, prepare_detail, linear_to_srgb
from run_refine import save_json, image8, montage


def angular(a,b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))


def run():
    out=Path(__file__).parent/'output'/'synthetic'; out.mkdir(parents=True,exist_ok=True)
    size=160; y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    n0=unit(np.stack([x*.48,-y*.4,np.ones_like(x)],-1))
    direction=np.zeros_like(n0); direction[...,0]=1
    direction=unit(direction-n0*np.sum(direction*n0,-1,keepdims=True))
    shape=np.sin(7*np.pi*x)*np.exp(-((x/.63)**2+(y/.72)**2))*0.085
    truth=unit(n0+direction*shape[...,None])
    valid=(x*x+y*y)<.84
    true_light=np.array([.29,.16,.36]); ambient=.08
    observed=ambient+np.maximum(truth@true_light,0)
    labels=(np.arange(size)[:,None]//20)*8+np.arange(size)[None,:]//20
    fitted_ambient,fitted_light,lighting=fit_light(n0,observed,valid)
    settings=Settings(detail_sigma=.7,low_frequency_sigma=9,tensor_sigma=1.8,
                      iterations=160,prior=.003,smoothness=.025,direction_prior=.07,noise_floor=.0004)
    detail=prepare_detail(n0,observed,valid,labels,fitted_ambient,fitted_light,settings)
    result,stats=solve(n0,detail['residual'],detail['confidence'],detail['direction'],fitted_light,settings)
    # Evaluate all fixed eligible pixels, including those the method skipped.
    before=angular(n0,truth); after=angular(result,truth)
    fixed=valid
    actual=dict(mean_angle_before=float(before[fixed].mean()),mean_angle_after=float(after[fixed].mean()),
                p95_angle_before=float(np.quantile(before[fixed],.95)),p95_angle_after=float(np.quantile(after[fixed],.95)),
                evaluation_pixels=int(fixed.sum()),evaluation_mask='Fixed entire eligible sphere, not only selected pixels.')
    known_result,known_stats=solve(n0,observed-(ambient+n0@true_light),valid.astype(float),direction,true_light,settings)
    known_after=angular(known_result,truth)
    actual['known_light_direction_prior_mean_angle_after']=float(known_after[fixed].mean())
    actual['pipeline_improved_mean_angle']=actual['mean_angle_after']<actual['mean_angle_before']
    report=dict(case='Curved normal field with tangent sinusoidal grooves, constant albedo, one diffuse light',
                true_light=true_light.tolist(),fitted_light=lighting,accuracy=actual,solver=stats,
                oracle_solver=known_stats,
                limitation='Synthetic data obeys the diffuse model. It does not validate real-image albedo, shadows, or camera errors.')
    save_json(out/'report.json',report)
    montage(out/'comparison.png',[('Synthetic observed shading',image8(linear_to_srgb(observed))),
        ('Base normal',image8(n0*.5+.5)),('Ground-truth normal',image8(truth*.5+.5)),
        ('Recovered normal',image8(result*.5+.5)),('Confidence',image8(detail['confidence'])),
        ('Recovered shading',image8(linear_to_srgb(ambient+np.maximum(result@true_light,0))))])
    np.savez_compressed(out/'ground_truth.npz',base=n0,truth=truth,result=result,valid=valid)
    print(__import__('json').dumps(report,indent=2))
    if not actual['pipeline_improved_mean_angle']:
        raise SystemExit('Synthetic pipeline did not improve mean angular error.')


if __name__=='__main__':
    run()
