"""Behavioral tests for full resolution, continuity and imperfect multi-light input."""
import unittest
import numpy as np
from scipy import ndimage as ndi
from normal_geometry import unit
from normal_multilight import (MultiSettings,solve_multilight,angular,
    sample_image,similarity_registration,detail_evidence)
from normal_structure import (connect_structures,compatible_edge,smooth_along,
    direction_weights,line_features)
from run_upgrade import choose_work_size


def synthetic(size=64):
    y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    base=unit(np.stack([.18*x,-.15*y,np.ones_like(x)],-1)).astype(np.float32)
    envelope=np.exp(-2*(x*x+y*y))
    delta=np.stack([.085*np.sin(5*np.pi*x)*envelope,
                    .055*np.cos(4*np.pi*y)*envelope,np.zeros_like(x)],-1)
    delta-=base*np.sum(base*delta,-1,keepdims=True)
    truth=unit(base+delta).astype(np.float32)
    lights=np.array([[-.5,.25,.7],[.55,.20,.65],[.04,.65,.6],[-.08,-.45,.8]])
    residual=np.stack([truth@light-base@light for light in lights]).astype(np.float32)
    confidence=np.ones_like(residual)
    return base,truth,lights,residual,confidence


class UpgradeTests(unittest.TestCase):
    def test_original_resolution_is_never_reduced(self):
        self.assertEqual(choose_work_size((1145,1374),450000,0),(1145,1374))
        w,h=choose_work_size((1145,1374),450000,2000000)
        self.assertGreaterEqual(w*h/(1145*1374)*450000,2000000)
        with self.assertRaisesRegex(ValueError,'refusing'):
            choose_work_size((5000,6000),100000,2000000,4096)

    def test_pixel_centers_and_identity_resampling(self):
        a=np.arange(35,dtype=np.float32).reshape(5,7)
        np.testing.assert_array_equal(sample_image(a,a.shape),a)
        # A linear source ramp should retain its subpixel value in the interior.
        enlarged=sample_image(a,(10,14))
        self.assertAlmostEqual(float(enlarged[3,3]),10.,places=5)

    def test_global_similarity_registration(self):
        rng=np.random.default_rng(7);a=rng.normal(size=(68,2))*100+300
        matrix=np.array([[1.03,-.02],[.02,1.03]]);offset=np.array([12,-8])
        source=a@matrix.T+offset
        m,t,report=similarity_registration(a,source)
        np.testing.assert_allclose(m,matrix,atol=1e-6)
        np.testing.assert_allclose(t,offset,atol=1e-5)
        self.assertLess(report['p95_landmark_residual'],1e-5)

    def test_clean_multilight_recovers_both_tangent_directions(self):
        base,truth,lights,r,c=synthetic()
        result,confidence,extra=solve_multilight(base,r,c,lights)
        before=float(angular(base,truth).mean());after=float(angular(result,truth).mean())
        self.assertLess(after,before*.20)
        self.assertGreater(float(confidence.mean()),.7)
        self.assertLess(extra['stats']['weighted_detail_rmse_after'],extra['stats']['weighted_detail_rmse_before']*.25)

    def test_zero_residual_does_not_invent_detail(self):
        base,truth,lights,r,c=synthetic(24)
        result,confidence,_=solve_multilight(base,np.zeros_like(r),c,lights)
        self.assertLess(float(angular(base,result).max()),1e-4)

    def test_collapsed_light_rank_falls_back_to_base(self):
        base=np.zeros((20,20,3),np.float32);base[...,2]=1
        lights=np.tile([.5,.1,.7],(4,1))
        r=np.ones((4,20,20),np.float32)*.04
        result,confidence,_=solve_multilight(base,r,np.ones_like(r),lights)
        np.testing.assert_array_equal(result,base)
        self.assertFalse(confidence.any())

    def test_shadow_protected_and_missing_pixels_unchanged(self):
        base,truth,lights,r,c=synthetic(32)
        c[:,10:18,10:18]=0
        c[1:,:,0:5]=0
        result,confidence,_=solve_multilight(base,r,c,lights)
        np.testing.assert_array_equal(result[10:18,10:18],base[10:18,10:18])
        np.testing.assert_array_equal(result[:,0:5],base[:,0:5])
        self.assertEqual(float(confidence[10:18,10:18].max()),0)

    def test_corrupt_reference_is_downweighted(self):
        base,truth,lights,r,c=synthetic(48)
        corrupted=r.copy();corrupted[1,12:36,12:36]+=.28
        result,confidence,extra=solve_multilight(base,corrupted,c,lights)
        baseline,_,_=solve_multilight(base,corrupted,c,lights,
                                      MultiSettings(robust_scale=100,iterations=1))
        region=np.s_[12:36,12:36]
        self.assertLess(float(angular(result,truth)[region].mean()),
                        float(angular(baseline,truth)[region].mean())*.6)
        self.assertLess(float(extra['observation_weights'][1][region].mean()),.15)

    def test_angle_bound_and_unit_norm_under_extreme_residual(self):
        base,truth,lights,r,c=synthetic(24)
        result,_,_=solve_multilight(base,r*60,c,lights)
        self.assertLessEqual(float(angular(base,result).max()),12.001)
        self.assertLess(float(np.abs(np.linalg.norm(result,axis=-1)-1).max()),2e-7)

    def test_nan_is_rejected(self):
        base,truth,lights,r,c=synthetic(16);r[0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'NaN'):
            solve_multilight(base,r,c,lights)

    def test_opposite_polarity_or_long_gap_cannot_link(self):
        shape=(40,60);labels=np.zeros(shape,np.int32);labels[:,30:]=1
        sem=np.ones(shape,np.int16);response=np.ones(shape,np.float32)
        valid=np.ones(shape,bool);threshold=np.ones(shape,np.float32)*.5
        a=dict(center=[24.,20.],tangent_dir=[1.,0.],region_id=0,semantic_id=1,
               polarity=1,strength=1.,scale=1.8,boundary_distance=4.)
        b=dict(a,center=[36.,20.],region_id=1)
        args=(labels,sem,response,threshold,valid,{(0,1)},18,12)
        self.assertIsNotNone(compatible_edge(a,b,*args))
        self.assertIsNone(compatible_edge(a,dict(b,polarity=-1),*args))
        response[:,26:35]=0
        self.assertIsNone(compatible_edge(a,b,*args))
        response[:]=1;valid[:,29:31]=False
        self.assertIsNone(compatible_edge(a,b,*args))

    def test_semantic_and_mesh_nonadjacency_prevent_linking(self):
        shape=(40,60);labels=np.zeros(shape,np.int32);labels[:,30:]=1
        sem=np.full(shape,11,np.int16);sem[:,30:]=12
        response=np.ones(shape,np.float32);valid=np.ones(shape,bool)
        a=dict(center=[24.,20.],tangent_dir=[1.,0.],region_id=0,semantic_id=11,
               polarity=1,strength=1.,scale=1.8,boundary_distance=4.)
        b=dict(a,center=[36.,20.],region_id=1,semantic_id=12)
        self.assertIsNone(compatible_edge(a,b,labels,sem,response,response*.5,valid,{(0,1)},18,12))
        sem[:]=1;a['semantic_id']=b['semantic_id']=1
        self.assertIsNone(compatible_edge(a,b,labels,sem,response,response*.5,valid,set(),18,12))

    def test_line_graph_crosses_neighboring_regions(self):
        y,x=np.mgrid[:96,:128]
        field=(-.07*np.exp(-((y-46)/2.2)**2)*(1+.12*np.sin(x*.35))).astype(np.float32)
        labels=(x//32).astype(np.int32)
        valid=np.ones(field.shape,bool);valid[:8]=False;valid[-8:]=False
        result=connect_structures(field,np.ones_like(field),valid,labels,np.ones_like(labels),.001)
        stats=result['graph']['stats']
        self.assertGreater(stats['chains'],0)
        self.assertGreater(stats['cross_region_edges'],0)
        self.assertGreater(float(result['confidence'][46,64]),float(result['confidence'][20,64]))

    def test_anisotropic_smoothing_preserves_cross_section(self):
        y,x=np.mgrid[:64,:96]
        field=np.exp(-((y-32)/1.7)**2)*(1+.3*np.sin(x*2.2))
        tangent=np.zeros((*field.shape,2),np.float32);tangent[...,0]=1
        wh,wv=direction_weights(tangent,np.ones_like(field),np.ones(field.shape,bool))
        result=smooth_along(field,wh,wv,iterations=8,amount=1.5)
        self.assertLess(float(result[32,8:-8].std()),float(field[32,8:-8].std())*.6)
        self.assertGreater(float(result[32].mean()),float(field[32].mean())*.94)
        self.assertLess(float(result[28].mean()),float(result[32].mean())*.03)

    def test_texture_free_native_evidence_stays_zero(self):
        base,truth,lights,r,c=synthetic(48)
        ambient=.08;observed=ambient+base@lights[0]
        valid=np.ones(observed.shape,bool)
        labels=(np.indices(observed.shape)[1]//12).astype(np.int32)
        detail=detail_evidence(base,observed,valid,labels,ambient,lights[0])
        self.assertLess(float(np.abs(detail['residual']).max()),1e-6)
        self.assertEqual(float(detail['confidence'].max()),0.)


if __name__=='__main__':
    unittest.main(verbosity=2)
