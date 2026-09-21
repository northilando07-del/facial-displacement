"""Geometry invariants and known-height reconstruction for document 77."""
import unittest
import copy
import numpy as np
from displacement_geometry import (Surface, Evidence, BARY_NODES, TEMPLATES,
    split_surface, solve_level, settings, differential, face_gradient,
    edges_of, deformed_normals, choose_refinement, aggregate)


def grid(n=7):
    x, y = np.meshgrid(np.linspace(-5, 5, n), np.linspace(-5, 5, n))
    vertices = np.column_stack([x.ravel(), y.ravel(), np.zeros(n*n)])
    faces = []
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i; b=a+1; c=a+n; d=c+1
            faces.extend([(a,b,d),(a,d,c)])
    faces = np.asarray(faces, np.int32)
    normals = np.tile([0.,0.,1.], (len(vertices),1))
    return Surface(vertices,normals,faces,vertices[faces,:2]/10+.5,
        np.zeros(len(faces),np.int32),np.arange(len(faces)),
        np.zeros(len(vertices)),np.zeros(len(vertices),np.int16))


def samples(surface, gradient=None, per_face=50):
    rng = np.random.default_rng(77)
    face = np.repeat(np.arange(len(surface.faces)), per_face)
    bary = rng.dirichlet([1,1,1], len(face))
    p = np.einsum('ni,nij->nj',bary,surface.rest[surface.faces[face]])
    g = np.zeros_like(p) if gradient is None else gradient(p)
    return Evidence(face,bary,g,np.ones(len(face)),np.zeros(len(face)),np.ones(len(face),np.int8))


class HierarchyTests(unittest.TestCase):
    def test_all_split_templates_cover_parent_without_overlap(self):
        for code, faces in TEMPLATES.items():
            p=BARY_NODES[np.asarray(faces)][:,:,1:]
            a,b=p[:,1]-p[:,0],p[:,2]-p[:,0]
            area=(a[:,0]*b[:,1]-a[:,1]*b[:,0])*.5
            self.assertTrue(np.all(area>0),code)
            self.assertAlmostEqual(float(area.sum()),.5,places=12)

    def test_adaptive_split_has_no_new_boundary_and_transports_samples(self):
        s=grid();e=samples(s)
        original=np.einsum('ni,nij->nj',e.bary,s.rest[s.faces[e.face]])
        selected=np.arange(len(s.faces))%5==0
        n,q,info=split_surface(s,e,selected,1)
        actual=np.einsum('ni,nij->nj',q.bary,n.rest[n.faces[q.face]])
        np.testing.assert_allclose(actual,original,atol=1e-12)
        edges,_,counts=edges_of(n.faces)
        self.assertLessEqual(int(counts.max()),2)
        p=n.rest[edges[counts==1]]
        on_boundary=np.any(np.all(np.isclose(np.abs(p[:,:,:2]),5),axis=1),axis=1)
        self.assertTrue(on_boundary.all())
        self.assertGreater(info['transition_faces'],0)

    def test_uv_seam_preserved_while_geometry_edge_is_shared(self):
        s=grid(2);s.uv[1]+=3
        e=samples(s)
        n,q,info=split_surface(s,e,[True,False],1)
        for face,parent in enumerate(info['parent_face']):
            expected=info['child_bary'][face]@s.uv[parent]
            np.testing.assert_allclose(n.uv[face],expected,atol=1e-7)
        self.assertLessEqual(int(edges_of(n.faces)[2].max()),2)

    def test_zero_signal_never_sculpts_and_never_refines(self):
        s=grid();e=samples(s);before=s.positions.copy();cfg=settings()
        report,stats,error,delta=solve_level(s,e,0,cfg)
        np.testing.assert_array_equal(s.positions,before)
        np.testing.assert_allclose(deformed_normals(s),s.directions,atol=1e-12)
        self.assertFalse(choose_refinement(s,stats,error,cfg,10).any())

    def test_known_plane_gradient_and_no_double_application(self):
        s=grid();e=samples(s,lambda p:np.tile([.04,-.02,0.],(len(p),1)))
        cfg=settings(dict(screening_length_mm=10000.))
        report,_,_,_=solve_level(s,e,0,cfg)
        expected=.04*s.rest[:,0]-.02*s.rest[:,1]
        np.testing.assert_allclose(s.height,expected,atol=2e-6)
        before=s.height.copy()
        again,_,_,delta=solve_level(s,e,0,cfg)
        self.assertLess(float(np.max(np.abs(delta))),1e-6)
        np.testing.assert_allclose(s.height,before,atol=1e-6)
        self.assertLess(report['residual_rms_after'],1e-6)

    def test_fine_solve_keeps_parent_vertices_and_reduces_residual(self):
        s=grid(5)
        def g(p):
            return np.column_stack([.08*np.cos(p[:,0]*1.2),.02*np.sin(p[:,1]),np.zeros(len(p))])
        e=samples(s,g,300);cfg=settings(dict(screening_length_mm=1000.))
        solve_level(s,e,0,cfg)
        n,q,info=split_surface(s,e,np.ones(len(s.faces),bool),1)
        parent=n.height[:len(s.rest)].copy()
        report,_,_,_=solve_level(n,q,1,cfg)
        np.testing.assert_array_equal(n.height[:len(s.rest)],parent)
        self.assertLess(report['residual_rms_after'],report['residual_rms_before'])
        self.assertEqual(report['boundary_and_parent_max_delta_mm'],0)

    def test_unobserved_components_and_domain_boundaries_are_pinned(self):
        s=grid();e=samples(s,lambda p:np.tile([.1,0,0.],(len(p),1)))
        e.weight[e.face>=len(s.faces)//2]=0
        original=s.height.copy()
        report,stats,_,_=solve_level(s,e,0,settings())
        missing=s.faces[np.unique(e.face[e.weight==0])].ravel()
        np.testing.assert_array_equal(s.height[missing],original[missing])
        e=samples(grid());e.domain[::2]=2
        self.assertTrue(aggregate(grid(),e)['mixed'].all())

    def test_invalid_settings_rejected(self):
        for overrides in ({'levels':-1},{'levels':6},{'levels':5.0},{'levels':True},
                          {'maximum_displacement_mm':float('nan')},
                          {'step_limits_mm':[0.]},{'millimeters_per_unit':0},
                          {'max_vertices':0},{'freeze_parent_vertices':1}):
            with self.assertRaises(ValueError):settings(overrides)

    def test_five_subdivisions_preserve_observations_and_parent_geometry(self):
        s=grid(3)
        e=samples(s,lambda p:np.column_stack([.05*np.cos(p[:,0]),
            .02*np.sin(p[:,1]),np.zeros(len(p))]),per_face=4096)
        cfg=settings({'levels':5})
        original=np.einsum('ni,nij->nj',e.bary,s.rest[s.faces[e.face]])
        solve_level(s,e,0,cfg)
        for level in range(1,6):
            before=s.positions.copy()
            s,e,info=split_surface(s,e,np.ones(len(s.faces),bool),level)
            report,_,_,_=solve_level(s,e,level,cfg)
            np.testing.assert_array_equal(s.positions[:len(before)],before)
            actual=np.einsum('ni,nij->nj',e.bary,s.rest[s.faces[e.face]])
            np.testing.assert_allclose(actual,original,atol=1e-11)
            self.assertEqual(len(s.faces),8*4**level)
            self.assertEqual(int(s.generation.max()),level)
            self.assertLessEqual(report['energy_after'],report['energy_before']+1e-10)
            self.assertLessEqual(report['max_delta_mm'],cfg['step_limits_mm'][level]+1e-12)
            self.assertEqual(int(edges_of(s.faces)[2].max()),2)

    def test_zero_signal_on_curved_surface_ignores_chain_priority(self):
        from displacement_geometry import unit
        s=grid();s.rest[:,2]=.05*np.sum(s.rest[:,:2]**2,axis=1)
        s.directions=unit(np.column_stack([-.1*s.rest[:,0],-.1*s.rest[:,1],np.ones(len(s.rest))]))
        e=samples(s);e.chain[:]=1
        result,stats,error,_=solve_level(s,e,0,settings())
        self.assertFalse(choose_refinement(s,stats,error,settings(),100).any())
        self.assertEqual(result['changed_vertices'],0)

    def test_zero_displacement_still_exports_neutral_data_map(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from export_displacement_maps import bake
        s=grid(2)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'mesh.npz'
            np.savez(source,faces=s.faces,corner_uv=s.uv,material_ids=s.materials,
                material_names=np.array(['skin']),height_mm=np.zeros(4),millimeters_per_unit=1.)
            result=bake(source,root/'maps',16,1.5,material_names=['skin'])
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['changed_texels'],0)
            data=np.load(root/'maps/skin/displacement_signed_mm.npy')
            np.testing.assert_array_equal(data,np.zeros((16,16)))
            self.assertTrue((root/'maps/skin/displacement_16.png').is_file())

    def test_displacement_data_quantization_stays_within_half_a_step(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from export_displacement_maps import bake
        s=grid(2)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'mesh.npz'
            np.savez(source,faces=s.faces,corner_uv=s.uv,material_ids=s.materials,
                material_names=np.array(['skin']),height_mm=np.full(4,.064362384,np.float32),
                millimeters_per_unit=1.)
            bake(source,root/'maps',32,1.5,material_names=['skin'])
            path=root/'maps/skin'
            original=np.load(path/'displacement_signed_mm.npy')
            decoded=(np.asarray(Image.open(path/'displacement_16.png'),dtype=float)/65535-.5)*3
            self.assertLessEqual(float(np.max(np.abs(decoded-original))),1.5/65535+1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
