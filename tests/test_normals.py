"""Meaningful numerical and I/O checks for the normal-refinement prototype."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import struct
import zlib
import numpy as np

from normal_geometry import (Mesh, Camera, unit, load_obj, calibrate_camera,
                             rasterize, corner_tangents, group_patches)
from normal_shading import Settings, solve, prepare_detail, tensor_direction
from run_refine import png16


def plane_fields(size=48):
    normal = np.zeros((size, size, 3)); normal[..., 2] = 1
    direction = np.zeros_like(normal); direction[..., 0] = 1
    confidence = np.ones((size, size))
    return normal, direction, confidence


class NormalTests(unittest.TestCase):
    def test_zero_residual_is_identity(self):
        n,d,c = plane_fields()
        result, stats = solve(n, np.zeros(c.shape), c, d, [.6,.2,.77], Settings(iterations=20))
        np.testing.assert_allclose(result, n, atol=1e-7)
        self.assertEqual(stats['objective_after'], 0)

    def test_bright_dark_sign_and_actual_angular_error(self):
        n,d,c = plane_fields()
        x = np.linspace(0, 4*np.pi, n.shape[1])
        truth = unit(n + d * (.075 * np.sin(x))[None,:,None])
        light = unit(np.array([.65,.12,.75]))
        residual = truth @ light - n @ light
        settings = Settings(iterations=100, prior=.002, smoothness=.01, direction_prior=.1)
        result, stats = solve(n,residual,c,d,light,settings)
        before = np.mean(np.linalg.norm(n-truth,axis=-1))
        after = np.mean(np.linalg.norm(result-truth,axis=-1))
        self.assertLess(after,before*.35)
        self.assertLess(stats['weighted_detail_rmse_after'],stats['weighted_detail_rmse_before']*.35)
        self.assertLess(stats['objective_after'],stats['objective_before'])

    def test_direction_sign_is_unoriented(self):
        n,d,c = plane_fields(20)
        residual = np.full(c.shape,.025)
        settings = Settings(iterations=20)
        a,_ = solve(n,residual,c,d,[.6,.2,.77],settings)
        b,_ = solve(n,residual,c,-d,[.6,.2,.77],settings)
        np.testing.assert_allclose(a,b,atol=1e-7)

    def test_grazing_light_sensitivity_and_angle_cap(self):
        n,d,c = plane_fields(20)
        result, stats = solve(n,np.ones(c.shape)*2,c,d,[.00001,0,1],Settings(iterations=40))
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(stats['normal_angle_max_degrees'],8.0001)
        self.assertLess(stats['normal_unit_length_max_error'],1e-10)

    def test_protected_pixels_do_not_change(self):
        n,d,c = plane_fields(30); c[:,12:18]=0
        result,_ = solve(n,np.ones(c.shape)*.04,c,d,[.6,0,.8],Settings(iterations=30))
        np.testing.assert_array_equal(result[:,12:18],n[:,12:18])

    def test_tangent_light_magnitude_matches_finite_difference(self):
        n=unit(np.array([.12,.34,1.])); light=np.array([.43,.18,.31])
        d=unit(np.cross(n,[0,1,0])); eps=1e-6
        actual=((unit(n+eps*d)@light)-(n@light))/eps
        expected=(light-(n@light)*n)@d
        self.assertAlmostEqual(actual,expected,places=6)

    def test_texture_free_sphere_does_not_invent_detail(self):
        y,x=np.mgrid[-1:1:64j,-1:1:64j]
        n=unit(np.stack([x*.3,-y*.3,np.ones_like(x)],-1))
        light=np.array([.22,.12,.38]); ambient=.08
        observed=ambient+n@light
        valid=(x*x+y*y)<.85
        labels=(np.arange(64)[:,None]//16)*4+np.arange(64)[None,:]//16
        detail=prepare_detail(n,observed,valid,labels,ambient,light,Settings())
        self.assertLess(float(np.abs(detail['residual']).max()),1e-8)
        self.assertEqual(int((detail['confidence']>.015).sum()),0)

    def test_tiny_two_pixel_patches_stay_finite(self):
        n,d,c=plane_fields(24)
        y,x=np.mgrid[:24,:24]
        observed=.4+.03*np.sin(x/3)
        labels=(np.arange(24*24)//2).reshape(24,24)
        detail=prepare_detail(n,observed,c.astype(bool),labels,.1,np.array([.2,.1,.3]),Settings())
        self.assertTrue(np.isfinite(detail['residual']).all())
        self.assertTrue(np.isfinite(detail['confidence']).all())
        self.assertTrue(all(np.isfinite(row['modal_residual']) for row in detail['patches']))

    def test_nonfinite_solver_inputs_are_rejected(self):
        n,d,c=plane_fields(12); residual=np.zeros(c.shape); residual[3,4]=np.nan
        with self.assertRaisesRegex(ValueError,'NaN'):
            solve(n,residual,c,d,[.6,0,.8],Settings())

    def test_depth_buffer_uses_front_surface(self):
        vertices=np.array([[0,0,0],[6,0,0],[0,-6,0],[0,0,1],[6,0,1],[0,-6,1]],float)
        faces=np.array([[0,1,2],[3,4,5]])
        mesh=Mesh(vertices,np.array([[0,0],[1,0],[0,1]]),faces,np.tile([0,1,2],(2,1)),
                  np.tile([0.,0.,1.],(2,3,1)),np.arange(2),np.zeros(2,int),['skin'],2,'synthetic')
        camera=Camera(np.zeros(3),np.eye(3),1.,np.zeros(2))
        r=rasterize(mesh,camera,(8,8))
        self.assertEqual(int(r['triangle'][1,1]),1)
        self.assertAlmostEqual(float(r['depth'][1,1]),1)
        point=r['barycentric'][1,1]@r['screen_vertices'][faces[1]]
        np.testing.assert_allclose(point,[1.5,1.5],atol=1e-6)

    def test_camera_recovers_rigid_orthographic_projection(self):
        from scipy.spatial.transform import Rotation
        rng=np.random.default_rng(2026)
        points=rng.normal(size=(68,3))*[5,8,2]+[0,155,8]
        camera=Camera(points.mean(0),Rotation.from_euler('xyz',[.08,-.05,.015]).as_matrix(),19,np.array([250,300]))
        target,_=camera.project(points)
        fitted,report=calibrate_camera(points,target)
        self.assertLess(report['max_pixels'],1e-5)
        np.testing.assert_allclose(fitted.project(points)[0],target,atol=1e-5)

    def test_mirrored_uv_handedness_and_source_unchanged(self):
        text='v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl skin\nf 1/1 2/2 3/3\n'
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'mesh.obj'; p.write_text(text)
            mesh=load_obj(p); t,b=corner_tangents(mesh)
            np.testing.assert_allclose(t[0],np.tile([1,0,0],(3,1)),atol=1e-8)
            np.testing.assert_allclose(b[0],np.tile([0,1,0],(3,1)),atol=1e-8)
            self.assertEqual(p.read_text(),text)
            mesh.uv[:,0]=1-mesh.uv[:,0]
            t,b=corner_tangents(mesh)
            handedness=np.sum(np.cross(mesh.corner_normals,t)*b,axis=-1)
            self.assertTrue(np.all(handedness<0))

    def test_png_is_real_16bit_rgb_and_roundtrips(self):
        rng=np.random.default_rng(9); a=rng.random((4,7,3))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'normal.png'; png16(p,a); raw=p.read_bytes()
            self.assertEqual(raw[:8],b'\x89PNG\r\n\x1a\n')
            offset=8; payload=[]
            while offset<len(raw):
                length=struct.unpack('>I',raw[offset:offset+4])[0]
                kind=raw[offset+4:offset+8]; data=raw[offset+8:offset+8+length]
                crc=struct.unpack('>I',raw[offset+8+length:offset+12+length])[0]
                self.assertEqual(crc,zlib.crc32(kind+data)&0xffffffff)
                if kind==b'IHDR': self.assertEqual(struct.unpack('>IIBBBBB',data),(7,4,16,2,0,0,0))
                if kind==b'IDAT': payload.append(data)
                offset+=length+12
            pixels=zlib.decompress(b''.join(payload)); rows=[]
            for i in range(4):
                row=pixels[i*(7*6+1):(i+1)*(7*6+1)]; self.assertEqual(row[0],0)
                rows.append(np.frombuffer(row[1:],dtype='>u2').reshape(7,3)/65535)
            np.testing.assert_allclose(np.array(rows),a,atol=1/65535)


if __name__=='__main__':
    unittest.main(verbosity=2)
