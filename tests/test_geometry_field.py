"""Known-geometry tests for document 44, independent of the real face images."""
import unittest
import numpy as np

from normal_geometry import Mesh, Camera, rasterize, unit
from normal_geometry_field import (geometry_settings, solve_geometry, surface_metric,
    integrate_height, split_height, height_to_normal, SurfaceDiffusion, height_edges,
    curl_rms, surface_support, resample_height, limit_height_edges)
from normal_multilight import angular


def flat_metric(shape, step=1., shear=0.):
    jx = np.zeros((*shape, 3), np.float32); jx[..., 0] = step
    jy = np.zeros_like(jx); jy[..., 0] = shear; jy[..., 1] = -step
    xx, xy, yy = np.sum(jx*jx, -1), np.sum(jx*jy, -1), np.sum(jy*jy, -1)
    return dict(jx=jx, jy=jy, xx=xx, xy=xy, yy=yy,
        area=np.sqrt(xx*yy-xy*xy), valid=np.ones(shape, bool))


def photometric_fixture():
    y, x = np.mgrid[:36, :44]
    base = np.zeros((*x.shape, 3), np.float32); base[..., 2] = 1
    normal = unit(np.stack([.055*np.sin(x/5), .04*np.cos(y/6), np.ones(x.shape)], -1)).astype(np.float32)
    lights = np.array([[-.4, .2, .7], [.45, .1, .6], [.03, .55, .7], [.12, -.35, .65]])
    ambient = np.full(4, .03)
    albedo = 1.+.12*np.sin(x/8)*np.cos(y/7)
    observations = np.stack([albedo*(a+normal@l) for l, a in zip(lights, ambient)]).astype(np.float32)
    return base, normal, lights, ambient, observations


class GeometryFieldTests(unittest.TestCase):
    def test_local_height_bounds_remove_spike_without_flattening_distant_detail(self):
        y, x = np.mgrid[:40, :48]
        metric = flat_metric(x.shape, .5, .15)
        base = np.zeros((*x.shape, 3), np.float32); base[..., 2] = 1
        truth = .008*x+.006*y
        height = truth.copy(); height[20, 24] += .6
        bounded, report = limit_height_edges(height, metric['valid'], metric, 8.)
        normal, _, cap = height_to_normal(base, bounded, metric['valid'], metric, 8.)
        self.assertTrue(report['converged'])
        self.assertEqual(cap['height_cap_scale'], 1.)
        self.assertLess(float(angular(base, normal).max()), 8.)
        np.testing.assert_allclose(bounded[:8], truth[:8], atol=1e-10)
        self.assertLess(curl_rms(*height_edges(bounded), metric['valid']), 1e-12)

    def test_local_height_bounds_leave_safe_fields_and_separate_islands_unchanged(self):
        y, x = np.mgrid[:28, :40]
        metric = flat_metric(x.shape)
        valid = (x < 14) | (x > 24)
        height = np.where(x < 14, .003*x, 8.)
        bounded, report = limit_height_edges(height, valid, metric, 8.)
        np.testing.assert_array_equal(bounded[valid], height[valid])
        self.assertFalse(bounded[~valid].any())
        self.assertEqual(report['changed_samples'], 0)

    def test_joint_normals_with_albedo_and_opposite_lights(self):
        base, truth, lights, ambient, obs = photometric_fixture()
        normals, conf, extra = solve_geometry(base, obs, np.ones_like(obs), lights, ambient)
        error = angular(normals, truth)
        self.assertGreater((conf > .015).mean(), .99)
        self.assertLess(float(error.mean()), .2)
        self.assertLess(float(np.quantile(error, .95)), .4)
        self.assertFalse(extra['stats']['polarity_voting'])

    def test_reflectance_only_does_not_become_wrinkles(self):
        base, _, lights, ambient, _ = photometric_fixture()
        y, x = np.indices(base.shape[:2])
        albedo = .8+.4*(np.sin(x/3) > 0)
        obs = np.stack([albedo*(a+base@l) for l, a in zip(lights, ambient)])
        normal, _, _ = solve_geometry(base, obs, np.ones_like(obs), lights, ambient)
        self.assertLess(float(angular(base, normal).max()), .15)

    def test_robust_fit_rejects_one_relighting_artifact(self):
        base, truth, lights, ambient, obs = photometric_fixture()
        damaged = obs.copy(); damaged[0, 8:28, 12:32] += .18
        normal, _, extra = solve_geometry(base, damaged, np.ones_like(obs), lights, ambient)
        naive, _, _ = solve_geometry(base, damaged, np.ones_like(obs), lights, ambient,
            dict(robust_scale=100., iterations=1))
        region = np.s_[8:28, 12:32]
        self.assertLess(float(angular(normal, truth)[region].mean()),
                        float(angular(naive, truth)[region].mean())*.7)
        self.assertGreater(extra['stats']['rejected_observations'], 0)

    def test_insufficient_and_degenerate_lights_fall_back_exactly(self):
        base, _, lights, ambient, obs = photometric_fixture()
        for n in (1, 2):
            normal, conf, _ = solve_geometry(base, obs[:n], np.ones_like(obs[:n]), lights[:n], ambient[:n])
            np.testing.assert_array_equal(normal, base)
            self.assertFalse(conf.any())
        light = np.repeat(lights[:1], 4, axis=0)
        normal, conf, _ = solve_geometry(base, obs, np.ones_like(obs), light, ambient)
        np.testing.assert_array_equal(normal, base)
        self.assertFalse(conf.any())

    def test_height_gradient_orientation_and_uniform_cap(self):
        shape = (24, 30); y, x = np.indices(shape)
        base = np.zeros((*shape, 3), np.float32); base[..., 2] = 1
        metric = flat_metric(shape, .5)
        height = .02*x+.03*y
        normal, _, _ = height_to_normal(base, height, metric['valid'], metric, 12.)
        expected = unit(np.array([-.04, .06, 1.]))
        np.testing.assert_allclose(normal[4:-4, 4:-4], np.broadcast_to(expected, normal[4:-4, 4:-4].shape), atol=1e-7)
        capped, h, stats = height_to_normal(base, height*10, metric['valid'], metric, 2.)
        self.assertLessEqual(float(angular(base, capped).max()), 2.00001)
        self.assertLess(stats['height_cap_scale'], 1.)
        self.assertLess(curl_rms(*height_edges(h), metric['valid']), 1e-12)

    def test_poisson_recovers_known_height(self):
        y, x = np.mgrid[:40, :48]
        truth = .2*np.cos(x/8)*np.cos(y/9)
        gy, gx = np.gradient(truth)
        metric = flat_metric(truth.shape, .5)
        height, stats = integrate_height(gx, gy, np.ones(truth.shape), metric['valid'], metric,
            dict(screening_length_mm=1000., cg_rtol=1e-7))
        error = height-truth; error -= error.mean()
        self.assertLess(float(np.sqrt(np.mean(error**2))), .003)
        self.assertTrue(stats['converged'])
        self.assertLess(stats['curl_after'], 1e-12)

    def test_unsupported_spur_cannot_flatten_a_valid_height_layer(self):
        shape = (30, 40); y, x = np.indices(shape)
        base = np.zeros((*shape, 3), np.float32); base[..., 2] = 1
        metric = flat_metric(shape, .5)
        valid = (y > 3) & (y < 26) & (x > 3) & (x < 26)
        valid[15, 26:35] = True
        height = .005*x+.004*y
        height[15, 28:35] = 10.
        normal, exported, stats = height_to_normal(base, height, valid, metric, 1.5)
        self.assertEqual(stats['height_cap_scale'], 1.)
        self.assertGreater(stats['unsupported_surface_samples'], 0)
        expected = unit(np.array([-.01, .008, 1.]))
        np.testing.assert_allclose(normal[8:20, 8:20],
            np.broadcast_to(expected, normal[8:20, 8:20].shape), atol=1e-7)
        self.assertFalse(exported[~surface_support(valid)].any())

    def test_one_dimensional_bridge_does_not_join_surface_islands(self):
        shape = (24, 30); y, x = np.indices(shape)
        metric = flat_metric(shape)
        valid = (x < 8) | (x > 20)
        valid[12, 8:21] = True
        gx = np.where(x < 8, .04, 0.)
        height, _ = integrate_height(gx, np.zeros(shape), np.ones(shape), valid, metric)
        np.testing.assert_allclose(height[x > 20], 0, atol=1e-12)
        support = surface_support(valid)
        # Include the triangular boundary tips in each constant island.
        field = np.where(x < 15, 2., 8.)
        smooth, _ = SurfaceDiffusion(valid, metric).smooth(field, 5., geometry_settings())
        np.testing.assert_allclose(smooth[(x < 8) & support], 2., atol=1e-8)
        np.testing.assert_array_equal(smooth[~support], 0)

    def test_masked_height_resampling_cannot_create_zero_height_cliffs(self):
        shape = (31, 35); y, x = np.indices(shape)
        valid = (x > 3) & (x < 31) & (y > 2) & (y < 28)
        valid[9:19, 13:21] = False
        height = np.where(valid, .2, 99.)
        for target_shape in (shape, (59, 67), (23, 29)):
            sampled, support = resample_height(height, valid, target_shape)
            np.testing.assert_allclose(sampled[support], .2, atol=3e-8)
            self.assertFalse(sampled[~support].any())
            base = np.zeros((*target_shape, 3), np.float32); base[..., 2] = 1
            normals, _, stats = height_to_normal(base, sampled, support,
                flat_metric(target_shape, .3), 1.5)
            self.assertLess(float(angular(base, normals).max()), 1e-4)
            self.assertEqual(stats['height_cap_scale'], 1.)

    def test_poisson_filters_nonintegrable_rotation(self):
        y, x = np.mgrid[-20:21, -20:21]
        metric = flat_metric(x.shape)
        _, report = integrate_height(-.03*y, .03*x, np.ones(x.shape), metric['valid'], metric)
        self.assertGreater(report['curl_before'], .05)
        self.assertLess(report['curl_after'], 1e-12)
        self.assertGreater(report['weighted_gradient_correction_rms'], .1)

    def test_mask_holes_separate_both_integration_and_filtering(self):
        shape = (28, 40); y, x = np.indices(shape)
        metric = flat_metric(shape)
        valid = (x < 14) | (x > 24)
        gx = np.where(x < 14, .04, 0.)
        height, _ = integrate_height(gx, np.zeros(shape), np.ones(shape), valid, metric)
        np.testing.assert_allclose(height[x > 24], 0, atol=1e-12)
        field = np.where(x < 14, 2., np.where(x > 24, 8., 0.))
        smooth, _ = SurfaceDiffusion(valid, metric).smooth(field, 5., geometry_settings())
        np.testing.assert_allclose(smooth[x < 14], 2., atol=1e-8)
        np.testing.assert_allclose(smooth[x > 24], 8., atol=1e-8)
        np.testing.assert_array_equal(smooth[~valid], 0)

    def test_mixed_metric_fem_has_correct_physical_energy(self):
        shape = (21, 24); y, x = np.indices(shape)
        metric = flat_metric(shape, .5, .2)
        operator = SurfaceDiffusion(metric['valid'], metric)
        # Physical coordinates are X=.5*x+.2*y, Y=-.5*y.
        field = .7*(.5*x+.2*y)-.3*(-.5*y)
        value = field.ravel()
        energy = float(value@(operator.stiffness@value))
        physical_area = (shape[0]-1)*(shape[1]-1)*.25
        self.assertAlmostEqual(energy/physical_area, .7**2+.3**2, places=5)

    def test_physical_filter_scale_invariant_across_resolution(self):
        amplitudes = []
        for side in (33, 65):
            y, x = np.mgrid[:side, :side]
            field = np.cos(np.pi*x/(side-1))*np.cos(np.pi*y/(side-1))
            metric = flat_metric(field.shape, 32./(side-1))
            smooth, _ = SurfaceDiffusion(metric['valid'], metric).smooth(field, 4., geometry_settings(dict(cg_rtol=1e-7)))
            amplitudes.append(float(np.sum(smooth*field)/np.sum(field**2)))
        expected = 1./(1.+.5*4.**2*2*(np.pi/32)**2)
        self.assertLess(abs(amplitudes[0]-amplitudes[1]), .008)
        self.assertLess(abs(amplitudes[1]-expected), .008)

    def test_height_bands_telescope(self):
        rng = np.random.default_rng(7)
        height = rng.normal(size=(25, 29))*.02
        metric = flat_metric(height.shape)
        bands, report = split_height(height, metric['valid'], metric)
        reconstructed = sum(bands[k] for k in ('fine_remainder', 'mid_fine', 'mid_large', 'low', 'illumination_remainder'))
        np.testing.assert_allclose(reconstructed, height, atol=1e-12)
        self.assertLess(report['reconstruction_max_error_mm'], 1e-12)

    def test_uv_stretch_and_mirror_do_not_change_surface_scale(self):
        vertices = np.array([[0., 0, 0], [4, 0, 0], [0, 4, 0], [4, 4, 0]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]], np.int32)
        uv = vertices[:, :2]/4.
        normals = np.tile([0., 0, 1], (2, 3, 1))
        mesh = Mesh(vertices, uv, triangles, triangles.copy(), normals,
            np.arange(2), np.zeros(2, np.int32), ['skin'], 2, 'synthetic')
        camera = Camera(np.zeros(3), np.eye(3), 4., np.array([1., 18.]))
        raster = rasterize(mesh, camera, (20, 20))
        a, _ = surface_metric(mesh, camera, raster, dict(millimeters_per_unit=1.))
        mesh.uv = uv*np.array([-.17, 2.3])+[.7, 0]
        b, _ = surface_metric(mesh, camera, raster, dict(millimeters_per_unit=1.))
        for key in ('xx', 'xy', 'yy', 'valid'):
            np.testing.assert_allclose(a[key], b[key], atol=1e-7)

    def test_invalid_configuration_and_nonfinite_observation_rejected(self):
        for options in (dict(sigmas_mm=[1, 1, 2, 3]), dict(millimeters_per_unit=0), dict(iterations=0)):
            with self.assertRaises(ValueError): geometry_settings(options)
        base, _, lights, ambient, obs = photometric_fixture()
        obs[0, 3, 4] = np.nan
        with self.assertRaises(ValueError):
            solve_geometry(base, obs, np.ones_like(obs), lights, ambient)


if __name__ == '__main__':
    unittest.main()
