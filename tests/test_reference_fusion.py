"""Behavioral regressions for the document-55 source-selection workflow."""
import unittest

import numpy as np

from normal_geometry_field import (geometry_settings, solve_geometry, integrate_height,
                                   height_to_normal, curl_rms, height_edges)
from normal_layers import analyze_bands
from normal_multilight import angular
from normal_reference_fusion import (fusion_settings, weighted_median, blend_weights,
    match_mid_structures, align_mid_evidence, fuse_reference_height, primary_slope,
    project_candidate_height)
from test_geometry_field import flat_metric, photometric_fixture


class ReferenceFusionTests(unittest.TestCase):
    def test_missing_source_samples_do_not_create_zero_height_cliffs(self):
        y, x = np.mgrid[:40, :48]
        metric = flat_metric(x.shape, .3)
        valid = np.ones(x.shape, bool)
        slope = np.zeros((*x.shape, 3), np.float32)
        slope[..., 0] = -.03
        observed = np.ones(x.shape, np.float32)
        observed[12:25, 20:29] = 0
        missing = slope.copy(); missing[observed == 0] = 0
        height, _ = project_candidate_height(missing, observed, slope, valid, metric,
            geometry_settings(dict(screening_length_mm=1000., cg_rtol=1e-7)))
        truth = .009*x
        difference = height-truth; difference -= difference.mean()
        self.assertLess(float(np.max(np.abs(difference))), .0001)
        self.assertLess(float(np.max(np.abs(np.diff(height, axis=1)))), .0091)

    def test_invalid_settings_rejected(self):
        for options in [dict(enabled=1), dict(match_radius_pixels=40),
                        dict(confidence_low=.8, confidence_high=.4),
                        dict(brightness_bins=True), dict(structure_fraction=1),
                        dict(primary_max_angle_degrees=float('nan'))]:
            with self.assertRaises(ValueError):
                fusion_settings(options)

    def test_weighted_median_rejects_one_far_outlier(self):
        values = np.array([[0., 2.], [1., 3.], [2., 4.], [100., 100.]])
        np.testing.assert_allclose(weighted_median(values, np.ones_like(values)), [1, 3])

    def test_continuous_weights_sum_to_one_and_do_not_cross_holes(self):
        shape = (25, 100)
        valid = np.ones(shape, bool); valid[:, 48:52] = False
        photo = np.zeros(shape); photo[:, 52:] = 1
        wm, wp, ws = blend_weights(photo, np.ones(shape), valid)
        np.testing.assert_allclose((wm+wp+ws)[valid], 1, atol=1e-7)
        self.assertFalse(wm[:, :48].any())
        np.testing.assert_allclose(wm[:, 52:], 1, atol=1e-7)
        self.assertFalse((wm+wp+ws)[~valid].any())
        photo = np.tile(np.linspace(0, 1, 100), (25, 1))
        wm, _, _ = blend_weights(photo, np.zeros(shape), np.ones(shape, bool))
        self.assertLess(float(np.max(np.abs(np.diff(wm, axis=1)))), .05)

    def test_screening_reference_is_exact_without_any_photo_edges(self):
        shape = (20, 24); y, x = np.indices(shape)
        truth = .03*np.sin(x/4)*np.cos(y/5)
        zero = np.zeros(shape)
        recovered, _ = integrate_height(zero, zero, zero, np.ones(shape, bool),
            flat_metric(shape), reference_height=truth)
        np.testing.assert_allclose(recovered, truth, atol=1e-10)

    def test_shifted_and_opposite_sign_lines_match_nearby_centers(self):
        y, x = np.mgrid[:72, :80]
        centers = [36, 39, 34, 40]
        fields = np.stack([(-1 if i == 1 else 1)*.08*np.exp(-.5*((x-c)/3)**2)
                           for i, c in enumerate(centers)]).astype(np.float32)
        valid = np.ones(x.shape, bool)
        result = match_mid_structures(fields, np.ones_like(fields), valid)
        self.assertGreater(result['report']['consensus_nodes'], 10)
        core = np.abs(result['anchor'][:, 0]-36) < 1
        self.assertTrue(core.any())
        for i, center in enumerate(centers):
            chosen = core & (result['scores'][i] > 0)
            self.assertTrue(chosen.any())
            self.assertLess(abs(float(np.median(result['offsets'][i, chosen, 0]))-(center-36)), 1.1)
        warped, weights = align_mid_evidence(fields, np.ones_like(fields), valid, result, 5.)
        self.assertTrue(np.isfinite(warped).all())
        self.assertGreater(int(((weights > .025).sum(0) >= 3).sum()), 50)

    def test_protected_strip_blocks_correspondence_and_warp(self):
        y, x = np.mgrid[:60, :80]
        fields = np.stack([.08*np.exp(-.5*((x-c)/2)**2) for c in [31, 36, 36, 36]]).astype(np.float32)
        valid = np.ones(x.shape, bool); valid[:, 33:35] = False
        weights = np.broadcast_to(valid, fields.shape).astype(np.float32).copy()
        result = match_mid_structures(fields, weights, valid)
        core = np.abs(result['anchor'][:, 0]-31) < 1
        self.assertTrue(core.any())
        self.assertFalse(result['consensus'][core].any())
        _, warped_weights = align_mid_evidence(fields, weights, valid, result, 5.)
        self.assertFalse(warped_weights[:, ~valid].any())

    def test_one_invented_line_cannot_form_multiview_consensus(self):
        y, x = np.mgrid[:60, :80]
        fields = np.zeros((4, *x.shape), np.float32)
        fields[0] = .08*np.exp(-.5*((x-35)/3)**2)
        result = match_mid_structures(fields, np.ones_like(fields), np.ones(x.shape, bool))
        self.assertEqual(result['report']['consensus_nodes'], 0)

    def test_primary_slope_is_bounded_and_zero_evidence_is_neutral(self):
        shape = (20, 30)
        base = np.zeros((*shape, 3), np.float32); base[..., 2] = 1
        direction = np.zeros_like(base); direction[..., 0] = 1
        slope = primary_slope(base, np.ones(shape)*10, [-.4, .2, .7], direction, 5.)
        self.assertLessEqual(float(np.linalg.norm(slope, axis=-1).max()), np.tan(np.deg2rad(5))+1e-7)
        np.testing.assert_array_equal(primary_slope(base, np.zeros(shape), [-.4, .2, .7], direction, 5.), 0)

    def test_single_reference_keeps_detail_and_marks_it_as_primary(self):
        y, x = np.mgrid[:40, :48]
        base = np.zeros((*x.shape, 3), np.float32); base[..., 2] = 1
        lights = np.array([[-.4, .2, .7]])
        residual = .04*(x-24)/4*np.exp(-.5*((x-24)/4)**2)
        obs = (.03+base@lights[0]+residual)[None].astype(np.float32)
        weights = np.ones_like(obs)
        normal, _, extra = solve_geometry(base, obs, weights, lights, [.03])
        direction = np.zeros_like(base); direction[..., 0] = 1
        arrays = dict(geometry_confidences=weights, mid_residuals=residual[None].astype(np.float32),
            low_residuals=np.zeros_like(obs), mid_lights=lights, mid_ambients=np.array([.03]),
            mid_anchor_direction=direction)
        result = fuse_reference_height(base, normal, extra, arrays, np.ones(x.shape, bool),
            flat_metric(x.shape, .3), geometry_settings())
        self.assertGreater(float(np.ptp(result['height'])), .005)
        self.assertGreater(result['report']['mean_primary_weight'], .99)
        self.assertEqual(result['report']['recovered_from_no_multilight_samples'], x.size)
        self.assertLess(curl_rms(*height_edges(result['height']), result['valid']), 1e-12)
        arrays['geometry_confidences'][:] = 0
        empty = fuse_reference_height(base, normal, extra, arrays, np.ones(x.shape, bool),
            flat_metric(x.shape, .3), geometry_settings())
        self.assertFalse(empty['height'].any())
        self.assertFalse(empty['confidence'].any())

    def test_consistent_reflectance_does_not_select_primary_by_low_signal(self):
        base, _, lights, ambient, _ = photometric_fixture()
        y, x = np.indices(base.shape[:2])
        albedo = 1.+.12*np.sin(x/5)
        prediction = np.stack([a+base@l for a, l in zip(ambient, lights)])
        obs = (prediction*albedo).astype(np.float32)
        normal, _, extra = solve_geometry(base, obs, np.ones_like(obs), lights, ambient,
            dict(confidence_as_blend_weight=True))
        self.assertGreater(float(extra['photo_confidence'].mean()), .75)
        direction = np.zeros_like(base); direction[..., 0] = 1
        arrays = dict(geometry_confidences=np.ones_like(obs), mid_residuals=obs-prediction,
            low_residuals=np.zeros_like(obs), mid_lights=lights, mid_ambients=ambient,
            mid_anchor_direction=direction)
        metric = flat_metric(x.shape, .3)
        result = fuse_reference_height(base, normal, extra, arrays, np.ones(x.shape, bool), metric, geometry_settings())
        self.assertLess(result['report']['mean_primary_weight'], .01)
        reconstructed, _, _ = height_to_normal(base, result['height'], result['valid'], metric, 8.)
        self.assertLess(float(angular(base, reconstructed).mean()), .2)

    def test_mid_graph_reconnects_brightness_partitions(self):
        y, x = np.mgrid[:100, :100]
        base = np.zeros((*x.shape, 3), np.float32); base[..., 2] = 1
        observed = (.3+.002*y+.07*np.exp(-.5*((x-50)/4)**2)).astype(np.float32)
        bands, _ = analyze_bands(base, observed, np.ones(x.shape, bool),
            np.zeros(x.shape, np.int32), np.ones(x.shape, np.int16), 0., np.array([0., 0., .4]),
            selected_names=('mid',), brightness_bins=4)
        self.assertGreater(bands['mid']['graph']['stats']['cross_brightness_edges'], 0)


if __name__ == '__main__':
    unittest.main()
