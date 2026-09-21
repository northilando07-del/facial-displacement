"""Behavioral tests for the document-33 layer and lip implementation."""
import unittest
import numpy as np

from normal_geometry import unit
from normal_layers import (isolated_blur, split_residual, lip_masks, lip_observation,
    analyze_bands, compose_normals, solve_band, consensus_confidence, DEFAULT_CAPS)
from normal_multilight import MultiSettings, angular
from test_upgrade import synthetic


def mouth_fixture(size=100, opened=False):
    lm = np.zeros((68, 2), float)
    lm[48:60] = [[20, 50], [30, 40], [42, 35], [50, 38], [58, 35], [70, 40],
                 [80, 50], [70, 60], [58, 65], [50, 66], [42, 65], [30, 60]]
    lm[60:68] = [[25, 50], [40, 46 if opened else 50], [50, 45 if opened else 50],
                 [60, 46 if opened else 50], [75, 50], [60, 56 if opened else 50],
                 [50, 57 if opened else 50], [40, 56 if opened else 50]]
    base = np.zeros((size, size, 3), np.float32); base[..., 2] = 1
    return lm, base, np.ones((size, size), bool)


class LayerTests(unittest.TestCase):
    def test_pyramid_reconstructs_raw_without_losing_remainders(self):
        rng = np.random.default_rng(31)
        raw = rng.normal(size=(80, 96)).astype(np.float32)*.04
        valid = np.ones(raw.shape, bool); valid[20:30, 40:48] = False
        result = split_residual(raw, valid, (.8, 2., 7., 18.))
        self.assertLess(result['reconstruction_max_error'], 3e-8)
        for name in ('high', 'mid', 'low', 'noise_remainder', 'illumination_remainder'):
            self.assertFalse(result[name][~valid].any())

    def test_scales_reject_invalid_or_reordered_values(self):
        a = np.zeros((8, 8), np.float32)
        for scales in [(1, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, np.nan)]:
            with self.assertRaises(ValueError):
                split_residual(a, np.ones_like(a, bool), scales)

    def test_constant_appearance_is_not_normal_detail(self):
        valid = np.ones((50, 60), bool); valid[20:30, 25:35] = False
        result = split_residual(np.full(valid.shape, .17, np.float32), valid, (.8, 2., 7., 18.))
        for name in ('high', 'mid', 'low'):
            self.assertLess(float(np.abs(result[name]).max()), 1e-6)

    def test_invalid_values_cannot_bleed_into_visible_surface(self):
        valid = np.zeros((50, 60), bool); valid[10:40, 10:50] = True
        a = np.full(valid.shape, np.nan, np.float32); a[valid] = .12
        result = isolated_blur(a, valid, 5)
        np.testing.assert_allclose(result[valid], .12, atol=1e-7)
        self.assertFalse(result[~valid].any())

    def test_upper_and_lower_lip_do_not_share_blur_statistics(self):
        valid = np.ones((50, 60), bool); valid[24:27] = False
        a = np.zeros(valid.shape, np.float32); a[:24] = .1; a[27:] = -.2
        result = isolated_blur(a, valid, 12)
        np.testing.assert_allclose(result[:24], .1, atol=1e-7)
        np.testing.assert_allclose(result[27:], -.2, atol=1e-7)

    def test_high_mid_low_separate_distinct_wavelengths(self):
        x = np.arange(768)[None, :]
        valid = np.ones((64, 768), bool)
        for wavelength, expected in [(6, 'high'), (24, 'mid'), (160, 'low')]:
            raw = np.broadcast_to(.05*np.sin(x*2*np.pi/wavelength), valid.shape).astype(np.float32)
            bands = split_residual(raw, valid, (.8, 3., 18., 64.))
            rms = {k: float(np.mean(bands[k][:, 200:-200]**2)) for k in ('high', 'mid', 'low')}
            self.assertEqual(max(rms, key=rms.get), expected)

    def test_closed_mouth_slit_and_outside_are_protected(self):
        lm, base, geom = mouth_fixture()
        masks = lip_masks(geom.shape, lm, geom, base)
        self.assertGreater(masks['domain'].sum(), 500)
        self.assertFalse(masks['domain'][49:52, 25:76].any())
        self.assertFalse(masks['domain'][~masks['outer']].any())

    def test_open_mouth_teeth_and_cavity_are_protected(self):
        lm, base, geom = mouth_fixture(opened=True)
        masks = lip_masks(geom.shape, lm, geom, base)
        self.assertFalse(masks['domain'][47:55, 40:60].any())
        geom[:] = False
        self.assertFalse(lip_masks(geom.shape, lm, geom, base)['domain'].any())

    def test_lips_use_own_color_and_reject_clipped_highlights(self):
        lm, base, geom = mouth_fixture()
        rgb = np.zeros((*geom.shape, 3), np.float32); rgb[:] = [.55, .18, .20]
        domain = lip_masks(geom.shape, lm, geom, base)['domain']
        rgb[39:45, 40:48] = 1.
        valid, _ = lip_observation(rgb, lm, geom, base, domain)
        self.assertGreater(valid.sum(), 100)
        self.assertFalse(valid[39:45, 40:48].any())
        self.assertFalse(valid[~domain].any())

    def test_zero_signal_and_invalid_domain_produce_no_detail(self):
        lm, base, geom = mouth_fixture()
        labels = np.zeros(geom.shape, np.int32)
        light = np.array([.4, .2, .7]); observed = .1+base@light
        for valid in (geom, np.zeros_like(geom)):
            layers, _ = analyze_bands(base, observed, valid, labels, labels+1, .1, light,
                                      sigmas=(.8, 2., 5., 12.), continuity=False)
            for value in layers.values():
                self.assertLess(float(np.abs(value['residual']).max()), 1e-6)
                self.assertFalse(value['confidence'].any())

    def test_zero_controls_restore_base_and_single_control_is_independent(self):
        base, truth, _, _, _ = synthetic(24)
        out = compose_normals(base, dict(mid=truth), dict(mid=0))
        np.testing.assert_array_equal(out, base)
        out = compose_normals(base, dict(mid=truth, high=base), dict(mid=1, high=10))
        self.assertLess(float(angular(out, truth).max()), .001)

    def test_neutral_layers_preserve_curved_float32_base_exactly(self):
        rng = np.random.default_rng(44)
        base = unit(rng.normal(size=(120, 160, 3))).astype(np.float32)
        layers = {name: base.copy() for name in ('low', 'mid', 'high', 'lips')}
        out = compose_normals(base, layers)
        np.testing.assert_array_equal(out, base)

    def test_composition_is_order_independent_and_not_rgb_averaging(self):
        base = np.zeros((4, 4, 3), np.float32); base[..., 2] = 1
        a = unit(base+np.array([.04, 0, 0])).astype(np.float32)
        b = unit(base+np.array([0, .06, 0])).astype(np.float32)
        expected = unit(base+np.array([.04, .06, 0]))
        first = compose_normals(base, dict(mid=a, high=b), dict(mid=1, high=1))
        second = compose_normals(base, dict(high=b, mid=a), dict(mid=1, high=1))
        np.testing.assert_allclose(first, expected, atol=1e-7)
        np.testing.assert_allclose(first, second, atol=1e-7)

    def test_extreme_layer_gains_remain_bounded_and_finite(self):
        base, truth, _, _, _ = synthetic(32)
        out = compose_normals(base, dict(mid=truth, high=truth), dict(mid=100, high=100))
        self.assertLess(float(angular(base, out).max()), 12.001)
        self.assertLess(float(np.abs(np.linalg.norm(out, axis=-1)-1).max()), 2e-7)
        with self.assertRaises(ValueError):
            compose_normals(base, dict(mid=truth), dict(mid=-1))

    def test_rank_loss_stays_base_even_when_lip_fallback_is_enabled(self):
        base = np.zeros((20, 20, 3), np.float32); base[..., 2] = 1
        lights = np.tile([.5, .1, .7], (4, 1))
        residual = np.full((4, 20, 20), .04, np.float32)
        out, confidence, _ = solve_band(base, residual, np.ones_like(residual), lights,
            np.broadcast_to([1., 0, 0], base.shape), name='lips', allow_anchor_fallback=True)
        np.testing.assert_array_equal(out, base)
        self.assertFalse(confidence.any())

    def test_consensus_keeps_clean_views_and_downweights_one_corrupt_reference(self):
        base, truth, lights, residual, confidence = synthetic(40)
        settings = MultiSettings(robust_scale=.012)
        clean, _ = consensus_confidence(base, residual, confidence, lights, settings)
        self.assertGreater(float(clean.mean()), .98)
        residual[1, 10:30, 10:30] += .24
        protected, stats = consensus_confidence(base, residual, confidence, lights, settings)
        self.assertLess(float(protected[1, 10:30, 10:30].mean()), .1)
        self.assertGreater(float(protected[[0, 2, 3], 10:30, 10:30].mean()), .75)
        self.assertGreater(stats['strongly_downweighted_observations'], 0)

    def test_consensus_does_not_invent_redundancy_when_views_are_missing(self):
        base, truth, lights, residual, confidence = synthetic(20)
        confidence[2:] = 0
        protected, stats = consensus_confidence(base, residual, confidence, lights, MultiSettings())
        np.testing.assert_array_equal(protected, confidence)
        self.assertEqual(stats['eligible_pixels'], 0)

    def test_lip_anchor_fallback_is_limited_to_observed_region(self):
        base, truth, lights, residual, confidence = synthetic(28)
        confidence[1:] = 0; confidence[:, :6] = 0
        direction = np.zeros_like(base); direction[..., 0] = 1
        direction -= base*np.sum(base*direction, -1, keepdims=True)
        out, conf, extra = solve_band(base, residual, confidence, lights, unit(direction),
            name='lips', allow_anchor_fallback=True,
            settings=MultiSettings(maximum_angle_degrees=DEFAULT_CAPS['lips']))
        self.assertGreater(extra['stats']['anchor_only_prior_samples'], 0)
        self.assertGreater(float(angular(base, out).max()), .01)
        self.assertLess(float(angular(base, out).max()), 3.001)
        np.testing.assert_array_equal(out[:6], base[:6])
        self.assertFalse(conf[:6].any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
