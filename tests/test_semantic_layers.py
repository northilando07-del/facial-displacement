"""Document 66 regressions using displaced lines and analytical photometry."""
import unittest
import numpy as np

from normal_semantic_layers import (semantic_settings, support_policy, match_scale_structures,
    fit_transverse_slope, build_semantic_heights, spread_proposals)
from normal_geometry_field import (geometry_settings, limit_height_edges, height_to_normal,
                                   height_edges, curl_rms)
from test_geometry_field import flat_metric


def line_fixture(centers=(70, 80, 62, 72), sigma=6., step=.25, shape=(120, 160)):
    y, x = np.indices(shape)
    fields = np.stack([(-1.)**i*.07*np.exp(-.5*((x-c)/sigma)**2) for i, c in enumerate(centers)]).astype(np.float32)
    metric = flat_metric(shape, step)
    return fields, np.ones_like(fields), metric['valid'], metric


class SemanticLayerTests(unittest.TestCase):
    def test_policy_fold_accepts_each_documented_case(self):
        primary = np.array([.9, .5, .1, .9, .9])
        auxiliary = np.array([[.9, .8, .8, 0, 0], [0, .8, .8, 0, 0], [0, 0, .8, 0, 0]])
        accepted, fallback = support_policy('fold', primary, auxiliary, 4)
        np.testing.assert_array_equal(accepted, [True, True, True, False, False])
        np.testing.assert_array_equal(fallback, [False, False, False, True, True])

    def test_policy_fine_needs_three_witnesses_and_never_falls_back(self):
        accepted, fallback = support_policy('fine', np.array([1., 1., 1.]),
            np.array([[1., 1., 0.], [1., 0., 0.], [0., 0., 0.]]), 4)
        np.testing.assert_array_equal(accepted, [True, False, False])
        self.assertFalse(fallback.any())
        a, b = support_policy('fine', np.ones(1), np.ones((1, 1)), 2)
        self.assertFalse(a.any() or b.any())

    def test_single_reference_keeps_only_tagged_fold_fallback(self):
        for name, expected in [('fold', True), ('wrinkle', False), ('fine', False)]:
            a, b = support_policy(name, np.ones(1), np.empty((0, 1)), 1)
            self.assertFalse(a.any())
            self.assertEqual(bool(b.any()), expected)

    def test_physical_fold_matching_accepts_large_drift_and_opposite_lights(self):
        fields, weights, valid, metric = line_fixture()
        result = match_scale_structures(fields, weights, valid, metric, 'fold')
        self.assertGreater(result['report']['consensus_nodes'], 40)
        self.assertGreater(result['report']['matched_nodes_per_view'][1], 40)
        self.assertLessEqual(result['report']['radius_mm'], 6.)
        self.assertGreater(len(result['report']['centerline_edges']), 20)
        dense = spread_proposals(fields, weights, valid, metric, result, 'fold')
        self.assertLessEqual(dense['max_shift_mm'], 5.00001)
        self.assertGreater(float(dense['accepted'].max()), .5)

    def test_two_views_can_confirm_fold_but_not_fine(self):
        fields, weights, valid, metric = line_fixture(centers=(70, 80))
        result = match_scale_structures(fields, weights, valid, metric, 'fold')
        self.assertGreater(result['report']['consensus_nodes'], 40)
        a, _ = support_policy('fine', np.ones(1), np.ones((1, 1)), 2)
        self.assertFalse(a.any())

    def test_three_auxiliary_views_recover_weak_primary(self):
        fields, weights, valid, metric = line_fixture()
        fields[0] = 0
        result = match_scale_structures(fields, weights, valid, metric, 'fold')
        self.assertEqual(result['report']['primary_proposals'], 0)
        self.assertGreater(result['report']['consensus_nodes'], 30)

    def test_short_auxiliary_fragments_can_confirm_a_long_primary_fold(self):
        fields, weights, valid, metric = line_fixture(centers=(70, 80), sigma=6.)
        # Primary is a 30 mm chain. The auxiliary confirms only a 5 mm fragment,
        # which must not itself meet the primary's 8 mm proposal requirement.
        fields[1, :45] = 0; fields[1, 65:] = 0
        result = match_scale_structures(fields, weights, valid, metric, 'fold')
        self.assertGreater(result['report']['consensus_nodes'], 5)
        self.assertGreater(result['report']['primary_proposals'], 30)

    def test_observed_single_view_fake_fold_is_rejected(self):
        fields, weights, valid, metric = line_fixture()
        fields[1:] = 0
        result = match_scale_structures(fields, weights, valid, metric, 'fold')
        self.assertGreater(result['report']['primary_proposals'], 20)
        self.assertEqual(result['report']['consensus_nodes'], 0)
        self.assertEqual(result['report']['fallback_nodes'], 0)

    def test_match_radius_uses_physical_units(self):
        fields, weights, valid, metric = line_fixture(centers=(70, 100), step=.25)
        result = match_scale_structures(fields, weights, valid, metric, 'fold')
        self.assertEqual(result['report']['consensus_nodes'], 0)

    def test_protected_strip_blocks_correspondences(self):
        fields, weights, valid, metric = line_fixture(centers=(70, 82))
        valid[:, 76:78] = False
        fields[:, ~valid] = 0
        weights[:, ~valid] = 0
        result = match_scale_structures(fields, weights, valid, metric, 'fold')
        self.assertEqual(result['report']['consensus_nodes'], 0)

    def test_short_narrow_marks_are_not_promoted_to_fold(self):
        fields, weights, valid, metric = line_fixture(sigma=1.)
        fields[:, :50] = 0; fields[:, 62:] = 0
        result = match_scale_structures(fields, weights, valid, metric, 'fold')
        self.assertEqual(result['report']['consensus_nodes'], 0)

    def test_fine_lines_need_three_supporting_images(self):
        fields, weights, valid, metric = line_fixture(centers=(50, 52, 48, 51),
                                                      sigma=1.8, step=.10, shape=(80, 100))
        fields[3] = 0
        result = match_scale_structures(fields, weights, valid, metric, 'fine')
        self.assertGreater(result['report']['consensus_nodes'], 20)
        fields[2] = 0
        result = match_scale_structures(fields, weights, valid, metric, 'fine')
        self.assertEqual(result['report']['consensus_nodes'], 0)

    def test_signed_fit_separates_albedo_from_transverse_geometry(self):
        shape = (24, 32)
        base = np.zeros((*shape, 3), np.float32); base[..., 2] = 1
        direction = np.zeros_like(base); direction[..., 0] = 1
        lights = np.array([[-.5, .1, .7], [.5, .2, .65], [.1, .5, .6], [-.2, -.4, .65]])
        ambient = np.full(4, .03)
        y, x = np.indices(shape)
        albedo = .08*np.sin(x/6)
        predicted = np.stack([a+base@l for a, l in zip(ambient, lights)])
        residual = predicted*albedo
        slope, _, quality, report = fit_transverse_slope(base, direction, residual,
            np.ones_like(residual), lights, ambient, 7.)
        self.assertLess(float(np.abs(slope).max()), 1e-5)
        self.assertGreater(float(quality.mean()), .98)
        self.assertFalse(report['full_2d_normal_inferred'])
        truth = .04*np.sin(x/5)
        residual += np.stack([direction@l for l in lights])*truth
        slope, _, quality, _ = fit_transverse_slope(base, direction, residual,
            np.ones_like(residual), lights, ambient, 7.)
        self.assertLess(float(np.abs(slope-truth).mean()), .001)

    def test_parallel_lights_cannot_claim_independent_geometry(self):
        shape = (12, 14)
        base = np.zeros((*shape, 3), np.float32); base[..., 2] = 1
        direction = np.zeros_like(base); direction[..., 0] = 1
        lights = np.tile([.4, .1, .7], (3, 1))
        residual = np.full((3, *shape), .02)
        _, _, quality, _ = fit_transverse_slope(base, direction, residual,
            np.ones_like(residual), lights, np.full(3, .03), 7.)
        self.assertFalse((quality > 0).any())

    def test_exact_height_completion_after_tiny_iteration_budget(self):
        y, x = np.indices((80, 90)); metric = flat_metric(x.shape, .5, .15)
        truth = .008*x+.006*y; height = truth.copy(); height[40, 45] += .8
        bounded, report = limit_height_edges(height, metric['valid'], metric, 8., maxiter=1)
        self.assertTrue(report['converged'])
        self.assertTrue(report['exact_envelope_completion'])
        self.assertLessEqual(report['maximum_relative_edge_violation'], .001)
        np.testing.assert_allclose(bounded[:8], truth[:8], atol=1e-10)
        base = np.zeros((*x.shape, 3), np.float32); base[..., 2] = 1
        _, _, normal_report = height_to_normal(base, bounded, metric['valid'], metric, 8.)
        self.assertEqual(normal_report['height_cap_scale'], 1.)
        self.assertLess(curl_rms(*height_edges(bounded), metric['valid']), 1e-12)

    def test_invalid_settings_are_rejected(self):
        for bad in [dict(enabled='yes'), dict(primary_strong=.2), dict(fallback_gain=float('nan')),
                    dict(policies={'fold': {'radius_mm': 100}}), dict(unknown=True)]:
            with self.assertRaises(ValueError):
                semantic_settings(bad)

    def test_end_to_end_fold_height_is_nonzero_and_all_layers_are_integrable(self):
        fields, weights, valid, metric = line_fixture(centers=(70, 70, 70, 70), sigma=7.)
        shape = valid.shape
        base = np.zeros((*shape, 3), np.float32); base[..., 2] = 1
        lights = np.array([[-.5, .1, .7], [.5, .2, .65], [.1, .5, .6], [-.2, -.4, .65]])
        ambient = np.full(4, .03)
        y, x = np.indices(shape)
        true_slope = .075*np.exp(-.5*((x-70)/7.)**2)
        predicted = np.stack([a+base@light for a, light in zip(ambient, lights)])
        observations = predicted+lights[:, 0, None, None]*true_slope
        arrays = dict(geometry_observations=observations.astype(np.float32),
            geometry_confidences=weights, geometry_lights=lights, geometry_ambients=ambient)
        result = build_semantic_heights(base, arrays, valid, metric,
            geometry_settings(), {'enabled': True})
        self.assertEqual(set(result['bands']), {'fold', 'wrinkle', 'fine'})
        self.assertGreater(result['report']['matching']['fold']['consensus_nodes'], 20)
        self.assertGreater(float(np.max(np.abs(result['bands']['fold']))), .005)
        self.assertFalse(result['report']['pore_reconstruction'])
        for name, height in result['bands'].items():
            self.assertTrue(result['report']['layers'][name]['integrability']['converged'])
            self.assertLess(curl_rms(*height_edges(height), valid), 1e-12)

    def test_webui_accepts_independent_gains_and_disables_pore_fill(self):
        from pathlib import Path
        import json
        import webui
        cfg = json.loads(Path('displacement.example.json').read_text(encoding='utf-8-sig'))
        cfg['semantic_layers']['enabled'] = True
        cfg['layers']['gains'] = {'fold': 2.2, 'wrinkle': .75, 'fine': .4, 'lips': 1.}
        cfg['protection']['fill_pores'] = True
        out = webui.validate(cfg)
        self.assertTrue(out['semantic_layers']['enabled'])
        self.assertFalse(out['protection']['fill_pores'])
        self.assertEqual(set(out['layers']['gains']), {'fold', 'wrinkle', 'fine', 'lips'})


if __name__ == '__main__':
    unittest.main()
