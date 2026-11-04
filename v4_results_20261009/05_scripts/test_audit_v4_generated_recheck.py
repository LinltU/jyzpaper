import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
from audit_v4_generated_recheck import RankReference, rules, Rechecker, DEFAULT_RUN, summary


class UnitTests(unittest.TestCase):
    def test_crossing_boundary_and_tail(self):
        c = Rechecker.__new__(Rechecker)
        c.rscale = c.pscale = (0, 1)
        n = 34
        p = pd.DataFrame({'peer_delta': np.zeros(n), 'peer_complete': np.ones(n, bool)})
        c.devices = {'toy': {'p': p, 'peer_scale': 1}}
        rz = np.zeros((1, n))
        pz = np.zeros((1, n))
        delta = np.zeros((1, n))
        delta[0, 32] = 20
        labels, reasons = c.classify('toy', rz, pz, delta)
        np.testing.assert_array_equal(labels, [[False, True]])
        self.assertEqual(reasons[0, 1], 'repeated_peer_jump')
        # Last one-point tail must be omitted, not silently resized to 32.
        c.devices['toy']['p'] = p.iloc[:33]
        labels, _ = c.classify('toy', rz[:, :33], pz[:, :33], delta[:, :33])
        self.assertEqual(labels.shape, (1, 1))

    def test_incomplete_peer_panel_does_not_create_jump(self):
        c = Rechecker.__new__(Rechecker)
        c.rscale = c.pscale = (0, 1)
        p = pd.DataFrame({'peer_delta': [0, 100, 0, 0], 'peer_complete': [True, False, True, True]})
        c.devices = {'toy': {'p': p, 'peer_scale': 1}}
        zero = np.zeros((1, 4))
        labels, _ = c.classify('toy', zero, zero, np.array([[0, 100, 0, 0]]))
        np.testing.assert_array_equal(labels, [[False]])

    def test_average_ties_roundtrip(self):
        x = np.array([0., 1., 1., 4., 9.])
        expected = norm.ppf(np.clip(pd.Series(x).rank(method='average', pct=True), 1e-4, 1-1e-4))
        np.testing.assert_allclose(RankReference(x).transform(x), expected)

    def test_extremes_and_monotonicity(self):
        out = RankReference([0, 0, 1, 2]).transform([-100, 0, .5, 1, 2, 100])
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((np.diff(out) >= 0).all())
        with self.assertRaises(ValueError):
            RankReference([np.nan])

    def test_rule_boundaries(self):
        result, reasons = rules(np.array([3, 3, 0, 0, 0]), np.array([3, -3, 0, 0, np.nan]),
                                np.array([.15, .15, 0, 0, .15]), np.array([2, 2, 0, 0, 2]),
                                np.array([0, 0, 0, 0, 2]), np.array([0, 0, 2, 0, 0]),
                                np.array([0, 0, 8, 12, np.nan]), np.array([1, 1, 1, 1, 0], bool), (0, 1), (0, 1))
        np.testing.assert_array_equal(result, [True, False, True, True, True])
        self.assertEqual(reasons[-1], 'no_peer_residual')


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = Rechecker(DEFAULT_RUN)

    def test_full_real_roundtrip(self):
        self.assertEqual(self.checker.real_roundtrip_count, 16910)
        self.assertEqual(int(self.checker.windows.window_anomaly_v4.sum()), 38)

    def test_best_method_counts(self):
        c = self.checker
        frame = c.file(DEFAULT_RUN / 'raw_4to6_IMDiffusion_seed42.npz', '4to6')
        stats = summary(frame)
        self.assertEqual(stats['windows'], 5408)
        self.assertEqual(stats['positive_to_normal'] + stats['positive_to_candidate'], 488)
        self.assertEqual(sum(stats[k] for k in ['normal_to_normal', 'normal_to_candidate', 'positive_to_normal', 'positive_to_candidate']), 5408)
        self.assertTrue(frame.rechecked_label.isin([0, 1]).all())
        self.assertTrue(frame.standalone_v4_status.str.startswith('not_evaluable').all())
        # Every sample is evaluated independently; duplicate ordering must not change results.
        row = frame.iloc[0]
        imei, sid = row.IMEI, int(row.source_id)
        rows = np.asarray(c.ctx.meta.iloc[sid].source_rows, int)
        with np.load(DEFAULT_RUN / 'raw_4to6_IMDiffusion_seed42.npz', allow_pickle=False) as z:
            val = z['windows'][0, :, 0]
        single = c.evaluate_device(imei, rows[None], val[None])
        double = c.evaluate_device(imei, np.tile(rows, (2, 1)), np.tile(val, (2, 1)))
        self.assertEqual(single['rechecked_label'][0], row.rechecked_label)
        np.testing.assert_array_equal(double['rechecked_label'], np.repeat(single['rechecked_label'], 2))


if __name__ == '__main__':
    unittest.main()
