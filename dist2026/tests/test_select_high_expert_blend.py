import unittest

import numpy as np

from experiments.select_high_expert_blend import simplex_weights


class HighExpertBlendTests(unittest.TestCase):
    def test_two_expert_grid_includes_endpoints(self):
        weights = list(simplex_weights(2, step=0.05))
        self.assertEqual(len(weights), 21)
        np.testing.assert_allclose(weights[0], [0.0, 1.0])
        np.testing.assert_allclose(weights[-1], [1.0, 0.0])

    def test_three_expert_grid_is_a_simplex(self):
        weights = list(simplex_weights(3, step=0.05))
        self.assertEqual(len(weights), 231)
        self.assertTrue(all(np.isclose(item.sum(), 1.0) for item in weights))
        self.assertTrue(all(np.all(item >= 0.0) for item in weights))

    def test_rejects_more_than_three_experts(self):
        with self.assertRaises(ValueError):
            list(simplex_weights(4))


if __name__ == "__main__":
    unittest.main()
