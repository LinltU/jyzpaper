import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.evaluate_diffusion_augmented_fmlp import (
    downstream_sample_weights,
    generation_vector_layout,
    match_marginal_moments,
    pack_generation_vectors,
)
from experiments.evaluate_diffusion_augmented_high_expert import select_quality_synthetic
from experiments.evaluate_diffusion_augmented_high_expert import tail_risk_multipliers


class DiffusionAugmentationTests(unittest.TestCase):
    def test_generation_vector_pack_includes_residual(self):
        columns = [
            "Measure", "Temp", "Hum", "Battery", "DeltaHours", "AH"
        ]
        windows = np.asarray([
            [[0.1, 20.0, 70.0, 90.0, 12.0, 10.0],
             [0.2, 21.0, 72.0, 89.0, 12.0, 11.0]],
        ], dtype=np.float32)
        targets = np.asarray([0.5], dtype=np.float32)
        metadata = [{
            "IMEI": "A",
            "Temp": 22.0,
            "Hum": 81.0,
            "Persistence": 0.2,
        }]

        vectors = pack_generation_vectors(
            windows, targets, metadata, columns
        )
        layout = generation_vector_layout(seq_len=2)

        self.assertEqual(tuple(vectors.shape), (1, 13))
        self.assertAlmostEqual(vectors[0, layout["target_temp"]], 22.0)
        self.assertAlmostEqual(vectors[0, layout["target_hum"]], 81.0)
        self.assertAlmostEqual(vectors[0, layout["target_residual"]], 0.3)

    def test_moment_matching_restores_reference_marginals(self):
        generated = np.asarray([
            [0.0, 2.0], [1.0, 3.0], [2.0, 4.0], [3.0, 5.0]
        ], dtype=np.float32)
        reference = np.asarray([
            [10.0, -5.0], [14.0, -1.0], [18.0, 3.0], [22.0, 7.0]
        ], dtype=np.float32)

        matched = match_marginal_moments(generated, reference, strength=1.0)

        np.testing.assert_allclose(
            matched.mean(axis=0), reference.mean(axis=0), atol=1e-6
        )
        np.testing.assert_allclose(
            matched.std(axis=0), reference.std(axis=0), atol=1e-6
        )

    def test_synthetic_weight_reduces_only_synthetic_samples(self):
        metadata = [
            {"Hum": 45.0},
            {"Hum": 85.0, "Synthetic": True},
        ]

        weights = downstream_sample_weights(metadata, synthetic_loss_weight=0.25)

        np.testing.assert_allclose(weights, np.asarray([1.1, 0.6]))

    def test_synthetic_quality_scales_global_weight(self):
        metadata = [
            {"Hum": 85.0, "Synthetic": True, "SyntheticQuality": 0.4},
        ]

        weights = downstream_sample_weights(metadata, synthetic_loss_weight=0.5)

        np.testing.assert_allclose(weights, np.asarray([0.48]))

    def test_quality_selection_prefers_realistic_candidates(self):
        seq_len = 2
        real = np.asarray([
            [0.1, 20.0, 82.0, 12.0, 0.2, 21.0, 84.0, 12.0,
             22.0, 85.0, 12.0, 0.1],
            [0.2, 21.0, 88.0, 12.0, 0.3, 22.0, 90.0, 12.0,
             23.0, 91.0, 12.0, 0.2],
            [0.3, 22.0, 94.0, 12.0, 0.4, 23.0, 96.0, 12.0,
             24.0, 97.0, 12.0, 0.3],
            [0.4, 23.0, 96.0, 12.0, 0.5, 24.0, 98.0, 12.0,
             25.0, 99.0, 12.0, 0.4],
        ], dtype=np.float32)
        candidates = np.concatenate([
            real + 0.05,
            real + np.asarray([10.0] * 8 + [0.0, 0.0, 0.0, 5.0], dtype=np.float32),
        ])
        scaler = __import__("sklearn.preprocessing").preprocessing.StandardScaler()
        scaler.fit(real)
        x = np.zeros((len(candidates), seq_len, 1), dtype=np.float32)
        y = np.arange(len(candidates), dtype=np.float32)
        metadata = [
            {"SyntheticID": str(index)} for index in range(len(candidates))
        ]

        _, _, selected_metadata, _, diagnostics = select_quality_synthetic(
            real, x, y, metadata, candidates, scaler, seq_len,
            select_count=4, minimum_quality=0.25,
        )

        self.assertEqual(len(selected_metadata), 4)
        self.assertLess(
            diagnostics["manifold_distance_ratio"]["selected_median"],
            diagnostics["manifold_distance_ratio"]["candidate_median"],
        )
        self.assertTrue(all(
            0.25 <= item["SyntheticQuality"] <= 1.0
            for item in selected_metadata
        ))

    def test_tail_risk_threshold_uses_only_real_very_humid_samples(self):
        targets = np.asarray([0.1, 0.3, 0.9, 5.0], dtype=np.float32)
        metadata = [
            {"Hum": 85.0, "Persistence": 0.0},
            {"Hum": 85.0, "Persistence": 0.0},
            {"Hum": 85.0, "Persistence": 0.0},
            {
                "Hum": 85.0,
                "Persistence": 0.0,
                "Synthetic": True,
            },
        ]

        multipliers, diagnostics = tail_risk_multipliers(
            targets, metadata, alpha=2.0 / 3.0, strength=1.0
        )

        self.assertAlmostEqual(
            diagnostics["threshold_absolute_persistence_residual"],
            0.5,
            places=6,
        )
        np.testing.assert_allclose(multipliers, [1.0, 1.0, 2.0, 2.0])
        self.assertEqual(diagnostics["real_tail_count"], 1)
        self.assertEqual(diagnostics["synthetic_tail_count"], 1)


if __name__ == "__main__":
    unittest.main()
