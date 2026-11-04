import unittest

import torch
import numpy as np

from model.history_conditioned_residual_ddpm import (
    HistoryConditionedResidualDDPM,
)
from experiments.evaluate_history_conditioned_residual_diffusion import (
    trend_initialized_baseline,
)


class HistoryConditionedResidualDDPMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = HistoryConditionedResidualDDPM(
            history_dim=12,
            target_context_dim=5,
            num_stages=4,
            num_devices=3,
            hidden_dim=32,
            depth=2,
            diffusion_steps=6,
        )
        self.history = torch.randn(4, 12)
        self.target = torch.randn(4, 5)
        self.stages = torch.tensor([0, 1, 2, 3])
        self.devices = torch.tensor([0, 1, 2, 0])

    def test_loss_is_finite_and_backpropagates(self):
        residual = torch.randn(4, 1)
        loss = self.model.loss(
            residual,
            self.history,
            self.target,
            self.stages,
            self.devices,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in self.model.parameters()
        ))

    def test_sample_shape(self):
        sample = self.model.sample(
            self.history,
            self.target,
            self.stages,
            self.devices,
        )
        self.assertEqual(tuple(sample.shape), (4, 1))
        self.assertTrue(torch.isfinite(sample).all())

    def test_rejects_mismatched_history_shape(self):
        with self.assertRaises(ValueError):
            self.model.loss(
                torch.randn(4, 1),
                torch.randn(4, 11),
                self.target,
                self.stages,
                self.devices,
            )

    def test_trend_baseline_is_bounded(self):
        windows = np.zeros((2, 3, 3), dtype=np.float32)
        windows[:, -1, 0] = [1.0, 2.0]
        windows[:, -1, 1] = [10.0, -10.0]
        metadata = [
            {"Persistence": 1.0, "ForecastDeltaHours": 12.0},
            {"Persistence": 2.0, "ForecastDeltaHours": 12.0},
        ]
        baseline = trend_initialized_baseline(
            windows,
            metadata,
            ["Measure", "MeasureRate", "DeltaHours"],
            trend_strength=0.5,
            trend_cap=0.4,
        )
        np.testing.assert_allclose(baseline, [1.2, 1.8], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
