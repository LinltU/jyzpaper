import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.humidity_conditional_ddpm import ConditionalDDPM


class ConditionalDDPMTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(7)
        return ConditionalDDPM(
            input_dim=6,
            num_stages=4,
            num_devices=3,
            hidden_dim=24,
            depth=2,
            dropout=0.0,
            diffusion_steps=8,
        )

    def test_training_loss_is_finite_and_backpropagates(self):
        model = self.make_model()
        clean = torch.randn(5, 6)
        stages = torch.tensor([0, 1, 2, 3, 3])
        devices = torch.tensor([0, 1, 2, 0, 1])

        loss = model.loss(clean, stages, devices)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_q_sample_and_generated_sample_shapes(self):
        model = self.make_model()
        clean = torch.randn(4, 6)
        timesteps = torch.tensor([0, 2, 4, 7])
        noisy = model.q_sample(clean, timesteps, noise=torch.zeros_like(clean))

        self.assertEqual(tuple(noisy.shape), (4, 6))
        generated = model.sample(
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 1, 2, 0]),
        )
        self.assertEqual(tuple(generated.shape), (4, 6))
        self.assertTrue(torch.isfinite(generated).all())

    def test_invalid_condition_shapes_are_rejected(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model.sample(torch.tensor([[0, 1]]), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            model.sample(torch.tensor([0, 1]), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
