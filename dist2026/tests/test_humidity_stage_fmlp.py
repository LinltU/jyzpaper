import os
import sys
import unittest

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.humidity_stage_fmlp import HumidityStageFMLP


class HumidityStageFMLPTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(11)
        return HumidityStageFMLP(
            seq_len=8,
            pred_len=2,
            enc_in=5,
            hidden_dim=16,
            expert_hidden_dim=8,
            dropout=0.0,
            kernel_size=3,
            target_index=0,
            humidity_index=2,
        )

    def test_output_shapes_and_ordered_quantiles(self):
        model = self.make_model()
        x = torch.randn(4, 8, 5)
        routes = torch.tensor([
            [35.0, 1.0],
            [57.0, -2.0],
            [72.0, 3.0],
            [88.0, 5.0],
        ])

        output = model(x, routes)
        quantiles = output["quantiles"]

        self.assertEqual(tuple(quantiles.shape), (4, 2, 3))
        self.assertEqual(tuple(output["gates"].shape), (4, 4))
        self.assertTrue(torch.all(quantiles[:, :, 0] <= quantiles[:, :, 1]))
        self.assertTrue(torch.all(quantiles[:, :, 1] <= quantiles[:, :, 2]))
        self.assertTrue(
            torch.allclose(output["gates"].sum(dim=1), torch.ones(4), atol=1e-6)
        )

    def test_initial_routing_matches_humidity_stages(self):
        model = self.make_model()
        x = torch.randn(4, 8, 5)
        routes = torch.tensor([
            [35.0, 0.0],
            [57.0, 0.0],
            [72.0, 0.0],
            [90.0, 0.0],
        ])

        gates = model(x, routes)["gates"]

        self.assertEqual(gates.argmax(dim=1).tolist(), [0, 1, 2, 3])

    def test_each_expert_receives_gradient(self):
        model = self.make_model()
        x = torch.randn(4, 8, 5)
        routes = torch.tensor([
            [35.0, 1.0],
            [57.0, -1.0],
            [72.0, 2.0],
            [88.0, 4.0],
        ])

        loss = model(x, routes)["quantiles"].sum()
        loss.backward()

        for expert in model.expert_heads:
            gradient = sum(
                parameter.grad.abs().sum().item()
                for parameter in expert.parameters()
                if parameter.grad is not None
            )
            self.assertGreater(gradient, 0.0)

    def test_invalid_routing_shape_is_rejected(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 8, 5), torch.randn(2, 3))

    def test_device_embedding_and_known_covariates(self):
        model = HumidityStageFMLP(
            seq_len=8,
            pred_len=1,
            enc_in=5,
            hidden_dim=16,
            expert_hidden_dim=8,
            dropout=0.0,
            kernel_size=3,
            target_index=0,
            humidity_index=2,
            num_devices=3,
            device_embedding_dim=4,
            known_covariate_dim=4,
        )
        x = torch.randn(3, 8, 5)
        routes = torch.tensor([[45.0, 1.0], [68.0, 2.0], [84.0, -1.0]])
        devices = torch.tensor([0, 1, 3])
        known = torch.randn(3, 4)

        output = model(
            x,
            routes,
            device_indices=devices,
            known_covariates=known,
        )

        self.assertEqual(tuple(output["quantiles"].shape), (3, 1, 3))
        with self.assertRaises(ValueError):
            model(x, routes, device_indices=devices)


if __name__ == "__main__":
    unittest.main()
