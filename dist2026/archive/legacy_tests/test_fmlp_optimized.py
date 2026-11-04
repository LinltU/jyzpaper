import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.fmlp import FMLP
from model.fmlp_ltsf import FMLP_LTSF
from train_unified import create_sequences_by_device, split_by_device_time


class FMLPModelTests(unittest.TestCase):
    def test_tabular_fmlp_supports_standardized_negative_targets(self):
        model = FMLP(input_dim=5, hidden_dim=8, dropout=0.0)
        with torch.no_grad():
            model.output_layer.weight.zero_()
            model.output_layer.bias.fill_(-1.0)

        output = model(torch.randn(4, 5))

        self.assertEqual(tuple(output.shape), (4,))
        self.assertTrue(torch.allclose(output, torch.full((4,), -1.0)))

    def test_ltsf_uses_exogenous_channels(self):
        torch.manual_seed(7)
        model = FMLP_LTSF(
            seq_len=6,
            pred_len=2,
            enc_in=3,
            hidden_dim=8,
            dropout=0.0,
            kernel_size=3,
            use_mlp=False,
            target_index=0,
        )
        x = torch.randn(2, 6, 3, requires_grad=True)

        output = model(x)
        output.sum().backward()

        self.assertEqual(tuple(output.shape), (2, 2, 1))
        self.assertGreater(x.grad[:, :, 1:].abs().sum().item(), 0.0)

    def test_ltsf_rejects_invalid_target_index(self):
        with self.assertRaises(ValueError):
            FMLP_LTSF(seq_len=6, pred_len=2, enc_in=3, target_index=3)


class TimeSplitTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for device_num, imei in enumerate(["A", "B"]):
            for step in range(20):
                rows.append({
                    "IMEI": imei,
                    "Timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=step),
                    "Measure": float(step + device_num),
                    "Hum": 40.0 + step,
                })
        self.df = pd.DataFrame(rows)

    def test_split_is_chronological_per_device_with_context(self):
        train, val, test = split_by_device_time(
            self.df,
            train_ratio=0.5,
            val_ratio=0.25,
            seq_len=3,
        )

        for imei in ["A", "B"]:
            train_group = train[train["IMEI"] == imei]
            val_group = val[val["IMEI"] == imei]
            test_group = test[test["IMEI"] == imei]

            self.assertEqual(len(train_group), 10)
            self.assertEqual(len(val_group), 8)
            self.assertEqual(len(test_group), 8)
            self.assertLess(train_group["Timestamp"].max(), val_group["Timestamp"].iloc[3])
            self.assertLess(val_group["Timestamp"].max(), test_group["Timestamp"].iloc[3])

    def test_sequence_index_points_to_last_forecast_step(self):
        features = ["Measure", "Hum"]
        _, y, _, indices = create_sequences_by_device(
            self.df,
            feature_cols=features,
            seq_len=3,
            pred_len=2,
        )

        first_target_index = indices[0]
        self.assertEqual(self.df.loc[first_target_index, "Measure"], y[0, -1])
        self.assertTrue(np.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
