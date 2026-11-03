"""Regression tests for actual failure modes, without private field data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "v4" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v4 = module("label_v4", "label_v4.py")
export = module("export_locations", "export_locations.py")


class Regressions(unittest.TestCase):
    def test_microsecond_storage_uses_real_tolerance_and_no_gap_bridge(self):
        start = pd.Timestamp("2026-01-01")
        frames = []
        # A has a missing peer at 2h: its nearest observation is 60min away.
        # B's rejected point is 40min away. Treating us as ns accepts both wrongly.
        for device, hours, measures in [
            ("A", [0, 1, 2, 3], [1., 2., 3., 4.]),
            ("B", [0, 1, 3, 3 + 40/60], [0., 0., 0., 0.]),
        ]:
            frames.append(pd.DataFrame({
                "device_id": device,
                "Timestamp": pd.Series([start + pd.Timedelta(hours=h) for h in hours],
                                       dtype="datetime64[us]"),
                "Measure": measures, "peer_group": "g", "humidity_bin": 0,
                "peer_tolerance_hours": .5, "env_residual_z": [0., 1., 2., 4.],
            }))
        result, matches, _ = v4.peers(pd.concat(frames, ignore_index=True))
        own = result.loc[result.device_id == "A"].reset_index(drop=True)
        own_matches = matches.loc[matches.device_id == "A"]
        np.testing.assert_allclose(own_matches.offset_minutes, [0., 0., -60., 0.])
        self.assertEqual(own_matches.accepted.tolist(), [1, 1, 0, 1])
        self.assertTrue(pd.isna(own.peer_delta.iloc[2]))
        self.assertTrue(pd.isna(own.peer_delta_jump_z.iloc[3]))
        rejected = matches.loc[(matches.device_id == "B") & (matches.accepted == 0)]
        self.assertEqual(len(rejected), 1)
        self.assertAlmostEqual(rejected.offset_minutes.iloc[0], -40.)
        self.assertTrue(rejected.peer_measure.isna().all())

    def test_only_explicit_device_locations_leave_ini(self):
        with tempfile.TemporaryDirectory() as temp:
            ini = Path(temp) / "config.ini"
            ini.write_text(
                "[DEFAULT]\npassword=DO_NOT_EXPORT\n"
                "111111111111111=INHERITED_NOT_LOCATION\n"
                "[DeviceLocations]\n000000000000001=example location\n",
                encoding="utf-8",
            )
            self.assertEqual(export.extract_locations(ini),
                             {"000000000000001": "example location"})


if __name__ == "__main__":
    unittest.main()
