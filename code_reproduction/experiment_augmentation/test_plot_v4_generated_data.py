"""Focused regression tests; run with python -m unittest test_plot_v4_generated_data."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import plot_v4_generated_data as plot


class PlotTests(unittest.TestCase):
    def test_selection_preserves_both_classes_and_is_deterministic(self):
        ids = np.arange(12)
        labels = np.array([0] * 8 + [1] * 4)
        sources = ids[::-1]
        selected = plot.select_windows(ids, labels, sources, 2)
        np.testing.assert_array_equal(selected, plot.select_windows(ids, labels, sources, 2))
        self.assertEqual(sum(labels[selected] == 0), 2)
        self.assertEqual(sum(labels[selected] == 1), 2)
        self.assertEqual(len(plot.select_windows(ids, labels, sources, 0)), 12)
        self.assertEqual(len(plot.select_windows(np.array([], dtype=int), labels, sources, 2)), 0)
        with self.assertRaises(ValueError):
            plot.select_windows(ids, labels, sources, -1)

    def test_interval_union_keeps_gaps(self):
        self.assertEqual(plot.merge_intervals([(3, 4), (0, 2), (1, 2), (0, 2), (4, 5)]), [(0, 2), (3, 5)])
        self.assertEqual(plot.merge_intervals([]), [])
        self.assertEqual(plot.merge_intervals([(1, 1)]), [(1, 1)])

    def test_windows_are_independent_and_negatives_visible(self):
        fig, ax = plot.plt.subplots()
        dates = np.array([[1, 2, 4, 5], [10, 11, 14, 18]])
        values = np.array([[0., 1., 2., 3.], [-4., 2., 1., 0.]])
        collection = plot.add_windows(ax, dates, values, plot.RED)
        self.assertEqual(len(collection.get_segments()), 2)
        np.testing.assert_array_equal(collection.get_segments()[1][:, 0], dates[1])
        plot.set_limits(ax, dates, values)
        self.assertLess(ax.get_ylim()[0], -4)
        self.assertGreater(ax.get_ylim()[1], 3)
        self.assertLess(ax.get_xlim()[0], 1)
        self.assertGreater(ax.get_xlim()[1], 18)
        plot.plt.close(fig)

    def context(self):
        times = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"])
        points = pd.DataFrame({"Timestamp": times, "Measure": [1., 2., 3., 4.], "v4_point_label": [0, 0, 1, 1]})
        meta = pd.DataFrame({"IMEI": ["001"], "label": [1], "source_rows": [[0, 1, 2, 3]]})
        return plot.Context({"ratios": {"2to8": .2}}, meta, {"001": points, "002": points.copy()}, {},
                            times.to_numpy()[None], np.ones((1, 4, 5)))

    def test_generated_validation_and_pickle_not_needed(self):
        ctx = self.context()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.npz"
            def write(**changes):
                fields = dict(windows=np.ones((1, 4, 5)), labels=np.array([1]), source_ids=np.array([0]),
                              source_imei=np.array(["001"], dtype=object))
                fields.update(changes)
                np.savez(path, **fields)
            write()
            values, labels, sources = plot.load_generated(path, ctx, "2to8")
            self.assertEqual(values.shape, (1, 4))
            for changes in [dict(source_ids=np.array([-1])), dict(source_ids=np.array([1])),
                            dict(source_ids=np.array([0.])), dict(labels=np.array([0])),
                            dict(labels=np.array([2])), dict(windows=np.full((1, 4, 5), np.nan)),
                            dict(windows=np.ones((1, 3, 5))), dict(labels=np.array([1, 1]))]:
                write(**changes)
                with self.assertRaises(ValueError):
                    plot.load_generated(path, ctx, "2to8")
            write()
            with patch.object(plot, "fit_ids", return_value=np.array([], dtype=int)):
                with self.assertRaisesRegex(ValueError, "fit/training"):
                    plot.load_generated(path, ctx, "2to8")

    def test_grid_no_positive_and_no_generated_device(self):
        ctx = self.context()
        ctx.meta["label"] = 0
        values = np.array([[-1., 0., 1., 2.]])
        with tempfile.TemporaryDirectory() as tmp:
            rows = plot.plot_grids(ctx, values, np.array([0]), np.array([0]), "test", Path(tmp), 0, 30, True)
            self.assertEqual(rows[0]["positive_windows"], 0)
            self.assertEqual(rows[0]["covered_real_points"], 4)
            self.assertEqual(rows[0]["nonpositive_measure_points"], 2)
            self.assertEqual(rows[1]["generated_windows"], 0)
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 3)
        self.assertEqual(plot.plt.get_fignums(), [])

    def test_fit_split_small_and_regular_devices(self):
        meta = pd.DataFrame({"IMEI": ["a"] * 20 + ["b"] * 4 + ["c"]})
        np.testing.assert_array_equal(plot.fit_ids(meta, .5), np.r_[np.arange(8), [20, 21, 24]])

    def test_real_context_checks_source_times_labels_and_leading_zero_imei(self):
        ctx = self.context()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            protocol = dict(ctx.protocol, features=["Measure", "a", "b", "c", "d"], window_length=4)
            (run / "protocol.json").write_text(json.dumps(protocol))
            p = ctx.points["001"].assign(IMEI="001")
            p.to_csv(run / "annotated_device_points_v4.csv", index=False)
            candidate = dict(IMEI="001", window_anomaly_v4=1, start_row=2, end_row=3,
                             start_time=str(p.Timestamp.iloc[2]), end_time=str(p.Timestamp.iloc[3]))
            pd.DataFrame([candidate]).to_csv(run / "v4_window_labels.csv", index=False)
            row = dict(IMEI="001", label=1, source_rows=[0, 1, 2, 3],
                       start_time=str(p.Timestamp.iloc[0]), end_time=str(p.Timestamp.iloc[3]))
            metadata_path = run / "window_metadata_v4.json"
            metadata_path.write_text(json.dumps([row]))
            real = np.ones((1, 4, 5))
            real[0, :, 0] = p.Measure
            np.savez(run / "window_dataset_v4.npz", x=real, y=np.array([1]))
            actual = plot.load_context(run)
            self.assertIn("001", actual.points)
            np.testing.assert_array_equal(actual.times, ctx.times)
            for changes in [dict(start_time="2025-01-01"), dict(source_rows=[0, 1, 2, 8]),
                            dict(source_rows=[0, 1, 3, 2]), dict(label=0), dict(IMEI="999")]:
                metadata_path.write_text(json.dumps([dict(row, **changes)]))
                with self.assertRaises(ValueError):
                    plot.load_context(run)
            metadata_path.write_text(json.dumps([row]))
            pd.DataFrame([dict(candidate, window_anomaly_v4=0)]).to_csv(run / "v4_window_labels.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Candidate/point"):
                plot.load_context(run)


@unittest.skipUnless(os.environ.get("V4_PLOT_RUN"), "Set V4_PLOT_RUN for full artifact validation")
class ArtifactTests(unittest.TestCase):
    def test_all_exported_figures_counts_labels_and_raw_statistics(self):
        run = Path(os.environ["V4_PLOT_RUN"])
        output = run / "figures/generated_data"
        ctx = plot.load_context(run)
        protocol = json.loads((output / "plot_protocol.json").read_text())
        summary = pd.read_csv(output / "configuration_summary.csv")
        audit = pd.read_csv(output / "device_generation_audit.csv", dtype={"IMEI": str})
        expected = len(protocol["ratios"]) * len(protocol["methods"]) * len(protocol["seeds"])
        self.assertEqual(len(summary), expected)
        self.assertEqual(len(audit), expected * len(ctx.points))
        self.assertFalse(summary.duplicated(["ratio", "method", "seed"]).any())
        generated_devices = ctx.meta.IMEI.to_numpy()
        for row in summary.itertuples():
            safe = row.method.replace("/", "_").replace("-", "_")
            raw = run / f"raw_{row.ratio}_{safe}_seed{row.seed}.npz"
            values, labels, sources = plot.load_generated(raw, ctx, row.ratio)
            self.assertEqual(row.generated_windows, len(values))
            self.assertEqual(row.positive_windows, int(labels.sum()))
            self.assertEqual(row.nonpositive_measure_points, int((values <= 0).sum()))
            self.assertEqual(row.measure_points, values.size)
            self.assertAlmostEqual(row.nonpositive_fraction, (values <= 0).mean())
            rows = audit[(audit.ratio == row.ratio) & (audit.method == row.method) & (audit.seed == row.seed)]
            self.assertEqual(set(rows.IMEI), set(ctx.points))
            for stat in rows.itertuples():
                ids = np.flatnonzero(generated_devices[sources] == stat.IMEI)
                self.assertEqual(stat.generated_windows, len(ids))
                self.assertEqual(stat.positive_windows, int(labels[ids].sum()))
                self.assertEqual(stat.nonpositive_measure_points, int((values[ids] <= 0).sum()))
                self.assertEqual(stat.measure_points, values[ids].size)
                if protocol["max_windows_per_device_class"] == 0:
                    self.assertEqual(stat.plotted_overlay_windows, len(ids))
                self.assertLessEqual(stat.point_coverage_fraction, 1)
            images = [output / row.overlay, output / row.localization]
            if protocol["details"]:
                images.append(images[0].parent / "local_window_comparison_by_device.png")
            for image in images:
                pixels = plot.plt.imread(image)
                self.assertGreater(pixels.shape[0], 2500)
                self.assertGreater(pixels.shape[1], 3500)
                self.assertGreater(pixels[:, :, :3].std(), .03)
        self.assertEqual(len(list(output.glob("*/*/*.png"))), expected * (3 if protocol["details"] else 2))

    def test_all_experiment_seeds_have_valid_provenance(self):
        run = Path(os.environ["V4_PLOT_RUN"])
        ctx = plot.load_context(run)
        count = 0
        for ratio in ctx.protocol["ratios"]:
            for method in ctx.protocol["methods"]:
                safe = method.replace("/", "_").replace("-", "_")
                for seed in ctx.protocol["seeds"]:
                    plot.load_generated(run / f"raw_{ratio}_{safe}_seed{seed}.npz", ctx, ratio)
                    count += 1
        self.assertEqual(count, len(ctx.protocol["ratios"]) * len(ctx.protocol["methods"]) * len(ctx.protocol["seeds"]))


if __name__ == "__main__":
    unittest.main()
