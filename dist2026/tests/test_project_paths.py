import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.project_paths import manifest_relative_path, resolve_project_path


class ProjectPathTests(unittest.TestCase):
    def test_resolves_path_relative_to_manifest_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "models" / "model.pth"
            manifest = root / "configs" / "ensemble_manifest.json"
            checkpoint.parent.mkdir()
            manifest.parent.mkdir()
            checkpoint.touch()
            relative = manifest_relative_path(checkpoint, manifest)
            manifest.write_text(
                json.dumps({"base_checkpoint": relative}), encoding="utf-8"
            )

            self.assertEqual(
                resolve_project_path(relative, relative_to=manifest.parent),
                checkpoint.resolve(),
            )

    def test_current_production_manifest_uses_relative_existing_paths(self):
        manifest_path = (
            ROOT
            / "humidity_stage_feature_ensemble_evaluation"
            / "ensemble_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("base_checkpoint", "high_humidity_checkpoint"):
            value = manifest[key]
            self.assertFalse(Path(value).is_absolute())
            self.assertTrue(
                resolve_project_path(value, relative_to=manifest_path.parent).is_file()
            )


if __name__ == "__main__":
    unittest.main()
