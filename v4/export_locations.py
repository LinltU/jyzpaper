"""Extract only explicitly declared DeviceLocations; never inherit DEFAULT secrets."""
import argparse
import configparser
import json
from pathlib import Path
import re


def extract_locations(ini: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(ini, encoding="utf-8-sig"):
        raise FileNotFoundError(ini)
    # SectionProxy.items() includes DEFAULT. Use only raw, explicit section entries.
    explicit = parser._sections.get("DeviceLocations", {})
    locations = {
        k: v for k, v in explicit.items()
        if re.fullmatch(r"\d{15}", k) and isinstance(v, str)
    }
    if not locations:
        raise ValueError("No explicit 15-digit device IDs in [DeviceLocations].")
    return locations


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--ini", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    mapping = extract_locations(args.ini)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(mapping)} explicit device locations. Keep the JSON private.")
