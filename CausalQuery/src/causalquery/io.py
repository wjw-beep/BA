import json
from pathlib import Path

import yaml


def read_data(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(stream)
        if path.suffix.lower() == ".json":
            return json.load(stream)
    raise ValueError(f"Unsupported input format: {path}")


def write_data(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        if path.suffix.lower() in {".yaml", ".yml"}:
            yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)
        elif path.suffix.lower() == ".json":
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        else:
            raise ValueError(f"Unsupported output format: {path}")
