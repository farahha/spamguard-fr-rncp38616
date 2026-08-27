from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """Charge la configuration YAML du projet."""
    config_path = path or PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
