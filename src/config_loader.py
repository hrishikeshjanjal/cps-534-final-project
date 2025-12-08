import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(path: Union[str, os.PathLike] = "configs/config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    return config
