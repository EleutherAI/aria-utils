"""Includes functionality for loading config files."""

import json

from pathlib import Path
from importlib import resources
from typing import Any, cast


def load_config(load_path: Path | str | None = None) -> dict[str, Any]:
    """Returns a dictionary loaded from the config.json file."""
    if load_path is not None:
        with open(load_path, "r") as f:
            return cast(dict[str, Any], json.load(f))
    else:
        with (
            resources.files("ariautils.config")
            .joinpath("config.json")
            .open("r") as f
        ):
            return cast(dict[str, Any], json.load(f))
