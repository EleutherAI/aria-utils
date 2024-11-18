"""Includes functionality for loading config files."""

import json

from importlib import resources
from typing import Any, cast


def load_config() -> dict[str, Any]:
    """Returns a dictionary loaded from the config.json file."""
    with (
        resources.files("ariautils.config")
        .joinpath("config.json")
        .open("r") as f
    ):
        return cast(dict[str, Any], json.load(f))
