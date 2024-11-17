"""Includes functionality for loading config files."""

import os
import json

from importlib import resources
from typing import Dict, Any, cast


def load_config() -> Dict[str, Any]:
    """Returns a dictionary loaded from the config.json file."""
    with (
        resources.files("ariautils.config")
        .joinpath("config.json")
        .open("r") as f
    ):
        return cast(Dict[str, Any], json.load(f))


# TODO: Move somewhere else
def load_maestro_metadata_json() -> Dict[str, Any]:
    """Loads MAESTRO metadata json ."""
    with (
        resources.files("ariautils.config")
        .joinpath("maestro_metadata.json")
        .open("r") as f
    ):
        return cast(Dict[str, Any], json.load(f))
