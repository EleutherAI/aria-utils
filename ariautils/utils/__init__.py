"""Miscellaneous utilities."""

import json
import logging

from importlib import resources
from typing import Any, cast

from .config import load_config


def get_logger(name: str | None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        if name is not None:
            formatter = logging.Formatter(
                "[%(asctime)s]: [%(levelname)s] [%(name)s] %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "[%(asctime)s]: [%(levelname)s] %(message)s"
            )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def load_maestro_metadata_json() -> dict[str, Any]:
    """Loads MAESTRO metadata json ."""
    with (
        resources.files("ariautils.config")
        .joinpath("maestro_metadata.json")
        .open("r") as f
    ):
        return cast(dict[str, Any], json.load(f))


__all__ = ["load_config", "load_maestro_metadata_json", "get_logger"]
