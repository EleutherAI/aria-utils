"""Miscellaneous utilities."""

import json
import logging

from importlib import resources
from pathlib import Path
from functools import lru_cache
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


@lru_cache(maxsize=1)
def load_maestro_metadata_json() -> dict[str, Any]:
    """Loads MAESTRO metadata json ."""
    with (
        resources.files("ariautils.config")
        .joinpath("maestro_metadata.json")
        .open("r") as f
    ):
        return cast(dict[str, Any], json.load(f))


@lru_cache(maxsize=1)
def load_aria_midi_metadata_json(
    metadata_load_path: Path | str | None = None,
) -> dict[int, dict[str, Any]]:
    """Loads MAESTRO metadata json."""
    if metadata_load_path is None:
        metadata_load_path = Path(
            str(
                resources.files("ariautils.config").joinpath(
                    "aria_midi_metadata.json"
                )
            )
        )
    with open(str(metadata_load_path), "r") as f:
        return {
            int(k): v
            for k, v in cast(dict[int, dict[str, Any]], json.load(f)).items()
        }


__all__ = [
    "load_config",
    "load_maestro_metadata_json",
    "load_aria_midi_metadata_json",
    "get_logger",
]
