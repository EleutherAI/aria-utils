"""Tests for tokenizers."""

import unittest
import copy

from importlib import resources
from pathlib import Path
from typing import Final

from ariautils.midi import MidiDict, normalize_midi_dict
from ariautils.tokenizer import AbsTokenizer
from ariautils.utils import get_logger


TEST_DATA_DIRECTORY: Final[Path] = Path(
    str(resources.files("tests").joinpath("assets", "data"))
)
RESULTS_DATA_DIRECTORY: Final[Path] = Path(
    str(resources.files("tests").joinpath("assets", "results"))
)


class TestAbsTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = get_logger(__name__ + ".TestAbsTokenizer")

    def test_normalize_midi_dict(self) -> None:
        def _test_normalize_midi_dict(
            _load_path: Path, _save_path: Path
        ) -> None:
            tokenizer = AbsTokenizer()
            midi_dict = MidiDict.from_midi(_load_path)
            midi_dict_copy = copy.deepcopy(midi_dict)

            normalized_midi_dict = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step,
                max_duration_ms=tokenizer.max_dur,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )
            normalized_twice_midi_dict = normalize_midi_dict(
                normalized_midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step,
                max_duration_ms=tokenizer.max_dur,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )
            self.assertDictEqual(
                normalized_midi_dict.get_msg_dict(),
                normalized_twice_midi_dict.get_msg_dict(),
            )
            self.assertDictEqual(
                midi_dict.get_msg_dict(),
                midi_dict_copy.get_msg_dict(),
            )
            normalized_midi_dict.to_midi().save(_save_path)

        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("arabesque_norm.mid")
        _test_normalize_midi_dict(load_path, save_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("pop_norm.mid")
        _test_normalize_midi_dict(load_path, save_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("basic_norm.mid")
        _test_normalize_midi_dict(load_path, save_path)

    def test_tokenize_detokenize(self) -> None:
        def _test_tokenize_detokenize(_load_path: Path) -> None:
            tokenizer = AbsTokenizer()
            midi_dict = MidiDict.from_midi(_load_path)

            midi_dict_1 = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step,
                max_duration_ms=tokenizer.max_dur,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            midi_dict_2 = normalize_midi_dict(
                midi_dict=tokenizer.detokenize(
                    tokenizer.tokenize(
                        midi_dict_1, remove_preceding_silence=False
                    )
                ),
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step,
                max_duration_ms=tokenizer.max_dur,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            self.assertDictEqual(
                midi_dict_1.get_msg_dict(),
                midi_dict_2.get_msg_dict(),
            )

        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        _test_tokenize_detokenize(_load_path=load_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        _test_tokenize_detokenize(_load_path=load_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        _test_tokenize_detokenize(_load_path=load_path)
