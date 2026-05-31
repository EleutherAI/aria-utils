"""Tests for MidiDict."""

import unittest
import tempfile
import shutil
import mido

from importlib import resources
from pathlib import Path
from typing import Final

from ariautils.midi import MidiDict
from ariautils.utils import get_logger


TEST_DATA_DIRECTORY: Final[Path] = Path(
    str(resources.files("tests").joinpath("assets", "data"))
)
RESULTS_DATA_DIRECTORY: Final[Path] = Path(
    str(resources.files("tests").joinpath("assets", "results"))
)


class TestMidiDict(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = get_logger(__name__ + ".TestMidiDict")

    def test_load(self) -> None:
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict = MidiDict.from_midi(load_path)

        self.logger.info(f"Num meta_msgs: {len(midi_dict.meta_msgs)}")
        self.logger.info(f"Num tempo_msgs: {len(midi_dict.tempo_msgs)}")
        self.logger.info(f"Num pedal_msgs: {len(midi_dict.pedal_msgs)}")
        self.logger.info(
            f"Num instrument_msgs: {len(midi_dict.instrument_msgs)}"
        )
        self.logger.info(f"Num note_msgs: {len(midi_dict.note_msgs)}")
        self.logger.info(f"ticks_per_beat: {midi_dict.ticks_per_beat}")
        self.logger.info(f"metadata: {midi_dict.metadata}")

    def test_save(self) -> None:
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict = MidiDict.from_midi(mid_path=load_path)
        midi_dict.to_midi().save(save_path)

    def test_tick_to_ms(self) -> None:
        CORRECT_LAST_NOTE_ONSET_MS: Final[int] = 220140
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict = MidiDict.from_midi(load_path)
        last_note = midi_dict.note_msgs[-1]
        last_note_onset_tick = last_note["tick"]
        last_note_onset_ms = midi_dict.tick_to_ms(last_note_onset_tick)
        self.assertEqual(last_note_onset_ms, CORRECT_LAST_NOTE_ONSET_MS)

    def test_calculate_hash(self) -> None:
        # Load two identical files with different filenames and metadata
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict_orig = MidiDict.from_midi(load_path)

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            shutil.copy(load_path, temp_file.name)
            midi_dict_temp = MidiDict.from_midi(temp_file.name)

        midi_dict_temp.meta_msgs.append({"type": "text", "data": "test"})
        midi_dict_temp.metadata["composer"] = "test"
        midi_dict_temp.metadata["composer"] = "test"
        midi_dict_temp.metadata["ticks_per_beat"] = -1

        self.assertEqual(
            midi_dict_orig.calculate_hash(), midi_dict_temp.calculate_hash()
        )

    def test_raw_pedal_values(self) -> None:
        expected_values = [0, 63, 64, 65, 127]
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        for value in expected_values:
            track.append(
                mido.Message(
                    "control_change",
                    control=64,
                    value=value,
                    time=0,
                    channel=0,
                )
            )

        with tempfile.NamedTemporaryFile(suffix=".mid") as temp_file:
            mid.save(temp_file.name)
            midi_dict = MidiDict.from_midi(temp_file.name)

        self.assertEqual(
            [msg["value"] for msg in midi_dict.pedal_msgs], expected_values
        )
        self.assertEqual(
            [msg["data"] for msg in midi_dict.pedal_msgs],
            [0, 0, 1, 1, 1],
        )

    def test_apply_pedal_threshold(self) -> None:
        midi_dict = MidiDict.from_msg_dict(
            {
                "meta_msgs": [],
                "tempo_msgs": [],
                "pedal_msgs": [
                    {
                        "type": "pedal",
                        "data": 0,
                        "value": 63,
                        "tick": 0,
                        "channel": 0,
                    },
                    {
                        "type": "pedal",
                        "data": 1,
                        "value": 72,
                        "tick": 1,
                        "channel": 0,
                    },
                    {
                        "type": "pedal",
                        "data": 1,
                        "value": 71,
                        "tick": 2,
                        "channel": 0,
                    },
                    {
                        "type": "pedal",
                        "data": 1,
                        "value": 127,
                        "tick": 3,
                        "channel": 0,
                    },
                ],
                "instrument_msgs": [],
                "note_msgs": [
                    {
                        "type": "note",
                        "data": {
                            "pitch": 60,
                            "start": 0,
                            "end": 50,
                            "velocity": 64,
                        },
                        "tick": 0,
                        "channel": 0,
                    }
                ],
                "ticks_per_beat": 480,
                "metadata": {},
            }
        )

        result = midi_dict.apply_pedal_threshold(72)

        self.assertIs(result, midi_dict)
        self.assertEqual(
            [msg["data"] for msg in midi_dict.pedal_msgs], [0, 1, 0, 1]
        )
        self.assertEqual(midi_dict.note_msgs[0]["data"]["end"], 50)

    def test_apply_pedal_threshold_buffer(self) -> None:
        midi_dict = MidiDict.from_msg_dict(
            {
                "meta_msgs": [],
                "tempo_msgs": [],
                "pedal_msgs": [
                    {
                        "type": "pedal",
                        "data": 0,
                        "value": value,
                        "tick": tick,
                        "channel": 0,
                    }
                    for tick, value in enumerate(
                        [0, 63, 72, 70, 57, 56, 65, 73, 60, 55]
                    )
                ],
                "instrument_msgs": [],
                "note_msgs": [
                    {
                        "type": "note",
                        "data": {
                            "pitch": 60,
                            "start": 0,
                            "end": 50,
                            "velocity": 64,
                        },
                        "tick": 0,
                        "channel": 0,
                    }
                ],
                "ticks_per_beat": 480,
                "metadata": {},
            }
        )

        result = midi_dict.apply_pedal_threshold(
            threshold=64, buffer=8, transitions_only=True
        )

        self.assertIs(result, midi_dict)
        self.assertEqual(
            [msg["tick"] for msg in midi_dict.pedal_msgs], [2, 5, 7, 9]
        )
        self.assertEqual(
            [msg["data"] for msg in midi_dict.pedal_msgs], [1, 0, 1, 0]
        )

    def test_resolve_pedal(self) -> None:
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath(
            "arabesque_pedal_resolved.mid"
        )
        midi_dict = MidiDict.from_midi(mid_path=load_path).resolve_pedal()
        midi_dict.to_midi().save(save_path)

    def test_remove_redundant_pedals(self) -> None:
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath(
            "arabesque_remove_redundant_pedals.mid"
        )
        midi_dict = MidiDict.from_midi(mid_path=load_path)
        self.logger.info(
            f"Num pedal_msgs before remove_redundant_pedals: {len(midi_dict.pedal_msgs)}"
        )

        midi_dict_adj_resolve = (
            MidiDict.from_midi(mid_path=load_path)
            .resolve_pedal()
            .remove_redundant_pedals()
        )
        midi_dict_resolve_adj = (
            MidiDict.from_midi(mid_path=load_path)
            .remove_redundant_pedals()
            .resolve_pedal()
        )

        self.logger.info(
            f"Num pedal_msgs after remove_redundant_pedals: {len(midi_dict_adj_resolve.pedal_msgs)}"
        )
        self.assertEqual(
            len(midi_dict_adj_resolve.pedal_msgs),
            len(midi_dict_resolve_adj.pedal_msgs),
        )

        for msg_1, msg_2 in zip(
            midi_dict_adj_resolve.note_msgs, midi_dict_resolve_adj.note_msgs
        ):
            self.assertDictEqual(msg_1, msg_2)

        midi_dict_adj_resolve.to_midi().save(save_path)
