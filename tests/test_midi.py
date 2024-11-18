import unittest

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
