"""Tests for tokenizers."""

import unittest
import copy

from importlib import resources
from pathlib import Path
from typing import Final

from ariautils.midi import MidiDict, normalize_midi_dict
from ariautils.tokenizer import AbsTokenizer, RelTokenizer
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
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )
            normalized_twice_midi_dict = normalize_midi_dict(
                normalized_midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
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
        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("transcription_norm.mid")
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
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
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
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            self.assertDictEqual(
                midi_dict_1.get_msg_dict(),
                midi_dict_2.get_msg_dict(),
            )

        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        _test_tokenize_detokenize(_load_path=load_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        _test_tokenize_detokenize(_load_path=load_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        _test_tokenize_detokenize(_load_path=load_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        _test_tokenize_detokenize(_load_path=load_path)

    def test_tokenize_detokenize_pedal(self) -> None:
        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("transcription_pedal.mid")

        midi_dict = MidiDict.from_midi(load_path)
        tokenizer = AbsTokenizer()

        seq = tokenizer.tokenize(midi_dict=midi_dict)
        tokenizer.detokenize(tokenized_seq=seq).to_midi().save(save_path)

        save_path = RESULTS_DATA_DIRECTORY.joinpath(
            "transcription_pedal_aug.mid"
        )
        data_aug = tokenizer.export_data_aug()

        for f in data_aug:
            seq = f(seq)

        tokenizer.detokenize(tokenized_seq=seq).to_midi().save(save_path)

    def test_pitch_aug(self) -> None:
        def _test_out_of_bounds(
            tokenizer: AbsTokenizer,
            midi_dict: MidiDict,
            pitch_aug: int,
        ) -> None:
            pitch_aug_fn = tokenizer.export_pitch_aug(pitch_aug)
            seq = tokenizer.tokenize(midi_dict, remove_preceding_silence=False)
            augmented_seq = pitch_aug_fn(seq, pitch_aug=pitch_aug)

            for tok_1, tok_2 in zip(seq, augmented_seq):
                if (
                    isinstance(tok_1, tuple)
                    and tok_1[0] in tokenizer.instruments_nd
                    and tok_2 == tokenizer.unk_tok
                ):
                    self.assertTrue(
                        tok_1[1] + pitch_aug not in set(range(0, 128))
                    )

        def _test_pitch_aug(
            tokenizer: AbsTokenizer,
            midi_dict: MidiDict,
            pitch_aug: int,
        ) -> None:
            midi_dict = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            pitch_aug_fn = tokenizer.export_pitch_aug(pitch_aug)
            seq = tokenizer.tokenize(midi_dict, remove_preceding_silence=False)
            augmented_seq = pitch_aug_fn(seq, pitch_aug=pitch_aug)
            midi_dict_aug = tokenizer.detokenize(augmented_seq)

            self.assertEqual(len(seq), len(augmented_seq))

            if tokenizer.unk_tok in augmented_seq:
                # Skip cases with unk tok
                self.logger.info(
                    f"Seen unk_tok on {load_path.name} for pitch_aug={pitch_aug}"
                )
                return

            midi_dict_aug = normalize_midi_dict(
                midi_dict=midi_dict_aug,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            for msg_no_aug, msg_aug in zip(
                midi_dict.note_msgs, midi_dict_aug.note_msgs
            ):
                if msg_no_aug["channel"] != 9:
                    self.assertEqual(
                        msg_no_aug["data"]["pitch"] + pitch_aug,
                        msg_aug["data"]["pitch"],
                    )

                    msg_no_aug_compare = copy.deepcopy(msg_no_aug)
                    msg_aug_compare = copy.deepcopy(msg_aug)
                    del msg_no_aug_compare["data"]["pitch"]  # type: ignore[misc]
                    del msg_aug_compare["data"]["pitch"]  # type: ignore[misc]
                    self.assertDictEqual(msg_no_aug_compare, msg_aug_compare)
                else:
                    self.assertDictEqual(msg_no_aug, msg_aug)

        tokenizer = AbsTokenizer()
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for pitch_aug in range(-30, 30):
            _test_pitch_aug(tokenizer, midi_dict, pitch_aug)
            _test_out_of_bounds(tokenizer, midi_dict, pitch_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for pitch_aug in range(-30, 30):
            _test_pitch_aug(tokenizer, midi_dict, pitch_aug)
            _test_out_of_bounds(tokenizer, midi_dict, pitch_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for pitch_aug in range(-30, 30):
            _test_pitch_aug(tokenizer, midi_dict, pitch_aug)
            _test_out_of_bounds(tokenizer, midi_dict, pitch_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for pitch_aug in range(-30, 30):
            _test_pitch_aug(tokenizer, midi_dict, pitch_aug)
            _test_out_of_bounds(tokenizer, midi_dict, pitch_aug)

    def test_velocity_aug(self) -> None:
        def _test_velocity_aug(
            tokenizer: AbsTokenizer,
            midi_dict: MidiDict,
            velocity_aug_step: int,
        ) -> None:
            midi_dict = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            velocity_aug_fn = tokenizer.export_velocity_aug(velocity_aug_step)
            seq = tokenizer.tokenize(midi_dict, remove_preceding_silence=False)
            augmented_seq = velocity_aug_fn(seq, aug_step=velocity_aug_step)
            midi_dict_aug = tokenizer.detokenize(augmented_seq)

            self.assertEqual(len(seq), len(augmented_seq))
            self.assertTrue(tokenizer.unk_tok not in augmented_seq)

            midi_dict_aug = normalize_midi_dict(
                midi_dict=midi_dict_aug,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            for msg_no_aug, msg_aug in zip(
                midi_dict.note_msgs, midi_dict_aug.note_msgs
            ):
                if msg_no_aug["channel"] == 9:
                    self.assertDictEqual(msg_no_aug, msg_aug)
                else:
                    _velocity = min(
                        max(
                            msg_no_aug["data"]["velocity"]
                            + velocity_aug_step * tokenizer.velocity_step,
                            tokenizer.velocity_step,
                        ),
                        tokenizer.max_velocity,
                    )
                    self.assertEqual(msg_aug["data"]["velocity"], _velocity)

                msg_no_aug_compare = copy.deepcopy(msg_no_aug)
                msg_aug_compare = copy.deepcopy(msg_aug)
                del msg_no_aug_compare["data"]["velocity"]  # type: ignore[misc]
                del msg_aug_compare["data"]["velocity"]  # type: ignore[misc]
                self.assertDictEqual(msg_no_aug_compare, msg_aug_compare)

        tokenizer = AbsTokenizer()
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for velocity_aug in range(-10, 10):
            _test_velocity_aug(tokenizer, midi_dict, velocity_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for velocity_aug in range(-10, 10):
            _test_velocity_aug(tokenizer, midi_dict, velocity_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for velocity_aug in range(-10, 10):
            _test_velocity_aug(tokenizer, midi_dict, velocity_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for velocity_aug in range(-10, 10):
            _test_velocity_aug(tokenizer, midi_dict, velocity_aug)

    def test_tempo_aug(self) -> None:
        def _quantize_time(_n: int | float, time_step: int) -> int:
            return round(_n / time_step) * time_step

        def _test_tempo_aug(
            tokenizer: AbsTokenizer,
            midi_dict: MidiDict,
            tempo_aug: float,
        ) -> None:
            midi_dict = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            tempo_aug_fn = tokenizer.export_tempo_aug(tempo_aug, mixup=False)
            seq = tokenizer.tokenize(midi_dict, remove_preceding_silence=False)
            augmented_seq = tempo_aug_fn(seq, tempo_aug=tempo_aug)
            midi_dict_aug = tokenizer.detokenize(augmented_seq)

            self.assertEqual(len(seq), len(augmented_seq))
            self.assertTrue(tokenizer.unk_tok not in augmented_seq)

            midi_dict_aug = normalize_midi_dict(
                midi_dict=midi_dict_aug,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_dur_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            for msg_no_aug, msg_aug in zip(
                midi_dict.note_msgs, midi_dict_aug.note_msgs
            ):
                _start_tick = _quantize_time(
                    msg_no_aug["data"]["start"] * tempo_aug,
                    time_step=tokenizer.time_step_ms,
                )
                _end_tick = min(
                    _start_tick + tokenizer.max_dur_ms,
                    _quantize_time(
                        msg_no_aug["data"]["end"] * tempo_aug,
                        time_step=tokenizer.time_step_ms,
                    ),
                )

                self.assertLessEqual(abs(msg_aug["tick"] - _start_tick), 10)
                self.assertLessEqual(
                    abs(msg_aug["data"]["start"] - _start_tick), 10
                )
                self.assertLessEqual(
                    abs(msg_aug["data"]["end"] - _end_tick), 10
                )

                msg_no_aug_compare = copy.deepcopy(msg_no_aug)
                msg_aug_compare = copy.deepcopy(msg_aug)
                del msg_no_aug_compare["tick"]  # type: ignore[misc]
                del msg_no_aug_compare["data"]["start"]  # type: ignore[misc]
                del msg_no_aug_compare["data"]["end"]  # type: ignore[misc]
                del msg_aug_compare["tick"]  # type: ignore[misc]
                del msg_aug_compare["data"]["start"]  # type: ignore[misc]
                del msg_aug_compare["data"]["end"]  # type: ignore[misc]
                self.assertDictEqual(msg_no_aug_compare, msg_aug_compare)

        tokenizer = AbsTokenizer()
        tempo_range = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for tempo_aug in tempo_range:
            _test_tempo_aug(tokenizer, midi_dict, tempo_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for tempo_aug in tempo_range:
            _test_tempo_aug(tokenizer, midi_dict, tempo_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for tempo_aug in tempo_range:
            _test_tempo_aug(tokenizer, midi_dict, tempo_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for tempo_aug in tempo_range:
            _test_tempo_aug(tokenizer, midi_dict, tempo_aug)


# TODO: Add pitch_aug test for relative tokenizer
class TestRelTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = get_logger(__name__ + ".TestRelTokenizer")

    def test_normalize_midi_dict(self) -> None:
        def _test_normalize_midi_dict(
            _load_path: Path, _save_path: Path
        ) -> None:
            tokenizer = RelTokenizer()
            midi_dict = MidiDict.from_midi(_load_path)
            midi_dict_copy = copy.deepcopy(midi_dict)

            normalized_midi_dict = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_time_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )
            normalized_twice_midi_dict = normalize_midi_dict(
                normalized_midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_time_ms,
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
        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("transcription_norm.mid")
        _test_normalize_midi_dict(load_path, save_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("pop_norm.mid")
        _test_normalize_midi_dict(load_path, save_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        save_path = RESULTS_DATA_DIRECTORY.joinpath("basic_norm.mid")
        _test_normalize_midi_dict(load_path, save_path)

    def test_tokenize_detokenize(self) -> None:
        def _test_tokenize_detokenize(_load_path: Path) -> None:
            tokenizer = RelTokenizer()
            midi_dict = MidiDict.from_midi(_load_path)

            midi_dict_1 = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_time_ms,
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
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_time_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            self.assertDictEqual(
                midi_dict_1.get_msg_dict(),
                midi_dict_2.get_msg_dict(),
            )

        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        _test_tokenize_detokenize(_load_path=load_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        _test_tokenize_detokenize(_load_path=load_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        _test_tokenize_detokenize(_load_path=load_path)
        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        _test_tokenize_detokenize(_load_path=load_path)

    def test_pitch_aug(self) -> None:
        def _test_out_of_bounds(
            tokenizer: RelTokenizer,
            midi_dict: MidiDict,
            pitch_aug: int,
        ) -> None:
            pitch_aug_fn = tokenizer.export_pitch_aug(pitch_aug)
            seq = tokenizer.tokenize(midi_dict, remove_preceding_silence=False)
            augmented_seq = pitch_aug_fn(seq, pitch_aug=pitch_aug)

            for tok_1, tok_2 in zip(seq, augmented_seq):
                if (
                    isinstance(tok_1, tuple)
                    and tok_1[0] in tokenizer.instruments_nd
                    and tok_2 == tokenizer.unk_tok
                ):
                    self.assertTrue(
                        tok_1[1] + pitch_aug not in set(range(0, 128))
                    )

        def _test_pitch_aug(
            tokenizer: RelTokenizer,
            midi_dict: MidiDict,
            pitch_aug: int,
        ) -> None:
            midi_dict = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_time_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            pitch_aug_fn = tokenizer.export_pitch_aug(pitch_aug)
            seq = tokenizer.tokenize(midi_dict, remove_preceding_silence=False)
            augmented_seq = pitch_aug_fn(seq, pitch_aug=pitch_aug)
            midi_dict_aug = tokenizer.detokenize(augmented_seq)

            self.assertEqual(len(seq), len(augmented_seq))

            if tokenizer.unk_tok in augmented_seq:
                # Skip cases with unk tok
                self.logger.info(
                    f"Seen unk_tok on {load_path.name} for pitch_aug={pitch_aug}"
                )
                return

            midi_dict_aug = normalize_midi_dict(
                midi_dict=midi_dict_aug,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_time_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            for msg_no_aug, msg_aug in zip(
                midi_dict.note_msgs, midi_dict_aug.note_msgs
            ):
                if msg_no_aug["channel"] != 9:
                    self.assertEqual(
                        msg_no_aug["data"]["pitch"] + pitch_aug,
                        msg_aug["data"]["pitch"],
                    )

                    msg_no_aug_compare = copy.deepcopy(msg_no_aug)
                    msg_aug_compare = copy.deepcopy(msg_aug)
                    del msg_no_aug_compare["data"]["pitch"]  # type: ignore[misc]
                    del msg_aug_compare["data"]["pitch"]  # type: ignore[misc]
                    self.assertDictEqual(msg_no_aug_compare, msg_aug_compare)
                else:
                    self.assertDictEqual(msg_no_aug, msg_aug)

        tokenizer = RelTokenizer()
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for pitch_aug in range(-30, 30):
            _test_pitch_aug(tokenizer, midi_dict, pitch_aug)
            _test_out_of_bounds(tokenizer, midi_dict, pitch_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for pitch_aug in range(-30, 30):
            _test_pitch_aug(tokenizer, midi_dict, pitch_aug)
            _test_out_of_bounds(tokenizer, midi_dict, pitch_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for pitch_aug in range(-30, 30):
            _test_pitch_aug(tokenizer, midi_dict, pitch_aug)
            _test_out_of_bounds(tokenizer, midi_dict, pitch_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for pitch_aug in range(-30, 30):
            _test_pitch_aug(tokenizer, midi_dict, pitch_aug)
            _test_out_of_bounds(tokenizer, midi_dict, pitch_aug)

    def test_velocity_aug(self) -> None:
        def _test_velocity_aug(
            tokenizer: RelTokenizer,
            midi_dict: MidiDict,
            velocity_aug_step: int,
        ) -> None:
            midi_dict = normalize_midi_dict(
                midi_dict=midi_dict,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_time_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            velocity_aug_fn = tokenizer.export_velocity_aug(velocity_aug_step)
            seq = tokenizer.tokenize(midi_dict, remove_preceding_silence=False)
            augmented_seq = velocity_aug_fn(seq, aug_step=velocity_aug_step)
            midi_dict_aug = tokenizer.detokenize(augmented_seq)

            self.assertEqual(len(seq), len(augmented_seq))
            self.assertTrue(tokenizer.unk_tok not in augmented_seq)

            midi_dict_aug = normalize_midi_dict(
                midi_dict=midi_dict_aug,
                ignore_instruments=tokenizer.config["ignore_instruments"],
                instrument_programs=tokenizer.config["instrument_programs"],
                time_step_ms=tokenizer.time_step_ms,
                max_duration_ms=tokenizer.max_time_ms,
                drum_velocity=tokenizer.config["drum_velocity"],
                quantize_velocity_fn=tokenizer._quantize_velocity,
            )

            for msg_no_aug, msg_aug in zip(
                midi_dict.note_msgs, midi_dict_aug.note_msgs
            ):
                if msg_no_aug["channel"] == 9:
                    self.assertDictEqual(msg_no_aug, msg_aug)
                else:
                    _velocity = min(
                        max(
                            msg_no_aug["data"]["velocity"]
                            + velocity_aug_step * tokenizer.velocity_step,
                            tokenizer.velocity_step,
                        ),
                        tokenizer.max_velocity,
                    )
                    self.assertEqual(msg_aug["data"]["velocity"], _velocity)

                msg_no_aug_compare = copy.deepcopy(msg_no_aug)
                msg_aug_compare = copy.deepcopy(msg_aug)
                del msg_no_aug_compare["data"]["velocity"]  # type: ignore[misc]
                del msg_aug_compare["data"]["velocity"]  # type: ignore[misc]
                self.assertDictEqual(msg_no_aug_compare, msg_aug_compare)

        tokenizer = RelTokenizer()
        load_path = TEST_DATA_DIRECTORY.joinpath("arabesque.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for velocity_aug in range(-10, 10):
            _test_velocity_aug(tokenizer, midi_dict, velocity_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("transcription.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for velocity_aug in range(-10, 10):
            _test_velocity_aug(tokenizer, midi_dict, velocity_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("pop.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for velocity_aug in range(-10, 10):
            _test_velocity_aug(tokenizer, midi_dict, velocity_aug)

        load_path = TEST_DATA_DIRECTORY.joinpath("basic.mid")
        midi_dict = MidiDict.from_midi(load_path)
        for velocity_aug in range(-10, 10):
            _test_velocity_aug(tokenizer, midi_dict, velocity_aug)
