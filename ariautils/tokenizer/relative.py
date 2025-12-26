"""Contains MIDI tokenizer with relative onset timings."""

import functools
import itertools
import random

from pathlib import Path
from typing import Final, Callable, Any, Concatenate, ParamSpec

P = ParamSpec("P")

from ariautils.midi import (
    MidiDict,
    MetaMessage,
    TempoMessage,
    PedalMessage,
    InstrumentMessage,
    NoteMessage,
    get_duration_ms,
)
from ariautils.utils import load_config, get_logger
from ariautils.tokenizer._base import Tokenizer, Token


logger = get_logger(__package__)


class RelTokenizer(Tokenizer):
    """MidiDict tokenizer implemented with relative onset timings.

    Designed to resemble the tokenizer used for MuseNet, this tokenizer
    represents the passage of time using 'wait' tokens. This format is
    reminiscent of MIDI, which separates note-on and note-off messages with
    'wait' messages which record the number of milliseconds to wait
    before processing the next message.

    Tokenization Schema:
        For non-percussion instruments:
            - Each note is represented by three consecutive tokens:
                1. [instrument, pitch, velocity]: Instrument class, MIDI pitch,
                    and velocity
                2. [duration]: Note duration in milliseconds
                3. [wait]: Time in milliseconds to wait before processing the
                    next note

        For percussion instruments:
            - Each note is represented by two consecutive tokens:
                1. [drum, note_number]: Percussion instrument and MIDI note
                    number
                2. [wait]: Time in milliseconds to wait before processing the
                    next note

    Notes:
        - Notes are ordered according to onset time
        - Sustain pedal effects are incorporated directly into note durations
        - Start (<S>) and end (<E>) tokens wrap the tokenized sequence, and
            prefix tokens for instrument, genre, composer, and form are
            prepended, i.e., before the <S> token
        - Various configuration settings affecting instrument processing,
            timing resolution, quantization levels, and prefix tokens can be
            adjusted in config.json at 'tokenizer.rel'.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        super().__init__()
        self.config = load_config(config_path)["tokenizer"]["rel"]
        self.name = "rel"

        self.max_time_ms: int = self.config["max_time_ms"]
        self.time_step_ms: int = self.config["time_step_ms"]

        self.time_step_quantizations = [
            self.time_step_ms * i
            for i in range((self.max_time_ms // self.time_step_ms) + 1)
        ]

        # Calculate velocity quantizations
        self.velocity_step: int = self.config["velocity_quantization_step"]
        self.velocity_quantizations = [
            i * self.velocity_step
            for i in range(int(127 / self.velocity_step) + 1)
        ]
        self.max_velocity = self.velocity_quantizations[-1]

        # _nd = no drum; _wd = with drum
        self.instruments_nd = [
            k
            for k, v in self.config["ignore_instruments"].items()
            if v is False
        ]
        self.instruments_wd = self.instruments_nd + ["drum"]

        # Prefix tokens
        self.prefix_tokens: list[Token] = [
            ("prefix", "instrument", x) for x in self.instruments_wd
        ]
        self.composer_names: list[str] = self.config["composer_names"]
        self.form_names: list[str] = self.config["form_names"]
        self.genre_names: list[str] = self.config["form_names"]
        self.prefix_tokens += [
            ("prefix", "composer", x) for x in self.composer_names
        ]
        self.prefix_tokens += [("prefix", "form", x) for x in self.form_names]
        self.prefix_tokens += [("prefix", "genre", x) for x in self.genre_names]

        # Build vocab
        self.wait_tokens: list[Token] = [
            ("wait", i) for i in self.time_step_quantizations
        ]
        self.dur_tokens: list[Token] = [
            ("dur", i) for i in self.time_step_quantizations
        ]
        self.drum_tokens: list[Token] = [("drum", i) for i in range(35, 82)]

        self.note_tokens: list[Token] = list(
            itertools.product(
                self.instruments_nd,
                [i for i in range(128)],
                self.velocity_quantizations,
            )
        )

        self.add_tokens_to_vocab(
            self.special_tokens
            + self.prefix_tokens
            + self.note_tokens
            + self.drum_tokens
            + self.dur_tokens
            + self.wait_tokens
        )
        self.pad_id = self.tok_to_id[self.pad_tok]

    def export_data_aug(self) -> list[Callable[[list[Token]], list[Token]]]:
        return [
            self.export_chord_mixup(),
            self.export_tempo_aug(max_tempo_aug=0.2),
            self.export_pitch_aug(5),
            self.export_velocity_aug(1),
        ]

    def _quantize_time(self, time: int) -> int:
        # This function will return values res >= 0 (inc. 0)
        return self._find_closest_int(time, self.time_step_quantizations)

    def _quantize_velocity(self, velocity: int) -> int:
        # This function will return values in the range 0 < res =< 127
        velocity_quantized = self._find_closest_int(
            velocity, self.velocity_quantizations
        )

        if velocity_quantized == 0 and velocity != 0:
            return self.velocity_step
        else:
            return velocity_quantized

    def _format(
        self, prefix: list[Token], unformatted_seq: list[Token]
    ) -> list[Token]:
        # If unformatted_seq is longer than 150 tokens insert diminish tok
        idx = -100 + random.randint(-10, 10)
        if len(unformatted_seq) > 150:
            if unformatted_seq[idx][0] == "dur":  # Don't want: note, <D>, dur
                unformatted_seq.insert(idx - 1, self.dim_tok)
            else:
                unformatted_seq.insert(idx, self.dim_tok)

        res = prefix + [self.bos_tok] + unformatted_seq + [self.eos_tok]

        return res

    def _tokenize_midi_dict(
        self, midi_dict: MidiDict, remove_preceding_silence: bool = True
    ) -> list[Token]:
        ticks_per_beat = midi_dict.ticks_per_beat
        midi_dict.remove_instruments(self.config["ignore_instruments"])

        if len(midi_dict.note_msgs) == 0:
            raise Exception("note_msgs is empty after ignoring instruments")

        channel_to_pedal_intervals = midi_dict._build_pedal_intervals()

        channels_used = {msg["channel"] for msg in midi_dict.note_msgs}

        channel_to_instrument = {
            msg["channel"]: midi_dict.program_to_instrument[msg["data"]]
            for msg in midi_dict.instrument_msgs
            if msg["channel"] != 9  # Exclude drums
        }
        # If non-drum channel is missing from instrument_msgs, default to piano
        for c in channels_used:
            if channel_to_instrument.get(c) is None and c != 9:
                channel_to_instrument[c] = "piano"

        # Calculate prefix
        prefix: list[Token] = [
            ("prefix", "instrument", x)
            for x in set(channel_to_instrument.values())
        ]
        if 9 in channels_used:
            prefix.append(("prefix", "instrument", "drum"))
        composer = midi_dict.metadata.get("composer")
        if composer and (composer in self.composer_names):
            prefix.insert(0, ("prefix", "composer", composer))
        form = midi_dict.metadata.get("form")
        if form and (form in self.form_names):
            prefix.insert(0, ("prefix", "form", form))
        genre = midi_dict.metadata.get("genre")
        if genre and (genre in self.genre_names):
            prefix.insert(0, ("prefix", "genre", genre))
        random.shuffle(prefix)

        tokenized_seq: list[Token] = []

        if remove_preceding_silence is False and len(midi_dict.note_msgs) > 0:
            initial_wait_duration = get_duration_ms(
                start_tick=0,
                end_tick=midi_dict.note_msgs[0]["data"]["start"],
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=ticks_per_beat,
            )

            while initial_wait_duration > self.max_time_ms:
                tokenized_seq.append(("wait", self.max_time_ms))
                initial_wait_duration -= self.max_time_ms

            initial_wait_duration = self._quantize_time(initial_wait_duration)
            if initial_wait_duration != 0:
                tokenized_seq.append(("wait", initial_wait_duration))

        num_notes = len(midi_dict.note_msgs)
        for i, msg in enumerate(midi_dict.note_msgs):
            # Special case instrument is a drum. This occurs exclusively when
            # MIDI channel is 9 when 0 indexing
            if msg["channel"] == 9:
                _pitch = msg["data"]["pitch"]
                tokenized_seq.append(("drum", _pitch))

            else:  # Non drum case (i.e. an instrument note)
                _instrument = channel_to_instrument[msg["channel"]]
                _pitch = msg["data"]["pitch"]
                _velocity = msg["data"]["velocity"]
                _start_tick = msg["data"]["start"]
                _end_tick = msg["data"]["end"]

                # Update _end_tick if affected by pedal
                for pedal_interval in channel_to_pedal_intervals[
                    msg["channel"]
                ]:
                    pedal_start, pedal_end = (
                        pedal_interval[0],
                        pedal_interval[1],
                    )
                    if (
                        pedal_start <= _start_tick < pedal_end
                        and _end_tick < pedal_end
                    ):
                        _end_tick = pedal_end

                _note_duration = get_duration_ms(
                    start_tick=_start_tick,
                    end_tick=_end_tick,
                    tempo_msgs=midi_dict.tempo_msgs,
                    ticks_per_beat=ticks_per_beat,
                )

                _velocity = self._quantize_velocity(_velocity)
                _note_duration = self._quantize_time(_note_duration)
                if _note_duration == 0:
                    _note_duration = self.time_step_ms

                tokenized_seq.append((_instrument, _pitch, _velocity))
                tokenized_seq.append(("dur", _note_duration))

            # Only add wait token if there is a msg after the current one
            if i <= num_notes - 2:
                _wait_duration = get_duration_ms(
                    start_tick=msg["data"]["start"],
                    end_tick=midi_dict.note_msgs[i + 1]["data"]["start"],
                    tempo_msgs=midi_dict.tempo_msgs,
                    ticks_per_beat=ticks_per_beat,
                )

                # If wait duration is longer than maximum quantized time step
                # append max_time_step tokens repeatedly
                while _wait_duration > self.max_time_ms:
                    tokenized_seq.append(("wait", self.max_time_ms))
                    _wait_duration -= self.max_time_ms

                _wait_duration = self._quantize_time(_wait_duration)
                if _wait_duration != 0:
                    tokenized_seq.append(("wait", _wait_duration))

        return self._format(
            prefix=prefix,
            unformatted_seq=tokenized_seq,
        )

    def tokenize(
        self,
        midi_dict: MidiDict,
        remove_preceding_silence: bool = True,
        **kwargs: Any,
    ) -> list[Token]:
        """Tokenizes a MidiDict object into a sequence.

        Args:
            midi_dict (MidiDict): The MidiDict to tokenize.
            remove_preceding_silence (bool): If true starts the sequence at
                onset=0ms by removing preceding silence. Defaults to False.

        Returns:
            list[Token]: A sequence of tokens representing the MIDI content.
        """

        return self._tokenize_midi_dict(
            midi_dict=midi_dict,
            remove_preceding_silence=remove_preceding_silence,
        )

    def _detokenize_midi_dict(self, tokenized_seq: list[Token]) -> MidiDict:
        # NOTE: These values chosen so that 1000 ticks = 1000ms, allowing us to
        # skip converting between ticks and ms
        instrument_programs = self.config["instrument_programs"]
        TICKS_PER_BEAT: Final[int] = 500
        TEMPO: Final[int] = 500000

        tempo_msgs: list[TempoMessage] = [
            {"type": "tempo", "data": TEMPO, "tick": 0}
        ]
        meta_msgs: list[MetaMessage] = []
        pedal_msgs: list[PedalMessage] = []
        instrument_msgs: list[InstrumentMessage] = []

        instrument_to_channel: dict[str, int] = {}

        # Add non-drum instrument_msgs, breaks at first note token
        channel_idx = 0
        for idx, tok in enumerate(tokenized_seq):
            if channel_idx == 9:  # Skip channel reserved for drums
                channel_idx += 1

            if tok in self.special_tokens:
                continue
            elif (
                tok[0] == "prefix"
                and tok[1] == "instrument"
                and tok[2] in self.instruments_wd
            ):
                # Process instrument prefix tokens
                if tok[2] in instrument_to_channel.keys():
                    logger.debug(f"Duplicate prefix {tok[2]}")
                    continue
                elif tok[2] == "drum":
                    instrument_msgs.append(
                        {
                            "type": "instrument",
                            "data": 0,
                            "tick": 0,
                            "channel": 9,
                        }
                    )
                    instrument_to_channel["drum"] = 9
                else:
                    instrument_msgs.append(
                        {
                            "type": "instrument",
                            "data": instrument_programs[tok[2]],
                            "tick": 0,
                            "channel": channel_idx,
                        }
                    )
                    instrument_to_channel[tok[2]] = channel_idx
                    channel_idx += 1
            elif tok[0] == "prefix":
                # Skip all other prefix tokens
                continue
            else:
                # Note, wait, or duration token
                start = idx
                break

        # Note messages
        note_msgs: list[NoteMessage] = []
        curr_tick = 0
        for curr_tok, next_tok in zip(
            tokenized_seq[start:], tokenized_seq[start + 1 :]
        ):
            if curr_tok in self.special_tokens:
                _curr_tok_type = "special"
            else:
                _curr_tok_type = curr_tok[0]

            if next_tok in self.special_tokens:
                _next_tok_type = "special"
            else:
                _next_tok_type = next_tok[0]

            if (
                _curr_tok_type == "special"
                or _curr_tok_type == "prefix"
                or _curr_tok_type == "dur"
            ):
                continue
            elif _curr_tok_type == "wait":
                assert isinstance(
                    curr_tok[1], int
                ), f"Expected int for wait, got {curr_tok[1]}"
                curr_tick += curr_tok[1]
            elif _curr_tok_type == "drum":
                assert isinstance(
                    curr_tok[1], int
                ), f"Expected int for onset, got {curr_tok[1]}"

                _pitch = curr_tok[1]
                _channel = instrument_to_channel["drum"]
                _velocity = self.config["drum_velocity"]
                _start_tick = curr_tick
                _end_tick = curr_tick + self.time_step_ms

                note_msgs.append(
                    {
                        "type": "note",
                        "data": {
                            "pitch": _pitch,
                            "start": _start_tick,
                            "end": _end_tick,
                            "velocity": _velocity,
                        },
                        "tick": _start_tick,
                        "channel": _channel,
                    }
                )

            elif (
                _curr_tok_type in self.instruments_nd
                and _next_tok_type == "dur"
            ):
                assert isinstance(
                    curr_tok[0], str
                ), f"Expected str for instrument, got {curr_tok[0]}"
                assert isinstance(
                    curr_tok[1], int
                ), f"Expected int for pitch, got {curr_tok[1]}"
                assert isinstance(
                    curr_tok[2], int
                ), f"Expected int for velocity, got {curr_tok[2]}"
                assert isinstance(
                    next_tok[1], int
                ), f"Expected int for duration, got {next_tok[1]}"

                _instrument = curr_tok[0]
                _pitch = curr_tok[1]
                _velocity = curr_tok[2]
                _start_tick = curr_tick
                _end_tick = curr_tick + next_tok[1]

                if _instrument not in instrument_to_channel.keys():
                    logger.warning(
                        f"Tried to decode note message for unexpected instrument: {_instrument} "
                    )
                else:
                    _channel = instrument_to_channel[_instrument]
                    note_msgs.append(
                        {
                            "type": "note",
                            "data": {
                                "pitch": _pitch,
                                "start": _start_tick,
                                "end": _end_tick,
                                "velocity": _velocity,
                            },
                            "tick": _start_tick,
                            "channel": _channel,
                        }
                    )

            else:
                logger.warning(
                    f"Unexpected token sequence: {curr_tok}, {next_tok}"
                )

        return MidiDict(
            meta_msgs=meta_msgs,
            tempo_msgs=tempo_msgs,
            pedal_msgs=pedal_msgs,
            instrument_msgs=instrument_msgs,
            note_msgs=note_msgs,
            ticks_per_beat=TICKS_PER_BEAT,
            metadata={},
        )

    def detokenize(self, tokenized_seq: list[Token], **kwargs: Any) -> MidiDict:
        """Detokenizes a MidiDict object.

        Args:
            tokenized_seq (list): The sequence of tokens to detokenize.

        Returns:
            MidiDict: A MidiDict reconstructed from the tokens.
        """

        return self._detokenize_midi_dict(tokenized_seq=tokenized_seq)

    def export_pitch_aug(
        self, max_pitch_aug: int
    ) -> Callable[Concatenate[list[Token], P], list[Token]]:
        """Exports a function that augments the pitch of all note tokens.

        Note that notes which fall out of the range (0, 127) will be replaced
        with the unknown token '<U>'.

        Args:
            max_pitch_aug (int): Returned function will randomly augment the pitch
                from a value in the range (-max_pitch_aug, max_pitch_aug).

        Returns:
            Callable[[list[Token], int], list[Token]]: Exported function.
        """

        def pitch_aug_seq(
            src: list[Token],
            unk_tok: str,
            _max_pitch_aug: int,
            pitch_aug: int | None = None,
        ) -> list[Token]:
            def pitch_aug_tok(tok: Token, _pitch_aug: int) -> Token:
                if isinstance(tok, str):  # Stand in for SpecialToken
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if (
                    _tok_type == "special"
                    or _tok_type == "prefix"
                    or _tok_type == "dur"
                    or _tok_type == "drum"
                    or _tok_type == "wait"
                ):
                    return tok
                else:
                    # Return augmented tok
                    assert (
                        isinstance(tok, tuple) and len(tok) == 3
                    ), f"Invalid note token"
                    (_instrument, _pitch, _velocity) = tok

                    if 0 <= _pitch + _pitch_aug <= 127:
                        return (_instrument, _pitch + _pitch_aug, _velocity)
                    else:
                        return unk_tok

            if pitch_aug is None:
                pitch_aug = random.randint(-_max_pitch_aug, _max_pitch_aug)

            return [pitch_aug_tok(x, pitch_aug) for x in src]

        return functools.partial(
            pitch_aug_seq,
            unk_tok=self.unk_tok,
            _max_pitch_aug=max_pitch_aug,
        )

    def export_velocity_aug(
        self, max_num_aug_steps: int
    ) -> Callable[Concatenate[list[Token], P], list[Token]]:
        """Exports a function which augments the velocity of all pitch tokens.

        Velocity values are clipped so that they don't fall outside of the
        valid range.

        Args:
            max_num_aug_steps (int): Returned function will randomly augment
                velocity in the range self.velocity_step * (-max_num_aug_steps,
                max_num_aug_steps).

        Returns:
            Callable[[list[Token], int], list[Token]]: Exported function.
        """

        def velocity_aug_seq(
            src: list[Token],
            min_velocity_step: int,
            max_velocity: int,
            _max_num_aug_steps: int,
            aug_step: int | None = None,
        ) -> list[Token]:
            def velocity_aug_tok(tok: Token, _velocity_aug: int) -> Token:
                if isinstance(tok, str):  # Stand in for SpecialToken
                    _tok_type = "special"
                else:
                    _tok_type = tok[0]

                if (
                    _tok_type == "special"
                    or _tok_type == "prefix"
                    or _tok_type == "dur"
                    or _tok_type == "drum"
                    or _tok_type == "wait"
                ):
                    # Return without changing
                    return tok
                else:
                    # Return augmented tok
                    assert isinstance(tok, tuple) and len(tok) == 3
                    (_instrument, _pitch, _velocity) = tok

                    assert isinstance(_pitch, int)
                    assert isinstance(_velocity, int)

                    # Check it doesn't go out of bounds
                    if _velocity + _velocity_aug >= max_velocity:
                        return (_instrument, _pitch, max_velocity)
                    elif _velocity + _velocity_aug <= min_velocity_step:
                        return (_instrument, _pitch, min_velocity_step)

                    return (_instrument, _pitch, _velocity + _velocity_aug)

            if aug_step is None:
                velocity_aug = min_velocity_step * random.randint(
                    -_max_num_aug_steps, _max_num_aug_steps
                )
            else:
                velocity_aug = aug_step * min_velocity_step

            return [velocity_aug_tok(x, velocity_aug) for x in src]

        return self.export_aug_fn_concat(
            functools.partial(
                velocity_aug_seq,
                min_velocity_step=self.velocity_step,
                max_velocity=self.max_velocity,
                _max_num_aug_steps=max_num_aug_steps,
            )
        )

    # TODO: Needs unit test
    def export_tempo_aug(
        self, max_tempo_aug: float
    ) -> Callable[Concatenate[list[Token], P], list[Token]]:
        """Exports a function which augments the tempo of a sequence of tokens.

        Args:
            max_tempo_aug (float): Returned function will randomly augment
                tempo by a factor in the range (1 - max_tempo_aug,
                1 + max_tempo_aug).

        Returns:
            Callable[[list[Token], float], list[Token]]: Exported function.
        """

        def tempo_aug_seq(
            src: list,
            time_step_ms: int,
            max_time_ms: int,
            _max_tempo_aug: float,
            tempo_aug: float | None = None,
        ) -> list[Token]:
            def _quantize_time_no_truncate(_n: int | float) -> int:
                return round(_n / time_step_ms) * time_step_ms

            def _append_wait_tokens(
                _res: list[Token], _wait_time_ms: int
            ) -> list[Token]:
                while _wait_time_ms > max_time_ms:
                    _res.append(("wait", max_time_ms))
                    _wait_time_ms -= max_time_ms

                _wait_time_ms = min(
                    _quantize_time_no_truncate(_wait_time_ms),
                    max_time_ms,
                )
                if _wait_time_ms != 0:
                    _res.append(("wait", _wait_time_ms))

                return res

            if tempo_aug is None:
                tempo_aug = random.uniform(
                    1 - max_tempo_aug, 1 + _max_tempo_aug
                )

            res = []
            wait_time_ms: float = 0
            src_ms: int = 0
            tgt_ms: float = 0
            for tok in src:
                if (
                    isinstance(tok, str) or tok[1] == "prefix"
                ):  # Stand in for SpecialToken
                    res.append(tok)
                    continue
                else:
                    tok_type = tok[0]

                if tok_type == "wait":
                    src_ms += tok[1]
                    wait_time_ms += (src_ms * tempo_aug) - tgt_ms
                    tgt_ms += tempo_aug * tok[1]
                elif tok_type == "dur":
                    dur_ms = min(
                        _quantize_time_no_truncate(tok[1] * tempo_aug),
                        max_time_ms,
                    )
                    res.append(("dur", dur_ms))
                else:  # Note msg
                    res = _append_wait_tokens(
                        res, _quantize_time_no_truncate(wait_time_ms)
                    )
                    res.append(tok)
                    wait_time_ms = 0

            return res

        return self.export_aug_fn_concat(
            functools.partial(
                tempo_aug_seq,
                time_step_ms=self.time_step_ms,
                max_time_ms=self.max_time_ms,
                _max_tempo_aug=max_tempo_aug,
            )
        )

    def export_chord_mixup(
        self,
    ) -> Callable[Concatenate[list[Token], P], list[Token]]:
        """Exports a function which augments order of concurrent note tokens.

        Concurrent note tokens are those which are not separated by a wait
        token.

        Returns:
            Callable[[list[Token]], list[Token]]: Exported function.
        """

        def chord_mixup(src: list[Token], unk_tok: str) -> list[Token]:
            stack: list[dict[str, Token]] = []
            for idx, tok in enumerate(src):
                if isinstance(tok, str):
                    tok_type = "special"
                else:
                    tok_type = tok[0]

                if (
                    tok_type == "special" or tok_type == "prefix"
                ) and tok != unk_tok:
                    # Skip special tok (when not unk), reset stack to be safe
                    stack = []
                elif tok_type == "wait" and len(stack) <= 1:
                    # Reset stack as it only contains one note
                    stack = []
                elif tok_type == "wait" and len(stack) > 1:
                    # Stack contains more than one note -> mix-up stack.
                    random.shuffle(stack)
                    num_toks = sum(len(note) for note in stack)
                    _idx = idx - num_toks

                    while stack:
                        entry = stack.pop()
                        if entry["note"] == unk_tok:
                            # This can happen if the note token has its pitch
                            # augmented out of the valid range. In this case we
                            # do not want to index it as it is not a note token
                            src[_idx] = entry["note"]
                            src[_idx + 1] = entry["dur"]
                            _idx += 2
                        elif entry["note"][0] == "drum":
                            # Drum case doesn't require a duration token
                            src[_idx] = entry["note"]
                            _idx += 1
                        else:
                            src[_idx] = entry["note"]
                            src[_idx + 1] = entry["dur"]
                            _idx += 2

                elif tok_type == "dur":
                    # Add dur to previously added note token if exists
                    if stack:
                        stack[-1]["dur"] = tok
                else:
                    # Note token -> append to stack
                    stack.append({"note": tok})

            return src

        return self.export_aug_fn_concat(
            functools.partial(
                chord_mixup,
                unk_tok=self.unk_tok,
            )
        )
