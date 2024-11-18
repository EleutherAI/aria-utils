"""Utils for data/MIDI processing."""

import re
import os
import json
import hashlib
import unicodedata
import mido

from collections import defaultdict
from pathlib import Path
from typing import (
    List,
    Dict,
    Any,
    Tuple,
    Final,
    Concatenate,
    Callable,
    TypeAlias,
    Literal,
    TypedDict,
    cast,
)

from mido.midifiles.units import tick2second
from ariautils.utils import load_config, load_maestro_metadata_json


class MetaMessage(TypedDict):
    """Meta message type corresponding text or copyright MIDI meta messages."""

    type: Literal["text", "copyright"]
    data: str


class TempoMessage(TypedDict):
    """Tempo message type corresponding to the set_tempo MIDI message."""

    type: Literal["tempo"]
    data: int
    tick: int


class PedalMessage(TypedDict):
    """Sustain pedal message type corresponding to control_change 64 MIDI messages."""

    type: Literal["pedal"]
    data: Literal[0, 1]  # 0 for off, 1 for on
    tick: int
    channel: int


class InstrumentMessage(TypedDict):
    """Instrument message type corresponding to program_change MIDI messages."""

    type: Literal["instrument"]
    data: int
    tick: int
    channel: int


class NoteData(TypedDict):
    pitch: int
    start: int
    end: int
    velocity: int


class NoteMessage(TypedDict):
    """Note message type corresponding to paired note_on and note_off MIDI messages."""

    type: Literal["note"]
    data: NoteData
    tick: int
    channel: int


MidiMessage: TypeAlias = (
    MetaMessage | TempoMessage | PedalMessage | InstrumentMessage | NoteMessage
)


class MidiDictData(TypedDict):
    """Type for MidiDict attributes in dictionary form."""

    meta_msgs: List[MetaMessage]
    tempo_msgs: List[TempoMessage]
    pedal_msgs: List[PedalMessage]
    instrument_msgs: List[InstrumentMessage]
    note_msgs: List[NoteMessage]
    ticks_per_beat: int
    metadata: Dict[str, Any]


class MidiDict:
    """Container for MIDI data in dictionary form.

    Args:
        meta_msgs (List[MetaMessage]): List of text or copyright MIDI meta messages.
        tempo_msgs (List[TempoMessage]): List of tempo change messages.
        pedal_msgs (List[PedalMessage]): List of sustain pedal messages.
        instrument_msgs (List[InstrumentMessage]): List of program change messages.
        note_msgs (List[NoteMessage]): List of note messages from paired note-on/off events.
        ticks_per_beat (int): MIDI ticks per beat.
        metadata (dict): Optional metadata key-value pairs (e.g., {"genre": "classical"}).
    """

    def __init__(
        self,
        meta_msgs: List[MetaMessage],
        tempo_msgs: List[TempoMessage],
        pedal_msgs: List[PedalMessage],
        instrument_msgs: List[InstrumentMessage],
        note_msgs: List[NoteMessage],
        ticks_per_beat: int,
        metadata: Dict[str, Any],
    ):
        self.meta_msgs = meta_msgs
        self.tempo_msgs = tempo_msgs
        self.pedal_msgs = pedal_msgs
        self.instrument_msgs = instrument_msgs
        self.note_msgs = sorted(note_msgs, key=lambda msg: msg["tick"])
        self.ticks_per_beat = ticks_per_beat
        self.metadata = metadata

        # Tracks if resolve_pedal() has been called.
        self.pedal_resolved = False

        # If tempo_msgs is empty, initalize to default
        if not self.tempo_msgs:
            DEFAULT_TEMPO_MSG: TempoMessage = {
                "type": "tempo",
                "data": 500000,
                "tick": 0,
            }
            self.tempo_msgs = [DEFAULT_TEMPO_MSG]
        # If tempo_msgs is empty, initalize to default (piano)
        if not self.instrument_msgs:
            DEFAULT_INSTRUMENT_MSG: InstrumentMessage = {
                "type": "instrument",
                "data": 0,
                "tick": 0,
                "channel": 0,
            }
            self.instrument_msgs = [DEFAULT_INSTRUMENT_MSG]

        self.program_to_instrument = self.get_program_to_instrument()

    @classmethod
    def get_program_to_instrument(cls) -> Dict[int, str]:
        """Return a map of MIDI program to instrument name."""

        PROGRAM_TO_INSTRUMENT: Final[Dict[int, str]] = (
            {i: "piano" for i in range(0, 7 + 1)}
            | {i: "chromatic" for i in range(8, 15 + 1)}
            | {i: "organ" for i in range(16, 23 + 1)}
            | {i: "guitar" for i in range(24, 31 + 1)}
            | {i: "bass" for i in range(32, 39 + 1)}
            | {i: "strings" for i in range(40, 47 + 1)}
            | {i: "ensemble" for i in range(48, 55 + 1)}
            | {i: "brass" for i in range(56, 63 + 1)}
            | {i: "reed" for i in range(64, 71 + 1)}
            | {i: "pipe" for i in range(72, 79 + 1)}
            | {i: "synth_lead" for i in range(80, 87 + 1)}
            | {i: "synth_pad" for i in range(88, 95 + 1)}
            | {i: "synth_effect" for i in range(96, 103 + 1)}
            | {i: "ethnic" for i in range(104, 111 + 1)}
            | {i: "percussive" for i in range(112, 119 + 1)}
            | {i: "sfx" for i in range(120, 127 + 1)}
        )

        return PROGRAM_TO_INSTRUMENT

    def get_msg_dict(self) -> MidiDictData:
        """Returns MidiDict data in dictionary form."""

        return {
            "meta_msgs": self.meta_msgs,
            "tempo_msgs": self.tempo_msgs,
            "pedal_msgs": self.pedal_msgs,
            "instrument_msgs": self.instrument_msgs,
            "note_msgs": self.note_msgs,
            "ticks_per_beat": self.ticks_per_beat,
            "metadata": self.metadata,
        }

    def to_midi(self) -> mido.MidiFile:
        """Inplace version of dict_to_midi."""

        return dict_to_midi(self.get_msg_dict())

    @classmethod
    def from_msg_dict(cls, msg_dict: MidiDictData) -> "MidiDict":
        """Inplace version of midi_to_dict."""

        assert msg_dict.keys() == {
            "meta_msgs",
            "tempo_msgs",
            "pedal_msgs",
            "instrument_msgs",
            "note_msgs",
            "ticks_per_beat",
            "metadata",
        }

        return cls(**msg_dict)

    @classmethod
    def from_midi(cls, mid_path: str | Path) -> "MidiDict":
        """Loads a MIDI file from path and returns MidiDict."""

        mid = mido.MidiFile(mid_path)
        return cls(**midi_to_dict(mid))

    def calculate_hash(self) -> str:
        msg_dict_to_hash = cast(dict, self.get_msg_dict())

        # Remove metadata before calculating hash
        msg_dict_to_hash.pop("meta_msgs")
        msg_dict_to_hash.pop("ticks_per_beat")
        msg_dict_to_hash.pop("metadata")

        return hashlib.md5(
            json.dumps(msg_dict_to_hash, sort_keys=True).encode()
        ).hexdigest()

    def tick_to_ms(self, tick: int) -> int:
        """Calculate the time (in milliseconds) in current file at a MIDI tick."""

        return get_duration_ms(
            start_tick=0,
            end_tick=tick,
            tempo_msgs=self.tempo_msgs,
            ticks_per_beat=self.ticks_per_beat,
        )

    def _build_pedal_intervals(self) -> Dict[int, List[List[int]]]:
        """Returns a mapping of channels to sustain pedal intervals."""

        self.pedal_msgs.sort(key=lambda msg: msg["tick"])
        channel_to_pedal_intervals = defaultdict(list)
        pedal_status: Dict[int, int] = {}

        for pedal_msg in self.pedal_msgs:
            tick = pedal_msg["tick"]
            channel = pedal_msg["channel"]
            data = pedal_msg["data"]

            if data == 1 and pedal_status.get(channel, None) is None:
                pedal_status[channel] = tick
            elif data == 0 and pedal_status.get(channel, None) is not None:
                # Close pedal interval
                _start_tick = pedal_status[channel]
                _end_tick = tick
                channel_to_pedal_intervals[channel].append(
                    [_start_tick, _end_tick]
                )
                del pedal_status[channel]

        # Close all unclosed pedals at end of track
        final_tick = self.note_msgs[-1]["data"]["end"]
        for channel, start_tick in pedal_status.items():
            channel_to_pedal_intervals[channel].append([start_tick, final_tick])

        return channel_to_pedal_intervals

    def resolve_overlaps(self) -> "MidiDict":
        """Resolves any note overlaps (inplace) between notes with the same
        pitch and channel. This is achieved by converting a pair of notes with
        the same pitch (a<b<c, x,y>0):

        [a, b+x], [b-y, c] -> [a, b-y], [b-y, c]

        Note that this should not occur if the note messages have not been
        modified, e.g., by resolve_overlap().
        """

        # Organize notes by channel and pitch
        note_msgs_c: Dict[int, Dict[int, List[NoteMessage]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for msg in self.note_msgs:
            _channel = msg["channel"]
            _pitch = msg["data"]["pitch"]
            note_msgs_c[_channel][_pitch].append(msg)

        # We can modify notes by reference as they are dictionaries
        for channel, msgs_by_pitch in note_msgs_c.items():
            for pitch, msgs in msgs_by_pitch.items():
                msgs.sort(
                    key=lambda msg: (msg["data"]["start"], msg["data"]["end"])
                )
                prev_off_tick = -1
                for idx, msg in enumerate(msgs):
                    on_tick = msg["data"]["start"]
                    off_tick = msg["data"]["end"]
                    if prev_off_tick > on_tick:
                        # Adjust end of previous (idx - 1) msg to remove overlap
                        msgs[idx - 1]["data"]["end"] = on_tick
                    prev_off_tick = off_tick

        return self

    def resolve_pedal(self) -> "MidiDict":
        """Extend note offsets according to pedal and resolve any note overlaps"""

        # If has been already resolved, we don't recalculate
        if self.pedal_resolved == True:
            print("Pedal has already been resolved")

        # Organize note messages by channel
        note_msgs_c = defaultdict(list)
        for msg in self.note_msgs:
            _channel = msg["channel"]
            note_msgs_c[_channel].append(msg)

        # We can modify notes by reference as they are dictionaries
        channel_to_pedal_intervals = self._build_pedal_intervals()
        for channel, msgs in note_msgs_c.items():
            for msg in msgs:
                note_end_tick = msg["data"]["end"]
                for pedal_interval in channel_to_pedal_intervals[channel]:
                    pedal_start, pedal_end = pedal_interval
                    if pedal_start < note_end_tick < pedal_end:
                        msg["data"]["end"] = pedal_end
                        break

        self.resolve_overlaps()
        self.pedal_resolved = True

        return self

    # TODO: Needs to be refactored
    def remove_redundant_pedals(self) -> "MidiDict":
        """Removes redundant pedal messages from the MIDI data in place.

        Removes all pedal on/off message pairs that don't extend any notes.
        Makes an exception for pedal off messages that coincide exactly with
        note offsets.
        """

        def _is_pedal_useful(
            pedal_start_tick: int,
            pedal_end_tick: int,
            note_msgs: List[NoteMessage],
        ) -> bool:
            # This logic loops through the note_msgs that could possibly
            # be effected by the pedal which starts at pedal_start_tick
            # and ends at pedal_end_tick. If there is note effected by the
            # pedal, then it returns early.

            note_idx = 0
            note_msg = note_msgs[0]
            note_start = note_msg["data"]["start"]

            while note_start <= pedal_end_tick and note_idx < len(note_msgs):
                note_msg = note_msgs[note_idx]
                note_start, note_end = (
                    note_msg["data"]["start"],
                    note_msg["data"]["end"],
                )

                if pedal_start_tick <= note_end <= pedal_end_tick:
                    # Found note for which pedal is useful
                    return True

                note_idx += 1

            return False

        def _process_channel_pedals(channel: int) -> None:
            pedal_msg_idxs_to_remove = []
            pedal_down_tick = None
            pedal_down_msg_idx = None

            note_msgs = [
                msg for msg in self.note_msgs if msg["channel"] == channel
            ]

            if not note_msgs:
                # No notes to process. In this case we remove all pedal_msgs
                # and then return early.
                for pedal_msg_idx, pedal_msg in enumerate(self.pedal_msgs):
                    pedal_msg_value, pedal_msg_tick, _channel = (
                        pedal_msg["data"],
                        pedal_msg["tick"],
                        pedal_msg["channel"],
                    )

                    if _channel == channel:
                        pedal_msg_idxs_to_remove.append(pedal_msg_idx)

                # Remove messages
                self.pedal_msgs = [
                    msg
                    for _idx, msg in enumerate(self.pedal_msgs)
                    if _idx not in pedal_msg_idxs_to_remove
                ]
                return

            for pedal_msg_idx, pedal_msg in enumerate(self.pedal_msgs):
                pedal_msg_value, pedal_msg_tick, _channel = (
                    pedal_msg["data"],
                    pedal_msg["tick"],
                    pedal_msg["channel"],
                )

                # Only process pedal_msgs for specified MIDI channel
                if _channel != channel:
                    continue

                # Remove never-closed pedal messages
                if (
                    pedal_msg_idx == len(self.pedal_msgs) - 1
                    and pedal_msg_value == 1
                ):
                    # Current msg is last one and ON  -> remove curr pedal_msg
                    pedal_msg_idxs_to_remove.append(pedal_msg_idx)

                # Logic for removing repeated pedal messages and updating
                # pedal_down_tick and pedal_down_idx
                if pedal_down_tick is None:
                    if pedal_msg_value == 1:
                        # Pedal is OFF and current msg is ON -> update
                        pedal_down_tick = pedal_msg_tick
                        pedal_down_msg_idx = pedal_msg_idx
                        continue
                    else:
                        # Pedal is OFF and current msg is OFF -> remove curr pedal_msg
                        pedal_msg_idxs_to_remove.append(pedal_msg_idx)
                        continue
                else:
                    if pedal_msg_value == 1:
                        # Pedal is ON and current msg is ON -> remove curr pedal_msg
                        pedal_msg_idxs_to_remove.append(pedal_msg_idx)
                        continue

                pedal_is_useful = _is_pedal_useful(
                    pedal_start_tick=pedal_down_tick,
                    pedal_end_tick=pedal_msg_tick,
                    note_msgs=note_msgs,
                )

                if pedal_is_useful is False:
                    # Pedal hasn't effected any notes -> remove
                    pedal_msg_idxs_to_remove.append(pedal_down_msg_idx)
                    pedal_msg_idxs_to_remove.append(pedal_msg_idx)

                # Finished processing pedal, set pedal state to OFF
                pedal_down_tick = None
                pedal_down_msg_idx = None

            # Remove messages
            self.pedal_msgs = [
                msg
                for _idx, msg in enumerate(self.pedal_msgs)
                if _idx not in pedal_msg_idxs_to_remove
            ]

        for channel in set([msg["channel"] for msg in self.pedal_msgs]):
            _process_channel_pedals(channel)

        return self

    def remove_instruments(self, config: dict) -> "MidiDict":
        """Removes all messages with instruments specified in config at:

        data.preprocessing.remove_instruments

        Note that drum messages, defined as those which occur on MIDI channel 9
        are not removed.
        """

        programs_to_remove = [
            i
            for i in range(1, 127 + 1)
            if config[self.program_to_instrument[i]] is True
        ]
        channels_to_remove = [
            msg["channel"]
            for msg in self.instrument_msgs
            if msg["data"] in programs_to_remove
        ]

        # Remove drums (channel 9) from channels to remove
        channels_to_remove = [i for i in channels_to_remove if i != 9]

        # Remove unwanted messages all type by looping over msgs types
        _msg_dict: Dict[str, List] = {
            "meta_msgs": self.meta_msgs,
            "tempo_msgs": self.tempo_msgs,
            "pedal_msgs": self.pedal_msgs,
            "instrument_msgs": self.instrument_msgs,
            "note_msgs": self.note_msgs,
        }

        for msgs_name, msgs_list in _msg_dict.items():
            setattr(
                self,
                msgs_name,
                [
                    msg
                    for msg in msgs_list
                    if msg.get("channel", -1) not in channels_to_remove
                ],
            )

        return self


# TODO: The sign has been changed. Make sure this function isn't used anywhere else
def _extract_track_data(
    track: mido.MidiTrack,
) -> Tuple[
    List[MetaMessage],
    List[TempoMessage],
    List[PedalMessage],
    List[InstrumentMessage],
    List[NoteMessage],
]:
    """Converts MIDI messages into format used by MidiDict."""

    meta_msgs: List[MetaMessage] = []
    tempo_msgs: List[TempoMessage] = []
    pedal_msgs: List[PedalMessage] = []
    instrument_msgs: List[InstrumentMessage] = []
    note_msgs: List[NoteMessage] = []

    last_note_on = defaultdict(list)
    for message in track:
        # Meta messages
        if message.is_meta is True:
            if message.type == "text" or message.type == "copyright":
                meta_msgs.append(
                    {
                        "type": message.type,
                        "data": message.text,
                    }
                )
            # Tempo messages
            elif message.type == "set_tempo":
                tempo_msgs.append(
                    {
                        "type": "tempo",
                        "data": message.tempo,
                        "tick": message.time,
                    }
                )
        # Instrument messages
        elif message.type == "program_change":
            instrument_msgs.append(
                {
                    "type": "instrument",
                    "data": message.program,
                    "tick": message.time,
                    "channel": message.channel,
                }
            )
        # Pedal messages
        elif message.type == "control_change" and message.control == 64:
            # Consistent with pretty_midi and ableton-live default behavior
            pedal_msgs.append(
                {
                    "type": "pedal",
                    "data": 0 if message.value < 64 else 1,
                    "tick": message.time,
                    "channel": message.channel,
                }
            )
        # Note messages
        elif message.type == "note_on" and message.velocity > 0:
            last_note_on[(message.note, message.channel)].append(
                (message.time, message.velocity)
            )
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            # Ignore non-existent note-ons
            if (message.note, message.channel) in last_note_on:
                end_tick = message.time
                open_notes = last_note_on[(message.note, message.channel)]

                notes_to_close = [
                    (start_tick, velocity)
                    for start_tick, velocity in open_notes
                    if start_tick != end_tick
                ]
                notes_to_keep = [
                    (start_tick, velocity)
                    for start_tick, velocity in open_notes
                    if start_tick == end_tick
                ]

                for start_tick, velocity in notes_to_close:
                    note_msgs.append(
                        {
                            "type": "note",
                            "data": {
                                "pitch": message.note,
                                "start": start_tick,
                                "end": end_tick,
                                "velocity": velocity,
                            },
                            "tick": start_tick,
                            "channel": message.channel,
                        }
                    )

                if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
                    # Note-on on the same tick but we already closed
                    # some previous notes -> it will continue, keep it.
                    last_note_on[(message.note, message.channel)] = (
                        notes_to_keep
                    )
                else:
                    # Remove the last note on for this instrument
                    del last_note_on[(message.note, message.channel)]

    return meta_msgs, tempo_msgs, pedal_msgs, instrument_msgs, note_msgs


def midi_to_dict(mid: mido.MidiFile) -> MidiDictData:
    """Converts mid.MidiFile into MidiDictData representation.

    Additionally runs metadata extraction according to config specified at:

    data.metadata.functions

    Args:
        mid (mido.MidiFile): A mido file object to parse.

    Returns:
        MidiDictData: A dictionary containing extracted MIDI data including notes,
            time signatures, key signatures, and other musical events.
    """

    metadata_config = load_config()["data"]["metadata"]
    # Convert time in mid to absolute
    for track in mid.tracks:
        curr_tick = 0
        for message in track:
            message.time += curr_tick
            curr_tick = message.time

    midi_dict_data: MidiDictData = {
        "meta_msgs": [],
        "tempo_msgs": [],
        "pedal_msgs": [],
        "instrument_msgs": [],
        "note_msgs": [],
        "ticks_per_beat": mid.ticks_per_beat,
        "metadata": {},
    }

    # Compile track data
    for mid_track in mid.tracks:
        meta_msgs, tempo_msgs, pedal_msgs, instrument_msgs, note_msgs = (
            _extract_track_data(mid_track)
        )
        midi_dict_data["meta_msgs"] += meta_msgs
        midi_dict_data["tempo_msgs"] += tempo_msgs
        midi_dict_data["pedal_msgs"] += pedal_msgs
        midi_dict_data["instrument_msgs"] += instrument_msgs
        midi_dict_data["note_msgs"] += note_msgs

    # Sort by tick (for note msgs, this will be the same as data.start_tick)
    midi_dict_data["tempo_msgs"] = sorted(
        midi_dict_data["tempo_msgs"], key=lambda x: x["tick"]
    )
    midi_dict_data["pedal_msgs"] = sorted(
        midi_dict_data["pedal_msgs"], key=lambda x: x["tick"]
    )
    midi_dict_data["instrument_msgs"] = sorted(
        midi_dict_data["instrument_msgs"], key=lambda x: x["tick"]
    )
    midi_dict_data["note_msgs"] = sorted(
        midi_dict_data["note_msgs"], key=lambda x: x["tick"]
    )

    for metadata_process_name, metadata_process_config in metadata_config[
        "functions"
    ].items():
        if metadata_process_config["run"] is True:
            metadata_fn = get_metadata_fn(
                metadata_process_name=metadata_process_name
            )
            fn_args: Dict = metadata_process_config["args"]

            collected_metadata = metadata_fn(mid, midi_dict_data, **fn_args)
            if collected_metadata:
                for k, v in collected_metadata.items():
                    midi_dict_data["metadata"][k] = v

    return midi_dict_data


def dict_to_midi(mid_data: MidiDictData) -> mido.MidiFile:
    """Converts MIDI information from dictionary form into a mido.MidiFile.

    This function performs midi_to_dict in reverse.

    Args:
        mid_data (dict): MIDI information in dictionary form.

    Returns:
        mido.MidiFile: The MIDI parsed from the input data.
    """

    assert mid_data.keys() == {
        "meta_msgs",
        "tempo_msgs",
        "pedal_msgs",
        "instrument_msgs",
        "note_msgs",
        "ticks_per_beat",
        "metadata",
    }, "Invalid json/dict."

    ticks_per_beat = mid_data["ticks_per_beat"]

    # Add all messages (not ordered) to one track
    track = mido.MidiTrack()
    end_msgs = defaultdict(list)

    for tempo_msg in mid_data["tempo_msgs"]:
        track.append(
            mido.MetaMessage(
                "set_tempo", tempo=tempo_msg["data"], time=tempo_msg["tick"]
            )
        )

    for pedal_msg in mid_data["pedal_msgs"]:
        track.append(
            mido.Message(
                "control_change",
                control=64,
                value=pedal_msg["data"]
                * 127,  # Stored in PedalMessage as 1 or 0
                channel=pedal_msg["channel"],
                time=pedal_msg["tick"],
            )
        )

    for instrument_msg in mid_data["instrument_msgs"]:
        track.append(
            mido.Message(
                "program_change",
                program=instrument_msg["data"],
                channel=instrument_msg["channel"],
                time=instrument_msg["tick"],
            )
        )

    for note_msg in mid_data["note_msgs"]:
        # Note on
        track.append(
            mido.Message(
                "note_on",
                note=note_msg["data"]["pitch"],
                velocity=note_msg["data"]["velocity"],
                channel=note_msg["channel"],
                time=note_msg["data"]["start"],
            )
        )
        # Note off
        end_msgs[(note_msg["channel"], note_msg["data"]["pitch"])].append(
            (note_msg["data"]["start"], note_msg["data"]["end"])
        )

    # Only add end messages that don't interfere with other notes
    for k, v in end_msgs.items():
        channel, pitch = k
        for start, end in v:
            add = True
            for _start, _end in v:
                if start < _start < end < _end:
                    add = False

            if add is True:
                track.append(
                    mido.Message(
                        "note_on",
                        note=pitch,
                        velocity=0,
                        channel=channel,
                        time=end,
                    )
                )

    # Magic sorting function
    def _sort_fn(msg: mido.Message) -> Tuple[int, int]:
        if hasattr(msg, "velocity"):
            return (msg.time, msg.velocity)  # pyright: ignore
        else:
            return (msg.time, 1000)  # pyright: ignore

    # Sort and convert from abs_time -> delta_time
    track = sorted(track, key=_sort_fn)
    tick = 0
    for msg in track:
        msg.time -= tick
        tick += msg.time

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid = mido.MidiFile(type=0)
    mid.ticks_per_beat = ticks_per_beat
    mid.tracks.append(track)

    return mid


def get_duration_ms(
    start_tick: int,
    end_tick: int,
    tempo_msgs: List[TempoMessage],
    ticks_per_beat: int,
) -> int:
    """Calculates elapsed time (in ms) between start_tick and end_tick."""

    # Finds idx such that:
    # tempo_msg[idx]["tick"] < start_tick <= tempo_msg[idx+1]["tick"]
    for idx, curr_msg in enumerate(tempo_msgs):
        if start_tick <= curr_msg["tick"]:
            break
    if idx > 0:  # Special case idx == 0 -> Don't -1
        idx -= 1

    # It is important that we initialise curr_tick & curr_tempo here. In the
    # case that there is a single tempo message the following loop will not run.
    duration = 0.0
    curr_tick = start_tick
    curr_tempo = tempo_msgs[idx]["data"]

    # Sums all tempo intervals before tempo_msgs[-1]["tick"]
    for curr_msg, next_msg in zip(tempo_msgs[idx:], tempo_msgs[idx + 1 :]):
        curr_tempo = curr_msg["data"]
        if end_tick < next_msg["tick"]:
            delta_tick = end_tick - curr_tick
        else:
            delta_tick = next_msg["tick"] - curr_tick

        duration += tick2second(
            tick=delta_tick,
            tempo=curr_tempo,
            ticks_per_beat=ticks_per_beat,
        )

        if end_tick < next_msg["tick"]:
            break
        else:
            curr_tick = next_msg["tick"]

    # Case end_tick > tempo_msgs[-1]["tick"]
    if end_tick > tempo_msgs[-1]["tick"]:
        curr_tempo = tempo_msgs[-1]["data"]
        delta_tick = end_tick - curr_tick

        duration += tick2second(
            tick=delta_tick,
            tempo=curr_tempo,
            ticks_per_beat=ticks_per_beat,
        )

    # Convert from seconds to milliseconds
    duration = duration * 1e3
    duration = round(duration)

    return duration


def _match_word(text: str, word: str) -> bool:
    def to_ascii(s: str) -> str:
        # Remove accents
        normalized = unicodedata.normalize("NFKD", s)
        return "".join(c for c in normalized if not unicodedata.combining(c))

    text = to_ascii(text)
    word = to_ascii(word)

    # If name="bach" this pattern will match "bach", "Bach" or "BACH" if
    # it is either proceeded or preceded by a "_" or " ".
    pattern = (
        r"(^|[\s_])("
        + word.lower()
        + r"|"
        + word.upper()
        + r"|"
        + word.capitalize()
        + r")([\s_]|$)"
    )

    if re.search(pattern, text, re.IGNORECASE):
        return True
    else:
        return False


def meta_composer_filename(
    mid: mido.MidiFile, msg_data: MidiDictData, composer_names: list
) -> Dict[str, str]:
    file_name = Path(str(mid.filename)).stem
    matched_names_unique = set()
    for name in composer_names:
        if _match_word(file_name, name):
            matched_names_unique.add(name)

    # Only return data if only one composer is found
    matched_names = list(matched_names_unique)
    if len(matched_names) == 1:
        return {"composer": matched_names[0]}
    else:
        return {}


def meta_form_filename(
    mid: mido.MidiFile, msg_data: MidiDictData, form_names: list
) -> Dict[str, str]:
    file_name = Path(str(mid.filename)).stem
    matched_names_unique = set()
    for name in form_names:
        if _match_word(file_name, name):
            matched_names_unique.add(name)

    # Only return data if only one composer is found
    matched_names = list(matched_names_unique)
    if len(matched_names) == 1:
        return {"form": matched_names[0]}
    else:
        return {}


def meta_composer_metamsg(
    mid: mido.MidiFile, msg_data: MidiDictData, composer_names: list
) -> Dict[str, str]:
    matched_names_unique = set()
    for msg in msg_data["meta_msgs"]:
        for name in composer_names:
            if _match_word(msg["data"], name):
                matched_names_unique.add(name)

    # Only return data if only one composer is found
    matched_names = list(matched_names_unique)
    if len(matched_names) == 1:
        return {"composer": matched_names[0]}
    else:
        return {}


# TODO: Needs testing
def meta_maestro_json(
    mid: mido.MidiFile,
    msg_data: MidiDictData,
    composer_names: list,
    form_names: list,
) -> Dict[str, str]:
    """Loads composer and form metadata from MAESTRO metadata json file.


    This should only be used when processing MAESTRO, it requires maestro.json
    to be in the working directory. This json files contains MAESTRO metadata in
    the form file_name: {"composer": str, "title": str}.
    """

    _file_name = Path(str(mid.filename)).name
    _file_name_without_ext = os.path.splitext(_file_name)[0]
    metadata = load_maestro_metadata_json().get(
        _file_name_without_ext + ".midi", None
    )
    if metadata == None:
        return {}

    matched_forms_unique = set()
    for form in form_names:
        if _match_word(metadata["title"], form):
            matched_forms_unique.add(form)

    matched_composers_unique = set()
    for composer in composer_names:
        if _match_word(metadata["composer"], composer):
            matched_composers_unique.add(composer)

    res = {}
    matched_composers = list(matched_composers_unique)
    matched_forms = list(matched_forms_unique)
    if len(matched_forms) == 1:
        res["form"] = matched_forms[0]
    if len(matched_composers) == 1:
        res["composer"] = matched_composers[0]

    return res


def meta_abs_path(mid: mido.MidiFile, msg_data: MidiDictData) -> Dict[str, str]:
    return {"abs_path": str(Path(str(mid.filename)).absolute())}


def get_metadata_fn(
    metadata_process_name: str,
) -> Callable[Concatenate[mido.MidiFile, MidiDictData, ...], Dict[str, str]]:
    name_to_fn: Dict[
        str,
        Callable[Concatenate[mido.MidiFile, MidiDictData, ...], Dict[str, str]],
    ] = {
        "composer_filename": meta_composer_filename,
        "composer_metamsg": meta_composer_metamsg,
        "form_filename": meta_form_filename,
        "maestro_json": meta_maestro_json,
        "abs_path": meta_abs_path,
    }

    fn = name_to_fn.get(metadata_process_name, None)
    if fn is None:
        raise ValueError(
            f"Error finding metadata function for {metadata_process_name}"
        )
    else:
        return fn


def test_max_programs(midi_dict: MidiDict, max: int) -> Tuple[bool, int]:
    """Returns false if midi_dict uses more than {max} programs."""
    present_programs = set(
        map(
            lambda msg: msg["data"],
            midi_dict.instrument_msgs,
        )
    )

    if len(present_programs) <= max:
        return True, len(present_programs)
    else:
        return False, len(present_programs)


def test_max_instruments(midi_dict: MidiDict, max: int) -> Tuple[bool, int]:
    present_instruments = set(
        map(
            lambda msg: midi_dict.program_to_instrument[msg["data"]],
            midi_dict.instrument_msgs,
        )
    )

    if len(present_instruments) <= max:
        return True, len(present_instruments)
    else:
        return False, len(present_instruments)


def test_note_frequency(
    midi_dict: MidiDict, max_per_second: float, min_per_second: float
) -> Tuple[bool, float]:
    if not midi_dict.note_msgs:
        return False, 0.0

    num_notes = len(midi_dict.note_msgs)
    total_duration_ms = get_duration_ms(
        start_tick=midi_dict.note_msgs[0]["data"]["start"],
        end_tick=midi_dict.note_msgs[-1]["data"]["end"],
        tempo_msgs=midi_dict.tempo_msgs,
        ticks_per_beat=midi_dict.ticks_per_beat,
    )

    if total_duration_ms == 0:
        return False, 0.0

    notes_per_second = (num_notes * 1e3) / total_duration_ms

    if notes_per_second < min_per_second or notes_per_second > max_per_second:
        return False, notes_per_second
    else:
        return True, notes_per_second


def test_note_frequency_per_instrument(
    midi_dict: MidiDict, max_per_second: float, min_per_second: float
) -> Tuple[bool, float]:
    num_instruments = len(
        set(
            map(
                lambda msg: midi_dict.program_to_instrument[msg["data"]],
                midi_dict.instrument_msgs,
            )
        )
    )

    if not midi_dict.note_msgs:
        return False, 0.0

    num_notes = len(midi_dict.note_msgs)
    total_duration_ms = get_duration_ms(
        start_tick=midi_dict.note_msgs[0]["data"]["start"],
        end_tick=midi_dict.note_msgs[-1]["data"]["end"],
        tempo_msgs=midi_dict.tempo_msgs,
        ticks_per_beat=midi_dict.ticks_per_beat,
    )

    if total_duration_ms == 0:
        return False, 0.0

    notes_per_second = (num_notes * 1e3) / total_duration_ms

    note_freq_per_instrument = notes_per_second / num_instruments
    if (
        note_freq_per_instrument < min_per_second
        or note_freq_per_instrument > max_per_second
    ):
        return False, note_freq_per_instrument
    else:
        return True, note_freq_per_instrument


def test_min_length(
    midi_dict: MidiDict, min_seconds: int
) -> Tuple[bool, float]:
    if not midi_dict.note_msgs:
        return False, 0.0

    total_duration_ms = get_duration_ms(
        start_tick=midi_dict.note_msgs[0]["data"]["start"],
        end_tick=midi_dict.note_msgs[-1]["data"]["end"],
        tempo_msgs=midi_dict.tempo_msgs,
        ticks_per_beat=midi_dict.ticks_per_beat,
    )

    if total_duration_ms / 1e3 < min_seconds:
        return False, total_duration_ms / 1e3
    else:
        return True, total_duration_ms / 1e3


def get_test_fn(
    test_name: str,
) -> Callable[Concatenate[MidiDict, ...], Tuple[bool, Any]]:
    name_to_fn: Dict[
        str, Callable[Concatenate[MidiDict, ...], Tuple[bool, Any]]
    ] = {
        "max_programs": test_max_programs,
        "max_instruments": test_max_instruments,
        "total_note_frequency": test_note_frequency,
        "note_frequency_per_instrument": test_note_frequency_per_instrument,
        "min_length": test_min_length,
    }

    fn = name_to_fn.get(test_name, None)
    if fn is None:
        raise ValueError(
            f"Error finding preprocessing function for {test_name}"
        )
    else:
        return fn
