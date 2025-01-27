"""Contains abstract tokenizer class."""

import functools

from typing import (
    Any,
    Final,
    Callable,
    TypeAlias,
)

from ariautils.midi import MidiDict


SpecialToken: TypeAlias = str
Token: TypeAlias = tuple[Any, ...] | str


class Tokenizer:
    """Abstract Tokenizer class for tokenizing MidiDict objects."""

    def __init__(self) -> None:
        self.name: str = ""

        self.bos_tok: Final[SpecialToken] = "<S>"
        self.eos_tok: Final[SpecialToken] = "<E>"
        self.pad_tok: Final[SpecialToken] = "<P>"
        self.unk_tok: Final[SpecialToken] = "<U>"
        self.dim_tok: Final[SpecialToken] = "<D>"

        self.special_tokens: list[SpecialToken] = [
            self.bos_tok,
            self.eos_tok,
            self.pad_tok,
            self.unk_tok,
            self.dim_tok,
        ]

        # These must be implemented in child class (abstract params)
        self.config: dict[str, Any] = {}
        self.vocab: tuple[Token, ...] = ()
        self.instruments_wd: list[str] = []
        self.instruments_nd: list[str] = []
        self.tok_to_id: dict[Token, int] = {}
        self.id_to_tok: dict[int, Token] = {}
        self.vocab_size: int = -1
        self.pad_id: int = -1

    def _tokenize_midi_dict(self, midi_dict: MidiDict) -> list[Token]:
        """Abstract method for tokenizing a MidiDict object into a sequence of
        tokens.

        Args:
            midi_dict (MidiDict): The MidiDict to tokenize.

        Returns:
            list[Token]: A sequence of tokens representing the MIDI content.
        """

        raise NotImplementedError

    def tokenize(self, midi_dict: MidiDict, **kwargs: Any) -> list[Token]:
        """Tokenizes a MidiDict object.

        This function should be overridden if additional transformations are
        required, e.g., adding additional tokens. The default behavior is to
        call tokenize_midi_dict.

        Args:
            midi_dict (MidiDict): The MidiDict to tokenize.
            **kwargs (Any): Additional keyword arguments passed to _tokenize_midi_dict.

        Returns:
            list[Token]: A sequence of tokens representing the MIDI content.
        """

        return self._tokenize_midi_dict(midi_dict, **kwargs)

    def _detokenize_midi_dict(self, tokenized_seq: list[Token]) -> MidiDict:
        """Abstract method for de-tokenizing a sequence of tokens into a
        MidiDict Object.

        Args:
            tokenized_seq (list[Token]): The sequence of tokens to detokenize.

        Returns:
            MidiDict: A MidiDict reconstructed from the tokens.
        """

        raise NotImplementedError

    def detokenize(self, tokenized_seq: list[Token], **kwargs: Any) -> MidiDict:
        """Detokenizes a MidiDict object.

        This function should be overridden if additional are required during
        detokenization. The default behavior is to call detokenize_midi_dict.

        Args:
            tokenized_seq (list): The sequence of tokens to detokenize.
            **kwargs (Any): Additional keyword arguments passed to detokenize_midi_dict.

        Returns:
            MidiDict: A MidiDict reconstructed from the tokens.
        """

        return self._detokenize_midi_dict(tokenized_seq, **kwargs)

    def export_data_aug(cls) -> list[Callable[[list[Token]], list[Token]]]:
        """Export a list of implemented data augmentation functions."""

        raise NotImplementedError

    def encode(self, unencoded_seq: list[Token]) -> list[int]:
        """Converts tokenized sequence into the corresponding list of ids."""

        def _enc_fn(tok: Token) -> int:
            return self.tok_to_id.get(tok, self.tok_to_id[self.unk_tok])

        if self.tok_to_id is None:
            raise NotImplementedError("tok_to_id")

        encoded_seq = [_enc_fn(tok) for tok in unencoded_seq]

        return encoded_seq

    def decode(self, encoded_seq: list[int]) -> list[Token]:
        """Converts list of ids into the corresponding list of tokens."""

        def _dec_fn(id: int) -> Token:
            return self.id_to_tok.get(id, self.unk_tok)

        if self.id_to_tok is None:
            raise NotImplementedError("id_to_tok")

        decoded_seq = [_dec_fn(idx) for idx in encoded_seq]

        return decoded_seq

    @classmethod
    def _find_closest_int(cls, n: int, sorted_list: list[int]) -> int:
        # Selects closest integer to n from sorted_list
        # Time ~ Log(n)

        if not sorted_list:
            raise ValueError("List is empty")

        left, right = 0, len(sorted_list) - 1
        closest = float("inf")

        while left <= right:
            mid = (left + right) // 2
            diff = abs(sorted_list[mid] - n)

            if diff < abs(closest - n):
                closest = sorted_list[mid]

            if sorted_list[mid] < n:
                left = mid + 1
            else:
                right = mid - 1

        return closest  # type: ignore[return-value]

    def add_tokens_to_vocab(self, tokens: list[Token] | tuple[Token]) -> None:
        """Utility function for safely adding extra tokens to vocab."""

        for token in tokens:
            assert token not in self.vocab

        self.vocab = self.vocab + tuple(tokens)
        self.tok_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_tok = {v: k for k, v in self.tok_to_id.items()}
        self.vocab_size = len(self.vocab)

    def export_aug_fn_concat(
        self, aug_fn: Callable[[list[Token]], list[Token]]
    ) -> Callable[[list[Token]], list[Token]]:
        """Transforms an augmentation function for concatenated sequences.

        This is useful for augmentation functions that are only defined for
        sequences which start with "<S>" and end with "<E>".

        Args:
            aug_fn (Callable[[list[Token]], list[Token]]): The augmentation
                function to transform.

        Returns:
            Callable[[list[Token]], list[Token]]: A transformed augmentation
                function that can handle concatenated sequences.
        """

        def _aug_fn_concat(
            src: list[Token],
            _aug_fn: Callable[[list[Token]], list[Token]],
            pad_tok: str,
            eos_tok: str,
            **kwargs: Any,
        ) -> list[Token]:
            # Split list on "<E>"
            initial_seq_len = len(src)
            src_sep = []
            prev_idx = 0
            for curr_idx, tok in enumerate(src, start=1):
                if tok == eos_tok:
                    src_sep.append(src[prev_idx:curr_idx])
                    prev_idx = curr_idx

            # Last sequence
            if prev_idx != curr_idx:
                src_sep.append(src[prev_idx:])

            # Augment
            src_sep = [
                _aug_fn(
                    _src,
                    **kwargs,
                )
                for _src in src_sep
            ]

            # Concatenate
            src_aug_concat = [tok for src_aug in src_sep for tok in src_aug]

            # Pad or truncate to original sequence length as necessary
            src_aug_concat = src_aug_concat[:initial_seq_len]
            src_aug_concat += [pad_tok] * (
                initial_seq_len - len(src_aug_concat)
            )

            return src_aug_concat

        return functools.partial(
            _aug_fn_concat,
            _aug_fn=aug_fn,
            pad_tok=self.pad_tok,
            eos_tok=self.eos_tok,
        )
