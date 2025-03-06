"""
Copyright 2025 Balacoon

Extracts acoustic tokens from audio
"""

import argparse
import logging
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from speech_proc.audio_dir import AudioDir
from speech_proc.nn import NN, append_arguments


def parse_args():
    ap = argparse.ArgumentParser(description="Extracts acoustic tokens from audio")
    append_arguments(ap)
    return ap.parse_args()


class AcousticTokensExtractor(NN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: add options which extractor to use
        local_path = hf_hub_download(
            repo_id="balacoon/vq4_50fps_24khz_vocoder",
            filename="analysis.jit",
        )
        self.load_model(local_path)

    def forward(self, data: torch.Tensor, data_len: torch.Tensor) -> torch.Tensor:
        return self._model(data)

    def save_data_wrap(self, output: np.ndarray, names: list[str], data_len: list[int]):
        self.save_data(
            output,
            names,
            data_len,
            "acoustic_tokens",
            "int16",
        )

    @staticmethod
    def description():
        return "Acoustic Tokens"


BATCH_SIZE = 8
MAX_DUR = 30.0


def extract_acoustic_tokens(
    in_dir: str,
    out_dir: str,
    ids_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_dur: Optional[float] = None,
):
    if not max_dur:
        max_dur = MAX_DUR
    if not batch_size:
        batch_size = BATCH_SIZE
    ids = None
    if ids_path:
        with open(ids_path, "r", encoding="utf-8") as fp:
            ids = [x.strip().split()[0] for x in fp.readlines()]
    audio_dir = AudioDir(in_dir)
    extractor = AcousticTokensExtractor(
        audio_dir,
        out_dir,
        batch_size,
        24000,
        max_dur,
        ids=ids,
    )
    extractor.process()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    extract_acoustic_tokens(
        args.in_dir,
        args.out_dir,
        args.ids,
        args.batch_size,
        args.max_dur,
    )
