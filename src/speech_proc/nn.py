"""
Copyright 2025 Balacoon

Applies neural network to audio directory
"""

import argparse
import concurrent
import logging
import os

import numpy as np
import torch
import tqdm
import time
from concurrent.futures import ThreadPoolExecutor

from speech_proc.audio_dir import AudioDir



def append_arguments(parser: argparse.ArgumentParser):
    """
    Appends arguments related to applying neural network
    to the data. Takes input/output directories, optional ids file
    that controls which utterances to work with, batch size, max duration
    """
    parser.add_argument(
        "-i",
        "--in-dir",
        required=True,
        help="Location of the directory with input audio data",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        help="Directory to create and put output data to",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="File with utterance ids, controlling which utterances to process",
    )
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Number of utterances to process in parallel.",
    )
    parser.add_argument(
        "--max-dur",
        default=None,
        type=float,
        help="Maximum duration of the audio files to process.",
    )


class NN:
    def __init__(
        self,
        in_audio_dir: AudioDir,
        out_dir: str,
        batch_size: int,
        sample_rate: int,
        max_dur: float,
        ids: list[str] = None,
    ):
        self._in_audio_dir = in_audio_dir
        self._out_dir = out_dir
        self._batch_size = batch_size
        self._max_dur = max_dur
        self._sample_rate = sample_rate
        self._model = None

        self._ids = ids if ids else self._in_audio_dir.get_ids()

        # Parallel validation of audio files
        def validate_id(audio_id):
            return (
                audio_id
                if self._in_audio_dir.is_valid(
                    audio_id, self._sample_rate, self._max_dur
                )
                else None
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm.tqdm(
                    executor.map(validate_id, self._ids),
                    total=len(self._ids),
                    desc="Validating audio files",
                )
            )

        ids_filt = [x for x in results if x is not None]
        logging.info(
            f"Reduced {len(self._ids)} audio files to {len(ids_filt)} after validation"
        )

        # Parallel retrieval of audio durations
        def get_duration(audio_id):
            return audio_id, self._in_audio_dir.get_duration(audio_id)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            duration_results = list(
                tqdm.tqdm(
                    executor.map(get_duration, ids_filt),
                    total=len(ids_filt),
                    desc="Collecting audio durations",
                )
            )

        id2dur = dict(duration_results)

        # Sort by duration (longest first)
        self._ids = sorted(ids_filt, key=lambda x: id2dur[x], reverse=True)

    def load_model(self, path: str):
        self._model = torch.jit.load(path).cuda()

    @staticmethod
    def pad_samples(arr: np.ndarray, expected_len: int) -> np.ndarray:
        """
        Pads samples array to the desired length
        """
        assert arr.ndim == 1
        current_len = arr.shape[0]
        if current_len == expected_len:
            return arr
        assert current_len < expected_len
        diff = expected_len - current_len
        padded_arr = np.pad(arr, (0, diff))
        return padded_arr

    def get_data(self, names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads audio data. Pads and packs it into batches.
        """
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(lambda x: self._in_audio_dir.read(x, sample_rate=self._sample_rate)[0], names))
        data_len = [x.shape[0] for x in data]
        padded_data = [self.pad_samples(x, max(data_len)) for x in data]
        # convert to torch tensors
        padded_data = [torch.tensor(x, dtype=torch.int16) for x in padded_data]
        batch = torch.stack(padded_data).cuda()
        data_len = torch.tensor(data_len, dtype=torch.int).cuda()
        return batch, data_len

    def forward(self, data: torch.Tensor, data_len: torch.Tensor) -> torch.Tensor:
        if not self._model:
            raise RuntimeError("Model is not loaded, did you call `load_model`")
        return self._model(data, data_len)

    def save_data(
        self,
        output: np.ndarray,
        names: list[str],
        data_len: list[int],
        out_stream: str,
        dtype: str,
    ):
        """
        Stores data in npz directory.

        If the npz file already exists, it updates the archive by adding new data
        or overwriting the specified stream if it already exists.

        Parameters:
            output (np.ndarray): Array of extracted features.
            names (list[str]): List of file names (without extension).
            data_len (list[int]): List of data lengths corresponding to each entry.
            out_stream (str): The key under which to store the data.
            dtype (str): which type to cast data to
        """
        assert output.shape[0] == len(names)
        assert len(names) == len(data_len)
        assert output.ndim in {2, 3}

        max_frames_num = output.shape[1]  # Get the max frames along the given dimension
        max_samples_num = max(data_len)

        for i, name in enumerate(names):
            samples_num = data_len[i]
            percent = samples_num / max_samples_num
            frames_num = int(percent * max_frames_num)

            # Slice based on frame_dim (keep other dimensions unchanged)
            arr = output[i, :frames_num, ...]  # Slice along 1st axis (after batch)

            out_path = os.path.join(self._out_dir, name + ".npz")
            # Load existing data if file exists
            if os.path.exists(out_path):
                with np.load(out_path, allow_pickle=True) as existing_data:
                    archive = dict(existing_data)  # Convert to mutable dict
            else:
                archive = {}

            # Update or add the new stream
            archive[out_stream] = arr.astype(dtype)

            # Save back to npz
            np.savez(out_path, **archive)

    def save_data_wrap(self, output: np.ndarray, names: list[str], data_len: list[int]):
        raise NotImplementedError

    @staticmethod
    def description():
        return "NN"

    def process(self):
        os.makedirs(self._out_dir, exist_ok=True)
        for i in tqdm.tqdm(
            range(0, len(self._ids), self._batch_size),
            desc=f"Applying {self.description()}",
        ):
            batch_names = self._ids[i : i + self._batch_size]
            data, data_len = self.get_data(batch_names)
            out_data = self.forward(data, data_len)
            self.save_data_wrap(
                out_data.detach().cpu().numpy(),
                batch_names,
                data_len.detach().cpu().numpy().tolist(),
            )
