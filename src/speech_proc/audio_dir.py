"""
Copyright 2025 Balacoon

Audio directory - object for handling audio dir, reading audio
"""

import glob
import os
from typing import Optional

import resampy
import soundfile as sf
import tqdm

AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "ogg"]


class AudioDir:
    def __init__(self, path: str):
        self._path = path

    def get_ids(
        self,
        to_sort: bool = False,
        expected_sample_rate: Optional[int] = None,
        max_dur: Optional[float] = None,
    ) -> list[str]:
        ids = []
        durations = []
        for path in tqdm.tqdm(
            glob.glob(os.path.join(self._path, "*")), desc="Getting audio IDs"
        ):
            if not any(path.endswith(x) for x in AUDIO_EXTENSIONS):
                continue
            if to_sort:
                info = self.get_info(path)
                if not self.is_valid_info(info, expected_sample_rate, max_dur):
                    continue
                durations.append(info[-1])
            ids.append(os.path.splitext(os.path.basename(path))[0])
        if to_sort:
            # Use zip to pair ids and durations, then sort by duration and extract ids
            sorted_pairs = sorted(
                zip(ids, durations), key=lambda pair: pair[1], reverse=True
            )
            ids = [pair[0] for pair in sorted_pairs]
        return ids

    def get_path(self, name: str) -> str:
        for suffix in AUDIO_EXTENSIONS:
            path = os.path.join(self._path, name + f".{suffix}")
            if os.path.isfile(path):
                return path
        return None

    def get_info(self, name: str) -> tuple[int, int, int, float]:
        """
        Reads meta info of an audio file, returning
        sample_rate, number of channels, precision, and duration.
        """
        if os.path.isfile(name):
            path = name
        else:
            path = self.get_path(name)

        try:
            info = sf.info(path)
        except RuntimeError:
            raise ValueError(f"Unsupported or corrupted audio file: {path}")

        sample_rate = info.samplerate
        channels = info.channels
        duration = info.duration
        precision = (
            int(info.subtype.split("-")[0]) if "-" in info.subtype else 16
        )  # Extract bit depth

        return sample_rate, channels, precision, duration

    def get_duration(self, name: str) -> float:
        _, _, _, duration = self.get_info(name)
        return duration

    def is_valid(
        self, name: str, expected_sample_rate: Optional[int] = None, max_dur: Optional[float] = None
    ) -> bool:
        path = self.get_path(name)
        if not path:
            return False
        return self.is_valid_info(self.get_info(name), expected_sample_rate, max_dur)

    def is_valid_info(
        self, info, expected_sample_rate: Optional[int] = None, max_dur: Optional[float] = None
    ) -> bool:
        sample_rate, channels, precision, duration = info
        if expected_sample_rate and expected_sample_rate > sample_rate:
            # this would require upsampling thats why we call it out
            # TODO: allow this with extra option
            return False
        if channels != 1 or precision != 16:
            return False
        if max_dur and duration > max_dur:
            return False
        return True

    def read(self, name: str, dtype="int16", sample_rate: Optional[int] = None):
        path = self.get_path(name)
        data, orig_sample_rate = sf.read(path, dtype=dtype)
        if sample_rate and sample_rate != orig_sample_rate:
            data = resampy.resample(
                data, orig_sample_rate, sample_rate, filter="kaiser_fast"
            )
        return data, sample_rate or orig_sample_rate
