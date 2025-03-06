"""
Copyright 2025 Balacoon

Audio directory - object for handling audio dir, reading audio
"""

import glob
import os
from typing import Optional

import soundfile as sf

AUDIO_EXTENSIONS = ["wav", "mp3", "flac", "ogg"]


class AudioDir:
    def __init__(self, path: str):
        self._path = path

    def get_ids(self) -> list[str]:
        ids = []
        for path in glob.glob(os.path.join(self._path, "*")):
            if any(path.endswith(x) for x in AUDIO_EXTENSIONS):
                ids.append(os.path.splitext(os.path.basename(path))[0])
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
        self, name: str, expected_sample_rate: int, max_dur: Optional[float] = None
    ) -> bool:
        path = self.get_path(name)
        if not path:
            return False
        sample_rate, channels, precision, duration = self.get_info(name)
        if expected_sample_rate != sample_rate:
            return False
        if channels != 1 or precision != 16:
            return False
        if max_dur and duration > max_dur:
            return False
        return True

    def read(self, name: str, dtype="int16"):
        path = self.get_path(name)
        data, sample_rate = sf.read(path, dtype=dtype)
        return data, sample_rate
