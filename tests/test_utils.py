# Copyright 2025 Balacoon

import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import soundfile as sf


@contextmanager
def create_dummy_audio(sample_rate=24000, num_files=3, duration=1.0):
    """
    Creates a temporary directory with dummy audio files.

    Args:
        sample_rate (int): Sample rate of the audio.
        num_files (int): Number of audio files to create.
        duration (float): Duration of each audio file in seconds.

    Yields:
        str: Path to the temporary directory.
    """
    temp_dir = tempfile.mkdtemp()

    try:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sine_wave = 1.0 * np.sin(2 * np.pi * 440 * t)
        for i in range(num_files):
            file_path = os.path.join(temp_dir, f"dummy_{i}.wav")
            samples = np.random.rand(int(sample_rate * duration))
            samples += sine_wave
            samples *= 32767
            samples = np.clip(samples, -32768, 32767)
            samples = samples.astype(np.int16)
            sf.write(file_path, samples.astype(np.int16), sample_rate)

        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
        pass
