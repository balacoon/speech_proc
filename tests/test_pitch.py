# Copyright 2025 Balacoon

import glob
import os
import shutil
import sys

# Get the directory of the current test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.append(TEST_DIR)

import numpy as np
from test_utils import create_dummy_audio  # Now directly importable

from speech_proc.pitch import extract_pitch


def test_semantic_tokens():
    with create_dummy_audio(
        sample_rate=24000, num_files=20, duration=10.0
    ) as dummy_audio_dir:
        out_dir = "tmp_pitch"
        extract_pitch(
            dummy_audio_dir,
            out_dir,
            nproc=4,
        )
        paths = list(glob.glob(os.path.join(out_dir, "*.npz")))
        assert len(paths) == 20
        for p in paths:
            archive = np.load(p)
            assert "pitch" in archive
            arr = archive["pitch"]
            assert arr.shape == (500,)  # 50 per second, minus one for win_size
            assert np.all(arr >= -30)
            assert np.all(arr < 30)
            assert not np.all(arr == 0)
        shutil.rmtree(out_dir)
