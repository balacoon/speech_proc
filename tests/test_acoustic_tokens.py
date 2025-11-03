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

from speech_proc.nano_acoustic_tokens import extract_tokens


def test_nano_acoustic_tokens():
    with create_dummy_audio(
        sample_rate=22050, num_files=20, duration=10.0
    ) as dummy_audio_dir:
        out_dir = "tmp_acoustic_tokens"
        extract_tokens(
            audio_dir_path=dummy_audio_dir,
            output_dir=out_dir,
            batch_size=8,
            max_dur=15.0,
            device="cpu",
        )
        paths = list(glob.glob(os.path.join(out_dir, "*.npz")))
        assert len(paths) == 20
        for p in paths:
            archive = np.load(p)
            assert "acoustic_tokens" in archive
            arr = archive["acoustic_tokens"]
            assert arr.shape == (8, 216)
            assert np.all(arr >= 0)
            assert np.all(arr < 2025)
            assert not np.all(arr == 0)
        shutil.rmtree(out_dir)
