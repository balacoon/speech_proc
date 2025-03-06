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

from speech_proc.semantic_tokens import extract_semantic_tokens


def test_semantic_tokens():
    with create_dummy_audio(
        sample_rate=16000, num_files=20, duration=10.0
    ) as dummy_audio_dir:
        out_dir = "tmp_semantic_tokens"
        extract_semantic_tokens(
            dummy_audio_dir,
            out_dir,
        )
        paths = list(glob.glob(os.path.join(out_dir, "*.npz")))
        assert len(paths) == 20
        for p in paths:
            archive = np.load(p)
            assert "semantic_tokens" in archive
            arr = archive["semantic_tokens"]
            assert arr.shape == (499,)  # 50 per second, minus one for win_size
            assert np.all(arr >= 0)
            assert np.all(arr < 1000)
            assert not np.all(arr == 0)
        shutil.rmtree(out_dir)
