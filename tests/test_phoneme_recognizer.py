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

from speech_proc.phoneme_recognizer import extract_features


def test_phoneme_recognizer():
    with create_dummy_audio(
        sample_rate=16000, num_files=5, duration=2.0
    ) as dummy_audio_dir:
        out_dir = "tmp_phoneme_features"
        extract_features(
            audio_dir_path=dummy_audio_dir,
            output_dir=out_dir,
            language="en",
            batch_size=2,
            device="cpu",
        )
        paths = list(glob.glob(os.path.join(out_dir, "*.npz")))
        assert len(paths) == 5
        for p in paths:
            archive = np.load(p)
            # Check that we have phoneme features (excluding "phone" and "tone")
            # The exact features depend on the model, but we expect at least some features
            assert len(archive.files) > 0, "Expected at least one feature in output"

            # Each feature should be an array of int32 values representing phoneme attributes
            for feature_name in archive.files:
                arr = archive[feature_name]
                # Features should have reasonable length (20ms frames for 2 seconds)
                # 2 seconds / 0.02 = 100 frames
                assert arr.shape[0] > 0, f"Feature {feature_name} should have frames"
                assert (
                    arr.shape[0] <= 100
                ), f"Feature {feature_name} has too many frames"
                # Should be int32 as specified in the code
                assert arr.dtype == np.int32
                # Values should be non-negative (argmax indices)
                assert np.all(arr >= 0)
        shutil.rmtree(out_dir)
