"""
Copyright 2025 Balacoon

Parallelized extraction of pitch from audio.
Requires SPTK to be installed.
"""

import argparse
import logging
import os
import shutil
import struct
import subprocess
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import numpy as np
import tqdm

from speech_proc.audio_dir import AudioDir

FRAME_RATE = 50
FMIN = 50.0
FMAX = 400.0
MODES = ["linear", "log", "relative_semitones"]


def check_sptk():
    """Checks if the SPTK `pitch` command is available in the environment."""
    if shutil.which("pitch") is None:
        raise RuntimeError(
            "SPTK `pitch` command is not available. Please install SPTK or add it to path."
        )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Parallel script for pitch (f0) extraction."
    )
    ap.add_argument(
        "-i", "--in-dir", required=True, help="Directory with input audio data"
    )
    ap.add_argument(
        "-o", "--out-dir", required=True, help="Directory to store output data"
    )
    ap.add_argument("--ids", default=None, help="File with utterance ids to process")
    ap.add_argument(
        "--frame-rate", type=int, default=FRAME_RATE, help="Number of frames per second"
    )
    ap.add_argument("--fmin", type=float, default=FMIN, help="Minimum expected pitch")
    ap.add_argument("--fmax", type=float, default=FMAX, help="Maximum expected pitch")
    ap.add_argument(
        "--out-type",
        choices=MODES,
        default=MODES[-1],
        help=f"Type of output ({str(MODES)})",
    )
    ap.add_argument(
        "--nproc",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use",
    )
    return ap.parse_args()


class PitchExtractor:
    """Handles pitch extraction for a subset of utterances in parallel."""

    def __init__(
        self,
        audio_dir: AudioDir,
        out_dir: str,
        frame_rate: int,
        fmin: float,
        fmax: float,
        out_type: str,
    ):
        self._audio_dir = audio_dir
        self._out_dir = out_dir
        self._frame_rate = frame_rate
        self._fmin = fmin
        self._fmax = fmax
        self._out_type = out_type
        os.makedirs(self._out_dir, exist_ok=True)

    def _extract_f0(self, data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extracts fundamental frequency (f0) using SPTK `pitch`."""
        sr_khz = sample_rate / 1000.0
        frame_shift = int(sample_rate / self._frame_rate)  # Compute frame shift

        # Pack data into binary format for processing
        packed_data = struct.pack(f"{len(data)}f", *data.tolist())

        args = [
            "pitch",
            "-s",
            str(sr_khz),
            "-p",
            str(frame_shift),
            "-L",
            str(self._fmin),
            "-H",
            str(self._fmax),
            "-o",
            "1",
        ]

        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
        )
        f0, _ = process.communicate(packed_data)

        # Unpack binary output
        f0 = np.array(struct.unpack(f"{len(f0) // 4}f", f0))
        f0[f0 < 0] = 0  # Replace negative values with zero (unvoiced frames)

        return f0

    def _convert_f0(self, f0: np.ndarray) -> np.ndarray:
        """Converts f0 values based on the selected output type."""
        if self._out_type == "linear":
            return f0
        elif self._out_type == "log":
            return np.where(f0 > 0, np.log(f0), 0)
        elif self._out_type == "relative_semitones":
            ref_freq = 440.0  # A4 reference frequency
            semitones = np.where(f0 > 0, 12 * np.log2(f0 / ref_freq) + 69, 0)
            median_semitone = np.median(semitones[semitones > 0])
            return np.where(f0 > 0, np.round(semitones - median_semitone), 0)
        else:
            raise ValueError(f"Unsupported output type: {self._out_type}")

    def process_id(self, name: str):
        """Extracts and saves pitch for a single utterance."""
        data, sr = self._audio_dir.read(name, dtype="int16")
        f0 = self._extract_f0(data, sr)
        f0 = self._convert_f0(f0)

        # Save to .npz file
        out_path = os.path.join(self._out_dir, f"{name}.npz")
        np.savez(out_path, pitch=f0)


def run_extraction(args):
    """Wrapper function to allow parallel processing."""
    extractor, ids = args
    for name in tqdm.tqdm(ids, desc=f"Processing {len(ids)} files", position=0):
        extractor.process_id(name)


def extract_pitch(
    in_dir: str,
    out_dir: str,
    ids_path: str = None,
    out_type: str = MODES[-1],
    nproc: int = cpu_count(),
    frame_rate: int = FRAME_RATE,
    fmin: float = FMIN,
    fmax: float = FMAX,
):
    """Runs parallel pitch extraction."""
    check_sptk()  # Ensure SPTK is installed
    audio_dir = AudioDir(in_dir)
    extractor = PitchExtractor(audio_dir, out_dir, frame_rate, fmin, fmax, out_type)
    # Split IDs into N equal parts for parallel execution
    if ids_path:
        with open(ids_path, "r", encoding="utf-8") as fp:
            ids = [x.strip().split()[0] for x in fp.readlines()]
    else:
        ids = audio_dir.get_ids()
    id_chunks = np.array_split(ids, nproc)

    with Pool(nproc) as pool:
        pool.map(run_extraction, [(extractor, chunk) for chunk in id_chunks])


def main():
    """Main entry point for the script."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    extract_pitch(
        args.in_dir,
        args.out_dir,
        ids=args.ids,
        out_type=args.out_type,
        nproc=args.nproc,
        frame_rate=args.frame_rate,
        fmin=args.fmin,
        fmax=args.fmax,
    )
