"""
Copyright 2025 Balacoon

Speaker embedding extraction using ONNX-exported Qwen3-TTS speaker encoder.
Expects an ONNX model that takes raw 24 kHz float32 mono audio and returns
a speaker embedding vector (see export_speaker_encoder.py).
"""

import argparse
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
import tqdm

from speech_proc.audio_dir import AudioDir

SAMPLE_RATE = 24000


def extract_speaker_embeddings(
    audio_dir_path: str,
    output_dir: str,
    onnx_path: str,
    min_dur: Optional[float] = None,
    max_dur: Optional[float] = None,
    device: str = "cuda",
):
    """
    Extract speaker embeddings from audio files using an ONNX model.

    Args:
        audio_dir_path: Path to directory containing audio files
        output_dir: Path to directory where npz files will be saved
        onnx_path: Path to the ONNX speaker encoder model
        min_dur: Minimum duration in seconds, files shorter than this are skipped
        max_dur: Maximum duration in seconds, files longer than this are skipped
        device: Device to run model on ('cpu' or 'cuda')
    """
    os.makedirs(output_dir, exist_ok=True)

    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        if device == "cuda":
            print("WARNING: CUDAExecutionProvider not available, falling back to CPU")
        providers = ["CPUExecutionProvider"]

    print(f"Loading ONNX model from {onnx_path} ...")
    sess = ort.InferenceSession(onnx_path, providers=providers)
    active_provider = sess.get_providers()[0]
    print(f"Running on {active_provider}")

    audio_dir = AudioDir(audio_dir_path)
    audio_ids = audio_dir.get_ids(
        to_sort=True,
        expected_sample_rate=None,
        min_dur=min_dur,
        max_dur=max_dur,
    )
    print(f"Found {len(audio_ids)} audio files")

    for audio_id in tqdm.tqdm(audio_ids, desc="Extracting speaker embeddings"):
        try:
            audio_data, _ = audio_dir.read(
                audio_id, dtype="float32", sample_rate=SAMPLE_RATE
            )
            audio_input = audio_data[np.newaxis, :].astype(np.float32)  # [1, T]
            (embedding,) = sess.run(None, {"audio": audio_input})
            embedding = embedding.astype(np.float16)  # [1, D] -> store as fp16

            output_path = os.path.join(output_dir, f"{audio_id}.npz")
            if os.path.exists(output_path):
                with np.load(output_path, allow_pickle=True) as existing_data:
                    archive = dict(existing_data)
            else:
                archive = {}

            archive["speaker_embedding"] = embedding.squeeze(0)  # [D]
            np.savez(output_path, **archive)

        except Exception as e:
            print(f"Error processing {audio_id}: {e}")
            continue

    print("Speaker embedding extraction complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker embeddings from audio files using ONNX speaker encoder"
    )
    parser.add_argument(
        "audio_dir", type=str, help="Path to directory containing audio files"
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to directory where npz files will be saved"
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Path to the ONNX speaker encoder model",
    )
    parser.add_argument(
        "--max-dur",
        type=float,
        default=None,
        help="Maximum duration in seconds, files longer than this are skipped",
    )
    parser.add_argument(
        "--min-dur",
        type=float,
        default=None,
        help="Minimum duration in seconds, files shorter than this are skipped",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: cuda)",
    )

    args = parser.parse_args()

    extract_speaker_embeddings(
        audio_dir_path=args.audio_dir,
        output_dir=args.output_dir,
        onnx_path=args.onnx_path,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        device=args.device,
    )


if __name__ == "__main__":
    main()
