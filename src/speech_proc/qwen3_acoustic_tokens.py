"""
Copyright 2025 Balacoon

Acoustic token extraction using Qwen3TTSTokenizer
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import tqdm

from speech_proc.audio_dir import AudioDir


def extract_tokens(
    audio_dir_path: str,
    output_dir: str,
    batch_size: int = 16,
    min_dur: Optional[float] = None,
    max_dur: Optional[float] = None,
    device: str = "cpu",
    model_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
):
    """
    Extract acoustic tokens from audio files using Qwen3TTSTokenizer.

    Args:
        audio_dir_path: Path to directory containing audio files
        output_dir: Path to directory where npz files will be saved
        batch_size: Number of audio files to process at once
        min_dur: Minimum duration in seconds, files shorter than this are skipped
        max_dur: Maximum duration in seconds, files longer than this are skipped
        device: Device to run model on ('cpu' or 'cuda')
        model_name: Qwen3 tokenizer model name (default: Qwen/Qwen3-TTS-Tokenizer-12Hz)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model {model_name}...")
    from qwen_tts import Qwen3TTSTokenizer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        model_name,
        device_map=device if device == "cpu" else f"{device}:0",
        dtype=dtype,
    )
    print(f"Model input sample rate: {tokenizer.get_input_sample_rate()}")
    print(f"Model type: {tokenizer.get_model_type()}")

    # Initialize audio directory
    audio_dir = AudioDir(audio_dir_path)
    audio_ids = audio_dir.get_ids(
        # allow any sample rate, tokenizer handles resampling internally
        to_sort=True,
        expected_sample_rate=None,
        min_dur=min_dur,
        max_dur=max_dur,
    )
    print(f"Found {len(audio_ids)} audio files")

    # Process audio files in batches
    num_batches = (len(audio_ids) + batch_size - 1) // batch_size

    for batch_idx in tqdm.tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(audio_ids))
        batch_ids = audio_ids[start_idx:end_idx]

        # Get audio paths for this batch
        batch_paths = []
        valid_batch_ids = []

        for audio_id in batch_ids:
            audio_path = audio_dir.get_path(audio_id)
            if audio_path is None:
                print(f"Warning: Could not find audio file for {audio_id}")
                continue
            batch_paths.append(audio_path)
            valid_batch_ids.append(audio_id)

        if not batch_paths:
            continue

        try:
            # Encode audio files - tokenizer handles loading and resampling
            with torch.no_grad():
                encoded = tokenizer.encode(batch_paths, return_dict=True)

            # Extract audio_codes from the output
            # For 12Hz model: audio_codes is List[torch.LongTensor] each (codes_len, num_quantizers)
            audio_codes_list = encoded.audio_codes

            # Process outputs for each audio file in the batch
            for i, audio_id in enumerate(valid_batch_ids):
                # Get tokens for this sample
                tokens = audio_codes_list[i].cpu().numpy().astype(np.int16)
                # Shape: (codes_len, num_quantizers) for 12Hz
                # Transpose to (num_quantizers, codes_len) to match nano acoustic tokens format
                tokens = tokens.T

                # Save to npz file
                output_path = os.path.join(output_dir, f"{audio_id}.npz")

                # Check if archive already exists
                if os.path.exists(output_path):
                    existing_data = np.load(output_path, allow_pickle=True)
                    archive_dict = dict(existing_data)  # Convert to a mutable dictionary
                    existing_data.close()  # Close the file after loading
                else:
                    archive_dict = {}  # Create a new dictionary if file doesn't exist

                # Add or update the 'tokens' array
                archive_dict["acoustic_tokens"] = tokens

                # Save the updated archive
                np.savez(output_path, **archive_dict)

        except Exception as e:
            print(f"Error processing batch starting at {batch_ids[0]}: {e}")
            continue

    print("Acoustic token extraction complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract acoustic tokens from audio files using Qwen3TTSTokenizer"
    )
    parser.add_argument(
        "audio_dir", type=str, help="Path to directory containing audio files"
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to directory where npz files will be saved"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Number of audio files to process at once (default: 16)",
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
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on (default: cpu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Qwen3 tokenizer model name (default: Qwen/Qwen3-TTS-Tokenizer-12Hz)",
    )

    args = parser.parse_args()

    extract_tokens(
        audio_dir_path=args.audio_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        device=args.device,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
