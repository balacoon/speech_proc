"""
Copyright 2025 Balacoon

Phoneme feature extraction using Allophant model
"""

import argparse
import os
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import tqdm
from allophant.estimator import Estimator
from allophant.dataset_processing import Batch

from speech_proc.audio_dir import AudioDir


FRAME_DUR = 0.02  # 20ms


def extract_features(
    audio_dir_path: str,
    output_dir: str,
    language: str,
    batch_size: int = 16,
    max_dur: Optional[float] = None,
    device: str = "cpu",
    model_name: str = "kgnlp/allophant",
):
    """
    Extract phoneme features from audio files using Allophant.

    Args:
        audio_dir_path: Path to directory containing audio files
        output_dir: Path to directory where npz files will be saved
        language: Language code for deriving phoneme inventory (e.g., 'en', 'es', 'de')
        batch_size: Number of audio files to process at once
        max_dur: Maximum duration in seconds, files longer than this are skipped
        device: Device to run model on ('cpu' or 'cuda')
        model_name: Hugging Face model name or path to checkpoint
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model {model_name}...")
    model, attribute_indexer = Estimator.restore(model_name, device=device)
    model_sample_rate = model.sample_rate

    # Get phoneme inventory for the language
    print(f"Loading phoneme inventory for language: {language}")
    inventory = attribute_indexer.phoneme_inventory(language)
    print(f"Inventory contains {len(inventory)} phonemes")

    # Get feature names supported by the model
    supported_features = attribute_indexer.feature_names
    print(f"Supported features: {supported_features}")

    # Create composition feature matrix for the inventory
    composition_matrix = attribute_indexer.composition_feature_matrix(inventory).to(
        device
    )

    # Initialize audio directory
    audio_dir = AudioDir(audio_dir_path)
    audio_ids = audio_dir.get_ids(
        to_sort=True, expected_sample_rate=model_sample_rate, max_dur=max_dur
    )
    print(f"Found {len(audio_ids)} audio files")

    # Process audio files in batches
    num_batches = (len(audio_ids) + batch_size - 1) // batch_size

    for batch_idx in tqdm.tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(audio_ids))
        batch_ids = audio_ids[start_idx:end_idx]

        # Load and prepare audio for this batch
        batch_audios = []
        batch_lengths = []

        for audio_id in batch_ids:
            try:
                # Read audio using AudioDir
                audio_data, sample_rate = audio_dir.read(
                    audio_id, dtype="float32", sample_rate=model_sample_rate
                )

                # Convert to tensor and add channel dimension if needed
                audio_tensor = torch.from_numpy(audio_data)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(
                        0
                    )  # Add channel dimension (1 x samples)

                batch_audios.append(audio_tensor)
                batch_lengths.append(audio_tensor.shape[1])

            except Exception as e:
                print(f"Error loading audio {audio_id}: {e}")
                continue

        if not batch_audios:
            continue

        # Pad audios to the same length
        max_length = max(batch_lengths)
        padded_audios = []
        for audio in batch_audios:
            if audio.shape[1] < max_length:
                padding = torch.zeros(audio.shape[0], max_length - audio.shape[1])
                audio = torch.cat([audio, padding], dim=1)
            padded_audios.append(audio)

        # Stack into batch tensor
        batch_audio_tensor = torch.cat(padded_audios, dim=0)
        batch_lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long)
        # Language ID can be 0 for inference
        batch_lang_ids = torch.zeros(len(batch_audios), dtype=torch.long)

        # Create batch
        batch = Batch(batch_audio_tensor, batch_lengths_tensor, batch_lang_ids)

        # Run model prediction
        with torch.no_grad():
            model_outputs = model.predict(batch.to(device), composition_matrix)

        # Process outputs for each audio file in the batch
        for i, audio_id in enumerate(batch_ids):
            output_features = {}

            # Extract features for this sample
            for feature_name in supported_features:

                # skip some features, they are not informative, duplicates
                if feature_name in ["phone", "tone"]:
                    continue

                # Get output for this feature (shape: frames x batch x dim)
                feature_output = model_outputs.outputs[feature_name][
                    :, i, :
                ]  # frames x dim

                # Get actual length for this sample
                actual_length_samples = batch_lengths[i]
                actual_length_frames = math.floor(
                    actual_length_samples / model_sample_rate / FRAME_DUR
                )
                feature_output = feature_output[
                    :actual_length_frames
                ]  # Trim to actual length
                feature_values = (
                    feature_output.argmax(dim=-1)
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )

                output_features[feature_name] = feature_values

            # Save to npz file
            output_path = os.path.join(output_dir, f"{audio_id}.npz")
            np.savez(output_path, **output_features)

    print("Feature extraction complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract phoneme features from audio files using Allophant"
    )
    parser.add_argument(
        "audio_dir", type=str, help="Path to directory containing audio files"
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to directory where npz files will be saved"
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        required=True,
        help="Language code for deriving phoneme inventory (e.g., 'en', 'es', 'de')",
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
        "-m",
        type=float,
        default=None,
        help="Maximum duration in seconds, files longer than this are skipped",
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
        default="kgnlp/allophant",
        help="Hugging Face model name or path to checkpoint (default: kgnlp/allophant)",
    )

    args = parser.parse_args()

    extract_features(
        audio_dir_path=args.audio_dir,
        output_dir=args.output_dir,
        language=args.language,
        batch_size=args.batch_size,
        max_dur=args.max_dur,
        device=args.device,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
