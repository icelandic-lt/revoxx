"""
Voice Activity Detection script for audio files.

This script applies VAD (voice activity detection) to a directory hierarchy of files and produces an output file
in JSON format that collects all non-silence parts of an audio file as time-stamps in seconds and some general statistics
"""

import argparse
import json
import os
import soundfile as sf
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from tqdm import tqdm


def get_audio_files(directory):
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".wav", ".flac", ".mp3")):
                audio_files.append(os.path.join(root, file))
    return audio_files


def process_audio(file_path, model, base_dir, use_dynamic_threshold):
    # Get audio info using soundfile instead of torchaudio
    info = sf.info(file_path)
    sample_rate = info.samplerate
    overall_length = info.frames / sample_rate

    # read audio file and convert to 16 kHz sample rate by default
    wav = read_audio(file_path)

    # First attempt with default threshold (0.5)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=False,
        min_speech_duration_ms=100,
        min_silence_duration_ms=50,
    )

    # If no speech detected and dynamic threshold is enabled, try again with lower threshold
    if not speech_timestamps and use_dynamic_threshold:
        print(
            f"No speech detected in {file_path} with default threshold. Trying with lower threshold..."
        )
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            return_seconds=False,
            min_speech_duration_ms=100,
            min_silence_duration_ms=50,
            threshold=0.1,
        )

    # Convert timestamps to seconds with high precision
    speech_segments = [
        [round(t["start"] / 16000, 3), round(t["end"] / 16000, 3)]
        for t in speech_timestamps
    ]

    result = {"overall": round(overall_length, 3), "timestamps": speech_segments}

    if speech_segments:
        result["begin"] = speech_segments[0][0]
        result["end"] = speech_segments[-1][1]
    else:
        print(
            f"Warning: No speech detected in {file_path}"
            + (" even with lower threshold." if use_dynamic_threshold else ".")
        )

    # Get relative path
    rel_path = os.path.relpath(file_path, base_dir)

    return rel_path, result


def main():
    parser = argparse.ArgumentParser(
        description="Apply VAD to audio files in a directory."
    )
    parser.add_argument("input_dir", help="Input directory containing audio files")
    parser.add_argument("output_file", help="Output JSON file to store results")
    parser.add_argument(
        "--use-dynamic-threshold",
        action="store_true",
        help="Use dynamic threshold for speech detection",
    )
    args = parser.parse_args()

    # Load Silero VAD model
    model = load_silero_vad()

    audio_files = get_audio_files(args.input_dir)
    results = {}

    for file_path in tqdm(audio_files, desc="Processing audio files"):
        try:
            rel_path, result = process_audio(
                file_path, model, args.input_dir, args.use_dynamic_threshold
            )
            results[rel_path] = result
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
