"""
Voice Activity Detection script for audio files.

This script applies VAD (voice activity detection) to a directory hierarchy of files and produces an output file
in JSON format that collects all non-silence parts of an audio file as time-stamps in seconds and some general statistics.

Supports two backends:
- OmniVAD (default): CPU-only, based on FireRedVAD DFSMN model via ncnn
- Silero VAD (optional): Requires PyTorch, install with pip install revoxx[silero]
"""

import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm


def get_audio_files(directory):
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".wav", ".flac", ".mp3")):
                audio_files.append(os.path.join(root, file))
    return audio_files


def process_audio(
    file_path, model, base_dir, use_dynamic_threshold, collect_warnings=False
):
    """Process a single audio file with Silero VAD.

    Requires silero_vad to be installed (pip install revoxx[silero]).
    """
    from silero_vad import read_audio, get_speech_timestamps

    info = sf.info(file_path)
    sample_rate = info.samplerate
    overall_length = info.frames / sample_rate

    wav = read_audio(file_path)

    warnings = []

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=False,
        min_speech_duration_ms=100,
        min_silence_duration_ms=50,
    )

    if not speech_timestamps and use_dynamic_threshold:
        warning_msg = f"No speech detected in {file_path} with default threshold. Trying with lower threshold..."
        if collect_warnings:
            warnings.append(warning_msg)
        else:
            print(warning_msg)

        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            return_seconds=False,
            min_speech_duration_ms=100,
            min_silence_duration_ms=50,
            threshold=0.1,
        )

    speech_segments = [
        [round(t["start"] / 16000, 3), round(t["end"] / 16000, 3)]
        for t in speech_timestamps
    ]

    result = {"overall": round(overall_length, 3), "timestamps": speech_segments}

    if speech_segments:
        result["begin"] = speech_segments[0][0]
        result["end"] = speech_segments[-1][1]
    else:
        warning_msg = f"Warning: No speech detected in {file_path}" + (
            " even with lower threshold." if use_dynamic_threshold else "."
        )
        if collect_warnings:
            warnings.append(warning_msg)
        else:
            print(warning_msg)

    rel_path = os.path.relpath(file_path, base_dir)

    if collect_warnings:
        return rel_path, result, warnings
    return rel_path, result


def process_dataset(
    dataset_path, output_filename="vad_silero.json", file_callback=None
):
    """Process all audio files in a dataset directory with Silero VAD.

    Args:
        dataset_path: Path to dataset directory
        output_filename: Name of the output JSON file
        file_callback: Optional callback(files_processed) called after each file

    Returns:
        Dictionary with files_processed count and warnings list
    """
    from pathlib import Path
    from silero_vad import load_silero_vad

    dataset_path = Path(dataset_path)
    vad_output = dataset_path / output_filename
    audio_files = get_audio_files(str(dataset_path))

    result_info = {"files_processed": 0, "warnings": []}

    if not audio_files:
        return result_info

    model = load_silero_vad()
    results = {}

    for file_path in audio_files:
        try:
            rel_path, result, file_warnings = process_audio(
                file_path,
                model,
                str(dataset_path),
                use_dynamic_threshold=True,
                collect_warnings=True,
            )
            results[rel_path] = result
            result_info["warnings"].extend(file_warnings)
            result_info["files_processed"] += 1
            if file_callback:
                file_callback(result_info["files_processed"])
        except Exception as e:
            result_info["warnings"].append(f"VAD error for {file_path}: {e}")

    import silero_vad

    output = {
        "_meta": {
            "backend": "Silero VAD",
            "version": getattr(silero_vad, "__version__", "unknown"),
            "model": "Silero VAD (PyTorch)",
        },
        "files": results,
    }

    with open(vad_output, "w") as f:
        json.dump(output, f, indent=2)

    return result_info


def _run_omnivad(audio_files, base_dir, output_file):
    """Run OmniVAD on audio files."""
    from scripts_module.omnivad_processor import process_audio as omnivad_process

    from omnivad import OmniVAD

    vad = OmniVAD()
    results = {}

    try:
        for file_path in tqdm(audio_files, desc="Processing (OmniVAD)"):
            try:
                rel_path, result = omnivad_process(file_path, vad, base_dir)
                results[rel_path] = result
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    finally:
        vad.close()

    import omnivad as omnivad_mod

    output = {
        "_meta": {
            "backend": "OmniVAD",
            "version": omnivad_mod.__version__,
            "model": "FireRedVAD DFSMN (ncnn)",
        },
        "files": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def _run_silero(audio_files, base_dir, output_file, use_dynamic_threshold):
    """Run Silero VAD on audio files."""
    from silero_vad import load_silero_vad
    import silero_vad

    model = load_silero_vad()
    results = {}

    for file_path in tqdm(audio_files, desc="Processing (Silero VAD)"):
        try:
            rel_path, result = process_audio(
                file_path, model, base_dir, use_dynamic_threshold
            )
            results[rel_path] = result
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    output = {
        "_meta": {
            "backend": "Silero VAD",
            "version": getattr(silero_vad, "__version__", "unknown"),
            "model": "Silero VAD (PyTorch)",
        },
        "files": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Apply VAD to audio files in a directory."
    )
    parser.add_argument("input_dir", help="Input directory containing audio files")
    parser.add_argument("output_file", help="Output JSON file to store results")
    parser.add_argument(
        "--backend",
        choices=["omnivad", "silero"],
        default="omnivad",
        help="VAD backend to use (default: omnivad)",
    )
    parser.add_argument(
        "--use-dynamic-threshold",
        action="store_true",
        help="Use dynamic threshold for Silero speech detection",
    )
    args = parser.parse_args()

    audio_files = get_audio_files(args.input_dir)

    if args.backend == "omnivad":
        _run_omnivad(audio_files, args.input_dir, args.output_file)
    else:
        _run_silero(
            audio_files, args.input_dir, args.output_file, args.use_dynamic_threshold
        )


if __name__ == "__main__":
    main()
