"""
OmniVAD processor for voice activity detection.

Uses OmniVAD (FireRedVAD DFSMN model via ncnn) for CPU-only speech detection.
Produces the same JSON output format as the Silero-based vadiate module.
"""

import json
import os
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from omnivad import OmniVAD

VAD_SAMPLE_RATE = 16000


def get_audio_files(directory):
    """Find all audio files recursively in a directory."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".wav", ".flac", ".mp3")):
                audio_files.append(os.path.join(root, file))
    return audio_files


def process_audio(file_path, vad, base_dir, collect_warnings=False):
    """Process a single audio file with OmniVAD.

    Args:
        file_path: Path to audio file
        vad: OmniVAD instance
        base_dir: Base directory for relative path calculation
        collect_warnings: If True, return warnings list instead of printing

    Returns:
        Tuple of (rel_path, result, warnings) if collect_warnings,
        otherwise (rel_path, result)
    """
    info = sf.info(file_path)
    sample_rate = info.samplerate
    overall_length = info.frames / sample_rate

    warnings = []

    # Read and resample to 16kHz mono float32
    data, sr = sf.read(file_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != VAD_SAMPLE_RATE:
        data = resample_poly(data, VAD_SAMPLE_RATE, sr).astype(np.float32)

    result_vad = vad.detect(data, sample_rate=VAD_SAMPLE_RATE)

    speech_segments = [
        [round(start, 3), round(end, 3)]
        for start, end in result_vad.get("timestamps", [])
    ]

    result = {"overall": round(overall_length, 3), "timestamps": speech_segments}

    if speech_segments:
        result["begin"] = speech_segments[0][0]
        result["end"] = speech_segments[-1][1]
    else:
        warning_msg = f"Warning: No speech detected in {file_path}."
        if collect_warnings:
            warnings.append(warning_msg)
        else:
            print(warning_msg)

    rel_path = os.path.relpath(file_path, base_dir)

    if collect_warnings:
        return rel_path, result, warnings
    return rel_path, result


def process_dataset(dataset_path, output_filename="vad.json", file_callback=None):
    """Process all audio files in a dataset directory.

    Args:
        dataset_path: Path to dataset directory
        output_filename: Name of the output JSON file
        file_callback: Optional callback(files_processed) called after each file

    Returns:
        Dictionary with files_processed count and warnings list
    """
    from pathlib import Path

    dataset_path = Path(dataset_path)
    vad_output = dataset_path / output_filename
    audio_files = get_audio_files(str(dataset_path))

    result_info = {"files_processed": 0, "warnings": []}

    if not audio_files:
        return result_info

    vad = OmniVAD()
    results = {}

    try:
        for file_path in audio_files:
            try:
                rel_path, result, file_warnings = process_audio(
                    file_path,
                    vad,
                    str(dataset_path),
                    collect_warnings=True,
                )
                results[rel_path] = result
                result_info["warnings"].extend(file_warnings)
                result_info["files_processed"] += 1
                if file_callback:
                    file_callback(result_info["files_processed"])
            except Exception as e:
                result_info["warnings"].append(f"VAD error for {file_path}: {e}")
    finally:
        vad.close()

    import omnivad

    output = {
        "_meta": {
            "backend": "OmniVAD",
            "version": omnivad.__version__,
            "model": "FireRedVAD DFSMN (ncnn)",
        },
        "files": results,
    }

    with open(vad_output, "w") as f:
        json.dump(output, f, indent=2)

    return result_info
