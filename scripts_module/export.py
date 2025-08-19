#!/usr/bin/env python3
"""Command-line tool for exporting Revoxx sessions to Talrómur 3 dataset format."""

import argparse
import sys
from pathlib import Path
from typing import List

from revoxx.dataset import DatasetExporter


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Revoxx sessions to Talrómur 3 dataset format",
        epilog="Example: revoxx-export --sessions-dir ./recordings "
        "--output-dir ./datasets --sessions session1.revoxx,session2.revoxx",
    )

    parser.add_argument(
        "--sessions-dir",
        type=Path,
        required=True,
        help="Base directory containing .revoxx session directories",
    )

    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for dataset"
    )

    parser.add_argument(
        "--sessions",
        type=str,
        help="Comma-separated list of session names to export (default: all)",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name for the output dataset (default: from first session speaker)",
    )

    parser.add_argument(
        "--format",
        choices=["wav", "flac"],
        default="flac",
        help="Output audio format (default: flac)",
    )

    parser.add_argument(
        "--zero-intensity",
        type=str,
        default="neutral",
        help="Comma-separated emotions to set intensity to 0 (default: neutral)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Display detailed statistics"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset without prompting",
    )

    return parser.parse_args()


def find_sessions(base_dir: Path, session_names: List[str] = None) -> List[Path]:
    """Find session directories.

    Args:
        base_dir: Base directory to search
        session_names: Optional list of specific session names to find

    Returns:
        List of paths to .revoxx session directories
    """
    sessions = []

    if session_names:
        for name in session_names:
            session_path = base_dir / name
            if session_path.exists() and session_path.is_dir():
                sessions.append(session_path)
            else:
                print(f"Warning: Session not found: {name}")
    else:
        for item in sorted(base_dir.iterdir()):
            if item.is_dir() and item.suffix == ".revoxx":
                sessions.append(item)

    return sessions


def main():
    """Main entry point."""
    args = parse_arguments()

    if not args.sessions_dir.exists():
        print(f"Error: Sessions directory not found: {args.sessions_dir}")
        sys.exit(1)

    if not args.output_dir.exists():
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Could not create output directory: {e}")
            sys.exit(1)

    session_names = args.sessions.split(",") if args.sessions else None
    session_paths = find_sessions(args.sessions_dir, session_names)

    if not session_paths:
        print("Error: No sessions found")
        sys.exit(1)

    print(f"Found {len(session_paths)} session(s) to export")

    if args.dataset_name:
        dataset_path = args.output_dir / args.dataset_name
        if dataset_path.exists():
            if args.force:
                import shutil

                shutil.rmtree(dataset_path)
                print(f"Removed existing dataset: {dataset_path}")
            else:
                response = input(
                    f"Dataset '{args.dataset_name}' exists. Overwrite? (y/n): "
                )
                if response.lower() != "y":
                    print("Export cancelled")
                    sys.exit(0)
                import shutil

                shutil.rmtree(dataset_path)

    zero_emotions = []
    if args.zero_intensity:
        zero_emotions = [e.strip() for e in args.zero_intensity.split(",")]

    exporter = DatasetExporter(
        output_dir=args.output_dir,
        audio_format=args.format,
        zero_intensity_emotions=zero_emotions,
    )

    try:
        print("Exporting sessions...")
        dataset_path, statistics = exporter.export_sessions(
            session_paths, dataset_name=args.dataset_name
        )

        print(f"\nDataset created: {dataset_path}")

        if args.verbose:
            print("\n--- Export Statistics ---")
            print(f"Sessions processed: {statistics['sessions_processed']}")
            print(f"Total utterances: {statistics['total_utterances']}")
            print(f"Emotions: {', '.join(statistics['emotions'])}")

            if statistics.get("file_counts"):
                print("\nFiles per emotion:")
                for emotion, count in statistics["file_counts"].items():
                    print(f"  {emotion}: {count}")

            if statistics.get("missing_recordings"):
                print(
                    f"\nWarning: {statistics['missing_recordings']} recordings were missing"
                )

    except Exception as e:
        print(f"Error: Export failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
