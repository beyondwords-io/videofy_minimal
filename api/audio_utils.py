from __future__ import annotations

import subprocess
from pathlib import Path


def get_duration_seconds(ffprobe_bin: str, audio_file: Path) -> float:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_file),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return max(0.0, float(result.stdout.strip() or 0.0))


def concat_mp3(ffmpeg_bin: str, inputs: list[Path], output_file: Path) -> None:
    if not inputs:
        raise ValueError("Cannot concatenate zero audio files")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    concat_file = output_file.parent / "concat.txt"
    concat_lines = [f"file '{path.resolve()}'" for path in inputs]
    concat_file.write_text("\n".join(concat_lines), encoding="utf-8")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_file),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def create_silence_mp3(ffmpeg_bin: str, duration_seconds: float, output_file: Path) -> None:
    if duration_seconds <= 0:
        raise ValueError("Silence duration must be positive")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-t",
        f"{duration_seconds:.3f}",
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_file),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
