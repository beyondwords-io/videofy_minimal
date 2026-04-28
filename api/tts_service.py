from __future__ import annotations

import json
import logging
import time
import subprocess
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen
from urllib.error import HTTPError

try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
except ImportError:  # pragma: no cover - backward compatibility
    from elevenlabs import VoiceSettings
    from elevenlabs import ElevenLabs

from . import audio_utils

logger = logging.getLogger(__name__)


class TTSService(Protocol):
    def synthesize_line(
        self,
        text: str,
        output_mp3: Path,
        voice_id: str | None = None,
        model_id: str = "eleven_turbo_v2_5",
        voice_settings: dict[str, Any] | None = None,
    ) -> None: ...

    def get_duration_seconds(self, audio_file: Path) -> float: ...

    def concat_mp3(self, inputs: list[Path], output_file: Path) -> None: ...

    def create_silence_mp3(self, duration_seconds: float, output_file: Path) -> None: ...


class ElevenLabsService:
    def __init__(self, api_key: str, voice_id: str, ffprobe_bin: str, ffmpeg_bin: str):
        self._api_key = api_key
        self._voice_id = voice_id
        self._ffprobe_bin = ffprobe_bin
        self._ffmpeg_bin = ffmpeg_bin
        self._client = ElevenLabs(api_key=self._api_key) if self._api_key else None

    def synthesize_line(
        self,
        text: str,
        output_mp3: Path,
        voice_id: str | None = None,
        model_id: str = "eleven_turbo_v2_5",
        voice_settings: dict[str, Any] | None = None,
    ) -> None:
        if not self._client:
            raise ValueError("ELEVENLABS_API_KEY is required for manuscript processing")

        output_mp3.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "voice_id": voice_id or self._voice_id,
            "model_id": model_id,
            "output_format": "mp3_44100_128",
            "text": text,
        }
        if voice_settings:
            try:
                payload["voice_settings"] = VoiceSettings(**voice_settings)
            except Exception:
                payload["voice_settings"] = voice_settings

        logger.info(
            "Calling ElevenLabs text_to_speech.convert with voice_id=%s model_id=%s",
            payload["voice_id"],
            model_id,
        )
        audio_stream = self._client.text_to_speech.convert(**payload)

        with output_mp3.open("wb") as handle:
            if isinstance(audio_stream, (bytes, bytearray)):
                handle.write(audio_stream)
            else:
                for chunk in audio_stream:
                    if chunk:
                        handle.write(chunk)

    def get_duration_seconds(self, audio_file: Path) -> float:
        return audio_utils.get_duration_seconds(self._ffprobe_bin, audio_file)

    def concat_mp3(self, inputs: list[Path], output_file: Path) -> None:
        audio_utils.concat_mp3(self._ffmpeg_bin, inputs, output_file)

    def create_silence_mp3(self, duration_seconds: float, output_file: Path) -> None:
        audio_utils.create_silence_mp3(self._ffmpeg_bin, duration_seconds, output_file)


_BW_POLL_TIMEOUT = 60


class BeyondWordsService:
    def __init__(self, api_key: str, project_id: str, ffprobe_bin: str, ffmpeg_bin: str):
        self._api_key = api_key
        self._project_id = project_id
        self._ffprobe_bin = ffprobe_bin
        self._ffmpeg_bin = ffmpeg_bin
        self._content_id: str | None = None

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"https://api.beyondwords.io/v1/projects/{self._project_id}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        req = Request(url, data=data, method=method)
        req.add_header("X-Api-Key", self._api_key)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "videofy-minimal/1.0")
        with urlopen(req) as resp:
            if resp.status == 204:
                return None
            return json.loads(resp.read().decode("utf-8"))

    def synthesize_line(
        self,
        text: str,
        output_mp3: Path,
        voice_id: str | None = None,
        model_id: str = "eleven_turbo_v2_5",
        voice_settings: dict[str, Any] | None = None,
    ) -> None:
        if not self._api_key:
            raise ValueError("BEYONDWORDS_API_KEY is required for manuscript processing")

        output_mp3.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "title": text[:80],
            "body": f"<p>{text}</p>",
        }
        if voice_id:
            try:
                payload["body_voice_id"] = int(voice_id)
            except (ValueError, TypeError):
                logger.warning("Invalid BeyondWords voice_id=%s, using project default", voice_id)

        if self._content_id is None:
            logger.info("Creating BeyondWords content for project=%s", self._project_id)
            content = self._request("POST", "/content", payload)
            self._content_id = content["id"]
            logger.info("BeyondWords content created id=%s, polling for audio", self._content_id)
        else:
            logger.info("Updating BeyondWords content id=%s", self._content_id)
            self._request("PUT", f"/content/{self._content_id}", payload)

        audio_url = self._poll_for_audio(self._content_id)
        self._download_audio(audio_url, output_mp3)
        logger.info("BeyondWords audio downloaded to %s", output_mp3)

    def _poll_for_audio(self, content_id: str) -> str:
        elapsed = 0.0
        delay = 1.0
        while elapsed < _BW_POLL_TIMEOUT:
            time.sleep(delay)
            elapsed += delay

            content = self._request("GET", f"/content/{content_id}")
            status = content.get("status")
            logger.info(
                "BeyondWords content id=%s status=%s (%.1fs elapsed)",
                content_id,
                status,
                elapsed,
            )
            if status == "processed":
                for entry in content.get("audio", []):
                    if entry.get("content_type") == "audio/mpeg" and isinstance(entry.get("url"), str):
                        return entry["url"]
                raise ValueError(
                    f"BeyondWords content {content_id} processed but no MP3 URL found in response"
                )
            if status in ("error", "skipped"):
                raise ValueError(f"BeyondWords content {content_id} status: {status}")

            delay = min(delay * 2, 5.0)

        raise TimeoutError(
            f"BeyondWords content {content_id} not processed after {_BW_POLL_TIMEOUT}s"
        )

    def _download_audio(self, url: str, output_mp3: Path) -> None:
        req = Request(url)
        with urlopen(req) as resp:
            with output_mp3.open("wb") as handle:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    handle.write(chunk)

    def get_duration_seconds(self, audio_file: Path) -> float:
        return audio_utils.get_duration_seconds(self._ffprobe_bin, audio_file)

    def concat_mp3(self, inputs: list[Path], output_file: Path) -> None:
        audio_utils.concat_mp3(self._ffmpeg_bin, inputs, output_file)

    def create_silence_mp3(self, duration_seconds: float, output_file: Path) -> None:
        audio_utils.create_silence_mp3(self._ffmpeg_bin, duration_seconds, output_file)
