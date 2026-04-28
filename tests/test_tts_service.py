import io
import json
from pathlib import Path
from unittest.mock import patch

import api.tts_service as tts_module


class FakeTextToSpeechAPI:
    def __init__(self):
        self.calls: list[dict] = []

    def convert(self, **kwargs):
        self.calls.append(kwargs)
        return [b"abc"]


class FakeElevenLabsClient:
    last_instance = None

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.text_to_speech = FakeTextToSpeechAPI()
        FakeElevenLabsClient.last_instance = self


def test_tts_service_calls_elevenlabs_convert_with_voice_id(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tts_module, "ElevenLabs", FakeElevenLabsClient)

    service = tts_module.ElevenLabsService(
        api_key="test-api-key",
        voice_id="fallback-voice",
        ffprobe_bin="ffprobe",
        ffmpeg_bin="ffmpeg",
    )

    out = tmp_path / "line.mp3"
    service.synthesize_line(
        text="Hei verden",
        output_mp3=out,
        voice_id="brand-voice-id",
        model_id="eleven_multilingual_v2",
        voice_settings={"stability": 1.0, "similarity_boost": 1.0},
    )

    client = FakeElevenLabsClient.last_instance
    assert client is not None
    assert len(client.text_to_speech.calls) == 1

    payload = client.text_to_speech.calls[0]
    assert payload["voice_id"] == "brand-voice-id"
    assert payload["model_id"] == "eleven_multilingual_v2"
    assert payload["text"] == "Hei verden"
    assert payload["output_format"] == "mp3_44100_128"
    assert "voice_settings" in payload
    assert out.read_bytes() == b"abc"


class FakeURLResponse:
    def __init__(self, data: bytes | dict, status: int = 200):
        if isinstance(data, dict):
            self._data = json.dumps(data).encode("utf-8")
        else:
            self._data = data
        self.status = status
        self._stream = io.BytesIO(self._data)

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            return self._data
        return self._stream.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_processed_response(content_id: str, audio_url: str = "https://cdn.beyondwords.io/test.mp3"):
    return FakeURLResponse({
        "id": content_id,
        "status": "processed",
        "audio": [{"content_type": "audio/mpeg", "url": audio_url}],
    })


def test_beyondwords_service_creates_then_updates_then_deletes(tmp_path: Path):
    responses = [
        FakeURLResponse({"id": "abc-123", "status": "queued"}),
        _make_processed_response("abc-123"),
        FakeURLResponse(b"audio-line-1"),
        FakeURLResponse({"id": "abc-123", "status": "queued"}),
        _make_processed_response("abc-123", "https://cdn.beyondwords.io/line2.mp3"),
        FakeURLResponse(b"audio-line-2"),
    ]
    call_index = {"i": 0}
    requests_made: list[str] = []

    def fake_urlopen(req, **kwargs):
        idx = call_index["i"]
        call_index["i"] += 1
        method = req.get_method() if hasattr(req, "get_method") else "GET"
        url = req.full_url if hasattr(req, "full_url") else str(req)
        requests_made.append(f"{method} {url}")
        return responses[idx]

    service = tts_module.BeyondWordsService(
        api_key="test-key", project_id="99", ffprobe_bin="ffprobe", ffmpeg_bin="ffmpeg",
    )

    out1 = tmp_path / "line1.mp3"
    out2 = tmp_path / "line2.mp3"
    with patch.object(tts_module, "urlopen", fake_urlopen), \
         patch.object(tts_module.time, "sleep"):
        service.synthesize_line(text="Line one", output_mp3=out1, voice_id="42")
        service.synthesize_line(text="Line two", output_mp3=out2, voice_id="42")

    assert out1.read_bytes() == b"audio-line-1"
    assert out2.read_bytes() == b"audio-line-2"
    assert len(requests_made) == 6


def test_beyondwords_service_passes_body_voice_id(tmp_path: Path):
    responses = [
        FakeURLResponse({"id": "abc-456", "status": "queued"}),
        _make_processed_response("abc-456"),
        FakeURLResponse(b"audio"),
    ]
    call_index = {"i": 0}
    post_body: dict = {}

    def fake_urlopen(req, **kwargs):
        idx = call_index["i"]
        call_index["i"] += 1
        if hasattr(req, "data") and req.data and req.get_method() == "POST":
            post_body.update(json.loads(req.data.decode("utf-8")))
        return responses[idx]

    service = tts_module.BeyondWordsService(
        api_key="test-key", project_id="99", ffprobe_bin="ffprobe", ffmpeg_bin="ffmpeg",
    )

    out = tmp_path / "line.mp3"
    with patch.object(tts_module, "urlopen", fake_urlopen), \
         patch.object(tts_module.time, "sleep"):
        service.synthesize_line(text="Test", output_mp3=out, voice_id="42")

    assert post_body["body_voice_id"] == 42
    assert "<p>Test</p>" in post_body["body"]
