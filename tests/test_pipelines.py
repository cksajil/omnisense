"""
Tests for OmniSense pipelines.
Uses lightweight mocks — we never hit real models in unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnisense.pipelines.audio import AudioPipeline
from omnisense.pipelines.base import BasePipeline

# ── BasePipeline ──────────────────────────────────────────────────────────────


class ConcreteTestPipeline(BasePipeline):
    """Minimal concrete subclass for testing the abstract base."""

    loaded = False

    def load(self) -> None:
        self.loaded = True

    def run(self, *args, **kwargs):
        return {"result": "ok"}


class TestBasePipeline:
    def test_auto_loads_on_first_call(self):
        p = ConcreteTestPipeline()
        assert not p._loaded
        p("anything")
        assert p._loaded
        assert p.loaded

    def test_load_called_only_once(self):
        p = ConcreteTestPipeline()
        p("first")
        p.loaded = False  # reset flag manually
        p("second")  # should NOT call load() again
        assert not p.loaded  # confirms load() was skipped

    def test_device_stored(self):
        p = ConcreteTestPipeline(device="cpu")
        assert p.device == "cpu"


# ── AudioPipeline ─────────────────────────────────────────────────────────────


class TestAudioPipeline:
    def test_raises_for_missing_file(self):
        pipeline = AudioPipeline()
        pipeline._loaded = True
        with pytest.raises(FileNotFoundError):
            pipeline.run("/nonexistent/path/file.wav")

    def test_raises_for_unsupported_extension(self, tmp_path):
        bad_file = tmp_path / "file.xyz"
        bad_file.write_text("dummy")
        pipeline = AudioPipeline()
        pipeline._loaded = True
        with pytest.raises(ValueError, match="Unsupported file type"):
            pipeline.run(bad_file)

    def test_raises_when_duration_exceeds_limit(self, tmp_path):
        audio = tmp_path / "long.wav"
        audio.write_bytes(b"\x00" * 100)

        pipeline = AudioPipeline()
        pipeline._loaded = True
        pipeline._model = MagicMock()

        with (
            patch("omnisense.pipelines.audio.get_audio_duration", return_value=99999),
            patch(
                "omnisense.pipelines.audio.extract_audio_from_video", return_value=audio
            ),
        ):
            with pytest.raises(ValueError, match="exceeds limit"):
                pipeline.run(audio)

    def test_run_returns_expected_keys(self, tmp_path):
        audio = tmp_path / "sample.wav"
        audio.write_bytes(b"\x00" * 100)

        mock_result = {
            "text": "Hello world this is a test.",
            "language": "en",
            "segments": [
                {"text": "Hello world", "start": 0.0, "end": 1.5, "words": []},
                {"text": "this is a test.", "start": 1.5, "end": 3.0, "words": []},
            ],
        }

        pipeline = AudioPipeline()
        pipeline._loaded = True
        pipeline._model = MagicMock()

        with (
            patch.object(pipeline, "_transcribe", return_value=mock_result),
            patch("omnisense.pipelines.audio.get_audio_duration", return_value=10.0),
        ):
            result = pipeline.run(audio)

        assert set(result.keys()) == {
            "transcript",
            "segments",
            "chunks",
            "language",
            "duration",
            "model",
        }
        assert result["transcript"] == "Hello world this is a test."
        assert result["language"] == "en"
        assert result["duration"] == 10.0
        assert isinstance(result["chunks"], list)

    def test_video_triggers_audio_extraction(self, tmp_path):
        video = tmp_path / "sample.mp4"
        video.write_bytes(b"\x00" * 100)
        extracted_audio = tmp_path / "sample_audio.wav"
        extracted_audio.write_bytes(b"\x00" * 100)

        mock_result = {
            "text": "Test transcript.",
            "language": "en",
            "segments": [
                {"text": "Test transcript.", "start": 0.0, "end": 2.0, "words": []}
            ],
        }

        pipeline = AudioPipeline()
        pipeline._loaded = True

        with (
            patch(
                "omnisense.pipelines.audio.extract_audio_from_video",
                return_value=extracted_audio,
            ) as mock_extract,
            patch("omnisense.pipelines.audio.get_audio_duration", return_value=5.0),
            patch.object(pipeline, "_transcribe", return_value=mock_result),
        ):
            result = pipeline.run(video)
            mock_extract.assert_called_once()

        assert result["transcript"] == "Test transcript."
