"""Audio transcription pipeline — Whisper."""

from omnisense.pipelines.base import BasePipeline


class AudioPipeline(BasePipeline):
    """Phase 2 implementation."""

    def load(self) -> None:  # noqa: D102
        pass

    def run(self, audio_path: str) -> dict:  # noqa: D102
        return {}
