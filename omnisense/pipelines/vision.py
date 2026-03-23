"""Vision analysis pipeline — CLIP + BLIP-2 + DETR."""

from omnisense.pipelines.base import BasePipeline


class VisionPipeline(BasePipeline):
    """Phase 3 implementation."""

    def load(self) -> None:  # noqa: D102
        pass

    def run(self, video_path: str) -> dict:  # noqa: D102
        return {}
