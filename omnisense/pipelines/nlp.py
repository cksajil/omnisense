"""NLP analysis pipeline — summarization, NER, zero-shot classification."""

from omnisense.pipelines.base import BasePipeline


class NLPPipeline(BasePipeline):
    """Phase 3 implementation."""

    def load(self) -> None:  # noqa: D102
        pass

    def run(self, text: str) -> dict:  # noqa: D102
        return {}
