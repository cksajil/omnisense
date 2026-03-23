"""
Vision analysis pipeline.

Runs three models over extracted video frames:
  1. openai/clip-vit-base-patch32          — zero-shot visual classification
  2. Salesforce/blip-image-captioning-base — natural language frame captions
  3. facebook/detr-resnet-50               — object detection with bounding boxes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
    DetrForObjectDetection,
    DetrImageProcessor,
)

from omnisense.config import DEVICE, MODELS
from omnisense.pipelines.base import BasePipeline
from omnisense.utils.logger import log
from omnisense.utils.vision import extract_frames, resize_image

COCO_LABELS = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VisionPipeline(BasePipeline):
    """
    Extracts visual intelligence from video frames.

    Usage:
        pipeline = VisionPipeline(device="cuda")
        result = pipeline("/path/to/video.mp4")

        # Or pass frames directly
        result = pipeline(frames=[pil_image1, pil_image2])

    Result shape:
        {
            "captions":         list[dict],
            "objects":          list[dict],
            "unique_objects":   list[str],
            "frame_count":      int,
            "clip_labels":      list[dict],
            "top_visual_label": str,
            "models":           dict,
        }
    """

    def __init__(self, device: str = DEVICE) -> None:
        super().__init__(device=device)
        self._clip_model = None
        self._clip_processor = None
        self._blip_model = None
        self._blip_processor = None
        self._detr_model = None
        self._detr_processor = None
        self._torch_device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all three vision models."""
        log.info("Loading vision models…")

        log.info(f"  [1/3] CLIP: {MODELS['clip']}")
        self._clip_processor = CLIPProcessor.from_pretrained(MODELS["clip"])
        self._clip_model = CLIPModel.from_pretrained(MODELS["clip"]).to(
            self._torch_device
        )
        self._clip_model.eval()

        log.info(f"  [2/3] BLIP: {MODELS['captioner']}")
        self._blip_processor = BlipProcessor.from_pretrained(MODELS["captioner"])
        self._blip_model = BlipForConditionalGeneration.from_pretrained(
            MODELS["captioner"]
        ).to(self._torch_device)
        self._blip_model.eval()

        log.info(f"  [3/3] DETR: {MODELS['detector']}")
        self._detr_processor = DetrImageProcessor.from_pretrained(
            MODELS["detector"], revision="no_timm"
        )
        self._detr_model = DetrForObjectDetection.from_pretrained(
            MODELS["detector"], revision="no_timm"
        ).to(self._torch_device)
        self._detr_model.eval()

        log.info("All vision models loaded ✓")

    # ── Core run ──────────────────────────────────────────────────────────────

    def run(
        self,
        media_input: str | Path | None = None,
        frames: list[Image.Image] | None = None,
        clip_labels: list[str] | None = None,
        detection_threshold: float = 0.7,
        max_frames: int = 20,
    ) -> dict[str, Any]:
        """
        Run full vision analysis on a video or list of frames.

        Args:
            media_input: Path to video file.
            frames: List of PIL images. Mutually exclusive with media_input.
            clip_labels: Labels for CLIP zero-shot classification.
            detection_threshold: Confidence threshold for DETR detections.
            max_frames: Max frames to process (GPU memory guard).

        Returns:
            Structured vision result dict.
        """
        if media_input is None and frames is None:
            raise ValueError("Provide either media_input path or frames list.")

        if frames is None:
            raw_frames = extract_frames(media_input, max_frames=max_frames)
            pil_frames = [resize_image(f["image"]) for f in raw_frames]
            timestamps = [f["timestamp"] for f in raw_frames]
        else:
            pil_frames = [resize_image(f) for f in frames[:max_frames]]
            timestamps = list(range(len(pil_frames)))

        if not pil_frames:
            log.warning("No frames to process")
            return self._empty_result()

        log.info(f"Processing {len(pil_frames)} frames…")

        # 1. BLIP captions
        captions = self._generate_captions(pil_frames, timestamps)
        log.info(f"Generated {len(captions)} captions")

        # 2. DETR object detection
        all_objects = self._detect_objects(pil_frames, timestamps, detection_threshold)
        unique_objects = sorted(set(o["label"] for o in all_objects))
        log.info(f"Detected {len(unique_objects)} unique object types")

        # 3. CLIP zero-shot classification
        default_labels = clip_labels or [
            "indoor scene",
            "outdoor scene",
            "people talking",
            "nature landscape",
            "urban environment",
            "sports activity",
            "presentation or lecture",
            "interview",
            "news broadcast",
        ]
        clip_result = self._clip_classify(pil_frames[:5], default_labels)
        log.info(f"CLIP top label: {clip_result[0]['label']}")

        return {
            "captions": captions,
            "objects": all_objects,
            "unique_objects": unique_objects,
            "frame_count": len(pil_frames),
            "clip_labels": clip_result,
            "top_visual_label": clip_result[0]["label"] if clip_result else "unknown",
            "models": {
                "clip": MODELS["clip"],
                "captioner": MODELS["captioner"],
                "detector": MODELS["detector"],
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_captions(
        self,
        frames: list[Image.Image],
        timestamps: list[float],
    ) -> list[dict]:
        """Generate natural language captions for each frame using BLIP."""
        captions = []
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            try:
                inputs = self._blip_processor(images=frame, return_tensors="pt").to(
                    self._torch_device
                )

                with torch.no_grad():
                    output = self._blip_model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=4,
                    )

                caption = self._blip_processor.decode(
                    output[0], skip_special_tokens=True
                )
                captions.append(
                    {
                        "frame_id": i,
                        "timestamp": ts,
                        "caption": caption,
                    }
                )
            except Exception as exc:
                log.warning(f"Caption failed for frame {i}: {exc}")
                captions.append(
                    {
                        "frame_id": i,
                        "timestamp": ts,
                        "caption": "caption unavailable",
                    }
                )
        return captions

    def _detect_objects(
        self,
        frames: list[Image.Image],
        timestamps: list[float],
        threshold: float,
    ) -> list[dict]:
        """Run DETR object detection on each frame."""
        all_detections = []
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            try:
                inputs = self._detr_processor(images=frame, return_tensors="pt").to(
                    self._torch_device
                )

                with torch.no_grad():
                    outputs = self._detr_model(**inputs)

                target_sizes = torch.tensor(
                    [frame.size[::-1]], device=self._torch_device
                )
                results = self._detr_processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=threshold,
                )[0]

                for score, label_id, box in zip(
                    results["scores"],
                    results["labels"],
                    results["boxes"],
                ):
                    label_idx = label_id.item()
                    label = (
                        COCO_LABELS[label_idx]
                        if label_idx < len(COCO_LABELS)
                        else "unknown"
                    )
                    if label == "N/A":
                        continue

                    all_detections.append(
                        {
                            "frame_id": i,
                            "timestamp": ts,
                            "label": label,
                            "score": round(score.item(), 3),
                            "box": [round(x, 1) for x in box.tolist()],
                        }
                    )

            except Exception as exc:
                log.warning(f"Detection failed for frame {i}: {exc}")

        return all_detections

    def _clip_classify(
        self,
        frames: list[Image.Image],
        labels: list[str],
    ) -> list[dict]:
        """Use CLIP to classify frames against text labels."""
        try:
            inputs = self._clip_processor(
                text=labels,
                images=frames,
                return_tensors="pt",
                padding=True,
            ).to(self._torch_device)

            with torch.no_grad():
                outputs = self._clip_model(**inputs)

            logits = outputs.logits_per_image
            avg_logits = logits.mean(dim=0)
            probs = avg_logits.softmax(dim=0)

            return sorted(
                [
                    {"label": label, "score": round(prob.item(), 4)}
                    for label, prob in zip(labels, probs)
                ],
                key=lambda x: x["score"],
                reverse=True,
            )
        except Exception as exc:
            log.warning(f"CLIP classification failed: {exc}")
            return [{"label": "unknown", "score": 0.0}]

    def _empty_result(self) -> dict:
        return {
            "captions": [],
            "objects": [],
            "unique_objects": [],
            "frame_count": 0,
            "clip_labels": [{"label": "unknown", "score": 0.0}],
            "top_visual_label": "unknown",
            "models": {
                "clip": MODELS["clip"],
                "captioner": MODELS["captioner"],
                "detector": MODELS["detector"],
            },
        }
