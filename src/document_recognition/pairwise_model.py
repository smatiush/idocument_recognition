from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import LayoutLMv3Config, LayoutLMv3Model, LayoutLMv3Processor

REQUIRED_PAIRWISE_ARTIFACT_FILES = (
    "pairwise_model.pt",
    "pairwise_model_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(slots=True)
class PairwiseModelConfig:
    backbone_config: dict[str, Any]
    classifier_dropout: float = 0.1


def validate_pairwise_model_artifact(model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    missing_files = [name for name in REQUIRED_PAIRWISE_ARTIFACT_FILES if not (model_dir / name).is_file()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"Incomplete pairwise model artifact at {model_dir}: missing {missing}")

    LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False)


class PairwiseLayoutLMv3Classifier(nn.Module):
    def __init__(
        self,
        backbone: LayoutLMv3Model,
        classifier_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(classifier_dropout)
        hidden_size = backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, 2),
        )

    @classmethod
    def from_pretrained_backbone(
        cls,
        pretrained_model_name: str,
        classifier_dropout: float = 0.1,
    ) -> "PairwiseLayoutLMv3Classifier":
        backbone = LayoutLMv3Model.from_pretrained(pretrained_model_name)
        return cls(backbone=backbone, classifier_dropout=classifier_dropout)

    @classmethod
    def from_saved(cls, model_dir: str | Path) -> "PairwiseLayoutLMv3Classifier":
        model_dir = Path(model_dir)
        validate_pairwise_model_artifact(model_dir)
        with (model_dir / "pairwise_model_config.json").open("r", encoding="utf-8") as file:
            config = PairwiseModelConfig(**json.load(file))

        backbone = LayoutLMv3Model(LayoutLMv3Config.from_dict(config.backbone_config))
        model = cls(backbone=backbone, classifier_dropout=config.classifier_dropout)
        state_dict = torch.load(model_dir / "pairwise_model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save(self, model_dir: str | Path, processor: LayoutLMv3Processor) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        config = PairwiseModelConfig(
            backbone_config=self.backbone.config.to_dict(),
            classifier_dropout=self.dropout.p,
        )

        torch.save(self.state_dict(), model_dir / "pairwise_model.pt")
        with (model_dir / "pairwise_model_config.json").open("w", encoding="utf-8") as file:
            json.dump(asdict(config), file, indent=2)
        processor.save_pretrained(str(model_dir))
        processor.image_processor.save_pretrained(str(model_dir))
        processor.tokenizer.save_pretrained(str(model_dir))
        validate_pairwise_model_artifact(model_dir)

    def _encode_page(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        bbox = self._sanitize_bbox_tensor(bbox)
        self._validate_bbox_tensor(bbox)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )
        return outputs.last_hidden_state[:, 0, :]

    def _sanitize_bbox_tensor(self, bbox: torch.Tensor) -> torch.Tensor:
        max_x = int(self.backbone.embeddings.x_position_embeddings.num_embeddings) - 1
        max_y = int(self.backbone.embeddings.y_position_embeddings.num_embeddings) - 1
        bbox = torch.nan_to_num(bbox, nan=0.0, posinf=0.0, neginf=0.0)
        bbox = bbox.to(dtype=torch.long)
        x1 = bbox[:, :, 0].clamp(min=0, max=max_x)
        y1 = bbox[:, :, 1].clamp(min=0, max=max_y)
        x2 = bbox[:, :, 2].clamp(min=0, max=max_x)
        y2 = bbox[:, :, 3].clamp(min=0, max=max_y)

        left = torch.minimum(x1, x2)
        top = torch.minimum(y1, y2)
        right = torch.maximum(x1, x2)
        bottom = torch.maximum(y1, y2)

        max_width = int(self.backbone.embeddings.w_position_embeddings.num_embeddings) - 1
        max_height = int(self.backbone.embeddings.h_position_embeddings.num_embeddings) - 1
        right = torch.minimum(right, left + max_width)
        bottom = torch.minimum(bottom, top + max_height)
        return torch.stack((left, top, right, bottom), dim=-1)

    def _validate_bbox_tensor(self, bbox: torch.Tensor) -> None:
        max_x = int(self.backbone.embeddings.x_position_embeddings.num_embeddings) - 1
        max_y = int(self.backbone.embeddings.y_position_embeddings.num_embeddings) - 1
        min_value = int(bbox.min().detach().cpu().item())
        max_left_right = int(bbox[:, :, [0, 2]].max().detach().cpu().item())
        max_top_bottom = int(bbox[:, :, [1, 3]].max().detach().cpu().item())
        max_width = int((bbox[:, :, 2] - bbox[:, :, 0]).max().detach().cpu().item())
        max_height = int((bbox[:, :, 3] - bbox[:, :, 1]).max().detach().cpu().item())

        if min_value < 0 or max_left_right > max_x or max_top_bottom > max_y:
            raise ValueError(
                "Invalid LayoutLMv3 bbox coordinates after sanitization: "
                f"min={min_value}, max_x_coord={max_left_right}/{max_x}, "
                f"max_y_coord={max_top_bottom}/{max_y}."
            )

        max_w = int(self.backbone.embeddings.w_position_embeddings.num_embeddings) - 1
        max_h = int(self.backbone.embeddings.h_position_embeddings.num_embeddings) - 1
        if max_width > max_w or max_height > max_h:
            raise ValueError(
                "Invalid LayoutLMv3 bbox size after sanitization: "
                f"max_width={max_width}/{max_w}, max_height={max_height}/{max_h}."
            )

    def forward(
        self,
        left_input_ids: torch.Tensor,
        left_attention_mask: torch.Tensor,
        left_bbox: torch.Tensor,
        left_pixel_values: torch.Tensor,
        right_input_ids: torch.Tensor,
        right_attention_mask: torch.Tensor,
        right_bbox: torch.Tensor,
        right_pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        left_embedding = self._encode_page(
            input_ids=left_input_ids,
            attention_mask=left_attention_mask,
            bbox=left_bbox,
            pixel_values=left_pixel_values,
        )
        right_embedding = self._encode_page(
            input_ids=right_input_ids,
            attention_mask=right_attention_mask,
            bbox=right_bbox,
            pixel_values=right_pixel_values,
        )

        combined = torch.cat(
            [
                left_embedding,
                right_embedding,
                torch.abs(left_embedding - right_embedding),
                left_embedding * right_embedding,
            ],
            dim=-1,
        )
        logits = self.classifier(self.dropout(combined))

        output: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            output["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return output
