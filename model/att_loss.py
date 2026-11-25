import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttrIndexer:
    def __init__(self, attr_names: list):
        self.attr_names = attr_names
        self.num_attrs = len(attr_names)
        self.name2idx = {name: i for i, name in enumerate(attr_names)}

    @classmethod
    def from_json(cls, json_path: str) -> "AttrIndexer":
        with open(json_path, "r") as f:
            attr_names = json.load(f)
        return cls(attr_names)

    def encode(self, obj_list: list) -> torch.Tensor:
        target = torch.zeros(self.num_attrs, dtype=torch.float32)
        for obj_name in obj_list:
            if obj_name in self.name2idx:
                target[self.name2idx[obj_name]] = 1.0
        return target

    def encode_batch(self, obj_lists: list) -> torch.Tensor:
        return torch.stack([self.encode(objs) for objs in obj_lists])


class VAAttrLoss(nn.Module):
    def __init__(
        self,
        attr_names: list,
        attr_mu: torch.Tensor,
        attr_L_inv: torch.Tensor,
        attr_log_det: torch.Tensor,
        attr_embeddings: torch.Tensor,
        temperature: float = 0.07,
        weight_sharpness: float = 1.0,
        weight_floor: float = 0.01,
    ):
        super().__init__()

        self.attr_names = attr_names
        self.num_attrs = len(attr_names)
        self.temperature = temperature
        self.weight_sharpness = weight_sharpness
        self.weight_floor = weight_floor

        self.register_buffer("attr_mu", attr_mu)
        self.register_buffer("attr_L_inv", attr_L_inv)
        self.register_buffer("attr_log_det", attr_log_det)
        self.register_buffer("attr_embeddings", F.normalize(attr_embeddings, dim=-1))

    @classmethod
    def from_precomputed(
        cls,
        stats_path: str,
        attr_embeddings: Dict[str, torch.Tensor],
        attr_list_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple["VAAttrLoss", Optional["AttrIndexer"]]:
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)

        available_sets = [set(stats.keys()), set(attr_embeddings.keys())]
        if attr_list_path is not None:
            with open(attr_list_path, "r") as f:
                gt_attr_names = json.load(f)
            available_sets.append(set(gt_attr_names))

        common_attrs = sorted(set.intersection(*available_sets))

        mu_list, L_inv_list, log_det_list, embed_list = [], [], [], []
        for attr in common_attrs:
            s = stats[attr]
            mu_list.append(torch.from_numpy(s["mu"]))
            L_inv_list.append(torch.from_numpy(s["L_inv"]))
            log_det_list.append(s["log_det"])
            embed_list.append(attr_embeddings[attr])

        loss_fn = cls(
            attr_names=common_attrs,
            attr_mu=torch.stack(mu_list),
            attr_L_inv=torch.stack(L_inv_list),
            attr_log_det=torch.tensor(log_det_list, dtype=torch.float32),
            attr_embeddings=torch.stack(embed_list),
            **kwargs,
        )

        indexer = AttrIndexer(common_attrs) if attr_list_path is not None else None

        return loss_fn, indexer

    def compute_weights(self, va: torch.Tensor) -> torch.Tensor:
        diff = va.unsqueeze(1) - self.attr_mu.unsqueeze(0)

        transformed = torch.einsum("aij,baj->bai", self.attr_L_inv, diff)
        mahal_sq = (transformed ** 2).sum(dim=-1)

        log_weights = -0.5 * self.weight_sharpness * mahal_sq
        weights = torch.exp(log_weights)  # (B, A)

        weights = weights.clamp(min=self.weight_floor)
        return weights

    def forward(
        self,
        z: torch.Tensor,
        va: torch.Tensor,
        target_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        weights = self.compute_weights(va)
        z_norm = F.normalize(z, dim=-1)
        logits = z_norm @ self.attr_embeddings.t() / self.temperature
        loss_per_attr = F.binary_cross_entropy_with_logits(
            logits, target_attr.float(), reduction="none"
        )
        weighted_loss = (weights * loss_per_attr).sum(dim=-1)
        loss = weighted_loss.mean()
        info = {
            "loss_attr": loss.item(),
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),
            "weight_min": weights.min().item(),
            "logits_mean": logits.mean().item(),
        }

        return loss, info

import torch
from typing import Optional, Dict

def extract_attr_clip_embeddings(
    attr_names: list,
    clip_model_path: str,
    templates: Optional[list] = None,
) -> Dict[str, torch.Tensor]:

    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = clip.load(clip_model_path, device=device)
    model.eval()

    if templates is None:
        templates = [
            "a photo of {}",
            "an image with {}",
            "a scene featuring {}",
            "a photo expressing {} emotion",
            "an image that feels {}",
        ]

    embeddings = {}

    with torch.no_grad():
        for attr_name in attr_names:

            clean_name = attr_name.split("_", 1)[-1].replace("_", " ")

            texts = [tmpl.format(clean_name) for tmpl in templates]
            tokens = clip.tokenize(texts).to(device)

            text_features = model.encode_text(tokens)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            avg_embed = text_features.mean(dim=0)

            avg_embed = avg_embed / avg_embed.norm()

            embeddings[attr_name] = avg_embed.cpu().float()

    return embeddings

