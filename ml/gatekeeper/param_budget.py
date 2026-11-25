"""
param_budget.py — Kiến trúc Factorized Embedding Dual-Branch Gatekeeper & kiểm toán ngân sách tham số (<20M).

Khớp chính xác thông số Blueprint v0.2.1 §5.1 và feature_schema.json (14 đặc trưng Tabular).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ml.gatekeeper.preprocessing import DEFAULT_SCHEMA_PATH, load_feature_schema


class SafePayGatekeeper(nn.Module):
    """Factorized Embedding Dual-Branch Gatekeeper (<20M params)."""

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        self.vocab_size = int(spec["vocab_size"])
        self.embed_dim = int(spec["embed_dim"])
        self.hidden_dim = int(spec["hidden_dim"])
        self.num_layers = int(spec["num_layers"])
        self.num_heads = int(spec["num_heads"])
        self.ffn_dim = int(spec["ffn_dim"])
        self.max_length = int(spec["max_length"])
        self.token_type_vocab_size = int(spec.get("token_type_vocab_size", 2))
        self.tabular_input_dim = int(spec["tabular_input_dim"])
        self.tabular_hidden_dim = int(spec["tabular_hidden_dim"])
        self.tabular_output_dim = int(spec["tabular_output_dim"])
        self.fusion_dim = int(spec["fusion_dim"])
        self.num_classes = int(spec["num_classes"])

        # Text Branch (Factorized Embedding ALBERT-style)
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.embed_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_length, self.hidden_dim)
        self.token_type_embedding = nn.Embedding(self.token_type_vocab_size, self.hidden_dim)
        self.embed_norm = nn.LayerNorm(self.hidden_dim)
        self.embed_dropout = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim),
            enable_nested_tensor=False,
        )

        # Tabular & Behavioral Branch
        self.tabular_mlp = nn.Sequential(
            nn.Linear(self.tabular_input_dim, self.tabular_hidden_dim),
            nn.LayerNorm(self.tabular_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.tabular_hidden_dim, self.tabular_output_dim),
        )

        # Fusion & Multi-Task Heads
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.decision_head = nn.Linear(self.fusion_dim, self.num_classes)
        self.risk_head = nn.Linear(self.fusion_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        tok_emb = self.embed_proj(self.token_embedding(input_ids))
        x = tok_emb + self.position_embedding(pos_ids) + self.token_type_embedding(token_type_ids)
        x = self.embed_dropout(self.embed_norm(x))

        src_key_padding_mask = attention_mask == 0
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        mask_f = attention_mask.unsqueeze(-1).to(encoded.dtype)
        h_text = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
        h_tab = self.tabular_mlp(tabular_features)

        h_fused = self.fusion_norm(torch.cat([h_text, h_tab], dim=-1))
        logits = self.decision_head(h_fused)
        risk_score = torch.sigmoid(self.risk_head(h_fused))
        return logits, risk_score


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def compute_param_budget(schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    schema = load_feature_schema(schema_path)
    spec = schema["model_budget_spec"]
    model = SafePayGatekeeper(spec)

    emb_params = (
        _count_params(model.token_embedding)
        + _count_params(model.embed_proj)
        + _count_params(model.position_embedding)
        + _count_params(model.token_type_embedding)
        + _count_params(model.embed_norm)
    )
    enc_params = _count_params(model.encoder)
    tab_params = _count_params(model.tabular_mlp)
    fusion_heads_params = (
        _count_params(model.fusion_norm)
        + _count_params(model.decision_head)
        + _count_params(model.risk_head)
    )
    total_params = _count_params(model)

    emb_limit = int(spec["embedding_param_limit"])
    tot_limit = int(spec["total_param_limit"])

    assert emb_params == (
        total_params - enc_params - tab_params - fusion_heads_params
    ), "Tổng tham số các khối không khớp."
    assert emb_params < emb_limit, f"Embedding params {emb_params:,} >= {emb_limit:,}"
    assert total_params < tot_limit, f"Total params {total_params:,} >= {tot_limit:,}"

    return {
        "vocab_size": int(spec["vocab_size"]),
        "tabular_input_dim": int(spec["tabular_input_dim"]),
        "embedding_params": emb_params,
        "token_embedding_only": _count_params(model.token_embedding),
        "encoder_params": enc_params,
        "tabular_params": tab_params,
        "fusion_heads_params": fusion_heads_params,
        "total_params": total_params,
        "embedding_limit": emb_limit,
        "total_limit": tot_limit,
        "fp32_size_mb": round((total_params * 4) / (1024 * 1024), 2),
        "est_onnx_int8_mb": round(total_params / (1024 * 1024), 2),
    }


if __name__ == "__main__":
    summary = compute_param_budget()
    print("=== SafePay Gatekeeper Parameter Budget Audit ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
