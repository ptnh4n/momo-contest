"""
config.py — Cấu hình cho LLM Orchestrator (Model-Agnostic) & nạp backend/.env an toàn.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Literal

from dotenv import load_dotenv

OrchestratorMode = Literal["auto", "llm", "deterministic"]

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_ENV_PATH = _BACKEND_DIR / ".env"

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = ("gemini-3.5-flash-lite",)
DEFAULT_THINKING_LEVEL = "NONE"


def load_backend_env(env_path: str | Path | None = None, *, force: bool = False) -> bool:
    """
    Nạp biến môi trường từ `backend/.env` (override=False).

    Khi chạy pytest (`PYTEST_CURRENT_TEST` có mặt) hoặc `SAFEPAY_LOAD_DOTENV=0`,
    bỏ qua tự động nạp `.env` nếu `force=False` để các unit test T04/T01
    chạy hoàn toàn cục bộ và xác định (không vô tình gọi Gemini API thật).
    """
    if not force:
        if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
            return False
        if os.environ.get("SAFEPAY_LOAD_DOTENV", "1").strip() == "0":
            return False

    target = Path(env_path) if env_path is not None else _DEFAULT_ENV_PATH
    if not target.exists():
        return False
    return bool(load_dotenv(dotenv_path=target, override=False))


def _parse_api_keys() -> tuple[str, ...]:
    keys: list[str] = []
    multi = os.environ.get("GEMINI_API_KEYS", "")
    if multi:
        keys.extend(k.strip() for k in multi.split(",") if k.strip())
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    if single:
        keys.append(single)
    return tuple(dict.fromkeys(keys))


def _parse_fallback_models() -> tuple[str, ...]:
    if "GEMINI_FALLBACK_MODELS" in os.environ:
        raw = os.environ.get("GEMINI_FALLBACK_MODELS", "")
        return tuple(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
    return DEFAULT_FALLBACK_MODELS


@dataclass(frozen=True)
class OrchestratorConfig:
    mode: OrchestratorMode = "auto"
    api_keys: tuple[str, ...] = ()
    model: str = DEFAULT_GEMINI_MODEL
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    thinking_level: str | None = None
    temperature: float = 0.0
    max_tool_steps: int = 6
    max_retries: int = 1

    @property
    def has_api_key(self) -> bool:
        return len(self.api_keys) > 0

    @property
    def model_chain(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([self.model, *self.fallback_models]))

    @classmethod
    def from_env(cls, mode_override: str | None = None) -> "OrchestratorConfig":
        raw_mode = (mode_override or os.environ.get("SAFEPAY_ORCHESTRATOR_MODE", "auto")).strip().lower()
        if raw_mode not in ("auto", "llm", "deterministic"):
            raw_mode = "auto"

        raw_thinking = os.environ.get("GEMINI_THINKING_LEVEL", DEFAULT_THINKING_LEVEL).strip().upper()
        thinking_level: str | None = raw_thinking if raw_thinking and raw_thinking != "NONE" else None

        model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

        return cls(
            mode=raw_mode,  # type: ignore[arg-type]
            api_keys=_parse_api_keys(),
            model=model,
            fallback_models=_parse_fallback_models(),
            thinking_level=thinking_level,
        )
