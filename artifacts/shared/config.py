# lambdas/shared/config.py
from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

def _fenv(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v != "" else float(default)
    except Exception:
        return float(default)

def _ienv(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    try:
        return int(v) if v != "" else int(default)
    except Exception:
        return int(default)

def _senv(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "") else default

@dataclass(frozen=True)
class Settings:
    # Infra / estado
    region: str
    ddb_table: str
    session_ttl_hours: float

    # LLM (Bedrock por defecto)
    llm_provider: str
    bedrock_model_id: str
    llm_temperature: float
    llm_max_tokens: int
    llm_top_p: float

    # Clasificador externo (tu API)
    classifier_api_url: Optional[str]
    classifier_api_key: Optional[str]
    classifier_timeout_ms: int

    # Reglas del agente
    judge_threshold: float
    max_judge_passes: int

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        region=_senv("AWS_REGION", "us-east-1") or "us-east-1",
        ddb_table=_senv("DDB_TABLE", "medical-agent-sessions") or "medical-agent-sessions",
        session_ttl_hours=_fenv("SESSION_TTL_HOURS", 0),

        llm_provider=(_senv("LLM_PROVIDER", "bedrock") or "bedrock").lower(),
        bedrock_model_id=_senv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620") or "anthropic.claude-3-5-sonnet-20240620",
        llm_temperature=_fenv("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_ienv("LLM_MAX_TOKENS", 600),
        llm_top_p=_fenv("LLM_TOP_P", 0.9),

        classifier_api_url=_senv("CLASSIFIER_API_URL"),
        classifier_api_key=_senv("CLASSIFIER_API_KEY"),
        classifier_timeout_ms=_ienv("CLASSIFIER_TIMEOUT_MS", 8000),

        judge_threshold=_fenv("JUDGE_THRESHOLD", 0.8),
        max_judge_passes=_ienv("MAX_JUDGE_PASSES", 3),
    )
