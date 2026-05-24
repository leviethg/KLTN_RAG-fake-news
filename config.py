"""
config.py – Centralised configuration for the Vietnamese Fake-News Verification
            system trained on the ViFactCheck dataset.

All tuneable knobs live here; import and use them across the pipeline:
    from config import cfg
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# ─────────────────────────────────────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = ROOT_DIR / "data"
CACHE_DIR  = ROOT_DIR / "cache"
MODEL_DIR  = ROOT_DIR / "models"
LOG_DIR    = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Sub-directories for each pipeline stage (created on first use)
STAGE_DIRS: dict[str, Path] = {
    "stage1_evidence_retrieval": CACHE_DIR / "stage1_evidence_retrieval",
    "stage2_claim_decompose":    CACHE_DIR / "stage2_claim_decompose",
    "stage3_evidence_scoring":   CACHE_DIR / "stage3_evidence_scoring",
    "stage4_verdict_llm":        CACHE_DIR / "stage4_verdict_llm",
    "stage5_mlp_classifier":     CACHE_DIR / "stage5_mlp_classifier",
}

for _d in [DATA_DIR, CACHE_DIR, MODEL_DIR, LOG_DIR, OUTPUT_DIR, *STAGE_DIRS.values()]:
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAME       = "ViFactCheck"
DATASET_HF_PATH    = "tarudesu/ViFactCheck"   # HuggingFace hub path
RAW_TRAIN_FILE     = DATA_DIR / "train.jsonl"
RAW_DEV_FILE       = DATA_DIR / "dev.jsonl"
RAW_TEST_FILE      = DATA_DIR / "test.jsonl"

# Label space  (N = 3)
NUM_LABELS   = 3
LABEL2ID: dict[str, int] = {
    "SUPPORTED":   0,
    "REFUTED":     1,
    "NEI":         2,   # Not Enough Information
}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Evidence Retrieval  (RAG hyper-parameters)
# ─────────────────────────────────────────────────────────────────────────────

RAG_TOP_K          = 5          # K: number of evidence passages retrieved
BM25_B             = 0.75       # BM25 length normalisation
BM25_K1            = 1.5        # BM25 term-frequency saturation
DENSE_MODEL_NAME   = "intfloat/multilingual-e5-base"
DENSE_INDEX_PATH   = CACHE_DIR / "faiss_index.bin"
HYBRID_ALPHA       = 0.5        # weight for dense score in hybrid retrieval
                                # final_score = α·dense + (1-α)·bm25


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Claim Decomposition (LLM-based, cached)
# ─────────────────────────────────────────────────────────────────────────────

MAX_SUB_CLAIMS     = 5          # maximum atomic sub-claims per input claim


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Evidence Scoring
# ─────────────────────────────────────────────────────────────────────────────

NLI_MODEL_NAME     = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_BATCH_SIZE     = 32
NLI_MAX_LENGTH     = 512


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – LLM Verdict (chain-of-thought explanation)
# ─────────────────────────────────────────────────────────────────────────────

LLM_TEMPERATURE    = 0.0        # deterministic for reproducibility
LLM_MAX_TOKENS     = 1024
LLM_SYSTEM_PROMPT  = (
    "Bạn là chuyên gia kiểm tra thông tin tiếng Việt. "
    "Dựa trên bằng chứng được cung cấp, hãy đưa ra phán quyết "
    "(SUPPORTED / REFUTED / NEI) kèm giải thích rõ ràng."
)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 – MLP Classifier hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

# Architecture
MLP_HIDDEN_DIMS    = [512, 256]         # hidden layer sizes
MLP_DROPOUT        = 0.3
MLP_ACTIVATION     = "gelu"

# Training
MLP_LEARNING_RATE  = 2e-4
MLP_BATCH_SIZE     = 32
MLP_EPOCHS         = 30
MLP_WEIGHT_DECAY   = 1e-4
MLP_WARMUP_RATIO   = 0.1               # fraction of steps for LR warm-up
MLP_GRAD_CLIP      = 1.0
MLP_EARLY_STOP_PATIENCE = 5

# Checkpointing
MLP_CHECKPOINT_DIR = MODEL_DIR / "mlp_checkpoints"
MLP_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Encoder backbone whose [CLS] embeddings feed into the MLP
ENCODER_MODEL_NAME = "vinai/phobert-base-v2"
ENCODER_MAX_LENGTH = 256
ENCODER_FREEZE     = True               # freeze encoder; only train MLP head


# ─────────────────────────────────────────────────────────────────────────────
# API configuration  (loaded from .env – never hard-code secrets here)
# ─────────────────────────────────────────────────────────────────────────────

class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key:    str  = Field(default="", alias="OPENAI_API_KEY")
    openai_model:      str  = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_base_url:   str  = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )

    # Google Gemini
    gemini_api_key:    str  = Field(default="", alias="GEMINI_API_KEY")
    gemini_model:      str  = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")

    # Active backend: "openai" | "gemini"
    llm_backend:       str  = Field(default="openai", alias="LLM_BACKEND")

    # Rate limiting
    api_max_retries:   int  = Field(default=3,    alias="API_MAX_RETRIES")
    api_retry_delay:   float = Field(default=2.0, alias="API_RETRY_DELAY")
    api_timeout:       float = Field(default=30.0, alias="API_TIMEOUT")


# Singleton – import `cfg` everywhere instead of instantiating repeatedly
cfg = APISettings()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
