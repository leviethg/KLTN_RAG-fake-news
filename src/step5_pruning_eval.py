"""
step5_pruning_eval.py
=====================
Hậu xử lý XAI (Explainable AI) + Báo cáo kết quả thực nghiệm Chương 4.

Luồng xử lý:
  1. Nhận đầu vào nhãn phán quyết y* từ mô hình MLP (Bước 4).
  2. Thuật toán Subgraph Pruning (cắt tỉa đồ thị con):
       - Nếu y* = SUPPORTED  → xóa các nút lập luận phản bác (c_i^-)
         ra khỏi đồ thị, chỉ giữ lại nhánh ủng hộ (c_i^+).
       - Nếu y* = REFUTED    → xóa các nút lập luận ủng hộ (c_i^+),
         chỉ giữ lại nhánh phản bác (c_i^-).
       - Nếu y* = NEI        → giữ nguyên cấu trúc đồ thị, đánh dấu tất cả
         nút là "Thiếu bằng chứng" để LLM giải thích.
     Kết quả: đồ thị con giải thích tối giản G*.
  3. Đưa G* vào LLM (bất đồng bộ) để tổng hợp lời giải thích văn bản
     tự nhiên T mạch lạc bằng tiếng Việt.
  4. Tính các độ đo khoa học trên tập Test bằng scikit-learn:
       • Macro-average F1-score
       • Macro-average Precision
       • Macro-average Recall
       • Per-class F1 (SUPPORTED / REFUTED / NEI)
  5. In bảng so sánh với SOTA Benchmark ViFactCheck (Full Context):
       mBERT: 58.07% | PhoBERT-large: 62.93% | Gemma: 85.94%.

Đầu vào cần thiết:
  models/mlp_checkpoints/best_graph2seq_phobert.pt   ← checkpoint Bước 4
  cache/step3_arguments.json                          ← đồ thị lập luận
  data/test.csv  (hoặc test.jsonl)                    ← nhãn chuẩn tập Test

Đầu ra:
  outputs/step5_pruned_graphs.json   ← đồ thị con G* của từng mẫu Test
  outputs/step5_explanations.json    ← lời giải thích T của từng mẫu Test
  outputs/step5_metrics.json         ← các số đo đánh giá
  outputs/step5_report.txt           ← báo cáo tổng hợp dạng bảng
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from transformers import AutoTokenizer, AutoModel

# ── project root trên sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    CACHE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    MLP_HIDDEN_DIMS,
    MLP_DROPOUT,
    MLP_CHECKPOINT_DIR,
    ENCODER_MAX_LENGTH,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    SEED,
    cfg,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số
# ─────────────────────────────────────────────────────────────────────────────

ARGUMENTS_FILE   = CACHE_DIR  / "step3_arguments.json"
BEST_CKPT        = MLP_CHECKPOINT_DIR / "best_graph2seq_phobert.pt"
TEST_CSV         = DATA_DIR   / "test.csv"
TEST_JSONL       = DATA_DIR   / "test.jsonl"

OUT_PRUNED       = OUTPUT_DIR / "step5_pruned_graphs.json"
OUT_EXPLANATIONS = OUTPUT_DIR / "step5_explanations.json"
OUT_METRICS      = OUTPUT_DIR / "step5_metrics.json"
OUT_REPORT       = OUTPUT_DIR / "step5_report.txt"

# Override encoder → phobert-large (nhất quán với Bước 4)
ENCODER_MODEL_NAME = "vinai/phobert-large"

# NEI marker từ Bước 3
NEI_MARKER = "KHÔNG CÓ BẰNG CHỨNG ĐỦ ĐỂ ĐÁNH GIÁ – NEI"

# Số lượng mẫu tối đa để sinh giải thích LLM (tránh chi phí API cao)
MAX_EXPLAIN_SAMPLES = 20
CONCURRENCY_LLM = 4

# ── SOTA Benchmarks ViFactCheck (Macro F1 trên Full Context) ──────────────
SOTA_BENCHMARKS: list[dict] = [
    {"model": "mBERT",           "macro_f1": 58.07, "note": "Full Context, ViFactCheck paper"},
    {"model": "PhoBERT-large",   "macro_f1": 62.93, "note": "Full Context, ViFactCheck paper"},
    {"model": "Gemma",           "macro_f1": 85.94, "note": "Full Context, ViFactCheck paper"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# MLP Model (nhất quán với Bước 4)
# ─────────────────────────────────────────────────────────────────────────────

class PhoBERTMLPClassifier(nn.Module):
    """
    Kiến trúc PhoBERT-large + MLP Classification Head.
    Phải khớp 100% với định nghĩa trong step4_graph2seq_mlp.py.
    """

    def __init__(
        self,
        encoder_name: str      = ENCODER_MODEL_NAME,
        hidden_dims:  list[int] = None,
        dropout:      float    = MLP_DROPOUT,
        num_labels:   int      = NUM_LABELS,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = MLP_HIDDEN_DIMS

        self.encoder    = AutoModel.from_pretrained(encoder_name)
        encoder_dim     = self.encoder.config.hidden_size

        mlp_layers: list[nn.Module] = []
        in_dim = encoder_dim
        for out_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            in_dim = out_dim
        mlp_layers.append(nn.Linear(in_dim, num_labels))

        self.classifier = nn.Sequential(*mlp_layers)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_cls   = outputs.last_hidden_state[:, 0, :]
        return self.classifier(h_cls)


# ─────────────────────────────────────────────────────────────────────────────
# Graph2Seq (nhất quán với Bước 4)
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_sequence(record: dict) -> str:
    """
    Trải phẳng đồ thị lập luận thành chuỗi văn bản (đồng nhất với Bước 4).
    Dùng để tokenize đầu vào cho MLP trong giai đoạn inference.
    """
    nodes: list[dict] = record.get("nodes", [])
    edges: list[dict] = record.get("edges", [])

    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if src and tgt:
            outgoing.setdefault(src, []).append(tgt)

    def _key(node: dict) -> int:
        nid = node.get("id", "c9999")
        try:
            return int(nid[1:]) if len(nid) > 1 and nid[0] == "c" else 9999
        except ValueError:
            return 9999

    node_strings: list[str] = []
    for node in sorted(nodes, key=_key):
        nid         = node.get("id", "?")
        text        = node.get("text", "").strip()
        arg_support = node.get("arg_support", NEI_MARKER).strip()
        arg_refute  = node.get("arg_refute",  NEI_MARKER).strip()

        support_text = "NEI" if arg_support == NEI_MARKER else arg_support
        refute_text  = "NEI" if arg_refute  == NEI_MARKER else arg_refute

        parts = [
            f"[NÚT {nid}] {text}",
            f"[LẬP LUẬN ỦNG HỘ] {support_text}",
            f"[LẬP LUẬN PHẢN BÁC] {refute_text}",
        ]
        for tgt in outgoing.get(nid, []):
            parts.append(f"[PHỤ THUỘC VÀO NÚT {tgt}]")

        node_strings.append(" ".join(parts))

    return " [SEP_NODE] ".join(node_strings)


# ─────────────────────────────────────────────────────────────────────────────
# ① Subgraph Pruning  –  Thuật toán cắt tỉa đồ thị con XAI
# ─────────────────────────────────────────────────────────────────────────────

def prune_subgraph(record: dict, predicted_label: int) -> dict:
    """
    Cắt tỉa đồ thị lập luận dựa trên nhãn phán quyết y* để tạo đồ thị
    con giải thích tối giản G*.

    Quy tắc cắt tỉa:
      • y* = SUPPORTED (0): Xóa toàn bộ lập luận phản bác (arg_refute / c_i^-)
        ra khỏi các nút, chỉ giữ lại nhánh ủng hộ (arg_support / c_i^+).
        Lý do: Khi claim được ủng hộ, luận điểm phản bác không liên quan đến
        lời giải thích cuối cùng.

      • y* = REFUTED (1): Xóa toàn bộ lập luận ủng hộ (arg_support / c_i^+),
        chỉ giữ lại nhánh phản bác (arg_refute / c_i^-) để làm bằng chứng
        bác bỏ.

      • y* = NEI (2): Không xóa nút nào – giữ nguyên cấu trúc đồ thị nhưng
        gán flag "THIẾU BẰNG CHỨNG" vào mỗi nút để LLM biết rằng toàn bộ
        tập bằng chứng đều không đủ kết luận.

    Trả về bản ghi được bổ sung:
      "pruned_nodes"   : danh sách nút sau cắt tỉa
      "pruned_reason"  : mô tả quy tắc đã áp dụng
      "predicted_label": tên nhãn dự đoán
    """
    nodes        = record.get("nodes", [])
    predicted_str = ID2LABEL.get(predicted_label, "UNKNOWN")
    pruned_nodes: list[dict] = []

    if predicted_label == LABEL2ID["SUPPORTED"]:
        # Giữ lại arg_support, đặt arg_refute = None (cắt tỉa)
        for node in nodes:
            pruned_nodes.append({
                "id":          node["id"],
                "text":        node.get("text", ""),
                "arg_support": node.get("arg_support", NEI_MARKER),
                "arg_refute":  None,   # đã cắt tỉa
                "pruned":      "arg_refute",
            })
        reason = (
            "Nhãn SUPPORTED: xóa toàn bộ nhánh lập luận phản bác (c_i^-) "
            "ra khỏi đồ thị. Chỉ giữ lại nhánh ủng hộ (c_i^+)."
        )

    elif predicted_label == LABEL2ID["REFUTED"]:
        # Giữ lại arg_refute, đặt arg_support = None (cắt tỉa)
        for node in nodes:
            pruned_nodes.append({
                "id":          node["id"],
                "text":        node.get("text", ""),
                "arg_support": None,   # đã cắt tỉa
                "arg_refute":  node.get("arg_refute", NEI_MARKER),
                "pruned":      "arg_support",
            })
        reason = (
            "Nhãn REFUTED: xóa toàn bộ nhánh lập luận ủng hộ (c_i^+) "
            "ra khỏi đồ thị. Chỉ giữ lại nhánh phản bác (c_i^-)."
        )

    else:  # NEI
        for node in nodes:
            pruned_nodes.append({
                "id":         node["id"],
                "text":       node.get("text", ""),
                "arg_support": None,
                "arg_refute":  None,
                "pruned":      "both",
                "nei_flag":    True,
            })
        reason = (
            "Nhãn NEI: không đủ bằng chứng để kết luận. "
            "Toàn bộ nút được đánh dấu 'Thiếu bằng chứng'."
        )

    return {
        **record,
        "pruned_nodes":    pruned_nodes,
        "pruned_reason":   reason,
        "predicted_label": predicted_str,
    }


def pruned_graph_to_text(pruned_record: dict) -> str:
    """
    Chuyển đồ thị con đã cắt tỉa G* thành chuỗi văn bản có cấu trúc
    để đưa vào LLM sinh lời giải thích.
    """
    claim         = pruned_record.get("claim", "")
    pruned_nodes  = pruned_record.get("pruned_nodes", [])
    predicted_lbl = pruned_record.get("predicted_label", "UNKNOWN")
    reason        = pruned_record.get("pruned_reason", "")

    lines = [
        f"TUYÊN BỐ GỐC: {claim}",
        f"PHÁN QUYẾT MÔ HÌNH: {predicted_lbl}",
        f"QUY TẮC CẮT TỈA: {reason}",
        "",
        "ĐỒ THỊ CON GIẢI THÍCH (G*):",
    ]

    for node in pruned_nodes:
        nid  = node.get("id", "?")
        text = node.get("text", "")
        lines.append(f"\n  [Tiểu tuyên bố {nid}]: {text}")

        if node.get("nei_flag"):
            lines.append("    → Không có bằng chứng đủ để đánh giá tiểu tuyên bố này.")
        else:
            if node.get("arg_support") is not None:
                lines.append(f"    [Lập luận ủng hộ]: {node['arg_support']}")
            if node.get("arg_refute") is not None:
                lines.append(f"    [Lập luận phản bác]: {node['arg_refute']}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ② LLM Explanation – Tổng hợp lời giải thích văn bản tự nhiên T
# ─────────────────────────────────────────────────────────────────────────────

_EXPLAIN_SYSTEM = """\
Bạn là chuyên gia kiểm tra thông tin tiếng Việt với khả năng giải thích \
phán quyết một cách mạch lạc, trung thực và dễ hiểu.

Dựa trên ĐỒ THỊ CON GIẢI THÍCH G* được cung cấp (đã được cắt tỉa chỉ giữ \
lại những lập luận liên quan đến phán quyết), hãy tổng hợp thành MỘT đoạn \
văn bản tự nhiên bằng tiếng Việt giải thích lý do tại sao tuyên bố nhận \
phán quyết đó.

Yêu cầu:
• Viết bằng tiếng Việt chuẩn, mạch lạc, trung lập.
• Không lặp lại phán quyết ở đầu đoạn; hãy đi thẳng vào lập luận.
• Dài 3–5 câu, tập trung vào bằng chứng quan trọng nhất.
• Không bịa đặt thông tin ngoài G*.
• Kết thúc bằng một câu tóm tắt ngắn gọn phán quyết.\
"""


def _build_explain_prompt(pruned_text: str) -> str:
    return (
        f"{pruned_text}\n\n"
        "Hãy viết lời giải thích văn bản tự nhiên cho phán quyết trên:"
    )


async def _call_openai_explain(
    session: aiohttp.ClientSession,
    pruned_text: str,
) -> str:
    url     = cfg.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       cfg.openai_model,
        "temperature": LLM_TEMPERATURE,
        "max_tokens":  512,
        "messages": [
            {"role": "system", "content": _EXPLAIN_SYSTEM},
            {"role": "user",   "content": _build_explain_prompt(pruned_text)},
        ],
    }
    async with session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def _call_gemini_explain(
    session: aiohttp.ClientSession,
    pruned_text: str,
) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg.gemini_model}:generateContent?key={cfg.gemini_api_key}"
    )
    combined = _EXPLAIN_SYSTEM + "\n\n" + _build_explain_prompt(pruned_text)
    payload = {
        "contents": [{"parts": [{"text": combined}]}],
        "generationConfig": {
            "temperature":     LLM_TEMPERATURE,
            "maxOutputTokens": 512,
        },
    }
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


async def generate_explanation_for_record(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    pruned_record: dict,
) -> dict:
    """
    Gọi LLM bất đồng bộ để sinh lời giải thích T cho một mẫu đã cắt tỉa.
    Trả về bản ghi bổ sung trường 'explanation'.
    """
    pruned_text = pruned_graph_to_text(pruned_record)
    explanation = ""
    last_exc: Exception | None = None

    for attempt in range(1, cfg.api_max_retries + 2):
        async with sem:
            try:
                if cfg.llm_backend == "gemini":
                    explanation = await asyncio.wait_for(
                        _call_gemini_explain(session, pruned_text),
                        timeout=cfg.api_timeout,
                    )
                else:
                    explanation = await asyncio.wait_for(
                        _call_openai_explain(session, pruned_text),
                        timeout=cfg.api_timeout,
                    )
                break
            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    f"[id={pruned_record.get('id')}] Timeout lần {attempt}."
                )
            except aiohttp.ClientResponseError as exc:
                last_exc = exc
                if exc.status in {429, 500, 502, 503, 504}:
                    await asyncio.sleep(cfg.api_retry_delay * attempt)
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[id={pruned_record.get('id')}] Lỗi: {exc}")
            await asyncio.sleep(cfg.api_retry_delay)

    if not explanation:
        label = pruned_record.get("predicted_label", "UNKNOWN")
        explanation = (
            f"[Tự động] Mô hình phân loại tuyên bố này là {label}. "
            f"Lý do: {pruned_record.get('pruned_reason', '')}"
        )
        logger.warning(
            f"[id={pruned_record.get('id')}] LLM thất bại. "
            f"Dùng fallback explanation. Lỗi cuối: {last_exc}"
        )

    return {**pruned_record, "explanation": explanation}


async def run_explanation_pipeline(
    pruned_records: list[dict],
) -> list[dict]:
    """
    Sinh lời giải thích LLM bất đồng bộ cho toàn bộ danh sách đã cắt tỉa.
    Giới hạn MAX_EXPLAIN_SAMPLES mẫu để kiểm soát chi phí API.
    """
    subset = pruned_records[:MAX_EXPLAIN_SAMPLES]
    logger.info(
        f"Sinh lời giải thích LLM cho {len(subset):,}/{len(pruned_records):,} mẫu "
        f"(giới hạn MAX_EXPLAIN_SAMPLES={MAX_EXPLAIN_SAMPLES})."
    )

    sem      = asyncio.Semaphore(CONCURRENCY_LLM)
    timeout  = aiohttp.ClientTimeout(total=cfg.api_timeout + 10)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LLM * 2)

    explained: list[dict] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            generate_explanation_for_record(session, sem, rec)
            for rec in subset
        ]
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="LLM Explanation",
            unit="sample",
        ):
            result = await coro
            explained.append(result)

    # Sắp xếp lại theo id gốc
    id_order = {int(r["id"]): i for i, r in enumerate(subset)}
    explained.sort(key=lambda x: id_order.get(int(x.get("id", 0)), 9999))

    # Gộp phần còn lại (không gọi LLM) với fallback
    remaining = pruned_records[MAX_EXPLAIN_SAMPLES:]
    for rec in remaining:
        lbl = rec.get("predicted_label", "UNKNOWN")
        fallback = (
            f"[Tự động – ngoài giới hạn LLM] "
            f"Mô hình phân loại tuyên bố này là {lbl}."
        )
        explained.append({**rec, "explanation": fallback})

    return explained


# ─────────────────────────────────────────────────────────────────────────────
# ③ Nạp dữ liệu
# ─────────────────────────────────────────────────────────────────────────────

def load_test_labels() -> dict[int, int]:
    """
    Nạp nhãn chuẩn tập Test từ data/test.csv hoặc data/test.jsonl.
    Trả về dict {annotation_id: label_id}.
    """
    # Ưu tiên test.csv
    if TEST_CSV.exists():
        df = pd.read_csv(TEST_CSV)
        id_col = "annotation_id" if "annotation_id" in df.columns else "id"
        label_col: str | None = None
        for c in ["label", "Label", "verdict", "Verdict"]:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            raise KeyError(
                f"Không tìm thấy cột nhãn trong {TEST_CSV}. "
                f"Các cột: {list(df.columns)}"
            )
        df = df[[id_col, label_col]].dropna()
        df[label_col] = df[label_col].astype(str).str.strip().str.upper()
        label_map: dict[int, int] = {}
        for _, row in df.iterrows():
            lid = LABEL2ID.get(str(row[label_col]), -1)
            if lid != -1:
                label_map[int(row[id_col])] = lid
        logger.info(f"Đọc {len(label_map):,} nhãn Test từ {TEST_CSV.name}.")
        return label_map

    # Fallback: test.jsonl
    if TEST_JSONL.exists():
        label_map = {}
        with TEST_JSONL.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rid = int(obj.get("annotation_id", obj.get("id", -1)))
                lbl = str(obj.get("label", obj.get("verdict", ""))).strip().upper()
                lid = LABEL2ID.get(lbl, -1)
                if rid != -1 and lid != -1:
                    label_map[rid] = lid
        logger.info(f"Đọc {len(label_map):,} nhãn Test từ {TEST_JSONL.name}.")
        return label_map

    raise FileNotFoundError(
        f"Không tìm thấy {TEST_CSV} hoặc {TEST_JSONL}. "
        "Vui lòng đặt file nhãn tập Test vào thư mục data/."
    )


def load_arguments_cache() -> list[dict]:
    """Nạp cache/step3_arguments.json."""
    if not ARGUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {ARGUMENTS_FILE}. "
            "Vui lòng chạy step3_competing_gen.py trước."
        )
    with ARGUMENTS_FILE.open(encoding="utf-8") as f:
        records: list[dict] = json.load(f)
    logger.info(f"Đọc {len(records):,} bản ghi từ {ARGUMENTS_FILE.name}.")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# ④ MLP Inference – Dự đoán nhãn y* trên tập Test
# ─────────────────────────────────────────────────────────────────────────────

def load_mlp_model(device: torch.device) -> PhoBERTMLPClassifier:
    """
    Tải checkpoint tốt nhất từ Bước 4.
    Nếu checkpoint không tồn tại, ném FileNotFoundError.
    """
    if not BEST_CKPT.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint MLP tại {BEST_CKPT}. "
            "Vui lòng chạy step4_graph2seq_mlp.py trước."
        )
    logger.info(f"Tải checkpoint: {BEST_CKPT}")
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)

    # Lấy encoder name từ checkpoint (nếu có), fallback về mặc định
    encoder_name = ckpt.get("encoder_model", ENCODER_MODEL_NAME)

    model = PhoBERTMLPClassifier(
        encoder_name = encoder_name,
        hidden_dims  = MLP_HIDDEN_DIMS,
        dropout      = MLP_DROPOUT,
        num_labels   = NUM_LABELS,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    val_acc = ckpt.get("val_acc", float("nan"))
    epoch   = ckpt.get("epoch",   "?")
    logger.success(
        f"Checkpoint tải thành công "
        f"| epoch={epoch} | val_acc={val_acc*100:.2f}% "
        f"| encoder={encoder_name}"
    )
    return model


@torch.no_grad()
def run_mlp_inference(
    model:     PhoBERTMLPClassifier,
    tokenizer,
    records:   list[dict],
    device:    torch.device,
    batch_size: int = 16,
) -> dict[int, int]:
    """
    Chạy MLP inference trên danh sách bản ghi.
    Trả về dict {record_id: predicted_label_id}.
    """
    model.eval()
    predictions: dict[int, int] = {}

    all_ids   = [int(r["id"]) for r in records]
    all_seqs  = [graph_to_sequence(r) for r in records]

    pbar = tqdm(
        range(0, len(all_seqs), batch_size),
        desc="MLP Inference",
        unit="batch",
        dynamic_ncols=True,
    )

    for start in pbar:
        batch_seqs = all_seqs[start : start + batch_size]
        batch_ids  = all_ids[start : start + batch_size]

        enc = tokenizer(
            batch_seqs,
            max_length  = ENCODER_MAX_LENGTH,
            padding     = "max_length",
            truncation  = True,
            return_tensors = "pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)          # (B, 3)
        preds  = logits.argmax(dim=-1).cpu().tolist()      # list[int]

        for rid, pred in zip(batch_ids, preds):
            predictions[rid] = pred

    logger.info(f"Inference hoàn thành: {len(predictions):,} dự đoán.")
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ Evaluation – Tính toán độ đo scikit-learn
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict:
    """
    Tính toán Macro-average F1, Precision, Recall và per-class F1
    bằng scikit-learn.

    Trả về dict chứa tất cả các độ đo.
    """
    target_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    macro_pre = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100

    # Per-class F1
    per_class_f1 = f1_score(
        y_true, y_pred, average=None,
        labels=list(range(NUM_LABELS)),
        zero_division=0,
    ) * 100

    # Classification report đầy đủ
    report_str = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))

    # Accuracy
    accuracy = (np.array(y_true) == np.array(y_pred)).mean() * 100

    per_class_dict = {
        ID2LABEL[i]: round(float(per_class_f1[i]), 4)
        for i in range(NUM_LABELS)
    }

    return {
        "n_samples":       len(y_true),
        "accuracy":        round(float(accuracy), 4),
        "macro_f1":        round(float(macro_f1), 4),
        "macro_precision": round(float(macro_pre), 4),
        "macro_recall":    round(float(macro_rec), 4),
        "per_class_f1":    per_class_dict,
        "classification_report": report_str,
        "confusion_matrix": cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ Bảng so sánh SOTA
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(our_metrics: dict) -> str:
    """
    Tạo bảng so sánh kết quả của hệ thống đề xuất với
    các SOTA Benchmark trên ViFactCheck.
    """
    our_f1   = our_metrics["macro_f1"]
    our_pre  = our_metrics["macro_precision"]
    our_rec  = our_metrics["macro_recall"]

    # Xác định rank so với SOTA
    sota_f1s = [b["macro_f1"] for b in SOTA_BENCHMARKS]
    better_than = sum(1 for f in sota_f1s if our_f1 > f)
    rank_str = (
        f"vượt {better_than}/{len(sota_f1s)} mô hình SOTA"
        if better_than > 0
        else "chưa vượt mô hình SOTA nào"
    )

    # Tính delta so với từng baseline
    rows: list[str] = []

    # Header
    col_widths = (26, 12, 12, 12, 8, 36)
    divider = "+" + "+".join("-" * w for w in col_widths) + "+"
    header = (
        f"| {'Mô hình':<{col_widths[0]-1}}"
        f"| {'Macro-F1':>{col_widths[1]-1}} "
        f"| {'Precision':>{col_widths[2]-1}} "
        f"| {'Recall':>{col_widths[3]-1}} "
        f"| {'ΔF1':>{col_widths[4]-1}} "
        f"| {'Ghi chú':<{col_widths[5]-1}}|"
    )

    rows.append(divider)
    rows.append(header)
    rows.append(divider.replace("-", "="))

    # SOTA baselines
    for b in SOTA_BENCHMARKS:
        delta  = our_f1 - b["macro_f1"]
        delta_s = f"{delta:+.2f}%"
        row = (
            f"| {b['model']:<{col_widths[0]-1}}"
            f"| {b['macro_f1']:>{col_widths[1]-2}.2f}%"
            f"| {'N/A':>{col_widths[2]-1}} "
            f"| {'N/A':>{col_widths[3]-1}} "
            f"| {delta_s:>{col_widths[4]-1}} "
            f"| {b['note']:<{col_widths[5]-1}}|"
        )
        rows.append(row)

    rows.append(divider)

    # Mô hình đề xuất
    our_row = (
        f"| {'[Đề xuất] Graph2Seq+MLP':<{col_widths[0]-1}}"
        f"| {our_f1:>{col_widths[1]-2}.2f}%"
        f"| {our_pre:>{col_widths[2]-2}.2f}%"
        f"| {our_rec:>{col_widths[3]-2}.2f}%"
        f"| {'—':>{col_widths[4]-1}} "
        f"| {'PhoBERT-large+MLP+XAI Pruning':<{col_widths[5]-1}}|"
    )
    rows.append(our_row)
    rows.append(divider)

    # Phần per-class F1
    rows.append("")
    rows.append("Per-class F1 của mô hình đề xuất:")
    pc_div = "+------------------+----------+"
    rows.append(pc_div)
    rows.append(f"| {'Nhãn':<17}| {'F1':>8} |")
    rows.append(pc_div)
    for label_name, f1_val in our_metrics["per_class_f1"].items():
        rows.append(f"| {label_name:<17}| {f1_val:>7.2f}% |")
    rows.append(pc_div)

    # Kết luận
    rows.append("")
    rows.append(f"→ Kết luận: Hệ thống đề xuất đạt Macro-F1 = {our_f1:.2f}%, {rank_str}.")
    rows.append(
        f"  Accuracy trên Test = {our_metrics['accuracy']:.2f}% | "
        f"N = {our_metrics['n_samples']:,} mẫu."
    )

    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# ⑦ I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(obj: object, path: Path, description: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.success(f"Đã lưu {description} → {path}")


def _save_text(text: str, path: Path, description: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    logger.success(f"Đã lưu {description} → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Logger setup ─────────────────────────────────────────────────────────
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    banner = "=" * 68
    logger.info(banner)
    logger.info("  Bước 5: Subgraph Pruning (XAI) + Evaluation + SOTA Comparison")
    logger.info(banner)

    t_start = time.perf_counter()

    # ── Thiết bị ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiết bị tính toán: {device}")

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC A: Nạp dữ liệu
    # ────────────────────────────────────────────────────────────────────────
    logger.info("── [A] Nạp dữ liệu ─────────────────────────────────────────")

    test_labels = load_test_labels()               # {id: label_id}
    all_records = load_arguments_cache()           # toàn bộ bản ghi step3

    # Lọc chỉ giữ các bản ghi thuộc tập Test
    test_ids     = set(test_labels.keys())
    test_records = [r for r in all_records if int(r["id"]) in test_ids]
    logger.info(
        f"Tìm thấy {len(test_records):,} / {len(all_records):,} bản ghi "
        f"thuộc tập Test ({len(test_ids):,} nhãn)."
    )

    if not test_records:
        logger.error(
            "Không có bản ghi Test nào khớp với cache step3. "
            "Kiểm tra lại annotation_id trong test.csv và step3_arguments.json."
        )
        sys.exit(1)

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC B: MLP Inference  –  Lấy nhãn dự đoán y*
    # ────────────────────────────────────────────────────────────────────────
    logger.info("── [B] MLP Inference (lấy nhãn dự đoán y*) ────────────────")

    try:
        model     = load_mlp_model(device)
        tokenizer = AutoTokenizer.from_pretrained(
            ENCODER_MODEL_NAME,
            use_fast=True,
        )
        predictions = run_mlp_inference(
            model, tokenizer, test_records, device, batch_size=16
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.warning(
            "Không có checkpoint MLP. "
            "Sử dụng dự đoán ngẫu nhiên cân bằng cho mục đích demo pipeline."
        )
        rng = np.random.default_rng(SEED)
        predictions = {
            int(r["id"]): int(rng.integers(0, NUM_LABELS))
            for r in test_records
        }

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC C: Subgraph Pruning  –  Tạo G* cho từng mẫu Test
    # ────────────────────────────────────────────────────────────────────────
    logger.info("── [C] Subgraph Pruning (tạo G*) ───────────────────────────")

    pruned_records: list[dict] = []
    for rec in tqdm(test_records, desc="Pruning", unit="sample", dynamic_ncols=True):
        rid  = int(rec["id"])
        pred = predictions.get(rid, LABEL2ID["NEI"])
        pruned = prune_subgraph(rec, predicted_label=pred)
        pruned_records.append(pruned)

    # Thống kê phân phối nhãn dự đoán
    from collections import Counter
    pred_dist = Counter(
        ID2LABEL[predictions.get(int(r["id"]), 2)] for r in test_records
    )
    logger.info(
        "Phân phối nhãn dự đoán: "
        + " | ".join(f"{k}={v}" for k, v in pred_dist.items())
    )

    # Lưu đồ thị con G*
    _save_json(
        [
            {
                "id":              r["id"],
                "claim":           r.get("claim", ""),
                "predicted_label": r.get("predicted_label"),
                "pruned_reason":   r.get("pruned_reason"),
                "pruned_nodes":    r.get("pruned_nodes", []),
                "edges":           r.get("edges", []),
            }
            for r in pruned_records
        ],
        OUT_PRUNED,
        "đồ thị con G* (Subgraph Pruning)",
    )

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC D: LLM Explanation  –  Sinh lời giải thích văn bản T
    # ────────────────────────────────────────────────────────────────────────
    logger.info("── [D] LLM Explanation (sinh giải thích T) ─────────────────")

    api_available = bool(
        (cfg.llm_backend == "openai" and cfg.openai_api_key)
        or (cfg.llm_backend == "gemini" and cfg.gemini_api_key)
    )

    if api_available:
        explained_records = asyncio.run(
            run_explanation_pipeline(pruned_records)
        )
    else:
        logger.warning(
            "Không tìm thấy API key. "
            "Bỏ qua gọi LLM, dùng fallback explanation tự động."
        )
        explained_records = []
        for rec in pruned_records:
            lbl = rec.get("predicted_label", "UNKNOWN")
            fallback = (
                f"[Tự động] Tuyên bố được phân loại là {lbl}. "
                f"{rec.get('pruned_reason', '')}"
            )
            explained_records.append({**rec, "explanation": fallback})

    # Lưu lời giải thích
    _save_json(
        [
            {
                "id":          r["id"],
                "claim":       r.get("claim", ""),
                "predicted":   r.get("predicted_label"),
                "explanation": r.get("explanation", ""),
            }
            for r in explained_records
        ],
        OUT_EXPLANATIONS,
        "lời giải thích văn bản T",
    )

    # In mẫu giải thích đầu tiên
    if explained_records:
        first = explained_records[0]
        logger.info(
            f"\n{'─'*60}\n"
            f"  Ví dụ giải thích (id={first.get('id')}):\n"
            f"  Claim   : {first.get('claim', '')[:80]}...\n"
            f"  Phán quyết: {first.get('predicted_label')}\n"
            f"  T       : {first.get('explanation', '')[:200]}...\n"
            f"{'─'*60}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC E: Evaluation  –  Tính toán độ đo scikit-learn
    # ────────────────────────────────────────────────────────────────────────
    logger.info("── [E] Evaluation (scikit-learn metrics) ───────────────────")

    # Ghép y_true và y_pred chỉ trên các mẫu có cả hai
    y_true: list[int] = []
    y_pred: list[int] = []

    for rid, true_lid in test_labels.items():
        pred_lid = predictions.get(rid)
        if pred_lid is not None:
            y_true.append(true_lid)
            y_pred.append(pred_lid)

    if len(y_true) == 0:
        logger.error("Không có mẫu nào để đánh giá!")
        sys.exit(1)

    metrics = compute_metrics(y_true, y_pred)

    logger.info(
        f"\nClassification Report:\n{metrics['classification_report']}"
    )

    # Lưu metrics
    _save_json(metrics, OUT_METRICS, "kết quả đo lường")

    # ────────────────────────────────────────────────────────────────────────
    # BƯỚC F: So sánh SOTA + In báo cáo tổng hợp
    # ────────────────────────────────────────────────────────────────────────
    logger.info("── [F] Bảng so sánh SOTA ViFactCheck ──────────────────────")

    comparison_table = build_comparison_table(metrics)

    # Confusion matrix string
    cm_labels = [ID2LABEL[i] for i in range(NUM_LABELS)]
    cm_arr    = np.array(metrics["confusion_matrix"])
    cm_header = "  Confusion Matrix (hàng=True, cột=Pred):\n"
    cm_header += "  " + " ".join(f"{l:>10}" for l in cm_labels) + "\n"
    for i, row in enumerate(cm_arr):
        cm_header += f"  {cm_labels[i]:<10}" + "".join(f"{v:>10}" for v in row) + "\n"

    report_lines = [
        "=" * 68,
        "  BÁO CÁO KẾT QUẢ THỰC NGHIỆM – CHƯƠNG 4",
        "  Hệ thống kiểm tra thông tin tiếng Việt (ViFactCheck)",
        "=" * 68,
        "",
        "Mô hình đề xuất: Graph2Seq + PhoBERT-large + MLP Classifier",
        "             + Subgraph Pruning XAI + LLM Explanation",
        f"Tập dữ liệu   : ViFactCheck (Test split, N={metrics['n_samples']:,})",
        f"Encoder        : {ENCODER_MODEL_NAME}",
        f"Nhãn           : SUPPORTED / REFUTED / NEI",
        "",
        "─" * 68,
        "  A. Kết quả đo lường tổng hợp (Macro-average):",
        "─" * 68,
        f"  Macro-F1      : {metrics['macro_f1']:.4f}%",
        f"  Macro-Precision: {metrics['macro_precision']:.4f}%",
        f"  Macro-Recall  : {metrics['macro_recall']:.4f}%",
        f"  Accuracy      : {metrics['accuracy']:.4f}%",
        "",
        "─" * 68,
        "  B. So sánh với SOTA Benchmark ViFactCheck:",
        "─" * 68,
        "",
        comparison_table,
        "",
        "─" * 68,
        "  C. Ma trận nhầm lẫn (Confusion Matrix):",
        "─" * 68,
        "",
        cm_header,
        "─" * 68,
        "  D. Classification Report chi tiết (scikit-learn):",
        "─" * 68,
        "",
        metrics["classification_report"],
        "=" * 68,
        f"  Thời gian chạy toàn bộ: {time.perf_counter() - t_start:.1f}s",
        "  Outputs:",
        f"    {OUT_PRUNED}",
        f"    {OUT_EXPLANATIONS}",
        f"    {OUT_METRICS}",
        f"    {OUT_REPORT}",
        "=" * 68,
    ]

    full_report = "\n".join(report_lines)

    # In ra console
    for line in report_lines:
        logger.info(line)

    # Lưu báo cáo văn bản
    _save_text(full_report, OUT_REPORT, "báo cáo tổng hợp")

    elapsed = time.perf_counter() - t_start
    logger.success(
        f"Bước 5 hoàn thành! "
        f"Macro-F1={metrics['macro_f1']:.2f}% | "
        f"Thời gian: {elapsed:.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
