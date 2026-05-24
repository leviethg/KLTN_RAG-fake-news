"""
step2_hybrid_rag.py
===================
Coarse-to-Fine Hybrid Retrieval cho tiếng Việt.

Luồng xử lý:
  1. Đọc cache/step1_graph.json  – đồ thị tiểu tuyên bố từ Bước 1.
  2. Đọc data/train.csv          – lấy cột Context làm kho tài liệu.
  3. Với mỗi claim:
       Giai đoạn 1 (Coarse) : Tách từ tiếng Việt bằng pyvi → BM25 (rank_bm25)
                               → top-20 câu ứng viên cho từng tiểu tuyên bố.
       Giai đoạn 2 (Fine)   : Cross-encoder đa ngôn ngữ (BAAI/bge-reranker-v2-m3)
                               → lọc chính xác top-5 câu bằng chứng E_i.
  4. Lưu kết quả vào cache/step2_rag.json.

Cấu trúc đầu ra (List[dict]):
  {
    "id":        int,
    "claim":     str,
    "nodes":     [{"id": str, "text": str}, ...],
    "edges":     [{"source": str, "target": str}, ...],
    "evidences": {
        "c0": ["<câu bằng chứng 1>", ...],   # top-5 cho nút gốc
        "c1": [...],                           # top-5 cho từng tiểu tuyên bố
        ...
    }
  }
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from tqdm import tqdm

# ── project root trên sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DATA_DIR,
    RAG_TOP_K,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_CSV          = DATA_DIR / "train.csv"
INPUT_FILE         = CACHE_DIR / "step1_graph.json"
OUTPUT_FILE        = CACHE_DIR / "step2_rag.json"

BM25_TOP_N         = 20           # số ứng viên sau giai đoạn Coarse
RERANK_TOP_K       = RAG_TOP_K    # số bằng chứng cuối cùng (= 5 từ config)
CROSSENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
CE_BATCH_SIZE      = 32
MIN_SENT_LEN       = 10           # bỏ câu quá ngắn (ký tự)

# ─────────────────────────────────────────────────────────────────────────────
# Tiện ích: tách câu
# ─────────────────────────────────────────────────────────────────────────────

# Tách sau dấu câu kết thúc câu (tiếng Việt + tiếng Anh)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """
    Tách văn bản tiếng Việt thành các câu riêng lẻ.
    Kết hợp tách theo dấu câu và theo dòng mới.
    """
    if not text or not isinstance(text, str):
        return []

    # Chuẩn hoá khoảng trắng thừa giữa các dòng
    text = re.sub(r"\r\n|\r", "\n", text.strip())

    raw_parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw_parts.extend(_SENT_BOUNDARY.split(line))

    return [s.strip() for s in raw_parts if len(s.strip()) >= MIN_SENT_LEN]


# ─────────────────────────────────────────────────────────────────────────────
# Tách từ tiếng Việt (pyvi)
# ─────────────────────────────────────────────────────────────────────────────

def vi_tokenize(text: str) -> list[str]:
    """Tách từ tiếng Việt bằng pyvi, trả về list token chữ thường."""
    segmented = ViTokenizer.tokenize(text)
    return segmented.lower().split()


# ─────────────────────────────────────────────────────────────────────────────
# Giai đoạn 1: Coarse Retrieval – BM25
# ─────────────────────────────────────────────────────────────────────────────

def bm25_retrieve(
    query: str,
    corpus_sents: list[str],
    top_n: int,
) -> list[tuple[str, float]]:
    """
    Xây BM25Okapi index trên corpus_sents đã tách từ tiếng Việt,
    truy hồi top_n câu ứng viên theo điểm BM25 giảm dần.

    Trả về: list[(sentence, bm25_score)]
    """
    tokenized_corpus = [vi_tokenize(s) for s in corpus_sents]
    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)

    tokenized_query = vi_tokenize(query)
    scores: np.ndarray = bm25.get_scores(tokenized_query)

    effective_n = min(top_n, len(corpus_sents))
    top_indices = np.argsort(scores)[::-1][:effective_n]
    return [(corpus_sents[int(i)], float(scores[i])) for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Giai đoạn 2: Fine Reranking – Cross-encoder
# ─────────────────────────────────────────────────────────────────────────────

def load_cross_encoder(model_name: str) -> CrossEncoder:
    logger.info(f"Nạp Cross-encoder: {model_name} ...")
    model = CrossEncoder(model_name, max_length=512)
    logger.info("Cross-encoder đã sẵn sàng.")
    return model


def rerank(
    cross_encoder: CrossEncoder,
    query: str,
    candidates: list[str],
    top_k: int,
    batch_size: int = CE_BATCH_SIZE,
) -> list[tuple[str, float]]:
    """
    Tính điểm tương tác ngữ nghĩa sâu (Cross-encoder) cho mỗi cặp
    (query, candidate), trả về top_k câu theo điểm giảm dần.

    Trả về: list[(sentence, cross_encoder_score)]
    """
    if not candidates:
        return []

    pairs = [[query, c] for c in candidates]
    raw_scores = cross_encoder.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    # raw_scores có thể là ndarray hoặc list
    scores = (
        raw_scores.tolist()
        if isinstance(raw_scores, np.ndarray)
        else list(raw_scores)
    )

    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_data() -> list[dict]:
    """Nạp cache/step1_graph.json."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_FILE}. Vui lòng chạy step1_decomposition.py trước."
        )
    with INPUT_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Đọc {len(data):,} bản ghi từ {INPUT_FILE}")
    return data


def load_context_map() -> dict[int, str]:
    """Nạp ánh xạ annotation_id → Context từ data/train.csv."""
    logger.info(f"Đọc Context từ {TRAIN_CSV} ...")
    df = pd.read_csv(TRAIN_CSV)

    # Chuẩn hoá cột id
    if "annotation_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "annotation_id"})
        else:
            raise KeyError("Không tìm thấy cột 'annotation_id' hoặc 'id' trong train.csv.")

    if "Context" not in df.columns:
        raise KeyError("Không tìm thấy cột 'Context' trong train.csv.")

    df = df[["annotation_id", "Context"]].dropna(subset=["Context"])
    mapping = {int(row["annotation_id"]): str(row["Context"]) for _, row in df.iterrows()}
    logger.info(f"Tổng {len(mapping):,} bản ghi có Context.")
    return mapping


def load_existing_output() -> dict[int, dict]:
    """Nạp kết quả đã xử lý (nếu có) để hỗ trợ resume."""
    if not OUTPUT_FILE.exists():
        return {}
    try:
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            records: list[dict] = json.load(f)
        cache = {int(r["id"]): r for r in records}
        logger.info(f"Resume: {len(cache):,} bản ghi đã có trong {OUTPUT_FILE}.")
        return cache
    except Exception as exc:
        logger.warning(f"Không thể nạp output cũ ({exc}). Bắt đầu lại từ đầu.")
        return {}


def save_output(records: list[dict]) -> None:
    """Ghi toàn bộ kết quả ra cache/step2_rag.json."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.success(f"Đã lưu {len(records):,} bản ghi vào {OUTPUT_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Xử lý một bản ghi
# ─────────────────────────────────────────────────────────────────────────────

def process_record(
    record: dict,
    context_map: dict[int, str],
    cross_encoder: CrossEncoder,
) -> dict:
    """
    Với một ClaimGraphRecord:
      - Tách câu từ Context của bài báo tương ứng.
      - Với từng tiểu tuyên bố (kể cả c0):
          Stage 1: BM25 → top-BM25_TOP_N ứng viên.
          Stage 2: Cross-encoder → top-RERANK_TOP_K bằng chứng E_i.
      Trả về bản ghi được bổ sung trường "evidences".
    """
    claim_id: int = int(record["id"])
    nodes: list[dict] = record.get("nodes", [])

    context_text = context_map.get(claim_id, "")
    corpus_sents = split_sentences(context_text)

    evidences: dict[str, list[str]] = {}

    if not corpus_sents:
        logger.warning(f"[id={claim_id}] Context trống hoặc không tồn tại. Bỏ qua truy hồi.")
        for node in nodes:
            evidences[node["id"]] = []
        return {**record, "evidences": evidences}

    for node in nodes:
        node_id   = node["id"]
        node_text = node["text"]

        # Giai đoạn 1 – BM25 coarse retrieval
        bm25_results = bm25_retrieve(node_text, corpus_sents, top_n=BM25_TOP_N)
        candidates   = [sent for sent, _ in bm25_results]

        # Giai đoạn 2 – Cross-encoder fine reranking
        reranked = rerank(cross_encoder, node_text, candidates, top_k=RERANK_TOP_K)
        evidences[node_id] = [sent for sent, _ in reranked]

    return {**record, "evidences": evidences}


# ─────────────────────────────────────────────────────────────────────────────
# Hàm chính
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    logger.info("=== Bước 2: Coarse-to-Fine Hybrid Retrieval ===")
    logger.info(
        f"BM25 top-N={BM25_TOP_N} | Cross-encoder top-K={RERANK_TOP_K} | "
        f"Model={CROSSENCODER_MODEL}"
    )

    graph_records = load_graph_data()
    context_map   = load_context_map()
    done_cache    = load_existing_output()

    # Xác định bản ghi cần xử lý mới
    pending = [r for r in graph_records if int(r["id"]) not in done_cache]
    logger.info(
        f"Cần xử lý: {len(pending):,} / {len(graph_records):,} "
        f"(đã có: {len(done_cache):,})"
    )

    if pending:
        cross_encoder = load_cross_encoder(CROSSENCODER_MODEL)

        for record in tqdm(pending, desc="Hybrid Retrieval", unit="claim"):
            out = process_record(record, context_map, cross_encoder)
            done_cache[int(out["id"])] = out

    # Giữ thứ tự theo annotation_id gốc
    id_order = [int(r["id"]) for r in graph_records]
    ordered  = [done_cache[i] for i in id_order if i in done_cache]

    save_output(ordered)
    logger.success("Bước 2 (Hybrid RAG) hoàn thành!")


if __name__ == "__main__":
    main()
