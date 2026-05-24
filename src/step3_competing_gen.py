"""
step3_competing_gen.py
======================
Sinh lập luận đối kháng (Competing Arguments) cho từng tiểu tuyên bố.

Luồng xử lý:
  1. Đọc cache/step2_rag.json  – đồ thị tiểu tuyên bố + bằng chứng E_i từ Bước 2.
  2. Với mỗi nút (c_i) trong đồ thị:
       Gọi LLM bất đồng bộ với prompt ép buộc hai vai trò phản biện độc lập:
         - Vai trò LUẬT SƯ BẢO VỆ : sinh arg_support (c_i^+) – giả định nhãn TRUE.
         - Vai trò CÔNG TỐ VIÊN  : sinh arg_refute  (c_i^-) – giả định nhãn FALSE.
       Nếu E_i không chứa thông tin liên quan, cả hai chuỗi phải là NEI_MARKER
       để pipeline sau nhận diện nhãn NEI tường minh.
  3. Nhúng hai chuỗi lập luận vào thuộc tính của từng nút.
  4. Lưu kết quả vào cache/step3_arguments.json.

Cấu trúc đầu ra (List[dict]):
  {
    "id":    int,
    "claim": str,
    "nodes": [
      {
        "id":          str,
        "text":        str,
        "arg_support": str,   # lập luận ủng hộ  (c_i^+) hoặc NEI_MARKER
        "arg_refute":  str,   # lập luận phản bác (c_i^-) hoặc NEI_MARKER
      },
      ...
    ],
    "edges":     [{"source": str, "target": str}, ...],
    "evidences": {"c0": [...], "c1": [...], ...}
  }
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as atqdm

# ── project root trên sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    CACHE_DIR,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    cfg,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE   = CACHE_DIR / "step2_rag.json"
OUTPUT_FILE  = CACHE_DIR / "step3_arguments.json"

CONCURRENCY  = 40          # số node được xử lý song song tối đa
RETRY_LIMIT  = cfg.api_max_retries
RETRY_DELAY  = cfg.api_retry_delay
TIMEOUT_SEC  = cfg.api_timeout
SAVE_EVERY   = 50         # ghi checkpoint sau mỗi N claim

# Chuỗi nhận diện NEI – xuất hiện trong cả arg_support lẫn arg_refute
NEI_MARKER   = "KHÔNG CÓ BẰNG CHỨNG ĐỦ ĐỂ ĐÁNH GIÁ – NEI"

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schema
# ─────────────────────────────────────────────────────────────────────────────

class ArgumentPair(BaseModel):
    """Cặp lập luận đối kháng cho một tiểu tuyên bố."""
    arg_support: str = Field(..., description="Lập luận ủng hộ (c_i^+) hoặc NEI_MARKER")
    arg_refute:  str = Field(..., description="Lập luận phản bác (c_i^-) hoặc NEI_MARKER")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
Bạn là một hệ thống lập luận đối kháng phục vụ kiểm tra thông tin tiếng Việt.
Với mỗi TIỂU TUYÊN BỐ và TẬP BẰNG CHỨNG đi kèm, bạn phải đóng hai vai trò \
phản biện HOÀN TOÀN ĐỘC LẬP và không được tham chiếu chéo lẫn nhau:

━━━ VAI TRÒ 1 – LUẬT SƯ BẢO VỆ (arg_support) ━━━
• Giả định tiểu tuyên bố là ĐÚNG (nhãn SUPPORTED).
• Chọn lọc và tổng hợp bằng chứng ủng hộ, xây dựng lập luận thuyết phục nhất.
• Chỉ dùng thông tin trong TẬP BẰNG CHỨNG; không được suy diễn vượt ra ngoài.

━━━ VAI TRÒ 2 – CÔNG TỐ VIÊN (arg_refute) ━━━
• Giả định tiểu tuyên bố là SAI (nhãn REFUTED).
• Tìm mâu thuẫn, thiếu sót hoặc thông tin phủ nhận trong bằng chứng.
• Xây dựng lập luận phản bác sắc bén nhất có thể từ TẬP BẰNG CHỨNG.

━━━ QUY TẮC XỬ LÝ KHI BẰNG CHỨNG THIẾU / KHÔNG LIÊN QUAN ━━━
Nếu TẬP BẰNG CHỨNG trống rỗng hoặc không chứa thông tin liên quan đến tiểu tuyên bố:
• CẢ HAI trường arg_support VÀ arg_refute phải được gán ĐÚNG chuỗi sau (không thêm bớt):
  "{NEI_MARKER}"
• Tuyệt đối KHÔNG suy diễn, bịa đặt hoặc dùng kiến thức ngoài bằng chứng.

━━━ ĐỊNH DẠNG ĐẦU RA ━━━
Chỉ trả về JSON thuần (không markdown, không giải thích):
{{
  "arg_support": "<lập luận ủng hộ hoặc chuỗi NEI>",
  "arg_refute":  "<lập luận phản bác hoặc chuỗi NEI>"
}}
"""


def build_user_prompt(sub_claim: str, evidences: list[str]) -> str:
    """Tạo user prompt với tiểu tuyên bố và danh sách bằng chứng."""
    if evidences:
        evidence_block = "\n".join(
            f"[{i + 1}] {sent}" for i, sent in enumerate(evidences)
        )
    else:
        evidence_block = "(Không có bằng chứng nào được truy hồi.)"

    return (
        f"TIỂU TUYÊN BỐ: {sub_claim}\n\n"
        f"TẬP BẰNG CHỨNG:\n{evidence_block}\n\n"
        "Hãy đóng hai vai trò độc lập và trả về JSON theo đúng định dạng đã quy định."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM API callers  (OpenAI-compatible & Gemini)
# ─────────────────────────────────────────────────────────────────────────────

async def _call_openai(
    session: aiohttp.ClientSession,
    sub_claim: str,
    evidences: list[str],
) -> str:
    """Gọi OpenAI-compatible chat/completions endpoint."""
    url = cfg.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":            cfg.openai_model,
        "temperature":      LLM_TEMPERATURE,
        "max_tokens":       LLM_MAX_TOKENS,
        "response_format":  {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(sub_claim, evidences)},
        ],
    }
    async with session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


async def _call_gemini(
    session: aiohttp.ClientSession,
    sub_claim: str,
    evidences: list[str],
) -> str:
    """Gọi Gemini generateContent endpoint."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg.gemini_model}:generateContent?key={cfg.gemini_api_key}"
    )
    combined = SYSTEM_PROMPT + "\n\n" + build_user_prompt(sub_claim, evidences)
    payload = {
        "contents": [{"parts": [{"text": combined}]}],
        "generationConfig": {
            "temperature":      LLM_TEMPERATURE,
            "maxOutputTokens":  LLM_MAX_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def call_llm(
    session: aiohttp.ClientSession,
    sub_claim: str,
    evidences: list[str],
) -> str:
    """Dispatcher chọn backend từ cfg.llm_backend."""
    if cfg.llm_backend == "gemini":
        return await _call_gemini(session, sub_claim, evidences)
    return await _call_openai(session, sub_claim, evidences)


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction & fallback
# ─────────────────────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _extract_json(raw: str) -> dict:
    """Trích xuất JSON từ phản hồi LLM (có thể chứa markdown fence)."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = _JSON_RE.search(raw)
    if not m:
        raise ValueError(f"Không tìm thấy JSON trong phản hồi:\n{raw[:300]}")
    return json.loads(m.group())


def _make_nei_pair() -> ArgumentPair:
    """Trả về cặp lập luận NEI khi LLM thất bại hoàn toàn."""
    return ArgumentPair(arg_support=NEI_MARKER, arg_refute=NEI_MARKER)


def parse_llm_response(raw: str) -> ArgumentPair:
    """Parse và validate phản hồi LLM thành ArgumentPair."""
    try:
        data = _extract_json(raw)
        return ArgumentPair.model_validate(data)
    except Exception as exc:
        logger.warning(f"Lỗi parse LLM response: {exc}. Dùng NEI fallback.")
        return _make_nei_pair()


# ─────────────────────────────────────────────────────────────────────────────
# Worker bất đồng bộ: xử lý một nút (sub-claim)
# ─────────────────────────────────────────────────────────────────────────────

async def process_node(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    claim_id: int,
    node: dict,
    evidences: list[str],
) -> dict:
    """
    Gọi LLM để sinh cặp lập luận đối kháng cho một nút trong đồ thị.
    Trả về nút được bổ sung 'arg_support' và 'arg_refute'.
    """
    node_id   = node["id"]
    node_text = node["text"]

    if not evidences:
        return {**node, "arg_support": NEI_MARKER, "arg_refute": NEI_MARKER}

    last_exc: Exception | None = None
    for attempt in range(1, RETRY_LIMIT + 2):
        async with sem:
            try:
                raw = await asyncio.wait_for(
                    call_llm(session, node_text, evidences),
                    timeout=TIMEOUT_SEC,
                )
                pair = parse_llm_response(raw)
                return {**node, "arg_support": pair.arg_support, "arg_refute": pair.arg_refute}

            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    f"[id={claim_id}/{node_id}] Timeout lần {attempt}. "
                    f"Thử lại sau {RETRY_DELAY}s..."
                )
            except aiohttp.ClientResponseError as exc:
                last_exc = exc
                if exc.status in {429, 500, 502, 503, 504}:
                    wait = RETRY_DELAY * attempt
                    logger.warning(
                        f"[id={claim_id}/{node_id}] HTTP {exc.status} lần {attempt}. "
                        f"Thử lại sau {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(f"[id={claim_id}/{node_id}] Lỗi lần {attempt}: {exc}")

            await asyncio.sleep(RETRY_DELAY)

    logger.error(
        f"[id={claim_id}/{node_id}] Thất bại sau {RETRY_LIMIT + 1} lần thử. "
        f"Lỗi cuối: {last_exc}. Dùng NEI fallback."
    )
    return {**node, "arg_support": NEI_MARKER, "arg_refute": NEI_MARKER}


# ─────────────────────────────────────────────────────────────────────────────
# Worker bất đồng bộ: xử lý toàn bộ một claim record
# ─────────────────────────────────────────────────────────────────────────────

async def process_record(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    record: dict,
) -> dict:
    """
    Xử lý tất cả các nút của một claim, thu thập cặp lập luận,
    trả về record hoàn chỉnh với nodes đã được bổ sung arg_support/arg_refute.
    """
    claim_id: int     = int(record["id"])
    nodes: list[dict] = record.get("nodes", [])
    evidences_map: dict[str, list[str]] = record.get("evidences", {})

    tasks = [
        process_node(
            session,
            sem,
            claim_id,
            node,
            evidences_map.get(node["id"], []),
        )
        for node in nodes
    ]

    augmented_nodes = await asyncio.gather(*tasks)

    return {
        "id":        record["id"],
        "claim":     record.get("claim", ""),
        "nodes":     list(augmented_nodes),
        "edges":     record.get("edges", []),
        "evidences": evidences_map,
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_rag_data() -> list[dict]:
    """Nạp cache/step2_rag.json."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_FILE}. Vui lòng chạy step2_hybrid_rag.py trước."
        )
    with INPUT_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Đọc {len(data):,} bản ghi từ {INPUT_FILE}")
    return data


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
    """Ghi toàn bộ kết quả ra cache/step3_arguments.json."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.success(f"Đã lưu {len(records):,} bản ghi vào {OUTPUT_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline bất đồng bộ chính
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(rag_records: list[dict], done_cache: dict[int, dict]) -> dict[int, dict]:
    """
    Chạy toàn bộ pipeline sinh lập luận đối kháng bất đồng bộ.
    Trả về done_cache được bổ sung kết quả mới.
    """
    pending = [r for r in rag_records if int(r["id"]) not in done_cache]
    logger.info(
        f"Cần xử lý: {len(pending):,} / {len(rag_records):,} "
        f"(đã có: {len(done_cache):,})"
    )

    if not pending:
        return done_cache

    sem     = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SEC + 10)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Tạo tất cả task nhưng xử lý theo thứ tự để checkpoint định kỳ
        tasks = [process_record(session, sem, r) for r in pending]

        processed_count = 0
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Competing Arguments",
            unit="claim",
        ):
            result = await coro
            done_cache[int(result["id"])] = result
            processed_count += 1

            if processed_count % SAVE_EVERY == 0:
                # Checkpoint trung gian theo thứ tự gốc
                id_order = [int(r["id"]) for r in rag_records]
                ordered  = [done_cache[i] for i in id_order if i in done_cache]
                save_output(ordered)
                logger.info(f"Checkpoint: đã lưu {len(ordered):,} bản ghi.")

    return done_cache


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

    logger.info("=== Bước 3: Competing Argument Generation ===")
    logger.info(
        f"Backend={cfg.llm_backend.upper()} | "
        f"Concurrency={CONCURRENCY} | "
        f"Retries={RETRY_LIMIT} | "
        f"NEI_MARKER='{NEI_MARKER}'"
    )

    rag_records = load_rag_data()
    done_cache  = load_existing_output()

    t0 = time.perf_counter()
    done_cache = asyncio.run(run_pipeline(rag_records, done_cache))
    elapsed = time.perf_counter() - t0

    # Ghi kết quả cuối cùng theo thứ tự annotation_id gốc
    id_order = [int(r["id"]) for r in rag_records]
    ordered  = [done_cache[i] for i in id_order if i in done_cache]

    save_output(ordered)
    logger.success(
        f"Bước 3 (Competing Arguments) hoàn thành! "
        f"{len(ordered):,} bản ghi trong {elapsed:.1f}s."
    )


if __name__ == "__main__":
    main()
