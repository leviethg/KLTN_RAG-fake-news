"""
step1_decomposition.py
======================
Phân rã câu claim thành đồ thị tiểu tuyên bố nguyên thủy (atomic sub-claim graph)
bằng cách gọi LLM bất đồng bộ (aiohttp) trên toàn bộ tập train.csv.

Cấu trúc đầu ra:
  cache/step1_graph.json  →  List[ClaimGraph]
      {
        "id":    int,             # annotation_id
        "claim": str,             # câu claim gốc
        "nodes": [{"id": str, "text": str}, ...],
        "edges": [{"source": str, "target": str}, ...]
      }

Node gốc luôn có id = "c0" và text = câu claim gốc.
Các tiểu tuyên bố được đánh id "c1", "c2", ...
Edge source→target biểu diễn quan hệ phụ thuộc logic c_i → c_j.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm as stdtqdm

# ── project root trên sys.path ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    CACHE_DIR,
    DATA_DIR,
    MAX_SUB_CLAIMS,
    cfg,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_CSV      = DATA_DIR / "train.csv"
OUTPUT_FILE    = CACHE_DIR / "step1_graph.json"

CONCURRENCY    = 15          # số request song song tối đa
MAX_RPM        = 300         # giới hạn request/phút (Gemini free = 15, dùng 12 để an toàn)
RETRY_LIMIT    = cfg.api_max_retries
RETRY_DELAY    = cfg.api_retry_delay
TIMEOUT_SEC    = cfg.api_timeout

# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Giới hạn tốc độ gọi API: đảm bảo khoảng cách tối thiểu `60/rpm` giây
    giữa các lần gọi thực sự. Lock nội bộ giúp serialize các lần `wait()`.
    """
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._next_allowed: float = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_allowed = time.monotonic() + self._interval


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class SubClaimNode(BaseModel):
    id:   str = Field(..., description="Định danh nút, ví dụ 'c0', 'c1', ...")
    text: str = Field(..., description="Nội dung văn bản của tiểu tuyên bố")


class SubClaimEdge(BaseModel):
    source: str = Field(..., description="ID nút nguồn")
    target: str = Field(..., description="ID nút đích")


class SubClaimGraph(BaseModel):
    """Kết quả LLM trả về cho một câu claim."""
    nodes: list[SubClaimNode] = Field(..., min_length=1)
    edges: list[SubClaimEdge] = Field(default_factory=list)

    @field_validator("nodes")
    @classmethod
    def root_node_exists(cls, v: list[SubClaimNode]) -> list[SubClaimNode]:
        ids = {n.id for n in v}
        if "c0" not in ids:
            raise ValueError("Danh sách nodes phải chứa nút gốc có id='c0'.")
        return v

    @field_validator("edges")
    @classmethod
    def edges_reference_valid_nodes(cls, edges: list[SubClaimEdge], info: Any) -> list[SubClaimEdge]:
        # Validation lazy – sẽ được thực thi sau khi nodes đã được validate
        return edges


class ClaimGraphRecord(BaseModel):
    """Bản ghi cuối cùng lưu vào file JSON."""
    id:    int
    claim: str
    nodes: list[SubClaimNode]
    edges: list[SubClaimEdge]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích ngôn ngữ học và kiểm tra thông tin tiếng Việt.
Nhiệm vụ của bạn là phân rã một câu CLAIM thành các tiểu tuyên bố nguyên thủy (atomic sub-claims) \
và xây dựng đồ thị phụ thuộc logic giữa chúng.

NGUYÊN TẮC BẮT BUỘC:
1. Mỗi tiểu tuyên bố phải là một mệnh đề đơn, không thể phân rã thêm.
2. Tìm các mối quan hệ phụ thuộc logic chéo: nếu hiểu c_i mà không có c_j thì sai → ghi edge c_j → c_i.
3. Nếu KHÔNG tồn tại phụ thuộc chéo nào, thêm edge từ TẤT CẢ các nút con về nút gốc c0.
4. Nút gốc c0 LUÔN là câu claim ban đầu (nguyên văn).
5. Số tiểu tuyên bố tối đa: {max_sub_claims} (không kể c0).
6. Chỉ trả về JSON thuần, không có markdown, không có giải thích ngoài JSON.
7. BẮT BUỘC ESCAPE NGOẶC KÉP: Nếu trong câu claim hoặc tiểu tuyên bố có chứa dấu ngoặc kép ("), bạn PHẢI escape nó thành \\\" bên trong chuỗi JSON để tránh làm vỡ cú pháp JSON.

ĐỊNH DẠNG JSON BẮT BUỘC:
{{
  "nodes": [
    {{"id": "c0", "text": "<câu claim gốc>"}},
    {{"id": "c1", "text": "<tiểu tuyên bố 1>"}},
    ...
  ],
  "edges": [
    {{"source": "c1", "target": "c2"}},
    ...
  ]
}}
""".format(max_sub_claims=MAX_SUB_CLAIMS)


def build_user_prompt(claim: str) -> str:
    return (
        f"CLAIM: {claim}\n\n"
        "Hãy phân rã CLAIM trên thành đồ thị tiểu tuyên bố theo đúng định dạng JSON đã quy định."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM API callers  (OpenAI-compatible & Gemini)
# ─────────────────────────────────────────────────────────────────────────────

async def _call_openai(
    session: aiohttp.ClientSession,
    claim: str,
) -> str:
    """Gọi OpenAI-compatible chat/completions endpoint."""
    url = cfg.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.openai_model,
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(claim)},
        ],
    }
    async with session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["choices"][0]["message"]["content"]


async def _call_gemini(
    session: aiohttp.ClientSession,
    claim: str,
) -> str:
    """Gọi Gemini generateContent endpoint."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg.gemini_model}:generateContent?key={cfg.gemini_api_key}"
    )
    combined = SYSTEM_PROMPT + "\n\n" + build_user_prompt(claim)
    payload = {
        "contents": [{"parts": [{"text": combined}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
        },
    }
    async with session.post(url, json=payload) as resp:
        if not resp.ok:
            body = await resp.text()
            # Phân biệt RESOURCE_EXHAUSTED (hết quota ngày) vs RATE_LIMIT_EXCEEDED (vượt RPM)
            try:
                err_data = json.loads(body)
                api_status = err_data.get("error", {}).get("status", "")
            except Exception:
                api_status = ""
            err = aiohttp.ClientResponseError(
                resp.request_info, resp.history,
                status=resp.status,
                message=f"[{api_status}] {body[:300]}",
            )
            raise err
        data = await resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def call_llm(session: aiohttp.ClientSession, claim: str) -> str:
    """Dispatcher chọn backend từ cfg.llm_backend."""
    if cfg.llm_backend == "gemini":
        return await _call_gemini(session, claim)
    return await _call_openai(session, claim)


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction & fallback graph builder
# ─────────────────────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _extract_json(raw: str) -> dict:
    """Trích xuất JSON từ chuỗi trả về của LLM (có thể chứa markdown)."""
    raw = raw.strip()
    # Loại bỏ code-fence nếu có
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    raw = re.sub(r"\s*```$", "", raw)
    m = _JSON_RE.search(raw)
    if not m:
        raise ValueError(f"Không tìm thấy JSON trong phản hồi LLM:\n{raw[:300]}")
    return json.loads(m.group())



def _make_fallback_graph(claim: str) -> SubClaimGraph:
    """Trả về đồ thị tối thiểu (chỉ nút gốc) khi LLM thất bại hoàn toàn."""
    return SubClaimGraph(
        nodes=[SubClaimNode(id="c0", text=claim)],
        edges=[],
    )


def _ensure_connectivity(graph: SubClaimGraph) -> SubClaimGraph:
    """
    Nếu không có cạnh nào → nối tất cả nút con về c0.
    Đảm bảo tính liên thông toàn cục của đồ thị.
    """
    if graph.edges:
        return graph

    child_ids = [n.id for n in graph.nodes if n.id != "c0"]
    if not child_ids:
        return graph

    new_edges = [SubClaimEdge(source=cid, target="c0") for cid in child_ids]
    return SubClaimGraph(nodes=graph.nodes, edges=new_edges)


# def parse_llm_response(raw: str, claim: str) -> SubClaimGraph:
#     """Phân tích phản hồi LLM, validate bằng Pydantic, đảm bảo liên thông."""
#     try:
#         data = _extract_json(raw)
#         graph = SubClaimGraph.model_validate(data)
#         graph = _ensure_connectivity(graph)
#         return graph
#     except Exception as exc:
#         logger.warning(f"Lỗi parse LLM response: {exc}. Dùng fallback graph.")
#         return _make_fallback_graph(claim)
def parse_llm_response(raw: str, claim: str) -> SubClaimGraph:
    """Phân tích phản hồi LLM, validate bằng Pydantic, đảm bảo liên thông."""
    try:
        data = _extract_json(raw)
        
        # 1. Bảo đảm data['nodes'] là danh sách hợp lệ
        if "nodes" not in data or not isinstance(data["nodes"], list):
            data["nodes"] = []
        
        # 2. Tự động kiểm tra và vá nút gốc c0 nếu LLM bỏ quên
        node_ids = {n.get("id") for n in data["nodes"] if isinstance(n, dict) and "id" in n}
        if "c0" not in node_ids:
            data["nodes"].insert(0, {"id": "c0", "text": claim})
            
        graph = SubClaimGraph.model_validate(data)
        graph = _ensure_connectivity(graph)
        return graph
    except Exception as exc:
        logger.warning(f"Lỗi parse LLM response: {exc}. Dùng fallback graph.")
        return _make_fallback_graph(claim)

# ─────────────────────────────────────────────────────────────────────────────
# Worker bất đồng bộ
# ─────────────────────────────────────────────────────────────────────────────

async def process_claim(
    session: aiohttp.ClientSession,
    rate_limiter: RateLimiter,
    row: dict,
) -> ClaimGraphRecord:
    """
    Xử lý một câu claim: gọi LLM với retry, parse kết quả,
    trả về ClaimGraphRecord sẵn sàng ghi vào file.
    """
    claim_id: int = int(row["annotation_id"])
    claim_text: str = str(row["claim"])

    last_exc: Exception | None = None
    max_attempts = RETRY_LIMIT + 1
    for attempt in range(1, max_attempts + 1):
        sleep_time: float = 0.0
        try:
            await rate_limiter.wait()
            raw = await asyncio.wait_for(
                call_llm(session, claim_text),
                timeout=TIMEOUT_SEC,
            )
            graph = parse_llm_response(raw, claim_text)
            return ClaimGraphRecord(
                id=claim_id,
                claim=claim_text,
                nodes=graph.nodes,
                edges=graph.edges,
            )
        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning(f"[id={claim_id}] Timeout lần {attempt}/{max_attempts}.")
            sleep_time = RETRY_DELAY
        except aiohttp.ClientResponseError as e:
            last_exc = e
            is_quota_exhausted = "RESOURCE_EXHAUSTED" in (e.message or "")
            if e.status == 429 and is_quota_exhausted:
                # Hết quota ngày – không có ích khi retry, dừng toàn bộ pipeline
                logger.error(
                    f"[id={claim_id}] Hết quota API (RESOURCE_EXHAUSTED). "
                    "Hãy kiểm tra quota tại https://aistudio.google.com hoặc đợi quota reset (thường reset lúc 0h PT)."
                )
                raise QuotaExhaustedError("RESOURCE_EXHAUSTED")
            elif e.status in (429, 500, 502, 503, 504):
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                jitter = random.uniform(0.0, 1.0)
                sleep_time = wait + jitter
                logger.warning(
                    f"[id={claim_id}] HTTP {e.status} (lần {attempt}/{max_attempts}), "
                    f"thử lại sau {sleep_time:.1f}s. Chi tiết: {e.message[:200]}"
                )
            else:
                logger.error(f"[id={claim_id}] HTTP {e.status} không thể retry. Chi tiết: {e.message[:200]}")
                break
        except Exception as e:
            last_exc = e
            logger.warning(f"[id={claim_id}] Lỗi lần {attempt}/{max_attempts}: {e}")
            sleep_time = RETRY_DELAY

        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    logger.error(f"[id={claim_id}] Thất bại sau {max_attempts} lần. Dùng fallback.")
    return ClaimGraphRecord(
        id=claim_id,
        claim=claim_text,
        nodes=[SubClaimNode(id="c0", text=claim_text)],
        edges=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hàm chính
# ─────────────────────────────────────────────────────────────────────────────

def load_train_data() -> list[dict]:
    """Đọc train.csv, chuẩn hoá tên cột, trả về list dict."""
    logger.info(f"Đọc dữ liệu từ {TRAIN_CSV} ...")
    df = pd.read_csv(TRAIN_CSV)

    # Chuẩn hoá: ưu tiên cột 'claim', fallback sang 'Statement'
    if "claim" in df.columns:
        df = df.rename(columns={"claim": "claim"})
    elif "Statement" in df.columns:
        df = df.rename(columns={"Statement": "claim"})
    else:
        raise KeyError("Không tìm thấy cột 'claim' hoặc 'Statement' trong train.csv.")

    # Chuẩn hoá id
    if "id" in df.columns:
        df = df.rename(columns={"id": "annotation_id"})
    elif "annotation_id" not in df.columns:
        raise KeyError("Không tìm thấy cột 'id' hoặc 'annotation_id' trong train.csv.")

    df = df[["annotation_id", "claim"]].dropna(subset=["claim"])
    logger.info(f"Tổng số claim cần xử lý: {len(df):,}")
    return df.to_dict(orient="records")


# def load_existing_cache() -> dict[int, ClaimGraphRecord]:
#     """Nạp kết quả đã xử lý từ file cache (nếu có) để resume."""
#     if not OUTPUT_FILE.exists():
#         return {}
#     try:
#         with OUTPUT_FILE.open(encoding="utf-8") as f:
#             raw_list: list[dict] = json.load(f)
#         cache = {r["id"]: ClaimGraphRecord.model_validate(r) for r in raw_list}
#         logger.info(f"Resume: tìm thấy {len(cache):,} bản ghi đã xử lý trong cache.")
#         return cache
#     except Exception as e:
#         logger.warning(f"Không thể nạp cache cũ ({e}), bắt đầu lại từ đầu.")
#         return {}

def load_existing_cache() -> dict[str, ClaimGraphRecord]:
    """Nạp kết quả đã xử lý từ file cache (nếu có) để resume bằng khóa kết hợp."""
    if not OUTPUT_FILE.exists():
        return {}
    try:
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            raw_list: list[dict] = json.load(f)
        # Sử dụng khóa kết hợp ID_Claim để tránh bị đè dữ liệu trùng ID
        cache = {f"{r['id']}_{r['claim']}": ClaimGraphRecord.model_validate(r) for r in raw_list}
        logger.info(f"Resume: tìm thấy {len(cache):,} bản ghi độc nhất đã xử lý trong cache.")
        return cache
    except Exception as e:
        logger.warning(f"Không thể nạp cache cũ ({e}), bắt đầu lại từ đầu.")
        return {}

def save_results(records: list[ClaimGraphRecord]) -> None:
    """Ghi toàn bộ kết quả ra cache/step1_graph.json."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in records]
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.success(f"Đã lưu {len(records):,} bản ghi vào {OUTPUT_FILE}")


SAVE_EVERY = 100   # lưu cache tạm mỗi N claim hoàn thành


class QuotaExhaustedError(Exception):
    """Ném ra khi API báo hết quota (RESOURCE_EXHAUSTED) – không thể retry."""


async def _worker(
    worker_id: int,
    session: aiohttp.ClientSession,
    queue: asyncio.Queue,
    rate_limiter: RateLimiter,
    cache: dict[int, ClaimGraphRecord],
    results: list[ClaimGraphRecord],
    lock: asyncio.Lock,
    stop_event: asyncio.Event,
    pbar,
) -> None:
    """Worker kéo task từ queue và xử lý tuần tự; nhiều worker chạy song song."""
    while not stop_event.is_set():
        try:
            row = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            result = await process_claim(session, rate_limiter, row)
        except QuotaExhaustedError:
            stop_event.set()
            queue.task_done()
            return
        except Exception as e:
            claim_id = int(row["annotation_id"])
            claim_text = str(row["claim"])
            logger.error(f"[worker={worker_id}] Lỗi không mong đợi id={claim_id}: {e}")
            result = ClaimGraphRecord(
                id=claim_id, claim=claim_text,
                nodes=[SubClaimNode(id="c0", text=claim_text)], edges=[],
            )
        async with lock:
            results.append(result)
           
            cache[f"{result.id}_{result.claim}"] = result
            pbar.update(1)
            if len(results) % SAVE_EVERY == 0:
                save_results(list(cache.values()))
                logger.info(f"Checkpoint: {len(results)} claim mới (tổng cache: {len(cache):,}).")
        queue.task_done()


async def run_pipeline(
    rows: list[dict],
    cache: dict[int, ClaimGraphRecord],
) -> list[ClaimGraphRecord]:
    """
    Queue-based worker pool: chỉ CONCURRENCY worker hoạt động cùng lúc.
    Rate limiter được chia sẻ giữa các worker → đảm bảo tổng tốc độ ≤ MAX_RPM.
    """
    queue: asyncio.Queue = asyncio.Queue()
    for row in rows:
        await queue.put(row)

    rate_limiter = RateLimiter(rpm=MAX_RPM)
    results: list[ClaimGraphRecord] = []
    lock = asyncio.Lock()
    stop_event = asyncio.Event()
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_SEC + 10)

    with stdtqdm(total=len(rows), desc="Decomposing claims", unit="claim") as pbar:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            workers = [
                asyncio.create_task(
                    _worker(i, session, queue, rate_limiter, cache, results, lock, stop_event, pbar)
                )
                for i in range(CONCURRENCY)
            ]
            await asyncio.gather(*workers)

    if stop_event.is_set():
        logger.error(
            "Pipeline dừng sớm do hết quota API. "
            f"Đã xử lý được {len(results):,}/{len(rows):,} claim trong lần chạy này."
        )

    return results


async def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    # Kiểm tra API key
    if cfg.llm_backend == "openai" and not cfg.openai_api_key:
        logger.error("OPENAI_API_KEY chưa được thiết lập trong file .env!")
        sys.exit(1)
    if cfg.llm_backend == "gemini" and not cfg.gemini_api_key:
        logger.error("GEMINI_API_KEY chưa được thiết lập trong file .env!")
        sys.exit(1)

    logger.info(f"Backend LLM: {cfg.llm_backend.upper()} | "
                f"Model: {cfg.openai_model if cfg.llm_backend == 'openai' else cfg.gemini_model} | "
                f"Concurrency: {CONCURRENCY}")


    rows = load_train_data()
    cache = load_existing_cache()

    # # 1. Lọc ra những claim chưa xử lý dựa trên khóa kết hợp
    # pending = []
    # for r in rows:
    #     key = f"{int(r['annotation_id'] if 'annotation_id' in r else r['id'])}_{str(r['claim'])}"
    #     if key not in cache:
    #         pending.append(r)
    # logger.info(f"Số claim cần xử lý mới: {len(pending):,} / {len(rows):,}")

    # if pending:
    #     await run_pipeline(pending, cache)

    # # 2. Ánh xạ ngược lại theo thứ tự 5,062 dòng gốc của file CSV
    # ordered = []
    # for r in rows:
    #     key = f"{int(r['annotation_id'] if 'annotation_id' in r else r['id'])}_{str(r['claim'])}"
    #     if key in cache:
    #         ordered.append(cache[key])

    # save_results(ordered)
    # logger.success("Bước 1 (Claim Decomposition) hoàn thành!")
    

    # 1. Lọc ra những claim thực sự chưa xử lý dựa trên khóa kết hợp
    pending = []
    for r in rows:
        key = f"{int(r['annotation_id'] if 'annotation_id' in r else r['id'])}_{str(r['claim'])}"
        if key not in cache:
            pending.append(r)
    logger.info(f"Số claim cần xử lý mới: {len(pending):,} / {len(rows):,}")

    if pending:
        await run_pipeline(pending, cache)

    # 2. Ánh xạ ngược lại theo đúng thứ tự và nội dung của 5,062 dòng gốc
    ordered = []
    for r in rows:
        key = f"{int(r['annotation_id'] if 'annotation_id' in r else r['id'])}_{str(r['claim'])}"
        if key in cache:
            ordered.append(cache[key])

    save_results(ordered)
    logger.success("Bước 1 (Claim Decomposition) hoàn thành chuẩn xác!")

if __name__ == "__main__":
    asyncio.run(main())
