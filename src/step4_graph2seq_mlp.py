"""
step4_graph2seq_mlp.py
======================
Graph2Seq (Incident Encoding) + vinai/phobert-large + MLP Classifier
cho bài toán phân loại thông tin tiếng Việt (SUPPORTED / REFUTED / NEI).

Luồng xử lý:
  1. Đọc cache/step3_arguments.json  (đồ thị lập luận đối kháng từ Bước 3).
  2. Đọc nhãn chuẩn từ data/train.csv.
  3. Graph2Seq: Trải phẳng đồ thị → chuỗi văn bản tuyến tính có meta-data indicator.
  4. Tokenize chuỗi bằng tokenizer của vinai/phobert-large.
  5. Kiến trúc: PhoBERT-large (frozen + fine-tune lớp cuối) → h_cls → MLP → Softmax.
  6. Vòng lặp huấn luyện với CrossEntropyLoss + AdamW, hiển thị Loss/Acc mỗi Epoch.
  7. Lưu checkpoint tốt nhất vào models/mlp_checkpoints/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from loguru import logger
from tqdm import tqdm

# ── Project root trên sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    CACHE_DIR,
    DATA_DIR,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    MLP_HIDDEN_DIMS,
    MLP_DROPOUT,
    MLP_LEARNING_RATE,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_WEIGHT_DECAY,
    MLP_GRAD_CLIP,
    MLP_EARLY_STOP_PATIENCE,
    MLP_CHECKPOINT_DIR,
    ENCODER_MAX_LENGTH,
    SEED,
)

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE  = CACHE_DIR / "step3_arguments.json"
TRAIN_CSV   = DATA_DIR  / "train.csv"

# Override encoder → phobert-large theo yêu cầu (config mặc định là phobert-base-v2)
ENCODER_MODEL_NAME   = "vinai/phobert-large"

# Tỷ lệ tách validation
VAL_SPLIT = 0.10

# Marker NEI nhận dạng từ Bước 3
NEI_MARKER = "KHÔNG CÓ BẰNG CHỨNG ĐỦ ĐỂ ĐÁNH GIÁ – NEI"

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Graph2Seq – Incident Encoding
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_sequence(record: dict) -> str:
    """
    Trải phẳng đồ thị lập luận thành chuỗi văn bản tuyến tính có cấu trúc meta-data.

    Mỗi nút được mã hóa theo định dạng:
        [NÚT {id}] {text}
        [LẬP LUẬN ỦNG HỘ] {arg_support}
        [LẬP LUẬN PHẢN BÁC] {arg_refute}
        [PHỤ THUỘC VÀO NÚT {target_id}]  ← chỉ thị topo (nếu có cạnh từ nút này)

    Ví dụ đầu ra:
        [NÚT c0] Vaccine COVID-19 có hiệu quả 95%. [LẬP LUẬN ỦNG HỘ] Thử nghiệm lâm sàng...
        [LẬP LUẬN PHẢN BÁC] Một số nghiên cứu... [PHỤ THUỘC VÀO NÚT c1] [SEP_NODE]
        [NÚT c1] Thử nghiệm giai đoạn 3 hoàn thành. ...

    Các nút được phân cách bởi token đặc biệt [SEP_NODE].
    """
    nodes: list[dict] = record.get("nodes", [])
    edges: list[dict] = record.get("edges", [])

    # Xây dựng bản đồ cạnh đi ra: source_id → [target_id, ...]
    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        if src and tgt:
            outgoing.setdefault(src, []).append(tgt)

    # Sắp xếp: c0 trước, sau đó c1, c2, ... theo thứ tự số
    def _node_sort_key(node: dict) -> int:
        nid = node.get("id", "c9999")
        try:
            return int(nid[1:]) if len(nid) > 1 and nid[0] == "c" else 9999
        except ValueError:
            return 9999

    sorted_nodes = sorted(nodes, key=_node_sort_key)

    node_strings: list[str] = []
    for node in sorted_nodes:
        nid         = node.get("id", "?")
        text        = node.get("text", "").strip()
        arg_support = node.get("arg_support", NEI_MARKER).strip()
        arg_refute  = node.get("arg_refute",  NEI_MARKER).strip()

        # Rút gọn NEI_MARKER thành "NEI" để tiết kiệm token
        support_text = "NEI" if arg_support == NEI_MARKER else arg_support
        refute_text  = "NEI" if arg_refute  == NEI_MARKER else arg_refute

        parts = [
            f"[NÚT {nid}] {text}",
            f"[LẬP LUẬN ỦNG HỘ] {support_text}",
            f"[LẬP LUẬN PHẢN BÁC] {refute_text}",
        ]

        # Chỉ thị phụ thuộc topo
        for tgt in outgoing.get(nid, []):
            parts.append(f"[PHỤ THUỘC VÀO NÚT {tgt}]")

        node_strings.append(" ".join(parts))

    return " [SEP_NODE] ".join(node_strings)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_arguments() -> list[dict]:
    """Nạp cache/step3_arguments.json, trả về danh sách bản ghi."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INPUT_FILE}. "
            "Vui lòng chạy step3_competing_gen.py trước."
        )
    with INPUT_FILE.open(encoding="utf-8") as f:
        records: list[dict] = json.load(f)
    logger.info(f"Đọc {len(records):,} bản ghi từ {INPUT_FILE.name}")
    return records


def load_labels() -> dict[int, int]:
    """
    Nạp nhãn từ data/train.csv.
    Hỗ trợ tên cột: 'annotation_id'/'id' và 'label'/'Label'/'verdict'/'Verdict'.
    Trả về dict {annotation_id: label_id}.
    """
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy {TRAIN_CSV}.")

    df = pd.read_csv(TRAIN_CSV)

    # Chuẩn hoá cột ID
    if "annotation_id" in df.columns:
        id_col = "annotation_id"
    elif "id" in df.columns:
        id_col = "id"
    else:
        raise KeyError(
            "Không tìm thấy cột 'annotation_id' hoặc 'id' trong train.csv."
        )

    # Chuẩn hoá cột nhãn
    label_col: Optional[str] = None
    for candidate in ["label", "Label", "verdict", "Verdict"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise KeyError(
            "Không tìm thấy cột nhãn (label/verdict) trong train.csv. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    df = df[[id_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()

    # Numeric label fallback: "0"→0, "1"→1, "2"→2
    _numeric_label_map: dict[str, int] = {
        str(v): v for v in ID2LABEL.keys()
    }

    label_map: dict[int, int] = {}
    unknown: set[str] = set()
    for _, row in df.iterrows():
        label_str = row[label_col]
        lid = LABEL2ID.get(label_str, _numeric_label_map.get(label_str, -1))
        if lid == -1:
            unknown.add(label_str)
        else:
            label_map[int(row[id_col])] = lid

    if unknown:
        logger.warning(f"Bỏ qua nhãn không hợp lệ: {unknown}")

    # Phân phối nhãn
    from collections import Counter
    dist = Counter(label_map.values())
    logger.info(
        f"Đọc {len(label_map):,} nhãn từ {TRAIN_CSV.name} | "
        + " | ".join(f"{ID2LABEL[k]}={v}" for k, v in sorted(dist.items()))
    )
    return label_map


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class Graph2SeqDataset(Dataset):
    """
    Dataset chuyển đổi đồ thị lập luận → chuỗi token hoá cho PhoBERT.

    Mỗi mẫu trả về dict:
        input_ids      : LongTensor (max_length,)
        attention_mask : LongTensor (max_length,)
        label          : LongTensor scalar
    """

    def __init__(
        self,
        records:    list[dict],
        labels:     dict[int, int],
        tokenizer,
        max_length: int = ENCODER_MAX_LENGTH,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples: list[tuple[str, int]] = []

        skipped = 0
        for rec in records:
            rid   = int(rec["id"])
            label = labels.get(rid, -1)
            if label == -1:
                skipped += 1
                continue
            seq = graph_to_sequence(rec)
            self.samples.append((seq, label))

        if skipped:
            logger.warning(f"Bỏ qua {skipped:,} bản ghi không tìm thấy nhãn.")
        logger.info(f"Dataset: {len(self.samples):,} mẫu hợp lệ.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq, label = self.samples[idx]

        enc = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (max_length,)
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model: PhoBERT-large Encoder + MLP Classification Head
# ─────────────────────────────────────────────────────────────────────────────

class PhoBERTMLPClassifier(nn.Module):
    """
    Kiến trúc phân tích sâu kết hợp:

        PhoBERT-large (frozen, fine-tune lớp Transformer cuối)
              │
              ▼  last_hidden_state[:, 0, :]  →  h_cls ∈ ℝ^1024
              │
        ┌─────┴────────────────────────────────────────┐
        │            MLP Classification Head           │
        │  Linear(1024 → 512) → ReLU → Dropout(0.3)   │
        │  Linear(512  → 256) → ReLU → Dropout(0.3)   │
        │  Linear(256  → 3)                            │
        └──────────────────────────────────────────────┘
              │
              ▼  logits ∈ ℝ^3  (dùng CrossEntropyLoss)
    """

    def __init__(
        self,
        encoder_name:        str       = ENCODER_MODEL_NAME,
        hidden_dims:         list[int] = None,
        dropout:             float     = MLP_DROPOUT,
        num_labels:          int       = NUM_LABELS,
        freeze_encoder:      bool      = True,
        num_unfreeze_layers: int       = 1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = MLP_HIDDEN_DIMS

        # ── Encoder Backbone ──────────────────────────────────────────────────
        logger.info(f"Tải encoder: {encoder_name}")
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim  = self.encoder.config.hidden_size  # 1024 cho phobert-large

        # ── Freeze Strategy ───────────────────────────────────────────────────
        if freeze_encoder:
            # Bước 1: đóng băng toàn bộ encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Bước 2: mở băng `num_unfreeze_layers` lớp Transformer cuối cùng
            total_layers  = self.encoder.config.num_hidden_layers
            unfreeze_from = total_layers - num_unfreeze_layers
            for i, layer in enumerate(self.encoder.encoder.layer):
                if i >= unfreeze_from:
                    for param in layer.parameters():
                        param.requires_grad = True

            # Bước 3: mở băng pooler nếu có (giúp h_cls phong phú hơn)
            if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
                for param in self.encoder.pooler.parameters():
                    param.requires_grad = True

        # ── MLP Classification Head ───────────────────────────────────────────
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

        # Khởi tạo trọng số MLP với Xavier uniform
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # Log thống kê tham số
        n_total     = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Tổng tham số: {n_total:,} | "
            f"Trainable: {n_trainable:,} ({n_trainable / n_total * 100:.2f}%) | "
            f"Frozen lớp transformer: {unfreeze_from if freeze_encoder else 0}/{total_layers if freeze_encoder else 0}"
        )

    def forward(
        self,
        input_ids:      torch.Tensor,  # (batch, seq_len)
        attention_mask: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:
        """Trả về logits (batch_size, num_labels)."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Vector ngữ cảnh toàn cục từ token [CLS]: (batch, hidden_dim)
        h_cls  = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(h_cls)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Tính độ chính xác trên một batch."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model:      PhoBERTMLPClassifier,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    scaler:     Optional[torch.cuda.amp.GradScaler] = None,
) -> tuple[float, float]:
    """
    Một epoch huấn luyện với mixed-precision tùy chọn.
    Trả về (avg_loss, avg_accuracy).
    """
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc="  [Train]", leave=False, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                MLP_GRAD_CLIP,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                MLP_GRAD_CLIP,
            )
            optimizer.step()

        acc         = compute_accuracy(logits.detach(), labels)
        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.1f}%")

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(
    model:     PhoBERTMLPClassifier,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """
    Đánh giá trên tập validation.
    Trả về (avg_loss, avg_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc="  [Val]  ", leave=False, unit="batch", dynamic_ncols=True)
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)

        acc         = compute_accuracy(logits, labels)
        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc*100:.1f}%")

    return total_loss / n_batches, total_acc / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )

    logger.info("=" * 64)
    logger.info("  Bước 4: Graph2Seq + PhoBERT-large + MLP Classifier")
    logger.info("=" * 64)

    # ── Thiết bị & Mixed Precision ───────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB | AMP: {use_amp}")
    else:
        logger.warning("Không tìm thấy GPU. Huấn luyện trên CPU (sẽ rất chậm).")

    # ── Nạp dữ liệu ──────────────────────────────────────────────────────────
    records = load_arguments()
    labels  = load_labels()

    # ── Tokenizer ────────────────────────────────────────────────────────────
    logger.info(f"Tải tokenizer: {ENCODER_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME)

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    full_dataset = Graph2SeqDataset(
        records    = records,
        labels     = labels,
        tokenizer  = tokenizer,
        max_length = ENCODER_MAX_LENGTH,
    )

    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    logger.info(f"Tập Train: {n_train:,} mẫu | Tập Val: {n_val:,} mẫu")

    # num_workers=0 để tương thích Windows (spawn) và tránh lỗi tokenizer
    train_loader = DataLoader(
        train_ds,
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=MLP_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PhoBERTMLPClassifier(
        encoder_name        = ENCODER_MODEL_NAME,
        hidden_dims         = MLP_HIDDEN_DIMS,       # [512, 256] từ config
        dropout             = MLP_DROPOUT,           # 0.3 từ config
        num_labels          = NUM_LABELS,            # 3
        freeze_encoder      = True,
        num_unfreeze_layers = 1,                     # chỉ fine-tune lớp Transformer cuối
    ).to(device)

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=MLP_LEARNING_RATE,
        weight_decay=MLP_WEIGHT_DECAY,
        eps=1e-8,
    )

    # CosineAnnealingLR: giảm LR mượt mà theo chu kỳ cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MLP_EPOCHS,
        eta_min=MLP_LEARNING_RATE * 1e-2,
    )

    scaler: Optional[torch.cuda.amp.GradScaler] = (
        torch.cuda.amp.GradScaler() if use_amp else None
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    MLP_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = MLP_CHECKPOINT_DIR / "best_graph2seq_phobert.pt"

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    patience_cnt  = 0

    header = (
        f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>8} | {'Val Acc':>8} | {'LR':>10}"
    )
    sep = "-" * len(header)

    logger.info(f"\n{sep}")
    logger.info(header)
    logger.info(sep)

    for epoch in range(1, MLP_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Hiển thị kết quả epoch
        row = (
            f"{epoch:>6d} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | "
            f"{val_loss:>8.4f} | {val_acc*100:>7.2f}% | {current_lr:>10.2e}"
        )
        logger.info(row)

        # Lưu checkpoint nếu tốt hơn
        is_better = (
            val_acc > best_val_acc
            or (val_acc == best_val_acc and val_loss < best_val_loss)
        )
        if is_better:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "scheduler_state":  scheduler.state_dict(),
                    "val_acc":          best_val_acc,
                    "val_loss":         best_val_loss,
                    "encoder_model":    ENCODER_MODEL_NAME,
                    "label2id":         LABEL2ID,
                    "id2label":         ID2LABEL,
                },
                best_ckpt_path,
            )
            logger.success(
                f"  ★ Checkpoint mới: val_acc={best_val_acc*100:.2f}% "
                f"val_loss={best_val_loss:.4f} → {best_ckpt_path.name}"
            )
        else:
            patience_cnt += 1
            if patience_cnt >= MLP_EARLY_STOP_PATIENCE:
                logger.warning(
                    f"  Early stopping tại epoch {epoch} "
                    f"(patience={MLP_EARLY_STOP_PATIENCE})."
                )
                break

    logger.info(sep)
    logger.success(
        f"Huấn luyện hoàn thành! "
        f"Best Val Acc: {best_val_acc*100:.2f}% | "
        f"Best Val Loss: {best_val_loss:.4f}"
    )
    logger.info(f"Checkpoint tốt nhất: {best_ckpt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
