"""
BioBERT embedding model wrapper.

Architecture & Pooling Strategy:
---------------------------------
BioBERT (bidirectional encoder representations from transformers, pre-trained on
PubMed abstracts and PMC full-text articles) is a token-level masked language model.
Unlike Sentence-BERT, it does NOT have a sentence-level contrastive training head,
so it outputs token-level hidden states of shape (batch, seq_len, 768).

To produce a unified sentence embedding we apply:
    Attention-Mask-Aware Mean Pooling:
        e = Σ(token_emb_i * mask_i) / Σ(mask_i)   [ignores padding tokens]
followed by L2 unit-vector normalization so cosine similarity = dot product.

This strategy is established in the NLP literature (Reimers & Gurevych 2019)
and acknowledged as the standard approach for BioBERT sentence representations
when fine-tuning for STS is not available.

Fallback behavior:
------------------
If BioBERT weights are not available locally (never downloaded or download
incomplete), the module gracefully falls back to Sentence-BERT embeddings
padded to 768 dimensions to maintain the BioBERT dimension contract.
The `is_fallback` attribute is set True in this case, allowing callers to
display a UI warning to the user.
"""

import logging
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from config import BIOBERT_MODEL_NAME, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

# ── Local cache path helpers ──────────────────────────────────────────────────
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_BIOBERT_SNAP_DIR = (
    _HF_CACHE
    / "models--monologg--biobert_v1.1_pubmed"
    / "snapshots"
)


def _find_biobert_local_snapshot() -> Optional[str]:
    """
    Returns the absolute path to the BioBERT snapshot directory if the weights
    are fully downloaded (model.safetensors or pytorch_model.bin must be > 1 MB),
    otherwise returns None.
    """
    if not _BIOBERT_SNAP_DIR.exists():
        return None

    for snap in sorted(_BIOBERT_SNAP_DIR.iterdir(), reverse=True):
        weight_files = list(snap.glob("*.safetensors")) + list(snap.glob("*.bin"))
        # Any weight file larger than 100 MB means it's a real model, not an empty stub
        if any(f.stat().st_size > 100 * 1024 * 1024 for f in weight_files):
            logger.info(f"BioBERT local snapshot found: {snap}")
            return str(snap)

    logger.info("BioBERT snapshot directory exists but no complete weight file found.")
    return None


class BioBERTModel:
    """
    BioBERT (Biomedical BERT) dense embedding extractor with attention-mask-aware mean pooling.

    Attributes:
        is_fallback (bool): True when the model falls back to SBERT due to missing weights.
        model_name  (str):  The HuggingFace model identifier being used.
    """

    _instance: Optional["BioBERTModel"] = None
    _tokenizer = None
    _model = None
    _is_fallback: bool = False

    @classmethod
    def get_instance(cls) -> "BioBERTModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, model_name: str = BIOBERT_MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if BioBERTModel._model is not None or BioBERTModel._is_fallback:
            # Already loaded (or already confirmed unavailable) by a previous instance
            self.tokenizer  = BioBERTModel._tokenizer
            self.model      = BioBERTModel._model
            self.is_fallback = BioBERTModel._is_fallback
            return

        logger.info(f"Loading BioBERT model: {self.model_name} on {self.device}")

        # Try 1: Load directly from verified local snapshot (no network)
        local_path = _find_biobert_local_snapshot()
        if local_path:
            try:
                BioBERTModel._tokenizer = AutoTokenizer.from_pretrained(
                    local_path, local_files_only=True
                )
                BioBERTModel._model = AutoModel.from_pretrained(
                    local_path, local_files_only=True
                )
                BioBERTModel._is_fallback = False
                logger.info("BioBERT loaded successfully from local snapshot.")
            except Exception as e_snap:
                logger.warning(f"Failed to load BioBERT from snapshot ({e_snap}).")
                BioBERTModel._model = None
                BioBERTModel._tokenizer = None

        # Try 2: Load from HuggingFace cache by model name (online or cached)
        if BioBERTModel._model is None:
            try:
                BioBERTModel._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                BioBERTModel._model = AutoModel.from_pretrained(
                    self.model_name, local_files_only=True
                )
                BioBERTModel._is_fallback = False
                logger.info("BioBERT loaded from HuggingFace local cache by model name.")
            except Exception as e_cache:
                logger.warning(
                    f"BioBERT not in local cache ({e_cache}). "
                    "Run 'python download_models.py' to download it. "
                    "Using Sentence-BERT as fallback for BioBERT embeddings."
                )
                BioBERTModel._is_fallback = True

        # Move to device if fully loaded
        if BioBERTModel._model is not None:
            BioBERTModel._model.to(self.device)
            BioBERTModel._model.eval()
            logger.info(
                f"BioBERT ready  (hidden_size={BioBERTModel._model.config.hidden_size}, "
                f"device={self.device})"
            )

        self.tokenizer   = BioBERTModel._tokenizer
        self.model       = BioBERTModel._model
        self.is_fallback = BioBERTModel._is_fallback

    # ── Pooling ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Attention-mask-aware mean pooling across the token sequence.

        Formula:
            e = Σ(token_emb_i * mask_i) / clamp(Σ(mask_i), min=1e-9)

        Ensures that padding tokens (mask=0) do not bias the sentence embedding.
        model_output[0]: last hidden states of shape (batch, seq_len, hidden_dim)
        """
        token_embeddings = model_output[0]  # (B, L, H)
        # Expand mask to match hidden dimension
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask       = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask  # (B, H)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Generates attention-mask-aware mean-pooled embeddings for medical texts.

        Args:
            texts:                Input string or list of strings.
            batch_size:           Processing batch size (memory management).
            normalize_embeddings: If True, returns L2-normalized unit vectors.

        Returns:
            numpy array of shape (N, 768) — the BioBERT hidden dimension.
        """
        is_single = isinstance(texts, str)
        input_list = [texts] if is_single else list(texts)

        if not input_list:
            return np.empty((0, 768), dtype=np.float32)

        # ── Fallback path ────────────────────────────────────────────────────
        if self.is_fallback or self.model is None:
            from models.sentence_bert import SentenceBERTModel
            sbert = SentenceBERTModel.get_instance()
            raw_embs = sbert.encode(input_list, normalize_embeddings=False)
            # Pad/truncate SBERT 384-dim → 768-dim to maintain BioBERT dimension contract
            padded = np.zeros((len(input_list), 768), dtype=np.float32)
            dim = min(raw_embs.shape[1], 768)
            padded[:, :dim] = raw_embs[:, :dim]
            if normalize_embeddings:
                norms = np.linalg.norm(padded, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                padded = padded / norms
            return padded

        # ── Real BioBERT path ────────────────────────────────────────────────
        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(input_list), batch_size):
                batch_texts = input_list[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                input_ids      = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                model_output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                pooled = self._mean_pooling(model_output, attention_mask)

                if normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)

                all_embeddings.append(pooled.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings.astype(np.float32)

    # ── Similarity ────────────────────────────────────────────────────────────

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Computes cosine similarity between two medical phrases using BioBERT embeddings.

        For L2-normalized vectors: cos(u, v) = dot(u, v)

        Returns:
            float in [-1.0, 1.0]; values near 1.0 indicate high semantic similarity.
        """
        embs = self.encode([text_a, text_b], normalize_embeddings=True)
        dot_product = float(np.dot(embs[0], embs[1]))
        return float(np.clip(dot_product, -1.0, 1.0))
