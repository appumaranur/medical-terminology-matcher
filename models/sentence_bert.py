"""
Sentence-BERT embedding model wrapper.

Loads the pre-trained 'all-MiniLM-L6-v2' model directly from the local
HuggingFace cache snapshot directory to avoid any network calls at runtime.
"""

import os
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import SBERT_MODEL_NAME, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

# Resolve the local snapshot path for all-MiniLM-L6-v2.
# Loading from the snapshot directory directly suppresses all HEAD / network retries.
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_SBERT_SNAP_DIR = (
    _HF_CACHE
    / "models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots"
)


def _find_local_snapshot() -> str:
    """
    Returns the absolute path to the latest SBERT snapshot directory,
    or falls back to the model name (for online loading if not cached).
    """
    if _SBERT_SNAP_DIR.exists():
        snaps = sorted(_SBERT_SNAP_DIR.iterdir())
        if snaps:
            snap_path = str(snaps[-1])
            logger.info(f"SBERT local snapshot found: {snap_path}")
            return snap_path
    logger.warning("SBERT local snapshot not found; will attempt online load.")
    return SBERT_MODEL_NAME


class SentenceBERTModel:
    """
    Sentence-BERT wrapper for generating dense sentence-level semantic representations.

    Utilizes pre-trained 'all-MiniLM-L6-v2' with optimized mean-pooling and projection
    trained explicitly for semantic textual similarity (STS) tasks.

    Embedding dimension: 384
    Distance metric:     Cosine similarity (equivalent to dot product on L2-normalized vectors)
    """

    _instance = None
    _model: SentenceTransformer = None

    @classmethod
    def get_instance(cls) -> "SentenceBERTModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, model_name: str = SBERT_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self) -> None:
        if SentenceBERTModel._model is not None:
            self.model = SentenceBERTModel._model
            return

        logger.info(f"Loading Sentence-BERT model on device: {self.device}")

        # Primary: load directly from local snapshot directory (no network calls)
        local_path = _find_local_snapshot()
        try:
            SentenceBERTModel._model = SentenceTransformer(local_path, device=self.device)
            logger.info("Sentence-BERT loaded successfully from local cache.")
        except Exception as e_local:
            logger.warning(f"Local load failed ({e_local}); attempting online load...")
            try:
                SentenceBERTModel._model = SentenceTransformer(
                    self.model_name, device=self.device
                )
                logger.info("Sentence-BERT loaded from HuggingFace Hub.")
            except Exception as e_online:
                logger.error(f"Failed to load Sentence-BERT: {e_online}")
                raise RuntimeError(
                    f"Cannot load Sentence-BERT model '{self.model_name}'. "
                    "Ensure the model is cached locally or internet access is available."
                ) from e_online

        self.model = SentenceBERTModel._model

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Encodes a single text string or list of texts into L2-normalized dense embedding vectors.

        Args:
            texts:               Input string or list of strings to encode.
            batch_size:          Batch size for GPU/CPU memory management.
            normalize_embeddings: If True, returns unit-norm vectors (required for cosine similarity).

        Returns:
            2D numpy array of shape (N, 384).
        """
        is_single = isinstance(texts, str)
        input_list = [texts] if is_single else list(texts)

        if not input_list:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        embeddings = self.model.encode(
            input_list,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )

        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        """Returns the embedding dimension (384 for all-MiniLM-L6-v2)."""
        try:
            return self.model.get_embedding_dimension()
        except AttributeError:
            return self.model.get_sentence_embedding_dimension()

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Computes cosine similarity between two medical texts.

        Formula: cos(u, v) = dot(u_norm, v_norm)  [for L2-normalized vectors]

        Returns:
            float in [-1.0, 1.0]; values near 1.0 indicate high semantic similarity.
        """
        embs = self.encode([text_a, text_b], normalize_embeddings=True)
        dot_product = float(np.dot(embs[0], embs[1]))
        # Clamp to [-1.0, 1.0] to guard against tiny floating-point imprecision
        return float(np.clip(dot_product, -1.0, 1.0))
