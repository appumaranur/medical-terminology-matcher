import logging
from typing import Dict, List, Union, Optional
import numpy as np
from models.sentence_bert import SentenceBERTModel
from models.biobert import BioBERTModel

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Centralized embedding service managing models, batch generation, and memory caching.
    """
    _sbert_cache: Dict[str, np.ndarray] = {}
    _biobert_cache: Dict[str, np.ndarray] = {}

    def __init__(self):
        self.sbert = SentenceBERTModel.get_instance()
        self._biobert: Optional[BioBERTModel] = None

    @property
    def biobert(self) -> BioBERTModel:
        if self._biobert is None:
            self._biobert = BioBERTModel.get_instance()
        return self._biobert

    def get_sbert_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Retrieves or computes a 1D normalized Sentence-BERT embedding vector for text."""
        clean_text = text.strip()
        if use_cache and clean_text in self._sbert_cache:
            return self._sbert_cache[clean_text]

        emb = self.sbert.encode(clean_text, normalize_embeddings=True)[0]
        if use_cache:
            self._sbert_cache[clean_text] = emb
        return emb

    def get_biobert_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Retrieves or computes a 1D normalized BioBERT embedding vector for text."""
        clean_text = text.strip()
        if use_cache and clean_text in self._biobert_cache:
            return self._biobert_cache[clean_text]

        emb = self.biobert.encode(clean_text, normalize_embeddings=True)[0]
        if use_cache:
            self._biobert_cache[clean_text] = emb
        return emb

    def get_batch_sbert_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Computes embeddings for a list of texts using caching where available."""
        results = []
        uncached_indices = []
        uncached_texts = []

        for idx, text in enumerate(texts):
            clean_text = text.strip()
            if use_cache and clean_text in self._sbert_cache:
                results.append((idx, self._sbert_cache[clean_text]))
            else:
                uncached_indices.append(idx)
                uncached_texts.append(clean_text)

        if uncached_texts:
            new_embs = self.sbert.encode(uncached_texts, normalize_embeddings=True)
            for orig_idx, clean_text, emb in zip(uncached_indices, uncached_texts, new_embs):
                if use_cache:
                    self._sbert_cache[clean_text] = emb
                results.append((orig_idx, emb))

        # Sort back into original order
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results], dtype=np.float32)

    def clear_cache(self) -> None:
        """Clears memory caches."""
        self._sbert_cache.clear()
        self._biobert_cache.clear()

