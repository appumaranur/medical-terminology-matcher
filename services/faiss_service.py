import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, MEDICAL_TERMS_CSV
from models.sentence_bert import SentenceBERTModel
from models.classifier import RelationshipClassifier

logger = logging.getLogger(__name__)

class FAISSService:
    """
    FAISS-based dense vector search service.
    Constructs an exact Inner Product (IndexFlatIP) index over L2-normalized Sentence-BERT embeddings.
    Indexes canonical medical terms, clinical synonyms, and semantic descriptions,
    mapping matches back to unique canonical terminology records.
    """
    _instance = None

    @classmethod
    def get_instance(cls) -> "FAISSService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.sbert = SentenceBERTModel.get_instance()
        self.dimension = self.sbert.get_dimension()
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.vector_to_record_idx: List[int] = []
        self._load_or_build()

    def _load_or_build(self) -> None:
        """Loads index and metadata from disk if present, else rebuilds from CSV."""
        meta_extra_path = str(FAISS_METADATA_PATH).replace(".json", "_extra.json")
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH) and os.path.exists(meta_extra_path):
            try:
                logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                with open(meta_extra_path, "r", encoding="utf-8") as f:
                    extra = json.load(f)
                    self.vector_to_record_idx = extra.get("vector_to_record_idx", [])
                logger.info(f"FAISS index loaded with {self.index.ntotal} vectors across {len(self.metadata)} unique terms.")
                return
            except Exception as e:
                logger.warning(f"Failed to read existing index: {e}. Rebuilding from scratch...")

        self.rebuild_index_from_csv()

    def rebuild_index_from_csv(self, csv_path: Optional[str] = None) -> int:
        """
        Reads terminology records from CSV, encodes full semantic representations
        (canonical terms, aliases/synonyms, and descriptive definitions),
        and builds a normalized FAISS IndexFlatIP.
        """
        import pandas as pd
        target_csv = csv_path or str(MEDICAL_TERMS_CSV)
        logger.info(f"Rebuilding FAISS index from: {target_csv}")

        if not os.path.exists(target_csv):
            logger.error(f"Medical terms CSV not found at {target_csv}")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.vector_to_record_idx = []
            return 0

        df = pd.read_csv(target_csv).fillna("")
        records = df.to_dict(orient="records")

        if not records:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            self.vector_to_record_idx = []
            return 0

        texts_to_embed: List[str] = []
        vector_to_record: List[int] = []

        for r_idx, r in enumerate(records):
            term = str(r.get("medical_term", "")).strip()
            desc = str(r.get("description", "")).strip()
            syn = str(r.get("synonyms", "")).strip()

            # 1. Canonical term representation
            texts_to_embed.append(term)
            vector_to_record.append(r_idx)

            # 2. Canonical with description
            if desc:
                texts_to_embed.append(f"{term}: {desc}")
                vector_to_record.append(r_idx)

            # 3. Individual synonyms
            if syn:
                for alias in syn.split(";"):
                    alias_clean = alias.strip()
                    if alias_clean and alias_clean.lower() != term.lower():
                        texts_to_embed.append(alias_clean)
                        vector_to_record.append(r_idx)

        logger.info(f"Encoding {len(texts_to_embed)} vector representations for {len(records)} records...")
        embeddings = self.sbert.encode(texts_to_embed, normalize_embeddings=True)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        new_index = faiss.IndexFlatIP(self.dimension)
        new_index.add(embeddings)

        self.index = new_index
        self.metadata = records
        self.vector_to_record_idx = vector_to_record

        # Persist index and metadata
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        meta_extra_path = str(FAISS_METADATA_PATH).replace(".json", "_extra.json")
        with open(meta_extra_path, "w", encoding="utf-8") as f:
            json.dump({"vector_to_record_idx": self.vector_to_record_idx}, f, indent=2)

        logger.info(f"FAISS index built successfully: {self.index.ntotal} vectors for {len(self.metadata)} terms.")
        return len(self.metadata)

    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search against the FAISS vector index.
        Deduplicates matches by canonical record and returns top_k ranked results.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        if self.index is None or self.index.ntotal == 0:
            self._load_or_build()
            if self.index is None or self.index.ntotal == 0:
                return []

        # Encode query into normalized vector
        query_vector = self.sbert.encode([clean_query], normalize_embeddings=True)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        # Retrieve top candidates
        k_to_fetch = min(self.index.ntotal, max(top_k * 5, 50))
        scores, indices = self.index.search(query_vector, k_to_fetch)

        results: List[Dict[str, Any]] = []
        seen_record_indices = set()

        raw_scores = scores[0]
        raw_indices = indices[0]

        for score, vec_idx in zip(raw_scores, raw_indices):
            if vec_idx < 0 or vec_idx >= len(self.vector_to_record_idx):
                continue

            record_idx = self.vector_to_record_idx[vec_idx]
            if record_idx in seen_record_indices:
                continue

            meta = self.metadata[record_idx]
            category = meta.get("category", "")

            if category_filter and category_filter.lower() != "all" and category.lower() != category_filter.lower():
                continue

            seen_record_indices.add(record_idx)
            sim_score = float(np.clip(score, -1.0, 1.0))
            relationship = RelationshipClassifier.classify(sim_score)
            badge_class = RelationshipClassifier.get_badge_class(relationship)

            results.append({
                "rank": len(results) + 1,
                "term_id": meta.get("term_id", f"ID_{record_idx}"),
                "medical_term": meta.get("medical_term", ""),
                "description": meta.get("description", ""),
                "category": category,
                "synonyms": meta.get("synonyms", ""),
                "similarity_score": round(sim_score, 4),
                "similarity_percentage": round(sim_score * 100, 1),
                "relationship": relationship,
                "badge_class": badge_class
            })

            if len(results) >= top_k:
                break

        return results

    def get_total_terms(self) -> int:
        """Returns the total number of unique indexed medical terms."""
        return len(self.metadata)
