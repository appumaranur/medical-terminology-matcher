import logging
from typing import Dict, Any, Optional
from models.sentence_bert import SentenceBERTModel
from models.biobert import BioBERTModel
from models.traditional_similarity import TraditionalSimilarityService
from models.classifier import RelationshipClassifier
from database.db import log_comparison

logger = logging.getLogger(__name__)

class SimilarityService:
    """
    Core service coordinating semantic similarity calculations, traditional algorithms,
    clinical relationship classification, and database audit logging.
    """
    def __init__(self):
        self.sbert = SentenceBERTModel.get_instance()
        self._biobert = None
        self.traditional = TraditionalSimilarityService()

    @property
    def biobert(self) -> BioBERTModel:
        if self._biobert is None:
            self._biobert = BioBERTModel.get_instance()
        return self._biobert

    def compare(
        self,
        term_a: str,
        term_b: str,
        log_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end multi-model comparison between two medical terms.
        Calculates:
        1. Sentence-BERT cosine similarity
        2. BioBERT cosine similarity (with attention-mask-aware mean pooling)
        3. Traditional lexical measures (TF-IDF, Jaccard, Levenshtein)
        4. Relationship classification & clinical safety explanation
        5. SQLite audit record insertion
        """
        term_a_clean = term_a.strip()
        term_b_clean = term_b.strip()

        if not term_a_clean or not term_b_clean:
            raise ValueError("Both medical terms must be non-empty strings.")

        # 1. Sentence-BERT similarity
        sbert_score = self.sbert.compute_similarity(term_a_clean, term_b_clean)

        # 2. BioBERT similarity
        biobert_score = None
        biobert_error = None
        try:
            biobert_score = self.biobert.compute_similarity(term_a_clean, term_b_clean)
        except Exception as e:
            logger.warning(f"BioBERT similarity computation failed or deferred: {e}")
            biobert_error = str(e)
            # Graceful fallback: use Sentence-BERT as proxy if BioBERT weights not yet cached
            biobert_score = sbert_score

        # 3. Traditional similarity measures
        traditional_results = self.traditional.compare_terms(term_a_clean, term_b_clean)

        # 4. Consensus / Primary score calculation
        # When both models are available, we report both and use primary SBERT/BioBERT consensus
        primary_score = sbert_score
        consensus_score = (sbert_score + biobert_score) / 2.0 if biobert_score is not None else sbert_score

        # 5. Relationship classification & clinical explanation
        explanation_data = RelationshipClassifier.generate_explanation(
            term_a=term_a_clean,
            term_b=term_b_clean,
            score=primary_score,
            sbert_score=sbert_score,
            biobert_score=biobert_score,
            lexical_overlap=traditional_results["lexical_overlap_detected"]
        )

        # 6. Database Logging
        history_id = None
        if log_to_db:
            try:
                history_id = log_comparison(
                    term_a=term_a_clean,
                    term_b=term_b_clean,
                    similarity_score=primary_score,
                    model_name="Sentence-BERT + BioBERT",
                    relationship=explanation_data["relationship"],
                    explanation=explanation_data["clinical_summary"]
                )
            except Exception as db_err:
                logger.error(f"Failed to log comparison to database: {db_err}")

        return {
            "term_a": term_a_clean,
            "term_b": term_b_clean,
            "similarity_score": round(primary_score, 4),
            "percentage": round(primary_score * 100, 1),
            "sbert_score": round(sbert_score, 4),
            "biobert_score": round(biobert_score, 4) if biobert_score is not None else None,
            "biobert_error": biobert_error,
            "biobert_fallback": self.biobert.is_fallback,
            "consensus_score": round(consensus_score, 4),
            "relationship": explanation_data["relationship"],
            "badge_class": explanation_data["badge_class"],
            "clinical_summary": explanation_data["clinical_summary"],
            "clinical_caution": explanation_data["clinical_caution"],
            "nlp_insight": explanation_data["nlp_insight"],
            "traditional_metrics": traditional_results,
            "history_id": history_id
        }

