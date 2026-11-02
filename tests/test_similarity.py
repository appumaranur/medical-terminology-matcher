import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from models.sentence_bert import SentenceBERTModel
from models.biobert import BioBERTModel
from models.traditional_similarity import (
    compute_levenshtein_distance,
    compute_levenshtein_similarity,
    compute_jaccard_similarity,
    compute_tfidf_similarity,
    TraditionalSimilarityService
)
from models.classifier import RelationshipClassifier
from services.similarity_service import SimilarityService


class TestMedicalSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sbert = SentenceBERTModel.get_instance()
        cls.biobert = BioBERTModel.get_instance()
        cls.similarity_service = SimilarityService()

    # ── Sentence-BERT ─────────────────────────────────────────────────────────

    def test_sbert_embedding_shape_and_norm(self):
        """SBERT must return shape (1, 384) with L2 norm = 1.0."""
        emb = self.sbert.encode("Myocardial Infarction", normalize_embeddings=True)
        self.assertEqual(emb.shape, (1, 384))
        norm = np.linalg.norm(emb[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_sbert_known_similar_pair(self):
        """Heart Attack vs Myocardial Infarction must score >= 0.55 (Equivalent tier)."""
        score = self.sbert.compute_similarity("Heart Attack", "Myocardial Infarction")
        self.assertGreaterEqual(score, 0.55)

    def test_sbert_known_unrelated_pair(self):
        """Appendicitis vs Quantum Physics must score near 0 or negative."""
        score = self.sbert.compute_similarity("Appendicitis", "Quantum Physics")
        self.assertLess(score, 0.20)

    # ── BioBERT ───────────────────────────────────────────────────────────────

    def test_biobert_embedding_shape_and_norm(self):
        """BioBERT must return shape (1, 768) with L2 norm = 1.0 (fallback or real)."""
        emb = self.biobert.encode("Myocardial Infarction", normalize_embeddings=True)
        self.assertEqual(emb.shape, (1, 768))
        norm = np.linalg.norm(emb[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

    # ── Traditional similarity ────────────────────────────────────────────────

    def test_traditional_similarity_metrics(self):
        """Traditional metrics must return valid ranges; Heart Attack vs Myocardial Infarction has zero word overlap."""
        s1, s2 = "Heart Attack", "Myocardial Infarction"
        lev_dist = compute_levenshtein_distance(s1, s2)
        lev_sim  = compute_levenshtein_similarity(s1, s2)
        jaccard  = compute_jaccard_similarity(s1, s2)
        tfidf    = compute_tfidf_similarity(s1, s2)

        self.assertGreater(lev_dist, 0)
        self.assertTrue(0.0 <= lev_sim <= 1.0)
        self.assertEqual(jaccard, 0.0)      # zero word overlap
        self.assertTrue(0.0 <= tfidf <= 1.0)

    def test_traditional_identical_strings(self):
        """Identical strings must give edit_distance=0, all similarities=1.0."""
        s = "Hypertension"
        self.assertEqual(compute_levenshtein_distance(s, s), 0)
        self.assertEqual(compute_levenshtein_similarity(s, s), 1.0)
        self.assertEqual(compute_jaccard_similarity(s, s), 1.0)
        self.assertAlmostEqual(compute_tfidf_similarity(s, s), 1.0, places=3)

    # ── Classifier ────────────────────────────────────────────────────────────

    def test_relationship_classifier_thresholds(self):
        """
        Verify all four relationship tiers fire at the correct score ranges.
        Recalibrated thresholds: Equivalent >= 0.55 | Related >= 0.40 | Moderate >= 0.20 | Unrelated < 0.20
        """
        self.assertIn("Equivalent", RelationshipClassifier.classify(0.85))   # well above 0.55
        self.assertIn("Equivalent", RelationshipClassifier.classify(0.60))   # just above 0.55 → Equivalent
        self.assertIn("Related",    RelationshipClassifier.classify(0.45))   # between 0.40–0.55 → Related
        self.assertIn("Moderate",   RelationshipClassifier.classify(0.25))   # between 0.20–0.40 → Moderately Similar
        self.assertIn("Unrelated",  RelationshipClassifier.classify(0.05))   # below 0.20 → Unrelated

    def test_classifier_returns_all_fields(self):
        """generate_explanation must return all required JSON keys."""
        result = RelationshipClassifier.generate_explanation(
            term_a="Heart Attack",
            term_b="Myocardial Infarction",
            score=0.60,
            sbert_score=0.60,
            biobert_score=0.58,
        )
        for key in ("relationship", "badge_class", "score", "clinical_summary", "clinical_caution", "nlp_insight"):
            self.assertIn(key, result)

    # ── End-to-end comparison pipeline ───────────────────────────────────────

    def test_end_to_end_pair_comparison(self):
        """Full SimilarityService.compare() must return all expected fields."""
        res = self.similarity_service.compare("Heart Attack", "Myocardial Infarction", log_to_db=False)
        for field in ("similarity_score", "sbert_score", "biobert_score", "consensus_score",
                      "relationship", "clinical_summary", "clinical_caution", "traditional_metrics",
                      "biobert_fallback"):
            self.assertIn(field, res)

    def test_end_to_end_score_range(self):
        """All similarity scores must lie in [-1.0, 1.0]."""
        res = self.similarity_service.compare("Hypertension", "High Blood Pressure", log_to_db=False)
        self.assertGreaterEqual(res["sbert_score"], -1.0)
        self.assertLessEqual(res["sbert_score"], 1.0)
        if res["biobert_score"] is not None:
            self.assertGreaterEqual(res["biobert_score"], -1.0)
            self.assertLessEqual(res["biobert_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
