import unittest
import os
from services.faiss_service import FAISSService
from config import FAISS_INDEX_PATH

class TestFAISSService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.faiss_service = FAISSService.get_instance()
        cls.faiss_service.rebuild_index_from_csv()

    def test_faiss_index_exists_and_populated(self):
        total = self.faiss_service.get_total_terms()
        self.assertGreater(total, 50)
        self.assertTrue(os.path.exists(FAISS_INDEX_PATH))

    def test_faiss_semantic_search_retrieval(self):
        results = self.faiss_service.search("Heart Attack", top_k=5)
        self.assertGreaterEqual(len(results), 1)
        top = results[0]
        self.assertEqual(top["rank"], 1)
        self.assertIn("medical_term", top)
        self.assertIn("similarity_score", top)
        self.assertIn("relationship", top)
        # Should match Myocardial Infarction or Cardiology
        self.assertTrue(
            "Myocardial" in top["medical_term"] or "Cardio" in top["category"] or top["similarity_score"] > 0.5
        )

    def test_faiss_category_filter(self):
        results = self.faiss_service.search("Sugar", top_k=5, category_filter="Endocrinology")
        for r in results:
            self.assertEqual(r["category"], "Endocrinology")

    def test_faiss_empty_query(self):
        results = self.faiss_service.search("", top_k=5)
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()

