import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import json
from app import app
from database.db import init_db


class TestMedicalAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        with app.app_context():
            init_db()

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def test_dashboard_endpoint(self):
        res = self.client.get("/api/dashboard")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("total_terms", data["data"])
        self.assertIn("category_distribution", data["data"])
        self.assertGreater(data["data"]["total_terms"], 0)

    # ── Model status ──────────────────────────────────────────────────────────

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("sbert_available", data["data"])
        self.assertIn("biobert_available", data["data"])
        self.assertTrue(data["data"]["sbert_available"])

    # ── Compare ───────────────────────────────────────────────────────────────

    def test_compare_endpoint_success(self):
        payload = {"term_a": "Heart Attack", "term_b": "Myocardial Infarction"}
        res = self.client.post(
            "/api/compare",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("similarity_score", data["data"])
        self.assertIn("relationship",     data["data"])
        self.assertIn("clinical_summary", data["data"])
        self.assertIn("sbert_score",      data["data"])
        self.assertIn("biobert_score",    data["data"])
        self.assertIn("biobert_fallback", data["data"])
        # Heart Attack vs Myocardial Infarction must be Equivalent (score >= 0.55)
        self.assertGreaterEqual(data["data"]["similarity_score"], 0.55)
        self.assertIn("Equivalent", data["data"]["relationship"])

    def test_compare_endpoint_missing_params(self):
        payload = {"term_a": "Heart Attack"}    # term_b missing
        res = self.client.post(
            "/api/compare",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 400)

    def test_compare_endpoint_empty_strings(self):
        payload = {"term_a": "", "term_b": "Hypertension"}
        res = self.client.post(
            "/api/compare",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 400)

    # ── FAISS Search ──────────────────────────────────────────────────────────

    def test_search_endpoint(self):
        payload = {"query": "Brain Attack", "top_k": 5, "category": "all"}
        res = self.client.post(
            "/api/search",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["results"]), 0)
        # Each result must have required fields
        first = data["results"][0]
        for field in ("rank", "medical_term", "similarity_score", "relationship"):
            self.assertIn(field, first)

    def test_search_endpoint_empty_query(self):
        payload = {"query": "", "top_k": 5}
        res = self.client.post(
            "/api/search",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 400)

    # ── History ───────────────────────────────────────────────────────────────

    def test_history_and_delete_endpoints(self):
        # 1. Fetch history
        res = self.client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        hist = res.get_json()
        self.assertEqual(hist["status"], "success")

        # 2. If records exist, delete the first one and verify 200
        if hist["count"] > 0:
            first_id = hist["data"][0]["id"]
            del_res = self.client.delete(f"/api/history/{first_id}")
            self.assertEqual(del_res.status_code, 200)

    # ── Model Comparison ──────────────────────────────────────────────────────

    def test_model_comparison_endpoint(self):
        payload = {"term_a": "Hypertension", "term_b": "High Blood Pressure"}
        res = self.client.post(
            "/api/model-comparison",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("comparison_table", data)
        self.assertEqual(len(data["comparison_table"]), 5)  # 3 traditional + 2 transformer
        # All scores must be in [0, 1]
        for row in data["comparison_table"]:
            self.assertGreaterEqual(row["score"], -1.0)
            self.assertLessEqual(row["score"], 1.0)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def test_evaluation_endpoint(self):
        res = self.client.get("/api/evaluation?model=Sentence-BERT&split=test&threshold=0.55")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        for field in ("accuracy", "f1_score", "roc_auc", "confusion_matrix",
                      "precision", "recall", "precision_at_k", "recall_at_k"):
            self.assertIn(field, data["data"])
        # Accuracy must be in valid range
        self.assertGreaterEqual(data["data"]["accuracy"], 0.0)
        self.assertLessEqual(data["data"]["accuracy"], 1.0)

    # ── Dataset ───────────────────────────────────────────────────────────────

    def test_dataset_terms_endpoint(self):
        res = self.client.get("/api/dataset/terms?page=1&per_page=10")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("terms", data["data"])
        self.assertGreater(data["data"]["total"], 0)
        # Dataset must have >= 100 terms
        self.assertGreaterEqual(data["data"]["total"], 100)


if __name__ == "__main__":
    unittest.main()
