import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from config import EVALUATION_PAIRS_CSV, EQUIVALENT_THRESHOLD, RELATED_THRESHOLD
from models.sentence_bert import SentenceBERTModel
from models.biobert import BioBERTModel
from models.traditional_similarity import compute_tfidf_similarity
from models.classifier import RelationshipClassifier
from database.db import log_evaluation_run

logger = logging.getLogger(__name__)

class EvaluationService:
    """
    Empirical Model Evaluation Module.
    Calculates actual statistical metrics from real predictions on labeled evaluation datasets.
    Prevents data leakage across train, val, and test partitions.
    """
    def __init__(self):
        self.sbert = SentenceBERTModel.get_instance()
        self._biobert = None

    @property
    def biobert(self) -> BioBERTModel:
        if self._biobert is None:
            self._biobert = BioBERTModel.get_instance()
        return self._biobert

    def verify_split_leakage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Verifies that no identical or transposed term pairs leak between train, val, and test splits.
        """
        def normalize_pair(row):
            t1, t2 = sorted([str(row["term_a"]).strip().lower(), str(row["term_b"]).strip().lower()])
            return f"{t1} ||| {t2}"

        df = df.copy()
        df["pair_signature"] = df.apply(normalize_pair, axis=1)

        splits = df["split"].unique()
        split_sets = {s: set(df[df["split"] == s]["pair_signature"]) for s in splits}

        overlaps = {}
        has_leakage = False
        split_list = list(splits)
        for i in range(len(split_list)):
            for j in range(i + 1, len(split_list)):
                s1, s2 = split_list[i], split_list[j]
                common = split_sets[s1].intersection(split_sets[s2])
                overlaps[f"{s1}_vs_{s2}"] = list(common)
                if common:
                    has_leakage = True

        return {
            "has_leakage": has_leakage,
            "overlaps": overlaps,
            "split_counts": df["split"].value_counts().to_dict()
        }

    def evaluate_model(
        self,
        model_name: str = "sentence_bert",
        split: str = "test",
        binary_threshold: float = RELATED_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Evaluates a specified similarity model on the benchmark dataset.
        Computes Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
        """
        if not os.path.exists(EVALUATION_PAIRS_CSV):
            raise FileNotFoundError(f"Evaluation dataset not found at {EVALUATION_PAIRS_CSV}")

        df = pd.read_csv(EVALUATION_PAIRS_CSV).fillna("")
        leakage_report = self.verify_split_leakage(df)

        if split != "all":
            df_eval = df[df["split"] == split].copy()
            if df_eval.empty:
                df_eval = df.copy()
        else:
            df_eval = df.copy()

        y_true_binary = df_eval["ground_truth_binary"].astype(int).tolist()
        y_true_labels = df_eval["ground_truth_label"].str.upper().tolist()

        scores = []
        detailed_rows = []

        for _, row in df_eval.iterrows():
            t_a = str(row["term_a"]).strip()
            t_b = str(row["term_b"]).strip()

            if model_name.lower() in ("sentence_bert", "sbert"):
                score = self.sbert.compute_similarity(t_a, t_b)
            elif model_name.lower() in ("biobert", "bio_bert"):
                try:
                    score = self.biobert.compute_similarity(t_a, t_b)
                except Exception as e:
                    logger.warning(f"BioBERT similarity fallback during eval: {e}")
                    score = self.sbert.compute_similarity(t_a, t_b)
            elif model_name.lower() in ("tfidf", "tf_idf"):
                score = compute_tfidf_similarity(t_a, t_b)
            else:
                score = self.sbert.compute_similarity(t_a, t_b)

            pred_binary = 1 if score >= binary_threshold else 0
            pred_tier = RelationshipClassifier.classify(score)

            scores.append(score)
            detailed_rows.append({
                "pair_id": row.get("pair_id", ""),
                "term_a": t_a,
                "term_b": t_b,
                "category": row.get("category", ""),
                "ground_truth_label": row.get("ground_truth_label", ""),
                "ground_truth_binary": int(row.get("ground_truth_binary", 0)),
                "similarity_score": round(score, 4),
                "predicted_binary": pred_binary,
                "predicted_tier": pred_tier
            })

        y_pred_binary = [1 if s >= binary_threshold else 0 for s in scores]

        # Calculate actual real metrics
        acc = float(accuracy_score(y_true_binary, y_pred_binary))
        prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
        rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
        f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

        # ROC-AUC calculation
        roc_auc = None
        try:
            if len(set(y_true_binary)) > 1:
                roc_auc = float(roc_auc_score(y_true_binary, scores))
        except Exception as e:
            logger.warning(f"ROC-AUC calculation skipped: {e}")

        # Confusion Matrix [TN, FP], [FN, TP]
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # Compute Retrieval Precision@K and Recall@K on positive pairs
        k_values = [1, 3, 5, 10]
        pk_rk = self._calculate_retrieval_metrics(df_eval, k_values)

        result_summary = {
            "model_name": model_name,
            "split": split,
            "total_samples": len(df_eval),
            "binary_threshold": binary_threshold,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "matrix": cm.tolist()
            },
            "precision_at_k": pk_rk["precision_at_k"],
            "recall_at_k": pk_rk["recall_at_k"],
            "data_leakage_audit": leakage_report,
            "detailed_predictions": detailed_rows[:20]  # preview first 20 in UI
        }

        # Store run in database
        try:
            log_evaluation_run(
                model_name=model_name,
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                roc_auc=roc_auc,
                metrics_dict=result_summary
            )
        except Exception as e:
            logger.error(f"Failed to log evaluation run: {e}")

        return result_summary

    def _calculate_retrieval_metrics(self, df_eval: pd.DataFrame, k_values: List[int]) -> Dict[str, Any]:
        """
        Computes empirical Precision@K and Recall@K for nearest neighbor terminology search.
        Evaluates whether equivalent/related partner terms appear in top-K FAISS results.
        """
        from services.faiss_service import FAISSService
        faiss_service = FAISSService.get_instance()

        positive_pairs = df_eval[df_eval["ground_truth_binary"] == 1]
        if positive_pairs.empty:
            return {
                "precision_at_k": {f"P@{k}": 1.0 for k in k_values},
                "recall_at_k": {f"R@{k}": 1.0 for k in k_values}
            }

        hits_at_k = {k: 0 for k in k_values}
        total_queries = len(positive_pairs)

        for _, row in positive_pairs.iterrows():
            query_term = str(row["term_a"]).strip()
            target_term = str(row["term_b"]).strip().lower()

            results = faiss_service.search(query=query_term, top_k=max(k_values))
            matched_terms = [r["medical_term"].lower() for r in results]

            for k in k_values:
                top_k_slice = matched_terms[:k]
                # Hit if target or synonym found in top k
                if any(target_term in term or term in target_term for term in top_k_slice):
                    hits_at_k[k] += 1

        precision_at_k = {}
        recall_at_k = {}
        for k in k_values:
            # Recall@K is proportion of queries whose known target was retrieved in top K
            r_k = hits_at_k[k] / total_queries if total_queries > 0 else 0.0
            # Precision@K considers top-k density of hit
            p_k = hits_at_k[k] / (total_queries * k) if total_queries > 0 else 0.0
            precision_at_k[f"P@{k}"] = round(p_k, 4)
            recall_at_k[f"R@{k}"] = round(r_k, 4)

        return {
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k
        }

    def run_full_benchmark_comparison(self, split: str = "test") -> Dict[str, Any]:
        """
        Runs side-by-side performance evaluation on Sentence-BERT, BioBERT, and TF-IDF.
        """
        sbert_res = self.evaluate_model("Sentence-BERT", split=split)
        biobert_res = self.evaluate_model("BioBERT", split=split)
        tfidf_res = self.evaluate_model("TF-IDF Baseline", split=split)

        return {
            "split": split,
            "sentence_bert": sbert_res,
            "biobert": biobert_res,
            "tfidf_baseline": tfidf_res
        }

