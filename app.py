import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

import config
from database.db import init_db, get_dashboard_stats, get_search_history, delete_history_item, clear_all_history
from services.similarity_service import SimilarityService
from services.faiss_service import FAISSService
from services.dataset_service import DatasetService
from services.evaluation_service import EvaluationService
from models.traditional_similarity import TraditionalSimilarityService

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Initialize Database on startup
with app.app_context():
    init_db()

# Lazy-loaded services
similarity_service = SimilarityService()
faiss_service = FAISSService.get_instance()
dataset_service = DatasetService()
evaluation_service = EvaluationService()

# -------------------------------------------------------------
# Frontend Page Routes
# -------------------------------------------------------------

@app.route("/")
def dashboard_view():
    return render_template("dashboard.html", active_page="dashboard")

@app.route("/compare")
def compare_view():
    term_a = request.args.get("term_a", "")
    term_b = request.args.get("term_b", "")
    return render_template("compare.html", active_page="compare", term_a=term_a, term_b=term_b)

@app.route("/search")
def search_view():
    q = request.args.get("q", "")
    return render_template("search.html", active_page="search", initial_query=q)

@app.route("/dataset")
def dataset_view():
    return render_template("dataset.html", active_page="dataset")

@app.route("/model-comparison")
def model_comparison_view():
    return render_template("model_comparison.html", active_page="model_comparison")

@app.route("/evaluation")
def evaluation_view():
    return render_template("evaluation.html", active_page="evaluation")

@app.route("/history")
def history_view():
    return render_template("history.html", active_page="history")

@app.route("/about")
def about_view():
    return render_template("about.html", active_page="about")

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    POST /api/compare
    JSON: { "term_a": str, "term_b": str }
    """
    try:
        data = request.get_json(force=True) or {}
        term_a = data.get("term_a", "").strip()
        term_b = data.get("term_b", "").strip()

        if not term_a or not term_b:
            return jsonify({"status": "error", "message": "Please provide both 'term_a' and 'term_b'."}), 400

        result = similarity_service.compare(term_a, term_b, log_to_db=True)
        return jsonify({"status": "success", "data": result}), 200

    except Exception as e:
        logger.error(f"Error in /api/compare: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/search", methods=["POST"])
def api_search():
    """
    POST /api/search
    JSON: { "query": str, "top_k": int, "category": str }
    """
    try:
        data = request.get_json(force=True) or {}
        query = data.get("query", "").strip()
        top_k = int(data.get("top_k", 10))
        category = data.get("category", "all")

        if not query:
            return jsonify({"status": "error", "message": "Query parameter cannot be empty."}), 400

        # Bound top_k between 1 and 50
        top_k = max(1, min(50, top_k))

        results = faiss_service.search(query=query, top_k=top_k, category_filter=category)

        # Log query to DB for analytics
        top_match_term = results[0]["medical_term"] if results else ""
        top_score = results[0]["similarity_score"] if results else 0.0
        from database.db import log_query
        try:
            log_query(query, top_match_term, top_score, "FAISS + SBERT")
        except Exception as log_err:
            logger.warning(f"Could not log query: {log_err}")

        return jsonify({
            "status": "success",
            "query": query,
            "total_matches": len(results),
            "top_k": top_k,
            "results": results
        }), 200

    except Exception as e:
        logger.error(f"Error in /api/search: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    """
    GET /api/dashboard
    Returns summary statistics, category breakdown, relationship distribution, popular queries.
    """
    try:
        stats = get_dashboard_stats()
        total_terms = faiss_service.get_total_terms()
        categories = dataset_service.get_category_counts()

        # Build response payload
        payload = {
            "total_terms": total_terms,
            "total_comparisons": stats["total_comparisons"],
            "total_queries": stats["total_queries"],
            "avg_similarity_score": stats["avg_score"],
            "category_distribution": categories,
            "relationship_distribution": stats["relationship_counts"],
            "popular_queries": stats["popular_queries"],
            "recent_comparisons": stats["recent_comparisons"]
        }
        return jsonify({"status": "success", "data": payload}), 200

    except Exception as e:
        logger.error(f"Error in /api/dashboard: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    """
    GET /api/history?search=...&model=...&limit=...&offset=...
    """
    try:
        search_kw = request.args.get("search", None)
        model_filt = request.args.get("model", None)
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        records = get_search_history(search_keyword=search_kw, model_filter=model_filt, limit=limit, offset=offset)
        return jsonify({"status": "success", "count": len(records), "data": records}), 200

    except Exception as e:
        logger.error(f"Error in /api/history: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history/<int:history_id>", methods=["DELETE"])
def api_delete_history(history_id: int):
    """
    DELETE /api/history/<id>
    """
    try:
        success = delete_history_item(history_id)
        if success:
            return jsonify({"status": "success", "message": f"Record {history_id} deleted."}), 200
        else:
            return jsonify({"status": "error", "message": "Record not found."}), 404

    except Exception as e:
        logger.error(f"Error deleting history: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """
    POST /api/history/clear
    """
    try:
        count = clear_all_history()
        return jsonify({"status": "success", "message": f"Cleared {count} history records."}), 200
    except Exception as e:
        logger.error(f"Error clearing history: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/dataset/terms", methods=["GET"])
def api_dataset_terms():
    """
    GET /api/dataset/terms?search=...&category=...&page=...&per_page=...
    """
    try:
        search = request.args.get("search", "")
        category = request.args.get("category", "all")
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 15))

        data = dataset_service.get_terms(search=search, category=category, page=page, per_page=per_page)
        return jsonify({"status": "success", "data": data}), 200

    except Exception as e:
        logger.error(f"Error in /api/dataset/terms: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/dataset/add", methods=["POST"])
def api_dataset_add():
    """
    POST /api/dataset/add
    JSON: { "medical_term": str, "description": str, "category": str, "synonyms": str }
    """
    try:
        data = request.get_json(force=True) or {}
        term = data.get("medical_term", "")
        desc = data.get("description", "")
        cat = data.get("category", "")
        syn = data.get("synonyms", "")

        ok, msg, created = dataset_service.add_term(
            medical_term=term,
            description=desc,
            category=cat,
            synonyms=syn
        )
        if ok:
            return jsonify({"status": "success", "message": msg, "data": created}), 201
        else:
            return jsonify({"status": "error", "message": msg}), 400

    except Exception as e:
        logger.error(f"Error in /api/dataset/add: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/dataset/<term_id>", methods=["DELETE"])
def api_dataset_delete(term_id: str):
    """
    DELETE /api/dataset/<term_id>
    """
    try:
        ok, msg = dataset_service.delete_term(term_id)
        if ok:
            return jsonify({"status": "success", "message": msg}), 200
        else:
            return jsonify({"status": "error", "message": msg}), 404

    except Exception as e:
        logger.error(f"Error in /api/dataset/delete: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/dataset/upload", methods=["POST"])
def api_dataset_upload():
    """
    POST /api/dataset/upload (multipart/form-data with file)
    """
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file part in request."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected."}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"status": "error", "message": "Only CSV files are allowed."}), 400

        ok, msg, stats = dataset_service.upload_dataset(file)
        if ok:
            return jsonify({"status": "success", "message": msg, "stats": stats}), 200
        else:
            return jsonify({"status": "error", "message": msg}), 400

    except Exception as e:
        logger.error(f"Error in /api/dataset/upload: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/evaluation", methods=["GET"])
def api_evaluation():
    """
    GET /api/evaluation?model=sentence_bert&split=test&threshold=0.60
    """
    try:
        model = request.args.get("model", "Sentence-BERT")
        split = request.args.get("split", "test")
        threshold = float(request.args.get("threshold", config.RELATED_THRESHOLD))

        results = evaluation_service.evaluate_model(model_name=model, split=split, binary_threshold=threshold)
        return jsonify({"status": "success", "data": results}), 200

    except Exception as e:
        logger.error(f"Error in /api/evaluation: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/evaluation/benchmark", methods=["GET"])
def api_evaluation_benchmark():
    """
    GET /api/evaluation/benchmark?split=test
    Returns full side-by-side comparison across Sentence-BERT, BioBERT, and TF-IDF.
    """
    try:
        split = request.args.get("split", "test")
        data = evaluation_service.run_full_benchmark_comparison(split=split)
        return jsonify({"status": "success", "data": data}), 200

    except Exception as e:
        logger.error(f"Error in /api/evaluation/benchmark: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/model-comparison", methods=["POST"])
def api_model_comparison():
    """
    POST /api/model-comparison
    JSON: { "term_a": str, "term_b": str }
    Returns exhaustive multi-algorithm similarity scores and detailed analysis.
    """
    try:
        data = request.get_json(force=True) or {}
        term_a = data.get("term_a", "").strip()
        term_b = data.get("term_b", "").strip()

        if not term_a or not term_b:
            return jsonify({"status": "error", "message": "Both terms are required."}), 400

        full_comp = similarity_service.compare(term_a, term_b, log_to_db=False)
        trad = full_comp["traditional_metrics"]

        comparison_table = [
            {
                "method_type": "Lexical / Character",
                "method_name": "Levenshtein Edit Distance",
                "score": trad["levenshtein_similarity"],
                "score_display": f"{trad['levenshtein_similarity'] * 100:.1f}%",
                "distance": trad["levenshtein_distance"],
                "mechanism": "Character edit distance (insertions, deletions, substitutions)",
                "limitation": "Fails completely on non-cognate synonyms (e.g., 'Heart Attack' vs 'Myocardial Infarction')."
            },
            {
                "method_type": "Lexical / Token Set",
                "method_name": "Jaccard Token Similarity",
                "score": trad["jaccard_word"],
                "score_display": f"{trad['jaccard_word'] * 100:.1f}%",
                "distance": "-",
                "mechanism": "Set intersection over union of word tokens |A ∩ B| / |A ∪ B|",
                "limitation": "Zero overlap whenever words differ, ignoring synonymy and semantic relatedness."
            },
            {
                "method_type": "Lexical / Statistical",
                "method_name": "TF-IDF + Cosine Similarity",
                "score": trad["tfidf_cosine"],
                "score_display": f"{trad['tfidf_cosine'] * 100:.1f}%",
                "distance": "-",
                "mechanism": "Character n-gram & word frequency inverse document frequency vectors",
                "limitation": "Captures subword morphemes (e.g. 'cardio-'), but lacks deep contextual semantics."
            },
            {
                "method_type": "Transformer / Dense Vector",
                "method_name": "Sentence-BERT (all-MiniLM-L6-v2)",
                "score": full_comp["sbert_score"],
                "score_display": f"{full_comp['sbert_score'] * 100:.1f}%",
                "distance": "-",
                "mechanism": "Dense 384-dimensional Siamese contrastive embeddings with cosine distance",
                "limitation": "Trained predominantly on general text; can occasionally miss hyper-specialized clinical abbreviations."
            },
            {
                "method_type": "Transformer / Biomedical",
                "method_name": "BioBERT (PubMed Contextual)",
                "score": full_comp["biobert_score"],
                "score_display": f"{full_comp['biobert_score'] * 100:.1f}%" if full_comp['biobert_score'] is not None else "N/A",
                "distance": "-",
                "mechanism": "768-dimensional biomedical masked language model with attention-mask-aware mean pooling",
                "limitation": "High computational cost; requires mean pooling for sentence embeddings."
            }
        ]

        research_commentary = (
            f"Analysis for '{term_a}' vs '{term_b}':\n"
            f"Traditional lexical methods (Levenshtein: {trad['levenshtein_similarity']:.2f}, "
            f"Jaccard: {trad['jaccard_word']:.2f}, TF-IDF: {trad['tfidf_cosine']:.2f}) measure exact "
            f"character and token string overlap. When clinical terms are expressed in lay language versus "
            f"formal Latinate or Greek roots (e.g. 'Heart Attack' vs 'Myocardial Infarction'), lexical methods "
            f"fail with near-zero scores. In contrast, Transformer models (Sentence-BERT: {full_comp['sbert_score']:.2f}, "
            f"BioBERT: {full_comp['biobert_score']:.2f}) project medical concepts into continuous dense vector spaces, "
            f"accurately capturing clinical synonymy and contextual affinity."
        )

        return jsonify({
            "status": "success",
            "term_a": term_a,
            "term_b": term_b,
            "comparison_table": comparison_table,
            "research_commentary": research_commentary,
            "relationship": full_comp["relationship"],
            "badge_class": full_comp["badge_class"]
        }), 200

    except Exception as e:
        logger.error(f"Error in /api/model-comparison: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/status", methods=["GET"])
def api_status():
    """
    GET /api/status
    Returns model availability status so the frontend can show BioBERT download prompts.
    """
    try:
        from models.biobert import BioBERTModel
        bb = BioBERTModel.get_instance()
        return jsonify({
            "status": "success",
            "data": {
                "sbert_available": True,
                "biobert_available": not bb.is_fallback,
                "biobert_fallback": bb.is_fallback,
                "biobert_model": config.BIOBERT_MODEL_NAME,
                "sbert_model": config.SBERT_MODEL_NAME,
                "biobert_download_cmd": "python download_models.py",
                "message": (
                    "BioBERT is ready." if not bb.is_fallback
                    else "BioBERT weights not downloaded. Run 'python download_models.py' to enable real BioBERT. "
                         "Currently using Sentence-BERT as proxy for BioBERT embeddings."
                )
            }
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/status: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    logger.info(f"Starting Medical Terminology Matching Web Application on {config.HOST}:{config.PORT}...")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

