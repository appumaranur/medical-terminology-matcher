"""
Configuration module for the Medical Terminology Matching System.

Import order matters: this module MUST be imported before any HuggingFace
library is imported so the offline env-vars take effect before those
libraries read them at module-load time.
"""

import os
from pathlib import Path

# ── HuggingFace Offline / Online Control ─────────────────────────────────────
# Set HF_HUB_OFFLINE=1 to prevent slow network retries when models are cached.
# Set FORCE_HF_ONLINE=1 (externally) to allow downloads (used by download_models.py).
_force_online = os.environ.get("FORCE_HF_ONLINE", "0") == "1"
if not _force_online:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    # Explicitly unset so HF libs can reach the network
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

# ── Directory Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR        = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
DATABASE_DIR    = BASE_DIR / "database"
MODELS_DIR      = BASE_DIR / "models"
SERVICES_DIR    = BASE_DIR / "services"
STATIC_DIR      = BASE_DIR / "static"
TEMPLATES_DIR   = BASE_DIR / "templates"

# Ensure runtime directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)
DATABASE_DIR.mkdir(exist_ok=True, parents=True)

# ── File Paths ────────────────────────────────────────────────────────────────
DB_PATH              = DATABASE_DIR / "medical_matcher.db"
SCHEMA_PATH          = DATABASE_DIR / "schema.sql"
MEDICAL_TERMS_CSV    = DATA_DIR / "medical_terms.csv"
EVALUATION_PAIRS_CSV = DATA_DIR / "evaluation_pairs.csv"
FAISS_INDEX_PATH     = VECTOR_STORE_DIR / "medical_terms.index"
FAISS_METADATA_PATH  = VECTOR_STORE_DIR / "metadata.json"

# ── Model Configurations ──────────────────────────────────────────────────────
# Sentence-BERT: general-purpose semantic similarity model (384-dim)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# BioBERT: PubMed-pretrained biomedical BERT (768-dim).
# monologg/biobert_v1.1_pubmed stores weights as safetensors (safe to load).
# dmis-lab/biobert-base-cased-v1.2 uses legacy pytorch_model.bin which is
# blocked by CVE-2025-32434 security patch in transformers >= 4.50.
BIOBERT_MODEL_NAME      = "monologg/biobert_v1.1_pubmed"
CLINICAL_BERT_FALLBACK  = "dmis-lab/biobert-base-cased-v1.2"  # kept for reference only

EMBEDDING_BATCH_SIZE = 32

# ── Similarity Classification Thresholds ──────────────────────────────────────
# Calibrated against the all-MiniLM-L6-v2 cosine similarity distribution on
# medical terminology pairs.  Empirical calibration notes:
#   - General SBERT (all-MiniLM-L6-v2) scores ~0.59 for non-cognate clinical
#     synonyms (e.g. "Heart Attack" vs "Myocardial Infarction") because the model
#     was not fine-tuned on biomedical STS.  Thresholds are set conservatively:
#   - "Equivalent" at 0.55 catches true clinical synonyms for SBERT;
#     BioBERT (when available) will score these pairs higher (~0.75+).
# Note: High semantic similarity does NOT automatically prove clinical equivalence.
EQUIVALENT_THRESHOLD          = 0.55   # score >= 0.55 → Equivalent / Highly Similar
RELATED_THRESHOLD             = 0.40   # score >= 0.40 → Related
MODERATELY_SIMILAR_THRESHOLD  = 0.20   # score >= 0.20 → Moderately Similar
                                        # score <  0.20 → Unrelated

RELATIONSHIP_CLASSES = {
    "EQUIVALENT": "Equivalent / Highly Similar",
    "RELATED":    "Related",
    "MODERATE":   "Moderately Similar",
    "UNRELATED":  "Unrelated",
}

# ── Web Server Configuration ──────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "med-term-matcher-secret-key-2026")
HOST  = os.environ.get("HOST",  "127.0.0.1")
PORT  = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")
