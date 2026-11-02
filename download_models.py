"""
Pre-download script for BioBERT and Sentence-BERT models.

Usage:
    python download_models.py

This script downloads both models to the HuggingFace cache so the app
can run fully offline afterwards. Requires internet access.
"""

import os
import sys
import logging
from pathlib import Path

# Allow online downloads (override the offline-first setting in config.py)
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Model names ────────────────────────────────────────────────────────────────
# monologg/biobert_v1.1_pubmed stores weights as safetensors (safe to load).
# dmis-lab/biobert-base-cased-v1.2 uses the legacy pytorch_model.bin format,
# which is blocked by a CVE-2025-32434 security patch in transformers ≥ 4.50.
BIOBERT_MODEL   = "monologg/biobert_v1.1_pubmed"
SBERT_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"


def download_sbert() -> bool:
    """Download/verify Sentence-BERT (all-MiniLM-L6-v2)."""
    logger.info(f"Downloading Sentence-BERT: {SBERT_MODEL} ...")
    try:
        model = SentenceTransformer(SBERT_MODEL)
        dim = model.get_sentence_embedding_dimension()
        logger.info(f"✓ Sentence-BERT ready  (embedding dim = {dim})")
        return True
    except Exception as exc:
        logger.error(f"✗ Sentence-BERT download failed: {exc}")
        return False


def download_biobert() -> bool:
    """Download/verify BioBERT (monologg/biobert_v1.1_pubmed)."""
    logger.info(f"Downloading BioBERT: {BIOBERT_MODEL} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
        logger.info("  Tokenizer ✓")
        model = AutoModel.from_pretrained(BIOBERT_MODEL)
        logger.info(f"  Model ✓  (hidden size = {model.config.hidden_size})")
        logger.info("✓ BioBERT ready")
        return True
    except Exception as exc:
        logger.error(f"✗ BioBERT download failed: {exc}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Medical Terminology Matcher — Model Pre-Download")
    logger.info("=" * 60)

    results = {
        "sentence-bert": download_sbert(),
        "biobert":       download_biobert(),
    }

    logger.info("")
    logger.info("Download summary:")
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        logger.info(f"  {name:20s}  {status}")

    all_ok = all(results.values())
    if all_ok:
        logger.info("")
        logger.info("All models downloaded. You can now run the app offline:")
        logger.info("    python app.py")
    else:
        logger.warning("")
        logger.warning("Some models failed to download.")
        logger.warning("The app will fall back to Sentence-BERT for any missing model.")

    sys.exit(0 if all_ok else 1)


