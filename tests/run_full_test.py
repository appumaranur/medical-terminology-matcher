"""
Full end-to-end pipeline validation test.
Run: python tests/run_full_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # must be first - sets HF_HUB_OFFLINE

from models.sentence_bert import SentenceBERTModel
from models.biobert import BioBERTModel
from models.classifier import RelationshipClassifier
from models.traditional_similarity import TraditionalSimilarityService
from services.faiss_service import FAISSService

# ── 1. SBERT ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("1. Sentence-BERT Model")
print("=" * 70)
sbert = SentenceBERTModel.get_instance()
print(f"   Embedding dim: {sbert.get_dimension()}")

pairs = [
    ("Heart Attack",       "Myocardial Infarction",    "Equivalent / Highly Similar"),
    ("Hypertension",       "High Blood Pressure",      "Equivalent / Highly Similar"),
    ("Diabetes Mellitus",  "High Blood Sugar",         "Related"),
    ("Stroke",             "Cerebrovascular Accident", "Equivalent / Highly Similar"),
    ("Fever",              "Pyrexia",                  "Equivalent / Highly Similar"),
    ("Pneumonia",          "Lung Infection",           "Equivalent / Highly Similar"),
    ("Appendicitis",       "Quantum Physics",          "Unrelated"),
]

print(f"\n   {'Pair':<60s} {'Score':>6}  Predicted                Expected")
print("   " + "-" * 100)
correct = 0
for a, b, expected in pairs:
    s = sbert.compute_similarity(a, b)
    rel = RelationshipClassifier.classify(s)
    hit = "OK" if rel == expected else "FAIL"
    if rel == expected:
        correct += 1
    print(f"   [{hit}] {(a + ' vs ' + b):<60s} {s:>6.4f}  {rel:<25s}  {expected}")

print(f"\n   SBERT Accuracy: {correct}/{len(pairs)} ({correct/len(pairs)*100:.0f}%)")

# ── 2. BioBERT ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. BioBERT Model")
print("=" * 70)
bb = BioBERTModel.get_instance()
print(f"   Fallback mode: {bb.is_fallback}")
if bb.is_fallback:
    print("   WARNING: BioBERT weights not downloaded. Run 'python download_models.py'")
    print("            Currently using Sentence-BERT as proxy for BioBERT.")
else:
    print(f"   Hidden size: {bb.model.config.hidden_size}")

for a, b, _ in pairs[:4]:
    s = bb.compute_similarity(a, b)
    rel = RelationshipClassifier.classify(s)
    print(f"   {(a + ' vs ' + b):<60s} {s:>6.4f}  {rel}")

# ── 3. Traditional Similarity ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. Traditional Similarity Algorithms")
print("=" * 70)
trad = TraditionalSimilarityService()
for a, b in [("Heart Attack", "Myocardial Infarction"), ("Hypertension", "High Blood Pressure")]:
    r = trad.compare_terms(a, b)
    print(f"   {a} vs {b}")
    print(f"     TF-IDF: {r['tfidf_cosine']:.4f}  Jaccard: {r['jaccard_word']:.4f}  Levenshtein: {r['levenshtein_similarity']:.4f}")

# ── 4. FAISS Search ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. FAISS Semantic Search")
print("=" * 70)
faiss_svc = FAISSService.get_instance()
print(f"   Index: {faiss_svc.index.ntotal} vectors, {faiss_svc.get_total_terms()} unique terms")

for query in ["Heart Attack", "Hypertension", "Diabetes"]:
    results = faiss_svc.search(query, top_k=5)
    print(f"\n   Query: '{query}'")
    for r in results:
        print(f"     {r['rank']}. {r['medical_term']:<35s} {r['similarity_score']:.4f}  {r['relationship']}")

print("\n" + "=" * 70)
print("ALL TESTS DONE")
print("=" * 70)

