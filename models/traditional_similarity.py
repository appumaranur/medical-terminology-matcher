import re
from typing import Dict, Any, List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def tokenize_words(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words."""
    return re.findall(r"\b\w+\b", text.lower())

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes Levenshtein (edit) distance between two strings using dynamic programming.
    Time Complexity: O(m * n), Space Complexity: O(min(m, n)).
    """
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    # Ensure s1 is shorter to optimize memory
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    previous_row = list(range(len(s1) + 1))
    for i, c2 in enumerate(s2):
        current_row = [i + 1] + [0] * len(s1)
        for j, c1 in enumerate(s1):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]

def compute_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Computes normalized Levenshtein similarity:
        Sim_Lev(s1, s2) = 1.0 - (LevenshteinDistance(s1, s2) / max(len(s1), len(s2)))
    Range: [0.0, 1.0]
    """
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = compute_levenshtein_distance(s1, s2)
    return float(max(0.0, 1.0 - (dist / max_len)))

def compute_jaccard_similarity(s1: str, s2: str, mode: str = "word") -> float:
    """
    Computes Jaccard Similarity coefficient:
        J(A, B) = |A ∩ B| / |A ∪ B|
    Supports word-level ('word') or character n-gram ('char3') mode.
    """
    if mode == "word":
        tokens_a: Set[str] = set(tokenize_words(s1))
        tokens_b: Set[str] = set(tokenize_words(s2))
    elif mode == "char3":
        # 3-gram character shingles
        clean_a = re.sub(r"\s+", " ", s1.lower().strip())
        clean_b = re.sub(r"\s+", " ", s2.lower().strip())
        tokens_a = {clean_a[i:i+3] for i in range(max(0, len(clean_a) - 2))} or {clean_a}
        tokens_b = {clean_b[i:i+3] for i in range(max(0, len(clean_b) - 2))} or {clean_b}
    else:
        tokens_a = set(s1.lower())
        tokens_b = set(s2.lower())

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return float(intersection / union) if union > 0 else 0.0

def compute_tfidf_similarity(s1: str, s2: str) -> float:
    """
    Computes TF-IDF vector cosine similarity between two strings.
    Uses character n-grams (2 to 4) combined with word tokens to handle medical sub-word affixes
    (e.g., 'cardio-', '-itis', '-ectomy', 'hyper-').
    """
    s1, s2 = s1.strip(), s2.strip()
    if not s1 or not s2:
        return 0.0
    if s1.lower() == s2.lower():
        return 1.0

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=True
        )
        tfidf_matrix = vectorizer.fit_transform([s1, s2])
        sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(np.clip(sim_matrix[0, 0], 0.0, 1.0))
    except Exception:
        # Fallback to word-level overlap
        return compute_jaccard_similarity(s1, s2, mode="word")

class TraditionalSimilarityService:
    """
    Aggregated service computing all traditional lexical/syntactic similarity measures.
    """
    @staticmethod
    def compare_terms(term_a: str, term_b: str) -> Dict[str, Any]:
        lev_dist = compute_levenshtein_distance(term_a, term_b)
        lev_sim = compute_levenshtein_similarity(term_a, term_b)
        jaccard_word = compute_jaccard_similarity(term_a, term_b, mode="word")
        jaccard_char = compute_jaccard_similarity(term_a, term_b, mode="char3")
        tfidf_sim = compute_tfidf_similarity(term_a, term_b)

        tokens_a = set(tokenize_words(term_a))
        tokens_b = set(tokenize_words(term_b))
        overlapping_words = list(tokens_a & tokens_b)

        return {
            "tfidf_cosine": round(tfidf_sim, 4),
            "jaccard_word": round(jaccard_word, 4),
            "jaccard_char3": round(jaccard_char, 4),
            "levenshtein_similarity": round(lev_sim, 4),
            "levenshtein_distance": lev_dist,
            "overlapping_tokens": overlapping_words,
            "lexical_overlap_detected": len(overlapping_words) > 0
        }

