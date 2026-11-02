from typing import Dict, Any, Optional
from config import EQUIVALENT_THRESHOLD, RELATED_THRESHOLD, MODERATELY_SIMILAR_THRESHOLD, RELATIONSHIP_CLASSES

class RelationshipClassifier:
    """
    Classifies continuous semantic similarity scores into clinical & semantic relationship tiers,
    generating clear, clinically responsible explanations.

    Threshold calibration (all-MiniLM-L6-v2):
        >= 0.55  →  Equivalent / Highly Similar  (clinical synonyms, lay-to-clinical equivalents)
        >= 0.40  →  Related                      (disease-symptom, etiology-manifestation)
        >= 0.20  →  Moderately Similar            (same organ system / specialty)
        <  0.20  →  Unrelated
    """

    @staticmethod
    def classify(score: float) -> str:
        """
        Maps a cosine similarity score in [-1.0, 1.0] to a relationship tier.
        """
        if score >= EQUIVALENT_THRESHOLD:
            return RELATIONSHIP_CLASSES["EQUIVALENT"]
        elif score >= RELATED_THRESHOLD:
            return RELATIONSHIP_CLASSES["RELATED"]
        elif score >= MODERATELY_SIMILAR_THRESHOLD:
            return RELATIONSHIP_CLASSES["MODERATE"]
        else:
            return RELATIONSHIP_CLASSES["UNRELATED"]

    @staticmethod
    def get_badge_class(relationship: str) -> str:
        """Returns Bootstrap color class corresponding to relationship tier."""
        if "Equivalent" in relationship:
            return "badge-success"
        elif "Related" in relationship:
            return "badge-primary"
        elif "Moderate" in relationship:
            return "badge-warning"
        else:
            return "badge-secondary"

    @staticmethod
    def generate_explanation(
        term_a: str,
        term_b: str,
        score: float,
        model_name: str = "Sentence-BERT",
        sbert_score: Optional[float] = None,
        biobert_score: Optional[float] = None,
        lexical_overlap: bool = False
    ) -> Dict[str, Any]:
        """
        Generates an evidence-based clinical and NLP explanation for the relationship.
        Incorporates cautionary guidance: High semantic similarity does NOT automatically
        prove clinical equivalence.
        """
        relationship = RelationshipClassifier.classify(score)

        # Clinical reasoning details
        if score >= EQUIVALENT_THRESHOLD:
            clinical_summary = (
                f"'{term_a}' and '{term_b}' demonstrate strong semantic concordance (score: {score:.3f}). "
                f"They frequently represent clinical synonyms, lay-to-clinical terminology equivalents "
                f"(such as 'Heart Attack' and 'Myocardial Infarction'), or near-identical pathophysiological diagnoses."
            )
            clinical_caution = (
                "Clinical Caution: While the models predict strong semantic equivalence, clinical workflows must "
                "verify specific diagnostic criteria, ICD-10/SNOMED-CT classifications, anatomical specifics, and staging."
            )
        elif score >= RELATED_THRESHOLD:
            clinical_summary = (
                f"'{term_a}' and '{term_b}' share a substantial thematic and physiological relationship (score: {score:.3f}). "
                f"This commonly reflects disease-symptom pairs, etiology-manifestation pairings "
                f"(e.g., 'Diabetes Mellitus' and 'Hyperglycemia'), or co-occurring comorbidities."
            )
            clinical_caution = (
                "Clinical Caution: These entities are related but not clinically interchangeable. Treating or coding "
                "one as the other in medical charts without clinical justification may lead to diagnostic inaccuracy."
            )
        elif score >= MODERATELY_SIMILAR_THRESHOLD:
            clinical_summary = (
                f"'{term_a}' and '{term_b}' have moderate contextual affinity (score: {score:.3f}). "
                f"They may belong to the same medical domain or organ system, but they denote distinct conditions, "
                f"interventions, or anatomical structures."
            )
            clinical_caution = (
                "These terms share broad physiological context but exhibit distinct clinical management pathways."
            )
        else:
            clinical_summary = (
                f"'{term_a}' and '{term_b}' exhibit negligible semantic correlation (score: {score:.3f}). "
                f"They likely pertain to completely separate medical disciplines, organ systems, or biological mechanisms."
            )
            clinical_caution = "No significant semantic or clinical overlap identified."

        nlp_insight = ""
        if sbert_score is not None and biobert_score is not None:
            diff = abs(sbert_score - biobert_score)
            if diff > 0.15:
                nlp_insight = (
                    f"Model Discrepancy Note: Sentence-BERT ({sbert_score:.3f}) and BioBERT ({biobert_score:.3f}) "
                    f"diverge by {diff:.3f}. Sentence-BERT captures general conversational and semantic phrasing, "
                    f"while BioBERT's biomedical pre-training on PubMed abstracts provides deeper domain-specific token representations."
                )
            else:
                nlp_insight = (
                    f"Consensus Agreement: Both Sentence-BERT ({sbert_score:.3f}) and BioBERT ({biobert_score:.3f}) "
                    f"exhibit close alignment on this medical term pair."
                )

        return {
            "relationship": relationship,
            "badge_class": RelationshipClassifier.get_badge_class(relationship),
            "score": round(score, 4),
            "clinical_summary": clinical_summary,
            "clinical_caution": clinical_caution,
            "nlp_insight": nlp_insight,
            "lexical_overlap": lexical_overlap
        }

