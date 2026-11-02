import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from config import MEDICAL_TERMS_CSV
from services.faiss_service import FAISSService

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["term_id", "medical_term", "description", "category", "synonyms"]

class DatasetService:
    """
    Manages medical terminology dataset CRUD operations, validation,
    CSV exports/imports, and automatic vector index synchronization.
    """

    @staticmethod
    def get_terms(
        search: Optional[str] = None,
        category: Optional[str] = None,
        page: int = 1,
        per_page: int = 20
    ) -> Dict[str, Any]:
        """
        Retrieves paginated and filtered medical terms from the primary CSV dataset.
        """
        if not os.path.exists(MEDICAL_TERMS_CSV):
            return {"terms": [], "total": 0, "page": page, "per_page": per_page, "pages": 0, "categories": []}

        df = pd.read_csv(MEDICAL_TERMS_CSV).fillna("")
        all_categories = sorted(list(df["category"].unique()))

        if search:
            search_lower = search.strip().lower()
            mask = (
                df["medical_term"].str.lower().str.contains(search_lower) |
                df["description"].str.lower().str.contains(search_lower) |
                df["synonyms"].str.lower().str.contains(search_lower) |
                df["term_id"].str.lower().str.contains(search_lower)
            )
            df = df[mask]

        if category and category.lower() != "all":
            df = df[df["category"].str.lower() == category.lower()]

        total = len(df)
        total_pages = max(1, (total + per_page - 1) // per_page)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page

        sliced_df = df.iloc[start_idx:end_idx]
        terms = sliced_df.to_dict(orient="records")

        return {
            "terms": terms,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": total_pages,
            "categories": all_categories
        }

    @staticmethod
    def add_term(
        medical_term: str,
        description: str,
        category: str,
        synonyms: str = "",
        term_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validates and adds a new medical term to the dataset.
        Rebuilds the FAISS index automatically upon successful addition.
        """
        term_clean = medical_term.strip()
        desc_clean = description.strip()
        cat_clean = category.strip()
        syn_clean = synonyms.strip()

        if not term_clean or not desc_clean or not cat_clean:
            return False, "Medical term, description, and category are required.", None

        if os.path.exists(MEDICAL_TERMS_CSV):
            df = pd.read_csv(MEDICAL_TERMS_CSV).fillna("")
        else:
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)

        # Check for duplicate medical term (case-insensitive)
        if not df.empty and df["medical_term"].str.lower().eq(term_clean.lower()).any():
            return False, f"Medical term '{term_clean}' already exists in dataset.", None

        # Generate unique term_id if not provided
        if not term_id:
            max_num = 0
            if not df.empty and "term_id" in df.columns:
                for tid in df["term_id"].dropna():
                    match = re.search(r"(\d+)", str(tid))
                    if match:
                        max_num = max(max_num, int(match.group(1)))
            term_id = f"MED{max_num + 1:03d}"
        else:
            term_id = term_id.strip().upper()
            if not df.empty and df["term_id"].str.upper().eq(term_id).any():
                return False, f"Term ID '{term_id}' already exists in dataset.", None

        new_row = {
            "term_id": term_id,
            "medical_term": term_clean,
            "description": desc_clean,
            "category": cat_clean,
            "synonyms": syn_clean
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(MEDICAL_TERMS_CSV, index=False)

        # Synchronize vector index
        try:
            faiss_service = FAISSService.get_instance()
            faiss_service.rebuild_index_from_csv()
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index after term addition: {e}")

        return True, "Medical term added successfully.", new_row

    @staticmethod
    def delete_term(term_id: str) -> Tuple[bool, str]:
        """
        Deletes a medical term record by term_id and rebuilds the FAISS index.
        """
        if not os.path.exists(MEDICAL_TERMS_CSV):
            return False, "Dataset file does not exist."

        df = pd.read_csv(MEDICAL_TERMS_CSV).fillna("")
        initial_len = len(df)
        df = df[df["term_id"].str.upper() != term_id.strip().upper()]

        if len(df) == initial_len:
            return False, f"Term with ID '{term_id}' not found."

        df.to_csv(MEDICAL_TERMS_CSV, index=False)

        # Synchronize vector index
        try:
            faiss_service = FAISSService.get_instance()
            faiss_service.rebuild_index_from_csv()
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index after term deletion: {e}")

        return True, f"Term '{term_id}' deleted successfully."

    @staticmethod
    def upload_dataset(file_storage) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates, parses, and replaces or merges an uploaded CSV dataset.
        Checks for missing values, invalid headers, and duplicate entries.
        """
        try:
            new_df = pd.read_csv(file_storage).fillna("")
        except Exception as e:
            return False, f"Failed to parse uploaded CSV: {e}", {}

        # Validate headers
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in new_df.columns]
        if missing_cols:
            return False, f"Uploaded CSV missing required columns: {', '.join(missing_cols)}. Expected: {', '.join(REQUIRED_COLUMNS)}", {}

        # Check for missing values in required fields
        blank_terms = new_df[new_df["medical_term"].str.strip() == ""]
        if len(blank_terms) > 0:
            return False, f"Uploaded CSV contains {len(blank_terms)} records with missing 'medical_term'.", {}

        # Check for duplicates in uploaded file
        duplicates = new_df[new_df.duplicated(subset=["medical_term"], keep="first")]
        dedup_count = len(duplicates)
        new_df = new_df.drop_duplicates(subset=["medical_term"], keep="first")

        # Ensure term_id is populated
        for idx, row in new_df.iterrows():
            if not str(row["term_id"]).strip():
                new_df.at[idx, "term_id"] = f"MED{idx + 1:03d}"

        # Save validated dataset
        new_df[REQUIRED_COLUMNS].to_csv(MEDICAL_TERMS_CSV, index=False)

        # Synchronize FAISS index
        faiss_service = FAISSService.get_instance()
        indexed_count = faiss_service.rebuild_index_from_csv()

        stats = {
            "total_records": len(new_df),
            "duplicates_removed": dedup_count,
            "indexed_count": indexed_count,
            "categories": sorted(list(new_df["category"].unique()))
        }

        return True, "Dataset uploaded, validated, and indexed successfully.", stats

    @staticmethod
    def get_category_counts() -> Dict[str, int]:
        """Returns distribution of terms across medical specialties."""
        if not os.path.exists(MEDICAL_TERMS_CSV):
            return {}
        df = pd.read_csv(MEDICAL_TERMS_CSV).fillna("")
        if "category" not in df.columns:
            return {}
        return df["category"].value_counts().to_dict()

