import sqlite3
import json
from typing import List, Dict, Any, Optional
from config import DB_PATH, SCHEMA_PATH

def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row factory configured."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes the database schema if tables do not exist."""
    with get_connection() as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()

def log_comparison(
    term_a: str,
    term_b: str,
    similarity_score: float,
    model_name: str,
    relationship: str,
    explanation: str = ""
) -> int:
    """Logs a single comparison to the search_history table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO search_history (term_a, term_b, similarity_score, model_name, relationship, explanation)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (term_a, term_b, round(similarity_score, 4), model_name, relationship, explanation)
        )
        conn.commit()
        return cursor.lastrowid

def log_query(query_term: str, top_match: str, top_score: float, model_name: str) -> int:
    """Logs a search query to the query_logs table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO query_logs (query_term, top_match, top_score, model_name)
            VALUES (?, ?, ?, ?)
            """,
            (query_term, top_match, round(top_score, 4), model_name)
        )
        conn.commit()
        return cursor.lastrowid

def get_search_history(
    search_keyword: Optional[str] = None,
    model_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Retrieves search history with optional filtering and pagination."""
    query = "SELECT * FROM search_history WHERE 1=1"
    params: List[Any] = []

    if search_keyword:
        query += " AND (term_a LIKE ? OR term_b LIKE ? OR relationship LIKE ?)"
        wildcard = f"%{search_keyword}%"
        params.extend([wildcard, wildcard, wildcard])

    if model_filter and model_filter.lower() != "all":
        query += " AND model_name = ?"
        params.append(model_filter)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def delete_history_item(history_id: int) -> bool:
    """Deletes a single history record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_history WHERE id = ?", (history_id,))
        conn.commit()
        return cursor.rowcount > 0

def clear_all_history() -> int:
    """Clears all records from search_history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_history")
        conn.commit()
        return cursor.rowcount

def get_dashboard_stats() -> Dict[str, Any]:
    """Computes summary statistics for the interactive dashboard."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Total comparisons
        cursor.execute("SELECT COUNT(*) AS total_comparisons, AVG(similarity_score) AS avg_score FROM search_history")
        row = cursor.fetchone()
        total_comparisons = row["total_comparisons"] or 0
        avg_score = round(row["avg_score"] or 0.0, 3)

        # Total queries performed
        cursor.execute("SELECT COUNT(*) AS total_queries FROM query_logs")
        total_queries = cursor.fetchone()["total_queries"] or 0

        # Most searched query terms
        cursor.execute(
            """
            SELECT query_term, COUNT(*) AS count
            FROM query_logs
            GROUP BY query_term
            ORDER BY count DESC
            LIMIT 5
            """
        )
        popular_queries = [dict(r) for r in cursor.fetchall()]

        # Relationship distribution in search history
        cursor.execute(
            """
            SELECT relationship, COUNT(*) AS count
            FROM search_history
            GROUP BY relationship
            """
        )
        relationship_counts = {r["relationship"]: r["count"] for r in cursor.fetchall()}

        # Recent comparisons
        cursor.execute(
            """
            SELECT id, term_a, term_b, similarity_score, model_name, relationship, created_at
            FROM search_history
            ORDER BY created_at DESC
            LIMIT 6
            """
        )
        recent_comparisons = [dict(r) for r in cursor.fetchall()]

        return {
            "total_comparisons": total_comparisons,
            "total_queries": total_queries,
            "avg_score": avg_score,
            "popular_queries": popular_queries,
            "relationship_counts": relationship_counts,
            "recent_comparisons": recent_comparisons
        }

def log_evaluation_run(
    model_name: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    roc_auc: Optional[float],
    metrics_dict: Dict[str, Any]
) -> int:
    """Saves an evaluation run to evaluation_history table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO evaluation_history (model_name, accuracy, precision, recall, f1_score, roc_auc, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                round(accuracy, 4),
                round(precision, 4),
                round(recall, 4),
                round(f1_score, 4),
                round(roc_auc, 4) if roc_auc is not None else None,
                json.dumps(metrics_dict)
            )
        )
        conn.commit()
        return cursor.lastrowid

