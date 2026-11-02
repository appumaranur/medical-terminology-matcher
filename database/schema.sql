-- SQLite Schema for Medical Terminology Matching System

CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term_a TEXT NOT NULL,
    term_b TEXT NOT NULL,
    similarity_score REAL NOT NULL,
    model_name TEXT NOT NULL,
    relationship TEXT NOT NULL,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_term TEXT NOT NULL,
    top_match TEXT,
    top_score REAL,
    model_name TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    accuracy REAL NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1_score REAL NOT NULL,
    roc_auc REAL,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metrics_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at);
CREATE INDEX IF NOT EXISTS idx_query_logs_query_term ON query_logs(query_term);

