"""
database.py
-----------
Thin wrapper around sqlite3. We use raw SQL instead of an ORM on purpose:
this is a portfolio project, and being able to show the exact schema and
exact queries in an interview is worth more than ORM convenience.

Swap-out note: for a real deployment, replace this module's connection
string with a PostgreSQL connection (e.g. via psycopg2 / SQLAlchemy) as
called out in the project doc's tech stack. The schema below maps 1:1
onto Postgres tables with minimal changes (INTEGER PRIMARY KEY -> SERIAL).
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "rcia.db")
DB_PATH = os.path.abspath(DB_PATH)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS internal_clauses (
    id TEXT PRIMARY KEY,
    doc_title TEXT NOT NULL,
    section TEXT,
    clause_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regulations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT,
    raw_text TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regulation_clauses (
    id TEXT PRIMARY KEY,
    regulation_id TEXT NOT NULL REFERENCES regulations(id),
    clause_text TEXT NOT NULL,
    obligation_actor TEXT,
    obligation_action TEXT,
    effective_date TEXT
);

CREATE TABLE IF NOT EXISTS review_items (
    id TEXT PRIMARY KEY,
    regulation_id TEXT NOT NULL REFERENCES regulations(id),
    regulation_clause_id TEXT NOT NULL REFERENCES regulation_clauses(id),
    internal_policy_id TEXT NOT NULL,
    internal_clause_id TEXT NOT NULL REFERENCES internal_clauses(id),
    impact_type TEXT NOT NULL,
    draft_redline TEXT,
    business_action TEXT,
    below_threshold INTEGER NOT NULL DEFAULT 0,
    citation_verified INTEGER NOT NULL,
    confidence REAL NOT NULL,
    risk TEXT NOT NULL,
    routing TEXT NOT NULL,
    second_hop_clauses TEXT,
    reasoning_trace TEXT NOT NULL,
    retrieval_score REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    reviewer_action TEXT,
    reviewer_notes TEXT,
    created_at TEXT NOT NULL,
    reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS eval_labels (
    id TEXT PRIMARY KEY,
    regulation_clause_text TEXT NOT NULL,
    internal_clause_id TEXT NOT NULL,
    is_impacted INTEGER NOT NULL,
    true_impact_type TEXT,
    notes TEXT
);
"""


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(review_items)").fetchall()]
        if "business_action" not in cols:
            conn.execute("ALTER TABLE review_items ADD COLUMN business_action TEXT")
        if "below_threshold" not in cols:
            conn.execute("ALTER TABLE review_items ADD COLUMN below_threshold INTEGER NOT NULL DEFAULT 0")
        if "internal_policy_id" not in cols:
            conn.execute("ALTER TABLE review_items ADD COLUMN internal_policy_id TEXT")


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def loads(s):
    if s is None:
        return None
    return json.loads(s)
