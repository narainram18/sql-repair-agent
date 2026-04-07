"""
sql_repair_env.py
-----------------
DBOps Agent Environment — SQL Query Repair
Meta OpenEnv Hackathon Submission

Architecture:
  - FastAPI app exposing /reset, /step, /state
  - In-memory SQLite database reseeded on every reset()
  - Three task difficulties: Easy, Medium, Hard
  - Pydantic-typed Action/Observation models
  - Deterministic graders returning scores in [0.0, 1.0]
"""

from __future__ import annotations

import json
import sqlite3
import traceback
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic Models (OpenEnv Spec)
# ---------------------------------------------------------------------------


class SQLAction(BaseModel):
    """Agent action sent to /step."""

    action_type: str = Field(
        description="Must be 'INSPECT_SCHEMA', 'TEST_QUERY', or 'SUBMIT_FINAL_QUERY'."
    )
    sql_command: Optional[str] = Field(
        default=None,
        description="The SQL query to execute. Required for TEST_QUERY and SUBMIT_FINAL_QUERY.",
    )


class SQLObservation(BaseModel):
    """Environment observation returned from /step and /reset."""

    current_schema: str = Field(
        description="DDL of the tables. Populated when action is INSPECT_SCHEMA."
    )
    target_output_preview: str = Field(
        description="Preview of expected data or explanation of target."
    )
    last_execution_status: str = Field(
        description="'SUCCESS', 'SYNTAX_ERROR', 'LOGIC_ERROR', or 'PENDING'."
    )
    last_query_result: Optional[str] = Field(
        default=None,
        description="JSON string of returned rows or SQLite error trace.",
    )
    step_reward: float = Field(description="Reward granted for the last action.")


class StepResponse(BaseModel):
    """Full response envelope from /step."""

    observation: SQLObservation
    done: bool
    total_reward: float
    step_count: int
    task_id: str


class ResetResponse(BaseModel):
    """Full response envelope from /reset."""

    observation: SQLObservation
    task_id: str
    broken_query: str
    message: str


class StateResponse(BaseModel):
    """Snapshot of current environment state from /state."""

    task_id: str
    step_count: int
    total_reward: float
    done: bool
    broken_query: str
    max_steps: int


# ---------------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------------

# Realistic seed data used across all tasks
_SEED_SQL = """
CREATE TABLE users (
    user_id     INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    signup_date TEXT    NOT NULL
);

CREATE TABLE products (
    product_id  INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    price       REAL    NOT NULL
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    product_id  INTEGER NOT NULL,
    order_date  TEXT    NOT NULL,
    quantity    INTEGER NOT NULL,
    FOREIGN KEY (user_id)    REFERENCES users(user_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Users: 8 users, including 2 with no orders (user_id 7, 8)
INSERT INTO users VALUES
  (1, 'Alice Johnson',  '2025-06-15'),
  (2, 'Bob Smith',      '2026-01-20'),
  (3, 'Carol White',    '2025-11-03'),
  (4, 'David Brown',    '2026-02-14'),
  (5, 'Eva Martinez',   '2025-08-30'),
  (6, 'Frank Lee',      '2026-03-01'),
  (7, 'Grace Kim',      '2026-01-05'),
  (8, 'Henry Zhang',    '2026-02-28');

-- Products across 4 categories
INSERT INTO products VALUES
  (1,  'Laptop Pro 15',    'Electronics', 1299.99),
  (2,  'USB-C Hub',        'Electronics',   49.99),
  (3,  'Desk Chair Ergo',  'Furniture',    399.99),
  (4,  'Standing Desk',    'Furniture',    549.99),
  (5,  'Python Cookbook',  'Books',         39.99),
  (6,  'SQL Mastery',      'Books',         34.99),
  (7,  'Noise Headphones', 'Electronics',  249.99),
  (8,  'Desk Lamp LED',    'Furniture',     79.99),
  (9,  'Cloud Arch Guide', 'Books',         44.99),
  (10, 'Webcam 4K',        'Electronics',  129.99);

-- Orders spread across 2025-11 to 2026-03 (users 7 & 8 have NO orders)
INSERT INTO orders VALUES
  (1,  1, 1,  '2025-11-10', 1),
  (2,  1, 5,  '2025-12-01', 2),
  (3,  2, 3,  '2026-01-22', 1),
  (4,  2, 7,  '2026-02-05', 1),
  (5,  3, 2,  '2025-11-18', 3),
  (6,  3, 10, '2026-01-09', 1),
  (7,  4, 4,  '2026-02-20', 1),
  (8,  4, 6,  '2026-03-01', 2),
  (9,  5, 1,  '2025-12-15', 1),
  (10, 5, 8,  '2026-01-30', 2),
  (11, 6, 9,  '2026-03-05', 1),
  (12, 6, 2,  '2026-03-10', 2),
  (13, 1, 7,  '2026-02-14', 1),
  (14, 3, 5,  '2026-02-28', 1),
  (15, 5, 6,  '2026-03-12', 3);
"""


def _build_db() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite database with seed data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SEED_SQL)
    conn.commit()
    return conn


def _get_schema(conn: sqlite3.Connection) -> str:
    """Return the DDL (CREATE TABLE statements) for all user tables."""
    cur = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name;"
    )
    rows = cur.fetchall()
    return "\n\n".join(row[0] for row in rows if row[0])


def _run_query(conn: sqlite3.Connection, sql: str) -> tuple[str, list[dict]]:
    """
    Execute a SQL query and return (status, rows).
    status is 'SUCCESS', 'SYNTAX_ERROR', or 'LOGIC_ERROR'.
    rows is a list of dicts (empty on error).
    """
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        return "SUCCESS", rows
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        # SQLite raises OperationalError for both syntax and runtime issues
        if any(kw in msg.lower() for kw in ("syntax", "no such", "near")):
            return "SYNTAX_ERROR", [{"error": msg}]
        return "LOGIC_ERROR", [{"error": msg}]
    except Exception as exc:
        return "SYNTAX_ERROR", [{"error": str(exc)}]


# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------


class BaseTask:
    """Abstract task base class."""

    task_id: str = "base"
    difficulty: str = "easy"
    broken_query: str = ""
    target_description: str = ""

    def ground_truth_rows(self, conn: sqlite3.Connection) -> list[dict]:
        """Execute the canonical correct query and return rows."""
        raise NotImplementedError

    def grade(self, conn: sqlite3.Connection, submitted_sql: str) -> float:
        """
        Deterministic grader. Returns score in [0.0, 1.0].
        Subclasses implement _score_rows() to evaluate the result.
        """
        status, rows = _run_query(conn, submitted_sql)
        if status != "SUCCESS":
            return 0.0
        return self._score_rows(conn, rows)

    def _score_rows(self, conn: sqlite3.Connection, rows: list[dict]) -> float:
        raise NotImplementedError

    def target_preview(self, conn: sqlite3.Connection) -> str:
        """Return a human-readable preview of the expected output."""
        rows = self.ground_truth_rows(conn)
        preview = json.dumps(rows[:3], indent=2)
        return (
            f"Expected {len(rows)} row(s). First up to 3 rows:\n{preview}\n"
            f"Description: {self.target_description}"
        )


# ── Task 1: Easy — Syntax & Typo Repair ─────────────────────────────────────

class EasySyntaxRepairTask(BaseTask):
    """
    Fix two typos:
      1. Table name  : 'user'  → 'users'
      2. SQL keyword : 'ODER BY' → 'ORDER BY'
    """

    task_id = "easy_syntax_repair"
    difficulty = "easy"
    broken_query = (
        "SELECT name, signup_date FROM user "
        "WHERE signup_date > '2026-01-01' "
        "ODER BY name"
    )
    target_description = (
        "Users who signed up after 2026-01-01, ordered alphabetically by name. "
        "Columns: name, signup_date."
    )
    _CORRECT_SQL = (
        "SELECT name, signup_date FROM users "
        "WHERE signup_date > '2026-01-01' "
        "ORDER BY name"
    )

    def ground_truth_rows(self, conn: sqlite3.Connection) -> list[dict]:
        _, rows = _run_query(conn, self._CORRECT_SQL)
        return rows

    def _score_rows(self, conn: sqlite3.Connection, rows: list[dict]) -> float:
        truth = self.ground_truth_rows(conn)
        if not rows:
            return 0.0
        # Normalise rows to sets of frozensets for order-insensitive comparison
        def _norm(r: dict) -> frozenset:
            return frozenset((k.lower(), str(v)) for k, v in r.items())

        truth_set = {_norm(r) for r in truth}
        agent_set = {_norm(r) for r in rows}

        if agent_set == truth_set:
            return 1.0
        # Partial: rows execute but data is wrong (e.g. missing ORDER BY is still correct data)
        # Check column overlap at least
        if truth_set & agent_set:
            return 0.5
        return 0.5  # Executes but wrong data


# ── Task 2: Medium — Logical Error & Missing LEFT JOIN ───────────────────────

class MediumJoinRepairTask(BaseTask):
    """
    Fix lifetime revenue calculation:
      - INNER JOIN → LEFT JOIN so users with 0 purchases appear
      - Add COALESCE(SUM(...), 0) to handle NULLs
    """

    task_id = "medium_join_repair"
    difficulty = "medium"
    broken_query = (
        "SELECT u.user_id, u.name, SUM(p.price * o.quantity) AS lifetime_revenue "
        "FROM users u "
        "INNER JOIN orders o ON u.user_id = o.user_id "
        "INNER JOIN products p ON o.product_id = p.product_id "
        "GROUP BY u.user_id, u.name "
        "ORDER BY u.user_id"
    )
    target_description = (
        "Lifetime revenue per user, including users with 0 purchases (revenue = 0.0). "
        "Columns: user_id, name, lifetime_revenue. All 8 users must appear."
    )
    _CORRECT_SQL = (
        "SELECT u.user_id, u.name, "
        "COALESCE(SUM(p.price * o.quantity), 0) AS lifetime_revenue "
        "FROM users u "
        "LEFT JOIN orders o ON u.user_id = o.user_id "
        "LEFT JOIN products p ON o.product_id = p.product_id "
        "GROUP BY u.user_id, u.name "
        "ORDER BY u.user_id"
    )

    def ground_truth_rows(self, conn: sqlite3.Connection) -> list[dict]:
        _, rows = _run_query(conn, self._CORRECT_SQL)
        return rows

    def _score_rows(self, conn: sqlite3.Connection, rows: list[dict]) -> float:
        truth = self.ground_truth_rows(conn)
        if not rows:
            return 0.0

        truth_ids = {r["user_id"] for r in truth}
        agent_ids = {r.get("user_id") for r in rows}

        all_users_present = truth_ids == agent_ids
        # Check revenue values match (rounded to 2dp)
        truth_rev = {
            r["user_id"]: round(float(r["lifetime_revenue"]), 2) for r in truth
        }
        agent_rev: dict[Any, float] = {}
        for r in rows:
            uid = r.get("user_id")
            rev_key = next(
                (k for k in r if "revenue" in k.lower() or "rev" in k.lower()), None
            )
            if uid and rev_key:
                try:
                    agent_rev[uid] = round(float(r[rev_key]), 2)
                except (TypeError, ValueError):
                    pass

        revenues_correct = truth_rev == agent_rev

        if all_users_present and revenues_correct:
            return 1.0
        if all_users_present and not revenues_correct:
            # All 8 users present but wrong math
            return 0.7
        if not all_users_present:
            # Missing 0-purchase users — INNER JOIN still used
            return 0.3
        return 0.3


# ── Task 3: Hard — Window Functions & CTE ───────────────────────────────────

class HardWindowFunctionTask(BaseTask):
    """
    Fix top-selling product category per month:
      - Use a CTE to aggregate monthly revenue per category
      - Apply RANK() OVER (PARTITION BY month ORDER BY revenue DESC)
      - Filter WHERE rank = 1
    """

    task_id = "hard_window_function"
    difficulty = "hard"
    broken_query = (
        "SELECT strftime('%Y-%m', o.order_date) AS month, "
        "p.category, SUM(p.price * o.quantity) AS revenue "
        "FROM orders o "
        "JOIN products p ON o.product_id = p.product_id "
        "GROUP BY month, p.category "
        "ORDER BY month, revenue DESC"
    )
    target_description = (
        "The single top-revenue product category for each calendar month. "
        "Columns: month, category, revenue. One row per month."
    )
    _CORRECT_SQL = """
WITH monthly_category_revenue AS (
    SELECT
        strftime('%Y-%m', o.order_date) AS month,
        p.category,
        SUM(p.price * o.quantity) AS revenue
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    GROUP BY month, p.category
),
ranked AS (
    SELECT
        month,
        category,
        revenue,
        RANK() OVER (PARTITION BY month ORDER BY revenue DESC) AS rnk
    FROM monthly_category_revenue
)
SELECT month, category, revenue
FROM ranked
WHERE rnk = 1
ORDER BY month
"""

    def ground_truth_rows(self, conn: sqlite3.Connection) -> list[dict]:
        _, rows = _run_query(conn, self._CORRECT_SQL)
        return rows

    def _score_rows(self, conn: sqlite3.Connection, rows: list[dict]) -> float:
        truth = self.ground_truth_rows(conn)
        if not rows:
            return 0.0

        truth_months = {r["month"] for r in truth}
        agent_months = {r.get("month") for r in rows}

        # Perfect match
        def _norm(r: dict) -> tuple:
            return (
                str(r.get("month", "")),
                str(r.get("category", "")),
                round(float(r.get("revenue", 0)), 2),
            )

        truth_tuples = {_norm(r) for r in truth}
        agent_tuples: set[tuple] = set()
        for r in rows:
            try:
                agent_tuples.add(_norm(r))
            except (TypeError, ValueError):
                pass

        if truth_tuples == agent_tuples:
            return 1.0

        # Returns all categories per month (no filtering by rank)
        if len(rows) > len(truth) and truth_months == agent_months:
            return 0.8

        # Groups by month only (no category breakdown or wrong category)
        if truth_months == agent_months and len(rows) == len(truth):
            return 0.4

        # Something ran but is structurally off
        if agent_months & truth_months:
            return 0.4

        return 0.0


# ---------------------------------------------------------------------------
# Environment Core
# ---------------------------------------------------------------------------

_TASKS: dict[str, BaseTask] = {
    t.task_id: t
    for t in [
        EasySyntaxRepairTask(),
        MediumJoinRepairTask(),
        HardWindowFunctionTask(),
    ]
}

MAX_STEPS = 20
STEP_PENALTY = -0.01
REWARD_INSPECT = 0.05
REWARD_TEST_ERROR = -0.10
REWARD_TEST_WRONG = 0.10


class SQLRepairEnvironment:
    """
    Stateful environment instance managing one episode.
    Thread-unsafe by design; one instance per request session.
    """

    def __init__(self) -> None:
        self.conn: Optional[sqlite3.Connection] = None
        self.task: Optional[BaseTask] = None
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.done: bool = False
        self._schema_cache: str = ""

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy_syntax_repair") -> ResetResponse:
        """Seed a new database and start a fresh episode."""
        if task_id not in _TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Valid: {list(_TASKS.keys())}"
            )

        # Tear down any existing connection
        if self.conn:
            self.conn.close()

        self.conn = _build_db()
        self.task = _TASKS[task_id]
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self._schema_cache = _get_schema(self.conn)

        obs = SQLObservation(
            current_schema="",
            target_output_preview=self.task.target_preview(self.conn),
            last_execution_status="PENDING",
            last_query_result=None,
            step_reward=0.0,
        )
        return ResetResponse(
            observation=obs,
            task_id=self.task.task_id,
            broken_query=self.task.broken_query,
            message=(
                f"Episode started. Task: [{self.task.difficulty.upper()}] "
                f"{self.task.task_id}. Max steps: {MAX_STEPS}."
            ),
        )

    def step(self, action: SQLAction) -> StepResponse:
        """Process one agent action and return the next observation."""
        if self.done:
            raise RuntimeError("Episode is finished. Call /reset to start a new one.")
        if self.conn is None or self.task is None:
            raise RuntimeError("Environment not initialised. Call /reset first.")

        self.step_count += 1
        reward = STEP_PENALTY  # per-step efficiency penalty
        status = "PENDING"
        result_json: Optional[str] = None
        schema_out = ""

        action_type = action.action_type.upper()

        # ── INSPECT_SCHEMA ───────────────────────────────────────────────────
        if action_type == "INSPECT_SCHEMA":
            reward += REWARD_INSPECT
            schema_out = self._schema_cache
            status = "SUCCESS"
            result_json = None

        # ── TEST_QUERY ───────────────────────────────────────────────────────
        elif action_type == "TEST_QUERY":
            if not action.sql_command:
                reward += REWARD_TEST_ERROR
                status = "SYNTAX_ERROR"
                result_json = json.dumps([{"error": "sql_command is required for TEST_QUERY"}])
            else:
                exec_status, rows = _run_query(self.conn, action.sql_command)
                status = exec_status
                result_json = json.dumps(rows[:20])  # cap preview rows
                if exec_status != "SUCCESS":
                    reward += REWARD_TEST_ERROR
                else:
                    reward += REWARD_TEST_WRONG  # executed; caller decides if correct

        # ── SUBMIT_FINAL_QUERY ───────────────────────────────────────────────
        elif action_type == "SUBMIT_FINAL_QUERY":
            if not action.sql_command:
                score = 0.0
                status = "SYNTAX_ERROR"
                result_json = json.dumps([{"error": "sql_command required for SUBMIT_FINAL_QUERY"}])
            else:
                score = self.task.grade(self.conn, action.sql_command)
                exec_status, rows = _run_query(self.conn, action.sql_command)
                status = exec_status
                result_json = json.dumps(rows[:20])
                reward += score  # grader score IS the submit reward
            self.done = True

        else:
            raise ValueError(
                f"Unknown action_type '{action.action_type}'. "
                "Must be INSPECT_SCHEMA, TEST_QUERY, or SUBMIT_FINAL_QUERY."
            )

        # Enforce max steps
        if self.step_count >= MAX_STEPS and not self.done:
            self.done = True

        self.total_reward += reward

        obs = SQLObservation(
            current_schema=schema_out,
            target_output_preview=self.task.target_preview(self.conn),
            last_execution_status=status,
            last_query_result=result_json,
            step_reward=round(reward, 4),
        )
        return StepResponse(
            observation=obs,
            done=self.done,
            total_reward=round(self.total_reward, 4),
            step_count=self.step_count,
            task_id=self.task.task_id,
        )

    def state(self) -> StateResponse:
        """Return a lightweight snapshot of the current episode state."""
        if self.task is None:
            raise RuntimeError("Environment not initialised. Call /reset first.")
        return StateResponse(
            task_id=self.task.task_id,
            step_count=self.step_count,
            total_reward=round(self.total_reward, 4),
            done=self.done,
            broken_query=self.task.broken_query,
            max_steps=MAX_STEPS,
        )

    def close(self) -> None:
        """Release the in-memory database."""
        if self.conn:
            self.conn.close()
            self.conn = None


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SQL Repair Environment",
    description="DBOps Agent — Meta OpenEnv Hackathon",
    version="1.0.0",
)

# Global environment instance (single-session; production would use per-session state)
_env = SQLRepairEnvironment()


class ResetRequest(BaseModel):
    task_id: str = Field(
        default="easy_syntax_repair",
        description="Task to load. One of: easy_syntax_repair, medium_join_repair, hard_window_function",
    )


@app.post("/reset", response_model=ResetResponse)
def api_reset(req: ResetRequest = ResetRequest()) -> ResetResponse:
    """Start a new episode. Reseeds the database and loads the specified task."""
    try:
        return _env.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResponse)
def api_step(action: SQLAction) -> StepResponse:
    """Execute one agent action and receive the next observation."""
    try:
        return _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=StateResponse)
def api_state() -> StateResponse:
    """Return the current episode state without advancing it."""
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "env": "sql-repair-env"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)