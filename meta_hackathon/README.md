# 🛠️ SQL Repair Environment — Meta OpenEnv Hackathon

**A DBOps Agent benchmark environment where an AI agent acts as a Database Reliability Engineer.**

Given a broken SQL query and an in-memory SQLite database, the agent must inspect the schema, iteratively test fixes, and submit a repaired query that produces the exact target output.

---

## 📐 Environment Specification

| Property  | Value                          |
| --------- | ------------------------------ |
| Framework | FastAPI + SQLite (`:memory:`)  |
| Port      | `7860`                         |
| Max steps | `20` per episode               |
| Tags      | `openenv`, `sql`, `agent-eval` |

---

## 🎯 Tasks

### Easy — `easy_syntax_repair`

**Goal:** Fix two typos in a `SELECT` statement.

- `user` → `users` (table name)
- `ODER BY` → `ORDER BY` (keyword)

**Broken query:**

```sql
SELECT name, signup_date FROM user WHERE signup_date > '2026-01-01' ODER BY name
```

### Medium — `medium_join_repair`

**Goal:** Fix lifetime revenue calculation to include users with zero purchases.

- `INNER JOIN` → `LEFT JOIN`
- Add `COALESCE(SUM(...), 0)` to handle NULLs

### Hard — `hard_window_function`

**Goal:** Find the top-selling product category per month using window functions.

- Write a CTE to aggregate monthly revenue per category
- Apply `RANK() OVER (PARTITION BY month ORDER BY revenue DESC)`
- Filter `WHERE rnk = 1`

---

## 🔌 API Endpoints

### `POST /reset`

Start a new episode.

**Request body:**

```json
{ "task_id": "easy_syntax_repair" }
```

Valid `task_id` values: `easy_syntax_repair`, `medium_join_repair`, `hard_window_function`

**Response:** Initial observation + broken query to fix.

---

### `POST /step`

Submit an agent action.

**Request body:**

```json
{
  "action_type": "TEST_QUERY",
  "sql_command": "SELECT name FROM users ORDER BY name"
}
```

**`action_type` options:**
| Action | Description | `sql_command` |
|---|---|---|
| `INSPECT_SCHEMA` | Get table DDL | Not required |
| `TEST_QUERY` | Run a candidate fix | Required |
| `SUBMIT_FINAL_QUERY` | Submit final answer (ends episode) | Required |

---

### `GET /state`

Returns current episode state (no side effects).

---

## 📊 Observation Space (`SQLObservation`)

| Field                   | Type             | Description                                            |
| ----------------------- | ---------------- | ------------------------------------------------------ |
| `current_schema`        | `string`         | DDL of all tables (populated on `INSPECT_SCHEMA`)      |
| `target_output_preview` | `string`         | Description + preview of expected output               |
| `last_execution_status` | `string`         | `SUCCESS`, `SYNTAX_ERROR`, `LOGIC_ERROR`, or `PENDING` |
| `last_query_result`     | `string \| null` | JSON-encoded rows (up to 20) or error trace            |
| `step_reward`           | `float`          | Reward for the last action                             |

---

## 🎮 Action Space (`SQLAction`)

| Field         | Type             | Description                                                         |
| ------------- | ---------------- | ------------------------------------------------------------------- |
| `action_type` | `string`         | `INSPECT_SCHEMA`, `TEST_QUERY`, or `SUBMIT_FINAL_QUERY`             |
| `sql_command` | `string \| null` | SQL to execute (required for `TEST_QUERY` and `SUBMIT_FINAL_QUERY`) |

---

## 💰 Reward Structure

| Event                         | Reward                         |
| ----------------------------- | ------------------------------ |
| Per step (efficiency penalty) | `-0.01`                        |
| `INSPECT_SCHEMA`              | `+0.05`                        |
| `TEST_QUERY` → SQLite error   | `-0.10`                        |
| `TEST_QUERY` → executes OK    | `+0.10`                        |
| `SUBMIT_FINAL_QUERY`          | `grader_score` (`0.0` – `1.0`) |

---

## 📈 Grader Rubrics

### Easy

| Result                      | Score |
| --------------------------- | ----- |
| SQLite exception            | `0.0` |
| Executes but wrong data     | `0.5` |
| Exact match to ground truth | `1.0` |

### Medium

| Result                                 | Score |
| -------------------------------------- | ----- |
| SQLite exception                       | `0.0` |
| Executes but excludes 0-purchase users | `0.3` |
| All users present, wrong revenue math  | `0.7` |
| Exact match                            | `1.0` |

### Hard

| Result                                            | Score |
| ------------------------------------------------- | ----- |
| SQLite exception                                  | `0.0` |
| Groups by month only                              | `0.4` |
| Returns all categories per month (no rank filter) | `0.8` |
| Exact match (one winner per month)                | `1.0` |

---

## 🖥️ Running Locally

```bash
# Build
docker build -t sql-repair-env .

# Run server
docker run -p 7860:7860 sql-repair-env

# Run baseline agent (new terminal)
pip install openai requests
API_BASE_URL=http://localhost:7860 MODEL_NAME=gpt-4o HF_TOKEN=<your_key> python inference.py
```

**Expected output format:**
