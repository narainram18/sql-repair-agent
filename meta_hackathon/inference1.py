"""
inference.py
------------
Baseline inference agent for the SQL Repair Environment.
Uses the OpenAI Python client to drive an LLM agent through the environment.

Required environment variables:
  API_BASE_URL  — Base URL of the running environment (e.g. http://localhost:7860)
  MODEL_NAME    — LLM model identifier (e.g. gpt-4o)
  HF_TOKEN      — Hugging Face token (used as OpenAI API key for routed models)

Stdout log format (strictly enforced):
  [START] task=<task_name> env=sql-repair-env model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
TASK_ID: str = os.environ.get("TASK_ID", "easy_syntax_repair")

# OpenAI client — HF_TOKEN is used as the API key when routing through HF Inference
openai_client = OpenAI(
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
)

MAX_AGENT_STEPS = 15  # Safety cap below env's MAX_STEPS=20

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------


def env_reset(task_id: str) -> dict[str, Any]:
    resp = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, sql_command: Optional[str] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"action_type": action_type}
    if sql_command is not None:
        payload["sql_command"] = sql_command
    resp = requests.post(f"{API_BASE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Agent Prompting Helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a Database Reliability Engineer (DRE) agent.
You are given a broken SQL query and must repair it to produce the correct target output.

You interact with a SQLite environment using three actions:
1. INSPECT_SCHEMA  — Retrieve the database table schemas (DDL). No sql_command needed.
2. TEST_QUERY      — Run a candidate SQL query and inspect the results. Provide sql_command.
3. SUBMIT_FINAL_QUERY — Submit your final repaired SQL. Provide sql_command. This ends the episode.

Strategy:
- Start with INSPECT_SCHEMA to understand the tables.
- Use TEST_QUERY to iteratively debug and fix the query and check the syntax too.
- When confident your query produces the correct output, use SUBMIT_FINAL_QUERY.
- Be efficient; every step has a small cost.

Respond ONLY with valid JSON in this exact format:
{"action_type": "...", "sql_command": "...or null"}
No explanation, no markdown, no extra text. Only the JSON object.
"""


def build_user_message(
    broken_query: str,
    target_preview: str,
    step_history: list[dict[str, Any]],
    last_obs: dict[str, Any],
) -> str:
    """Construct the user turn message from current environment state."""
    history_str = ""
    if step_history:
        lines = []
        for h in step_history[-6:]:  # last 6 steps to stay within context
            lines.append(
                f"  Step {h['step']}: action={h['action']} "
                f"status={h['status']} reward={h['reward']:.2f}"
            )
            if h.get("result_preview"):
                lines.append(f"    Result preview: {h['result_preview'][:300]}")
        history_str = "Recent step history:\n" + "\n".join(lines)

    schema_section = ""
    if last_obs.get("current_schema"):
        schema_section = f"\nDatabase Schema:\n{last_obs['current_schema']}\n"

    return f"""Broken query to fix:
{broken_query}

Target output description:
{target_preview}
{schema_section}
{history_str}

Last execution status: {last_obs.get('last_execution_status', 'PENDING')}
Last result: {str(last_obs.get('last_query_result', 'None'))[:400]}

What is your next action? Respond with JSON only."""


# ---------------------------------------------------------------------------
# Main Agent Loop
# ---------------------------------------------------------------------------


def run_agent(task_id: str = TASK_ID) -> None:
    # ── Reset environment ────────────────────────────────────────────────────
    reset_data = env_reset(task_id)
    broken_query: str = reset_data["broken_query"]
    last_obs: dict[str, Any] = reset_data["observation"]
    task_name: str = reset_data["task_id"]

    print(f"[START] task={task_name} env=sql-repair-env model={MODEL_NAME}")
    sys.stdout.flush()

    step_history: list[dict[str, Any]] = []
    all_rewards: list[float] = []
    step_num = 0
    final_score = 0.0
    success = False
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # ── Agent loop ───────────────────────────────────────────────────────────
    for _ in range(MAX_AGENT_STEPS):
        # Build the user message with full context
        user_content = build_user_message(
            broken_query=broken_query,
            target_preview=last_obs.get("target_output_preview", ""),
            step_history=step_history,
            last_obs=last_obs,
        )
        messages.append({"role": "user", "content": user_content})

        # Call the LLM
        error_msg: Optional[str] = None
        action_type_str = "UNKNOWN"
        step_reward = 0.0
        done = False

        try:
            completion = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=512,
                temperature=0.0,
            )
            raw_response = completion.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": raw_response})

            # Parse LLM JSON output
            # Strip any accidental markdown fences
            clean = raw_response.strip().strip("```json").strip("```").strip()
            action_obj = json.loads(clean)
            action_type_str = action_obj.get("action_type", "UNKNOWN").upper()
            sql_cmd: Optional[str] = action_obj.get("sql_command") or None

            # Execute action in environment
            step_data = env_step(action_type_str, sql_cmd)
            step_num = step_data["step_count"]
            last_obs = step_data["observation"]
            step_reward = last_obs["step_reward"]
            done = step_data["done"]

            all_rewards.append(step_reward)

            # Track history for context
            result_raw = last_obs.get("last_query_result") or ""
            step_history.append(
                {
                    "step": step_num,
                    "action": action_type_str,
                    "status": last_obs.get("last_execution_status", ""),
                    "reward": step_reward,
                    "result_preview": str(result_raw)[:200],
                }
            )

            # Determine final score on terminal step
            if done and action_type_str == "SUBMIT_FINAL_QUERY":
                # Score = reward on submit (grader output, 0.0-1.0)
                # The submit reward = grader score (minus step penalty already baked in)
                final_score = max(0.0, step_reward + 0.01)  # add back step penalty
                success = final_score >= 1.0

        except json.JSONDecodeError as exc:
            error_msg = f"JSON parse error: {exc}"
            all_rewards.append(0.0)
            step_num += 1
            done = False
        except requests.HTTPError as exc:
            error_msg = f"HTTP error: {exc}"
            all_rewards.append(0.0)
            step_num += 1
            done = False
        except Exception as exc:
            error_msg = f"Unexpected error: {exc}"
            all_rewards.append(0.0)
            step_num += 1
            done = False

        # ── Emit [STEP] log ──────────────────────────────────────────────────
        done_str = "true" if done else "false"
        error_str = error_msg if error_msg else "null"
        print(
            f"[STEP] step={step_num} action={action_type_str} "
            f"reward={step_reward:.2f} done={done_str} error={error_str}"
        )
        sys.stdout.flush()

        if done:
            break

    # ── Emit [END] log ───────────────────────────────────────────────────────
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    print(
        f"[END] success={success_str} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    run_agent()


    