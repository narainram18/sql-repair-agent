"""
debug_hard.py
-------------
Shows exactly what SQL the model submits for the hard task.
Run this to diagnose why score is 0.40
"""
import json
import os
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860").rstrip("/")

# Step 1: Reset to hard task
r = requests.post(f"{API_BASE_URL}/reset", json={"task_id": "hard_window_function"})
data = r.json()
print("=== BROKEN QUERY ===")
print(data["broken_query"])
print()
print("=== TARGET OUTPUT ===")
print(data["observation"]["target_output_preview"])
print()

# Step 2: Run the CORRECT window function query directly
correct_sql = """WITH monthly AS (
    SELECT strftime('%Y-%m', o.order_date) AS month,
           p.category,
           SUM(p.price * o.quantity) AS revenue
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    GROUP BY month, p.category
),
ranked AS (
    SELECT month, category, revenue,
           RANK() OVER (PARTITION BY month ORDER BY revenue DESC) AS rnk
    FROM monthly
)
SELECT month, category, revenue
FROM ranked
WHERE rnk = 1
ORDER BY month"""

print("=== TESTING CORRECT SQL ===")
r2 = requests.post(f"{API_BASE_URL}/step", json={
    "action_type": "TEST_QUERY",
    "sql_command": correct_sql
})
d2 = r2.json()
print("Status:", d2["observation"]["last_execution_status"])
print("Result:", d2["observation"]["last_query_result"])
print("Reward:", d2["observation"]["step_reward"])
print()

# Step 3: Submit it
print("=== SUBMITTING CORRECT SQL ===")
r3 = requests.post(f"{API_BASE_URL}/reset", json={"task_id": "hard_window_function"})
r4 = requests.post(f"{API_BASE_URL}/step", json={
    "action_type": "SUBMIT_FINAL_QUERY",
    "sql_command": correct_sql
})
d4 = r4.json()
print("Score (step_reward):", d4["observation"]["step_reward"])
print("Done:", d4["done"])
print("Total reward:", d4["total_reward"])
