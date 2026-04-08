"""
Inference Script — AdaptiveLearner-v0
=====================================
MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=AdaptiveLearner-v0 model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import numpy as np
from openai import OpenAI

from tasks.easy_task import make_easy_task
from tasks.medium_task import make_medium_task
from tasks.hard_task import make_hard_task
from tasks.graders import grade_easy, grade_medium, grade_hard

# ── Config ────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if HF_TOKEN is None:
    print("[WARNING] HF_TOKEN environment variable is not set. LLM inference will fail.")

BENCHMARK = "AdaptiveLearner-v0"

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)


def _build_prompt(obs_list, n_concepts):
    """Build a concise prompt for the LLM agent."""
    return (
        f"You are an RL agent optimizing an adaptive learning path.\n"
        f"Observation (flattened array): {obs_list}\n"
        f"The first {n_concepts} values are mastery levels (0-1) for each concept.\n"
        f"The next {n_concepts} values are prerequisite flags (1=met, 0=not met).\n"
        f"Action space: integers 0 to {n_concepts}.\n"
        f"  0-{n_concepts - 1}: teach that concept\n"
        f"  {n_concepts}: NO-OP (student rests)\n"
        f"Pick the best action. Respond with ONLY a single integer."
    )


def run_task(task_name, env, grader_fn):
    """Run a single task and emit structured stdout logs."""
    obs, info = env.reset()
    n = env.n_concepts

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    done = False
    step_num = 0
    rewards_list = []

    while not done:
        obs_list = obs.tolist()

        # ── Decide action ─────────────────────────────────────────────
        prompt = _build_prompt(obs_list, n)
        error = "null"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )
            action_str = response.choices[0].message.content.strip()
            action = int("".join(filter(str.isdigit, action_str)))
        except Exception as e:
            action = n  # fallback to NO-OP on LLM error
            error = str(e).replace("\n", " ")

        # Bounds check
        action = min(max(0, action), n)

        # ── Step ──────────────────────────────────────────────────────
        try:
            obs, reward, done, trunc, info = env.step(action)
            done = done or trunc
        except Exception as e:
            reward = 0.0
            done = True
            error = str(e).replace("\n", " ")

        step_num += 1
        rewards_list.append(reward)

        done_str = "true" if done else "false"
        print(
            f"[STEP] step={step_num} action={action} "
            f"reward={reward:.2f} done={done_str} error={error}"
        )

    # ── Score via grader ──────────────────────────────────────────────
    actions_taken = list(range(step_num))  # placeholder list of length step_num
    score = grader_fn(env, actions_taken, rewards_list)

    success_str = "true" if score > 0.0 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    
    # Exact strict format: [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
    print(
        f"[END] success={success_str} steps={step_num} "
        f"rewards={rewards_str}"
    )

    return score


def main():
    # ── Task registry ─────────────────────────────────────────────────
    tasks = [
        ("easy", make_easy_task, grade_easy),
        ("medium", make_medium_task, grade_medium),
        ("hard", make_hard_task, grade_hard),
    ]

    for task_name, make_fn, grader_fn in tasks:
        env = make_fn()
        run_task(task_name, env, grader_fn)


if __name__ == "__main__":
    main()
