"""
Graders for each task.
Each grader runs the environment to completion, computes a normalized score in [0.0, 1.0].
"""
import numpy as np
from env.learning_env import AdaptiveLearnerEnv


def grade_episode(env: AdaptiveLearnerEnv, actions: list[int], rewards: list[float]) -> float:
    """
    Compute a normalized score in [0.0, 1.0] from a completed episode.
    
    Score components:
      - Mean mastery (weighted 60%)
      - Reward-based score (weighted 30%)
      - Efficiency bonus (weighted 10%)
    """
    state = env.state()
    mastery_levels = state['mastery_levels']
    mean_mastery = float(np.mean(mastery_levels))
    
    total_reward = sum(rewards)
    # Normalize reward: map from [-max_steps, max_steps] to [0, 1]
    reward_score = max(0.0, min(1.0, (total_reward + env.max_steps) / (2 * env.max_steps)))
    
    # Efficiency: fewer steps to reach mastery is better
    steps_used = len(actions)
    efficiency = max(0.0, 1.0 - (steps_used / env.max_steps)) if env.max_steps > 0 else 0.0
    
    score = (mean_mastery * 0.6) + (reward_score * 0.3) + (efficiency * 0.1)
    # The hackathon validator requires scores strictly between (0, 1) exclusive.
    return max(0.001, min(0.999, score))


def grade_easy(env: AdaptiveLearnerEnv, actions: list[int], rewards: list[float]) -> float:
    """Grader for the Easy task (5 concepts, 50 steps)."""
    return grade_episode(env, actions, rewards)


def grade_medium(env: AdaptiveLearnerEnv, actions: list[int], rewards: list[float]) -> float:
    """Grader for the Medium task (15 concepts, 100 steps)."""
    return grade_episode(env, actions, rewards)


def grade_hard(env: AdaptiveLearnerEnv, actions: list[int], rewards: list[float]) -> float:
    """Grader for the Hard task (35 concepts, 150 steps)."""
    return grade_episode(env, actions, rewards)
