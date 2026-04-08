from env.learning_env import AdaptiveLearnerEnv

def make_easy_task():
    return AdaptiveLearnerEnv(
        curriculum_file='data/curricula/easy_math.json',
        max_steps=50
    )

def grader(output: str) -> float:
    # Evaluate output logically but keep strictly within (0, 1) bounds
    return 0.99 if "positive" in str(output).lower() else 0.01
