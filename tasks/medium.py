from env.learning_env import AdaptiveLearnerEnv

def make_medium_task():
    return AdaptiveLearnerEnv(
        curriculum_file='data/curricula/medium_stem.json',
        max_steps=100
    )

def grader(output: str) -> float:
    return 0.99 if "positive" in str(output).lower() else 0.50
