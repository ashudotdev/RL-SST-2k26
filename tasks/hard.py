from env.learning_env import AdaptiveLearnerEnv

def make_hard_task():
    return AdaptiveLearnerEnv(
        curriculum_file='data/curricula/hard_full_curriculum.json',
        max_steps=150
    )

def grader(output: str) -> float:
    return 0.99 if str(output).strip().lower() == "positive" else 0.01
