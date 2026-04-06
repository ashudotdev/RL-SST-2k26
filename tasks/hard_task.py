from env.learning_env import AdaptiveLearnerEnv

def make_hard_task():
    return AdaptiveLearnerEnv(
        curriculum_file='data/curricula/hard_full_curriculum.json',
        max_steps=150
    )
