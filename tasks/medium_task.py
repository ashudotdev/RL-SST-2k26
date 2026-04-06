from env.learning_env import AdaptiveLearnerEnv

def make_medium_task():
    return AdaptiveLearnerEnv(
        curriculum_file='data/curricula/medium_stem.json',
        max_steps=100
    )
