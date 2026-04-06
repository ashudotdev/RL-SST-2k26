from env.learning_env import AdaptiveLearnerEnv

def make_easy_task():
    return AdaptiveLearnerEnv(
        curriculum_file='data/curricula/easy_math.json',
        max_steps=50
    )
