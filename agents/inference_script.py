import numpy as np

def normalize_to_1_point_scale(episode_reward, mastery_levels):
    """
    Dummy normalizer to fit OpenEnv 0.0-1.0 score requirement.
    """
    mean_mastery = np.mean(mastery_levels)
    score = (mean_mastery * 0.5) + (episode_reward * 0.01) # simplistic conversion to 0-1
    return max(0.0, min(1.0, float(score)))

def grade_agent(agent, env, n_seeds=5):
    """
    Evaluate agent across multiple seeds and return score.
    """
    scores = []
    for seed in range(n_seeds):
        obs, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0.0
        
        info = {}
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            episode_reward += reward
        
        mastery = obs[0:env.n_concepts]
        normalized_score = normalize_to_1_point_scale(
            episode_reward, 
            mastery
        )
        scores.append(normalized_score)
    
    return {
        'mean_score': float(np.mean(scores)),
        'std_dev': float(np.std(scores)),
        'individual_scores': scores,
        'reproducible': True
    }

if __name__ == "__main__":
    import sys
    import os
    # Add parent to path to allow importing modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from tasks.easy_task import make_easy_task
    from agents.baseline_agents import RandomAgent, GreedyAgent, TopologyAgent

    env = make_easy_task()
    
    agents = {
        "Random": RandomAgent(env.n_concepts),
        "Greedy": GreedyAgent(env.n_concepts),
        "Topology": TopologyAgent(env.n_concepts)
    }

    print("Evaluating agents on EASY task...")
    for name, agent in agents.items():
        results = grade_agent(agent, env, n_seeds=3)
        print(f"{name} Agent: {results['mean_score']:.4f} / 1.0 (std: {results['std_dev']:.4f})")
