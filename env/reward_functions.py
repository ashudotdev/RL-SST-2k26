def compute_reward(
    old_mastery: float,
    new_mastery: float,
    time_spent: float,
    prerequisites_met: bool,
    concept_newly_enabled: bool,
    engagement: float,
    fatigue: float,
    step: int,
    mean_mastery: float
) -> float:
    """
    Compute multi-component reward normalized to [-1, 1].
    """
    # 1. Learning Gain Reward (+2.0 to +10.0 scaled down)
    learning_gain = new_mastery - old_mastery
    reward_learning = learning_gain * 10.0
    
    # 2. Efficiency Reward (+0 to +3.0)
    efficiency = learning_gain / (time_spent / 60.0) if time_spent > 0 else 0
    reward_efficiency = min(3.0, efficiency)
    
    # 3. Prerequisite Bonus (+1.0 if applicable)
    reward_prerequisite_bonus = 1.0 if (prerequisites_met and concept_newly_enabled) else 0.0
    
    # 4. Engagement Penalty (-0.5 to 0.0)
    reward_engagement = -0.5 if engagement < 0.3 else 0.0
    
    # 5. Fatigue Penalty (-0.0 to -1.0)
    reward_fatigue = -1.0 if fatigue > 0.8 else -0.1 * fatigue
    
    # 6. Curriculum Completion Bonus (0.0 to +5.0)
    reward_progress = mean_mastery * 5.0 if step % 10 == 0 else 0.0
    
    # Final weighted sum
    reward = (
        reward_learning * 0.4 +
        reward_efficiency * 0.25 +
        reward_prerequisite_bonus * 0.15 +
        reward_engagement * 0.1 +
        reward_fatigue * 0.05 +
        reward_progress * 0.05
    )
    
    # Clip to [-1.0, 1.0] range
    return max(-1.0, min(1.0, reward))
