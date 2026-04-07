import numpy as np
from typing import Tuple

def learn(
    current_mastery: float,
    lesson_duration: int,      # minutes spent
    lesson_difficulty: float,  # 0.0=easy, 1.0=hard
    student_ability: float,    # 0.0=slow learner, 1.0=gifted
    prerequisite_met: bool,    # Are prerequisites mastered?
) -> Tuple[float, float, float]:
    """
    Returns:
        new_mastery: Updated mastery level
        time_spent: Actual time student spent
        engagement: Engagement score [0, 1]
    """
    
    # Diminishing returns on mastery
    remaining_to_learn = 1.0 - current_mastery
    
    # Learning rate depends on prerequisites & ability
    if not prerequisite_met:
        learning_rate = 0.01  # Very slow without prerequisites
    else:
        learning_rate = 0.1 * student_ability  # Fast if prerequisites met
    
    # Time scales with difficulty
    time_needed = lesson_difficulty * 30  # 0-30 minutes
    
    # Actually spent time varies
    time_spent = time_needed * (0.8 + 0.4 * np.random.random())
    
    # Learning follows power law: benefit = coefficient * time^exponent
    efficiency = learning_rate * (time_spent ** 0.8)
    new_mastery = min(1.0, current_mastery + efficiency * remaining_to_learn)
    
    # Engagement decreases with repetition on same topic
    engagement = max(0.3, 1.0 - (current_mastery * 0.5))
    
    return new_mastery, time_spent, engagement

def forget(mastery: float, days_since_reviewed: int) -> float:
    """
    Ebbinghaus forgetting curve: M(t) = M(0) * exp(-t/S)
    S = strength (scales with mastery)
    """
    strength = 10 + mastery * 50  # Stronger mastery = slower forgetting
    decay_rate = np.exp(-days_since_reviewed / strength)
    return mastery * decay_rate
