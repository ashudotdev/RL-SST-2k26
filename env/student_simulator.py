import numpy as np
from typing import Tuple
from .learning_curves import learn, forget

class StudentSimulator:
    """
    Simulates a student's learning process.
    Acts as a wrapper around the mathematical formulas defined in learning_curves.py.
    """
    def __init__(self):
        pass

    def learn(
        self,
        current_mastery: float,
        lesson_duration: int,
        lesson_difficulty: float,
        student_ability: float,
        prerequisite_met: bool,
    ) -> Tuple[float, float, float]:
        """
        Simulate learning a concept.
        Returns:
            new_mastery, time_spent, engagement
        """
        return learn(
            current_mastery,
            lesson_duration,
            lesson_difficulty,
            student_ability,
            prerequisite_met
        )

    def forget(self, mastery: float, days_since_reviewed: int) -> float:
        """
        Simulate forgetting over time.
        Returns:
            new_mastery after forgetting
        """
        return forget(mastery, days_since_reviewed)
