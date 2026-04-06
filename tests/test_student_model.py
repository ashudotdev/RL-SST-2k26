import pytest
from env.student_simulator import StudentSimulator

def test_learn_without_prerequisites():
    simulator = StudentSimulator()
    new_mastery, time_spent, engagement = simulator.learn(
        current_mastery=0.0,
        lesson_duration=30,
        lesson_difficulty=0.5,
        student_ability=0.8,
        prerequisite_met=False
    )
    # Learning should be very small
    assert new_mastery < 0.2
    assert engagement > 0

def test_learn_with_prerequisites():
    simulator = StudentSimulator()
    new_mastery, time_spent, engagement = simulator.learn(
        current_mastery=0.0,
        lesson_duration=30,
        lesson_difficulty=0.5,
        student_ability=0.8,
        prerequisite_met=True
    )
    # Learning should be higher
    assert new_mastery > 0.1
    assert time_spent > 0

def test_forgetting_curve():
    simulator = StudentSimulator()
    mastery_initial = 1.0
    mastery_after = simulator.forget(mastery_initial, days_since_reviewed=5)
    assert mastery_after < mastery_initial
