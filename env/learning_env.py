import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .student_simulator import StudentSimulator
from .curriculum_loader import CurriculumLoader
from .reward_functions import compute_reward

class AdaptiveLearnerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, curriculum_file: str, max_steps: int = 100):
        super().__init__()
        self.curriculum = CurriculumLoader(curriculum_file)
        self.n_concepts = self.curriculum.n_concepts
        self.max_steps = max_steps
        
        self.student_model = StudentSimulator()
        
        # Action space: 0 to n_concepts (0 = NO-OP, 1 to n = teach concept_id)
        # Wait, the plan says 0 is NO-OP and 1-n: teach concept_id
        # BUT the code in the plan says:
        # action ∈ {0, 1, 2, ..., n_concepts}
        # and step() function: `if action < 0 or action >= self.n_concepts: raise ValueError` and uses `self.mastery_levels[action]` directly!
        # So action space is actually just 0 to n_concepts-1 for concepts, and n_concepts for NO-OP.
        # Let's align with the `step()` function code in the plan, where action 0 to n_concepts-1 are concepts,
        # but wait, it doesn't mention NO-OP handling except "if action == n_concepts: no-op".
        
        self.action_space = spaces.Discrete(self.n_concepts + 1)
        
        # Observation space shape
        # mastery_levels: n_concepts
        # prerequisites_met: n_concepts
        # time_invested: n_concepts
        # days_since_review: n_concepts
        # student_info: 10
        obs_shape = self.n_concepts * 4 + 10
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )
        
        # Internal state
        self.student_ability = 0.5
        self.mastery_levels = np.zeros(self.n_concepts)
        self.time_invested = np.zeros(self.n_concepts)
        self.days_since_review = np.zeros(self.n_concepts)
        self.step_count = 0
        self.episode_reward = 0.0
        self.fatigue = 0.0
        self.engagement = 1.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.student_ability = np.random.uniform(0.3, 0.9)
        self.mastery_levels = np.zeros(self.n_concepts)
        self.time_invested = np.zeros(self.n_concepts)
        self.days_since_review = np.zeros(self.n_concepts)
        
        self.step_count = 0
        self.episode_reward = 0.0
        self.fatigue = 0.0
        self.engagement = 1.0
        
        obs_dict = self.state()
        return self._flatten_obs(obs_dict), {}
        
    def _forgetting_curve_decay(self) -> np.ndarray:
        decayed = np.zeros(self.n_concepts)
        for i in range(self.n_concepts):
            decayed[i] = self.student_model.forget(self.mastery_levels[i], self.days_since_review[i])
        return decayed

    def _check_prerequisites(self, action: int) -> bool:
        prereqs = self.curriculum.get_prerequisites(action)
        mastery_with_decay = self._forgetting_curve_decay()
        for p in prereqs:
            if mastery_with_decay[p] < 0.8: # Threshold to consider 'mastered' enough
                return False
        return True

    def _is_curriculum_complete(self) -> bool:
        mastery_with_decay = self._forgetting_curve_decay()
        return np.all(mastery_with_decay >= 0.95)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # NO-OP action
        if action == self.n_concepts:
            # Student rests
            self.fatigue = max(0.0, self.fatigue - 0.2)
            self.engagement = min(1.0, self.engagement + 0.1)
            self.days_since_review += 1
            self.mastery_levels = self._forgetting_curve_decay()
            
            self.step_count += 1
            done = self.step_count >= self.max_steps or self._is_curriculum_complete()
            obs_dict = self.state()
            return self._flatten_obs(obs_dict), 0.0, done, False, {"step": self.step_count}

        if action < 0 or action > self.n_concepts:
            raise ValueError(f"Invalid action: {action}")
            
        prerequisites_met = self._check_prerequisites(action)
        concept_difficulties = self.curriculum.get_difficulty_array()
        
        old_mastery = self.mastery_levels[action]
        new_mastery, time_spent, engagement = self.student_model.learn(
            current_mastery=old_mastery,
            lesson_duration=30,
            lesson_difficulty=concept_difficulties[action],
            student_ability=self.student_ability,
            prerequisite_met=prerequisites_met,
        )
        
        self.mastery_levels[action] = new_mastery
        self.time_invested[action] += time_spent
        self.days_since_review[action] = 0
        self.fatigue = min(1.0, self.fatigue + 0.05)
        self.engagement = engagement
        
        mean_mastery = np.mean(self.mastery_levels)
        concept_newly_enabled = False # simplification (would track previous eligible state)
        
        reward = compute_reward(
            old_mastery, new_mastery, time_spent,
            prerequisites_met, concept_newly_enabled,
            self.engagement, self.fatigue, self.step_count,
            mean_mastery
        )
        
        done = self.step_count >= self.max_steps or self._is_curriculum_complete()
        
        self.step_count += 1
        self.episode_reward += reward
        
        self.days_since_review += 1
        self.mastery_levels = self._forgetting_curve_decay()
        
        info = {
            'step': self.step_count,
            'total_mastery': float(np.mean(self.mastery_levels)),
            'episode_reward': self.episode_reward,
            'concept_fully_learned': int(new_mastery >= 1.0),
            'prerequisites_met': prerequisites_met,
            'student_ability': self.student_ability,
        }
        
        obs_dict = self.state()
        return self._flatten_obs(obs_dict), reward, done, False, info

    def state(self) -> Dict:
        mastery_with_decay = self._forgetting_curve_decay()
        return {
            'mastery_levels': mastery_with_decay.copy(),
            'prerequisites_met': np.array([
                self._check_prerequisites(i) for i in range(self.n_concepts)
            ]),
            'time_invested': self.time_invested.copy(),
            'days_since_review': self.days_since_review.copy(),
            'concept_difficulty': self.curriculum.get_difficulty_array(),
            'student_ability': self.student_ability,
            'current_engagement': self.engagement,
            'fatigue_level': self.fatigue,
            'total_study_time': float(np.sum(self.time_invested) / 60),
            'concepts_mastered': int(np.sum(mastery_with_decay >= 0.95)),
            'episode_progress': float(np.mean(mastery_with_decay)),
            'episode_reward': self.episode_reward,
            'step': self.step_count,
            'max_steps': self.max_steps,
        }
        
    def _flatten_obs(self, obs_dict: Dict) -> np.ndarray:
        parts = [
            obs_dict['mastery_levels'],
            obs_dict['prerequisites_met'],
            obs_dict['time_invested'],
            obs_dict['days_since_review'],
            np.array([
                obs_dict['student_ability'],
                obs_dict['current_engagement'],
                obs_dict['fatigue_level'],
                obs_dict['total_study_time'],
                obs_dict['concepts_mastered'],
                obs_dict['episode_progress'],
                obs_dict['episode_reward'],
                obs_dict['step'],
                obs_dict['max_steps'],
                0.0 # padding to make it 10 stats
            ])
        ]
        return np.concatenate(parts).astype(np.float32)
