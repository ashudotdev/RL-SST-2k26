import unittest
import numpy as np
from env.learning_env import AdaptiveLearnerEnv

class TestLearningEnv(unittest.TestCase):
    def test_env_initialization(self):
        env = AdaptiveLearnerEnv("data/curricula/easy_math.json", max_steps=50)
        self.assertEqual(env.n_concepts, 5)
        
    def test_env_reset(self):
        env = AdaptiveLearnerEnv("data/curricula/easy_math.json", max_steps=50)
        obs, info = env.reset()
        self.assertEqual(len(obs), 5 * 4 + 10)
        
    def test_env_step(self):
        env = AdaptiveLearnerEnv("data/curricula/easy_math.json", max_steps=50)
        env.reset()
        obs, reward, done, trunc, info = env.step(0) # action 0
        self.assertTrue(isinstance(reward, float))
        self.assertFalse(done)

if __name__ == '__main__':
    unittest.main()
