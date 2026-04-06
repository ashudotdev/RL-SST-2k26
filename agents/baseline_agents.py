import numpy as np

class BaseAgent:
    def __init__(self, n_concepts: int):
        self.n_concepts = n_concepts
        self.noop_action = n_concepts
        
    def _parse_obs(self, obs: np.ndarray):
        mastery = obs[0:self.n_concepts]
        prereqs = obs[self.n_concepts:2*self.n_concepts]
        return mastery, prereqs

class RandomAgent(BaseAgent):
    def select_action(self, obs: np.ndarray):
        mastery, prereqs = self._parse_obs(obs)
        learnable = np.where(mastery < 0.95)[0]
        if len(learnable) == 0:
            return self.noop_action
        return np.random.choice(learnable)

class GreedyAgent(BaseAgent):
    def select_action(self, obs: np.ndarray):
        mastery, prereqs = self._parse_obs(obs)
        scores = []
        for i in range(self.n_concepts):
            if mastery[i] < 0.95 and prereqs[i]:
                scores.append(1.0 - mastery[i])
            else:
                scores.append(-1.0)
        
        best = np.argmax(scores)
        if scores[best] == -1.0:
            return self.noop_action
        return best

class TopologyAgent(BaseAgent):
    def select_action(self, obs: np.ndarray):
        mastery, prereqs = self._parse_obs(obs)
        scores = []
        for i in range(self.n_concepts):
            if mastery[i] < 0.95 and prereqs[i]:
                # simplistic approximation of topology awareness
                # values lower indices (which are typically prerequisites for later ones)
                weight = 1.0 + (self.n_concepts - i) * 0.1
                scores.append((1.0 - mastery[i]) * weight)
            else:
                scores.append(-1.0)
                
        best = np.argmax(scores)
        if scores[best] == -1.0:
            return self.noop_action
        return best
