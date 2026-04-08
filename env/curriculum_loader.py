import json
import networkx as nx
from typing import List, Dict, Any
import numpy as np

class CurriculumLoader:
    def __init__(self, curriculum_file: str):
        try:
            with open(curriculum_file, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"Error loading curriculum file {curriculum_file}: {e}")
            self.data = {"concepts": []}
            
        self.concepts = self.data.get('concepts', [])
        self.n_concepts = len(self.concepts)
        self.concept_names = [c['name'] for c in self.concepts]
        self.concept_difficulties = [c.get('difficulty', 0.5) for c in self.concepts]
        
        # Build DAG
        self.graph = nx.DiGraph()
        for i, concept in enumerate(self.concepts):
            self.graph.add_node(i, name=concept['name'])
            for prereq in concept.get('prerequisites', []):
                if prereq in self.concept_names:
                    prereq_idx = self.concept_names.index(prereq)
                    self.graph.add_edge(prereq_idx, i)
                
    def get_difficulty_array(self) -> np.ndarray:
        return np.array(self.concept_difficulties)

    def get_prerequisites(self, concept_idx: int) -> List[int]:
        return list(self.graph.predecessors(concept_idx))
        
    def get_sparse_representation(self) -> Dict[str, List[int]]:
        # convert integer node keys to string to be JSON compatible if needed
        return {str(k): v for k, v in nx.to_dict_of_lists(self.graph).items()}
