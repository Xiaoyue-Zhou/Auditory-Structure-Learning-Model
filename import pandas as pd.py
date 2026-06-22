import pandas as pd
import numpy as np
from dataclasses import dataclass

# =======================
# Utilities
# =======================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def row_normalize(matrix, eps=1e-12):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / (row_sums + eps)


# =======================
# Experimental graph
# =======================

def build_exp_graph():
    n_node = 11
    community = np.array([0, 0, 0, 0, 0, 
                          1, 1, 1, 1, 1, 1])
    
    node_degree = np.array([4, 3, 3, 4, 4, 
                            4, 3, 3, 4, 4, 4])
    
    edges = [
        # community 1
        (0, 1), (0, 2), (0, 3), (0, 10),
        (1, 0), (1, 3), (1, 4),
        (2, 0), (2, 3), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 4),
        (4, 1), (4, 2), (4, 3), (4, 5),

        # community 2   
        (5, 4), (5, 6), (5, 8), (5, 9),
        (6, 5), (6, 7), (6, 9),
        (7, 6), (7, 8), (7, 10),
        (8, 5), (8, 7), (8, 9), (8, 10),
        (9, 5), (9, 6), (9, 8), (9, 10),
        (10, 0), (10, 7), (10, 8), (10, 9)
    ]

    adjacency = np.zeros((n_node, n_node))
    for i, j in edges:
        adjacency[i, j] = True

    return adjacency, community, node_degree

def random_walk(adjacency, length, rng):
    # rng - np.random.default_rng(seed)

    n_nodes = adjacency.shape[0]
    current_node = rng.integers(n_nodes)
    seq = [current_node]

    for _ in range(length - 1):
        neighbors = np.flatnonzero(adjacency[current_node])
        current_node = rng.choice(neighbors)
        seq.append(current_node)
    
    return np.array(seq, dtype=int)

def make_trials(adjacency, community, rng):
    # refer to MATLAB codes for detail
    # TBC ------------------------

    # return trial_seq
    pass


# =======================
# Model parameters
# =======================
@dataclass
class ModelParameters:

    beta: float = 0.06
    lr: float = 1.0
    reset_current_activity: bool = True

    decision_gain: float = 20.0
    criterion_offset = 0.0

    @property
    def trace_decay(self):
        return np.exp(-self.beta)


# =======================
# Associative memory model
# =======================
class HebbianSequenceModel:
    def __init__(self, n_nodes, params):
        self.n_nodes = n_nodes
        self.params = params

        self.W = np.zeros(
            (n_nodes, n_nodes),
            dtype=float
        )

        self.exposure_seq = None
    
    def learn_exposure(self, sequence):
        sequence = np.asarray(sequence, dtype=int)

        if sequence.ndim != 1:
            raise ValueError("Exposure sequence must be a 1-d array.")
        
        # store exposure sequence
        self.exposure_seq = sequence.copy()

        trace = np.zeros(self.n_nodes, dtype=float)

        trace(sequence[0]) = 1
        for current in sequence[1:]:
            # trace decay exponentially
            trace *= self.params.trace_decay

            # strengthening of association is positively associated with trace strength
            self.W[:,current] += (self.params.lr * trace)

            # activate the current tone
            if self.params.reset_current_activity:
                trace[current] = 1.0
            else:
                trace[current] += 1.0
        
        return self
    
    def association_matrix(self, normalize=True):
        '''
        1. remove diagnal of weight matrix
        2. row normalization
        '''

        matrix = self.W.copy()
        np.fill_diagonal(matrix, 0.0)

        if normalize:
            matrix = row_normalize(matrix)
        
        return matrix
    
    def get_familiarity(self, cue, target, normalize=True):
        matrix = self.association_matrix(normalize=normalize)
        return float(matrix[cue, target])
    
    def decision_evidence(self, cue, target):
        familiarity = self.get_familiarity(cue, 
                                           target, 
                                           normalize=True)
        uniform_baseline = 1.0 / (self.n_nodes - 1)
        evidence = familiarity - uniform_baseline - self.params.criterion_offset

        return evidence
    
    def choice_probability(self, cue, target):
        evidence = self.decision_evidence(cue, target)
        p_yes = sigmoid(evidence * self.params.decision_gain)

        return float(p_yes)
    
    def simulate_choice(self, cue, target, rng):
        p_yes = self.choice_probability(cue, target)
        yes = int(rng.random() < p_yes)
        return{'p_yes': p_yes,
                'yes': yes,
        }
    
    def copy(self):
        copied_model = HebbianSequenceModel(
            n_nodes=self.n_nodes,
            params=self.params,
        )

        copied_model.W = self.W.copy()

        if self.exposure_seq is not None:
            copied_model.exposure_seq = (
                self.exposure_seq.copy()
            )

        return copied_model
    
    def consolidation():
        pass

def evaluate_model(model, 
                   trial, 
                   adjacency, 
                   rng):
    # TBC
    pass


# ==============================
# Day 1 virtual experiment
# ==============================
time_ms = 6 * 60 * 1000
iti_ms = 80
tone_ms = 100
exposure_length = time_ms // (iti_ms + tone_ms)

def Exp_simulation_day1(n_subj=29,
                        exposure_length=exposure_length,
                        beta=0.06):
    
    seeds = range(n_subj)
    adjacency, community, node_degree = build_exp_graph()
    params = ModelParameters(beta=beta)

    all_data = []
    models = []

    for iSubj in range(n_subj):
        # different seed for each subject
        rng = np.random.default_rng(str(1+seeds[iSubj]))

        trial = make_trials(adjacency, community, rng)
        exposure_seq = random_walk(adjacency, 
                                   exposure_length, 
                                   rng)
        
        mdl_day1 = HebbianSequenceModel(
            n_nodes=adjacency.shape[0],
            params=params
        )

        mdl_day1.learn_exposure(exposure_seq)
        
        # 先假设test model的时候没有更新权重
        cur_data = evaluate_model(mdl_day1, trial)

        all_data.append(cur_data)
        models.append(mdl_day1)