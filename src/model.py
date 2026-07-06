from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.utils import sigmoid, row_normalize

@dataclass
class ModelParameters:

    beta: float = 0.06
    reset_current_activity: bool = True

    decision_gain: float = 20.0
    criterion_offset: float = 0.0

    lr: float = 1.0 # exposure learing rate
    lr_test_prefix:float = 1.0 # learning prefix of test sequences
    lr_test_probe:float = 0.1  # learning probe of test sequences

    @property
    def trace_decay(self):
        return np.exp(-self.beta)
    
class HebbianSequenceModel:
    def __init__(self, n_nodes, params):
        self.n_nodes = n_nodes
        self.params = params

        self.W = np.zeros(
            (n_nodes, n_nodes),
            dtype=float
        )

        self.exposure_seq = None

    def update_tone(self, trace, tone_curr, lr):
        tone_curr = int(tone_curr)

        # decay of trace
        trace *= self.params.trace_decay

        # update according to trace strength
        self.W[:,tone_curr] += (lr * trace)

        # activate current tone
        if self.params.reset_current_activity:
            trace[tone_curr] = 1.0
        else:
            trace[tone_curr] += 1.0

        return trace
    
    def learn_exposure(self, sequence):
        sequence = np.asarray(sequence, dtype=int)

        if sequence.ndim != 1:
            raise ValueError("Exposure sequence must be a 1-d array.")
        
        # store exposure sequence
        self.exposure_seq = sequence.copy()

        trace = np.zeros(self.n_nodes, dtype=float)

        trace[sequence[0]] = 1
        for current in sequence[1:]:
            trace = self.update_tone(
                trace=trace, 
                tone_curr=current,
                lr=self.params.lr)
        
        return self
    
    def run_test_phase(self, trials):
        # initialize output
        results = (
            trials
            .sort_values(['block', 'trial_in_block'], kind='stable')
            .reset_index(drop=True)
            .copy()
        )

        sequences = np.stack(
            results['sequence'].to_numpy()
        ).astype(int)

        n_trial, n_tone_per_trial = sequences.shape
        n_prefix = n_tone_per_trial - 1

        # initialize readout (familiarity)
        familiarity = np.empty(n_trial, dtype=float)
        decision_evidence = np.empty(n_trial, dtype=float)
        p_yes = np.empty(n_trial, dtype=float)

        lr_prefix = self.params.lr_test_prefix
        lr_probe  = self.params.lr_test_probe

        for iTrial in range(n_trial):
            seq_cur = sequences[iTrial,:]
            cue = seq_cur[-2]
            target = seq_cur[-1]

            # reset trace at the beginning of each trial
            trace = np.zeros(self.n_nodes, dtype=float)
            trace[seq_cur[0]] = 1.0
            for tone in seq_cur[1:n_prefix]:
                trace = self.update_tone(trace=trace,
                                         tone_curr=tone,
                                         lr = lr_prefix)
        
            # readout before learning the final tone
            ass_matrix = self.association_matrix(normalize=True)
            familiarity[iTrial] = ass_matrix[cue, target]
            uniform_baseline = 1 / (self.n_nodes - 1)
            decision_evidence[iTrial] = (
                familiarity[iTrial] 
                - uniform_baseline 
                - self.params.criterion_offset
             )
            p_yes[iTrial] = sigmoid(self.params.decision_gain * decision_evidence[iTrial])

            # learn the final tone
            trace = self.update_tone(trace=trace,
                                     tone_curr=target,
                                     lr = lr_probe)
            # add model readouts to the trial table
            results['familiarity'] = familiarity
            results['decision_evidence'] = decision_evidence
            results['p_yes'] = p_yes
            
            legal = results['legal'].to_numpy(dtype=bool) 
            results['expected_acc'] = np.where(legal, p_yes, 1-p_yes)
        return results
    
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
    
    def consolidation(self, conso_type):
        # conso_type is a string describing possible consolidation mechanism
        # learning rate should be different from online learning
        # beta sould be larger than online learning

        # global decay + replay with larger beta?
        if conso_type == 'veridical':
            pass
        elif conso_type == 'associate_generative':
            pass
        elif conso_type == 'selective':
            pass
        elif conso_type == 'weight-space':
            # sharpening / diffusion / competitive normalization
            pass
        pass

def evaluate_model(model, 
                   trials, 
                   adjacency):
    results = trials.copy()

    cues = results['cue'].to_numpy(dtype=int)
    targets = results['target'].to_numpy(dtype=int)

    # associative familiarity
    ass_matrix = model.association_matrix(normalize=True)
    familiarity = ass_matrix[cues, targets]

    # decision readout
    baseline_uniform = 1 / (model.n_nodes - 1)
    decision_evidence = familiarity - baseline_uniform - model.params.criterion_offset
    p_yes = sigmoid(model.params.decision_gain * decision_evidence)

    # expected accuracy
    expected_acc = np.where(results['legal'].to_numpy(dtype=bool), p_yes, 1-p_yes)
    
    results['familiarity'] = familiarity
    results['decision_evidence'] = decision_evidence
    results['p_yes'] = p_yes
    results['exp_acc'] = expected_acc
    
    return results
