from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from src.utils import sigmoid, row_normalize

@dataclass
class ModelParameters:

    beta: float = 0.06
    reset_current_activity: bool = True

    decision_gain: float = 20.0
    criterion_offset: float = 0.0

    readout_mode: str = 'normalized'  # alternatives: 'contrast'

    lr: float = 1.0 # exposure learing rate
    lr_test_prefix:float = 1.0 # learning prefix of test sequences
    lr_test_probe:float = 0.1  # learning probe of test sequences

    @property
    def trace_decay(self):
        return np.exp(-self.beta)

@dataclass
class ConsolidationParameters:
    # currently implemented type
    conso_type: str = 'veridical'

    # global synaptic retention before replay
    global_retention: float = 0.9

    # number of replay episodes sample from the original exposure sequence
    n_replay: int = 100

    # length of each replay episode
    replay_length: int = 20

    beta_consolidation: float = 0.30
    lr_consolidation: float = 0.01
    normalize_lag1_strength: bool = True

    replay_temperature: float = 1.0

    # for prioritized replay
    priority_count_alpha: float = 0.5
    priority_surprisal_gamma: float = 1.0
    priority_eps: float = 1e-12


class HebbianSequenceModel:
    def __init__(self, n_nodes, params):
        self.n_nodes = n_nodes
        self.params = params

        self.W = np.zeros(
            (n_nodes, n_nodes),
            dtype=float
        )

        self.C = np.zeros((n_nodes, n_nodes), dtype=float) # transition count
        self.S = np.zeros((n_nodes, n_nodes), dtype=float) # surprisal sum

        self.exposure_seq = None

    def update_tone(self, trace, tone_curr, lr=None, trace_decay=None):
        tone_curr = int(tone_curr)

        if lr is None:
            lr = self.params.lr

        if trace_decay is None:
            trace_decay = self.params.trace_decay

        # decay of trace
        trace *= trace_decay

        # update according to trace strength
        self.W[:,tone_curr] += (lr * trace)

        # activate current tone
        if self.params.reset_current_activity:
            trace[tone_curr] = 1.0
        else:
            trace[tone_curr] += 1.0

        return trace
    
    def _predict_trans_prob(self, cue, target, eps=1e-12):
        cue = int(cue)
        target = int(target)
        row = self.W[cue,:].copy().astype(float)

        row[cue] = 0.0 # exclude self transition
        row[row<0.0] = 0.0 # numerical safety
        row_sum = row.sum()

        if row_sum < eps:
            if target == cue:
                return eps
            return 1.0 / (self.n_nodes - 1)
        
        p = row[target] / row_sum # normalization
        return float(np.clip(p, eps, 1.0)) # bounded within [0 1.0]

    def _record_trans_suprisal(self, cue, target, eps=1e-12):
        cue = int(cue)
        target = int(target)
        p_model = self._predict_trans_prob(cue, target, eps=eps)
        surprisal = -np.log(p_model + eps)
        
        self.C[cue, target] += 1.0
        self.S[cue, target] += surprisal
        
        return surprisal

    def learn_exposure(self, sequence):
        sequence = np.asarray(sequence, dtype=int)

        if sequence.ndim != 1:
            raise ValueError("Exposure sequence must be a 1-d array.")
        
        # store exposure sequence
        self.exposure_seq = sequence.copy()

        trace = np.zeros(self.n_nodes, dtype=float)
        trace[sequence[0]] = 1.0

        prev = int(sequence[0])

        for current in sequence[1:]:
            self._record_trans_suprisal(prev, current)
            trace = self.update_tone(
                trace=trace, 
                tone_curr=current,
                lr=self.params.lr)
            prev = current
        
        return self
    
    def transition_readout(self, cue, target):
        '''
        Compute both [1] representation level and [2] behavioural level readouts.
        familiarity_normalized:
            row-normalized association strength
            this is a representation-level readout
        familiarity_contrast:
            raw association minus row-average association
            this is a behavioural-level readout
            reflecting confidence (amount of availavble memory), which is influenced by global decay
        '''
        cue = int(cue)
        target = int(target)

        row_raw = self.W[cue,:].copy()
        row_raw[cue] = 0.0
        row_sum = row_raw.sum()
        uniform_baseline = 1.0 / (self.n_nodes - 1)

        # normalize familiarity
        if row_sum > 0:
            familiarity_normalized = row_raw[target] / row_sum
        else:
            familiarity_normalized = uniform_baseline
        
        # raw contrast familiarity
        row_mean = row_sum / (self.n_nodes - 1)
        familiarity_contrast = row_raw[target] - row_mean

        # choose behaviour evidence
        if self.params.readout_mode == 'normalized':
            familiarity = familiarity_normalized
            decision_evidence = familiarity - uniform_baseline - self.params.criterion_offset

        elif self.params.readout_mode == 'contrast':
            familiarity = familiarity_contrast
            decision_evidence = familiarity - self.params.criterion_offset
        else:
            raise ValueError(f"Unknown readout mode: {self.params.readout_mode}")
        
        p_yes = sigmoid(self.params.decision_gain * decision_evidence)
     
        return {
            "familiarity": familiarity,
            "familiarity_normalized": familiarity_normalized,
            "familiarity_contrast": familiarity_contrast,
            "decision_evidence": decision_evidence,
            "p_yes": p_yes,
            "row_strength": row_sum,
        }

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
        familiarity_normalized = np.empty(n_trial, dtype=float)
        familiarity_contrast = np.empty(n_trial, dtype=float)
        decision_evidence = np.empty(n_trial, dtype=float)
        p_yes = np.empty(n_trial, dtype=float)
        row_strength = np.empty(n_trial, dtype=float)

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
            readout = self.transition_readout(cue, target)
            familiarity[iTrial] = readout['familiarity']
            familiarity_normalized[iTrial] = readout['familiarity_normalized']
            familiarity_contrast[iTrial] = readout['familiarity_contrast']
            decision_evidence[iTrial] = readout['decision_evidence']
            p_yes[iTrial] = readout['p_yes']
            row_strength[iTrial] = readout['row_strength']

            # learn the final tone
            trace = self.update_tone(trace=trace,
                                     tone_curr=target,
                                     lr = lr_probe)
            
        # add model readouts to the trial table
        results['familiarity'] = familiarity
        results['familiarity_normalized'] = familiarity_normalized
        results['familiarity_contrast'] = familiarity_contrast
        results['decision_evidence'] = decision_evidence
        results['p_yes'] = p_yes
        results['row_strength'] = row_strength
        
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
        readout = self.transition_readout(cue, target)
        return float(readout['decision_evidence'])
    
    def choice_probability(self, cue, target):
        readout = self.transition_readout(cue, target)
        return float(readout['p_yes'])
    
    def simulate_choice(self, cue, target, rng):
        p_yes = self.choice_probability(cue, target)
        yes = int(rng.random() < p_yes)
        return{'p_yes': p_yes,
                'yes': yes,
        }
    
    def copy(self):
        copied_model = HebbianSequenceModel(
            n_nodes=self.n_nodes,
            params=replace(self.params),
        )

        copied_model.W = self.W.copy()

        if self.exposure_seq is not None:
            copied_model.exposure_seq = (
                self.exposure_seq.copy()
            )
        
        copied_model.C = self.C.copy()
        copied_model.S = self.S.copy()

        return copied_model
    
    def _replay_sequence(self, sequence, lr, trace_decay):
        sequence = np.asarray(sequence, dtype=int)
        trace = np.zeros(self.n_nodes, dtype=float)
        trace[sequence[0]] = 1.0

        for tone_curr in sequence[1:]:
            trace = self.update_tone(
                trace=trace,
                tone_curr=tone_curr,
                lr=lr,
                trace_decay=trace_decay,
            )
        return self
    
    def consolidate_veridical(self, rng, conso_params=None):
        '''
        Veridical replay consolidation
        Mechanism:
            [1] apply global decay to W
            [2] sample contiguous fragments from the stored exposre sequence
            [3] replay those fragments with a consolidation-specific beta and learning rate
        
        This replay is veridical as all replayed sequences are sampled from the actual exposure sequence, rather than generated from W.
        '''
        if conso_params is None:
            conso_params = ConsolidationParameters(conso_type='veridical')

        if self.exposure_seq is None:
            raise ValueError("No exposure sequence stored in the model. \n Run learn_exposure() first.")
        
        exposure_length = len(self.exposure_seq)
        if conso_params.replay_length > exposure_length:
            raise ValueError(f"Replay length ({conso_params.replay_length}) cannot exceed exposure sequence length ({exposure_length}).")
        

        self.W *= conso_params.global_retention

        trace_decay_conso = np.exp(-1*conso_params.beta_consolidation)

        if conso_params.normalize_lag1_strength:
            lr_effective = (conso_params.lr_consolidation / trace_decay_conso)
        else:
            lr_effective = conso_params.lr_consolidation
        
        max_start = exposure_length - conso_params.replay_length + 1
        
        for _ in range(conso_params.n_replay):
            start = rng.integers(0, max_start)
            replay_fragment = self.exposure_seq[start:start+conso_params.replay_length]
            self._replay_sequence(replay_fragment,
                                   lr=lr_effective,
                                   trace_decay=trace_decay_conso)
        return self

    # Generative replay helper functions
    def _transition_matrix_from_W(self, temperature=1):
        W_gen = self.W.copy().astype(float)
        
        # remove self transitions
        np.fill_diagonal(W_gen, 0.0)

        # keep only non-negative weights (though all weights should be positive)
        W_gen[W_gen<0] = 0.0

        # temperature transform
        if temperature - 1.0 > 1e-5:
            W_gen = np.power(W_gen, 1 / temperature)
        
        P = np.zeros_like(W_gen, dtype=float)
        for i in range(self.n_nodes):
            row_sum = W_gen[i].sum()
            if row_sum > 0:
                P[i] = W_gen[i] / row_sum
            else:
                P[i] = 1.0 / (self.n_nodes - 1)
                P[i,i] = 0.0
        return P
    
    def _sample_generative_sequence(self, trans_mat, rng, replay_length):
        current = int(rng.integers(self.n_nodes))
        sequence = np.empty(replay_length, dtype=int)

        sequence[0] = current
        for t in range(1, replay_length):
            probs = trans_mat[current].copy()
            probs = probs / probs.sum() # 其实已经是sum=1了，为了safty再做一次
            current = int(rng.choice(self.n_nodes, p=probs))

            sequence[t] = current
        
        return sequence

    def consolidate_generative(self, rng, conso_params=None):
        '''
        Generative replay consolidation
        Mechanism:
            [1] build a replay transition matrix from the learned W
            [2] global decay
            [3] generate replay sequences from the transition matrix
            [4] replay those generated sequences using the same Hebbian learning rule
        '''

        # build fixed generator from current learned W
        trans_mat = self._transition_matrix_from_W(temperature=conso_params.replay_temperature)

        # global decay
        self.W *= conso_params.global_retention

        # consolidation-specific trace decay
        trace_decay_conso = np.exp(-1 * conso_params.beta_consolidation)
        if conso_params.normalize_lag1_strength:
            lr_effective = conso_params.lr_consolidation / trace_decay_conso
        else:
            lr_effective = conso_params.lr_consolidation

        # generate & learn the sequences
        for _ in range(conso_params.n_replay):
            replay_fragment = self._sample_generative_sequence(trans_mat,
                                                               rng=rng,
                                                               replay_length=conso_params.replay_length)
            self._replay_sequence(replay_fragment,
                                   lr=lr_effective,
                                   trace_decay=trace_decay_conso)
        return self

    # selective (prioritized) replay helper functions
    def _priority_mat_from_surprisal(self, 
                                     count_alpha=0.5,
                                     surprisal_gamma=1.0,
                                     eps=1e-12):
        counts = self.C.copy()
        np.fill_diagonal(counts, 0.0)

        observed = counts > 0
        mean_surprisal = np.zeros_like(counts, dtype=float)
        mean_surprisal[observed] = self.S[observed] / np.maximum(self.C[observed], 1.0)

        priority = np.zeros_like(counts, dtype=float)
        priority[observed] = (
            np.power(counts[observed], 
                     count_alpha) 
            * np.power(mean_surprisal[observed] + eps,
                       surprisal_gamma)
        )
        np.fill_diagonal(priority, 0.0)

        return priority

    def _direct_trans_mat_from_C(self, eps=1e-12):
        counts = self.C.copy()
        np.fill_diagonal(counts, 0.0)
        
        P = np.zeros_like(counts, dtype=float)
        for i in range(self.n_nodes):
            row_sum = counts[i,:].sum()
            if row_sum > eps:
                P[i] = counts[i] / row_sum
            else:
                P[i] = 1.0 / (self.n_nodes - 1)
                P[i, i] = 0.0
        
        return P

    def _sample_edge_from_matrix(self, matrix, rng):
        flat = matrix.ravel().astype(float)
        total = flat.sum()
        probs = flat / total

        edge_idx = int(rng.choice(flat.size, p=probs))
        return divmod(edge_idx, self.n_nodes)

    def _sample_next_from_transmat(self, trans_mat, current, rng):
        current = int(current)
        probs = trans_mat[current].copy()
        probs_sum = probs.sum()

        probs = probs / probs_sum # normalization
        return int(rng.choice(self.n_nodes, p=probs))

    def _sample_prioritzed_sequence(self, priority_mat, trans_mat, rng, replay_length):
        # seed based prioritized replay
        seed_i, seed_j = self._sample_edge_from_matrix(priority_mat, rng)
        
        sequence = np.empty(replay_length, dtype=int)
        sequence[0] = seed_i
        sequence[1] = seed_j
        current = seed_j

        for t in range(2, replay_length):
            current = self._sample_next_from_transmat(trans_mat,
                                                      current,
                                                      rng)
            sequence[t] = current
            
        return sequence

    def consolidate_selective(self, rng, conso_params=None):
        '''
        Prioritzied replay consolidation

        Mechanism:
            [1] build priority matrix from exposure-time transition surprisal
            [2] build continuation transition matrix from experienced transition counts
            [3] apply global decay
            [4] generate replay sequences seeded by high-priority transitions
            [5] replay-learn the generated sequences with consolidation-specific beta & lr
        '''

        if conso_params is None:
            conso_params = ConsolidationParameters(conso_type='selective')
        
        priority_mat = self._priority_mat_from_surprisal(count_alpha=conso_params.priority_count_alpha,
                                                         surprisal_gamma=conso_params.priority_surprisal_gamma,
                                                         eps=conso_params.priority_eps)
        trans_mat = self._direct_trans_mat_from_C(eps=conso_params.priority_eps)

        # global decay
        self.W *= conso_params.global_retention

        # consolidation-specific trace decay
        trace_decay_conso = np.exp(-1 * conso_params.beta_consolidation)
        if conso_params.normalize_lag1_strength:
            lr_effective = conso_params.lr_consolidation / trace_decay_conso
        else:
            lr_effective = conso_params.lr_consolidation
        
        # generate & learn replay sequences
        for _ in range(conso_params.n_replay):
            fragment = self._sample_prioritzed_sequence(priority_mat,
                                                        trans_mat,
                                                        rng,
                                                        conso_params.replay_length)
            self._replay_sequence(fragment,
                                  lr=lr_effective,
                                  trace_decay=trace_decay_conso)
        
        return self

    def consolidation(self, conso_type, rng=None, conso_params=None):
        '''
        Dispatcher for consolidation mechanisms.
        '''
        if rng is None:
            rng = np.random.default_rng()

        if conso_params is None:
            conso_params = ConsolidationParameters(conso_type=conso_type)


        if conso_type == 'veridical':
            return self.consolidate_veridical(rng=rng, conso_params=conso_params)
        elif conso_type == 'generative':
            return self.consolidate_generative(rng=rng, conso_params=conso_params)
        elif conso_type == 'selective':
            return self.consolidate_selective(rng=rng, conso_params=conso_params)
        elif conso_type == 'weight-space':
            # sharpening / diffusion / competitive normalization
            raise NotImplementedError("This method is not implemented yet.")
        else:
            raise ValueError(f"Unknown consolidation type: {conso_type}")

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
