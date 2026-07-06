import numpy as np
import pandas as pd


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

def random_walk(adjacency, length, rng, start_node=-1):
    n_nodes = adjacency.shape[0]

    if start_node >= 0:
        current_node = start_node
    else:
        current_node = rng.integers(n_nodes)
    seq = [current_node]

    for _ in range(length - 1):
        neighbors = np.flatnonzero(adjacency[current_node])
        current_node = rng.choice(neighbors)
        seq.append(current_node)
    
    return np.array(seq, dtype=int)

def make_trials(adjacency,
                community,
                rng,
                n_block=10,
                n_tone_per_trial=10):
    '''
    adjacency: np.array [n_node, n_node]
    community: np.array [n_node, 1]
    '''

    n_nodes = adjacency.shape[0]

    # remove possible self-transitions
    adjacency = adjacency.copy()
    np.fill_diagonal(adjacency, 0.0)

    # emulate all posible direct final transitions
    cue, target = np.where(
        ~np.eye(n_nodes, dtype=bool)
    )

    pairs = np.column_stack([cue, target])
    is_legal = adjacency[cue, target] > 0
    is_within = (community[cue] == community[target])

    within_legal = pairs[is_legal & is_within]       # 36
    between_legal = pairs[is_legal & ~is_within]     # 4
    within_illegal = pairs[~is_legal & is_within]    # 14
    between_illegal = pairs[~is_legal & ~is_within]  # 26 out of 56

    # each block - n(legal) == n(illegal)
    n_legal = within_legal.shape[0] + between_legal.shape[0]
    n_between_illegal = n_legal - within_illegal.shape[0]

    # construct each block
    block_tables = []
    for iBlc in range(n_block):
        selected_between_illegal = (
            between_illegal[
                rng.choice(between_illegal.shape[0], 
                           size=n_between_illegal,
                           replace=False)
            ]
        )

        final_pairs = np.vstack(
            [
                within_legal,
                between_legal,
                within_illegal,
                selected_between_illegal
            ]
        )

        # randomize trial order within the block
        final_pairs = final_pairs[rng.permutation(final_pairs.shape[0])]
        final_targets = final_pairs[:, 1]
        final_cue = final_pairs[:, 0]

        # generate first 9 tones & flip
        n_trial_per_block = final_pairs.shape[0]
        prefix = np.empty((n_trial_per_block, n_tone_per_trial-1), dtype=int)
        for iRow in range(n_trial_per_block):
            reversed = random_walk(adjacency, 
                                   length=n_tone_per_trial-1, 
                                   rng=rng, 
                                   start_node=final_cue[iRow])
            prefix[iRow, :] = reversed[::-1]

        sequence = np.column_stack([prefix, final_targets])
        legal = adjacency[final_cue, final_targets] > 0
        within = community[final_cue] == community[final_targets]

        membership_label = np.where(within, 'within', 'between')
        legal_label = np.where(legal, 'legal', 'illegal')
        trial_type = np.char.add(membership_label, '_')
        trial_type = np.char.add(trial_type, legal_label)

        regularity = np.where(within == legal, 'regular', 'irregular')

        block_tbl = pd.DataFrame(
            {
                "block": iBlc,
                "trial_in_block": np.arange(n_trial_per_block),
                "cue": final_cue,
                "target": final_targets,
                'legal': legal,
                "within": within,
                "trial_type": trial_type,
                "regularity": regularity
            }
        )
        block_tbl['sequence'] = list(sequence)
        block_tables.append(block_tbl)
    
    trials_all = pd.concat(block_tables, ignore_index=True)
    n_trial = trials_all.shape[0]
    trials_s1 = trials_all.iloc[: int(n_trial//2), :]
    trials_s2 = trials_all.iloc[int(n_trial//2):, :]


    return trials_s1, trials_s2
