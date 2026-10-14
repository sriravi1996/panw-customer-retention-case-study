"""Parent bootstrap row assembly, preserving repeated-parent multiplicity."""
import numpy as np


def resample_parent_rows(df, sampled_parents, parent_indices):
    """Equivalent to concatenating one dataframe copy per parent occurrence.

    Gather indices first, then copy once. Preserve sample order, row order and
    distinct account identities for every occurrence of a repeated parent.
    """
    positions, occurrences, counter = [], [], {}
    for parent in sampled_parents:
        occ = counter.get(parent, 0)
        counter[parent] = occ + 1
        indices = parent_indices[parent]
        positions.extend(indices)
        occurrences.extend([occ] * len(indices))
    rows = df.loc[positions].copy().reset_index(drop=True)
    rows['_boot_acct'] = (rows['purchasing_account'].astype(str) + '__'
                          + np.asarray(occurrences).astype(str))
    return rows


def rank_contracts(df, score_column):
    """Rank all policies consistently; break equal scores by contract ID only."""
    return df.sort_values([score_column, 'contract_id'], ascending=[False, True], kind='mergesort')
