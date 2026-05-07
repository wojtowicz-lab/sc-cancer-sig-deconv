import numpy as np
import pandas as pd

def crosstable(labels_a, labels_b, normalize=False):
    table = pd.crosstab(labels_a, labels_b,
        rownames=['Cluster A'],
        colnames=['Cluster B'],
        normalize=normalize)
    return table

def clr(df_exposures, numerical_zero=1e-10):
    """
    Centered log-ratio (CLR) transformation of compositional exposure data.
    
    Zeros are replaced with a small epsilon (2/3 of the smallest non-zero value)
    prior to transformation to ensure numerical stability. Rows are re-normalized
    after zero replacement to maintain the unit-sum constraint.
    
    Parameters
    ----------
    df_exposures : pd.DataFrame
        Per-cell signature exposures (rows = cells, columns = signatures).
    numerical_zero : float
        Values below this threshold are treated as zero. Default is 1e-10.
    
    Returns
    -------
    pd.DataFrame
        CLR-transformed exposures.
    """
    df = df_exposures.copy()

    # replace near-zero values with epsilon to avoid log(0)
    epsilon = (2/3) * df[df > numerical_zero].min().min()
    df[df <= numerical_zero] = epsilon
    
    # re-normalize rows to maintain unit-sum constraint
    df = df.div(df.sum(axis=1), axis=0)
    
    # centered log-ratio transformation
    df = df.apply(lambda x: np.log(x / np.exp(np.log(x).mean())), axis=1)
    
    return df

def test_signature_exposures(df_exposures, labels, clr_transform=True, numerical_zero=1e-10):
    """
    Test for differential signature activity between clusters using Kruskal-Wallis test.
    Parameters
    ----------
    df_exposures : pd.DataFrame
        Per-cell signature exposures (rows = cells, columns = signatures).
    labels : pd.Series
        Cluster labels (e.g., from ScUnit.get_labels()). Noise points (-1) are excluded.
    clr_transform : bool
        If True, apply CLR transformation prior to testing. Default is True.
    numerical_zero : float
        Values below this threshold are treated as zero. Default is 1e-10.
    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'stat', 'eta_squared', 'p_value', 'p_value_bh',
        and either 'auroc' (k=2) or 'auroc_min' and 'auroc_avg' (k>2).
    """
    from scipy.stats import kruskal
    from statsmodels.stats.multitest import multipletests
    from sklearn.metrics import roc_auc_score
    from itertools import combinations

    def _eta_squared(groups):
        all_values = np.concatenate(groups)
        grand_mean = all_values.mean()
        ss_total = ((all_values - grand_mean) ** 2).sum()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        return ss_between / ss_total if ss_total > 0 else 0.0

    def _auroc(groups_raw, unique_labels):
        aurocs = []
        for l1, l2 in combinations(range(len(unique_labels)), 2):
            g1, g2 = groups_raw[l1], groups_raw[l2]
            y_true = np.array([0] * len(g1) + [1] * len(g2))
            y_score = np.concatenate([g1, g2])
            auroc = roc_auc_score(y_true, y_score)
            aurocs.append(max(auroc, 1 - auroc))  # ensure AUROC >= 0.5
        return aurocs

    df_raw = df_exposures.copy().reset_index(drop=True)
    df = clr(df_raw, numerical_zero=numerical_zero) if clr_transform else df_raw
    mask = labels.values != -1
    df = df.loc[mask].reset_index(drop=True)
    df_raw = df_raw.loc[mask].reset_index(drop=True)
    labels_clean = labels.values[mask]
    unique_labels = np.unique(labels_clean)
    k = len(unique_labels)

    results = {}
    for sig in df.columns:
        groups = [df.loc[labels_clean == label, sig].values for label in unique_labels]
        groups_raw = [df_raw.loc[labels_clean == label, sig].values for label in unique_labels]
        stat, pval = kruskal(*groups)
        aurocs = _auroc(groups_raw, unique_labels)
        entry = {
            'stat': stat,
            'p_value': pval,
            'eta_squared': _eta_squared(groups_raw),
        }
        if k == 2:
            entry['auroc'] = aurocs[0]
        else:
            entry['auroc_min'] = min(aurocs)
            entry['auroc_avg'] = np.mean(aurocs)
        results[sig] = entry

    result_df = pd.DataFrame(results).T
    _, pvals_bh, _, _ = multipletests(result_df['p_value'], method='fdr_bh')
    result_df['p_value_bh'] = pvals_bh

    cols = ['stat', 'eta_squared', 'p_value', 'p_value_bh']
    cols += ['auroc'] if k == 2 else ['auroc_min', 'auroc_avg']
    return result_df[cols]