def crosstable(labels_a, labels_b, normalize=False):
    table = pd.crosstab(labels_a, labels_b,
                        rownames=['Cluster A'],
                        colnames=['Cluster B'],
                        normalize='all' if normalize else False)
    return table