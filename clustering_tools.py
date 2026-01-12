import pandas as pd
import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import umap
import warnings
import seaborn as sns

from sklearn.metrics import silhouette_score
from scipy.optimize import nnls

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="IProgress not found")

class sample:
    _DATASETS = {
        'mutations': 'df_mutations',
        'contexts': 'df_contexts',
        'substitutions': 'df_substitutions',
    }

    def __init__(self, id, filepath=None):
        if filepath is None:
            filepath = f'../data/funnell_tables/{id}_sbs_counts.csv'    # Default filepath based on ID
        self.id = id
        self.df_mutations = pd.read_csv(filepath).set_index('cell_id')   # Load data into a DataFrame
        self.df_contexts = self._aggregate_mutations()
        self.df_substitutions = self._aggregate_mutations(kind='substitution')
        self.nrows = self.df_mutations.shape[0]     # Store number of rows
        self.embeddings = {}    # Cache UMAP embeddings by parameters
        self.cluster_labels = {}    # Cache HDBSCAN labels by parameters
        self.current_umap_params = None     # Track the current UMAP parameters
        self.current_hdbscan_params = None  # Track the current HDBSCAN parameters

    def _aggregate_mutations(self, kind='context'):
        """
        Collapse mutation counts either by trinucleotide context or substitution type.
        
        Parameters
        ----------
        kind : str
            'context'  -> collapse to trinucleotide context (e.g., A[C>A]A -> ACA)
            'substitution' -> collapse by substitution type (e.g., A[C>A]A -> C>A)
        """
        df = self.df_mutations.copy()
        
        if kind == 'context':
            col_map = {col: col[0] + col[2] + col[-1] for col in df.columns}  # ACA, ACT, ...
        elif kind == 'substitution':
            col_map = {col: col[2:5] for col in df.columns}  # C>A, C>G, ...
        else:
            raise ValueError("kind must be 'context' or 'substitution'")
        
        return df.rename(columns=col_map).groupby(level=0, axis=1).sum()

    def _make_key(self, **params):
        # Helper to create a hashable key from parameters for caching
        return tuple(sorted(params.items()))

    def umap(self, data='mutations', metric='cosine', n_components=20, random_state=42, **kwargs):
        params_key = self._make_key(
            data=data,
            metric=metric,
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )

        if params_key in self.embeddings:
            self.current_umap_params = params_key
            return self

        try:
            df = getattr(self, self._DATASETS[data]).copy()
        except KeyError:
            raise ValueError(f"Unknown data='{data}'. Choose from {list(self._DATASETS)}")

        if metric == 'hellinger':
            df = np.sqrt(df)
            df = df.div(df.sum(axis=1), axis=0)
            umap_metric = 'euclidean'
        else:
            umap_metric = metric

        reducer = umap.UMAP(
            metric=umap_metric,
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )

        self.embeddings[params_key] = reducer.fit_transform(df)
        self.current_umap_params = params_key
        return self

    def hdbscan(self, min_cluster_size=30, min_samples=5, **kwargs):
        # Compute HDBSCAN clustering on the current UMAP embedding (or reuse from cache)
        if self.current_umap_params is None:
            raise ValueError("Run umap() first")  # Need embedding first

        embedding = self.embeddings[self.current_umap_params]
        hdbscan_key = self._make_key(min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs)

        # If this combination of embedding + HDBSCAN parameters exists, reuse it
        if self.current_umap_params in self.cluster_labels and hdbscan_key in self.cluster_labels[self.current_umap_params]:
            self.current_hdbscan_params = hdbscan_key
            return self

        # Otherwise, fit HDBSCAN from scratch
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs)
        labels = clusterer.fit_predict(embedding)

        # Cache labels
        self.cluster_labels.setdefault(self.current_umap_params, {})[hdbscan_key] = labels
        self.current_hdbscan_params = hdbscan_key
        return self

    def deconv(self, signatures, cellwise=True, noise=False):
        """
        Deconvolve signatures using NNLS for each cluster.
        
        Parameters:
        -----------
        signatures : pd.DataFrame
            Signature matrix (index = mutations/contexts/substitutions, columns = signature names)
        cellwise : bool
            If True, run NNLS per cell. If False, sum mutations per cluster first.
        noise : bool
            If False, exclude cluster -1 (noise points)
        
        Returns:
        --------
        dict
            Dictionary with cluster labels as keys and DataFrames/Series as values
        """
        if self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() first")
        
        labels = self.get_labels().values
        sig_names = signatures.index
        n_features = signatures.shape[1]
        
        # Match to appropriate df based on number of features
        if n_features == self.df_mutations.shape[1]:
            df = self.df_mutations
        elif n_features == self.df_contexts.shape[1]:
            df = self.df_contexts
        elif n_features == self.df_substitutions.shape[1]:
            df = self.df_substitutions
        else:
            raise ValueError(f"Signatures shape {n_features} doesn't match any data type. "
                            f"Expected {self.df_mutations.shape[1]} or "
                            f"{self.df_contexts.shape[1]} or {self.df_substitutions.shape[1]}")
        
        # Check if columns match (signatures columns must equal df columns)
        if not df.columns.equals(signatures.columns):
            raise ValueError(f"Column mismatch: data has {list(df.columns[:5])}... "
                            f"but signatures has {list(signatures.columns[:5])}...")
        
        S = signatures.T
        results = {}
        unique_clusters = np.unique(labels)
        
        if not noise:
            unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster in unique_clusters:
            mask = labels == cluster
            M = df.iloc[mask].values
            cell_ids = df.iloc[mask].index
            
            if cellwise:
                M_norm = M / M.sum(axis=1, keepdims=True)
                exposures = []
                for i in range(M_norm.shape[0]):
                    coeffs, _ = nnls(S, M_norm[i, :])
                    exposures.append(coeffs)
                df_result = pd.DataFrame(exposures, columns=sig_names, index=cell_ids)
                df_result = df_result.div(df_result.sum(axis=1), axis=0)
            else:
                v = M.sum(axis=0)
                v = v / v.sum()
                coeffs, _ = nnls(S, v)
                coeffs = coeffs / coeffs.sum()
                df_result = pd.Series(coeffs, index=sig_names, name=f"cluster_{cluster}")
            
            results[cluster] = df_result

        return results
    
    def get_labels(self):
        # Return cluster labels as a Pandas Series
        if self.current_umap_params is None or self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() first")  # Ensure clustering has been done
        return pd.Series(
            self.cluster_labels[self.current_umap_params][self.current_hdbscan_params],
            name='cluster_label'
        )
    
    def silhouette_score(self, ignore_noise=True, metric='euclidean'):
        if self.current_umap_params is None or self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() first.")

        X = self.embeddings[self.current_umap_params]
        labels = self.cluster_labels[self.current_umap_params][self.current_hdbscan_params]

        if ignore_noise:
            mask = labels != -1
            X = X[mask]
            labels = labels[mask]

        # Silhouette score requires at least 2 clusters
        if len(np.unique(labels)) < 2:
            raise ValueError("Silhouette score requires at least 2 clusters.")

        return silhouette_score(X, labels, metric=metric)

    def summary(self):        
        if self.current_umap_params is None or self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() first before getting summary.")
        
        labels = self.cluster_labels[self.current_umap_params][self.current_hdbscan_params]
        s = pd.Series(labels, name="cluster_label")

        return s.value_counts().sort_index()

    def plot_clustering(self, show_noise=True, s=15, alpha=0.7, cmap='Dark2', figsize=(6, 4), title=None):
        """
        Scatter plot of the current UMAP embedding colored by HDBSCAN labels
        """
    
        if self.current_umap_params is None or self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() before calling plot().")

        emb = self.embeddings[self.current_umap_params]
        labels = self.cluster_labels[self.current_umap_params][self.current_hdbscan_params]

        # Ensure embedding has at least 2 dimensions
        if emb.shape[1] < 2:
            raise ValueError("UMAP embedding must have at least 2 components to plot.")

        # Optionally filter out noise points (-1)
        if not show_noise:
            mask = labels != -1
            emb_plot = emb[mask]
            labels_plot = labels[mask]
            if emb_plot.shape[0] == 0:
                raise ValueError("No points to plot after removing noise (-1).")
        else:
            emb_plot = emb
            labels_plot = labels

        x = emb_plot[:, 0]
        y = emb_plot[:, 1]

        plt.figure(figsize=figsize)
        sc = plt.scatter(x, y, c=labels_plot, cmap=cmap, s=s, alpha=alpha)
        cbar = plt.colorbar(sc)
        cbar.set_label('Cluster')

        if title is None:
            title = f"HDBSCAN clusters on UMAP — ID: {self.id}"
        plt.title(title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.show()

    def plot_deconv(self, signatures, show_noise=False, figsize=(16, 6)):
        """
        Plot boxplots of signature exposures (cellwise deconvolution) by cluster,
        overlaid with cluster-level deconvolution results as red dots.
        
        Parameters:
        -----------
        signatures : pd.DataFrame
            Signature matrix (index = signature names, columns = mutations/contexts/substitutions)
        show_noise : bool
            If True, include cluster -1 (noise points)
        figsize : tuple
            Figure size
        """
        if self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() first")
        
        # Determine which dataframe to use based on signature shape
        n_features = signatures.shape[1]
        if n_features == self.df_mutations.shape[1]:
            df = self.df_mutations
        elif n_features == self.df_contexts.shape[1]:
            df = self.df_contexts
        elif n_features == self.df_substitutions.shape[1]:
            df = self.df_substitutions
        else:
            raise ValueError(f"Signatures shape {n_features} doesn't match any data type.")
        
        # Get deconvolution results
        cellwise_results = self.deconv(signatures, cellwise=True, noise=show_noise)
        cluster_results = self.deconv(signatures, cellwise=False, noise=show_noise)
        
        # Get labels
        labels = self.get_labels().values
        unique_clusters = sorted([c for c in np.unique(labels) if show_noise or c != -1])
        
        # Prepare data for boxplot (cellwise exposures)
        df_plot_list = []
        for cluster in unique_clusters:
            if cluster in cellwise_results:
                cluster_exp = cellwise_results[cluster]
                cluster_exp['label'] = cluster
                df_plot_list.append(cluster_exp)
        
        df_long = pd.concat(df_plot_list, ignore_index=False)
        df_long = df_long.melt(id_vars=['label'], var_name='signature', value_name='exposure')
        
        # Create plot
        plt.figure(figsize=figsize)
        ax = sns.boxplot(
            data=df_long,
            x='signature',
            y='exposure',
            hue='label',
            palette='Set2',
            showfliers=False
        )
        
        # Overlay cluster-level results as dark gray diamonds, aligned with boxplot centers
        sig_names = signatures.index.tolist()
        n_clusters = len(unique_clusters)
        
        for sig_idx, sig_name in enumerate(sig_names):
            for cluster_idx, cluster in enumerate(unique_clusters):
                if cluster in cluster_results:
                    cluster_val = cluster_results[cluster].loc[sig_name]
                    x_pos = sig_idx + (cluster_idx - (n_clusters - 1) / 2) * (0.8 / n_clusters)
                    ax.plot(x_pos, cluster_val, marker='D', color='black', markersize=4.5, 
                        zorder=5, linestyle='none', mec='black', mew=0.5)
            
        plt.xlabel("Signature")
        plt.ylabel("Exposure")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        
        return self


def crosstable(labels_a, labels_b, normalize=False):
    table = pd.crosstab(labels_a, labels_b,
                        rownames=['Cluster A'],
                        colnames=['Cluster B'],
                        normalize='all' if normalize else False)
    return table

