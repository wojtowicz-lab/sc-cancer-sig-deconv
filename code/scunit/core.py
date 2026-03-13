import pandas as pd
import numpy as np
import hdbscan
import umap
import warnings

from scipy.optimize import nnls
from sklearn.metrics import silhouette_score
from hdbscan.validity import validity_index

from .plotting import PlottingMixin

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="IProgress not found")

class ScUnit(PlottingMixin):
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

    def _cluster_score(self, scorer, ignore_noise=True, metric='euclidean'):
        if self.current_umap_params is None or self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() first.")

        X = self.embeddings[self.current_umap_params].astype(np.float64)  # <- ważne
        labels = self.cluster_labels[self.current_umap_params][self.current_hdbscan_params]

        if ignore_noise:
            mask = labels != -1
            X = X[mask]
            labels = labels[mask]

        if len(np.unique(labels)) < 2:
            raise ValueError("Score requires at least 2 clusters.")

        return scorer(X, labels, metric=metric)

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

        try:
            sil = self.silhouette_score()
            dbcv = self.dbcv_score()
            print(f"Silhouette: {sil:.4f}\nDBCV: {dbcv:.4f}")
        except ValueError:
            print("Could not compute scores (too few clusters).")

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
        return self._cluster_score(
            scorer=silhouette_score,
            ignore_noise=ignore_noise,
            metric=metric
        )

    def dbcv_score(self, ignore_noise=True, metric='euclidean'):
        return float(self._cluster_score(
            scorer=validity_index,
            ignore_noise=ignore_noise,
            metric=metric
        ))

    def summary(self):        
        if self.current_umap_params is None or self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() first before getting summary.")
        
        labels = self.cluster_labels[self.current_umap_params][self.current_hdbscan_params]
        s = pd.Series(labels, name="cluster_label")

        return s.value_counts().sort_index()

