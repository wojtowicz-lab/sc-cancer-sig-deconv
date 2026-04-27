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

    def __init__(self, path, min_mutations=0):
        """
        Parameters
        ----------
        path : str, optional
            Path to the CSV file.
        min_mutations : int, optional
            Minimum number of mutations per cell (row sum). Cells below this threshold are excluded. Default is 0 (no filtering).
        """
        df = pd.read_csv(path).set_index('cell_id')
        if min_mutations > 0:
            df = df[df.sum(axis=1) >= min_mutations]
        self.df_mutations = df
        
        self.df_contexts = self._aggregate_mutations()
        self.df_substitutions = self._aggregate_mutations(kind='substitution')
        self.nrows = self.df_mutations.shape[0]     # Store number of rows
        self.embeddings = {}    # Cache UMAP embeddings by parameters
        self.cluster_labels = {}    # Cache HDBSCAN labels by parameters
        self.current_umap_params = None     # Track the current UMAP parameters
        self.current_hdbscan_params = None  # Track the current HDBSCAN parameters

        mutation_counts = self.df_mutations.sum(axis=1)
        print(f"Loaded {self.nrows} cells.")
        print(
            f"Mutations per cell:\n"
            f"\tmin:  {mutation_counts.min():.0f}\n"
            f"\tmean: {mutation_counts.mean():.0f}\n"
            f"\tmax:  {mutation_counts.max():.0f}"
        )

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
    
    def deconv(self, signatures, aggregate_by=None):
        """
        Deconvolve mutational signatures using NNLS for each cell or cluster.

        Parameters
        ----------
        signatures : pd.DataFrame
            Signature matrix (rows = mutation types, columns = signature names).
        aggregate_by : array-like, optional
            Cluster labels (e.g., from get_labels()). If provided, mutations are summed
            per cluster before NNLS, excluding noise points (-1).

        Returns
        -------
        pd.DataFrame
            Exposures per cell or per cluster (rows = cells/clusters, columns = signature names),
            normalized to sum to 1.
        """
        n_features = signatures.shape[0]

        if n_features == self.df_mutations.shape[1]:
            df = self.df_mutations
        elif n_features == self.df_contexts.shape[1]:
            df = self.df_contexts
        elif n_features == self.df_substitutions.shape[1]:
            df = self.df_substitutions
        else:
            raise ValueError(
                f"Signatures shape {n_features} doesn't match any data type. "
                f"Expected {self.df_mutations.shape[1]}, "
                f"{self.df_contexts.shape[1]}, or {self.df_substitutions.shape[1]}."
            )

        if not df.columns.equals(signatures.index):
            raise ValueError(
                f"Column mismatch: data has {list(df.columns[:5])}... "
                f"but signatures has {list(signatures.index[:5])}..."
            )

        if aggregate_by is not None:
            labels = pd.Series(aggregate_by.values, index=df.index)
            df = df[labels != -1].groupby(labels[labels != -1]).sum()

        S = signatures.values
        M = df.values
        M_norm = M / M.sum(axis=1, keepdims=True)

        exposures = []
        for i in range(M_norm.shape[0]):
            coeffs, _ = nnls(S, M_norm[i, :])
            exposures.append(coeffs)

        result = pd.DataFrame(exposures, index=df.index, columns=signatures.columns)
        return result.div(result.sum(axis=1), axis=0)

    def passthrough(self, data='mutations'):
        """
        Skip dimensionality reduction and use raw data directly for clustering.
        
        Parameters
        ----------
        data : str
            Which data to use ('mutations', 'contexts', 'substitutions').
        """
        key = self._make_key(method='passthrough', data=data)
        if key not in self.embeddings:
            self.embeddings[key] = getattr(self, self._DATASETS[data]).values
        self.current_umap_params = key
        return self