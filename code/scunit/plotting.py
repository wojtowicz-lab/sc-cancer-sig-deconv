import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class PlottingMixin:
    def plot_embedding(self, show_noise=True, s=15, alpha=0.7, cmap='Dark2', figsize=(6, 4), title=None):
        if self.current_umap_params is None or self.current_hdbscan_params is None:
            raise ValueError("Run umap() and hdbscan() before calling plot().")

        emb = self.embeddings[self.current_umap_params]
        labels = self.cluster_labels[self.current_umap_params][self.current_hdbscan_params]

        if emb.shape[1] < 2:
            raise ValueError("UMAP embedding must have at least 2 components to plot.")

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

        unique_labels = sorted(np.unique(labels_plot))
        cmap_obj = plt.get_cmap(cmap, len(unique_labels))
        color_map = {lab: cmap_obj(i) for i, lab in enumerate(unique_labels)}

        plt.figure(figsize=figsize)
        for lab in unique_labels:
            mask = labels_plot == lab
            label_str = 'noise' if lab == -1 else f'cluster {lab}'
            plt.scatter(x[mask], y[mask], c=[color_map[lab]], s=s, alpha=alpha, label=label_str)

        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        if title is None:
            title = f"HDBSCAN clusters on UMAP — ID: {self.id}"
        plt.title(title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.show()

        return self

    def plot_exposures(self, signatures, show_noise=False, figsize=(16, 6)):
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
                cluster_exp = cellwise_results[cluster].copy()
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
