import subprocess
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import scanpy.external as sce
import umap
import flowsom as fs
import pyVIA.core as via
import anndata as ad
from LogClass import LoggerSetup
import warnings
import shap
import fasttreeshap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Clustering:
    """
    A class responsible for performing clustering on single-cell data using different methods.
    """

    def __init__(self, adata, output_folder, method, k_coef, knn, resolution, maxclus, thread, runtime, batchcov, root_user, fnull, path_cycombine,
                 markertoinclude,marker_array):
        """
        Initialize the Clustering class with the given parameters.

        :param adata: AnnData object containing the data.
        :param output_folder: Output folder for temporary files.
        :param method: Clustering method to use ('phenograph', 'flowsom', or 'via').
        :param k_coef: Number of nearest neighbors for Phenograph clustering.
        :param knn: Number of K-Nearest Neighbors for VIA clustering.
        :param resolution: Resolution parameter for VIA clustering.
        :param maxclus: Maximum number of clusters for meta-clustering.
        :param thread: Number of threads to use for clustering.
        :param runtime: Runtime option ('Full' or 'Clustering').
        :param batchcov: Batch covariate for correction.
        :param root_user: Root user for VIA clustering.
        :param fnull: File object for null output redirection.
        :param path_cycombine: Path to the R script for batch correction.
        """
        self.adata = adata
        self.output_folder = output_folder
        self.method = method
        self.k_coef = k_coef
        self.knn = knn
        self.resolution = resolution
        self.maxclus = maxclus
        self.thread = thread
        self.runtime = runtime
        self.batchcov = batchcov
        self.root_user = root_user
        self.fnull = fnull
        self.path_cycombine = path_cycombine
        self.markertoinclude = markertoinclude
        self.marker_array = marker_array
        self.log = LoggerSetup.setup_logging()

    sc.settings.verbosity = 0

    def runclustering(self):
        """
        Executes clustering analysis on the dataset using the specified method ('phenograph', 'flowsom', or 'via').
        :return: Updated AnnData object with clustering results.
        """
        self.log.info("PART 2")
        self.log.info(f"{self.method.capitalize()} Clustering")

        # Log markers used and excluded for clustering
        self._log_markers()

        # Backup the original AnnData object
        self.adataback = self.adata.copy()

        # Subset AnnData and prepare data
        self._prepare_data()

        # Perform batch correction if specified
        if self.batchcov is True:
            self._batch_correction()

        # Standardize data
        self.adata.X = StandardScaler().fit_transform(self.adata.X)

        # Execute clustering based on the specified method
        if self.method.lower() == 'phenograph':
            self._run_phenograph()
        elif self.method.lower() == 'flowsom':
            self._run_flowsom()
        elif self.method.lower() == 'via':
            self._run_via()
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

        # Update AnnData with clustering results
        self._update_results()

        return self.adata

    def _log_markers(self):
        """Logs markers used and excluded for clustering."""
        self.log.info(f"Markers used for {self.method.capitalize()} clustering:")
        for marker in self.markertoinclude:
            self.log.info(f" + {marker}")

        if self.marker_array:
            self.log.info(f"Markers excluded from {self.method.capitalize()} clustering:")
            for marker in self.marker_array:
                self.log.info(f" - {marker}")

    def _prepare_data(self):
        """Subsets AnnData, adds an index column, and saves data for batch correction."""
        self.adata.raw = self.adata
        self.adata.var['highly_variable'] = False
        self.adata.var.loc[self.markertoinclude, 'highly_variable'] = True
        self.adata = self.adata[:, self.adata.var.highly_variable]
        #self.adata = self.adata[:, self.markertoinclude]
        self.adata.var['n'] = range(len(self.adata.var))
        data_df = self.adata.to_df()
        data_df['batch'] = self.adata.obs[self.batchcov]
        data_df.to_csv(f"{self.output_folder}/tmp.csv", index=False)

    def _batch_correction(self):
        """Performs batch correction using an external R script."""
        self.log.info("Performs batch correction using cyCombine")
        subprocess.check_call(
            ['Rscript', '--vanilla', self.path_cycombine, f"{self.output_folder}/tmp.csv", self.output_folder,
             str(self.maxclus), str(self.adata.shape[1])],
            stdout=self.fnull, stderr=self.fnull
        )
        corrected_csv_path = f"{self.output_folder}/corrected_data.csv"
        corrected_data = pd.read_csv(corrected_csv_path).drop(columns=["id", "label", "batch"])
        self.adatacorrected = ad.AnnData(corrected_data)
        self.adata.X = self.adatacorrected.X

    def _run_phenograph(self):
        """Executes Phenograph clustering."""
        if self.runtime == 'Clustering':
            self.communities, self.graph, self.Q = sce.tl.phenograph(
                self.adata.X,
                k=int(self.k_coef),
                seed=42,
                clustering_algo="leiden",
                directed=True,
                primary_metric="euclidean",
                q_tol=0.05,
                prune=False,
                min_cluster_size=1,
                n_jobs=int(self.thread)
            )
        elif self.runtime == 'Full':
            if self.log:
                self.log.info("PART 3")
                self.log.info("Run Umap")
            self.adata.obsm['X_umap'] = umap.UMAP().fit_transform(self.adata.X)
            if self.log:
                self.log.info("Run Clustering")
            self.communities, self.graph, self.Q = sce.tl.phenograph(
                self.adata.X,
                k = int(self.k_coef),
                seed = 42,
                clustering_algo = "leiden",
                directed = True,
                primary_metric = "euclidean",
                q_tol = 0.05,
                prune = False,
                min_cluster_size = 1,
                n_jobs = int(self.thread)
            )

    def _run_flowsom(self):
        """Executes FlowSOM clustering."""
        if self.runtime == 'Clustering':
            fsom = fs.FlowSOM(
                self.adata, cols_to_use=list(range(len(self.adata.var))),
                xdim=10, ydim=10, n_clusters=self.maxclus, seed=42
            )
            self.communities = fsom.metacluster_labels
        elif self.runtime == 'Full':
            if self.log:
                self.log.info("PART 3")
                self.log.info("Run Umap")
            self.adata.obsm['X_umap'] = umap.UMAP().fit_transform(self.adata.X)
            if self.log:
                self.log.info("Run Clustering")
            fsom = fs.FlowSOM(
                self.adata, cols_to_use=list(range(len(self.adata.var))),
                xdim=10, ydim=10, n_clusters=self.maxclus, seed=42
            )
            self.communities = fsom.metacluster_labels
            self.adata.obs['MetaCluster_Flowsom'] = pd.Categorical(fsom.cluster_labels)

    def _run_via(self):
        """Executes VIA clustering."""
        if self.runtime == 'Clustering':
            p = via.VIA(
                self.adata.X, random_seed=42, knn=int(self.knn), root_user=self.root_user,
                jac_weighted_edges=False, distance='l2',
                small_pop=10, resolution_parameter=self.resolution, num_threads=int(self.thread)
            )
            p.run_VIA()
            self.communities = p.labels
        elif self.runtime == 'Full':
            if self.log:
                self.log.info("PART 3")
                self.log.info("Run Umap")
            self.adata.obsm['X_umap'] = umap.UMAP().fit_transform(self.adata.X)
            if self.log:
                self.log.info("Run Clustering")
            p = via.VIA(
                self.adata.X, random_seed=42, knn=int(self.knn), root_user=self.root_user,
                jac_weighted_edges=False, distance='l2',
                small_pop=10, resolution_parameter=self.resolution, num_threads=int(self.thread)
            )
            p.run_VIA()
            self.communities = p.labels

    def _update_results(self):
        """Updates the AnnData object with clustering results."""
        self.adata.obs['pheno_leiden'] = pd.Categorical(self.communities)
        self.adata.obs['cluster'] = pd.Categorical(self.communities)

        self.adataback.obs['pheno_leiden'] = pd.Categorical(self.communities)
        self.adataback.obs['cluster'] = pd.Categorical(self.communities)
        if 'X_umap' in self.adata.obsm:
            self.adataback.obsm['X_umap'] = self.adata.obsm['X_umap']

    def subset_adata_by_cluster(self, cluster_key="pheno_leiden", fraction=0.1, random_state=42):
        """
        Subset an AnnData object to include only a fraction of cells from each cluster, with reproducibility.

        :param cluster_key: The name of the column in adata.obs that contains the cluster assignments.
        :param fraction: The fraction of cells to sample from each cluster.
        :param random_state: The seed for random number generation to ensure reproducibility.
        :return: A new AnnData object containing only the sampled cells.
        """
        np.random.seed(random_state)  # Set the random seed for reproducibility

        # Get the unique clusters
        clusters = self.adata.obs[cluster_key].unique()

        # Initialize an empty list to hold the indices of the selected cells
        selected_indices = []

        # For each cluster, randomly sample the specified fraction of cells
        for cluster in clusters:
            # Get the indices of the cells in the current cluster
            cluster_indices = self.adata.obs[self.adata.obs[cluster_key] == cluster].index.tolist()

            # Determine how many cells to sample (fraction of the cluster)
            sample_size = max(1, int(len(cluster_indices) * fraction))  # Ensure at least one cell is selected

            # Randomly sample the cells with the set seed
            sampled_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)

            # Add the sampled indices to the list of selected indices
            selected_indices.extend(sampled_indices)

        # Subset the AnnData object to include only the selected cells
        self.new_adata = self.adata[selected_indices].copy()

    import matplotlib.pyplot as plt
    import shap

    def run_shap_explainability(self, cell_idx=None):
        """
        Perform explainability using fasttreeshap for clustering results.
        SHAP values will be computed to explain the clusters based on the gene expression data.

        :param cell_idx: (optional) Index of the cell to plot waterfall SHAP values for. If None, a random cell will be selected.
        """
        self.log.info("Running SHAP explainability for clustering results.")

        try:
            # Prepare input and target for the classifier
            X = self.new_adata.X  # Gene expression data
            y = self.new_adata.obs['pheno_leiden'].cat.codes  # Convert categorical clusters to numeric

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Train a RandomForest classifier
            self.log.info("Training RandomForest classifier for SHAP.")
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=self.thread)
            model.fit(X_train, y_train)

        except Exception as e:
            self.log.error(f"Error during model training: {e}")
            return

        try:
            # Create a fasttreeshap explainer for the model
            self.log.info("Creating fasttreeshap explainer and computing SHAP values.")
            explainer = fasttreeshap.TreeExplainer(model, algorithm='auto', n_jobs=self.thread)

            # Compute SHAP values for the test set
            shap_values = explainer.shap_values(X_test)

        except Exception as e:
            self.log.error(f"Error during SHAP value computation: {e}")
            return

        try:
            # Plot and save summary of SHAP values for all clusters
            self.log.info("Generating and saving SHAP summary plot.")
            shap.summary_plot(shap_values, X_test, feature_names=self.adata.var_names, show=False)

            # Set facecolor to white, remove grid, and save as PDF
            plt.gcf().set_facecolor('white')
            plt.gca().grid(False)  # Remove the grey grid
            plt.savefig(f"{self.output_folder}/shap_summary_all_clusters.pdf", format='pdf', facecolor='white')
            plt.close()

            # Optionally, save SHAP summary plots per class/cluster with cluster names
            self.log.info("Saving SHAP plots for individual clusters.")
            for i, cluster in enumerate(np.unique(y_test)):
                cluster_label = f"Cluster {i + 1}"  # Replace class labels with Cluster 1 to N
                shap.summary_plot(shap_values[i], X_test, feature_names=self.adata.var_names, show=False)

                # Set facecolor to white, remove grid, and save as PDF
                plt.gcf().set_facecolor('white')
                plt.gca().grid(False)  # Remove the grey grid
                plt.title(cluster_label)  # Update the title to use 'Cluster X'
                plt.savefig(f"{self.output_folder}/shap_summary_{cluster_label}.pdf", format='pdf', facecolor='white')
                plt.close()

            self.log.info("SHAP explainability completed and plots saved.")

        except Exception as e:
            self.log.error(f"Error during SHAP plotting or saving: {e}")




