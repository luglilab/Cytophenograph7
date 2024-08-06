import subprocess
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import scanpy.external as sce
import umap
import flowsom as fs
import pyVIA.core as via
import anndata as ad
import logging
import warnings

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
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        # Ensure that the logging handler is set up only once
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.log.addHandler(handler)

    sc.settings.verbosity = 0
    warnings.filterwarnings("ignore", category = FutureWarning)

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
        if self.batchcov:
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

        # Run additional analyses and visualizations if runtime is 'Full'
        #if self.runtime == 'Full':
        #    self._run_full_analysis()

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
        self.adata = self.adata[:, self.markertoinclude]
        self.adata.var['n'] = range(len(self.adata.var))

        data_df = self.adata[:, self.markertoinclude].to_df()
        data_df['batch'] = self.adata.obs[self.batchcov]
        data_df.to_csv(f"{self.output_folder}/tmp.csv", index=False)

    def _batch_correction(self):
        """Performs batch correction using an external R script."""
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

    def _run_full_analysis(self):
        """Runs additional analyses and visualizations if runtime is 'Full'."""
        self.generation_concatenate()
        self.plot_umap()
        # self.plot_umap_expression()
        self.plot_frequency()
        self.plot_cell_clusters()
        self.plot_cell_obs()
        self.matrixplot()
