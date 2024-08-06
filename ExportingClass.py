import os
import scipy.sparse
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
import warnings
from LogClass import LoggerSetup
warnings.filterwarnings("ignore")

class Exporting:
    def __init__(self, adata, output_folder, analysis_name, runtime, tool, k_coef=None, knn=None):
        """
        Initialize the Exporting class with necessary attributes.

        :param adata: AnnData object containing the data
        :param output_folder: Path to the output folder
        :param analysis_name: Name of the analysis
        :param runtime: Runtime mode (Full, UMAP, Clustering)
        :param tool: Clustering tool used (Phenograph, VIA, FlowSOM)
        :param k_coef: Coefficient for Phenograph clustering (optional)
        :param knn: Number of nearest neighbors for VIA (optional)
        """
        self.adata = adata
        self.output_folder = output_folder
        self.analysis_name = analysis_name
        self.runtime = runtime
        self.tool = tool
        self.k_coef = k_coef
        self.knn = knn
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Set up logging
        self.log = LoggerSetup.setup_logging()
        sc.settings.verbosity = 0


    sc.settings.verbosity = 0
    warnings.filterwarnings("ignore", category = FutureWarning)

    def exporting(self):
        """
        Export data to an .h5ad file.
        """
        if self.log:
            self.log.info("PART 4")
            self.log.info("Output Generation")

        # Attempt to remove 'remove_from_FM' column if it exists
        try:
            del self.adata.obs['remove_from_FM']
        except KeyError:
            pass

        if self.runtime != 'UMAP':
            self._handle_non_umap_runtime()
        else:
            self._handle_umap_runtime()

        self._cleanup_files()

        if self.log:
            self.log.info("PART 5")
            self.log.info("End")

    def _handle_non_umap_runtime(self):
        """
        Handle exporting when runtime is not UMAP.
        """
        if self.tool == "Phenograph":
            self._export_phenograph()
        elif self.tool == "VIA":
            self._export_via()
        else:
            self._export_flowsom()

    def _export_phenograph(self):
        """
        Export data specific to Phenograph clustering.
        """
        cluster_col = f"{self.tool}_{self.k_coef}"
        self.adata.obs[cluster_col] = self.adata.obs['cluster'].astype("str")
        del self.adata.obs['cluster']
        del self.adata.obs[cluster_col]
        self.adata.obs.rename(columns={"pheno_leiden": "Phenograph_cluster"}, inplace=True)
        self._scale_data()
        self._write_h5ad()

    def _export_via(self):
        """
        Export data specific to VIA clustering.
        """
        cluster_col = f"{self.tool}_{self.knn}"
        self.adata.obs[cluster_col] = self.adata.obs['cluster'].astype("str")
        del self.adata.obs['cluster']
        del self.adata.obs[cluster_col]
        cluster_tool_col = f"{self.tool}_cluster"
        # self.adata.obs[cluster_tool_col] = self.adata.obs[cluster_tool_col].astype('category')
        self._scale_data()
        self._write_h5ad()

    def _export_flowsom(self):
        """
        Export data specific to FlowSOM clustering.
        """
        del self.adata.obs['cluster']
        self._scale_data()
        self._write_h5ad()
        self._remove_intermediate_files()

    def _handle_umap_runtime(self):
        """
        Handle exporting when runtime is UMAP.
        """
        self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.X)
        self.adata.X = self.adata.layers['scaled01']
        self.adata.layers['scaled01'] = scipy.sparse.csr_matrix(self.adata.layers['scaled01'])
        self._write_h5ad()

    def _scale_data(self):
        """
        Scale data using MinMaxScaler and update adata.
        """
        self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.layers['raw_value'])
        self.adata.X = self.adata.layers['scaled01']
        self.adata.layers['scaled01'] = scipy.sparse.csr_matrix(self.adata.layers['scaled01'])

    def _write_h5ad(self):
        """
        Write adata to an .h5ad file.
        """
        if self.tool == "FlowSOM":
            try:
                del self.adata.obs['distance_to_bmu']
                del self.adata.obs['metaclustering']
                del self.adata.var
            except FileNotFoundError:
                pass
        output_path = "/".join([self.output_folder, ".".join([self.analysis_name, "h5ad"])])
        self.adata.write(output_path)

    def _remove_intermediate_files(self):
        """
        Remove temporary files if they exist.
        """
        try:
            os.remove(os.path.join(self.output_folder, "tmp.csv"))
            os.remove(os.path.join(self.output_folder, "corrected_data.csv"))
        except FileNotFoundError:
            pass

    def _cleanup_files(self):
        """
        Clean up intermediate and concatenated files.
        """
        try:
            os.remove(os.path.join(self.output_folder, f"{self.analysis_name}_ConcatenatedCells_concatenate_after_QC.fcs"))
            os.remove(os.path.join(self.output_folder, f"{self.analysis_name}_ConcatenatedCells.fcs"))
        except FileNotFoundError:
            pass
