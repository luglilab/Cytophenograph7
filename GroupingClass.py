import os
import pandas as pd
import fcsy
from LogClass import LoggerSetup
import warnings
warnings.filterwarnings("ignore")

class Grouping:
    def __init__(self, adata, output_folder, tool, analysis_name, runtime):
        """
        Initialize the Grouping class with necessary attributes.

        :param adata: AnnData object containing the data
        :param output_folder: Path to the output folder
        :param tool: Clustering tool used (Phenograph, VIA, FlowSOM)
        :param analysis_name: Name of the analysis
        :param runtime: Runtime mode (Full, UMAP, Clustering)
        """
        self.adata = adata
        self.output_folder = output_folder
        self.tool = tool
        self.analysis_name = analysis_name
        self.runtime = runtime
        self.adataback = adata.copy()  # Back up of original data
        self.log = LoggerSetup.setup_logging()
        warnings.filterwarnings("ignore")

    def createdir(self, path):
        """
        Create directory if it does not exist.

        :param path: Directory path to create
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def groupbycluster(self):
        """
        Generate CSV and FCS files for each cluster.
        """
        # Create directory for cluster outputs
        self.createdir("/".join([self.output_folder, "".join(["Cluster", self.tool])]))

        # Restore original data
        self.adata_original =self.adata.raw.to_adata()
        self.adata_original.obs['cluster'] = self.adata_original.obs['cluster'].astype(int)
        # Iterate through each unique cluster
        for cluster in range(self.adata_original.obs['cluster'].unique().min(), self.adata_original.obs['cluster'].unique().max() + 1):
            self.tmp = self.adata_original[self.adata_original.obs['cluster'] == cluster].to_df().astype(int)

            # Add UMAP coordinates if in 'Full' mode
            if self.runtime == 'Full':
                self.tmp = pd.merge(self.tmp, pd.DataFrame(self.adata_original.obsm['X_umap'], index = self.adata_original.obs_names,
                                   columns = ['UMAP_1', 'UMAP_2']),left_index = True,right_index = True)
            # Add tool-specific cluster information
            if self.tool == "Phenograph":
                self.tmp['Phenograph'] = cluster
            elif self.tool == "VIA":
                self.tmp['VIA'] = cluster
            else:
                self.tmp['FlowSOM'] = cluster

            # Export CSV and FCS files
            output_path = "/".join(
                [self.output_folder, "".join(["Cluster", self.tool]), f"{self.analysis_name}_{cluster}.csv"])
            self.tmp.to_csv(output_path, header = True, index = False)
            fcs_path = "/".join(
                [self.output_folder, "".join(["Cluster", self.tool]), f"{self.analysis_name}_{cluster}.fcs"])
            fcsy.write_fcs(self.tmp, fcs_path)

    def groupbysample(self):
        """
        Generate CSV and FCS files for each sample, along with cluster frequencies.
        """
        # Create directory for sample outputs
        self.createdir("/".join([self.output_folder, "".join(["Sample", self.tool])]))

        # Prepare data
        self.tmp = self.adata_original.to_df().astype(int)
        self.tmp.set_index(self.adata.obs['Sample'], inplace = True)

        if self.runtime != 'UMAP':
            self.adata_original.obs['cluster'] = self.adata_original.obs['cluster'].astype(int)
            self.createdir("/".join([self.output_folder, "".join(["ClusterFrequency", self.tool])]))

        # Add UMAP coordinates if not in 'Clustering' mode
        if self.runtime != 'Clustering':
            self.tmp['UMAP_1'] = self.adata_original.obsm['X_umap'][:, 0]
            self.tmp['UMAP_2'] = self.adata_original.obsm['X_umap'][:, 1]

        if self.runtime != 'UMAP':
            self.tmp[self.tool] = self.adata_original.obs['cluster'].values
            self.tmp["cluster"] = self.adata_original.obs['cluster'].values

            # Unique filenames and clusters
            unique_filenames = self.adata_original.obs['Sample'].unique()
            unique_clusters = self.adata_original.obs['cluster'].unique()

            # Generate cluster frequency and count files
            self._generate_cluster_frequency_and_count(unique_clusters, unique_filenames)

            # Remove 'cluster' column before saving samples
            del self.tmp['cluster']

            # Save sample files
            for filename in unique_filenames:
                sample_data = self.tmp.loc[self.tmp.index == filename]
                sample_path = "/".join(
                    [self.output_folder, "".join(["Sample", self.tool]), f"{filename}_{self.analysis_name}.csv"])
                sample_data.to_csv(sample_path, header = True, index = False)

                fcs_sample_path = "/".join(
                    [self.output_folder, "".join(["Sample", self.tool]), f"{filename}_{self.analysis_name}.fcs"])
                fcsy.write_fcs(sample_data, fcs_sample_path)
        else:
            self._export_samples_without_clustering()

    def _generate_cluster_frequency_and_count(self, unique_clusters, unique_filenames):
        """
        Helper function to generate cluster frequency and count files.

        :param unique_clusters: Unique cluster identifiers
        :param unique_filenames: Unique sample filenames
        """
        dfCounts = pd.DataFrame(index = range(min(unique_clusters), max(unique_clusters) + 1))

        # Total percentage file
        for filename in unique_filenames:
            cluster_counts = self.tmp.loc[self.tmp.index == filename].cluster.value_counts(normalize = True)
            dfCounts[filename] = cluster_counts.reindex(self.tmp.cluster.unique(), fill_value = 0)

        dfCounts = dfCounts * 100
        dfCounts.index.name = 'Cluster'

        total_percentage_path = "/".join(
            [self.output_folder, "".join(["ClusterFrequency", self.tool]), "TotalPercentage.csv"])
        dfCounts.to_csv(total_percentage_path)

        # Total counts file
        dfCounts = pd.DataFrame(index = range(min(unique_clusters), max(unique_clusters) + 1))

        for filename in unique_filenames:
            cluster_counts = self.tmp.loc[self.tmp.index == filename].cluster.value_counts()
            dfCounts[filename] = cluster_counts.reindex(self.tmp.cluster.unique(), fill_value = 0)

        dfCounts.index.name = 'Cluster'

        total_counts_path = "/".join([self.output_folder, "".join(["ClusterFrequency", self.tool]), "TotalCounts.csv"])
        dfCounts.to_csv(total_counts_path)

    def _export_samples_without_clustering(self):
        """
        Helper function to export samples without clustering information.
        """
        unique_filenames = self.adata_original.obs['Sample'].unique()
        for filename in unique_filenames:
            sample_data = self.tmp.loc[self.tmp.index == filename]
            sample_path = "/".join(
                [self.output_folder, "".join(["Sample", self.tool]), f"{filename}_{self.analysis_name}.csv"])
            sample_data.to_csv(sample_path, header = True, index = False)

            fcs_sample_path = "/".join(
                [self.output_folder, "".join(["Sample", self.tool]), f"{filename}_{self.analysis_name}.fcs"])
            fcsy.write_fcs(sample_data, fcs_sample_path)
