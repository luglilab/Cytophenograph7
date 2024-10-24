import os
import pandas as pd
import fcsy
from LogClass import LoggerSetup
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")


class Grouping:
    def __init__(self, adata, output_folder, tool, listfeature, analysis_name, runtime):
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
        self.adata_original = adata.raw.to_adata()  # Back up of original data
        self.log = LoggerSetup.setup_logging()
        self.listfeature = listfeature
        warnings.filterwarnings("ignore")

    def createdir(self, path):
        """
        Create directory if it does not exist.

        :param path: Directory path to create
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def export_data(self, data, output_folder, filename):
        """
        Helper function to export CSV and FCS files.

        :param data: DataFrame to export
        :param output_folder: Path to save the files
        :param filename: The base filename to use for both CSV and FCS
        """
        # Export CSV
        csv_path = os.path.join(output_folder, f"{filename}.csv")
        data.to_csv(csv_path, header=True, index=False)
        # self.log.info(f"Exported CSV: {csv_path}")

        # Export FCS
        fcs_path = os.path.join(output_folder, f"{filename}.fcs")
        fcsy.write_fcs(data, fcs_path)
        # self.log.info(f"Exported FCS: {fcs_path}")

    def groupbycluster(self):
        """
        Generate CSV and FCS files for each cluster.
        """
        # Create directory for cluster outputs
        cluster_output_path = os.path.join(self.output_folder, f"Cluster{self.tool}")
        self.createdir(cluster_output_path)

        # Restore original data
        self.adata_original.obs['cluster'] = self.adata_original.obs['cluster'].astype(int)

        # Iterate through each unique cluster
        for cluster in self.adata_original.obs['cluster'].unique():
            self.tmp = self.adata_original[self.adata_original.obs['cluster'] == cluster].to_df().astype(int)

            # Add UMAP coordinates if in 'Full' mode
            if self.runtime == 'Full':
                umap_df = pd.DataFrame(self.adata_original.obsm['X_umap'], index=self.adata_original.obs_names,
                                       columns=['UMAP_1', 'UMAP_2'])
                self.tmp = pd.merge(self.tmp, umap_df, left_index=True, right_index=True)

            # Add tool-specific cluster information
            self.tmp[self.tool] = cluster

            # Export data
            self.export_data(self.tmp, cluster_output_path, f"{self.analysis_name}_{cluster}")

    def groupbysample(self):
        """
        Generate CSV and FCS files for each sample, along with cluster frequencies.
        """
        # Create directory for sample outputs
        sample_output_path = os.path.join(self.output_folder, f"Sample{self.tool}")
        self.createdir(sample_output_path)

        # Prepare data
        self.tmp = self.adata_original.to_df().astype(int)
        self.tmp.set_index(self.adata.obs['Sample'], inplace=True)

        if self.runtime != 'UMAP':
            self.adata_original.obs['cluster'] = self.adata_original.obs['cluster'].astype(int)
            cluster_frequency_path = os.path.join(self.output_folder, f"ClusterFrequency{self.tool}")
            self.createdir(cluster_frequency_path)

        # Add UMAP coordinates if not in 'Clustering' mode
        if self.runtime != 'Clustering':
            self.tmp['UMAP_1'] = self.adata_original.obsm['X_umap'][:, 0]
            self.tmp['UMAP_2'] = self.adata_original.obsm['X_umap'][:, 1]

        if self.runtime != 'UMAP':
            self.tmp[self.tool] = self.adata_original.obs['cluster'].values
            self.tmp['cluster'] = self.adata_original.obs['cluster'].values

            # Unique filenames and clusters
            unique_filenames = self.adata_original.obs['Sample'].unique()
            unique_clusters = self.adata_original.obs['cluster'].unique()

            # Generate cluster frequency and count files
            self._generate_cluster_frequency_and_count(unique_clusters)

            # Remove 'cluster' column before saving samples
            del self.tmp['cluster']

            # Save sample files
            for filename in unique_filenames:
                sample_data = self.tmp.loc[self.tmp.index == filename]
                self.export_data(sample_data, sample_output_path, f"{filename}_{self.analysis_name}")
        else:
            self._export_samples_without_clustering()

    def _export_samples_without_clustering(self):
        """
        Helper function to export samples without clustering information.
        """
        unique_filenames = self.adata_original.obs['Sample'].unique()
        sample_output_path = os.path.join(self.output_folder, f"Sample{self.tool}")

        for filename in unique_filenames:
            sample_data = self.tmp.loc[self.tmp.index == filename]
            self.export_data(sample_data, sample_output_path, f"{filename}_{self.analysis_name}")

    def _generate_cluster_frequency_and_count(self, unique_clusters):
        """
        Helper function to generate cluster frequency and count files, including additional features from adata.obs.

        :param unique_clusters: Unique cluster identifiers
        """
        df_absolute = pd.DataFrame(index=range(min(unique_clusters), max(unique_clusters) + 1))

        # Total percentage file
        df_relative = pd.DataFrame(index=df_absolute.index)
        self.listfeature.append('Sample')
        for feature in self.listfeature:  # Loop over self.listfeature instead of filename
            if feature in self.adata.obs.columns:
                self.log.info(f"Generating frequency and count files for feature: {feature}")

                # Absolute and relative frequency dataframes for the current feature
                df_feature_absolute = pd.DataFrame(index=range(min(unique_clusters), max(unique_clusters) + 1))
                df_feature_relative = pd.DataFrame(index=df_feature_absolute.index)

                # Calculate absolute counts for the feature
                abs_counts = (
                    self.adata.obs.groupby('cluster')[feature]
                    .value_counts()
                    .reset_index(name='count')  # Reset index to get a simple DataFrame
                    .pivot(index='cluster', columns=feature, values='count')  # Pivot the table for reindexing
                )

                # Calculate relative counts (percentages) for the feature
                rel_counts = (
                        abs_counts.div(abs_counts.sum(axis=1), axis=0) * 100  # Convert to percentages
                )

                # Reindex to match cluster index range and fill missing values with 0
                abs_counts = abs_counts.reindex(df_feature_absolute.index, fill_value=0)
                rel_counts = rel_counts.reindex(df_feature_relative.index, fill_value=0)

                # Fill in the DataFrames for absolute and relative counts
                df_feature_absolute = abs_counts
                df_feature_relative = rel_counts

                # Set index name
                df_feature_absolute.index.name = 'Cluster'
                df_feature_relative.index.name = 'Cluster'

                # Export frequency and count data for the current feature
                feature_frequency_path = os.path.join(self.output_folder, f"ClusterFrequency{self.tool}", feature)
                self.createdir(feature_frequency_path)

                df_feature_relative.to_csv(os.path.join(feature_frequency_path, f"{feature}_TotalPercentage.csv"))
                df_feature_absolute.to_csv(os.path.join(feature_frequency_path, f"{feature}_TotalCounts.csv"))

                self.log.info(f"Exported frequency and count files for feature: {feature}")

    # def generate_balloon_plot(self):
    #     """
    #     Generates a balloon plot with `self.listfeature` (excluding 'Sample') on the X-axis
    #     and markers (genes) from `adata.var.index` on the Y-axis. Dot size represents relative frequency,
    #     and dot color represents the median fluorescence intensity (MFI). Saves the plot as a PDF.
    #     """
    #     sns.set_style("whitegrid")
    #
    #     # Create a new folder for saving the plot
    #     self.log.info("Generating and saving balloon plot.")
    #     self._folder = "/".join([self.output_folder, "Dotplot"])
    #     self.createdir(self._folder)
    #
    #     # Extract the observation dataframe
    #     obs_df = self.adata.obs
    #
    #     # Iterate over each marker in `adata.var.index`
    #     for marker in self.adata.var.index:
    #         # Get the marker index in `adata.var`
    #         marker_index = self.adata.var_names.get_indexer([marker])
    #
    #         # Calculate the mean expression for the marker using .layers['raw_value']
    #         mean_expression = self.adata.layers['raw_value'][:, marker_index].flatten()
    #
    #         # Add this as a new column in the obs dataframe
    #         obs_df['mean_expression'] = mean_expression
    #
    #         # Prepare the features from `self.listfeature` excluding 'Sample'
    #         x_features = [feat for feat in self.listfeature if feat != 'Sample']
    #
    #         # Create a grouping feature by concatenating selected x_features
    #         obs_df['group'] = obs_df[x_features].astype(str).agg('_'.join, axis=1)
    #
    #         # Calculate the total number of cells per group
    #         total_cells_per_group = obs_df.groupby('group').size()
    #
    #         # Calculate frequency and mean expression grouped by 'group' and 'pheno_leiden'
    #         grouped_data = obs_df.groupby(['group', 'pheno_leiden']).agg(
    #             frequency=('mean_expression', 'size'),
    #             mean_value=('mean_expression', 'mean')).reset_index()
    #
    #         # Merge the total cell count to calculate relative frequency
    #         grouped_data = grouped_data.merge(total_cells_per_group.rename('total_cells'),
    #                                           on='group')
    #
    #         # Calculate relative frequency
    #         grouped_data['relative_frequency'] = grouped_data['frequency'] / grouped_data['total_cells']
    #
    #         # Define normalization for the hue (MFI) and color palette
    #         norm = plt.Normalize(grouped_data['mean_value'].min(), grouped_data['mean_value'].max())
    #         palette = "RdYlBu_r"  # Use the color palette from the provided code
    #
    #         # Plotting the dotplot using the refined style
    #         g = sns.relplot(data=grouped_data, x="group", y="pheno_leiden",
    #                         hue="mean_value", size="relative_frequency", sizes=(20, 200),
    #                         hue_norm=norm, palette=palette, aspect=1.8, height=6)
    #
    #         # Create colorbar and associate it with the correct axis (g.ax)
    #         cbar = plt.colorbar(ScalarMappable(cmap=palette, norm=norm), ax=g.ax, shrink=0.7)
    #         pos = cbar.ax.get_position()
    #         cbar.ax.set_position([pos.x0, pos.y0 + pos.height * (1 - 0.7), pos.width, pos.height])
    #
    #         # Manually create the size legend
    #         handles, labels = g.ax.get_legend_handles_labels()
    #
    #         # Remove the automatically generated legend
    #         g.legend.remove()
    #
    #         # Add the refined legend for size (relative frequency)
    #         g.ax.legend(handles[1:], labels[1:], title='Relative Frequency',
    #                     loc='lower left', bbox_to_anchor=(1.035, 0), frameon=False)
    #
    #         # Adjust x-axis tick positions
    #         g.ax.xaxis.set_major_locator(MultipleLocator(1))
    #
    #         # Set the title and axis labels
    #         g.set(title=f'Mean Expression: {marker}', xlabel='_'.join(x_features), ylabel='Pheno Leiden Cluster')
    #
    #         # Adjust layout and save the plot as PDF in the new folder
    #         plot_path = "/".join([self._folder, f"BalloonPlot_{marker}.pdf"])
    #         plt.tight_layout()
    #         plt.savefig(plot_path, format='pdf')  # Save plot as PDF
    #         plt.close()
    #
    #         self.log.info(f"Balloon plot for {marker} saved as PDF at {plot_path}")












