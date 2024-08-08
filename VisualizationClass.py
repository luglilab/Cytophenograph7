import pandas as pd
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from seaborn import color_palette
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from LogClass import LoggerSetup
import warnings
import pacmap
from seaborn import color_palette
warnings.filterwarnings("ignore")
from pypdf import PdfWriter, PdfReader, PageObject, Transformation

class Visualization:
    """
    A class responsible for visualizing results from single-cell data clustering analysis.
    """

    def __init__(self, adata, output_folder, tool, runtime,analysis_name):
        """
        Initialize the Visualization class.
        :param adata: AnnData object containing the data.
        :param output_folder: Output folder for saving plots.
        :param tool: The clustering tool used (e.g., Phenograph, VIA, FlowSOM).
        """
        self.adata = adata
        self.output_folder = output_folder
        self.tool = tool
        #self.palette28 = palette28
        #self.palette102 = palette102
        self.dpi = 300
        self.runtime = runtime
        self.analysis_name = analysis_name
        # Ensure that the logging handler is set up only once
        self.log = LoggerSetup.setup_logging()
        sc.settings.autoshow = False
        sc.settings.set_figure_params(dpi = 300, facecolor = 'white', dpi_save = 330,
                                      figsize = (10, 10))

    def generation_concatenate(self):
        """
        Function to concatenate the results of the clustering and the original adata object.
        Return: adata object with the results of the clustering
        """
        if self.runtime == 'Full':
            adata_df = self.adata.to_df()
            umap_df = pd.DataFrame(self.adata.obsm['X_umap'], index = self.adata.obs_names,
                                   columns = ['UMAP_1', 'UMAP_2'])
            self.tmp_df = pd.merge(adata_df, umap_df, left_index = True, right_index = True)
            pd.merge(self.tmp_df, self.adata.obs[['cluster',
                                                  'Sample', 'Cell_type',
                                                  'EXP',
                                                  'ID', 'Time_point',
                                                  'Condition']], left_index=True,
                     right_index=True).to_csv("/".join([self.output_folder, ".".join([self.analysis_name, "csv"])]),
                                              header=True, index=False)
        elif self.runtime == 'UMAP':
            self.tmp_df = pd.merge(pd.DataFrame(self.adata.X,
                                                columns=self.adata.var_names,
                                                index=self.adata.obs.index).astype(int),
                                   pd.DataFrame(self.adata.obsm['X_umap'], columns=['UMAP_1', 'UMAP_2'],
                                                index=self.adata.obs.index),
                                   right_index=True,
                                   left_index=True)
            pd.merge(self.tmp_df, self.adata.obs[['Sample', 'Cell_type',
                                                  'EXP',
                                                  'ID', 'Time_point',
                                                  'Condition']], left_index=True,
                     right_index=True).to_csv("/".join([self.output_folder, ".".join([self.analysis_name, "csv"])]),
                                              header=True, index=False)
        elif self.runtime == 'Clustering':
            self.tmp_df = pd.merge(pd.DataFrame(self.adata.X,
                                                columns=self.adata.var_names,
                                                index=self.adata.obs.index).astype(int),
                                   self.adata.obs[['cluster', 'Sample', 'Cell_type',
                                                   'EXP',
                                                   'ID', 'Time_point',
                                                   'Condition']],
                                   right_index=True,
                                   left_index=True)
            self.tmp_df.to_csv("/".join([self.output_folder, ".".join([self.analysis_name, "csv"])]), header=True,
                               index=False)

    def plot_umap(self):
        """
        Function for generating PDF files with UMAP plots.
        Returns: PDF files with UMAP plots.
        """
        if self.runtime == 'Full':
            # create output directory
            self.outfig = self.output_folder
            self.UMAP_folder = "/".join([self.outfig, "UMAP"])
            self.createdir(self.UMAP_folder)
            sc.settings.figdir = self.UMAP_folder
            self.palette = color_palette("husl", len(self.adata.obs["pheno_leiden"].unique()))
            self.dotsize=30
            # # set palette
            # if len(self.adata.obs["pheno_leiden"].unique()) < 28:
            #     self.palette = self.palette28
            # else:
            #     self.palette = color_palette("husl", len(self.adata.obs["pheno_leiden"].unique()))

            # plot umap + clustering
            sc.pl.umap(self.adata, color="pheno_leiden",
                       legend_fontoutline=2, show=False, add_outline=False, frameon=False,
                       title="UMAP Plot", palette=self.palette,
                       s=self.dotsize, save=".".join(["".join([str(self.tool), "_cluster"]), "pdf"]))
            sc.pl.umap(self.adata, color="pheno_leiden",
                       legend_fontoutline=4, show=False, add_outline=False, frameon=False,
                       legend_loc='on data', title="UMAP Plot", palette=self.palette,
                       s=self.dotsize, save="_legend_on_data.".join(["".join([str(self.tool), "_cluster"]), "pdf"]))
            # format svg
            sc.pl.umap(self.adata, color="pheno_leiden",
                       legend_fontoutline=4, show=False, add_outline=False, frameon=False,
                       legend_loc='on data', title="UMAP Plot", palette=self.palette,
                       s=self.dotsize, save="_legend_on_data.".join(["".join([str(self.tool), "_cluster"]), "svg"]))
            # plot umap with info file condition
            for _ in ['Sample', 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    sc.pl.umap(self.adata, color=_, legend_fontoutline=2, show=False, add_outline=False,
                               frameon=False,
                               title="UMAP Plot",
                               s=self.dotsize, save=".".join(["_".join([str(self.tool), _]), "pdf"]))
                else:
                    continue
            # plot umap grouped with gray background
            for _ in ['Cell_type', 'EXP', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    for batch in list(self.adata.obs[_].unique()):
                        sc.pl.umap(self.adata, color=_, groups=[batch], na_in_legend=False,
                                   title="UMAP Plot",
                                   legend_fontoutline=2, show=False, add_outline=False, frameon=False,
                                   s=self.dotsize, save=".".join(["_".join([_ + str(batch), _]), "pdf"]))
                else:
                    continue
            # scale data
            self.adata.X = self.adata.layers['raw_value']
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.X)
            for _ in list(self.adata.var_names.unique()):
                if self.scaler is True:
                    sc.pl.umap(self.adata, color=_, show=False, layer="raw_value",
                               legend_fontoutline=1, na_in_legend=False, s=self.dotsize,
                               title=_, cmap='turbo', groups=[_],
                               save=".".join([''.join(e for e in _ if e.isalnum()), "pdf"])
                               )
                else:
                    sc.pl.umap(self.adata, color=_, show=False, layer="scaled01",
                               legend_fontoutline=1, na_in_legend=False, s=self.dotsize,
                               title=_, cmap='turbo', groups=[_],
                               save=".".join([''.join(e for e in _ if e.isalnum()), "pdf"])
                               )
        elif self.runtime == 'UMAP':
            sc.settings.figdir = self.outfig
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.adata.layers['scaled01'] = scaler.fit_transform(self.adata.X)
            for _ in list(self.adata.var_names.unique()):
                sc.pl.umap(self.adata, color=_, show=False, layer="scaled01",
                           legend_fontoutline=1, na_in_legend=False, s=self.dotsize, frameon=False,
                           title=_, cmap='turbo', groups=[_],
                           save=".".join([''.join(e for e in _ if e.isalnum()), "pdf"])
                           )
            for _ in ['Sample', 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    sc.pl.umap(self.adata, color=_,
                               cmap=self.palette, legend_fontoutline=2, show=False, add_outline=False,
                               frameon=False,
                               title="UMAP Plot",
                               s=self.dotsize, save=".".join(["_".join([str(self.tool), _]), "pdf"]))
                else:
                    continue
            for _ in ['Cell_type', 'EXP', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    for batch in list(self.adata.obs[_].unique()):
                        sc.pl.umap(self.adata, color=_, groups=[batch], na_in_legend=False,
                                   title="UMAP Plot",
                                   legend_fontoutline=2, show=False, add_outline=False, frameon=False,
                                   s=self.dotsize, save=".".join(["_".join([_ + str(batch), _]), "pdf"])
                                   )
                else:
                    continue
        elif self.runtime == 'Clustering':
            pass

    def plot_umap_expression(self):
        """
        Function to plot the UMAP expression for all variables.
        Returns: Combined UMAP plot with all variables.
        """
        if self.runtime == 'Full':
            self.subsample_adata_plotting()
            self.adata_downsampled.obs['Clustering'] = self.adata_downsampled.obs['pheno_leiden'].astype(str)

            umapFiles = ["umap" + ".".join([''.join(e for e in _ if e.isalnum()), "pdf"]) for _ in list(self.adata.var_names.unique())]
            umapFiles.append("umap" + ".".join([str(self.tool) + "_cluster", "pdf"]))
            numberFiles = len(umapFiles)
            if numberFiles == 0: return
            numberColumns = 4
            numberRows = (numberFiles // numberColumns) + (numberFiles % numberColumns > 0)
            clusterFileIndex = (numberRows - 1) * numberColumns
            umapFiles.insert(clusterFileIndex, umapFiles.pop())
            for file in range(numberFiles):
                pdfFile = open("/".join([self.outfig, "UMAP", umapFiles[file]]), 'rb')
                reader = PdfReader(pdfFile)
                page = reader.pages[0]
                if file % numberColumns == 0:
                    if file == 0:
                        width = page.mediabox.width
                        height = page.mediabox.height
                        mergedPage = PageObject.create_blank_page(None, width * numberColumns, height * numberRows)
                    else:
                        mergedPage.add_transformation(Transformation().translate(0, height))
                        mergedPage.merge_page(pageRow)
                    pageRow = PageObject.create_blank_page(None, width * numberColumns, height)
                if file == clusterFileIndex + 1:
                    pageRow.add_transformation(Transformation().scale(.95, 1))
                pageRow.add_transformation(Transformation().translate(width, 0))
                pageRow.merge_page(page)
                if file == numberFiles - 1:
                    mergedPage.add_transformation(Transformation().translate(0, height))
                    mergedPage.merge_page(pageRow)
            writer = PdfWriter()
            writer.add_page(mergedPage)
            with open("/".join([self.outfig, "UMAP", "umap" + ".".join([str(self.tool) + "_ALL", "pdf"])]), 'wb') as f:
                writer.write(f)
        else:
            pass

    def plot_frequency(self):
        """
        Plot barplot with frequency.
        Returns: Barplot with frequency.
        """
        if self.runtime != 'UMAP':
            self.FREQUENCY_folder = "/".join([self.outfig, "BARPLOT_FREQUENCY"])
            self.createdir(self.FREQUENCY_folder)
            fig, (ax1) = plt.subplots(1, 1, figsize=(17 / 2.54, 17 / 2.54))
            ax1 = self.adata.obs.groupby("pheno_leiden")["Sample"].value_counts(
                normalize=True).unstack().plot.barh(
                stacked=True,
                legend=False,
                ax=ax1,
                color=self.palette, linewidth=2)
            ax1.set_xlabel("Percentage Frequency")
            ax1.set_ylabel("Cluster")
            ax1.grid(False)
            ax1.legend(bbox_to_anchor=(1.2, 1.0))
            ### save
            fig.savefig("/".join([self.FREQUENCY_folder, ".".join(["ClusterFrequencyNormalized", "pdf"])]),
                        dpi=self.dpi, bbox_inches='tight',
                        format="pdf")
            fig.savefig("/".join([self.FREQUENCY_folder, ".".join(["ClusterFrequencyNormalized", "svg"])]),
                        dpi=self.dpi, bbox_inches='tight',
                        format="svg")
            #
            for _ in ['Sample', 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    fig, (ax3) = plt.subplots(1, 1, figsize=(17 / 2.54, 17 / 2.54))
                    ax3 = self.adata.T.var.groupby(_)["pheno_leiden"].value_counts(
                        normalize=True).unstack().plot.barh(
                        stacked=True,
                        legend=False,
                        color=self.palette,
                        ax=ax3,
                        fontsize=5, linewidth=2)
                    ax3.set_xlabel("Cluster Percentage Frequency")
                    ax3.set_ylabel(_)
                    ax3.grid(False)
                    ax3.legend(bbox_to_anchor=(1.2, 1.0))
                    fig.savefig("/".join(
                        [self.FREQUENCY_folder, ".".join(["".join([_, "ClusterFrequencyNormalized"]), "pdf"])]),
                                dpi=self.dpi, bbox_inches='tight',
                                format="pdf")
            #
            fig, (ax2) = plt.subplots(1, 1, figsize=(17 / 2.54, 17 / 2.54))
            ax2 = self.adata.obs.groupby("pheno_leiden")["Sample"].value_counts(
                normalize=False).unstack().plot.barh(
                stacked=True,
                legend=False, ax=ax2, color=self.palette, linewidth=5)
            ax2.set_xlabel("Relative Frequency")
            ax2.set_ylabel("Cluster")
            ax2.grid(False)
            ax2.legend(bbox_to_anchor=(1.2, 1.0), fontsize=12, frameon=False, title = _)
            fig.savefig("/".join([self.FREQUENCY_folder, ".".join(["ClusterFrequencyNotNormalized", "pdf"])]),
                        dpi=self.dpi, bbox_inches='tight',
                        format="pdf")
            fig.savefig("/".join([self.FREQUENCY_folder, ".".join(["ClusterFrequencyNotNormalized", "svg"])]),
                        dpi=self.dpi, bbox_inches='tight',
                        format="svg")
            self.plot_frequency_ptz()
        else:
            pass

    def plot_frequency_ptz(self):
        """
        Function to plot frequency of clusters per combination of variables (Time_point, Condition, Sample, Cell_type, EXP).
        Returns: Barplots with frequency per combination of variables.
        """
        # Ensure the required columns are present
        required_columns = ["Time_point", "Condition", "Sample", "Cell_type", "EXP", "pheno_leiden"]
        missing_columns = [col for col in required_columns if col not in self.adata.obs.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Initialize a list to store the plots
        fig_list = []

        # Define combinations of columns to loop over
        group_combinations = ["Time_point", "Condition", "Sample", "Cell_type", "EXP"]

        for group_col in group_combinations:  # Iterate through each column name
            # Check if there is more than one unique value in group_col
            unique_values = self.adata.obs[group_col].drop_duplicates()
            if len(unique_values) <= 1:
                self.log.info(f"Skipping plot for {group_col} as there is only one unique value.")
                continue

            # Compute frequencies for each combination of group_col and pheno_leiden
            df = self.adata.obs.groupby([group_col, "pheno_leiden"]).size().unstack(fill_value = 0)

            # Normalize to percentage
            df = df.divide(df.sum(axis = 1), axis = 0) * 100

            # Replace NaN or infinite values with zero
            df.replace([np.inf, -np.inf], np.nan, inplace = True)
            df.fillna(0, inplace = True)

            # Create a new figure for each plot
            fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = True, figsize = (20, 10))
            fig_list.append(fig)  # Add the figure to the list

            if not df.empty and np.isfinite(df.values).all():
                # Perform hierarchical clustering and plot dendrogram
                Z = linkage(df, method = 'ward', optimal_ordering = True)
                dn = dendrogram(Z, get_leaves = True, orientation = 'left', labels = df.index, no_plot = True)
                dendrogram(Z, get_leaves = True, orientation = 'left', labels = df.index,
                           color_threshold = 0, above_threshold_color = 'k', ax = ax1)
                df.loc[dn['ivl']].plot.barh(stacked = True, ax = ax2, color = self.palette, legend = False,
                                            linewidth=1)

                # Set axis labels and appearance
                ax1.set(yticklabels = [], xticklabels = [])
                ax1.grid(False)
                ax2.tick_params(left = False)
                ax2.grid(False)
                ax1.axis('off')
                ax2.set_ylabel(" ")
                ax2.set_xlabel("Percentage Frequency")
                ax2.legend(bbox_to_anchor = (1.2, 1.0), title = 'Cluster',fontsize=12, frameon=False)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
            else:
                ax1.text(0.5, 0.5, f'No data available for {group_col}', horizontalalignment = 'center',
                         verticalalignment = 'center', fontsize = 12, transform = ax1.transAxes)
                ax2.text(0.5, 0.5, f'No data available for {group_col}', horizontalalignment = 'center',
                         verticalalignment = 'center', fontsize = 12, transform = ax2.transAxes)

            # Save the figure
            fig.savefig(f"{self.FREQUENCY_folder}/SampleFrequency_{group_col}_Clusterized.pdf",
                        dpi = self.dpi, bbox_inches = 'tight', format = "pdf")

    def plot_cell_clusters(self):
        """
        Plot the cell clusters on the UMAP projection.
        Returns: A PDF file with cell clusters plotted.
        """
        if self.runtime == 'Full':
            self.umap = pd.DataFrame(self.adata_downsampled.obsm['X_umap'], index=self.adata_downsampled.obs_names)
            clusters = self.adata_downsampled.obs['pheno_leiden']
            tsne = self.umap.copy()
            tsne.columns = ['x', 'y']

            # Cluster colors
            n_clusters = len(set(clusters))
            cluster_colors = pd.Series(
                sns.color_palette(self.palette, n_clusters), index=set(clusters)
            )

            # Set up figure
            n_cols = 6
            n_rows = int(np.ceil(n_clusters / n_cols))
            fig = plt.figure(figsize=[2 * n_cols, 2 * (n_rows + 2)], dpi=300)
            gs = plt.GridSpec(
                n_rows + 2, n_cols, height_ratios=np.append([0.75, 0.75], np.repeat(1, n_rows))
            )

            # Clusters
            ax = plt.subplot(gs[0:2, 2:4])
            ax.scatter(tsne["x"], tsne["y"], s=6, color=cluster_colors[clusters[tsne.index]])
            ax.set_axis_off()

            # Branch probabilities
            for i, cluster in enumerate(set(clusters)):
                row = int(np.floor(i / n_cols))
                ax = plt.subplot(gs[row + 2, i % n_cols])
                ax.scatter(tsne.loc[:, "x"], tsne.loc[:, "y"], s=3, color="lightgrey")
                cells = clusters.index[clusters == cluster]
                ax.scatter(
                    tsne.loc[cells, "x"],
                    tsne.loc[cells, "y"],
                    s=3,
                    color=cluster_colors[cluster],
                )
                ax.set_axis_off()
                ax.set_title(cluster, fontsize=10)
            fig.tight_layout()
            fig.savefig("/".join([self.UMAP_folder, ".".join(["umapCELL_clusters_all", "pdf"])]))
            plt.close(fig)
        else:
            pass

    def plot_cell_obs(self):
        """
        Plot UMAP projections with cell-type information overlayed.
        Returns: UMAP plots saved in the specified format.
        """
        sc.settings.figdir = self.UMAP_folder
        if self.runtime != 'Clustering':
            for _ in ['Cell_type', 'EXP', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    sc.pl.umap(self.adata_downsampled,
                               color=['Clustering', _],
                               show=False,
                               layer="scaled01",
                               legend_fontoutline=1, frameon=False,
                               na_in_legend=False, s=self.dotsize, cmap='turbo',
                               save=".".join(["".join([str(self.tool), _ + "_ALL"]), "pdf"])
                               )
                else:
                    continue
        else:
            pass

    def matrixplot(self):
        """
        Function for the generation of matrixplot sc.pl.matrixplot.
        Returns: matrixplot.
        """
        self.matrixplot_folder = "/".join([self.outfig, "HEATMAP"])
        self.createdir(self.matrixplot_folder)
        sc.settings.figdir = self.matrixplot_folder
        if self.runtime != 'UMAP':
            sc.pl.matrixplot(sc.pp.scale(self.adata,max_value=6,copy=True,layer="raw_value"),
                             list(self.adata.var_names), "pheno_leiden",
                             dendrogram=True, vmin=-1, vmax=1, cmap='RdBu_r',layer = "raw_value",
                             show=False, swap_axes=False, return_fig=False, use_raw=False, log=False,
                             save=".".join(["matrixplot_mean_z_score", "pdf"]))
            sc.pl.matrixplot(sc.pp.scale(self.adata,max_value=6,copy=True,layer="raw_value"),
                             list(self.adata.var_names), "pheno_leiden",
                             dendrogram=True, vmin=-1, vmax=1, cmap='RdBu_r', use_raw=False, log=False,
                             show=False, swap_axes=False, return_fig=False,layer = "raw_value",
                             save=".".join(["matrixplot_mean_z_score", "svg"]))
            sc.pl.matrixplot(self.adata, list(self.adata.var_names), "pheno_leiden",
                             dendrogram=True, cmap='Blues', standard_scale='var', layer = "raw_value",
                             colorbar_title='column scaled\nexpression',
                             swap_axes=False, return_fig=False,
                             show=False,
                             save=".".join(["matrixplot_column_scaled_expression", "pdf"]))
        else:
            pass

    def subsample_adata_plotting(self):
        """
        Subsample the AnnData object for plotting.
        Returns: Subsampled AnnData object.
        """
        np.random.seed(42)
        indices = np.random.choice(self.adata.obs.index, size=min(5000, len(self.adata.obs.index)), replace=False)
        self.adata_downsampled = self.adata[indices].copy()

    def createdir(self, directory):
        """
        Helper function to create a directory if it doesn't exist.
        :param directory: The path of the directory to create.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def plot_pacmap(self):
        """
        Function for generating PDF files with PACMAP plots.
        Returns: PDF files with PACMAP plots.
        """
        if self.runtime == 'Full':
            # create output directory
            self.outfig = self.output_folder
            self.PACMAP_folder = "/".join([self.outfig, "PACMAP"])
            self.createdir(self.PACMAP_folder)
            sc.settings.figdir = self.PACMAP_folder
            self.palette = color_palette("husl", len(self.adata.obs["pheno_leiden"].unique()))
            self.dotsize = 30
            # # set palette
            # if len(self.adata.obs["pheno_leiden"].unique()) < 28:
            #     self.palette = self.palette28
            # else:
            #     self.palette = color_palette("husl", len(self.adata.obs["pheno_leiden"].unique()))

            # plot umap + clustering
            # Prepare PACMAP model
            pacmap_model = pacmap.PaCMAP(n_components = 2, n_neighbors = 10, MN_ratio = 0.5, FP_ratio = 2.0)

            # Fit and transform data
            self.adata.obsm['X_umap'] = pacmap_model.fit_transform(self.adata.X)
            sc.pl.umap(self.adata, color = "pheno_leiden",
                       legend_fontoutline = 2, show = False, add_outline = False, frameon = False,
                       title = "UMAP Plot", palette = self.palette,
                       s = self.dotsize, save = ".".join(["".join([str(self.tool), "_cluster"]), "pdf"]))
            sc.pl.umap(self.adata, color = "pheno_leiden",
                       legend_fontoutline = 4, show = False, add_outline = False, frameon = False,
                       legend_loc = 'on data', title = "UMAP Plot", palette = self.palette,
                       s = self.dotsize, save = "_legend_on_data.".join(["".join([str(self.tool), "_cluster"]), "pdf"]))
            # format svg
            sc.pl.umap(self.adata, color = "pheno_leiden",
                       legend_fontoutline = 4, show = False, add_outline = False, frameon = False,
                       legend_loc = 'on data', title = "UMAP Plot", palette = self.palette,
                       s = self.dotsize, save = "_legend_on_data.".join(["".join([str(self.tool), "_cluster"]), "svg"]))
            # plot umap with info file condition
            for _ in ['Sample', 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    sc.pl.umap(self.adata, color = _, legend_fontoutline = 2, show = False, add_outline = False,
                               frameon = False,
                               title = "UMAP Plot",
                               s = self.dotsize, save = ".".join(["_".join([str(self.tool), _]), "pdf"]))
                else:
                    continue
            # plot umap grouped with gray background
            for _ in ['Cell_type', 'EXP', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    for batch in list(self.adata.obs[_].unique()):
                        sc.pl.umap(self.adata, color = _, groups = [batch], na_in_legend = False,
                                   title = "UMAP Plot",
                                   legend_fontoutline = 2, show = False, add_outline = False, frameon = False,
                                   s = self.dotsize, save = ".".join(["_".join([_ + str(batch), _]), "pdf"]))
                else:
                    continue
            # scale data
            self.adata.X = self.adata.layers['raw_value']
            self.scaler = MinMaxScaler(feature_range = (0, 1))
            self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.X)
            for _ in list(self.adata.var_names.unique()):
                if self.scaler is True:
                    sc.pl.umap(self.adata, color = _, show = False, layer = "raw_value",
                               legend_fontoutline = 1, na_in_legend = False, s = self.dotsize,
                               title = _, cmap = 'turbo', groups = [_],
                               save = ".".join([''.join(e for e in _ if e.isalnum()), "pdf"])
                               )
                else:
                    sc.pl.umap(self.adata, color = _, show = False, layer = "scaled01",
                               legend_fontoutline = 1, na_in_legend = False, s = self.dotsize,
                               title = _, cmap = 'turbo', groups = [_],
                               save = ".".join([''.join(e for e in _ if e.isalnum()), "pdf"])
                               )
        elif self.runtime == 'UMAP':
            sc.settings.figdir = self.outfig
            scaler = MinMaxScaler(feature_range = (0, 1))
            self.adata.layers['scaled01'] = scaler.fit_transform(self.adata.X)
            for _ in list(self.adata.var_names.unique()):
                sc.pl.umap(self.adata, color = _, show = False, layer = "scaled01",
                           legend_fontoutline = 1, na_in_legend = False, s = self.dotsize, frameon = False,
                           title = _, cmap = 'turbo', groups = [_],
                           save = ".".join([''.join(e for e in _ if e.isalnum()), "pdf"])
                           )
            for _ in ['Sample', 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    sc.pl.umap(self.adata, color = _,
                               cmap = self.palette, legend_fontoutline = 2, show = False, add_outline = False,
                               frameon = False,
                               title = "UMAP Plot",
                               s = self.dotsize, save = ".".join(["_".join([str(self.tool), _]), "pdf"]))
                else:
                    continue
            for _ in ['Cell_type', 'EXP', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    for batch in list(self.adata.obs[_].unique()):
                        sc.pl.umap(self.adata, color = _, groups = [batch], na_in_legend = False,
                                   title = "UMAP Plot",
                                   legend_fontoutline = 2, show = False, add_outline = False, frameon = False,
                                   s = self.dotsize, save = ".".join(["_".join([_ + str(batch), _]), "pdf"])
                                   )
                else:
                    continue
        elif self.runtime == 'Clustering':
            pass
