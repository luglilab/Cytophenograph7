import warnings
import anndata
import glob
import os
import sys
import pandas as pd
import phenograph as pg
import scanpy as sc
import pyVIA.core as via
import logging
import scanpy.external as sce
import tempfile
import matplotlib
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import fcsy
from fcsy import DataFrame
import subprocess
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import scprep
import anndata as ad
import pacmap
import umap
from pypdf import PdfWriter, PdfReader, PageObject, Transformation
import flowsom as fs
matplotlib.use('Agg')
import seaborn as sns
warnings.filterwarnings('ignore')


tmp = tempfile.NamedTemporaryFile()
sc.settings.autoshow = False
sc.settings.set_figure_params(dpi = 300, facecolor = 'white', dpi_save = 330,
                              figsize = (10, 10))
sc.settings.verbosity = 0
warnings.filterwarnings("ignore", category = FutureWarning)
from palette import palette28,palette102


class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.INFO: "###%(msg)s",
        logging.WARNING: "$$$%(msg)s",
        logging.ERROR: "@@@%(msg)s",
        "DEFAULT": "%(msg)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Cytophenograph:
    def __init__(self, info_file, input_folder, output_folder, k_coef, marker_list, analysis_name, thread, tool, batch,
                 batchcov, mindist, spread, runtime, knn, resolution, maxclus, downsampling, cellnumber,
                 filetype,arcsinh):
        self.info_file = info_file
        self.input_folder = input_folder
        self.output_folder = output_folder
        # self.k_coef = k_coef
        self.marker_list = marker_list
        # analysis file name 
        import re
        self.analysis_name = re.sub('[\W]+', '', analysis_name.replace(' ', '_'))
        self.thread = thread
        self.tool = tool
        self.tmp_df = pd.DataFrame()
        self.adata = None
        self.adata = None
        self.embedding = None
        self.palette = None
        self.marker = None
        self.markertoinclude = None
        self.marker_array = None
        self.flowsomshape = None
        self.anndata_list = []
        self.outfig = None
        self.tmp = None
        self.dpi = 300
        self.fileformat = "pdf"  # insert svg to change figure format
        self.newheader = []
        self.n_neighbors = 5
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.scanorama = batch
        self.batchcov = batchcov
        self.runtime = runtime
        self.cleaning = {}
        self.target_cells = 0.1

        if self.tool == "Phenograph":
            self.k_coef = k_coef
        if self.tool == "VIA":
            self.knn = knn
            self.resolution = resolution
        self.maxclus = int(maxclus)
        #self.flowsomDF = pd.DataFrame()
        self.listmarkerplot = None
        self.concatenate_fcs = None
        self.path_flowai = os.path.dirname(os.path.realpath(__file__)) + '/flowai.Rscript'
        # self.path_flowsom = os.path.dirname(os.path.realpath(__file__)) + '/flowsom.Rscript'
        self.path_cycombine = os.path.dirname(os.path.realpath(__file__)) + '/cycombine.Rscript'
        self.mindist = float(mindist)
        self.spread = float(spread)
        self.downsampling = downsampling
        self.cellnumber = cellnumber
        self.filetype = filetype
        if self.filetype == "FCS":
            self.arcsinh = arcsinh
        else:
            self.arcsinh = False
        self.root_user = [1]
        self.fnull = open(os.devnull, 'w')

        ch = logging.StreamHandler()
        ch.setFormatter(CustomFormatter())
        self.log.addHandler(ch)
        self.palette28 = palette28
        self.palette102 = palette102

        if self.runtime == 'UMAP': self.tool = 'UMAP'
        self.log.info("""
        #     _____      _              _                                            _      __      ________ 
        #    / ____|    | |            | |                                          | |     \ \    / /____  |
        #   | |    _   _| |_ ___  _ __ | |__   ___ _ __   ___   __ _ _ __ __ _ _ __ | |__    \ \  / /    / / 
        #   | |   | | | | __/ _ \| '_ \| '_ \ / _ \ '_ \ / _ \ / _` | '__/ _` | '_ \| '_ \    \ \/ /    / /  
        #   | |___| |_| | || (_) | |_) | | | |  __/ | | | (_) | (_| | | | (_| | |_) | | | |    \  /    / /   
        #    \_____\__, |\__\___/| .__/|_| |_|\___|_| |_|\___/ \__, |_|  \__,_| .__/|_| |_|     \(_)  /_/    
        #           __/ |        | |                            __/ |         | |                            
        #          |___/         |_|                           |___/          |_|                            """)
        self.log.info("Runtime: {}".format(self.runtime))
        self.log.info("DownSampling: {}".format(self.downsampling))
        if self.downsampling != 'All':
            self.log.info(" >> Event Number: {}".format(self.cellnumber))
        self.log.warning("PART 1")

    def read_fcs(self, path_csv_file):
        """
        Read FCS file version 3 and convert in pandas dataframe
        Returns: Pandas Dataframe

        """
        df = DataFrame.from_fcs(path_csv_file, channel_type = 'multi')
        df.columns = df.columns.map(' :: '.join)
        df.columns = df.columns.str.replace('[\",\']', '')
        df.columns = df.columns.str.rstrip(': ')

        if self.downsampling == "Balanced":
            if self.cellnumber < df.shape[0]:
                df = df.sample(n = int(self.cellnumber), random_state = 42,
                               ignore_index = True)
            else:
                pass
        barcode = []
        names = os.path.basename(path_csv_file)
        num_lines = df.shape[0]
        for _ in range(1, num_lines + 1):
            barcode.append("_".join([names.split(".")[0], str(_)]))
        df.index = barcode
        return df

    def read_info_file(self):
        """
        Read info file methods
        :return: pandas dataframe
        """
        df_info = pd.read_excel(self.info_file, header = 0)
        return df_info

    def import_all_event(self):
        """
        scan csv folder, save csv files names in list
        :return: list
        """
        # change folder
        os.chdir(self.input_folder)
        # create array with all csvfile name and path
        if self.filetype == "CSV":
            all_files = glob.glob(os.path.join(self.input_folder, "*.csv"))
        else:
            all_files = glob.glob(os.path.join(self.input_folder, "*.fcs"))
        # get files name
        names = [os.path.basename(x) for x in all_files]
        #
        list_with_file_name_and_path = []
        for file_, name in zip(all_files, names):
            list_with_file_name_and_path.append(file_)
        return list_with_file_name_and_path

    def create_df(self, path_csv_file):
        """
        create dataframe file csv
        :return: pandas dataframe
        """
        df = pd.read_csv(path_csv_file, header = 0)
        if self.downsampling == "Balanced":
            if self.cellnumber < df.shape[0]:
                df = df.sample(n = int(self.cellnumber), random_state = 42,
                               ignore_index = True)
            else:
                # self.log.info("It was not possible to downsample because the fixed number of events is greater than the original number of events. Decrease the threshold for downsampling.")
                pass
        barcode = []
        names = os.path.basename(path_csv_file)
        num_lines = df.shape[0]
        for _ in range(1, num_lines + 1):
            barcode.append("_".join([names.split(".")[0], str(_)]))
        df.index = barcode
        return df

    def concatenate_dataframe(self, info_file, csv_list):
        """

        :param csv_list:
        :return:
        """
        self.log.info("Files concatenation")
        # create empy list for save several df
        pandas_df_list = []
        # create list with anndata object
        # loop over csv file name
        for i in range(len(csv_list)):
            if self.filetype == "CSV":
                # append df in pandas_df_list
                pandas_df_list.append(self.create_df(csv_list[i]))
            else:
                pandas_df_list.append(self.read_fcs(csv_list[i]))
        # check header
        if all([len(pandas_df_list[0].columns.intersection(df.columns)) == pandas_df_list[0].shape[1]
                for df in pandas_df_list]):
            try:
                for i in range(len(pandas_df_list)):
                    # save column with Sample name in list
                    Sample_list = info_file["Sample"].tolist()
                    # check if Sample name are in the anndata index
                    if pandas_df_list[i].index[0][:-2] in Sample_list:
                        ann_tmp = anndata.AnnData(pandas_df_list[i])
                        ann_tmp.obs['Sample'] = pandas_df_list[i].index[0][:-2]
                        #
                        cell_type = info_file['Cell_type'].loc[info_file['Sample'] == pandas_df_list[i].index[0][:-2]]
                        ann_tmp.obs['Cell_type'] = ''.join(e for e in cell_type.to_string().split(" ")[-1] if e.isalnum())
                        #
                        exp = info_file['EXP'].loc[info_file['Sample'] == pandas_df_list[i].index[0][:-2]]
                        ann_tmp.obs['EXP'] = ''.join(e for e in exp.to_string().split(" ")[-1] if e.isalnum())
                        #
                        id = info_file['ID'].loc[info_file['Sample'] == pandas_df_list[i].index[0][:-2]]
                        ann_tmp.obs['ID'] = ''.join(e for e in id.to_string().split(" ")[-1] if e.isalnum())
                        #
                        time_point = info_file['Time_point'].loc[info_file['Sample'] == pandas_df_list[i].index[0][:-2]]
                        # ann_tmp.obs['Time_point'] = time_point.to_string().split(" ")[-1]
                        ann_tmp.obs['Time_point'] = ''.join(e for e in time_point.to_string().split(" ")[-1] if e.isalnum())
                        #

                        condition = info_file['Condition'].loc[info_file['Sample'] == pandas_df_list[i].index[0][:-2]]
                        ann_tmp.obs['Condition'] = ''.join(e for e in condition.to_string().split(" ")[-1] if e.isalnum())
                        #
                        count = info_file['Count'].loc[info_file['Sample'] == pandas_df_list[i].index[0][:-2]]
                        ann_tmp.obs['Count'] = ''.join(e for e in count.to_string().split(" ")[-1] if e.isalnum())
                        self.anndata_list.append(ann_tmp)
                    else:
                        self.log.error("Error, this file {0} is not in the column Sample of Info File.\nPlease check sample name and Info File".format(pandas_df_list[i].index[0][:-2]))
                        sys.exit(1)
                if len(self.anndata_list) == 1:
                    self.adata = self.anndata_list[0]
                    self.adata.layers['raw_value'] = self.adata.X
                else:
                    tmp = self.anndata_list[0]
                    self.anndata_list.pop(0)
                    self.adata = tmp.concatenate(self.anndata_list)
                    newheader = []
                    for _ in list(self.adata.var_names):
                        newheader.append(_.split(":: ")[-1])
                    self.adata.var_names = newheader
                    self.adata.layers['raw_value'] = self.adata.X
            except (ValueError, Exception):
                self.log.error("Error. Please check Info File Header or Data Files Header.")
                sys.exit(1)
        else:
            self.log.error("Error. Please check Info File Header or Data Files Header.")
            sys.exit(1)
        self.tmp_df = pd.DataFrame(self.adata.X, index=self.adata.obs.index)
        if self.downsampling == "Fixed":
            if self.cellnumber < self.adata.shape[0]:
                sc.pp.subsample(self.adata, n_obs=self.cellnumber, random_state=42)
            else:
                pass
        self.cleaning.update({"Before QC":self.adata.shape[0]})
        non_zero_cells_mask = (self.adata.X > 0).all(axis = 1)
        self.adata = self.adata[non_zero_cells_mask]
        self.log.info("{0} cells undergo to clustering analysis".format(self.adata.shape[0]))
        self.adata.layers['raw_value'] = self.adata.X
        self.adata.raw = self.adata
        return self.adata.var_names_make_unique()

    def transformation(self):
        if (self.filetype == "FCS") and (self.arcsinh == True):
            self.adata.layers['arcin'] = scprep.transform.arcsinh(self.adata.X, cofactor=150)

    def create_barplot(self):
        """
        Create a barplot and export with the self.cleaning dictionary
        :return:
        """
        self.QC_folder = "/".join([self.outfig, "QC_PLOTS"])
        self.createdir(self.QC_folder)
        ax = sns.barplot(data = pd.DataFrame.from_dict(self.cleaning, orient = 'index').reset_index(),
                    y = 0, x = 'index')
        ax.bar_label(ax.containers[0], fmt = '%.0f')
        plt.grid(False)
        plt.ylabel("Number of cells")
        plt.xlabel("Cleaning steps")
        plt.savefig("/".join([self.QC_folder, ".".join(["cleaning", self.fileformat])]), dpi = 300, bbox_inches = 'tight', format = self.fileformat)

    def correct_scanorama(self):
        """
        This function runs Scanorama
        Returns: corrected adata
        """
        print(self.adata)

        self.adata = self.adata[:, self.markertoinclude].copy()
        self.adata.layers['raw_value'] = self.adata.X
        self.adata.layers['scaled'] = sc.pp.scale(self.adata, max_value = 6,
                                                         zero_center = True, copy = True).X
        self.anndata_list = [self.adata[self.adata.obs[self.batchcov] == i] for i in
                             self.adata.obs[self.batchcov].unique()]
        self.corrected = scanorama.correct_scanpy(self.anndata_list,
                                                  return_dense = True,
                                                  return_dimred = True,
                                                  approx = False,
                                                  verbose = 0,
                                                  seed = 42)
        self.corrected_dataset = self.corrected[0].concatenate(self.corrected[1:],
                                                               join = 'inner',
                                                               batch_key = self.batchcov)
        self.corrected_dataset.layers['raw_value'] = self.adata.layers['raw_value']
        self.corrected_dataset.layers['scaled'] = self.adata.layers['scaled']
        return self.corrected_dataset

    def loadmarkers(self):
        """
        Read marker filename with path
        :return: array with filepath and filename
        """
        if os.path.exists(self.marker_list) is True:
            markerfilepath = os.path.split(self.marker_list)[0]
            markerfilename = os.path.split(self.marker_list)[1]
            return markerfilepath, markerfilename
        else:
            self.log.error("File does not exist.")
            sys.exit(1)

    def checkmarkers(self):
        """
        Check if marker in file is also a column of conc file
        :return:
        """
        # read marker file
        self.marker_array = [line.rstrip() for line in open(self.marker_list)]
        newmarker = []
        for _ in self.marker_array:
            newmarker.append(_.split(":: ")[-1])
        self.marker_array = newmarker
        if len(self.anndata_list) > 1:
            # read concatenate file
            for i in range(len(self.marker_array)):
                if self.marker_array[i] in self.adata.var_names.to_list():
                    continue
                else:
                    self.log.error("Marker {} not found in Matrix.".format(self.marker_array[i]))
                    sys.exit(1)
            return self.marker_array
        else:
            return self.marker_array

    def splitmarker(self):
        """
        function for split marker in two list
        return: list of marker to include in analysis
        """
        self.marker = self.adata.var_names.to_list()
        self.markertoinclude = [i for i in self.marker if i not in self.marker_array]
        return self.markertoinclude

    def runumap(self):
        """
        Function for UMAP generation
        return: UMAP embedding
        """
        self.log.warning("PART 3")
        if self.runtime != 'Clustering':
            self.log.info("UMAP (Uniform Manifold Approximation and Projection) generation")
            #reducer = umap.UMAP(random_state = 42, n_neighbors = self.n_neighbors, min_dist = self.mindist,
            #                    spread = self.spread)
            # embedding = reducer.fit_transform(self.adata.X)
            #return embedding
            # cml_var_explained = np.cumsum(self.adata.uns['pca']['variance_ratio'])
            # min_cml_frac = 0.5
            # n_pcs = next(idx for idx, cml_frac in enumerate(cml_var_explained) if cml_frac > min_cml_frac)
            sc.pp.neighbors(self.adata, n_neighbors = 10, n_pcs = self.n_pcs)
            sc.tl.umap(self.adata)
        else:
            self.log.info("Skipping UMAP (Uniform Manifold Approximation and Projection) generation")

    def subsample_adata_plotting(self):
        """
        Function for subsample adata for plotting
        Returns: anndata object

        """
        if self.runtime != 'Clustering':
            adatas = [self.adata[self.adata.obs['pheno_leiden'].astype("category").isin([clust])] for clust in
                      self.adata.obs['pheno_leiden'].astype("category").cat.categories]
            for dat in adatas:
                    sc.pp.subsample(dat, fraction = self.target_cells, random_state = 42)

            self.adata_downsampled = adatas[0].concatenate(*adatas[1:])
            # return self.adata_downsampled
        else:
            self.log.info("Skipping subsampling adata for plotting")

    def plot_umap(self):
        """
        Function per generation of pdf files with umap plot
        return: pdf files
        """
        if self.runtime == 'Full':
            # create output directory
            self.outfig = self.output_folder
            self.UMAP_folder = "/".join([self.outfig, "UMAP"])
            self.createdir(self.UMAP_folder)
            sc.settings.figdir = self.UMAP_folder

            # set palette
            if len(self.adata.obs["pheno_leiden"].unique()) < 28:
                self.palette = self.palette28
            else:
                self.palette = self.palette102
            # plot umap + clustering
            sc.pl.umap(self.adata, color = "pheno_leiden",
                       legend_fontoutline = 2, show = False, add_outline = False, frameon = False,
                       title = "UMAP Plot", palette = self.palette,
                       s = 50, save = ".".join(["".join([str(self.tool), "_cluster"]), "pdf"]))
            sc.pl.umap(self.adata, color = "pheno_leiden",
                       legend_fontoutline = 4, show = False, add_outline = False, frameon = False,
                       legend_loc = 'on data', title = "UMAP Plot", palette = self.palette,
                       s = 50, save = "_legend_on_data.".join(["".join([str(self.tool), "_cluster"]), self.fileformat]))
            # format svg
            sc.pl.umap(self.adata, color = "pheno_leiden",
                       legend_fontoutline = 4, show = False, add_outline = False, frameon = False,
                       legend_loc = 'on data', title = "UMAP Plot", palette = self.palette,
                       s = 50, save = "_legend_on_data.".join(["".join([str(self.tool), "_cluster"]), 'svg']))
            # plot umap with info file condition
            for _ in ['Sample', 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    sc.pl.umap(self.adata, color = _, legend_fontoutline = 2, show = False, add_outline = False,
                               frameon = False,
                               title = "UMAP Plot",
                               s = 50, save = ".".join(["_".join([str(self.tool), _]), "pdf"]))
                else:
                    continue
            # plot umap grouped with gray background
            for _ in ['Cell_type', 'EXP', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    for batch in list(self.adata.obs[_].unique()):
                        sc.pl.umap(self.adata, color = _, groups = [batch], na_in_legend = False,
                                   title = "UMAP Plot",
                                   legend_fontoutline = 2, show = False, add_outline = False, frameon = False,
                                   s = 50, save = ".".join(["_".join([_ + str(batch), _]), "pdf"]))
                else:
                    continue
            # scale data
            self.scaler = MinMaxScaler(feature_range = (0, 1))
            self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.X)
            for _ in list(self.adata.var_names.unique()):
                if self.scaler is True:
                    sc.pl.umap(self.adata, color = _, show = False, layer = "raw_value",
                               legend_fontoutline = 1, na_in_legend = False, s = 30,
                               title = _, cmap = 'turbo', groups = [_],
                               save = ".".join([''.join(e for e in _ if e.isalnum()), self.fileformat])
                               )
                else:
                    sc.pl.umap(self.adata, color = _, show = False, layer = "scaled01",
                               legend_fontoutline = 1, na_in_legend = False, s = 30,
                               title = _, cmap = 'turbo', groups = [_],
                               save = ".".join([''.join(e for e in _ if e.isalnum()), self.fileformat])
                               )
        elif self.runtime == 'UMAP':
            sc.settings.figdir = self.outfig
            scaler = MinMaxScaler(feature_range = (0, 1))
            #print(self.adata.X)
            self.adata.layers['scaled01'] = scaler.fit_transform(self.adata.X)
            for _ in list(self.adata.var_names.unique()):
                sc.pl.umap(self.adata, color = _, show = False, layer = "scaled01",
                           legend_fontoutline = 1, na_in_legend = False, s = 30, frameon = False,
                           title = _, cmap = 'turbo', groups = [_],
                           save = ".".join([''.join(e for e in _ if e.isalnum()), self.fileformat])
                           )
            for _ in ['Sample', 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    sc.pl.umap(self.adata, color = _,
                               cmap = self.palette, legend_fontoutline = 2, show = False, add_outline = False,
                               frameon = False,
                               title = "UMAP Plot",
                               s = 50, save = ".".join(["_".join([str(self.tool), _]), "pdf"]))
                else:
                    continue
            for _ in ['Cell_type', 'EXP', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    for batch in list(self.adata.obs[_].unique()):
                        sc.pl.umap(self.adata, color = _, groups = [batch], na_in_legend = False,
                                   title = "UMAP Plot",
                                   legend_fontoutline = 2, show = False, add_outline = False, frameon = False,
                                   s = 50, save = ".".join(["_".join([_ + str(batch), _]), "pdf"])
                                   )
                else:
                    continue
            # self.plot_cell_obs()
        elif self.runtime == 'Clustering':
            pass

    def plot_expression_and_save_pdf(self, exp_column, clip_limit=(-5, 5),
                                     line_width=1.5):
        """
        Plot expression data and save as a PDF.

        Parameters:
        - adata: AnnData object containing the data.
        - exp_column: The column in adata.obs used for hue in the plot (e.g., 'EXP').
        - markers: List of gene markers to include in the plot.
        - output_pdf: Filename for the output PDF.
        - clip_limit: Tuple specifying the min and max clipping values for the density plot.
        - line_width: Width of the outline for the density plots.
        """
        # Downsampling function to reduce the dataset to 10,000 observations
        def downsample_data(adata, target_size=10000):
            # Calculate the number of samples to draw from each group based on proportions
            group_counts = self.adata_downsampled.obs[exp_column].value_counts(normalize = True) * target_size
            group_counts = group_counts.round().astype(int)

            # Sample indices for each group
            sampled_indices = []
            for group, count in group_counts.items():
                group_indices = adata.obs[adata.obs[exp_column] == group].index
                sampled_indices.extend(np.random.choice(group_indices, count, replace = False))

            # Return a downsampled AnnData object
            return adata[sampled_indices]

        # Downsample the data
        self.adata_10000 = downsample_data(self.adata_downsampled)

        # Transpose the data so that markers are columns and observations are rows
        df = pd.DataFrame(self.adata_10000.X, columns = self.adata_10000.var.index,
                          index = self.adata_10000.obs[exp_column])

        # Melt the DataFrame to long format
        df_long = df.reset_index().melt(id_vars = exp_column, var_name = 'Marker', value_name = 'Expression')

        # Initialize the FacetGrid object with Marker as the rows and EXP as the hue
        # Choose a diverging color palette
        pal = sns.color_palette("Spectral", df_long[exp_column].nunique())  # "Spectral" is a good diverging palette
        g = sns.FacetGrid(df_long, row = 'Marker', hue = exp_column, aspect = 15, height = 1.2, palette = pal,
                          sharex = False)

        # Draw the densities with outline
        def plot_density(data, color, **kwargs):
            sns.kdeplot(data = data, x = 'Expression', bw_adjust = 0.5, clip = clip_limit,
                        fill = True, alpha = 0.5, linewidth = line_width, common_norm = False, color = color, **kwargs)

        g.map_dataframe(plot_density)

        # Properly label each subplot with the corresponding Marker name
        def label_marker(ax, marker):
            ax.text(-0.1, 0, marker, fontweight = "bold", color = "black",
                    ha = "right", va = "center", transform = ax.transAxes, fontsize = 10)

        # Loop over each subplot to add the marker labels
        for ax, marker in zip(g.axes.flat, df_long['Marker'].unique()):
            label_marker(ax, marker)  # Call the label function with the current axis and marker

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace = -0.5)

        # Add a legend to the plot
        g.add_legend(title = f"{exp_column}", bbox_to_anchor = (1.05, 1), loc = 'upper left', borderaxespad = 0.)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks = [], ylabel = "", xlabel = "")

        # Hide x-axis ticks and labels for all but the last subplot
        for ax in g.axes.flat[:-1]:
            ax.set_xticks([])  # Remove x-ticks
            ax.set_xticklabels([])  # Remove x-tick labels

        # Remove the x-label from the last subplot
        g.axes.flat[-1].set_xlabel("")  # No x-label

        # Ensure x-ticks are visible for the last plot
        g.axes.flat[-1].tick_params(axis = 'x', which = 'both', length = 5)

        # Remove spines for a cleaner look
        g.despine(bottom = True, left = True)

        # Adjust the size of the plot to fit within A4 dimensions
        g.fig.set_size_inches(8.27, 11.69)  # A4 size in inches

        self.outfig = self.output_folder
        self.JOYPLOT = "/".join([self.outfig, "JOYPLOT"])
        ".".join(["".join([str(self.tool), "_cluster"]), "pdf"])
        # Save the plot to a PDF with high resolution
        plt.savefig("/".join([self.JOYPLOT, ".".join([exp_column, self.fileformat])]),
                    dpi = 300, bbox_inches = 'tight')

        # Show the plot
        plt.show()

    def plot_cell_clusters(self):
        if self.runtime == 'Full':
            self.umap = pd.DataFrame(self.adata_downsampled.obsm['X_umap'], index = self.adata_downsampled.obs_names)
            clusters = self.adata_downsampled.obs['pheno_leiden']
            tsne = self.umap.copy()
            tsne.columns = ['x', 'y']

            # Cluster colors
            n_clusters = len(set(clusters))
            cluster_colors = pd.Series(
                sns.color_palette(self.palette, n_clusters), index = set(clusters)
            )

            # Set up figure
            n_cols = 6
            n_rows = int(np.ceil(n_clusters / n_cols))
            fig = plt.figure(figsize = [2 * n_cols, 2 * (n_rows + 2)], dpi = 300)
            gs = plt.GridSpec(
                n_rows + 2, n_cols, height_ratios = np.append([0.75, 0.75], np.repeat(1, n_rows))
            )

            # Clusters
            ax = plt.subplot(gs[0:2, 2:4])
            ax.scatter(tsne["x"], tsne["y"], s = 6, color = cluster_colors[clusters[tsne.index]])
            ax.set_axis_off()

            # Branch probabilities
            for i, cluster in enumerate(set(clusters)):
                row = int(np.floor(i / n_cols))
                ax = plt.subplot(gs[row + 2, i % n_cols])
                ax.scatter(tsne.loc[:, "x"], tsne.loc[:, "y"], s = 3, color = "lightgrey")
                cells = clusters.index[clusters == cluster]
                ax.scatter(
                    tsne.loc[cells, "x"],
                    tsne.loc[cells, "y"],
                    s = 3,
                    color = cluster_colors[cluster],
                )
                ax.set_axis_off()
                ax.set_title(cluster, fontsize = 10)
            fig.tight_layout()
            fig.savefig("/".join([self.UMAP_folder, ".".join(["umapCELL_clusters_all", self.fileformat])]))
            plt.close(fig)
        else:
            pass

    def plot_umap_expression(self):
        if self.runtime == 'Full':
            self.subsample_adata_plotting()
            self.adata_downsampled.obs['Clustering'] = self.adata_downsampled.obs['pheno_leiden'].astype(str)

            umapFiles = ["umap" + ".".join([''.join(e for e in _ if e.isalnum()), self.fileformat]) for _ in list(self.adata.var_names.unique())]
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
            with open("/".join([self.outfig, "UMAP", "umap" + ".".join([str(self.tool) + "_ALL", self.fileformat])]), 'wb') as f:
                writer.write(f)

            # sc.pl.umap(self.adata_downsampled,
            #            color = ['Clustering'] + list(self.adata_downsampled.var_names),
            #            show = False,
            #            layer = "scaled01",
            #            legend_fontoutline = 1,frameon=False,
            #            na_in_legend = False, s = 50, cmap = 'turbo',
            #            save = ".".join(["".join([str(self.tool), "_ALL"]), self.fileformat])
            #            )
            # sc.pl.umap(self.adata_downsampled,
            #            color = ['Clustering'] + list(self.adata_downsampled.var_names),
            #            show = False,
            #            layer = "scaled01",
            #            legend_fontoutline = 1,frameon=False,
            #            na_in_legend = False, s = 50, cmap = 'turbo',
            #            save = ".".join(["".join([str(self.tool), "_ALL"]), 'svg'])
            #            )
        else:
            pass

    def find_obs_not_unique(self):
        """
        Find obs columns that are not unique
        Returns: list of obs columns that are not unique

        """
        self.obs_not_unique = []
        for _ in self.adata_downsampled.obs.columns:
            if len(self.adata_downsampled.obs[_].unique()) == 1:
                self.obs_not_unique.append(_)
            else:
                continue
        return self.obs_not_unique

    def plot_cell_obs(self):
        if self.runtime != 'Clustering':
            for _ in ['Cell_type', 'EXP', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    sc.pl.umap(self.adata_downsampled,
                               color=['Clustering',_],
                               show=False,
                               layer="scaled01",
                               legend_fontoutline=1, frameon=False,
                               na_in_legend=False, s=50, cmap='turbo',
                               save=".".join(["".join([str(self.tool), _+"_ALL"]), self.fileformat])
                               )
                else:
                    continue
        else:
            pass

    def matrixplot(self):
        """
        Function for the generation of matrixplot sc.pl.matrixplot
        return: matrixplot
        """
        self.matrixplot_folder = "/".join([self.outfig, "HEATMAP"])
        self.createdir(self.matrixplot_folder)
        sc.settings.figdir = self.matrixplot_folder
        if self.runtime != 'UMAP':
            sc.pl.matrixplot(self.adata, list(self.adata.var_names), "pheno_leiden",
                             dendrogram = True, vmin = -2, vmax = 2, cmap = 'RdBu_r',
                             show = False, swap_axes = False, return_fig = False,use_raw = False, log=False,
                             save = ".".join(["matrixplot_mean_z_score", self.fileformat]))
            sc.pl.matrixplot(self.adata, list(self.adata.var_names), "pheno_leiden",
                             dendrogram = True, vmin = -2, vmax = 2, cmap = 'RdBu_r', use_raw = False, log=False,
                             show = False, swap_axes = False, return_fig = False,
                             save = ".".join(["matrixplot_mean_z_score", 'svg']))
            sc.pl.matrixplot(self.adata, list(self.adata.var_names), "pheno_leiden",
                             dendrogram = True, cmap = 'Blues', standard_scale = 'var',
                             colorbar_title = 'column scaled\nexpression',
                             swap_axes = False, return_fig = False,
                             show = False,
                             save = ".".join(["matrixplot_column_scaled_expression", self.fileformat]))
        else:
            pass

    def createfcs(self):
        """
        Function for the generation of fcs files and FlowAI QC
        return: Anndata with filtered cells
        """
        try:
            self.outfig = "/".join([self.output_folder, "".join(["Figures", self.tool])])
            self.createdir(self.outfig)
            self.log.info("Perform Flow Auto QC with FlowAI tool.")
            df = pd.DataFrame(self.adata.X, columns = self.adata.var.index, index = self.adata.obs.index)
            self.concatenate_fcs = "/".join([self.output_folder,
                                             "Test_ConcatenatedCells.fcs"])
            if 'Time' in df.columns:
                fcsy.write_fcs(path = self.concatenate_fcs, df = df)
                subprocess.check_call(['Rscript', '--vanilla',
                                       self.path_flowai, self.concatenate_fcs,
                                       self.output_folder], stdout = self.fnull, stderr = self.fnull)
                df = fcsy.read_fcs("/".join([self.output_folder,
                                            "Test_ConcatenatedCells_concatenate_after_QC.fcs"]))
                df.set_index(self.adata.obs.index, inplace = True)
                self.adata.obs['remove_from_FM'] = df['remove_from_FM']
                self.adata = self.adata[(self.adata.obs['remove_from_FM'] < 10000), :]
                self.adata.layers['raw_value'] = self.adata.X
                self.log.info("{0} cells after FlowAI analysis".format(self.adata.shape[0]))
                self.cleaning.update({ "After QC": self.adata.shape[0]})
                self.create_barplot()
                return self.adata
            else:
                # self.log.info("Time channel not found. Skip QC")
                pass
        except:
            # self.log.info("Impossible to complete Flow Auto QC. Check Time channel.")
            pass

    def runclustering(self, method='Phenograph'):
        """
        Executes clustering analysis on the dataset using the specified method ('phenograph', 'flowsom', or 'via').
        :param method: Clustering method to use ('phenograph', 'flowsom', or 'via').
        :return: Updated AnnData object with clustering results.
        """
        self.log.warning("PART 2")
        self.log.info(f"{method.capitalize()} Clustering")

        # Log markers used and excluded for clustering
        self._log_markers(method)

        # Backup the original AnnData object
        self.adataback = self.adata.copy()

        # Subset AnnData and prepare data
        self._prepare_data()

        #self.plot_expression_and_save_pdf("EXP")
        # Perform batch correction if specified
        if self.scanorama:
            self._batch_correction()

        # Standardize data
        self.adata.X = StandardScaler().fit_transform(self.adata.X)

        # Execute clustering based on the specified method
        if method.lower() == 'phenograph':
            self._run_phenograph()
        elif method.lower() == 'flowsom':
            self._run_flowsom()
        elif method.lower() == 'via':
            self._run_via()
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # Update AnnData with clustering results
        self._update_results()

        # Run additional analyses and visualizations if runtime is 'Full'
        if self.runtime == 'Full':
            self._run_full_analysis()

        return self.adata

    def _log_markers(self, method):
        """Logs markers used and excluded for clustering."""
        self.log.info(f"Markers used for {method.capitalize()} clustering:")
        for marker in self.markertoinclude:
            self.log.info(f" + {marker}")

        if self.marker_array:
            self.log.info(f"Markers excluded from {method.capitalize()} clustering:")
            for marker in self.marker_array:
                self.log.info(f" - {marker}")

    def _prepare_data(self):
        """Subsets AnnData, adds an index column, and saves data for batch correction."""
        self.adata.raw = self.adata
        self.adata = self.adata[:, self.markertoinclude]
        self.adata.var['n'] = range(len(self.adata.var))

        data_df = self.adata[:, self.markertoinclude].to_df()
        data_df['batch'] = self.adata.obs[self.batchcov]
        data_df.to_csv(f"{self.output_folder}/tmp.csv", index = False)

    def _batch_correction(self):
        """Performs batch correction using an external R script."""
        subprocess.check_call(
            ['Rscript', '--vanilla', self.path_cycombine, f"{self.output_folder}/tmp.csv", self.output_folder,
             str(self.maxclus), str(self.adata.shape[1])],
            stdout = self.fnull, stderr = self.fnull
        )
        corrected_csv_path = f"{self.output_folder}/corrected_data.csv"
        corrected_data = pd.read_csv(corrected_csv_path).drop(columns = ["id", "label", "batch"])
        self.adatacorrected = ad.AnnData(corrected_data)
        self.adata.X = self.adatacorrected.X

    def _run_phenograph(self):
        """Executes Phenograph clustering."""
        if self.runtime == 'Clustering':
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
        elif self.runtime == 'Full':
            self.adata.obsm['X_umap'] = umap.UMAP().fit_transform(self.adata.X)
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
                self.adata, cols_to_use = list(range(len(self.adata.var))),
                xdim = 10, ydim = 10, n_clusters = self.maxclus, seed = 42
            )
            self.communities = fsom.metacluster_labels
        elif self.runtime == 'Full':
            self.adata.obsm['X_umap'] = umap.UMAP().fit_transform(self.adata.X)
            fsom = fs.FlowSOM(
                self.adata, cols_to_use = list(range(len(self.adata.var))),
                xdim = 10, ydim = 10, n_clusters = self.maxclus, seed = 42
            )
            self.communities = fsom.metacluster_labels
            self.adata.obs['MetaCluster_Flowsom'] = pd.Categorical(fsom.cluster_labels)

    def _run_via(self):
        """Executes VIA clustering."""
        if self.runtime == 'Clustering':
            p = via.VIA(
                self.adata.X, random_seed = 42, knn = int(self.knn), root_user = self.root_user,
                jac_weighted_edges = False, distance = 'l2',
                small_pop = 10, resolution_parameter = self.resolution, num_threads = int(self.thread)
            )
            p.run_VIA()
            self.communities = p.labels
        elif self.runtime == 'Full':
            self.adata.obsm['X_umap'] = umap.UMAP().fit_transform(self.adata.X)
            p = via.VIA(
                self.adata.X, random_seed = 42, knn = int(self.knn), root_user = self.root_user,
                jac_weighted_edges = False, distance = 'l2',
                small_pop = 10, resolution_parameter = self.resolution, num_threads = int(self.thread)
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
        self.plot_umap_expression()
        self.plot_frequency()
        self.plot_cell_clusters()
        self.plot_cell_obs()
        self.matrixplot()

    def generation_concatenate(self):
        """
        Function to concatenate the results of the clustering and the original adata object
        return: adata object with the results of the clustering
        """
        if self.runtime == 'Full':
            self.tmp_df = pd.merge(pd.DataFrame(self.adata.X,
                                                columns = self.adata.var_names,
                                                index = self.adata.obs.index).astype(int),
                                   pd.DataFrame(self.adata.obsm['X_umap'], columns = ['UMAP_1', 'UMAP_2'],
                                                index = self.adata.obs.index),
                                   right_index = True,
                                   left_index = True)
            pd.merge(self.tmp_df, self.adata.obs[['cluster',
                                                  'Sample', 'Cell_type',
                                                  'EXP',
                                                  'ID', 'Time_point',
                                                  'Condition']], left_index = True,
                     right_index = True).to_csv("/".join([self.output_folder, ".".join([self.analysis_name, "csv"])]),
                                                header = True, index = False)
        elif self.runtime == 'UMAP':
            self.tmp_df = pd.merge(pd.DataFrame(self.adata.X,
                                                columns = self.adata.var_names,
                                                index = self.adata.obs.index).astype(int),
                                   pd.DataFrame(self.embedding, columns = ['UMAP_1', 'UMAP_2'],
                                                index = self.adata.obs.index),
                                   right_index = True,
                                   left_index = True)
            pd.merge(self.tmp_df, self.adata.obs[['Sample', 'Cell_type',
                                                  'EXP',
                                                  'ID', 'Time_point',
                                                  'Condition']], left_index = True,
                     right_index = True).to_csv("/".join([self.output_folder, ".".join([self.analysis_name, "csv"])]),
                                                header = True, index = False)
        elif self.runtime == 'Clustering':
            self.tmp_df = pd.merge(pd.DataFrame(self.adata.X,
                                                columns = self.adata.var_names,
                                                index = self.adata.obs.index).astype(int),
                                   self.adata.obs[['cluster', 'Sample', 'Cell_type',
                                                   'EXP',
                                                   'ID', 'Time_point',
                                                   'Condition']],
                                   right_index = True,
                                   left_index = True)
            self.tmp_df.to_csv("/".join([self.output_folder, ".".join([self.analysis_name, "csv"])]), header = True,
                               index = False)

    def plot_frequency(self):
        """
        Plot barplot with frequency
        return: barplot with frequency
        """
        if self.runtime != 'UMAP':
            self.FREQUENCY_folder = "/".join([self.outfig, "BARPLOT_FREQUENCY"])
            self.createdir(self.FREQUENCY_folder)
            fig, (ax1) = plt.subplots(1, 1, figsize = (17 / 2.54, 17 / 2.54))
            ax1 = self.adata.obs.groupby("pheno_leiden")["Sample"].value_counts(
                normalize = True).unstack().plot.barh(
                stacked = True,
                legend = False,
                ax = ax1,
                color = self.palette)
            ax1.set_xlabel("Percentage Frequency")
            ax1.set_ylabel("Cluster")
            ax1.grid(False)
            ax1.legend(bbox_to_anchor = (1.2, 1.0))
            ### save
            fig.savefig("/".join([self.FREQUENCY_folder , ".".join(["ClusterFrequencyNormalized", self.fileformat])]),
                        dpi = self.dpi, bbox_inches = 'tight',
                        format = self.fileformat)
            fig.savefig("/".join([self.FREQUENCY_folder , ".".join(["ClusterFrequencyNormalized", 'svg'])]),
                        dpi = self.dpi, bbox_inches = 'tight',
                        format = 'svg')
            #
            for _ in ['Sample', 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']:
                if len(self.adata.obs[_].unique()) > 1:
                    fig, (ax3) = plt.subplots(1, 1, figsize = (17 / 2.54, 17 / 2.54))
                    ax3 = self.adata.T.var.groupby(_)["pheno_leiden"].value_counts(
                        normalize = True).unstack().plot.barh(
                        stacked = True,
                        legend = False,
                        color = self.palette,
                        ax = ax3,
                        fontsize = 5)
                    ax3.set_xlabel("Cluster Percentage Frequency")
                    ax3.set_ylabel(_)
                    ax3.grid(False)
                    ax3.legend(bbox_to_anchor = (1.2, 1.0))
                    fig.savefig("/".join(
                        [self.FREQUENCY_folder , ".".join(["".join([_, "ClusterFrequencyNormalized"]), self.fileformat])]),
                                dpi = self.dpi, bbox_inches = 'tight',
                                format = self.fileformat)
            #
            fig, (ax2) = plt.subplots(1, 1, figsize = (17 / 2.54, 17 / 2.54))
            ax2 = self.adata.obs.groupby("pheno_leiden")["Sample"].value_counts(
                normalize = False).unstack().plot.barh(
                stacked = True,
                legend = False,ax = ax2,color = self.palette)
            ax2.set_xlabel("Relative Frequency")
            ax2.set_ylabel("Cluster")
            ax2.grid(False)
            ax2.legend(bbox_to_anchor = (1.2, 1.0))
            fig.savefig("/".join([self.FREQUENCY_folder , ".".join(["ClusterFrequencyNotNormalized", self.fileformat])]),
                        dpi = self.dpi, bbox_inches = 'tight',
                        format = self.fileformat)
            fig.savefig("/".join([self.FREQUENCY_folder , ".".join(["ClusterFrequencyNotNormalized", 'svg'])]),
                        dpi = self.dpi, bbox_inches = 'tight',
                        format = 'svg')
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
        group_combinations = [
            ["Time_point", "Condition"],
            ["Sample", "Cell_type"],
            ["EXP", "Time_point"]
        ]

        for group_cols in group_combinations:
            # Check if there is more than one unique combination in group_cols
            unique_combinations = self.adata.obs[group_cols].drop_duplicates()
            if len(unique_combinations) <= 1:
                print(f"Skipping plot for {', '.join(group_cols)} as there is only one unique combination.")
                continue

            # Compute frequencies for each combination of group_cols and pheno_leiden
            df = self.adata.obs.groupby(group_cols + ["pheno_leiden"]).size().unstack(fill_value = 0)

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
                df.loc[dn['ivl']].plot.barh(stacked = True, ax = ax2, color = self.palette, legend = False)

                # Set axis labels and appearance
                ax1.set(yticklabels = [], xticklabels = [])
                ax1.grid(False)
                ax2.tick_params(left = False)
                ax2.grid(False)
                ax1.axis('off')
                ax2.set_ylabel(" ")
                ax2.set_xlabel("Percentage Frequency")
                ax2.legend(bbox_to_anchor = (1.2, 1.0), title = 'Cluster')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.spines['left'].set_visible(False)
            else:
                ax1.text(0.5, 0.5, f'No data available for {", ".join(group_cols)}', horizontalalignment = 'center',
                         verticalalignment = 'center', fontsize = 12, transform = ax1.transAxes)
                ax2.text(0.5, 0.5, f'No data available for {", ".join(group_cols)}', horizontalalignment = 'center',
                         verticalalignment = 'center', fontsize = 12, transform = ax2.transAxes)

            # Save the figure
            fig.savefig(f"{self.FREQUENCY_folder}/SampleFrequency_{'_'.join(group_cols)}_Clusterized.{self.fileformat}",
                        dpi = self.dpi, bbox_inches = 'tight', format = self.fileformat)

        # Optionally, display all figures if running interactively
        for fig in fig_list:
            plt.show()

    def createdir(self, dirpath):
        """
        Make dir function and check if directory is already exists
        :param dirpath: string with path and directory name
        :return:
        """
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

    def groupbycluster(self):
        """
        Function for generation of csv with different clusters
        :return:
        """
        # make dir
        self.createdir("/".join([self.output_folder, "".join(["Cluster", self.tool])]))
        self.adata = self.adataback
        self.adata.obs['cluster'] = self.adata.obs['cluster'].astype(int)
        for _ in range(self.adata.obs['cluster'].unique().min(), self.adata.obs['cluster'].unique().max() + 1):
            self.tmp = self.adata[self.adata.obs['cluster'].isin([_])].to_df()
            self.tmp = self.tmp.astype(int)
            if self.runtime == 'Full':
                self.tmp['UMAP_1'] = pd.DataFrame(
                    self.adata[self.adata.obs['cluster'].isin([_])].obsm['X_umap'][:, 0]).set_index(self.tmp.index)
                self.tmp['UMAP_2'] = pd.DataFrame(
                    self.adata[self.adata.obs['cluster'].isin([_])].obsm['X_umap'][:, 1]).set_index(self.tmp.index)
            if (self.tool == "Phenograph"):
                self.tmp['Phenograph'] = _
            elif (self.tool == "VIA"):
                self.tmp['VIA'] = _
            else:
                self.tmp['FlowSOM'] = _
                #self.tmp['MetaCluster_FlowSOM'] = self.adata[self.adata.obs['metaclustering'].isin([int(_)]), :].obs[
                #    'clustering'].values
            self.tmp.to_csv("/".join([self.output_folder, "".join(["Cluster", self.tool]),
                                      "".join([self.analysis_name, "_", str(_), ".csv"])]), header=True, index=False)
            fcsy.write_fcs(self.tmp, "/".join([self.output_folder, "".join(["Cluster", self.tool]),
                                               "".join([self.analysis_name, "_", str(_), ".fcs"])]))

    def groupbysample(self):
        """
        Function for generation of csv with different clusters
        """
        # make dir
        self.createdir("/".join([self.output_folder, "".join(["Sample", self.tool])]))
        # tmp df
        self.tmp = self.adata.to_df()
        self.tmp = self.tmp.astype(int)
        # change index
        self.tmp.set_index(self.adata.obs['Sample'], inplace = True)
        if self.runtime != 'UMAP':
            self.adata.obs['cluster'] = self.adata.obs['cluster'].astype(int)
            self.createdir("/".join([self.output_folder, "".join(["ClusterFrequency", self.tool])]))
        # create columns
        if self.runtime != 'Clustering':
            self.tmp['UMAP_1'] = self.adata.obsm['X_umap'][:, 0]
            self.tmp['UMAP_2'] = self.adata.obsm['X_umap'][:, 1]
        if self.runtime != 'UMAP':
            self.tmp[self.tool] = self.adata.obs['cluster'].values

            # if self.tool == 'FlowSOM':
            #     self.tmp['MetaCluster_FlowSOM'] = self.adata.obs['MetaCluster_FlowSOM'].values
            # else:
            #     pass
            self.tmp["cluster"] = self.adata.obs['cluster'].values
            # get unique filenames
            unique_filename = self.adata.obs['Sample'].unique()
            # get unique number of cluster
            unique_Phenograph = self.adata.obs['cluster'].unique()
            #
            dfCounts = pd.DataFrame(index = range(min(unique_Phenograph), max(unique_Phenograph) + 1))
            # generate Tot_percentage file
            for i in range(len(unique_filename)):
                dfCounts[unique_filename[i]] = self.tmp.loc[self.tmp.index == unique_filename[i]].cluster.value_counts(
                    normalize = True).reindex(self.tmp.cluster.unique(), fill_value = 0)
            # compute percentage
            dfCounts = dfCounts * 100
            # save
            dfCounts.index.name = 'Cluster'
            dfCounts.to_csv("/".join(["/".join([self.output_folder, "".join(["ClusterFrequency", self.tool])]),
                                      "".join(["TotalPercentage", ".csv"])]))
            # create empty dataframe
            dfCounts = pd.DataFrame(index = range(min(unique_Phenograph), max(unique_Phenograph) + 1))
            # generate Tot_counts file
            for i in range(len(unique_filename)):
                dfCounts[unique_filename[i]] = self.tmp.loc[
                    self.tmp.index == unique_filename[i]].cluster.value_counts().reindex(
                    self.tmp.cluster.unique(), fill_value = 0)
            # save
            dfCounts.index.name = 'Cluster'
            dfCounts.to_csv("/".join(["/".join([self.output_folder, "".join(["ClusterFrequency", self.tool])]),
                                      "".join(["TotalCounts", ".csv"])]))
            del self.tmp['cluster']
            # save samples
            for i in range(len(unique_filename)):
                dfCounts[unique_filename[i]] = self.tmp.loc[self.tmp.index == unique_filename[i]].to_csv(
                    "/".join([self.output_folder, "".join(["Sample", self.tool]),
                              "".join([str(unique_filename[i]), "_", self.analysis_name, ".csv"])]),
                    header=True, index=False)
                fcsy.write_fcs(self.tmp.loc[self.tmp.index == unique_filename[i]],
                               "/".join([self.output_folder, "".join(["Sample", self.tool]),
                                         "".join([str(unique_filename[i]), "_", self.analysis_name, ".fcs"])]))
        else:
            unique_filename = self.adata.obs['Sample'].unique()
            for i in range(len(unique_filename)):
                self.tmp.loc[self.tmp.index == unique_filename[i]].to_csv(
                    "/".join([self.output_folder, "".join(["Sample", self.tool]),
                              "".join([str(unique_filename[i]), "_", self.analysis_name, ".csv"])]),
                    header=True, index=False)
                fcsy.write_fcs(self.tmp.loc[self.tmp.index == unique_filename[i]],
                               "/".join([self.output_folder, "".join(["Sample", self.tool]),
                                         "".join([str(unique_filename[i]), "_", self.analysis_name, ".fcs"])]))

    def runtimeumap(self):
        """
        Function for execution of phenograph analysis
        :return:
        """
        self.log.warning("PART 2")
        self.log.info("UMAP Dimensional Reduction")
        self.outfig = "/".join([self.output_folder, "".join(["Figures", self.tool])])
        self.createdir(self.outfig)
        self.log.info("Markers used for UMAP computation:")
        for i in self.markertoinclude:
            self.log.info(" + " + i)
        #self.adata = self.adata[:, self.markertoinclude].copy()
        if len(self.marker_array):
            self.log.info("Markers excluded for UMAP computation:")
        for i in self.marker_array:
            self.log.info(" - " + i)
        sc.tl.pca(self.adata, svd_solver='arpack',n_comps=len(self.adata.var.index)-1)
        sc.pp.scale(self.adata, max_value = 10,copy = False)
        # self.adata.X = self.adata.layers['scaled']
        if self.scanorama is True:
            sce.pp.harmony_integrate(self.adata, self.batchcov)
            # print(self.adata)
            min_cml_frac = 0.5
            cml_var_explained = np.cumsum(self.adata.uns['pca']['variance_ratio'])
            n_pcs = next(idx for idx, cml_frac in enumerate(cml_var_explained) if cml_frac > min_cml_frac)
            sc.pp.neighbors(self.adata, n_neighbors = 10, n_pcs = n_pcs,use_rep = 'X_pca_harmony')
            sc.tl.umap(self.adata)
            #
            # self.adata = self.correct_scanorama()
            # self.embedding = self.runumap()
            # self.adata.obsm['X_umap'] = self.embedding
            # self.adata.obsm['X_umap'] = self.embedding
        else:
            cml_var_explained = np.cumsum(self.adata.uns['pca']['variance_ratio'])
            x = range(len(self.adata.uns['pca']['variance_ratio']))
            y = cml_var_explained
            #plt.scatter(x, y, s = 4)
            #plt.xlabel('PC')
            #plt.ylabel('Cumulative variance explained')
            #plt.title('Cumulative variance explained by PCs')
            min_cml_frac = 0.5
            n_pcs = next(idx for idx, cml_frac in enumerate(cml_var_explained) if cml_frac > min_cml_frac)
            sc.pp.neighbors(self.adata, n_neighbors = 10, n_pcs = n_pcs)
            sc.tl.umap(self.adata)
            # self.embedding = self.runumap()
            # self.adata.obsm['X_umap'] = self.embedding
            # self.adata.obsm['X_umap'] = self.embedding
        self.adata = self.adata.raw.to_adata()
        self.generation_concatenate()
        self.plot_umap()
        return self.adata

    def exporting(self):
        """
        Export to h5ad file.
        """
        self.log.warning("PART 4")
        self.log.info("Output Generation")
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        try:
            del self.adata.obs['remove_from_FM']
        except:
            pass
        if self.runtime != 'UMAP':
            if self.tool == "Phenograph":
                self.adata.obs[self.tool + "_" + str(self.k_coef)] = self.adata.obs['cluster'].astype("str")
                del self.adata.obs['cluster']
                del self.adata.obs[self.tool + "_" + str(self.k_coef)]
                self.adata.obs.rename(columns = {"pheno_leiden": "Phenograph_cluster"}, inplace = True)
                #self.adata.obs["".join([str(self.tool), "_cluster"])] = self.adata.obs["pheno_leiden"].astype('category')
                self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.layers['raw_value'])
                self.adata.X = self.adata.layers['scaled01']
                self.adata.layers['scaled01'] = scipy.sparse.csr_matrix(self.adata.layers['scaled01'])
                #TODO order columns as initial obj
                self.adata.write("/".join([self.output_folder, ".".join([self.analysis_name, "h5ad"])]))
            elif self.tool == "VIA":
                self.adata.obs[self.tool + "_" + str(self.knn)] = self.adata.obs['cluster'].astype("str")
                del self.adata.obs['cluster']
                del self.adata.obs[self.tool + "_" + str(self.knn)]
                self.adata.obs["".join([str(self.tool), "_cluster"])] = self.adata.obs[
                    "".join([str(self.tool), "_cluster"])].astype('category')
                self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.layers['raw_value'])
                self.adata.X = self.adata.layers['scaled01']
                self.adata.layers['scaled01'] = scipy.sparse.csr_matrix(self.adata.layers['scaled01'])
                self.adata.write("/".join([self.output_folder, ".".join([self.analysis_name, "h5ad"])]))
            else:
                del self.adata.obs['cluster']
                self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.layers['raw_value'])
                self.adata.X = self.adata.layers['scaled01']
                self.adata.layers['scaled01'] = scipy.sparse.csr_matrix(self.adata.layers['scaled01'])
                self.adata.write("/".join([self.output_folder, ".".join([self.analysis_name, "h5ad"])]))
                try:
                    os.remove("/".join([self.output_folder, "tmp.csv"]))
                    os.remove("/".join([self.output_folder, "corrected_data.csv"]))
                except:
                    pass
        else:
            self.adata.layers['scaled01'] = self.scaler.fit_transform(self.adata.X)
            self.adata.X = self.adata.layers['scaled01']
            self.adata.layers['scaled01'] = scipy.sparse.csr_matrix(self.adata.layers['scaled01'])
            self.adata.write("/".join([self.output_folder, ".".join([self.analysis_name, "h5ad"])]))
        try:
            os.remove("/".join([self.output_folder, "".join([self.analysis_name, "_ConcatenatedCells_concatenate_after_QC.fcs"])]))
            os.remove("/".join([self.output_folder, "".join([self.analysis_name, "_ConcatenatedCells.fcs"])]))
        except:
            pass
        self.log.warning("PART 5")

