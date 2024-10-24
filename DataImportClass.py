import os
import sys
import glob
import pandas as pd
import anndata
import scanpy as sc
import fcsy
import warnings
from fcsy import DataFrame
from LogClass import LoggerSetup
warnings.filterwarnings("ignore")

class DataImporter:
    """
    A class responsible for importing and preparing data for analysis.
    """

    def __init__(self, input_folder, info_file, marker_list, filetype, cellnumber, downsampling):
        """
        Initialize the DataImporter class with the given parameters.

        :param input_folder: Path to the folder containing input files.
        :param info_file: Path to the info file.
        :param marker_list: Path to the marker list file.
        :param filetype: Type of input files (CSV or FCS).
        :param cellnumber: Number of cells for downsampling.
        :param downsampling: Downsampling method (All, Balanced, Fixed).
        """
        self.input_folder = input_folder
        self.info_file = info_file
        self.marker_list = marker_list
        self.filetype = filetype
        self.cellnumber = cellnumber
        self.downsampling = downsampling
        self.anndata_list = []  # List to hold AnnData objects for each file
        self.cleaning = {}  # Dictionary to hold cleaning statistics
        # Set up logging
        self.log = LoggerSetup.setup_logging()
        sc.settings.verbosity = 0


    def read_info_file(self):
        """
        Read the info file and return it as a pandas DataFrame.

        :return: DataFrame containing the info file data.
        """
        df_info = pd.read_excel(self.info_file, header=0)
        return df_info

    def import_all_event(self):
        """
        Scan the input folder and return a list of all CSV or FCS file paths.

        :return: List of file paths.
        """
        # Change the current directory to the input folder
        os.chdir(self.input_folder)

        # Find all files with the specified filetype in the input folder
        if self.filetype == "CSV":
            all_files = glob.glob(os.path.join(self.input_folder, "*.csv"))
        else:
            all_files = glob.glob(os.path.join(self.input_folder, "*.fcs"))

        return all_files

    def concatenate_dataframe(self, info_file, csv_list):
        """
        Concatenate dataframes from the list of CSV/FCS files and return an AnnData object.

        :param info_file: DataFrame containing the info file data.
        :param csv_list: List of CSV/FCS file paths.
        :return: Concatenated AnnData object and markers order.
        """
        self.log.info("Files concatenation")
        pandas_df_list = []  # List to hold individual DataFrames

        # Iterate through the list of CSV/FCS files
        for file_path in csv_list:
            if self.filetype == "CSV":
                pandas_df_list.append(self.create_df(file_path))
            else:
                pandas_df_list.append(self.read_fcs(file_path))

        # Ensure all DataFrames have the same set of columns
        if all([len(pandas_df_list[0].columns.intersection(df.columns)) == pandas_df_list[0].shape[1]
                for df in pandas_df_list]):
            try:
                # Iterate through each DataFrame and create AnnData objects
                for df in pandas_df_list:
                    # Extract sample name from index
                    sample_name = df.index[0].rsplit('_', 1)[0]

                    # Check if the sample name is in the info file
                    if sample_name in info_file["Sample"].tolist():
                        # Create AnnData object for each DataFrame
                        ann_tmp = anndata.AnnData(df)
                        ann_tmp.obs['Sample'] = sample_name
                        ann_tmp.obs['Cell_type'] = self._get_obs_value(info_file, 'Cell_type', sample_name)
                        ann_tmp.obs['EXP'] = self._get_obs_value(info_file, 'EXP', sample_name)
                        ann_tmp.obs['ID'] = self._get_obs_value(info_file, 'ID', sample_name)
                        ann_tmp.obs['Time_point'] = self._get_obs_value(info_file, 'Time_point', sample_name)
                        ann_tmp.obs['Condition'] = self._get_obs_value(info_file, 'Condition', sample_name)
                        ann_tmp.obs['Count'] = self._get_obs_value(info_file, 'Count', sample_name)

                        # Append AnnData object to the list
                        self.anndata_list.append(ann_tmp)
                    else:
                        self.log.error(f"Error, the file {sample_name} is not in the Sample column of the Info File.")
                        sys.exit(1)

                # Concatenate all AnnData objects into one
                if len(self.anndata_list) == 1:
                    self.adata = self.anndata_list[0]
                else:
                    self.adata = self.anndata_list[0].concatenate(self.anndata_list[1:], index_unique=None)

                # Set raw values and var names

                self.adata.layers['raw_value'] =  self.adata.X
                self.adata.var_names = [name.split(":: ")[-1] for name in self.adata.var_names]

            except (ValueError, Exception) as e:
                self.log.error(f"Error. Please check the Info File Header or Data Files Header. Details: {e}")
                sys.exit(1)
        else:
            self.log.error(
                "Error. Columns mismatch between data files. Please check the Info File Header or Data Files Header.")
            sys.exit(1)

        # Perform downsampling if the method is 'Fixed' and cellnumber is less than total cells
        if self.downsampling == "Fixed" and self.cellnumber < self.adata.shape[0]:
            sc.pp.subsample(self.adata, n_obs=self.cellnumber, random_state=42)

        # Update cleaning statistics
        self.cleaning.update({"Before QC": self.adata.shape[0]})

        # Filter out rows with zero values across all columns
        self.adata = self.adata[(self.adata.X > 0).all(axis=1)]
        self.log.info(f"{self.adata.shape[0]} cells undergo clustering analysis")

        # Update layers and raw data
        # self.adata.layers['raw_value'] = self.adata.X
        self.adata.raw = self.adata
        return self.adata

    def _get_obs_value(self, info_file, column, sample_name):
        """
        Retrieve the observation value from the info file for a given column and sample name.

        :param info_file: DataFrame containing the info file data.
        :param column: Column name to retrieve the value from.
        :param sample_name: Sample name to use for filtering.
        :return: Retrieved value.
        """
        # Extract the value from the specified column for the given sample name
        value = info_file[column].loc[info_file['Sample'] == sample_name]
        return ''.join(e for e in value.to_string().split()[-1] if e.isalnum())

    def loadmarkers(self):
        """
        Read the marker list file and return the file path and name.

        :return: Tuple containing the file path and file name.
        """
        if os.path.exists(self.marker_list):
            return os.path.split(self.marker_list)
        else:
            self.log.error("Marker list file does not exist.")
            sys.exit(1)

    def checkmarkers(self):
        """
        Check if the markers in the file are present in the concatenated data.

        :return: List of markers.
        """
        # Read marker list from file
        self.marker_array = [line.rstrip() for line in open(self.marker_list)]
        self.marker_array = [marker.split(":: ")[-1] for marker in self.marker_array]

        # Verify each marker's presence in the AnnData var_names
        for marker in self.marker_array:
            if marker not in self.adata.var_names:
                self.log.error(f"Marker {marker} not found in Matrix.")
                sys.exit(1)

        return self.marker_array

    def splitmarker(self):
        """
        Split markers into included and excluded lists.

        :return: List of markers to include in the analysis.
        """
        # Create a list of markers to include, excluding those in the marker array
        self.marker = self.adata.var_names.to_list()
        self.markertoinclude = [i for i in self.marker if i not in self.marker_array]
        return self.markertoinclude

    def create_df(self, path_csv_file):
        """
        Create a dataframe from a CSV file.

        :param path_csv_file: Path to the CSV file.
        :return: DataFrame created from the CSV file.
        """
        df = pd.read_csv(path_csv_file, header=0)

        # Perform balanced downsampling if necessary
        if self.downsampling == "Balanced" and self.cellnumber < df.shape[0]:
            df = df.sample(n=int(self.cellnumber), random_state=42, ignore_index=True)

        # Set the DataFrame index to the generated barcode
        df.index = self._generate_barcode(df, path_csv_file)
        return df

    def read_fcs(self, path_fcs_file):
        """
        Read an FCS file and convert it into a pandas DataFrame.

        :param path_fcs_file: Path to the FCS file.
        :return: DataFrame created from the FCS file.
        """
        df = DataFrame.from_fcs(path_fcs_file, channel_type='multi')

        # Clean and adjust DataFrame column names
        df.columns = df.columns.map(' :: '.join)
        df.columns = df.columns.str.replace('[\",\']', '')
        df.columns = df.columns.str.rstrip(': ')

        # Perform balanced downsampling if necessary
        if self.downsampling == "Balanced" and self.cellnumber < df.shape[0]:
            df = df.sample(n=int(self.cellnumber), random_state=42, ignore_index=True)

        # Set the DataFrame index to the generated barcode
        df.index = self._generate_barcode(df, path_fcs_file)
        return df

    def _generate_barcode(self, df, file_path):
        """
        Generate a unique barcode for each row in the DataFrame based on the file name.

        :param df: DataFrame for which barcodes are to be generated.
        :param file_path: File path to extract the base name for the barcode.
        :return: List of generated barcodes.
        """
        # Extract base name from file path
        base_name = os.path.basename(file_path).split(".")[0]

        # Generate a unique barcode for each row in the DataFrame
        return [f"{base_name}_{i}" for i in range(1, df.shape[0] + 1)]

    def generate_combination_obs(self):
        """
        Generate new observation columns in obs for all possible combinations of
        'Cell_type', 'EXP', 'ID', 'Time_point', and 'Condition', using 2 or 3 features,
        only if all selected columns have more than one unique value.

        Returns a list containing the original columns and new combination column names.
        """
        from itertools import combinations

        # List of column names to combine
        columns = ['Cell_type', 'EXP', 'ID', 'Time_point', 'Condition']

        # Check if all necessary columns exist in obs
        for col in columns:
            if col not in self.adata.obs.columns:
                self.log.error(f"Column {col} is missing in the AnnData object's obs.")
                return

        # Initialize list with the original columns
        col_list = columns.copy()

        # Generate combinations with 2 and 3 features
        for r in [2, 3]:  # r=2 for two-feature combinations, r=3 for three-feature combinations
            for combo in combinations(columns, r):
                # Check if all columns in the combination have more than one unique value
                if all(self.adata.obs[col].nunique() > 1 for col in combo):
                    # Create a new obs column for each valid combination
                    new_col_name = "_".join(combo)
                    self.adata.obs[new_col_name] = self.adata.obs[combo[0]].astype(str)

                    # Concatenate values for the other columns in the combination
                    for col in combo[1:]:
                        self.adata.obs[new_col_name] += "_" + self.adata.obs[col].astype(str)

                    # Add the new column name to the list
                    col_list.append(new_col_name)

        self.log.info("New combination columns created in obs for all valid 2- and 3-feature combinations.")

        return col_list

