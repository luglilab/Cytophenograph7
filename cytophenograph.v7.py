import sys
import optparse
import os
from DataImportClass import DataImporter
from ClusteringClass import Clustering
from VisualizationClass import Visualization
from PrintLogoClass import PrintLogo
from GroupingClass import Grouping
from ExportingClass import Exporting
from LogClass import LoggerSetup  # Correct import from ClassLog
import warnings
from StreamClass import StreamTrajectory
import traceback

def parse_arguments():
    """
    Parse command line arguments.

    :return: Parsed arguments.
    """
    parser = optparse.OptionParser(
        usage = 'python ./Cytophenograph/cytophenograph.v7.py -i $abs_path/Cytophenograph/Test_dataset/CD8_Panel_II_channelvalues_GA_downSampled/ -o $abs_path/Cytophenograph/output_test -k 300 -m $abs_path/Cytophenograph/Test_dataset/CD8_bulk_markers_to_exclude.txt -n Test -t 10 -p $abs_path/Cytophenograph/Test_dataset/Info_file_bulk_Test.xlsx -c VIA',
        version = '7.0'
    )
    parser.add_option('-a', action = "store_true", dest = "arcsin", default = False,
                      help = 'Perform arcsinh transformation on data.')
    parser.add_option('-b', action = "store_true", dest = "batch", default = False,
                      help = 'Perform batch correction with cyCombine.')
    parser.add_option('-c', type = 'choice', choices = ['Phenograph', 'VIA', 'FlowSOM'], dest = "clustering",
                      default = "Phenograph",
                      help = 'Tool selecting for clustering. Options: [Phenograph, VIA, FlowSOM].')
    parser.add_option('-d', action = "store", dest = "mindist", default = 0.5, type = float,
                      help = 'min_dist parameter for UMAP generation.')
    parser.add_option('-e', type = 'choice', choices = ['Sample', 'Cell_type', 'ID', 'EXP', 'Time_point', 'Condition'],
                      dest = "batchcov", default = "Sample", help = 'Covariate to correct with cyCombine.')
    parser.add_option('-f', type = 'choice', choices = ['All', 'Balanced', 'Fixed'], dest = "downsampling",
                      default = "All", help = 'Concatenation Method: [All, Balanced, Fixed].')
    parser.add_option('-g', action = "store", dest = "cellnumber", default = 1000, type = int,
                      help = 'Number of events for downsampling.')
    parser.add_option('-i', action = "store", dest = "input_folder", help = 'Absolute path of folder with CSV files.')
    parser.add_option('-k', action = "store", dest = "kmeancoef", default = '30', type = 'string',
                      help = 'Number of nearest neighbors for Phenograph. Default: 30')
    parser.add_option('-l', action = "store", dest = "fileformat", choices = ['CSV', 'FCS'], default = 'CSV',
                      type = 'choice', help = 'Format of input files. Default: CSV')
    parser.add_option('-m', action = "store", dest = "markerlist",
                      help = 'Text file with features to exclude during clustering.')
    parser.add_option('-n', action = "store", dest = "analysis_name", help = 'Analysis name.')
    parser.add_option('-o', action = "store", dest = "output_folder",
                      help = 'Absolute path of output folder. Use an empty folder.')
    parser.add_option('-p', action = "store", dest = "pheno", default = "Condition",
                      help = 'Excel file with metadata: Sample-Cell_type-EXP-ID-Time_point-Condition-Count.')
    parser.add_option('-r', type = 'choice', choices = ['Full', 'UMAP', 'Clustering'], dest = "runtime",
                      default = "Full", help = 'Runtime option: [Full, UMAP, Clustering].')
    parser.add_option('-s', action = "store", dest = "spread", default = 1.0, type = float,
                      help = 'Spread parameter for UMAP generation.')
    parser.add_option('-t', action = "store", dest = "thread", type = int, default = 1, help = 'Number of jobs.')
    parser.add_option('-w', action = "store", dest = "knn", default = 30, type = int,
                      help = 'Number of K-Nearest Neighbors for VIA. Range: 5-100. Default: 30')
    parser.add_option('-y', action = "store", dest = "maxclus", default = 31, type = int,
                      help = 'Exact number of clusters for meta-clustering. Max: 31.')
    parser.add_option('-z', action = "store", dest = "resolution", default = 1.0, type = float,
                      help = 'Resolution for VIA clustering. Range: 0.2-1.5. Default: 1.0')
    parser.add_option('-x', action = "store_true", dest = "compensation", default = False,
                      help = 'Apply compensation to the data. Default: False')
    parser.add_option('-j', action = "store_true", dest = "transformation", default = False,
                      help = 'Apply transformation to the data. Default: False')
    parser.add_option('--dimensionality_reduction', type = 'choice', choices = ['UMAP', 'PaCMAP'],
                      dest = "dimensionality_reduction", default = "UMAP",
                      help = 'Dimensionality reduction method: [UMAP, PaCMAP]. Default: UMAP')
    parser.add_option('-T', action = "store_true", dest = "trajectory_analysis", default = False,
                      help = "Perform trajectory analysis using StreamTrajectory. Default: False.")
    return parser.parse_args()


def main():
    # Set up the logger using LoggerSetup from ClassLog
    logger = LoggerSetup.setup_logging()

    PrintLogo()  # Display the ASCII art logo

    options, args = parse_arguments()
    DictInfo = dict()

    try:
        # Initialize the DataImporter with the parsed arguments
        data_importer = DataImporter(
            input_folder = options.input_folder,
            info_file = options.pheno,
            marker_list = options.markerlist,
            filetype = options.fileformat,
            cellnumber = options.cellnumber,
            downsampling = options.downsampling
        )

        # Read and process the info file
        DictInfo["Infofile"] = data_importer.read_info_file()

        # Import all events from the specified files
        DictInfo["List_csv_files"] = data_importer.import_all_event()

        # Concatenate dataframes into an AnnData object
        DictInfo["adata_conc"] = data_importer.concatenate_dataframe(DictInfo["Infofile"], DictInfo["List_csv_files"])

        # Load and verify markers
        DictInfo["pathmarkerfile"], DictInfo["basenamemarkerfilepath"] = data_importer.loadmarkers()
        DictInfo["markertoexclude"] = data_importer.checkmarkers()
        DictInfo["markertoinclude"] = data_importer.splitmarker()

        # Initialize Clustering with necessary parameters
        clustering = Clustering(
            adata = DictInfo["adata_conc"],
            output_folder = options.output_folder,
            method = options.clustering,
            k_coef = options.kmeancoef,
            knn = options.knn,
            resolution = options.resolution,
            maxclus = options.maxclus,
            thread = options.thread,
            runtime = options.runtime,
            batchcov = options.batchcov,
            root_user = [1],
            fnull = open(os.devnull, 'w'),
            path_cycombine = os.path.dirname(os.path.realpath(__file__)) + '/cycombine.Rscript',
            markertoinclude = DictInfo["markertoinclude"],
            marker_array = DictInfo["markertoexclude"]
        )

        # Perform clustering and analysis based on runtime option
        if options.runtime != 'UMAP':
            DictInfo["clustered_adata"] = clustering.runclustering()

            # Initialize Grouping for organizing data by cluster or sample
            grouping = Grouping(
                adata = DictInfo["clustered_adata"],
                output_folder = options.output_folder,
                tool = options.clustering,
                analysis_name = options.analysis_name,
                runtime = options.runtime
            )

            # Execute grouping methods
            grouping.groupbycluster()
            grouping.groupbysample()

            # Initialize Visualization to generate plots and visual analysis
            visualization = Visualization(
                adata = DictInfo["clustered_adata"],
                output_folder = options.output_folder,
                tool = options.clustering,
                runtime = options.runtime,
                analysis_name = options.analysis_name
            )
            visualization.generation_concatenate()
            visualization.plot_umap()
            visualization.plot_umap_expression()
            visualization.plot_frequency()
            visualization.plot_frequency_ptz()
            visualization.plot_cell_clusters()
            visualization.plot_cell_obs()
            visualization.matrixplot()

            # Initialize Exporting for data exportation
            exporting = Exporting(
                adata = DictInfo["clustered_adata"],
                output_folder = options.output_folder,
                analysis_name = options.analysis_name,
                runtime = options.runtime,
                tool = options.clustering,
                k_coef = options.kmeancoef,
                knn = options.knn
            )

            # Run exporting method to save results
            exporting.exporting()
            # Perform trajectory analysis if the user has requested it
            if options.trajectory_analysis:
                logger.info("Performing trajectory analysis...")
                trajectory = StreamTrajectory(
                    adata=DictInfo["clustered_adata"],
                    output_folder=options.output_folder,
                    tool = 'Phenograph',
                    runtime = options.runtime,
                    analysis_name = options.analysis_name
                )
                trajectory.elastic_principal_graph()

        elif options.runtime == 'UMAP':
            logger.info("Running UMAP specific operations...")

    except Exception as e:
        print(traceback.format_exc())
        # Capture and log any exceptions that occur during execution
        logger.error(f"Execution Error: {str(e)}")
        sys.exit(1)  # Exit the program with an error code


if __name__ == '__main__':
    main()
