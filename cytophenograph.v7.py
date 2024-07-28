import sys
import traceback
import optparse
from PhenoFunctions_v7 import Cytophenograph

def parse_arguments():
    parser = optparse.OptionParser(
        usage='python ./Cytophenograph/cytophenograph.v7.py -i $abs_path/Cytophenograph/Test_dataset/CD8_Panel_II_channelvalues_GA_downSampled/ -o $abs_path/Cytophenograph/output_test -k 300 -m $abs_path/Cytophenograph/Test_dataset/CD8_bulk_markers_to_exclude.txt -n Test -t 10 -p $abs_path/Cytophenograph/Test_dataset/Info_file_bulk_Test.xlsx -c VIA',
        version='7.0'
    )
    parser.add_option('-a', action="store_true", dest="arcsin", default=False, help='Perform arcsinh transformation on data.')
    parser.add_option('-b', action="store_true", dest="batch", default=False, help='Perform batch correction with Scanorama.')
    parser.add_option('-c', type='choice', choices=['Phenograph', 'VIA', 'FlowSOM'], dest="clustering", default="Phenograph", help='Tool selecting for clustering. Options: [Phenograph, VIA, FlowSOM].')
    parser.add_option('-d', action="store", dest="mindist", default=0.5, type=float, help='min_dist parameter for UMAP generation.')
    parser.add_option('-e', type='choice', choices=['Sample', 'Cell_type', 'ID', 'EXP', 'Time_point', 'Condition'], dest="batchcov", default="Sample", help='Covariate to correct with Scanorama.')
    parser.add_option('-f', type='choice', choices=['All', 'Balanced', 'Fixed'], dest="downsampling", default="All", help='Concatenation Method: [All, Balanced, Fixed].')
    parser.add_option('-g', action="store", dest="cellnumber", default=1000, type=int, help='Number of events for downsampling.')
    parser.add_option('-i', action="store", dest="input_folder", help='Absolute path of folder with CSV files.')
    parser.add_option('-k', action="store", dest="kmeancoef", default='30', type='string', help='Number of nearest neighbors for Phenograph. Default: 30')
    parser.add_option('-l', action="store", dest="fileformat", choices=['CSV', 'FCS'], default='CSV', type='choice', help='Format of input files. Default: CSV')
    parser.add_option('-m', action="store", dest="markerlist", help='Text file with features to exclude during clustering.')
    parser.add_option('-n', action="store", dest="analysis_name", help='Analysis name.')
    parser.add_option('-o', action="store", dest="output_folder", help='Absolute path of output folder. Use an empty folder.')
    parser.add_option('-p', action="store", dest="pheno", default="Condition", help='Excel file with metadata: Sample-Cell_type-EXP-ID-Time_point-Condition-Count.')
    parser.add_option('-r', type='choice', choices=['Full', 'UMAP', 'Clustering'], dest="runtime", default="Full", help='Runtime option: [Full, UMAP, Clustering].')
    parser.add_option('-s', action="store", dest="spread", default=1.0, type=float, help='Spread parameter for UMAP generation.')
    parser.add_option('-t', action="store", dest="thread", type=int, default=1, help='Number of jobs.')
    parser.add_option('-w', action="store", dest="knn", default=30, type=int, help='Number of K-Nearest Neighbors for VIA. Range: 5-100. Default: 30')
    parser.add_option('-y', action="store", dest="maxclus", default=31, type=int, help='Exact number of clusters for meta-clustering. Max: 31.')
    parser.add_option('-z', action="store", dest="resolution", default=1.0, type=float, help='Resolution for VIA clustering. Range: 0.2-1.5. Default: 1.0')

    return parser.parse_args()

def main():
    options, args = parse_arguments()
    DictInfo = dict()

    run = Cytophenograph(
        info_file=options.pheno,
        input_folder=options.input_folder,
        output_folder=options.output_folder,
        k_coef=options.kmeancoef,
        marker_list=options.markerlist,
        analysis_name=options.analysis_name,
        thread=options.thread,
        tool=options.clustering,
        batch=options.batch,
        batchcov=options.batchcov,
        mindist=options.mindist,
        spread=options.spread,
        runtime=options.runtime,
        knn=options.knn,
        resolution=options.resolution,
        maxclus=options.maxclus,
        downsampling=options.downsampling,
        cellnumber=options.cellnumber,
        filetype=options.fileformat,
        arcsinh=options.arcsin
    )

    try:
        DictInfo["Infofile"] = run.read_info_file()
        DictInfo["List_csv_files"] = run.import_all_event()
        DictInfo["adata_conc"] = run.concatenate_dataframe(DictInfo["Infofile"], DictInfo["List_csv_files"])
        DictInfo["pathmarkerfile"], DictInfo["basenamemarkerfilepath"] = run.loadmarkers()
        DictInfo["markertoexclude"] = run.checkmarkers()
        DictInfo["markertoinclude"] = run.splitmarker()

        if options.runtime != 'UMAP':
            if options.clustering == "Phenograph":
                DictInfo["phenograph_adata"] = run.runphenograph()
            elif options.clustering == "VIA":
                DictInfo["via_adata"] = run.runvia()
            elif options.clustering == "FlowSOM":
                DictInfo["flowsom_adata"] = run.runflowsom()
            run.groupbycluster()
            run.groupbysample()
            run.exporting()

        if options.runtime == 'UMAP':
            run.runtimeumap()
            run.groupbysample()
            run.exporting()

    except Exception as e:
        traceback.print_tb(sys.exc_info()[2])
        print(f"Error: {e.args}")

        run.log.error("Execution Error!")
        sys.exit(1)

if __name__ == '__main__':
    main()
