import argparse
from mismatch.TNMismatchClustering import TNMismatchClustering
from mismatch.MTLSuperPoints import create_MTLSuperPoints
from mismatch.AssignGroup import AssignGroup

def parse_weights(weight_str):
    weights = {}
    if weight_str:
        for pair in weight_str.split(","):
            if ":" not in pair:
                raise ValueError(f"Invalid format in --feature_weights: {pair}")
            feature, weight = pair.split(":")
            weights[str(feature).strip()] = float(weight)
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Mismatch Toolkit: Choose a mode to run.\n\n"
                    "Mismatch Clustering: Perform robust modeling of mismatch between paired variables "
                    "(e.g., tau and thickness), apply hierarchical clustering on residual patterns, and "
                    "optionally generate cluster-wise heatmaps and NIfTI maps.\n\n"
                    "MTL Superpoints: Partition MTL meshes into spatially contiguous regions (superpoints) "
                    "using label-wise graph-based clustering and compute regional mean thickness."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # 🔍 Subparser for TNMismatchClustering
    tn_parser = subparsers.add_parser("TNMismatchClustering", help="🔍 Model T-N mismatch and generate data-driven mismatch groups")
    tn_parser.add_argument("--input", required=True,
                           help=" Input CSV with ROI-level tau and thickness data per subject.")
    tn_parser.add_argument("--out", required=True,
                           help=" Output CSV for subject-level cluster labels.")
    tn_parser.add_argument("--out_resid", required=True,
                           help=" Output CSV for residuals (subject x ROI).")
    tn_parser.add_argument("--independent", required=True,
                           help=" Variable to predict from (e.g., tau). Should be part of column names.")
    tn_parser.add_argument("--predict", required=True,
                           help=" Variable to predict (e.g., thickness). Should match column name pattern.")
    tn_parser.add_argument("--sd", type=float, default=1.5,
                           help=" Standard deviation threshold for residual binarization. Use 0 to skip. Default=1.5.")
    tn_parser.add_argument("--n_clusters", type=int, default=None,
                           help=" Number of clusters. If not specified, elbow method is used.")
    tn_parser.add_argument("--cov", nargs="*",
                           help=" Optional covariates for RLM. Use *varname for ROI-specific covariates.")
    tn_parser.add_argument("--no_log", action="store_true",
                           help=" Disable log10 transform on the independent variable.")
    tn_parser.add_argument("--plot", action="store_true",
                           help=" Save dendrogram plot of clustering.")
    tn_parser.add_argument("--require_residual_map", action="store_true",
                           help=" Generate NIfTI residual maps by cluster.")
    tn_parser.add_argument("--atlas", choices=["DKT", "BrainCOLOR"],
                           help=" Use a predefined atlas (DKT or BrainCOLOR).")
    tn_parser.add_argument("--custom_atlas_path",
                           help=" Path to custom atlas segmentation NIfTI.")
    tn_parser.add_argument("--custom_label_csv",
                           help=" CSV mapping atlas label indices to ROI names.")
    tn_parser.add_argument("--feature_weights",type=str,
                           help="Comma-separated feature:weight pairs (e.g., region1:2.65,region2:2.65)")

    

    # 🧩 Subparser for MTLSuperPoints
    mtl_parser = subparsers.add_parser("MTLSuperPoints", help="🧩 Run MTL superpoint mesh partitioning and thickness extraction")
    mtl_parser.add_argument("--left_csv", required=True,
                            help=" CSV file with left hemisphere subject ID, MRIDATE, and VTK path.")
    mtl_parser.add_argument("--right_csv", required=True,
                            help=" CSV file with right hemisphere subject ID, MRIDATE, and VTK path.")
    mtl_parser.add_argument("--template_left", required=True,
                            help=" Left hemisphere template mesh with label array.")
    mtl_parser.add_argument("--template_right", required=True,
                            help=" Right hemisphere template mesh with label array.")
    mtl_parser.add_argument("--output_dir", required=True,
                            help=" Directory to store output VTK and thickness .txt files.")
    mtl_parser.add_argument("--num_partitions", type=int, default=50,
                            help=" Total number of triangle partitions (default = 50), and required to use 50.")
    mtl_parser.add_argument("--final_csv", required=True,
                            help=" Output CSV with merged left and right mean thickness per subject.")



        # 🎯 Subparser for AssignGroup
    assign_parser = subparsers.add_parser("AssignGroup", help="🎯 Assign subjects to T-N mismatch clusters using reference centroids")
    assign_parser.add_argument("--subject", required=True, help="📄 Merged thickness CSV")
    assign_parser.add_argument("--biomarker", required=True, help="🧬 CSV with pTau217, ICV, and Amygdala volumes")
    assign_parser.add_argument("--out", required=True, help="📁 Output CSV for assigned clusters")
    assign_parser.add_argument("--ref", default=None, help="📂 Path to ADNI reference model directory (optional)")


    args = parser.parse_args()


    if args.mode == "TNMismatchClustering":
        feature_weights = parse_weights(args.feature_weights)

        clustering = TNMismatchClustering(
            atlas=args.atlas,
            custom_atlas_path=args.custom_atlas_path,
            custom_label_csv=args.custom_label_csv
        )
        clustering.run_pipeline(
            input_csv=args.input,
            independent=args.independent,
            predict=args.predict,
            sd_thresh=args.sd,
            n_clusters=args.n_clusters,
            output_prefix=args.out.replace("_clusters.csv", ""),
            generate_maps=args.require_residual_map,
            covariates=args.cov,
            feature_weights=feature_weights  
        )


    elif args.mode == "MTLSuperPoints":
        create_MTLSuperPoints(
            left_csv=args.left_csv,
            right_csv=args.right_csv,
            template_left=args.template_left,
            template_right=args.template_right,
            output_dir=args.output_dir,
            num_partitions=args.num_partitions,
            final_csv=args.final_csv
        )
    elif args.mode == "AssignGroup":
        AssignGroup(
            subject_path=args.subject,
            biomarker_path=args.biomarker,
            output_path=args.out,
            adni_reference_dir=args.ref
        )




if __name__ == "__main__":
    main()
