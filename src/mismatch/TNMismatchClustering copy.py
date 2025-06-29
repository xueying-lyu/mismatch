# tn_mismatch_clustering.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import nibabel as nib

DKT_LABEL_MAP = {
    1002: 'lh.caudalanteriorcingulate', 1003: 'lh.caudalmiddlefrontal', 1005: 'lh.cuneus',
    1006: 'lh.entorhinal', 1007: 'lh.fusiform', 1008: 'lh.inferiorparietal', 1009: 'lh.inferiortemporal',
    1010: 'lh.isthmuscingulate', 1011: 'lh.lateraloccipital', 1012: 'lh.lateralorbitofrontal',
    1013: 'lh.lingual', 1014: 'lh.medialorbitofrontal', 1015: 'lh.middletemporal', 1016: 'lh.parahippocampal',
    1017: 'lh.paracentral', 1018: 'lh.parsopercularis', 1019: 'lh.parsorbitalis', 1020: 'lh.parstriangularis',
    1021: 'lh.pericalcarine', 1022: 'lh.postcentral', 1023: 'lh.posteriorcingulate', 1024: 'lh.precentral',
    1025: 'lh.precuneus', 1026: 'lh.rostralanteriorcingulate', 1027: 'lh.rostralmiddlefrontal',
    1028: 'lh.superiorfrontal', 1029: 'lh.superiorparietal', 1030: 'lh.superiortemporal',
    1031: 'lh.supramarginal', 1034: 'lh.transversetemporal', 1035: 'lh.insula',
    2002: 'rh.caudalanteriorcingulate', 2003: 'rh.caudalmiddlefrontal', 2005: 'rh.cuneus',
    2006: 'rh.entorhinal', 2007: 'rh.fusiform', 2008: 'rh.inferiorparietal', 2009: 'rh.inferiortemporal',
    2010: 'rh.isthmuscingulate', 2011: 'rh.lateraloccipital', 2012: 'rh.lateralorbitofrontal',
    2013: 'rh.lingual', 2014: 'rh.medialorbitofrontal', 2015: 'rh.middletemporal',
    2016: 'rh.parahippocampal', 2017: 'rh.paracentral', 2018: 'rh.parsopercularis',
    2019: 'rh.parsorbitalis', 2020: 'rh.parstriangularis', 2021: 'rh.pericalcarine',
    2022: 'rh.postcentral', 2023: 'rh.posteriorcingulate', 2024: 'rh.precentral',
    2025: 'rh.precuneus', 2026: 'rh.rostralanteriorcingulate', 2027: 'rh.rostralmiddlefrontal',
    2028: 'rh.superiorfrontal', 2029: 'rh.superiorparietal', 2030: 'rh.superiortemporal',
    2031: 'rh.supramarginal', 2034: 'rh.transversetemporal', 2035: 'rh.insula'
}



BrainCOLOR_LABEL_MAP = {
    100: 'Right.ACgG.anterior.cingulate.gyrus',
    101: 'Left.ACgG.anterior.cingulate.gyrus',
    102: 'Right.AIns.anterior.insula',
    103: 'Left.AIns.anterior.insula',
    104: 'Right.AOrG.anterior.orbital.gyrus',
    105: 'Left.AOrG.anterior.orbital.gyrus',
    106: 'Right.AnG.angular.gyrus',
    107: 'Left.AnG.angular.gyrus',
    108: 'Right.Calc.calcarine.cortex',
    109: 'Left.Calc.calcarine.cortex',
    112: 'Right.CO.central.operculum',
    113: 'Left.CO.central.operculum',
    114: 'Right.Cun.cuneus',
    115: 'Left.Cun.cuneus',
    116: 'Right.Ent.entorhinal.area',
    117: 'Left.Ent.entorhinal.area',
    118: 'Right.FO.frontal.operculum',
    119: 'Left.FO.frontal.operculum',
    120: 'Right.FRP.frontal.pole',
    121: 'Left.FRP.frontal.pole',
    122: 'Right.FuG.fusiform.gyrus',
    123: 'Left.FuG.fusiform.gyrus',
    124: 'Right.GRe.gyrus.rectus',
    125: 'Left.GRe.gyrus.rectus',
    128: 'Right.IOG.inferior.occipital.gyrus',
    129: 'Left.IOG.inferior.occipital.gyrus',
    132: 'Right.ITG.inferior.temporal.gyrus',
    133: 'Left.ITG.inferior.temporal.gyrus',
    134: 'Right.LiG.lingual.gyrus',
    135: 'Left.LiG.lingual.gyrus',
    136: 'Right.LOrG.lateral.orbital.gyrus',
    137: 'Left.LOrG.lateral.orbital.gyrus',
    138: 'Right.MCgG.middle.cingulate.gyrus',
    139: 'Left.MCgG.middle.cingulate.gyrus',
    140: 'Right.MFC.medial.frontal.cortex',
    141: 'Left.MFC.medial.frontal.cortex',
    142: 'Right.MFG.middle.frontal.gyrus',
    143: 'Left.MFG.middle.frontal.gyrus',
    144: 'Right.MOG.middle.occipital.gyrus',
    145: 'Left.MOG.middle.occipital.gyrus',
    146: 'Right.MOrG.medial.orbital.gyrus',
    147: 'Left.MOrG.medial.orbital.gyrus',
    148: 'Right.MPoG.postcentral.gyrus.medial.segment',
    149: 'Left.MPoG.postcentral.gyrus.medial.segment',
    150: 'Right.MPrG.precentral.gyrus.medial.segment',
    151: 'Left.MPrG.precentral.gyrus.medial.segment',
    152: 'Right.MSFG.superior.frontal.gyrus.medial.segment',
    153: 'Left.MSFG.superior.frontal.gyrus.medial.segment',
    154: 'Right.MTG.middle.temporal.gyrus',
    155: 'Left.MTG.middle.temporal.gyrus',
    156: 'Right.OCP.occipital.pole',
    157: 'Left.OCP.occipital.pole',
    160: 'Right.OFuG.occipital.fusiform.gyrus',
    161: 'Left.OFuG.occipital.fusiform.gyrus',
    162: 'Right.OpIFG.opercular.part.of.IFG',
    163: 'Left.OpIFG.opercular.part.of.IFG',
    164: 'Right.OrIFG.orbital.part.of.IFG',
    165: 'Left.OrIFG.orbital.part.of.IFG',
    166: 'Right.PCgG.posterior.cingulate.gyrus',
    167: 'Left.PCgG.posterior.cingulate.gyrus',
    168: 'Right.PCu.precuneus',
    169: 'Left.PCu.precuneus',
    170: 'Right.PHG.parahippocampal.gyrus',
    171: 'Left.PHG.parahippocampal.gyrus',
    172: 'Right.PIns.posterior.insula',
    173: 'Left.PIns.posterior.insula',
    174: 'Right.PO.parietal.operculum',
    175: 'Left.PO.parietal.operculum',
    176: 'Right.PoG.postcentral.gyrus',
    177: 'Left.PoG.postcentral.gyrus',
    178: 'Right.POrG.posterior.orbital.gyrus',
    179: 'Left.POrG.posterior.orbital.gyrus',
    180: 'Right.PP.planum.polare',
    181: 'Left.PP.planum.polare',
    182: 'Right.PrG.precentral.gyrus',
    183: 'Left.PrG.precentral.gyrus',
    184: 'Right.PT.planum.temporale',
    185: 'Left.PT.planum.temporale',
    186: 'Right.SCA.subcallosal.area',
    187: 'Left.SCA.subcallosal.area',
    190: 'Right.SFG.superior.frontal.gyrus',
    191: 'Left.SFG.superior.frontal.gyrus',
    192: 'Right.SMC.supplementary.motor.cortex',
    193: 'Left.SMC.supplementary.motor.cortex',
    194: 'Right.SMG.supramarginal.gyrus',
    195: 'Left.SMG.supramarginal.gyrus',
    196: 'Right.SOG.superior.occipital.gyrus',
    197: 'Left.SOG.superior.occipital.gyrus',
    198: 'Right.SPL.superior.parietal.lobule',
    199: 'Left.SPL.superior.parietal.lobule',
    200: 'Right.STG.superior.temporal.gyrus',
    201: 'Left.STG.superior.temporal.gyrus',
    202: 'Right.TMP.temporal.pole',
    203: 'Left.TMP.temporal.pole',
    204: 'Right.TrIFG.triangular.part.of.IFG',
    205: 'Left.TrIFG.triangular.part.of.IFG',
    206: 'Right.TTG.transverse.temporal.gyrus',
    207: 'Left.TTG.transverse.temporal.gyrus',
    31: 'Right.Amygdala', 32: 'Left.Amygdala', 47: 'Right.Hippocampus', 48: 'Left.Hippocampus',
    35: 'Brain.Stem', 36: 'Right.Caudate', 37: 'Left.Caudate',
    44: 'Right.Cerebral.White.Matter', 45: 'Left.Cerebral.White.Matter'
}

ATLAS_CONFIG = {
    "DKT": {
        "path": "/project/wolk/xyl/tn_mismatch/tn_mismatch/preprocess/tpl-ADNINormalAgingANTs_res-01_atlas-DKT_desc-31_dseg.nii.gz",
        "label_map": DKT_LABEL_MAP
    },
    "BrainCOLOR": {
        "path": "/project/wolk/xyl/tn_mismatch/tn_mismatch/preprocess/antsMalfLabeling.nii.gz",
        "label_map": BrainCOLOR_LABEL_MAP
    }
}


class TNMismatchClustering:
    def __init__(self, atlas=None, custom_atlas_path=None, custom_label_csv=None):
        self.atlas = atlas
        self.custom_atlas_path = custom_atlas_path
        self.custom_label_csv = custom_label_csv
        self.seg_path = None
        self.label_map = None
        self.allowed_labels = None
        self._load_atlas()

    def _load_atlas(self):
        if self.custom_atlas_path and self.custom_label_csv:
            self.seg_path = self.custom_atlas_path
            self.label_map = self.load_custom_label_csv(self.custom_label_csv)
            self.allowed_labels = set(self.label_map.keys())
        elif self.atlas:
            atlas_info = ATLAS_CONFIG.get(self.atlas)
            self.seg_path = atlas_info["path"]
            self.label_map = atlas_info["label_map"]
            self.allowed_labels = set(self.label_map.keys())

    def load_custom_label_csv(self, label_csv):
        df = pd.read_csv(label_csv)
        colnames = [c.lower() for c in df.columns]
        if "index" in colnames and "name" in colnames:
            index_col = df.columns[colnames.index("index")]
            name_col = df.columns[colnames.index("name")]
        elif "labelid" in colnames and "labelname" in colnames:
            index_col = df.columns[colnames.index("labelid")]
            name_col = df.columns[colnames.index("labelname")]
        else:
            raise ValueError("CSV must contain either 'index,name' or 'LabelID,LabelName' columns")
        return {int(row[index_col]): row[name_col] for _, row in df.iterrows()}

    def extract_matched_tau_thickness(self, df, independent, predict):
        tau_cols = [c for c in df.columns if c.endswith(f"_{independent}")]
        matched = []
        unmatched = []
        for c in tau_cols:
            base = c.replace(f"_{independent}", "")
            target = f"{base}_{predict}"
            if target in df.columns:
                matched.append((c, target))
            else:
                unmatched.append(base)
        if unmatched:
            raise ValueError(f"Missing {predict} for {', '.join(unmatched)}")
        tau_cols_final = [pair[0] for pair in matched]
        thickness_cols_final = [pair[1] for pair in matched]
        rois = [c.replace(f"_{independent}", "") for c in tau_cols_final]
        return df[tau_cols_final], df[thickness_cols_final], rois

    def compute_bisquare_residuals(self, tau_df, thickness_df, log_transform=True, cov_df=None, covariates=None):
        n_subjects, n_rois = tau_df.shape
        residuals = np.zeros((n_subjects, n_rois))

        roi_cov_usage = {}       # track ROI-specific covariate usage: {cov: set(roi_idx)}
        global_cov_used = set()  # track used global covariates
        missing_globals = set()  # track skipped global covariates
        all_roi_specific_cov = set()  # track all specified ROI-specific covariates (e.g., "amyloid")

        for i in range(n_rois):
            tau = tau_df.iloc[:, i]
            if log_transform:
                tau = np.log10(tau)

            thick = thickness_df.iloc[:, i]
            mask = tau.notna() & thick.notna()

            # Base design matrix with tau
            X = pd.DataFrame({'tau': tau})

            if covariates and cov_df is not None:
                for cov in covariates:
                    if cov.startswith("*"):
                        base = cov[1:]
                        all_roi_specific_cov.add(base)
                        roi_col = tau_df.columns[i].replace("_tau", f"_{base}")
                        if roi_col in cov_df.columns:
                            X[cov] = cov_df[roi_col]
                            roi_cov_usage.setdefault(base, set()).add(i)
                    else:
                        if cov in cov_df.columns:
                            X[cov] = cov_df[cov]
                            global_cov_used.add(cov)
                        else:
                            missing_globals.add(cov)

            # Ensure numeric
            non_numeric_cols = X.select_dtypes(exclude=["number"]).columns
            if len(non_numeric_cols) > 0:
                raise TypeError(f"❌ Non-numeric covariates detected in model for ROI {tau_df.columns[i]}: {', '.join(non_numeric_cols)}")

            try:
                X_masked = sm.add_constant(X[mask])
                y = thick[mask]
                rlm_model = sm.RLM(y, X_masked, M=sm.robust.norms.TukeyBiweight()).fit()
                pred = rlm_model.predict(sm.add_constant(X))
                residuals[:, i] = thick - pred
            except Exception as e:
                print(f"⚠️ RLM failed for ROI {tau_df.columns[i]}: {e}")
                residuals[:, i] = np.nan

        # === Summary messages ===
        if global_cov_used:
            print(f"    - Global covariates used: {', '.join(sorted(global_cov_used))}")
        if missing_globals:
            print(f"    - ⚠️ Global covariates NOT found and skipped: {', '.join(sorted(missing_globals))}")

        for base in sorted(all_roi_specific_cov):
            if base not in roi_cov_usage:
                print(f"    - ⚠️ ROI-specific covariate NOT found in any ROI: {base}")
            elif len(roi_cov_usage[base]) == n_rois:
                print(f"    - All ROIs used ROI-specific covariate: {base}")
            else:
                print(f"    - ROI-specific covariate used in {len(roi_cov_usage[base])} out of {n_rois} ROIs: {base}")

        return pd.DataFrame(residuals, columns=thickness_df.columns)

    #def binarize_residuals(self, res_df, sd_thresh):
    #    if sd_thresh == 0:
    #        return res_df
    #    bin_df = pd.DataFrame(index=res_df.index, columns=res_df.columns)
    #    for col in res_df.columns:
    #        sd = np.nanstd(res_df[col])
    #        bin_df[col] = res_df[col].apply(lambda x: -1 if x < -sd_thresh * sd else (1 if x > sd_thresh * sd else 0))
    #    return bin_df.fillna(0)

    def binarize_residuals(self, res_df, sd_thresh, feature_weights=None):
        if sd_thresh == 0:
            return res_df

        bin_df = pd.DataFrame(index=res_df.index, columns=res_df.columns)
        for col in res_df.columns:
            sd = np.nanstd(res_df[col])
            bin_df[col] = res_df[col].apply(
                lambda x: -1 if x < -sd_thresh * sd else (1 if x > sd_thresh * sd else 0)
            )

        bin_df = bin_df.fillna(0)

        if feature_weights:
            for col in bin_df.columns:
                base = col.split("_")[0]  # Extract '1002' from '1002_thickness'
                weight = feature_weights.get(base)
                if weight is not None:
                    bin_df[col] = bin_df[col].astype(float) * weight

            unmatched = set(feature_weights.keys()) - set([col.split("_")[0] for col in bin_df.columns])
            if unmatched:
                print(f"⚠️ Warning: These features in --feature_weights were not matched to any residual columns: {', '.join(unmatched)}")





        return bin_df


    def run_hierarchical_clustering(self, bin_df, n_clusters=None):
        distance_matrix = pdist(bin_df.values, metric="euclidean")
        linkage_matrix = linkage(distance_matrix, method="ward")
        if n_clusters is None:
            distances = linkage_matrix[:, 2]
            diff = np.diff(distances[::-1])
            elbow_index = np.argmax(diff[:9]) + 1
            n_clusters = elbow_index + 1
            print("\n🔎 Auto-selecting number of clusters based on elbow method...")
            print(f"📐 Optimal number of clusters estimated: {n_clusters}")
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
        return cluster_labels, linkage_matrix

    def determine_cluster_type_all_clusters(self, cluster_means, cluster_sizes):
        canonical_id = min(
            cluster_means.keys(),
            key=lambda cid: (np.nanmean(np.abs(cluster_means[cid])), -cluster_sizes[cid])
        )
        canonical_profile = cluster_means[canonical_id]
        cluster_types = {}
        for cid, mean_profile in cluster_means.items():
            if cid == canonical_id:
                cluster_types[cid] = "Likely Canonical"
            else:
                mean_residual = np.nanmean(mean_profile)
                canonical_mean_residual = np.nanmean(canonical_profile)
                if mean_residual < canonical_mean_residual:
                    cluster_types[cid] = "Likely Vulnerable"
                else:
                    cluster_types[cid] = "Likely Resilient"
        return cluster_types

    def map_residuals_to_nifti(self, cluster_id, mean_resid, rois, out_path):
        seg_img = nib.load(self.seg_path)
        seg_data = seg_img.get_fdata()
        out_data = np.zeros_like(seg_data, dtype=np.float32)
        roi_map = {int(r): mean_resid[i] for i, r in enumerate(rois)}
        for label, value in roi_map.items():
            out_data[seg_data == label] = value
        out_img = nib.Nifti1Image(out_data, affine=seg_img.affine, header=seg_img.header)
        nib.save(out_img, out_path)

    def plot_heatmap(self, df, out_path, rois=None, title=None):
        plt.figure(figsize=(max(20, df.shape[1] * 0.3), max(5, df.shape[0] * 0.3)))
        sns.heatmap(df, cmap="vlag", center=0, cbar_kws={"label": "Residual"},
                    xticklabels=rois if rois else True, yticklabels=False)
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.title(title or "Residual Heatmap")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def print_centered(self, text):
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 100
        print(text.center(width))
    
    @staticmethod
    def report_pairwise_correlations(tau_df, thickness_df, rois):
        sig_count = 0
        trend_count = 0
        total = len(rois)

        for i in range(total):
            tau = tau_df.iloc[:, i]
            thick = thickness_df.iloc[:, i]
            mask = tau.notna() & thick.notna()
            if mask.sum() < 3:
                continue
            _, pval = pearsonr(tau[mask], thick[mask])
            if pd.notna(pval):
                if pval < 0.05:
                    sig_count += 1
                elif pval < 0.1:
                    trend_count += 1

        print(f"\n Pearson correlation summary between independent and dependent variables:")

        print(f"  - {sig_count} out of {total} ROIs show significant correlation (p < 0.05)")
        print(f"  - {trend_count} out of {total} ROIs show a trend (0.05 < p < 0.1)")

        if (sig_count + trend_count) < (0.3 * total):
            print("   ⚠️ Caution: Looks like the correlation between your variables is on the low side.")
            print("     Keep in mind - Mismatch modeling works best when there's at least some relationship!")
    


              



    def plot_dendrogram(self, linkage_matrix, labels, out_path):
        plt.figure(figsize=(12, 6))
        dendrogram(
            linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=linkage_matrix[-(len(set(labels))-1), 2] if len(set(labels)) > 1 else 0
        )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Subject Index")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"🧬 Dendrogram saved to: {out_path}")
    
    def run_pipeline(self, input_csv, independent, predict, sd_thresh=1.5, n_clusters=None,
                 output_prefix="output", log_transform=True, generate_maps=True, covariates=None, feature_weights=None):

        self.print_centered("🌟 Welcome to the Mismatch Universe! 🌟")
        self.print_centered("🧠 PATCH Lab warmly welcomes you to explore brain heterogenity together!! 😊")

        df = pd.read_csv(input_csv)
        ids = df["ID"]
        tau_df, thickness_df, rois = self.extract_matched_tau_thickness(df, independent, predict)

        print(f"🗺️  Using predefined atlas: {self.atlas}")
        print(f"👤 Number of subjects: {df.shape[0]}")
        print(f"🧠 Number of ROIs: {len(rois)}")

        self.report_pairwise_correlations(tau_df, thickness_df, rois)

        

        print("📈 Running robust bisquare regression...")
        residuals = self.compute_bisquare_residuals(tau_df, thickness_df, log_transform=log_transform,
                                            cov_df=df, covariates=covariates)
        residuals.insert(0, "ID", ids)

        print(f"📊 Using residuals for clustering (SD={sd_thresh})...")
        binary_df = self.binarize_residuals(residuals.drop(columns="ID"), sd_thresh, feature_weights=feature_weights)


        print("🔗 Running Ward.D2 clustering...")
        labels, linkage_matrix = self.run_hierarchical_clustering(binary_df, n_clusters)
        residuals["T_N_cluster"] = labels

        global_mismatch_scores = residuals.drop(columns=["ID", "T_N_cluster"]).sum(axis=1) * -1
        result = pd.DataFrame({"ID": ids, "T_N_cluster": labels,
                               "Global_Mismatch_Score": global_mismatch_scores})

        print("\n📊 Cluster Membership Counts:")
        for cid, count in result["T_N_cluster"].value_counts().sort_index().items():
            print(f"  Cluster {cid}: {count} subjects")

        heatmap_dir = f"{output_prefix}_cluster_heatmaps"
        os.makedirs(heatmap_dir, exist_ok=True)

        cluster_means = {}
        cluster_sizes = {}
        cluster_heatmap_paths = []
        cluster_nii_paths = []




        roi_names = [self.label_map.get(int(r), r) for r in rois] if self.label_map else rois
        #roi_names = [self.label_map.get(r, r) for r in rois] if self.label_map else rois


        for cid in sorted(set(labels)):
            cluster_df = residuals[residuals["T_N_cluster"] == cid]
            cluster_means[cid] = cluster_df.drop(columns=["ID", "T_N_cluster"]).mean()
            cluster_sizes[cid] = len(cluster_df)
            heatmap_path = os.path.join(heatmap_dir, f"cluster_{cid}_residuals.png")
            cluster_heatmap_paths.append((cid, heatmap_path))

            self.plot_heatmap(
                cluster_df.drop(columns=["ID", "T_N_cluster"]),
                heatmap_path,
                roi_names,
                f"Residual Heatmap - Cluster {cid}"
            )
            print(f"🖼️ Residual heatmap saved for Cluster {cid} at: {heatmap_path}")

            if generate_maps and self.seg_path:
                mean_resid = cluster_means[cid].values
                nii_path = os.path.join(heatmap_dir, f"cluster_{cid}_residual_map.nii.gz")
                self.map_residuals_to_nifti(cid, mean_resid, rois, nii_path)
                print(f"🧠 Residual map for cluster {cid} saved to: {nii_path}")
                cluster_nii_paths.append((cid, nii_path))

        residuals_path = f"{output_prefix}_residuals.csv"
        residuals.to_csv(residuals_path, index=False)
        print(f"📁 Residuals saved to: {residuals_path}")

        cluster_path = f"{output_prefix}_clusters.csv"
        result.to_csv(cluster_path, index=False)
        print(f"📁 Cluster labels saved to: {cluster_path}")

        print(f"🗺️  Using predefined atlas: {self.atlas}")
        cluster_types = self.determine_cluster_type_all_clusters(cluster_means, cluster_sizes)
        for cid, ctype in cluster_types.items():
            print(f"✅ Cluster {cid} is {ctype}")

        name_df = pd.DataFrame([(cid, cluster_types[cid]) for cid in cluster_types],
                               columns=["T_N_cluster", "Cluster_Label"])
        final_result = result.merge(name_df, on="T_N_cluster")
        result_path = f"{output_prefix}_clusters_with_names.csv"
        final_result.to_csv(result_path, index=False)
        print(f"📁 Cluster labels with names saved to: {result_path}")

        full_heatmap = f"{output_prefix}_residual_heatmap_all.png"
        self.plot_heatmap(
            residuals.drop(columns=["ID", "T_N_cluster"]),
            full_heatmap,
            roi_names,
            "Residual Heatmap - All Subjects"
        )
        print(f"🖼️ Full residual heatmap saved to: {full_heatmap}")

        dendro_path = f"{output_prefix}_dendrogram.png"
        self.plot_dendrogram(linkage_matrix, labels=[str(i) for i in ids], out_path=dendro_path)

        return {
            "residual_df": residuals,
            "cluster_df": result,
            "final_df": final_result,
            "residual_csv": residuals_path,
            "cluster_csv": cluster_path,
            "final_csv": result_path,
            "cluster_heatmaps": cluster_heatmap_paths,
            "cluster_nifti_maps": cluster_nii_paths,
            "all_subjects_heatmap": full_heatmap,
            "dendrogram_path": dendro_path
        }
