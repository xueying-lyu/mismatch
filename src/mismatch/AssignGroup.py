import pandas as pd
import numpy as np
import argparse
import os
from mismatch import adni_reference  


def AssignGroup(subject_path, biomarker_path, output_path, binarization_sd=0.5, adni_reference_dir=None):
    if adni_reference_dir is None:
        adni_reference_dir = os.path.join(os.path.dirname(__file__), "adni_reference")

    def load_data(thickness_csv, biomarker_csv):
        thickness = pd.read_csv(thickness_csv)
        biomarkers = pd.read_csv(biomarker_csv)
        merged = thickness.merge(biomarkers, on="ID")
        return merged

    def load_adni_metadata():
        coeff = pd.read_csv(os.path.join(adni_reference_dir, "adni_model_coefficients.csv"))
        residual_sd = pd.read_csv(os.path.join(adni_reference_dir, "adni_residual_sd.csv"))
        centroids = pd.read_csv(os.path.join(adni_reference_dir, "adni_cluster_centroids.csv"))
        return coeff, residual_sd, centroids

    def predict_thickness(model_row, ptau_value, icv_value):
        intercept = model_row["Intercept"]
        tau_coeff = model_row["Tau_Coeff"]
        icv_coeff = model_row.get("ICV_Coeff", np.nan)
        pred = intercept + tau_coeff * np.log10(ptau_value)
        if not pd.isna(icv_coeff):
            pred += icv_coeff * icv_value
        return pred


    
    def binarize_residuals(residuals, sd_vector, threshold=0.5):
        binary = pd.DataFrame(index=residuals.index, columns=residuals.columns)
        for col in residuals.columns:
            sd_val = sd_vector.get(col, 1)
            z_scores = residuals[col].astype(float) / sd_val  # No rounding
            binary[col] = z_scores.apply(
                lambda z: -1 if z < -threshold else (1 if z > threshold else 0)
            )
        return binary

    

    def duplicate_specific_features(df):

        df = df.copy()
        amygdala_cols = ["x_left_Amygdala_volume", "x_right_Amygdala_volume"]
        weight = np.sqrt(7)

        for col in amygdala_cols:
            if col in df.columns:
                df[col] = df[col].astype(float) * weight
            else:
                raise ValueError(f"❌ Missing expected amygdala column: {col}")
        
        return df


    

    def euclidean_na(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        if np.sum(mask) == 0:
            return np.inf
        return np.sqrt(np.sum((a[mask] - b[mask]) ** 2))

    



    def assign_clusters(test_bin, centroid_bin):
        cluster_ids = centroid_bin["Cluster"].tolist()
        cluster_centers = centroid_bin.drop(columns=["Cluster"]).values
        assigned_labels = []
        all_distances = []
        for i in range(len(test_bin)):
            subj = test_bin.iloc[i].values.astype(float)
            dists = [euclidean_na(subj, center) for center in cluster_centers]
            assigned_labels.append(cluster_ids[np.argmin(dists)])
            all_distances.append(dists)
        cluster_label_map = {1: "Resilient", 2: "Vulnerable", 3: "Canonical"}
        dist_col_names = [f"Dist_Cluster{cluster_label_map.get(int(cid), f'Unknown_{cid}')}" for cid in cluster_ids]
        dist_df = pd.DataFrame(all_distances, columns=dist_col_names)
        return assigned_labels, dist_df

    test_df = load_data(subject_path, biomarker_path)
    coeff_df, adni_sd_df, centroid_df = load_adni_metadata()
    sd_series = dict(zip(adni_sd_df.Region, adni_sd_df.SD))

    roi_cols = coeff_df["Region"].unique()
    residual_matrix = pd.DataFrame(index=test_df.index)

    residual_dict = {}

    for roi in roi_cols:
        if roi not in test_df.columns:
            print(f"⚠️ Skipping missing ROI in subject data: {roi}")
            continue
        model_row = coeff_df[coeff_df["Region"] == roi].iloc[0]
        pred = test_df.apply(lambda row: predict_thickness(model_row, row["pTau217"], row["ICV"]), axis=1)
        residual_dict[roi] = test_df[roi] - pred

    # Efficiently create DataFrame from dictionary
    residual_matrix = pd.DataFrame(residual_dict)
    residual_matrix.insert(0, "ID", test_df["ID"])  # Add ID column as first column



    residual_matrix = pd.concat([test_df[["ID"]], residual_matrix], axis=1)


    #residual_matrix.to_csv("residuals_combined_atm.csv", index=False)

    residual_only = residual_matrix.drop(columns=["ID"])
    binary_resid = binarize_residuals(residual_only, sd_series, threshold=binarization_sd)

    expected_amygdala_cols = ["x_left_Amygdala_volume", "x_right_Amygdala_volume"]
    amygdala_cols = [col for col in residual_only.columns if col in expected_amygdala_cols]
    assert len(amygdala_cols) == 2, f"❌ Expected 2 Amygdala residual columns (left and right), but found: {amygdala_cols}"

    binary_resid = duplicate_specific_features(binary_resid)




    assigned_clusters, distance_df = assign_clusters(binary_resid, centroid_df)

    result = test_df[["ID"]].copy()
    cluster_label_map = {1: "Resilient", 2: "Vulnerable", 3: "Canonical"}
    result["Assigned_Cluster"] = [cluster_label_map.get(c, f"Unknown_{c}") for c in assigned_clusters]
    result = pd.concat([result, distance_df], axis=1)
    result.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")
    return result  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply ADNI T-N mismatch model to new subjects")
    parser.add_argument("--subject", required=True, help="Merged thickness CSV")
    parser.add_argument("--biomarker", required=True, help="CSV with pTau217, ICV, and Amygdala volumes")
    parser.add_argument("--out", required=True, help="Output CSV for assigned clusters")
    parser.add_argument("--ref", default=None, help="Path to ADNI reference model directory")
    
    args = parser.parse_args()

    results = AssignGroup(
        subject_path=args.subject,
        biomarker_path=args.biomarker,
        output_path=args.out,
        adni_reference_dir=args.ref
    ) 