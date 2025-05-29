import pytest
import pandas as pd
import os
from mismatch import AssignGroup
from mismatch.TNMismatchClustering import TNMismatchClustering


def test_imports():
    """Test that key modules can be imported."""
    assert AssignGroup is not None
    assert TNMismatchClustering is not None


def test_assign_group_function(tmp_path):
    """Smoke test for AssignGroup using minimal dummy files."""
    import numpy as np
    from mismatch.AssignGroup import AssignGroup

    # Create test output subfolder
    test_out = tmp_path / "assign_group_test"
    test_out.mkdir()

    # Define file paths
    subject_path = test_out / "subject.csv"
    biomarker_path = test_out / "biomarker.csv"
    output_path = test_out / "output.csv"
    ref_dir = test_out / "adni_reference"
    ref_dir.mkdir()

    # Dummy subject thickness data
    pd.DataFrame({
        "ID": [1],
        "x_left_Amygdala_volume": [1.1],
        "x_right_Amygdala_volume": [1.2],
        "x_left_0": [0.1],
        "x_right_0": [0.2],
    }).to_csv(subject_path, index=False)

    # Dummy biomarker data
    pd.DataFrame({
        "ID": [1],
        "pTau217": [0.5],
        "ICV": [1.6],
    }).to_csv(biomarker_path, index=False)

    # Minimal mock ADNI model coeffs
    pd.DataFrame({
        "Region": ["x_left_0", "x_right_0", "x_left_Amygdala_volume", "x_right_Amygdala_volume"],
        "Intercept": [2, 2, 1, 1],
        "Tau_Coeff": [-0.1, -0.1, -0.2, -0.2],
        "ICV_Coeff": [0.5, 0.5, 0.1, 0.1],
    }).to_csv(ref_dir / "adni_model_coefficients.csv", index=False)

    # Residual SDs
    pd.DataFrame({
        "Region": ["x_left_0", "x_right_0", "x_left_Amygdala_volume", "x_right_Amygdala_volume"],
        "SD": [1, 1, 1, 1],
    }).to_csv(ref_dir / "adni_residual_sd.csv", index=False)

    # Mock cluster centroids
    pd.DataFrame({
        "Cluster": [1, 2, 3],
        "x_left_0": [-1, 0, 1],
        "x_right_0": [-1, 0, 1],
        "x_left_Amygdala_volume": [-1, 0, 1],
        "x_right_Amygdala_volume": [-1, 0, 1],
        **{f"x_left_Amygdala_volume_dup{i}": [-1, 0, 1] for i in range(1, 7)},
        **{f"x_right_Amygdala_volume_dup{i}": [-1, 0, 1] for i in range(1, 7)},
    }).to_csv(ref_dir / "adni_cluster_centroids.csv", index=False)

    # Run AssignGroup
    result = AssignGroup(
        subject_path=str(subject_path),
        biomarker_path=str(biomarker_path),
        output_path=str(output_path),
        adni_reference_dir=str(ref_dir)
    )

    # Check outputs
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert "Assigned_Cluster" in df.columns
    assert len(df) == 1


def test_clustering_pipeline_runs(tmp_path):
    """Test if TNMismatchClustering runs with a small input and stores results in a dedicated folder."""
    pipeline = TNMismatchClustering(atlas="DKT")
    test_out = tmp_path / "clustering_test"
    test_out.mkdir()

    input_csv = "tests/Fake_DKT_input_with_Covariates2.csv"  # You must have this test CSV prepared
    output_prefix = test_out / "output"

    # Run the pipeline
    results = pipeline.run_pipeline(
        input_csv=input_csv,
        independent="tau",
        predict="thickness",
        sd_thresh=1.0,
        n_clusters=3,
        output_prefix=str(output_prefix),
        generate_maps=False,
        covariates=["age", "amyloid"]
    )

    # Check outputs
    assert results is not None
    assert os.path.exists(f"{output_prefix}_clusters.csv")
    assert os.path.exists(f"{output_prefix}_residuals.csv")
