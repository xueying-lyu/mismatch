# mismatch: A Python Package for Modeling Biomarker Discordance

`mismatch` is a Python package and analysis pipeline for modeling discordance between biological markers, with a focus on neuroimaging and biomarker data in Alzheimer’s disease.

The `mismatch` approach was originally developed to study the discordance between tau pathology (T) and neurodegeneration (N) in Alzheimer's disease (AD), with the goal of identifying phenotypes associated with non-AD factors beyond tau driving neurodegeneration. It enables the discovery of spatially distinct **T-N phenotypes** by clustering regional mismatch patterns: 

- ⚫ **Canonical (N ~ T)**: Neurodegeneration aligns with tau burden, reflecting typical AD patients.
- 🔴 **Vulnerable (N > T)**: Greater-than-expected neurodegeneration, suggesting additional co-morbidities (e.g., LATE, vascular).
- 🔵 **Resilient (N < T)**: Less-than-expected neurodegeneration, potentially reflecting resilient/protective factors.

Acknowledgement
============
If you use `mismatch`, please cite the following core papers:

[Medial temporal lobe Tau-Neurodegeneration mismatch from MRI and plasma biomarkers identifies vulnerable and resilient phenotypes with AD](https://www.medrxiv.org/content/10.1101/2025.08.17.25333859v1)

[Tau-neurodegeneration mismatch reveals vulnerability and resilience to comorbidities in Alzheimer's continuum](https://alz-journals.onlinelibrary.wiley.com/doi/full/10.1002/alz.13559)

[Tau-Atrophy Variability Reveals Phenotypic Heterogeneity in Alzheimer's Disease](https://onlinelibrary.wiley.com/doi/10.1002/ana.26233)

[Dissociation of tau pathology and neuronal hypometabolism within the ATN framework of Alzheimer’s disease](https://www.nature.com/articles/s41467-022-28941-1)

Thank you for supporting this project!

---

Purpose and Use Case
============

In heterogeneous diseases such as Alzheimer’s disease, downstream outcomes such as neurodegeneration are strongly associated with tau pathology but are also influenced by comorbid pathologies and resilience factors. However, direct in vivo biomarkers for non-AD contributors (e.g. TDP-43) are often limited. When tau alone does not fully account for neurodegenerative changes, their mismatch relationship can reflect meaningful biological processes rather than just "noise". In our work, we leverage T-N mismatch to identify data-driven T-N phenotypes showed association with LATE-NC, vascular pathology, and resilience. Modeling mismatch offers an approach for patients stratifications and offering insights into vulnerability and protective mechanisms beyond canonical pathology.

Notably, although originally developed to study Tau–Neurodegeneration (T–N) mismatch in Alzheimer's disease, the mismatch package is a **general framework** for modeling biomarker discordance. It can be applied to **other imaging or non-imaging biomarkers across various domains** where expected vs. observed relationships are informative.

**Caution**:
**Mismatch modeling assumes there is a meaningful relationship between the independent and predicted variables.
If the association is too weak, the residuals may not represent biologically meaningful mismatch.**


---

Installation
============

```sh
pip install mismatch
python3 -m mismatch --help
```

Or, if you want to use the latest development code and install in "editable" mode:

```bash
git clone https://github.com/xueying-lyu/mismatch.git
cd mismatch
pip install -e .
```

---

Core Functionalities
============

The `mismatch` package is organized into modular components centered around three core functionalities:

## 1. mismatch modeling and phenotypic discovery
### - `mismatch.TNMismatchClustering` 
**Purpose**: Identifies mismatch-based phenotypes by modeling regional discordance between an independent biomarker (e.g., tau) and a predicted biomarker (e.g., neurodegeneration). It applies robust linear regression to model the expected relationship between biomarkers, calculates subject-level regional residuals as mismatch, and cluster indviduals into data-driven phenotypes based on these mismatch patterns. This enables identification of canonical (N~T), vulnerable (N>T), and resilient (N<T) phenotypes with distinct mismatch patterns.

The module generates:
- Data-driven phenotype assignments
- Subject-level mismatch residual patterns  
- Subject-level global mismatch scores 
- Visualizations of group-level mismatch patterns  

It offers flexible and configurable options to customize the modeling process, such as:
- Specifying region sepcific or global covariates  
- Setting standard deviation thresholds for binarization of residuals
- Choosing number of clusters or estimated number 
- Using provided or user-customized atlases  

## 2. Individualized Phenotype Assignment

In our recent work, we performed T-N mismatch clustering within the medial temporal lobe (MTL) on ADNI participants to identify robust phenotypes. We then translated this framework to real-world clinical cohorts, including patients undergoing anti-amyloid therapy, by assigning individuals to the pretrained ADNI-derived mismatch phenotypes—**without the need to re-run clustering**. 

We provide this group assignment functionality in the mismatch package, enabling individual-level phenotype classification for research and clinical cohorts.

### - `mismatch.MTLSuperPoints`  
**Purpose**: Parcellates medial temporal lobe (MTL) surface meshes into anatomically-constrained superpoints using PyMetis and computes superpoint-wise cortical thickness measures. These regional thickness features serve as the input for T-N phenotype assignment.

This step requires preprocessing structural MRI data using the [CRASHS](https://github.com/pyushkevich/crashs) pipeline, which generates surface meshes aligned to an MTL template for consistent feature extraction across subjects.

### - `mismatch.AssignGroup`  
**Purpose**: Assigns each subject to one of the pretrained T-N mismatch phenotypes using ADNI-derived centroids and standardization parameters. This allows fast, individualized group membership assignment aligned with established groups without need to redo clustering.

This individual-level assignment module supports scalable phenotyping in new datasets and clinical applications.


mismatch Tutorials and Usage
============
## mismatch tutorial

We provide a [Jupyter notebook tutorial](notebook.ipynb) for using `mismatch` package.

## mismatch Command-Line Usage
The `mismatch` package also supports command line version besides the python function. 

Run `python3 -m mismatch <Mode> --help` to view available options.

The toolkit supports three command-line modes:

### TNMismatchClustering

Run `python3 -m mismatch TNMismatchClustering --help` to view available options: 

* `--input` Path to the input CSV containing ROI-level biomarker data (e.g., region1_tau, region1_thickness, region2_tau, region2_thickness).
* `--independent` Biomarker used as the predictor (e.g., tau. Must match column pattern like regionX_tau). 
* `--predict` Biomarker to be predicted (e.g., thickness. Must match column pattern like regionX_thickness).
* `--sd` Standard deviation threshold for residual binarization (default = 1.5) for clustering. Use 0 to disable and use raw residual for clustering.
* `--n_clusters` Choose number of clusters for hierarchical clustering. If not specified, the optimal number estimated by elbow method is used.
* `--cov` Optional covariates for modeling included in the input CSV (e.g., age). This will add age as covariate for all regions. For ROI-specific, use *variable_name to match ROI names (e.g., "*amyloid",  CSV must contain regionX_amyloid). This will only include amyloid as covariate for regionX with regionX_amyloid columns in the input csv. 
* `--feature_weights` Optional weights to apply to each ROI’s binarized mismatch value before clustering. Provide as comma-separated region:weight pairs (e.g.,region31:2.65,region32:1.5). These weights scale the contribution of each ROI in the clustering step.
* `--atlas` Choose predefined atlas to use (DKT or BrainCOLOR). It cannot be used together with --custom_atlas_path or --custom_label_csv
* `--custom_atlas_path` Path to custom atlas segmentation NIfTI file if not using predefined atlas. Must be used with --custom_label_csv.
* `--custom_label_csv` Path to a CSV mapping segmentation label indices to region names.(Required if using --custom_atlas_path.)(CSV should include columns like LabelID, LabelName.)
* `--no_log` If set, do NOT apply log10 to the independent variable (tau).
* `--plot` Save dendrogram and cluster visualization.
* `--require_residual_map` If set, generate NIfTI residual maps for each cluster using atlas segmentation. Requires atlas segmentation (either use --atlas or --custom_atlas_path) to map the mismatch patterns to a map. Only for neuroimaging biomarekers. 

example using neuroimaging biomarkers: 
```bash
python3 -m mismatch TNMismatchClustering \
  --input simulated_data/simulated_data_BrainCOLOR_label_names.csv \
  --out results/demo_run_clusters.csv \
  --out_resid results/demo_run_mismatch_output.csv \
  --independent tau \
  --predict thickness \
  --sd 1.5 \
  --n_clusters 6 \
  --plot \
  --require_residual_map \
  --atlas BrainCOLOR \
  --cov "*amy" "age" \
  --feature_weights 31:2.64,32:2.64
```

**! Note**:
*The `TNMismatchClustering` function is not limited to neuroimaging data.
Users can apply this tool to model mismatch between any paired biomarkers (e.g., fluid biomarkers, cognitive scores), not just imaging-derived features.*

In such cases, simply exclude the following neuroimaging-specific options:

--atlas \
--custom_atlas_path \
--custom_label_csv \
--require_residual_map 

The clustering and residual computation pipeline will still work with ROI-level or variable-level input data.



### MTLSuperPoints & AssignGroup

These modules are used together to assign individuals from a clinical or research cohort to pretrained T-N mismatch group derived from ADNI in our published paper. It enables individualized phenotype assignment in real-world cohorts without rerunning clustering.

-**`MTLSuperPoints`**

Run `python3 -m mismatch MTLSuperPoints` --help to view available options:

`--left_csv`, `--right_csv` CSV files containing subject IDs, scan dates, and mesh paths for left/right hemisphere. \
`--template_left`, `--template_right` Template meshes vtk file with anatomical label arrays.\
`--output_dir` Directory to save output files (including parcellations and subject level thickness txt).\
`--num_partitions` Number of triangle partitions per hemisphere (default = 50). Required to use 50 for doing the same T-N group assignment \
`--final_csv` Output CSV with merged thickness measures across parcellated super-points for each subject.

Example:

```bash
python3 -m mismatch MTLSuperPoints \
  --left_csv ./simulated_data/crashs/simulated_manifest_left.csv \
  --right_csv ./simulated_data/crashs/simulated_manifest_right.csv \
  --template_left ./simulated_data/crashs/template_shoot_left.vtk \
  --template_right ./simulated_data/crashs/template_shoot_right.vtk \
  --output_dir results/crashs_output \
  --final_csv results/crashs_output/final_output.csv
```

-**`AssignGroup`**

Run `python3 -m mismatch AssignGroup --help` to view available options:

`--subject` CSV with superpoint-wise thickness measures from MTLSuperPoints, which is the output CSV from `MTLSuperPoints`. \
`--biomarker` CSV containing features including pTau217, ICV, amygdala volumes. \
`--out` Output CSV for group assignment results. \
`--ref` (Optional) Custom path to a reference model directory (e.g., trained on your own dataset instead of default ADNI). Use this if you trained mismatch clusters on a new cohort and want to apply them elsewhere.

Example:
```bash
python3 -m mismatch AssignGroup \
  --subject results/crashs_output/final_output.csv \
  --biomarker ./simulated_data/crashs/simulated_biomarkers.csv \
  --out results/Assigned_TN_Phenotype.csv  
```

Join the mismatch Universe and Stay Curious
============

mismatch is a framework for rethinking what we often dismiss as “outliers” or “noise” in biomarker relationships. Whether you're studying disease heterogeneity or simply curious about why biology doesn’t always follow the rules, mismatch encourages you to look closer.

We're continuously working to improve and expand the toolkit. Contributions, issues, and new use cases are always welcome. Please feel free to open a pull request or share your usage experiences with us.

💡  Found an unexpected mismatch? It might be telling a story—one that a single biomarker alone can't explain.







---






