# Conformal Anchors

   **Conformal Anchors** is a framework that turns any classifier handling missing values into a reliable hypothesis-rejection tool. It addresses two limitations common in high-stakes decision support: first, most systems assume full information, whereas in practice only partial observations are available and additional evidence can be acquired; second, when error costs are asymmetric, it is often more important to safely exclude dangerous hypotheses than to commit to a single prediction.
   
   Conformal Anchors answer questions of the form: *which features should be observed, and which values should they take, to safely reject a queried hypothesis?* To this end, the framework identifies simple, actionable, and locally consistent if-then rules over observed or acquirable features that exclude the queried label from the **conformal prediction set** — inheriting marginal guarantees on the erroneous exclusion of the true label. Three construction strategies are supported, trading off coverage, compactness, and runtime: one-shot (OS), maximum-gain union pruning (MG), and coverage-greedy union pruning (CG).
   
   This repository evaluates the framework on a synthetic medical diagnosis scenario: 100 000 patients described by generic symptoms and group-specific diagnostic tests, labeled with one of 10 diseases across two organ groups (Lung, Stomach). The underlying classifier is a XGBoost model.

## Setup

**Requirements:** Python 3.12, [conda](https://docs.conda.io/en/latest/miniconda.html)

### 1. Create and activate a conda environment

```bash
conda create -n conformal-anchors python=3.12
conda activate conformal-anchors
```

### 2. Install dependencies

Install the project in editable mode. This also adds the project root to `sys.path`, which is required for the `anchor` package to be resolved correctly.

```bash
pip install -e .
```

## Project Structure

### Data

| File | Description |
|------|-------------|
| `dataset_basic_setting.py` | Disease names, feature bounds, and missingness configuration |
| `data/tests_v1.py` | v1 test distributions: 3 diseases per organ group, less class-separable |
| `data/tests_v2.py` | v2 test distributions: 5 diseases per organ group, used throughout all experiments |
| `generated_datasets/synthetic_patient_basic_setting_v1.csv` | Synthetic patient dataset generated with the v1 test distributions (3 diseases per group) |
| `generated_datasets/synthetic_patient_basic_setting_v2.csv` | Synthetic patient dataset generated with the v2 test distributions (100 000 patients, 5 diseases per group) |
| `generated_datasets/synthetic_patient_basic_setting_v2_binned.csv` | v2 dataset with continuous features discretised into interpretable categories |

### Model

| File | Description |
|------|-------------|
| `binning.py` | Discretises continuous features into interpretable categories |
| `model_training.py` | Trains the XGBoost classifier and exports all data splits |
| `xgb_model.json` | Serialised pre-trained XGBoost model ready to be loaded by experiment scripts |

### Conformal framework

| File | Description |
|------|-------------|
| `prediction_set.py` | Split-conformal calibration and prediction set construction |
| `subset.py` | Gower-distance k-NN neighbourhood selection |

### Experiments

| File | Description |
|------|-------------|
| `evaluation_experiment.py` | Main benchmark: OS, MG, and CG modes evaluated on 100 test instances |
| `complete_experiment_excluding_TRUE_label.py` | Marginal guarantee stress test where the excluded label is the true label |
| `specific_experiment_case_studies.py` | Qualitative case studies on individual patients |
| `beam_size_experiment.py` | Ablation: union coverage vs. beam size |
| `beam_size_coverage_runtime.py` | Ablation: coverage and runtime vs. beam size |

### External dependency

| File | Description |
|------|-------------|
| `anchor/` | Vendored fork of the [Anchors](https://github.com/marcotcr/anchor) tabular explainer library |

## Usage

```bash
python evaluation_experiment.py
python complete_experiment_excluding_TRUE_label.py
python specific_experiment_case_studies.py
python beam_size_experiment.py
python beam_size_coverage_runtime.py
```

All scripts load the pre-trained model (`xgb_model.json`) and datasets automatically and use `numpy.random.seed(1)` for reproducibility.
