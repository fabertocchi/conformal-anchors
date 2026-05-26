# conformal-anchors

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