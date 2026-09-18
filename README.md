# Trajectory Clustering Comparison Methodology

Reference implementation for a comparative aircraft-trajectory clustering study using real-world ADS-B data.

## Overview

This repository isolates the clustering-comparison methodology used to evaluate multiple unsupervised learning approaches on equal-length aircraft trajectory representations. The research workflow compares:

- K-means;
- DBSCAN;
- HDBSCAN;
- Gaussian mixture models (GMM).

Candidate solutions are evaluated with standard internal clustering metrics together with algorithm-specific diagnostics such as noise fraction for density-based methods.

The original study used a representative five-hour LaGuardia-area ADS-B dataset containing 343 processed aircraft trajectories. That research dataset is not redistributed here. A deterministic synthetic dataset and test suite are included so the methodology can be exercised without the original data.

## What this repository demonstrates

- preparation of equal-length multivariate aircraft trajectories;
- flattening of trajectory time series into model-ready feature vectors;
- standardized preprocessing for distance-based clustering;
- K-means, DBSCAN, HDBSCAN, and GMM candidate fitting;
- silhouette, Davies-Bouldin, and Calinski-Harabasz evaluation;
- explicit treatment of density-clustering noise points;
- comparable candidate summaries across algorithms;
- a self-contained synthetic demonstration and regression tests.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── src/
│   └── trajectory_clustering/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── models.py
│       └── evaluation.py
├── examples/
│   └── run_synthetic_demo.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
└── docs/
    ├── REPRODUCIBILITY.md
    └── RESEARCH_CODE_PROVENANCE.md
```

## Quick start

Install the direct dependencies:

```bash
pip install -r requirements.txt
```

Run the synthetic comparison:

```bash
python -m examples.run_synthetic_demo
```

Run the tests:

```bash
pytest -q
```

## Relationship to the research code

This repository is a cleaned **reference implementation** of the comparison methodology rather than a bit-for-bit copy of the larger PhD research repository. It preserves the core algorithm families, trajectory representation, evaluation metrics, and treatment of noise while removing unrelated thesis orchestration and aircraft-categorization interpretation code.

The earlier public repository `aircraft-behaviour-categorization-model` corresponds to a different published study focused on aircraft categorization using clustering. This repository is intentionally narrower: its purpose is the **comparative clustering methodology** itself.

## Data

The original ADS-B study dataset and generated candidate-result artifacts are not redistributed here. See `docs/REPRODUCIBILITY.md` for the expected trajectory table format and the distinction between the synthetic demo and full study reproduction.

## Scope and limitations

Internal clustering metrics quantify properties of a candidate partition; they do not by themselves establish that clusters correspond to operationally meaningful aircraft categories. Interpretation and domain validation remain separate steps.

## Citation

The associated manuscript citation will be added when its final bibliographic information is available. A machine-readable repository citation is provided in `CITATION.cff`.

## Author

**Nicolas Vincent-Boulay**  
Aerospace engineering PhD candidate and machine-learning researcher
