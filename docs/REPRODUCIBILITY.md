# Reproducibility

## Study and demo separation

This repository contains a self-contained reference implementation of the clustering-comparison methodology. The original research study used a processed five-hour LaGuardia-area ADS-B dataset containing 343 aircraft trajectories represented at 64 equal-length sequence steps with trajectory duration and original point count appended as two scalar features. That study dataset is not redistributed here.

The included synthetic example is intended to verify the computational workflow, not to recreate the published/manuscript numerical results.

## Method-level workflow

From the repository root:

    python -m venv .venv
    pip install -e ".[dev]"
    python -m examples.run_synthetic_demo
    pytest -q

The example:

1. creates four deterministic synthetic trajectory families;
2. interpolates every trajectory to 64 sequence steps and appends trajectory duration plus original point count;
3. standardizes the feature matrix;
4. fits K-means, DBSCAN, HDBSCAN, and GMM candidates;
5. reports cluster counts, noise, silhouette, Davies-Bouldin, Calinski-Harabasz, and runtime.

## Expected real-data schema

The preprocessor expects long-form trajectory observations with:

- one trajectory identifier column;
- one numeric time column;
- three or more numeric trajectory feature columns.

For the research workflow, the principal trajectory representation is three-dimensional position over time.

## Metric convention

Noise labels from DBSCAN and HDBSCAN are retained in assignments and reported through noise count/fraction. Noise samples are excluded from silhouette, Davies-Bouldin, and Calinski-Harabasz calculations.

## Determinism

K-means and GMM use a fixed random seed by default in this reference implementation. DBSCAN and HDBSCAN are deterministic for fixed inputs and parameters.
