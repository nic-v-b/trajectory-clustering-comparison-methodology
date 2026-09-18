# Research-code provenance

This repository isolates the algorithm-comparison component of a larger aircraft-trajectory clustering research workflow.

The maintained PhD research code contains additional functionality for:

- thesis experiment orchestration;
- candidate sweeps and saved-result management;
- figure generation;
- cluster interpretation;
- aircraft metadata enrichment;
- comparison with rule-based aircraft categories.

Those concerns are intentionally excluded here.

The reference implementation preserves the central comparison pattern used by the research code:

- fixed-length trajectory interpolation;
- optional standardization;
- K-means, DBSCAN, HDBSCAN, and Gaussian-mixture clustering;
- explicit density-clustering noise handling;
- silhouette, Davies-Bouldin, and Calinski-Harabasz evaluation.

This repository is therefore suitable for understanding and exercising the methodology without exposing the private PhD repository or conflating the clustering-comparison study with the separate aircraft-categorization publication.
