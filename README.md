# Dockerized Clustering Pipeline

This project provides a dockerized pipeline that:

- Loads a dataset from `/app/data` (CSV)
- Standardizes numerical columns
- One-hot encodes categorical columns
- Reduces to 2D using PCA or UMAP
- Clusters using KMeans or HDBSCAN
- Saves outputs (clustered CSV and 2D plot) to `/app/output`

## Project Structure

- `src/main.py` — main entry point
- `requirements.txt` — Python dependencies
- `Dockerfile` — container definition
- `data/` — place your CSV here (mounted into the container)
- `output/` — results written here (mounted from the container)

## Build

```bash
docker build -t binclust:latest .
```

## Run (mount local data and output directories)

Assuming you have a local `data/` directory containing `your_data.csv` and an `output/` directory for results:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  binclust:latest
```

By default, the container will:
- Auto-detect the first CSV in `/app/data`
- Use PCA for dimensionality reduction
- Use KMeans for clustering

### Changing methods and parameters

- Use UMAP instead of PCA and HDBSCAN instead of KMeans:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  binclust:latest \
  --reducer umap --clusterer hdbscan
```

- Specify KMeans clusters and random seed:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  binclust:latest \
  --n-clusters 5 --random-state 123
```

- Specify an explicit input CSV (if multiple present):

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  binclust:latest \
  --input /app/data/your_data.csv
```

## Outputs

- `/app/output/clustered.csv` — original data plus columns: `emb_x`, `emb_y`, `cluster`
- `/app/output/embedding.png` — 2D scatter plot colored by cluster
- `/app/output/cluster_explanations.csv` — per-cluster summary by sex and binaryClass with TSH measured rate, TT4 and FTI stats
- `/app/output/cluster_explanations_overall.csv` — overall per-cluster summary
- `/app/output/cluster_explanations.md` — human-readable summary highlighting key stats per cluster

## Notes

- Missing values are imputed: numeric with median, categorical with most frequent.
- Numeric-but-nominal columns (low cardinality) are treated as categorical automatically.
  - Criteria: unique values <= `--nominal-unique-threshold` (default 20) OR unique/non-null ratio <= `--nominal-ratio-threshold` (default 0.05).
  - Configure via flags: `--nominal-unique-threshold` and `--nominal-ratio-threshold`.
- Categorical encoding uses `OneHotEncoder(handle_unknown="ignore")`.
- If multiple CSVs exist in `/app/data`, the first one (by directory listing order) is used unless `--input` is specified.
- For HDBSCAN, points labeled `-1` are considered noise.
- The 2D plot excludes these noise points so clusters are clearer; the CSV still contains all rows (including noise labeled as `-1`).
- The plot uses a dark background and a dynamic viridis color palette sized to the number of clusters, which helps distinguish many clusters (especially with HDBSCAN).
- Cluster explanation files summarize TSH measured rate and TT4/FTI statistics by cluster, stratified by sex and binaryClass. When "TT4 measured" / "FTI measured" columns exist, TT4/FTI statistics are computed only on rows where they are measured.
