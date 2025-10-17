# Dockerized Clustering Pipeline

This project provides a dockerized pipeline that:

- Loads a dataset from `/app/data` (Parquet preferred, CSV supported)
- Standardizes numerical columns
- One-hot encodes categorical columns
- Reduces to 2D using PCA or UMAP
- Clusters using KMeans or HDBSCAN
- Saves outputs (clustered CSV and 2D plot) to `/app/outputs/outputs-{reducer}-{clusterer}` by default (can be overridden with --outputs-dir)

## Project Structure

- `src/main.py` — main entry point
- `requirements.txt` — Python dependencies
- `Dockerfile` — container definition
- `data/` — place your CSV here (mounted into the container)
- `outputs/` — base directory on the host that you can mount to `/app/outputs` inside the container. By default, results go to `/app/outputs/outputs-{reducer}-{clusterer}` (e.g., `/app/outputs/outputs-pca-kmeans`).

## Build

```bash
docker build -t binclust:latest .
```

## Run (mount local data and output directories)

Assuming you have a local `data/` directory containing `your_data.csv` and an `output/` directory for results:

```bash
docker run --rm \
  -v (pwd)/src:/app/src \
  -v (pwd)/data:/app/data \
  -v (pwd)/outputs:/app/outputs \
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
  -v (pwd)/src:/app/src \
  -v (pwd)/data:/app/data \
  -v (pwd)/outputs:/app/outputs \
  binclust:latest --reducer umap --clusterer hdbscan
```

- Specify KMeans clusters and random seed:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  binclust:latest \
  --n-clusters 5 --random-state 123
```

- Specify an explicit input CSV (if multiple present):

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  binclust:latest \
  --input /app/data/your_data.csv
```

## Environment variables via .env

The app can load environment variables from a .env file using python-dotenv.

- Recommended location: src/.env (a .env exists already in this repo as an example)
- Also supported: a project-root .env or any .env discoverable upwards from the working directory
- Variables in the .env will NOT override variables already present in the environment (override=False)

Example src/.env:

```
MLFLOW_TRACKING_URI="http://69.62.69.86:5000"
MLFLOW_EXPERIMENT_NAME="BinClust"
```

You can keep secrets out of the image by mounting src/ at runtime; main.py will attempt to load src/.env automatically.

## MLflow Experiment Tracking

This project can optionally log experiments with MLflow.

Flags:
- `--mlflow` to enable MLflow tracking
- `--mlflow-tracking-uri` to set the tracking server (defaults to env `MLFLOW_TRACKING_URI` or local `./mlruns` if not set)
- `--mlflow-experiment` to choose experiment name (default `BinClust`)
- `--mlflow-run-name` to set a custom run name
- `--mlflow-artifacts` to also upload the generated outputs as artifacts

Example (local tracking store mounted to host):

```bash
mkdir -p mlruns
# Start a run with MLflow logging and artifact upload
docker run --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/mlruns:/app/mlruns \
  -e MLFLOW_TRACKING_URI=file:/app/mlruns \
  binclust:latest \
  --reducer umap --clusterer hdbscan \
  --mlflow --mlflow-artifacts --mlflow-experiment BinClust --mlflow-run-name "umap-hdbscan"
```

If you have an MLflow Tracking Server available, set `--mlflow-tracking-uri` or the env var `MLFLOW_TRACKING_URI` to point to it (e.g., `http://mlflow:5000`).

Note: The container sets `GIT_PYTHON_REFRESH=quiet` to silence GitPython initialization warnings when Git is not present. This does not affect tracking; it only hides the noisy startup message about missing Git.

### HDBSCAN parameters and metrics
You can control HDBSCAN with the following flags:
- `--hdb-min-cluster-size` (int): min_cluster_size
- `--hdb-min-samples` (int): min_samples
- `--hdb-cluster-selection-method` (eom|leaf): cluster selection method
- `--hdb-cluster-selection-epsilon` (float): epsilon for cluster selection
- `--hdb-metric` (str): distance metric (e.g., euclidean, manhattan, cosine)

Example:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  binclust:latest \
  --reducer umap --clusterer hdbscan \
  --hdb-min-cluster-size 100 --hdb-min-samples 5 \
  --hdb-cluster-selection-method eom --hdb-cluster-selection-epsilon 0.05 \
  --hdb-metric cosine
```

### Drug status 3D Z-axis
The embedding plot is now 3D. The Z axis encodes drug status derived from a joblib map file `tiroides_viva_processed_id_map.joblib` with keys `on_drugs` and `not_on_drugs`:
- IDs in `on_drugs` plot at z=1
- IDs in `not_on_drugs` plot at z=0
- IDs not present in the map use a default z (0 by default)

Flags:
- `--id-column`: name of the identifier column to match against the sets (default: `id`).
- `--drug-map-path`: path to the joblib file (default: `<data-dir>/tiroides_viva_processed_id_map.joblib`).
- `--drug-z-default`: z value for IDs not present in either set (default: 0.0).

Example:
```bash
python src/main.py --reducer umap --clusterer hdbscan \
  --input data/tiroides_viva_processed.parquet \
  --id-column id \
  --drug-map-path data/tiroides_viva_processed_id_map.joblib
```

### HDBSCAN metrics logged
When using `--clusterer hdbscan` with `--mlflow`, the following metrics are logged:
- DBCV (as `DBCV`) using `relative_validity_` from HDBSCAN
- Noise ratio (as `NOISE` and `hdbscan_noise_ratio`)
- Cluster count excluding noise (as `COUNT` and `hdbscan_n_clusters`)
- Additionally, `hdbscan_noise_count` is logged

### Epsilon-cover selection of nearest not_on_drugs
You can compute, per cluster, an epsilon-cover over points labeled on_drugs in the 3D embedding (emb_x, emb_y, emb_z). For each cover center, the closest point labeled not_on_drugs is selected. The result is saved as a joblib file containing a dict {cluster_label: [selected_not_on_drugs_ids]}. When enabled, the pipeline logs per cluster how many balls (epsilon-cover centers) were used, e.g., `[INFO] Cluster 3: epsilon-cover centers=12`.

Flags:
- `--epsilon-cover-enabled`: enable this step
- `--epsilon <float>`: radius in embedding units (default 0.2)
- `--epsilon-use-cluster-scope`: if set, restrict nearest not_on_drugs search to the same cluster; otherwise search globally
- `--epsilon-output <path>`: optional explicit output path; by default saved to `<outputs_dir>/selected_not_on_drugs_by_cluster.joblib`
- `--epsilon-cover-plot`: also generate an interactive 3D plot showing balls (approx), centers, and lines to selected points
- `--epsilon-cover-plot-output <path>`: optional path for that plot (default: `<outputs_dir>/epsilon_cover.html`)

Example:
```bash
python src/main.py --reducer umap --clusterer hdbscan \
  --input data/tiroides_viva_processed.parquet \
  --drug-map-path data/tiroides_viva_processed_id_map.joblib \
  --id-column id \
  --epsilon-cover-enabled --epsilon 0.3 --epsilon-use-cluster-scope --epsilon-cover-plot
```

Output file structure example:
```python
{
  0: [1234, 5678, 9012],
  1: [3456, 7890],
  # ... per cluster label, noise -1 skipped with empty list
}
```

## Outputs

- `/app/outputs/outputs-{reducer}-{clusterer}/clustered.csv` — original data plus columns: `emb_x`, `emb_y`, `emb_z`, `cluster`
- `/app/outputs/outputs-{reducer}-{clusterer}/embedding.html` — interactive 3D scatter plot (Plotly) colored by cluster, with Z axis indicating drug status (0=not_on_drugs, 1=on_drugs). Also attempts `embedding.png` if kaleido is available
- `/app/outputs/outputs-{reducer}-{clusterer}/cluster_explanations.csv` — per-cluster summary using preferred columns (TSH_resultado, T4L_resultado, edadnum, peso, sexo, nefro, diabetes, hta) when available; if these columns are not present, the explanations are skipped
- `/app/outputs/outputs-{reducer}-{clusterer}/cluster_explanations_overall.csv` — overall per-cluster summary
- `/app/outputs/outputs-{reducer}-{clusterer}/cluster_explanations.md` — human-readable summary highlighting key stats per cluster

## Notes

- Missing values are imputed: numeric with median, categorical with most frequent.
- Numeric-but-nominal columns (low cardinality) are treated as categorical automatically.
  - Criteria: unique values <= `--nominal-unique-threshold` (default 20) OR unique/non-null ratio <= `--nominal-ratio-threshold` (default 0.05).
  - Configure via flags: `--nominal-unique-threshold` and `--nominal-ratio-threshold`.
- Categorical encoding uses `OneHotEncoder(handle_unknown="ignore")`.
- If multiple CSVs exist in `/app/data`, the first one (by directory listing order) is used unless `--input` is specified.
- For HDBSCAN, points labeled `-1` are considered noise.
- The 2D plot excludes these noise points so clusters are clearer; the CSV still contains all rows (including noise labeled as `-1`).
- The interactive Plotly plot uses a dark theme and a viridis color scale, which helps distinguish many clusters (especially with HDBSCAN).
- Cluster explanation files prioritize the preferred columns for summaries (TSH_resultado, T4L_resultado, edadnum, peso for numeric; sexo, nefro, diabetes, hta for categorical). If these columns are not present in your dataset, explanation files will be skipped to avoid misleading outputs.
