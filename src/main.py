import argparse
import os
import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import umap
import hdbscan

# SciPy KDTree for geometric queries
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None  # optional; we will guard usage

# Plotly for interactive plots
import plotly.express as px
# For loading id map
try:
    from joblib import load as joblib_load
except Exception:
    try:
        import joblib
        joblib_load = joblib.load  # type: ignore[attr-defined]
    except Exception:
        joblib_load = None

# Load environment variables from .env early (supports src/.env or project root .env)
try:
    from dotenv import load_dotenv, find_dotenv
    # Prefer explicit src/.env, then auto-discovery
    dotenv_path = None
    for candidate in [Path(__file__).parent / ".env", Path.cwd() / ".env"]:
        if candidate.exists():
            dotenv_path = candidate
            break
    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=False)
        print(f"[INFO] Loaded .env from {dotenv_path}")
    else:
        # As a fallback, search upwards for a .env if present
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
            print(f"[INFO] Loaded .env from {found}")
except Exception:
    # dotenv is optional; proceed without if unavailable
    pass

# Optional MLflow import (installed via requirements). We'll guard usage by a CLI flag.
# Silence GitPython initialization warnings if git is not present

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
try:
    import mlflow
    from mlflow import MlflowClient
except Exception:
    mlflow = None
    MlflowClient = None

NUMBER_OF_OBSERVATIONS_TO_CONSIDER = 60000 #Adjust to not overwhelm the system
def find_input_in_dir(data_dir: Path) -> Path:
    # Prefer parquet if present, else fall back to CSV
    parquets = list(data_dir.glob("*.parquet"))
    if parquets:
        if len(parquets) > 1:
            print(f"[INFO] Multiple Parquet files found in {data_dir}. Using the first one: {parquets[0].name}")
        return parquets[0]
    csvs = list(data_dir.glob("*.csv"))
    if csvs:
        if len(csvs) > 1:
            print(f"[INFO] Multiple CSV files found in {data_dir}. Using the first one: {csvs[0].name}")
        return csvs[0]
    raise FileNotFoundError(f"No Parquet or CSV files found in {data_dir}")


def detect_column_types(
    df: pd.DataFrame,
    nominal_unique_threshold: int = 20,
    nominal_ratio_threshold: float = 0.05,
) -> Tuple[list, list]:
    """
    Detect numeric and categorical columns. Numeric columns with low cardinality
    are treated as categorical (nominal) based on either an absolute unique-count
    threshold or a ratio of unique values to non-null count.
    """
    # Initial detection by dtype
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Ensure no duplication between lists
    cat_cols = [c for c in cat_cols if c not in numeric_cols]

    # Drop columns that are all NaN
    numeric_cols = [c for c in numeric_cols if not df[c].isna().all()]
    cat_cols = [c for c in cat_cols if not df[c].isna().all()]

    # Reclassify numeric-but-nominal columns
    reclassified = []
    for col in list(numeric_cols):
        series = df[col]
        non_null = series.notna().sum()
        if non_null == 0:
            continue
        nunique = series.nunique(dropna=True)
        ratio = nunique / non_null if non_null > 0 else 1.0
        if (nunique <= nominal_unique_threshold) or (ratio <= nominal_ratio_threshold and nunique < non_null):
            # Treat as categorical
            numeric_cols.remove(col)
            if col not in cat_cols:
                cat_cols.append(col)
            reclassified.append(col)

    if reclassified:
        print(
            f"[INFO] Reclassified numeric columns as categorical due to low cardinality: {reclassified}"
        )
        print(
            f"[INFO] Criteria: unique<= {nominal_unique_threshold} or unique/non-null <= {nominal_ratio_threshold}"
        )

    return numeric_cols, cat_cols


def build_preprocessor(numeric_cols, cat_cols) -> ColumnTransformer:
    numeric_pipeline = (
        SimpleImputer(strategy="median"),
        StandardScaler(with_mean=True, with_std=True),
    )
    # Create a small pipeline-like function for ColumnTransformer
    from sklearn.pipeline import Pipeline

    numeric_pipe = Pipeline([
        ("imputer", numeric_pipeline[0]),
        ("scaler", numeric_pipeline[1]),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def reduce_dimensions(X: np.ndarray, method: str, random_state: Optional[int] = 42,
                      umap_n_neighbors: int = 15, umap_min_dist: float = 0.1) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(X)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
                            random_state=random_state)
        return reducer.fit_transform(X)
    else:
        raise ValueError("Unknown reducer. Use 'pca' or 'umap'.")


def cluster_data(X2d: np.ndarray, method: str, random_state: Optional[int] = 42, n_clusters: int = 8,
                 hdb_min_cluster_size: int = 10,
                 hdb_min_samples: int = 3,
                 hdb_cluster_selection_method:str="eom",
                 hdb_cluster_selection_epsilon: float = 0.01,
                 hdb_metric: str = "euclidean",
                 ):
    """Cluster the 2D embedding.
    Returns a tuple: (labels, model) for both methods to allow access to model-specific attributes.
    """
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = model.fit_predict(X2d)
        return labels, model
    elif method == "hdbscan":
        model = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size, metric=hdb_metric,
                                gen_min_span_tree=True,
                                prediction_data=True,
                                cluster_selection_method=hdb_cluster_selection_method,
                                cluster_selection_epsilon=hdb_cluster_selection_epsilon,
                                min_samples=hdb_min_samples,
                               )
        labels = model.fit_predict(X2d)
        dbcv = float(getattr(model, "relative_validity_", np.nan))
        print(f"[INFO] HDBSCAN relative_validity_ = {dbcv}")
        return labels, model
    else:
        raise ValueError("Unknown clusterer. Use 'kmeans' or 'hdbscan'.")


def plot_embedding(df_out: pd.DataFrame, output_path: Path, title: str) -> None:
    """Create an interactive Plotly 3D scatter plot and write HTML to disk.

    - Excludes HDBSCAN noise points (cluster == -1) from the plot.
    - Uses a dark theme and viridis color scale.
    - Z axis encodes drug status: 0 = not_on_drugs, 1 = on_drugs (see emb_z column).
    - Saves to an HTML file. If output_path has a .png suffix, we will instead write
      a sibling HTML named similarly (embedding.html) for backward compatibility,
      and try to also export a static PNG if kaleido is available.
    """
    # Determine outputs HTML path
    if output_path.suffix.lower() == ".png":
        html_path = output_path.with_suffix(".html")
    else:
        html_path = output_path

    # Exclude HDBSCAN noise points (labeled as -1) from the plot
    if "cluster" in df_out.columns and (df_out["cluster"] == -1).any():
        df_plot = df_out[df_out["cluster"] != -1].copy()
    else:
        df_plot = df_out

    if df_plot.empty:
        # Compose an empty figure with message
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            annotations=[dict(text="All points labeled as noise by HDBSCAN<br>No clusters to display.",
                              x=0.5, y=0.5, showarrow=False, font=dict(size=14))],
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        # Try to save PNG if requested path is PNG and kaleido is available
        if output_path.suffix.lower() == ".png":
            try:
                fig.write_image(str(output_path))
            except Exception:
                print("[WARN] Static PNG export requires kaleido. HTML plot was saved instead.")
        return

    # Build interactive 3D scatter
    fig = px.scatter_3d(
        df_plot,
        x="emb_x",
        y="emb_y",
        z="emb_z",
        color="cluster",
        color_continuous_scale="viridis" if df_plot["cluster"].dtype.kind in {"i","u","f"} else None,
        title=title,
        template="plotly_dark",
        hover_data={col: True for col in df_plot.columns if col not in ["emb_x", "emb_y", "emb_z"]},
    )

    # Ensure markers are small and semi-opaque for dense plots
    fig.update_traces(marker=dict(size=3, opacity=0.85))
    fig.update_layout(legend_title_text="Cluster", width=900, height=700,
                      scene=dict(zaxis_title="on_drugs (0/1)"))

    # Save interactive HTML
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    # If a PNG was requested, try exporting a static image via kaleido
    if output_path.suffix.lower() == ".png":
        try:
            fig.write_image(str(output_path))
        except Exception:
            print("[WARN] Static PNG export requires kaleido. Saved interactive HTML at:", html_path)


def epsilon_cover(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Cover a set of points in R^3 with balls of radius epsilon and return centers.

    Parameters:
    - points: (N, 3) array of coordinates
    - epsilon: radius of covering balls
    """
    if cKDTree is None:
        raise RuntimeError("scipy is required for epsilon_cover but is not available")
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 3), dtype=float)
    tree = cKDTree(pts)
    uncovered = np.ones(len(pts), dtype=bool)
    centers: List[np.ndarray] = []
    while np.any(uncovered):
        idx = np.where(uncovered)[0][0]
        center = pts[idx]
        centers.append(center)
        neighbors = tree.query_ball_point(center, epsilon)
        uncovered[np.asarray(neighbors, dtype=int)] = False
    return np.vstack(centers) if centers else np.empty((0, 3), dtype=float)


def select_not_on_drugs_by_cluster(
    df_out: pd.DataFrame,
    id_col: str,
    cluster_col: str,
    xyz_cols: Tuple[str, str, str],
    on_set: set,
    not_set: set,
    epsilon: float,
    use_cluster_scope: bool = True,
) -> Dict[int, List]:
    """For each cluster, epsilon-cover the on_drugs points, then for each center,
    select the closest not_on_drugs point. Returns {cluster_label: [ids,...]}.

    If use_cluster_scope=True, nearest not_on_drugs are searched within the same
    cluster; otherwise searched globally across all clusters.
    """
    if cKDTree is None:
        raise RuntimeError("scipy is required for selection step but is not available")

    xcol, ycol, zcol = xyz_cols
    if any(c not in df_out.columns for c in [id_col, cluster_col, xcol, ycol, zcol]):
        raise ValueError("Missing required columns for selection")

    result: Dict[int, List] = {}
    # Determine cluster labels excluding noise -1 if present
    clusters = sorted([c for c in df_out[cluster_col].unique()])
    for cl in clusters:
        # Optionally skip noise cluster -1
        try:
            if cl == -1:
                result[cl] = []
                continue
        except Exception:
            pass

        sub = df_out[df_out[cluster_col] == cl]
        if sub.empty:
            result[cl] = []
            continue
        # Separate on-drugs and not-on-drugs in this cluster (or global for not)
        mask_on = sub[id_col].isin(on_set)
        on_pts = sub.loc[mask_on, [xcol, ycol, zcol]].to_numpy(dtype=float)
        if use_cluster_scope:
            not_df = sub
        else:
            not_df = df_out
        mask_not = not_df[id_col].isin(not_set)
        not_df = not_df.loc[mask_not, [id_col, xcol, ycol, zcol]].copy()

        selected_ids: List = []
        if len(on_pts) == 0 or len(not_df) == 0:
            print(f"[INFO] Cluster {cl}: epsilon-cover centers=0 (on_drugs points={len(on_pts)}, not_on_drugs candidates={len(not_df)})")
            result[cl] = selected_ids
            continue

        centers = epsilon_cover(on_pts, epsilon)
        print(f"[INFO] Cluster {cl}: epsilon-cover centers={len(centers)}")
        # Build KDTree on not_on_drugs points
        tree = cKDTree(not_df[[xcol, ycol, zcol]].to_numpy(dtype=float))
        dists, idxs = tree.query(centers, k=1)
        # Map back to ids
        if np.ndim(idxs) == 0:
            idxs = np.array([int(idxs)])
        else:
            idxs = idxs.astype(int)
        sel = list(dict.fromkeys(not_df.iloc[idxs][id_col].tolist()))  # unique, keep order
        result[cl] = sel
    return result


def plot_epsilon_cover_3d(
    df_out: pd.DataFrame,
    id_col: str,
    cluster_col: str,
    xyz_cols: Tuple[str, str, str],
    on_set: set,
    not_set: set,
    epsilon: float,
    use_cluster_scope: bool,
    output_path: Path,
    max_spheres_per_cluster: int = 200,
) -> None:
    """Create a 3D plot showing:
    - All points (emb_x, emb_y, emb_z) colored by cluster (light markers)
    - Epsilon-cover centers for on_drugs per cluster (bigger markers)
    - Lines from each center to the selected not_on_drugs point.

    For performance, spheres (balls) are approximated as optional small Mesh3d
    spheres only when the number of centers in a cluster is <= max_spheres_per_cluster.
    Otherwise, only centers+lines are drawn.
    """
    import plotly.graph_objects as go

    xcol, ycol, zcol = xyz_cols

    # Base figure with all points excluding noise -1 for clarity
    if cluster_col in df_out.columns and (df_out[cluster_col] == -1).any():
        df_plot = df_out[df_out[cluster_col] != -1].copy()
    else:
        df_plot = df_out

    fig = go.Figure()
    # Add all points
    for cl, sub in df_plot.groupby(cluster_col):
        fig.add_trace(go.Scatter3d(
            x=sub[xcol], y=sub[ycol], z=sub[zcol],
            mode="markers",
            name=f"Cluster {cl}",
            marker=dict(size=2, opacity=0.5),
            hovertext=sub[id_col].astype(str),
            hoverinfo="text",
        ))

    # For lines and centers, iterate clusters
    clusters = sorted(df_out[cluster_col].unique())
    for cl in clusters:
        if cl == -1:
            continue
        sub = df_out[df_out[cluster_col] == cl]
        if sub.empty:
            continue
        on_mask = sub[id_col].isin(on_set)
        on_pts = sub.loc[on_mask, [xcol, ycol, zcol]].to_numpy(dtype=float)
        if use_cluster_scope:
            not_df = sub
        else:
            not_df = df_out
        not_df = not_df[not_df[id_col].isin(not_set)][[id_col, xcol, ycol, zcol]].copy()
        if len(on_pts) == 0 or len(not_df) == 0:
            continue
        centers = epsilon_cover(on_pts, epsilon)
        if len(centers) == 0:
            continue
        # nearest not_on_drugs per center
        tree = cKDTree(not_df[[xcol, ycol, zcol]].to_numpy(dtype=float))
        dists, idxs = tree.query(centers, k=1)
        if np.ndim(idxs) == 0:
            idxs = np.array([int(idxs)])
        else:
            idxs = idxs.astype(int)
        matched = not_df.iloc[idxs]

        # centers scatter
        fig.add_trace(go.Scatter3d(
            x=centers[:,0], y=centers[:,1], z=centers[:,2],
            mode="markers",
            name=f"C{cl} centers",
            marker=dict(size=5, opacity=0.95, symbol="cross"),
        ))

        # lines from center to matched point
        # build segment traces batched to reduce trace count
        seg_x, seg_y, seg_z = [], [], []
        for i in range(len(centers)):
            seg_x += [centers[i,0], matched.iloc[i][xcol], None]
            seg_y += [centers[i,1], matched.iloc[i][ycol], None]
            seg_z += [centers[i,2], matched.iloc[i][zcol], None]
        fig.add_trace(go.Scatter3d(
            x=seg_x, y=seg_y, z=seg_z,
            mode="lines",
            name=f"C{cl} matches",
            line=dict(width=2),
            opacity=0.7,
        ))

        # Optional spheres: skip by default for performance; include when small
        if len(centers) <= max_spheres_per_cluster:
            # Draw coarse spheres using parametric points aggregated in one Mesh3d per cluster
            # We approximate by plotting small transparent markers instead of heavy meshes for performance.
            # Represent balls as faint halos via additional markers at centers with larger size.
            fig.add_trace(go.Scatter3d(
                x=centers[:,0], y=centers[:,1], z=centers[:,2],
                mode="markers",
                name=f"C{cl} balls (approx)",
                marker=dict(size=12, opacity=0.1, color="lightblue"),
                showlegend=False,
            ))

    fig.update_layout(
        template="plotly_dark",
        title=f"Epsilon-cover (epsilon={epsilon})",
        width=1000, height=800,
        scene=dict(zaxis_title="on_drugs (0/1)")
    )

    # Save
    output_path = Path(output_path)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    # Try PNG export if requested
    if output_path.suffix.lower() == ".png":
        try:
            fig.write_image(str(output_path))
        except Exception:
            print("[WARN] Static PNG export requires kaleido. Saved interactive HTML instead.")

def plot_selected_scatter_matrix(
    df_out: pd.DataFrame,
    selection_by_cluster: Dict[int, List],
    id_col: str,
    output_path: Path,
) -> None:
    """Plot a scatter matrix (Plotly Express) for the rows whose IDs were
    selected_not_on_drugs_by_cluster. Points are colored by 'cluster'.

    We use the embedding coordinates (emb_x, emb_y, emb_z) as dimensions to
    provide a compact, meaningful view regardless of original feature set.
    """
    # Flatten selected IDs
    selected_ids: List = []
    for cl, ids in selection_by_cluster.items():
        if not ids:
            continue
        selected_ids.extend(list(ids))
    selected_ids = list(dict.fromkeys(selected_ids))  # unique preserving order

    if not selected_ids:
        print("[WARN] No selected IDs to plot in scatter matrix.")
        return

    if id_col not in df_out.columns:
        raise ValueError(f"ID column '{id_col}' not found in df_out.")

    dims = [c for c in ["emb_x", "emb_y", "emb_z"] if c in df_out.columns]
    if len(dims) < 2:
        print("[WARN] Not enough embedding dimensions found to build a scatter matrix.")
        return

    df_sel = df_out[df_out[id_col].isin(selected_ids)].copy()
    if df_sel.empty:
        print("[WARN] Filtered selection is empty; nothing to plot.")
        return

    fig = px.scatter_matrix(
        df_sel,
        dimensions=dims,
        color="cluster" if "cluster" in df_sel.columns else None,
        title="Scatter Matrix of Selected Not-On-Drugs by Cluster",
        template="plotly_dark",
        hover_name=id_col if id_col in df_sel.columns else None,
    )
    # Make points a bit larger for readability
    fig.update_traces(diagonal_visible=False, marker=dict(size=5, opacity=0.85))

    output_path = Path(output_path)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def plot_selected_scatter_matrix_original(
    df: pd.DataFrame,
    selection_by_cluster: Dict[int, List],
    id_col: str,
    dims: List[str],
    output_path: Path,
) -> None:
    """Plot a scatter matrix using the original (pre-preprocessing) numeric
    features for the rows whose IDs were selected_not_on_drugs_by_cluster.

    - Only numeric columns from `dims` will be used.
    - Colors points by 'cluster' if present.
    """
    # Flatten selected IDs
    selected_ids: List = []
    for _, ids in selection_by_cluster.items():
        if not ids:
            continue
        selected_ids.extend(list(ids))
    selected_ids = list(dict.fromkeys(selected_ids))

    if not selected_ids:
        print("[WARN] No selected IDs to plot in original-feature scatter matrix.")
        return

    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in dataframe.")

    # Keep only available numeric dims
    available_dims = [c for c in dims if c in df.columns]
    numeric_dims = [c for c in available_dims if pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_dims) < 2:
        print("[WARN] Not enough numeric original features to plot a scatter matrix (need >=2).")
        return

    df_sel = df[df[id_col].isin(selected_ids)].copy()
    if df_sel.empty:
        print("[WARN] Filtered selection is empty for original-feature plot; nothing to plot.")
        return

    fig = px.scatter_matrix(
        df_sel,
        dimensions=numeric_dims,
        color="cluster" if "cluster" in df_sel.columns else None,
        title="Scatter Matrix (Original Features) of Selected Not-On-Drugs by Cluster",
        template="plotly_dark",
        hover_name=id_col if id_col in df_sel.columns else None,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=5, opacity=0.85))

    output_path = Path(output_path)
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def generate_cluster_explanations(df_out: pd.DataFrame, output_dir: Path) -> None:
    """Generate per-cluster explanations.

    Preferred behavior: when any of the preferred clustering columns are present,
    summarize those instead of legacy thyroid columns. Fallback to legacy logic
    if none of the preferred columns exist.
    """
    preferred_numeric = ["TSH_resultado", "T4L_resultado", "edadnum", "peso"]
    preferred_categorical = ["sexo", "nefro", "diabetes", "hta"]

    has_any_preferred = any(c in df_out.columns for c in preferred_numeric + preferred_categorical)

    # Exclude HDBSCAN noise cluster -1 from explanations if present
    mask_non_noise = ~(df_out.get("cluster", pd.Series([-1]*len(df_out))).eq(-1))
    data = df_out[mask_non_noise].copy()

    if has_any_preferred:
        # Use preferred columns for explanations
        group_cols = [c for c in ["cluster", "sexo"] if c in data.columns]
        if "cluster" not in group_cols:
            print("[WARN] 'cluster' column missing; cannot generate explanations.")
            return

        agg_spec = {"count": ("cluster", "size")}

        # Numeric summaries
        for col in preferred_numeric:
            if col in data.columns:
                agg_spec[f"{col}_mean"] = (col, lambda s: pd.to_numeric(s, errors='coerce').mean())
                agg_spec[f"{col}_median"] = (col, lambda s: pd.to_numeric(s, errors='coerce').median())

        # Rates for binary categorical flags
        def _rate(series):
            s = pd.to_numeric(series, errors='coerce')
            return float((s == 1).mean()) if len(s) else np.nan

        for col in [c for c in preferred_categorical if c in data.columns and c != "sexo"]:
            agg_spec[f"{col}_rate"] = (col, _rate)

        grouped = data.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
        grouped_overall = data.groupby(["cluster"], dropna=False).agg(**agg_spec).reset_index()

        # Save to CSV
        out_csv = output_dir / "cluster_explanations.csv"
        out_csv_overall = output_dir / "cluster_explanations_overall.csv"
        grouped.to_csv(out_csv, index=False)
        grouped_overall.to_csv(out_csv_overall, index=False)
        print(f"[INFO] Wrote cluster explanations (preferred) to {out_csv} and {out_csv_overall}")

        # Markdown summary
        md_lines = [
            "# Cluster Explanations (Preferred Columns)",
            "",
            "Notes:",
            "- *_rate fields are fractions where the column equals 1.",
            "- *_mean and *_median computed on numeric preferred columns.",
            "",
        ]
        for cl in sorted(grouped_overall["cluster"].unique()):
            md_lines.append(f"## Cluster {cl}")
            row_overall = grouped_overall[grouped_overall["cluster"] == cl].iloc[0]
            overall_parts = [f"n={int(row_overall['count'])}"]
            for col in preferred_numeric:
                if f"{col}_median" in grouped_overall.columns:
                    overall_parts.append(f"{col} median={row_overall[f'{col}_median']:.2f}")
            for col in [c for c in preferred_categorical if c in data.columns and c != "sexo"]:
                cname = f"{col}_rate"
                if cname in grouped_overall.columns:
                    overall_parts.append(f"{col} rate={row_overall[cname]:.2f}")
            md_lines.append("Overall: " + ", ".join(overall_parts))

            sub = grouped[grouped["cluster"] == cl]
            for _, r in sub.iterrows():
                seg_parts = [f"- "]
                if "sexo" in sub.columns:
                    seg_parts.append(f"sexo={r['sexo']}")
                seg_parts.append(f"n={int(r['count'])}")
                for col in preferred_numeric:
                    mcol = f"{col}_median"
                    if mcol in sub.columns and pd.notna(r[mcol]):
                        seg_parts.append(f"{col} median={r[mcol]:.2f}")
                for col in [c for c in preferred_categorical if c in data.columns and c != "sexo"]:
                    cname = f"{col}_rate"
                    if cname in sub.columns and pd.notna(r[cname]):
                        seg_parts.append(f"{col} rate={r[cname]:.2f}")
                md_lines.append(", ".join(seg_parts))
            md_lines.append("")

        md_path = output_dir / "cluster_explanations.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))
        print(f"[INFO] Wrote Markdown cluster explanations to {md_path}")
        return

    # No preferred columns found -> skip explanations per requirement to remove legacy columns
    print("[WARN] Preferred columns for explanations not found. Skipping cluster explanations.")
    return


def main():
    debugging = True

    root = Path("../") if debugging else Path("/app/")

    parser = argparse.ArgumentParser(description="Standardize numeric, one-hot encode categorical, reduce to 2D (PCA/UMAP), then cluster (KMeans/HDBSCAN)")
    parser.add_argument("--input", type=str, default=None, help="Path to input file (Parquet preferred, CSV supported). If not set, auto-detect in --data-dir.")
    parser.add_argument("--data-dir", type=str, default= root / "data", help="Directory to search for an input file if --input not provided.")
    parser.add_argument("--outputs-dir", type=str, default=None, help="Directory to write outputs. If not set, will use /app/outputs/outputs-{reducer}-{clusterer}.")
    parser.add_argument("--reducer", type=str, choices=["pca", "umap"], default="pca")
    parser.add_argument("--clusterer", type=str, choices=["kmeans", "hdbscan"], default="kmeans")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of clusters for KMeans.")
    parser.add_argument("--hdb-min-cluster-size", type=int, default=500, help="min_cluster_size for HDBSCAN.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)

    # Drug z-axis mapping configuration
    parser.add_argument("--id-column", type=str, default="id", help="Name of the ID column used to map on_drugs vs not_on_drugs")
    parser.add_argument("--drug-map-path", type=str, default=None, help="Path to tiroides_viva_processed_id_map.joblib; defaults to <data-dir>/tiroides_viva_processed_id_map.joblib if not provided")
    parser.add_argument("--drug-z-default", type=float, default=0.0, help="Default z value when ID not present in map (0=not on drugs)")

    parser.add_argument("--nominal-unique-threshold", type=int, default=5, help="Treat numeric columns with <= this many unique values as categorical (nominal).")
    parser.add_argument("--nominal-ratio-threshold", type=float, default=0.00005, help="Treat numeric columns as categorical if unique/non-null ratio <= this value.")

    # HDBSCAN hyperparameters
    parser.add_argument("--hdb-min-samples", type=int, default=3, help="min_samples for HDBSCAN (core distance neighborhood size)")
    parser.add_argument("--hdb-cluster-selection-method", type=str, choices=["eom","leaf"], default="eom", help="HDBSCAN cluster_selection_method")
    parser.add_argument("--hdb-cluster-selection-epsilon", type=float, default=0.01, help="HDBSCAN cluster_selection_epsilon")
    parser.add_argument("--hdb-metric", type=str, default="euclidean", help="Distance metric for HDBSCAN (e.g., euclidean, manhattan, cosine)")

    # MLflow related optional flags

    # Epsilon-cover selection options
    parser.add_argument("--epsilon-cover-enabled", action="store_true", help="Enable epsilon-cover selection of closest not_on_drugs by cluster")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Radius epsilon for covering balls in embedding space (units of emb_x,emb_y,emb_z)")
    parser.add_argument("--epsilon-use-cluster-scope", action="store_true", help="Limit nearest not_on_drugs search to same cluster")
    parser.add_argument("--epsilon-output", type=str, default=None, help="Optional path for joblib output; defaults to <outputs_dir>/selected_not_on_drugs_by_cluster.joblib")
    parser.add_argument("--epsilon-cover-plot", action="store_true", help="Also generate a 3D plot showing balls, centers, and lines to selected points")
    parser.add_argument("--epsilon-cover-plot-output", type=str, default=None, help="Optional path for epsilon cover plot HTML; defaults to <outputs_dir>/epsilon_cover.html")

    # MLflow related optional flags
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow experiment tracking")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", None), help="MLflow tracking URI (defaults to env MLFLOW_TRACKING_URI or local ./mlruns)")
    parser.add_argument("--mlflow-experiment", type=str, default=os.getenv("MLFLOW_EXPERIMENT_NAME", "BinClust"), help="MLflow experiment name")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Optional MLflow run name")
    parser.add_argument("--mlflow-artifacts", action="store_true", help="Also log output files as MLflow artifacts")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # Determine outputs directory: /app/outputs/outputs-{reducer}-{clusterer} if not overridden
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
    else:
        outputs_dir = root / "outputs"/ f"outputs-{args.reducer}-{args.clusterer}"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_input_in_dir(data_dir)

    print(f"[INFO] Loading dataset: {input_path}")
    try:
        if input_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_path, engine="pyarrow")
        elif input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        else:
            # Try parquet first, then CSV
            try:
                df = pd.read_parquet(input_path, engine="pyarrow")
            except Exception:
                df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to read input {input_path}: {e}")
        sys.exit(1)

    if df.empty:
        print("[ERROR] The input dataset is empty.")
        sys.exit(1)


    sample_n = NUMBER_OF_OBSERVATIONS_TO_CONSIDER
    if len(df) > sample_n:
        try:
            df = df.sample(n=sample_n, random_state=args.random_state).reset_index(drop=True)
            print(f"[INFO] Sampled {sample_n} rows out of {len(df)} for processing.")
        except Exception as e:
            print(f"[WARN] Failed to sample rows due to: {e}. Proceeding with full dataset.")

    # Feature selection: prefer user-requested clustering columns this columns must be in root/data/columns.pkl
    # Otherwise, use legacy thyroid columns

    with open(root / "data/columns.pkl", "rb") as f:
        columns_dict =pickle.load(f)
    preferred_features = columns_dict["preferred_features"]
    preferred_numeric = columns_dict["preferred_numeric"]

    existing = [c for c in preferred_features if c in df.columns]
    if existing:
        missing = [c for c in preferred_features if c not in df.columns]
        if missing:
            print(f"[WARN] Some requested features are missing and will be skipped: {missing}")
        df_features = df[existing].copy()
        # Define categorical columns explicitly among the selected set
        forced_cats = [c for c in ["sexo", "nefro", "diabetes", "hta"] if c in df_features.columns]
        # Force numeric to be numeric only (never categorical)
        for col in preferred_numeric:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce")
        forced_nums = [c for c in df_features.columns if c not in forced_cats]
        # Detect additional types only within the selected features, but enforce the forced sets
        auto_nums, auto_cats = detect_column_types(df_features,
            nominal_unique_threshold=args.nominal_unique_threshold,
            nominal_ratio_threshold=args.nominal_ratio_threshold,
        )
        # Merge with forced classification
        # Ensure forced numeric columns are not categorized as categorical by auto detection
        for must_num in preferred_numeric:
            if must_num in auto_cats:
                auto_cats = [c for c in auto_cats if c != must_num]
            if must_num in df_features.columns and must_num not in auto_nums:
                auto_nums.append(must_num)
        numeric_cols = sorted(set(auto_nums).intersection(forced_nums)) + [c for c in forced_nums if c not in auto_nums]
        cat_cols = sorted(set(auto_cats).union(forced_cats))
        df_for_model = df_features
        print(f"[INFO] Using requested feature subset: {existing}")
    else:
        print("[WARN] None of the requested features found. Falling back to automatic feature detection over all columns.")
        auto_nums, auto_cats = detect_column_types(
            df,
            nominal_unique_threshold=args.nominal_unique_threshold,
            nominal_ratio_threshold=args.nominal_ratio_threshold,
        )
        numeric_cols, cat_cols = auto_nums, auto_cats
        df_for_model = df

    print(f"[INFO] Detected {len(numeric_cols)} numeric and {len(cat_cols)} categorical columns.")

    if len(numeric_cols) == 0 and len(cat_cols) == 0:
        print("[ERROR] No usable columns found (numeric or categorical).")
        sys.exit(1)

    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    print("[INFO] Preprocessing (impute + scale + one-hot)...")
    X = preprocessor.fit_transform(df_for_model)

    # Get feature names after ColumnTransformer (numeric + one-hot categorical)
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = feature_names.tolist()
    except Exception:
        # Fallback: build names manually
        feature_names = []
        try:
            # Numeric transformer names come through as 'num__<col>'
            feature_names.extend([f"num__{c}" for c in preprocessor.transformers_[0][2]])
            # Categorical one-hot names via underlying encoder
            cat_pipeline = preprocessor.named_transformers_["cat"]
            ohe = cat_pipeline.named_steps["onehot"]
            ohe_names = ohe.get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
            feature_names.extend(ohe_names)
        except Exception:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

    print(f"[INFO] Feature matrix shape after preprocessing: {X.shape}")
    with open(outputs_dir / "preprocessed_data.pkl", "wb") as f:
        pickle.dump({"X": X, "feature_names": feature_names}, f)

    print(f"[INFO] Reducing to 2D using {args.reducer.upper()}...")
    X2d = reduce_dimensions(
        X,
        method=args.reducer,
        random_state=args.random_state,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )

    print(f"[INFO] Clustering using {args.clusterer.upper()}...")
    labels, model = cluster_data(
        X2d,
        method=args.clusterer,
        random_state=args.random_state,
        n_clusters=args.n_clusters,
        hdb_min_cluster_size=args.hdb_min_cluster_size,
        hdb_min_samples=args.hdb_min_samples,
        hdb_cluster_selection_method=args.hdb_cluster_selection_method,
        hdb_cluster_selection_epsilon=args.hdb_cluster_selection_epsilon,
        hdb_metric=args.hdb_metric,
    )

    # Optional MLflow logging (parameters, basic metrics)
    if args.mlflow and mlflow is not None:
        try:
            tracking_uri = args.mlflow_tracking_uri or (Path.cwd() / "mlruns").as_uri()
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            run_ctx = mlflow.start_run(run_name=args.mlflow_run_name)
            # Log parameters
            mlflow.log_params({
                "reducer": args.reducer,
                "clusterer": args.clusterer,
                "n_clusters": args.n_clusters,
                "hdb_min_cluster_size": args.hdb_min_cluster_size,
                "hdb_min_samples": args.hdb_min_samples,
                "hdb_cluster_selection_method": args.hdb_cluster_selection_method,
                "hdb_cluster_selection_epsilon": args.hdb_cluster_selection_epsilon,
                "hdb_metric": args.hdb_metric,
                "random_state": args.random_state,
                "umap_n_neighbors": args.umap_n_neighbors,
                "umap_min_dist": args.umap_min_dist,
                "nominal_unique_threshold": args.nominal_unique_threshold,
                "nominal_ratio_threshold": args.nominal_ratio_threshold,
                "n_rows": int(len(df)),
                "n_features_numeric": int(len(numeric_cols)),
                "n_features_categorical": int(len(cat_cols)),
            })
            # Compute simple metrics
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            metrics = {}
            unique_labels = np.unique(labels)
            # Silhouette score only valid when >1 cluster and no single label
            if args.clusterer == "kmeans" and len(unique_labels) > 1:
                try:
                    metrics["silhouette_2d"] = float(silhouette_score(X2d, labels))
                except Exception:
                    pass
            # General compactness metric
            if len(unique_labels) > 1:
                try:
                    metrics["davies_bouldin_2d"] = float(davies_bouldin_score(X2d, labels))
                except Exception:
                    pass
            # HDBSCAN specifics
            if args.clusterer == "hdbscan":
                n_noise = int((labels == -1).sum())
                n_clusters_found = int(len([l for l in unique_labels if l != -1]))
                # DBCV (Density-Based Clustering Validation) from HDBSCAN
                try:
                    dbcv = float(getattr(model, "relative_validity_", np.nan))
                    print(f"[INFO] HDBSCAN relative_validity_ = {dbcv}")
                except Exception:
                    dbcv = np.nan
                noise_ratio = float(n_noise / len(labels)) if len(labels) else 0.0
                metrics.update({
                    # Original keys (backward compatible)
                    "hdbscan_noise_count": n_noise,
                    "hdbscan_noise_ratio": noise_ratio,
                    "hdbscan_n_clusters": n_clusters_found,
                    # Requested names
                    "DBCV": dbcv,
                    "NOISE": noise_ratio,
                    "COUNT": n_clusters_found,
                })
            if metrics:
                mlflow.log_metrics(metrics)
        except Exception as e:
            print(f"[WARN] MLflow logging failed to initialize/log params/metrics: {e}")
            run_ctx = None
    else:
        run_ctx = None

    df_out = df.copy()
    # Attach embeddings/clusters aligned with df_for_model's row order
    # If df_for_model is a subset of columns only (same rows), alignment is 1:1
    df_out["emb_x"] = X2d[:, 0]
    df_out["emb_y"] = X2d[:, 1]
    df_out["cluster"] = labels

    # Compute emb_z from drug map (on_drugs -> 1, not_on_drugs -> 0)
    emb_z = np.full(len(df_out), float(args.drug_z_default), dtype=float)
    id_col = args.id_column
    # Resolve joblib map path
    drug_map_path = Path(args.drug_map_path) if args.drug_map_path else (data_dir / "tiroides_viva_processed_id_map.joblib")
    on_set, not_set = set(), set()
    try:
        if joblib_load is not None and drug_map_path.exists():
            m = joblib_load(str(drug_map_path))
            if isinstance(m, dict):
                if "on_drugs" in m:
                    on_set = set(m["on_drugs"]) if m["on_drugs"] is not None else set()
                    print(f"On drugs length {len(on_set)}")
                if "not_on_drugs" in m:
                    not_set = set(m["not_on_drugs"]) if m["not_on_drugs"] is not None else set()
        else:
            print(f"[WARN] Drug map not loaded (path not found or joblib unavailable): {drug_map_path}")
    except Exception as e:
        print(f"[WARN] Failed to load drug map from {drug_map_path}: {e}")

    if id_col in df_out.columns:
        try:
            ids = df_out[id_col]
            mask_on = ids.isin(on_set)
            mask_not = ids.isin(not_set)
            emb_z[mask_on.values] = 1.0
            emb_z[mask_not.values] = 0.0
        except Exception as e:
            print(f"[WARN] Failed to map emb_z using id column '{id_col}': {e}")
    else:
        print(f"[WARN] ID column '{id_col}' not found in data; emb_z will use default {args.drug_z_default} for all rows.")

    df_out["emb_z"] = emb_z + np.random.normal(0.0, 0.05, len(df_out))

    csv_out = outputs_dir / "clustered.csv"
    img_out = outputs_dir / "embedding.png"
    html_out = outputs_dir / "embedding.html"

    print(f"[INFO] Writing results to {csv_out}, {html_out} (PNG attempted if kaleido available)")
    df_out.to_csv(csv_out, index=False)

    title = f"Reducer={args.reducer.upper()} | Clusterer={args.clusterer.upper()}"
    # Save interactive Plotly HTML (and static PNG if kaleido available)
    plot_embedding(df_out, html_out, title)

    # Generate per-cluster explanations based on preferred columns; skip if not present
    try:
        generate_cluster_explanations(df_out, outputs_dir)
    except Exception as e:
        print(f"[WARN] Failed to generate cluster explanations: {e}")

    # Perform epsilon-cover selection if enabled
    try:
        if args.epsilon_cover_enabled:
            # Determine output path
            out_path = Path(args.epsilon_output) if args.epsilon_output else (outputs_dir / "selected_not_on_drugs_by_cluster.joblib")
            # Ensure we have joblib.dump
            try:
                from joblib import dump as joblib_dump
            except Exception:
                import joblib  # type: ignore
                joblib_dump = joblib.dump  # type: ignore
            if joblib_dump is None:
                raise RuntimeError("joblib is required to write selection output but is not available")
            # Use emb_x, emb_y, emb_z columns for geometry
            selection = select_not_on_drugs_by_cluster(
                df_out=df_out,
                id_col=id_col,
                cluster_col="cluster",
                xyz_cols=("emb_x", "emb_y", "emb_z"),
                on_set=on_set,
                not_set=not_set,
                epsilon=args.epsilon,
                use_cluster_scope=args.epsilon_use_cluster_scope,
            )
            joblib_dump(selection, str(out_path))
            print(f"[INFO] Wrote epsilon-cover selection to {out_path}")
            # Also create a scatter matrix for the selected IDs, colored by cluster
            try:
                scatter_matrix_path = outputs_dir / "selected_scatter_matrix.html"
                plot_selected_scatter_matrix(
                    df_out=df_out,
                    selection_by_cluster=selection,
                    id_col=id_col,
                    output_path=scatter_matrix_path,
                )
                print(f"[INFO] Wrote selected scatter matrix to {scatter_matrix_path}")
            except Exception as sm_e:
                print(f"[WARN] Failed to generate selected scatter matrix: {sm_e}")
            # Also create a scatter matrix using original (pre-preprocessing) numeric features
            try:
                orig_dims = [c for c in numeric_cols if c in df_out.columns]
                if len(orig_dims) >= 2:
                    scatter_matrix_orig_path = outputs_dir / "selected_scatter_matrix_original.html"
                    plot_selected_scatter_matrix_original(
                        df=df_out,
                        selection_by_cluster=selection,
                        id_col=id_col,
                        dims=orig_dims,
                        output_path=scatter_matrix_orig_path,
                    )
                    print(f"[INFO] Wrote original-feature selected scatter matrix to {scatter_matrix_orig_path}")
                else:
                    print("[WARN] Not enough numeric original features found to build original scatter matrix.")
            except Exception as sm2_e:
                print(f"[WARN] Failed to generate original-feature scatter matrix: {sm2_e}")
            # Optional 3D plot of balls, centers, and match lines
            if args.epsilon_cover_plot:
                cover_plot_path = Path(args.epsilon_cover_plot_output) if args.epsilon_cover_plot_output else (outputs_dir / "epsilon_cover.html")
                try:
                    plot_epsilon_cover_3d(
                        df_out=df_out,
                        id_col=id_col,
                        cluster_col="cluster",
                        xyz_cols=("emb_x", "emb_y", "emb_z"),
                        on_set=on_set,
                        not_set=not_set,
                        epsilon=args.epsilon,
                        use_cluster_scope=args.epsilon_use_cluster_scope,
                        output_path=cover_plot_path,
                    )
                    print(f"[INFO] Wrote epsilon-cover 3D plot to {cover_plot_path}")
                except Exception as pe:
                    print(f"[WARN] Epsilon-cover plot generation failed: {pe}")
    except Exception as e:
        print(f"[WARN] Epsilon-cover selection step failed: {e}")

    # Log artifacts to MLflow if requested
    if args.mlflow and mlflow is not None:
        try:
            if args.mlflow_artifacts:
                # Log primary outputs if they exist
                for p in [csv_out, html_out]:
                    if p.exists():
                        mlflow.log_artifact(str(p), artifact_path="outputs")
                # Log optional explanation files if present
                for name in [
                    "cluster_explanations.csv",
                    "cluster_explanations_overall.csv",
                    "cluster_explanations.md",
                ]:
                    p = outputs_dir / name
                    if p.exists():
                        mlflow.log_artifact(str(p), artifact_path="outputs")
        except Exception as e:
            print(f"[WARN] MLflow artifact logging failed: {e}")
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
