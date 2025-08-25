import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import umap
import hdbscan

import matplotlib.pyplot as plt
import seaborn as sns


def find_csv_in_dir(data_dir: Path) -> Path:
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    if len(csvs) > 1:
        print(f"[INFO] Multiple CSV files found in {data_dir}. Using the first one: {csvs[0].name}")
    return csvs[0]


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
                 hdb_min_cluster_size: int = 10) -> np.ndarray:
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = model.fit_predict(X2d)
        return labels
    elif method == "hdbscan":
        model = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size)
        labels = model.fit_predict(X2d)
        return labels
    else:
        raise ValueError("Unknown clusterer. Use 'kmeans' or 'hdbscan'.")


def plot_embedding(df_out: pd.DataFrame, output_path: Path, title: str) -> None:
    # Use a dark theme for better contrast with many clusters
    plt.style.use("dark_background")
    plt.figure(figsize=(8, 6))

    # Exclude HDBSCAN noise points (labeled as -1) from the plot
    if "cluster" in df_out.columns and (df_out["cluster"] == -1).any():
        df_plot = df_out[df_out["cluster"] != -1].copy()
    else:
        df_plot = df_out

    if df_plot.empty:
        # All points are noise (e.g., HDBSCAN labeled everything as -1)
        plt.text(0.5, 0.5, "All points labeled as noise by HDBSCAN\nNo clusters to display.",
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return

    unique_labels = sorted(df_plot["cluster"].unique())
    # Use a viridis palette with as many colors as the number of clusters
    n_colors = max(len(unique_labels), 1)
    palette = sns.color_palette("viridis", n_colors=n_colors)
    sns.scatterplot(data=df_plot, x="emb_x", y="emb_y", hue="cluster", palette=palette, s=30, linewidth=0)
    plt.title(title)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_cluster_explanations(df_out: pd.DataFrame, output_dir: Path) -> None:
    required_by_cols = ["sex", "binaryClass"]
    value_cols = []
    # Determine availability of columns
    has_tsh_measured = "TSH measured" in df_out.columns
    has_tt4 = "TT4" in df_out.columns
    has_fti = "FTI" in df_out.columns
    has_tt4_measured = "TT4 measured" in df_out.columns
    has_fti_measured = "FTI measured" in df_out.columns

    if not all(col in df_out.columns for col in required_by_cols):
        missing = [c for c in required_by_cols if c not in df_out.columns]
        print(f"[WARN] Missing columns for grouping: {missing}. Skipping cluster explanations.")
        return

    # Exclude HDBSCAN noise cluster -1 from explanations if present
    mask_non_noise = ~(df_out.get("cluster", pd.Series([-1]*len(df_out))).eq(-1))
    data = df_out[mask_non_noise].copy()

    # Prepare aggregations
    agg_dict = {"cluster": "size"}  # we'll rename size to 'n'
    if has_tsh_measured:
        agg_dict["TSH measured"] = lambda s: float(np.mean(pd.to_numeric(s, errors='coerce') == 1)) if len(s) else np.nan
    if has_tt4:
        agg_dict["TT4_mean"] = ("TT4", lambda s: pd.to_numeric(s, errors='coerce').mean())
        agg_dict["TT4_median"] = ("TT4", lambda s: pd.to_numeric(s, errors='coerce').median())
    if has_fti:
        agg_dict["FTI_mean"] = ("FTI", lambda s: pd.to_numeric(s, errors='coerce').mean())
        agg_dict["FTI_median"] = ("FTI", lambda s: pd.to_numeric(s, errors='coerce').median())

    # Because pandas named aggregations require different syntax, build accordingly
    group_cols = ["cluster", "sex", "binaryClass"]

    def pct_tsh_measured(x):
        s = pd.to_numeric(x, errors='coerce')
        return float((s == 1).mean())

    agg_spec = {"count": ("cluster", "size")}
    if has_tsh_measured:
        agg_spec["tsh_measured_rate"] = ("TSH measured", pct_tsh_measured)
    def _mean_with_flag(vals, flag=None):
        s = pd.to_numeric(vals, errors='coerce')
        if flag is not None:
            mask = pd.to_numeric(flag, errors='coerce') == 1
            s = s[mask]
        return float(s.mean())

    def _median_with_flag(vals, flag=None):
        s = pd.to_numeric(vals, errors='coerce')
        if flag is not None:
            mask = pd.to_numeric(flag, errors='coerce') == 1
            s = s[mask]
        return float(s.median())

    if has_tt4:
        if has_tt4_measured:
            agg_spec["tt4_mean"] = ("TT4", lambda s: _mean_with_flag(s, data.loc[s.index, "TT4 measured"]))
            agg_spec["tt4_median"] = ("TT4", lambda s: _median_with_flag(s, data.loc[s.index, "TT4 measured"]))
        else:
            agg_spec["tt4_mean"] = ("TT4", lambda s: _mean_with_flag(s))
            agg_spec["tt4_median"] = ("TT4", lambda s: _median_with_flag(s))
    if has_fti:
        if has_fti_measured:
            agg_spec["fti_mean"] = ("FTI", lambda s: _mean_with_flag(s, data.loc[s.index, "FTI measured"]))
            agg_spec["fti_median"] = ("FTI", lambda s: _median_with_flag(s, data.loc[s.index, "FTI measured"]))
        else:
            agg_spec["fti_mean"] = ("FTI", lambda s: _mean_with_flag(s))
            agg_spec["fti_median"] = ("FTI", lambda s: _median_with_flag(s))

    grouped = data.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()

    # Overall per-cluster summary (not stratified by sex/binaryClass)
    grouped_overall = data.groupby(["cluster"], dropna=False).agg(**agg_spec).reset_index()

    # Save to CSV
    out_csv = output_dir / "cluster_explanations.csv"
    out_csv_overall = output_dir / "cluster_explanations_overall.csv"
    grouped.to_csv(out_csv, index=False)
    grouped_overall.to_csv(out_csv_overall, index=False)
    print(f"[INFO] Wrote cluster explanations to {out_csv} and {out_csv_overall}")

    # Also write a simple Markdown summary for readability
    md_lines = ["# Cluster Explanations", "", "Notes:",
                "- tsh_measured_rate: fraction of rows with 'TSH measured' == 1.",
                "- tt4_* and fti_* computed from available numeric values.", ""]
    for cl in sorted(grouped_overall["cluster"].unique()):
        md_lines.append(f"## Cluster {cl}")
        row_overall = grouped_overall[grouped_overall["cluster"] == cl].iloc[0]
        overall_parts = [f"n={int(row_overall['count'])}"]
        if has_tsh_measured:
            overall_parts.append(f"TSH measured rate={row_overall['tsh_measured_rate']:.2f}")
        if has_tt4:
            overall_parts.append(f"TT4 median={row_overall['tt4_median']:.2f}")
        if has_fti:
            overall_parts.append(f"FTI median={row_overall['fti_median']:.2f}")
        md_lines.append("Overall: " + ", ".join(overall_parts))
        # Per sex/binaryClass
        sub = grouped[grouped["cluster"] == cl]
        for _, r in sub.iterrows():
            seg = f"- sex={r['sex']}, binaryClass={r['binaryClass']}: n={int(r['count'])}"
            if has_tsh_measured:
                seg += f", TSH measured rate={r['tsh_measured_rate']:.2f}"
            if has_tt4:
                seg += f", TT4 median={r['tt4_median']:.2f}"
            if has_fti:
                seg += f", FTI median={r['fti_median']:.2f}"
            md_lines.append(seg)
        md_lines.append("")

    md_path = output_dir / "cluster_explanations.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"[INFO] Wrote Markdown cluster explanations to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Standardize numeric, one-hot encode categorical, reduce to 2D (PCA/UMAP), then cluster (KMeans/HDBSCAN)")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV. If not set, auto-detect in --data-dir.")
    parser.add_argument("--data-dir", type=str, default="/app/data", help="Directory to search for a CSV if --input not provided.")
    parser.add_argument("--output-dir", type=str, default="/app/output", help="Directory to write outputs.")
    parser.add_argument("--reducer", type=str, choices=["pca", "umap"], default="pca")
    parser.add_argument("--clusterer", type=str, choices=["kmeans", "hdbscan"], default="kmeans")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of clusters for KMeans.")
    parser.add_argument("--hdb-min-cluster-size", type=int, default=10, help="min_cluster_size for HDBSCAN.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--nominal-unique-threshold", type=int, default=20, help="Treat numeric columns with <= this many unique values as categorical (nominal).")
    parser.add_argument("--nominal-ratio-threshold", type=float, default=0.05, help="Treat numeric columns as categorical if unique/non-null ratio <= this value.")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        csv_path = Path(args.input)
    else:
        csv_path = find_csv_in_dir(data_dir)

    print(f"[INFO] Loading dataset: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV {csv_path}: {e}")
        sys.exit(1)

    if df.empty:
        print("[ERROR] The input dataset is empty.")
        sys.exit(1)

    numeric_cols, cat_cols = detect_column_types(
        df,
        nominal_unique_threshold=args.nominal_unique_threshold,
        nominal_ratio_threshold=args.nominal_ratio_threshold,
    )
    print(f"[INFO] Detected {len(numeric_cols)} numeric and {len(cat_cols)} categorical columns.")

    if len(numeric_cols) == 0 and len(cat_cols) == 0:
        print("[ERROR] No usable columns found (numeric or categorical).")
        sys.exit(1)

    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    print("[INFO] Preprocessing (impute + scale + one-hot)...")
    X = preprocessor.fit_transform(df)

    print(f"[INFO] Feature matrix shape after preprocessing: {X.shape}")

    print(f"[INFO] Reducing to 2D using {args.reducer.upper()}...")
    X2d = reduce_dimensions(
        X,
        method=args.reducer,
        random_state=args.random_state,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )

    print(f"[INFO] Clustering using {args.clusterer.upper()}...")
    labels = cluster_data(
        X2d,
        method=args.clusterer,
        random_state=args.random_state,
        n_clusters=args.n_clusters,
        hdb_min_cluster_size=args.hdb_min_cluster_size,
    )

    df_out = df.copy()
    df_out["emb_x"] = X2d[:, 0]
    df_out["emb_y"] = X2d[:, 1]
    df_out["cluster"] = labels

    csv_out = output_dir / "clustered.csv"
    img_out = output_dir / "embedding.png"

    print(f"[INFO] Writing results to {csv_out} and {img_out}")
    df_out.to_csv(csv_out, index=False)

    title = f"Reducer={args.reducer.upper()} | Clusterer={args.clusterer.upper()}"
    plot_embedding(df_out, img_out, title)

    # Generate per-cluster explanations based on TSH measured, TT4, and FTI by sex and binaryClass
    try:
        generate_cluster_explanations(df_out, output_dir)
    except Exception as e:
        print(f"[WARN] Failed to generate cluster explanations: {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
