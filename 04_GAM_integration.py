#Script 04 – Integration Between Environmental Degradation and Functional
            Profiles (GAM analysis)

#Description
-----------

The hierarchical structure avoids statistical circularity:

  Layer 1 – Landscape pressure
    HFP  →  limnological variables
    HFP  →  biogeochemical genes
    HFP  →  ARG classes

  Layer 2 – Water chemistry mediation
    limnological variables  →  biogeochemical genes
    limnological variables  →  ARG classes

  Layer 3 – Gene–ARG association
    biogeochemical genes  →  ARG classes

GAMs are fitted with smoothing splines (pygam.LinearGAM).  All abundances
are log₁₀-transformed before modelling.  p-values are FDR-corrected
(Benjamini-Hochberg) within each layer.

The script also:
  - Identifies the most important direct predictors of each gene/ARG class
    (by pseudo-R² of single-predictor GAM)
  - Produces the network figure (Figure 5): force-directed graph where
    nodes = variables, genes, ARG classes; edges = significant GAM
    associations (FDR < 0.05), line width ∝ |β|

#Input files
-----------
  - data/metadata.csv          : SampleID, Degradation, PI, HFP,
                                   + limnological variables
  - data/DRAM_CPM.csv          : biogeochemical gene CPM matrix
                                   (filtered, same as Script 03)
  - data/ARG_RPKM.csv          : ARG class RPKM matrix



#Dependencies
------------
  Python 3.12 | pandas 2.x | numpy 1.x | scipy 1.11+ | matplotlib 3.8
  pygam 0.8.0 | networkx 3.x | statsmodels 0.14

Usage
-----
  python 04_GAM_integration.py

# ---------------------------------------------------------------------------


import os
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

try:
    from pygam import LinearGAM, s
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
    print("WARNING: pygam not installed. Install with: pip install pygam")
    print("Falling back to OLS for all GAM fits.")
    import statsmodels.api as sm

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    print("WARNING: networkx not installed. Install with: pip install networkx")
    print("Network figure (Fig 5) will be skipped.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  Paths & settings
# ---------------------------------------------------------------------------
DATA_DIR  = "data"
FIG_DIR   = "figures"
TABLE_DIR = "tables"

os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

COL_PRES = "#2E8B57"
COL_DEG  = "#CC3333"
DPI      = 300

FDR_THRESHOLD = 0.05



LIMNO_VARS = [
    "Ammonia", "CDOM", "Dissolved_oxygen", "Chlorophyll",
    "Turbidity", "pH", "Conductivity", "Temperature"
]

# ---------------------------------------------------------------------------
# 1.  Data loading
# ---------------------------------------------------------------------------
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load metadata, biogeochemical CPM, and ARG RPKM.

    Returns
    -------
    meta     : pd.DataFrame – samples × variables
    gene_cpm : pd.DataFrame – gene × samples (CPM)
    arg_rpkm : pd.DataFrame – ARG class × samples (RPKM)
    """
    meta     = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"),
                            index_col="SampleID")
    gene_cpm = pd.read_csv(os.path.join(DATA_DIR, "DRAM_CPM.csv"),
                            index_col=0)
    arg_rpkm = pd.read_csv(os.path.join(DATA_DIR, "ARG_RPKM.csv"),
                            index_col=0)

    shared = (meta.index
              .intersection(gene_cpm.columns)
              .intersection(arg_rpkm.columns))
    return meta.loc[shared], gene_cpm[shared], arg_rpkm[shared]


# ---------------------------------------------------------------------------
# 2.  GAM fitting helpers
# ---------------------------------------------------------------------------
def fit_gam_single(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit a single-predictor GAM (smoothing spline, n_splines=4 for n=14)
    and return pseudo-R², linear slope approximation (β), and p-value.

    Falls back to OLS if pygam is unavailable.
    """
    x = x.astype(float)
    y = y.astype(float)

    if PYGAM_AVAILABLE:
        try:
            gam = LinearGAM(s(0, n_splines=4)).fit(
                x.reshape(-1, 1), y
            )
            # Pseudo-R² = 1 - (sum_sq_resid / sum_sq_total)
            y_hat  = gam.predict(x.reshape(-1, 1))
            ss_res = ((y - y_hat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            p_val  = gam.statistics_["p_values"][0]

            # Approximate linear slope via first/last predicted values
            x_range = np.linspace(x.min(), x.max(), 100)
            y_range = gam.predict(x_range.reshape(-1, 1))
            beta    = np.polyfit(x_range, y_range, 1)[0]

            return {"R2": round(r2, 4), "beta": round(beta, 6),
                    "p_value": p_val, "method": "GAM"}
        except Exception:
            pass

    # OLS fallback
    import statsmodels.api as sm_local
    X      = sm_local.add_constant(x)
    model  = sm_local.OLS(y, X).fit()
    return {
        "R2":      round(model.rsquared, 4),
        "beta":    round(model.params[1], 6),
        "p_value": model.pvalues[1],
        "method":  "OLS"
    }


def run_gam_layer(predictors:  list,
                  responses:   list,
                  pred_data:   pd.DataFrame,
                  resp_data:   pd.DataFrame,
                  layer_name:  str) -> pd.DataFrame:
    """
    Run all single-predictor GAMs for one layer (predictor × response matrix).
    Applies FDR (Benjamini-Hochberg) correction within the layer.

    Parameters
    ----------
    predictors  : list of predictor variable/feature names
    responses   : list of response variable/feature names
    pred_data   : pd.DataFrame with predictor values (columns = predictors)
    resp_data   : pd.DataFrame with response values (rows = responses)
    layer_name  : label for this analysis layer

    Returns
    -------
    df_layer : pd.DataFrame with all Predictor × Response combinations
    """
    rows = []
    for pred, resp in itertools.product(predictors, responses):
        if pred not in pred_data.columns:
            continue
        if resp not in resp_data.index:
            continue

        x = pred_data[pred].values
        y = np.log10(resp_data.loc[resp].values + 1)

        res = fit_gam_single(x, y)
        rows.append({
            "Layer":     layer_name,
            "Predictor": pred,
            "Response":  resp,
            "R2":        res["R2"],
            "beta":      res["beta"],
            "p_value":   res["p_value"],
            "method":    res["method"]
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # FDR correction
    _, fdr, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["FDR"]         = fdr
    df["significant"] = fdr < FDR_THRESHOLD
    return df.sort_values("p_value")


# ---------------------------------------------------------------------------
# 3.  All layers
# ---------------------------------------------------------------------------
def run_all_layers(meta:     pd.DataFrame,
                   gene_cpm: pd.DataFrame,
                   arg_rpkm: pd.DataFrame) -> dict:
    """
    Run GAM analysis for all three layers and return a dict of DataFrames.
    """
    limno_avail   = [v for v in LIMNO_VARS if v in meta.columns]
    genes         = gene_cpm.index.tolist()
    arg_classes   = arg_rpkm.index.tolist()

    # Build predictor frames
    limno_frame = meta[limno_avail].copy()
    hfp_frame   = meta[["HFP"]].copy()

    # Gene CPM log₁₀-transformed (responses treated as predictor in layer 3)
    gene_frame  = np.log10(gene_cpm + 1).T   # samples × genes
    gene_frame.columns = gene_cpm.index

    results = {}

    # --- Layer 1: HFP → limnological variables -----------------------------
    limno_resp = meta[limno_avail].T   # variables as rows, samples as cols
    print("\n[Layer 1a] HFP → limnological variables …")
    results["HFP_to_limno"] = run_gam_layer(
        ["HFP"], limno_avail,
        hfp_frame, limno_resp, "HFP→Limno"
    )

    # --- Layer 1b: HFP → biogeochemical genes ------------------------------
    print("[Layer 1b] HFP → biogeochemical genes …")
    results["HFP_to_genes"] = run_gam_layer(
        ["HFP"], genes,
        hfp_frame, gene_cpm, "HFP→Gene"
    )

    # --- Layer 1c: HFP → ARG classes ---------------------------------------
    print("[Layer 1c] HFP → ARG classes …")
    results["HFP_to_ARGs"] = run_gam_layer(
        ["HFP"], arg_classes,
        hfp_frame, arg_rpkm, "HFP→ARG"
    )

    # --- Layer 2a: limnological → genes ------------------------------------
    print("[Layer 2a] Limnological → biogeochemical genes …")
    results["Limno_to_genes"] = run_gam_layer(
        limno_avail, genes,
        limno_frame, gene_cpm, "Limno→Gene"
    )

    # --- Layer 2b: limnological → ARGs -------------------------------------
    print("[Layer 2b] Limnological → ARG classes …")
    results["Limno_to_ARGs"] = run_gam_layer(
        limno_avail, arg_classes,
        limno_frame, arg_rpkm, "Limno→ARG"
    )

    # --- Layer 3: genes → ARGs ---------------------------------------------
    print("[Layer 3]  Biogeochemical genes → ARG classes …")
    results["Genes_to_ARGs"] = run_gam_layer(
        genes, arg_classes,
        gene_frame, arg_rpkm, "Gene→ARG"
    )

    return results


# ---------------------------------------------------------------------------
# 4.  Save all tables
# ---------------------------------------------------------------------------
def save_tables(results: dict) -> pd.DataFrame:
    """Save per-layer CSVs and one combined summary table."""
    fname_map = {
        "HFP_to_limno"  : "GAM_HFP_to_limno.csv",
        "HFP_to_genes"  : "GAM_HFP_to_genes.csv",
        "HFP_to_ARGs"   : "GAM_HFP_to_ARGs.csv",
        "Limno_to_genes": "GAM_limno_to_genes.csv",
        "Limno_to_ARGs" : "GAM_limno_to_ARGs.csv",
        "Genes_to_ARGs" : "GAM_genes_to_ARGs.csv",
    }
    all_dfs = []
    for key, fname in fname_map.items():
        if key in results and not results[key].empty:
            path = os.path.join(TABLE_DIR, fname)
            results[key].to_csv(path, index=False)
            print(f"  Saved: {path}")
            all_dfs.append(results[key])

    df_all = pd.concat(all_dfs, ignore_index=True)
    fname_all = os.path.join(TABLE_DIR, "GAM_summary_all.csv")
    df_all.to_csv(fname_all, index=False)
    print(f"  Saved: {fname_all}")
    return df_all


# ---------------------------------------------------------------------------
# 5.  Print summary statistics (matching paper text)
# ---------------------------------------------------------------------------
def print_summary(results: dict) -> None:
    """Print the counts matching the paper's Results text."""
    print("\n" + "="*60)
    print("  GAM analysis summary")
    print("="*60)

    for label, key in [
        ("HFP → limno",   "HFP_to_limno"),
        ("HFP → genes",   "HFP_to_genes"),
        ("HFP → ARGs",    "HFP_to_ARGs"),
        ("Limno → genes", "Limno_to_genes"),
        ("Limno → ARGs",  "Limno_to_ARGs"),
        ("Genes → ARGs",  "Genes_to_ARGs"),
    ]:
        if key not in results or results[key].empty:
            continue
        df  = results[key]
        n   = len(df)
        sig = df["significant"].sum()
        print(f"  {label:<22} {sig:>3} / {n:>3} significant (FDR<0.05)")

    print()


# ---------------------------------------------------------------------------
# 6.  Figure 5 – Network figure
# ---------------------------------------------------------------------------
def plot_network(results: dict,
                 limno_avail: list,
                 genes:       list,
                 arg_classes: list) -> None:
    """
    Force-directed network graph.

    Node types:
      • HFP (single landscape node)
      • Limnological variables
      • Biogeochemical genes
      • ARG classes

    Edges: significant GAM associations (FDR < 0.05).
    Edge colour: blue (negative β), red (positive β).
    Edge width: scaled by |β|.
    """
    if not NX_AVAILABLE:
        print("  Skipping network figure (networkx not available).")
        return

    G    = nx.DiGraph()
    edge_colours = []
    edge_widths  = []

    def add_edges(df_layer: pd.DataFrame, node_type_src: str,
                  node_type_tgt: str) -> None:
        for _, row in df_layer[df_layer["significant"]].iterrows():
            src = row["Predictor"]
            tgt = row["Response"]
            G.add_node(src, node_type=node_type_src)
            G.add_node(tgt, node_type=node_type_tgt)
            G.add_edge(src, tgt,
                       beta=row["beta"], R2=row["R2"], FDR=row["FDR"])

    type_map = {
        "HFP_to_limno"  : ("HFP",            "Limnological"),
        "HFP_to_genes"  : ("HFP",            "Biogeochemical"),
        "HFP_to_ARGs"   : ("HFP",            "ARG"),
        "Limno_to_genes": ("Limnological",    "Biogeochemical"),
        "Limno_to_ARGs" : ("Limnological",    "ARG"),
        "Genes_to_ARGs" : ("Biogeochemical",  "ARG"),
    }

    for key, (src_type, tgt_type) in type_map.items():
        if key in results and not results[key].empty:
            add_edges(results[key], src_type, tgt_type)

    if G.number_of_nodes() == 0:
        print("  Network is empty – no significant associations found.")
        return

    # Collect edge attributes in order
    edges      = list(G.edges(data=True))
    e_colours  = []
    e_widths   = []
    beta_vals  = np.array([d["beta"] for _, _, d in edges])
    max_beta   = np.abs(beta_vals).max() if len(beta_vals) > 0 else 1.0

    for _, _, d in edges:
        e_colours.append("#2980B9" if d["beta"] < 0 else "#E74C3C")
        e_widths.append( max(0.5, 3.0 * abs(d["beta"]) / max_beta) )

    # Node colours by type
    node_colours = [
        NODE_COLS.get(G.nodes[n].get("node_type", "ARG"), "grey")
        for n in G.nodes()
    ]

    # Layout – fixed seed for reproducibility
    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=200)

    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colours, node_size=700, alpha=0.9
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=7,
        labels={n: n for n in G.nodes()}
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=e_colours,
        width=e_widths,
        alpha=0.8,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        connectionstyle="arc3,rad=0.1"
    )

    # Legend – node types
    legend_patches = [
        mpatches.Patch(facecolor=col, label=label)
        for label, col in NODE_COLS.items()
    ]
    legend_patches += [
        plt.Line2D([0], [0], color="#E74C3C", lw=2, label="Positive β"),
        plt.Line2D([0], [0], color="#2980B9", lw=2, label="Negative β"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
              frameon=False, title="Node/Edge type", title_fontsize=9)

    ax.set_title(
        "Figure 5 – Integrated GAM network: HFP, limnological variables,\n"
        "biogeochemical genes, and ARG classes (FDR < 0.05)",
        fontsize=11, fontweight="bold"
    )
    ax.axis("off")
    fig.tight_layout()

    fname = os.path.join(FIG_DIR, "fig5_GAM_network.tiff")
    fig.savefig(fname, dpi=DPI, compression="tiff_lzw")
    print(f"  Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\nLoading data …")
    meta, gene_cpm, arg_rpkm = load_all()
    print(f"  Samples        : {len(meta)}")
    print(f"  Biogeoch. genes: {gene_cpm.shape[0]}")
    print(f"  ARG classes    : {arg_rpkm.shape[0]}")

    limno_avail = [v for v in LIMNO_VARS if v in meta.columns]
    genes       = gene_cpm.index.tolist()
    arg_classes = arg_rpkm.index.tolist()

    # Pre-transform HFP for GAM predictor frame
    meta_gam = meta.copy()
    meta_gam["HFP"] = meta_gam["HFP"].astype(float)

    print("\nRunning GAM layers …")
    results = run_all_layers(meta_gam, gene_cpm, arg_rpkm)

    print("\nSaving tables …")
    df_all = save_tables(results)

    print_summary(results)

    print("[Figure 5] Plotting GAM network …")
    plot_network(results, limno_avail, genes, arg_classes)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
