#Script 03 – Functional Pathways Related to Biogeochemical Cycles

#Description
-----------
This script analyses the functional gene profiles associated with
biogeochemical cycles along the environmental degradation gradient.  It:

  1. Loads DRAM v1.5.1 gene abundance data (CPM-normalised)
  2. Applies the gene selection criteria (detection ≥ 12/14 metagenomes;
     mean CPM ≥ 4 in ≥ 1 conservation state; canonical DRAM-v/KEGG marker)
  3. Computes Spearman rank correlations between PI and each marker gene
  4. Fits Gaussian GLMs (CPM ~ β₀ + β₁ × PI) with Bonferroni correction
  5. Calculates three functional integrity indices (FII, AAR, GGI)
  6. Produces Figure 4 (coordinated biogeochemical cycle shifts) and
     Figure S6 (R² summary)

#Canonical marker genes analysed (18 total)
-------------------------------------------
  Nitrogen : nifH, amoA, nosZ, nirK, nirS, nxrA/B
  Carbon   : mcrA, pmoA (pmoA-A / pmoA-B), cbbL/rbcL
  Sulfur   : dsrA/B, soxB
  Phosphorus: phoD/phoX, pstS, ppK/ppx–gppA
  Metals   : merA, arsC
  Aromatic : aclA/B

#Functional indices
------------------
  FII = Σ(beneficial genes) / Σ(detrimental genes)
        where beneficial = nosZ, pmoA-B, soxB, cbbL/rbcL, nirK+nirS, ppx-gppA
        and   detrimental = mcrA, pmoA-A, norB, dsrA/B, amoA

  AAR = Σ(aerobic genes) / Σ(anaerobic genes)
        where aerobic    = nosZ, pmoA-B, soxB, cbbL/rbcL, nirK+nirS
        and   anaerobic  = mcrA, dsrA/B, pmoA-A, norB

  GGI = (28 × mcrA) + (265 × norB) − (28 × pmoA-B) − (265 × nosZ)
        (IPCC AR6 GWP100 coefficients: CH₄ = 28, N₂O = 265)

#Input files
-----------
  - data/DRAM_CPM.csv    : gene × samples matrix (CPM values, from DRAM-v)
  - data/metadata.csv    : SampleID, Degradation, PI, HFP


#Dependencies
------------
  Python 3.12 | pandas 2.x | numpy 1.x | scipy 1.11+ | matplotlib 3.8
  statsmodels 0.14

Usage
-----
  python 03_biogeochemical_cycles.py

# ---------------------------------------------------------------------------

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  Paths & global settings
# ---------------------------------------------------------------------------
DATA_DIR  = "data"
FIG_DIR   = "figures"
TABLE_DIR = "tables"

os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

COL_PRES = "#2E8B57"
COL_DEG  = "#CC3333"
DPI      = 300

# Bonferroni threshold (18 genes)
ALPHA_BONF = 0.05 / 18   # ≈ 0.00278

# ---------------------------------------------------------------------------
# 1.  Canonical gene lists
# ---------------------------------------------------------------------------
ALL_GENES = [
    "nifH", "amoA", "nosZ", "mcrA", "pmoA-B", "pmoA-A",
    "dsrA/B", "norB", "ppK/ppx-gppA", "soxB",
    "cbbL/rbcL", "phoD/phoX", "NiFe-hyd",
    "nxrA/B", "pstS", "merA", "arsC", "aclA/B"
]

CYCLE_MAP = {
    "Nitrogen" : ["nifH", "amoA", "nosZ", "nxrA/B"],
    "Carbon"   : ["mcrA", "pmoA-B", "pmoA-A", "cbbL/rbcL"],
    "Sulfur"   : ["dsrA/B", "soxB"],
    "Phosphorus": ["phoD/phoX", "pstS", "ppK/ppx-gppA"],
    "Metals"   : ["merA", "arsC"],
    "Other"    : ["nifH", "NiFe-hyd", "aclA/B", "norB"]
}

# Colour per biogeochemical cycle (for Figure 4)
CYCLE_COLS = {
    "Nitrogen"  : "#4472C4",
    "Carbon"    : "#70AD47",
    "Sulfur"    : "#FFC000",
    "Phosphorus": "#9E480E",
    "Metals"    : "#7030A0",
    "Other"     : "#808080"
}

# Index gene classification
BENEFICIAL  = ["nosZ", "pmoA-B", "soxB", "cbbL/rbcL", "nirK+nirS", "ppK/ppx-gppA"]
DETRIMENTAL = ["mcrA", "pmoA-A", "norB", "dsrA/B", "amoA"]
AEROBIC     = ["nosZ", "pmoA-B", "soxB", "cbbL/rbcL", "nirK+nirS"]
ANAEROBIC   = ["mcrA", "dsrA/B", "pmoA-A", "norB"]

GWP_CH4  = 28
GWP_N2O  = 265


# ---------------------------------------------------------------------------
# 2.  Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DRAM CPM table and sample metadata.

    Returns
    -------
    cpm  : pd.DataFrame  – gene × samples (CPM)
    meta : pd.DataFrame  – samples × variables (PI, HFP, Degradation, …)
    """
    cpm  = pd.read_csv(os.path.join(DATA_DIR, "DRAM_CPM.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"), index_col="SampleID")
    shared = cpm.columns.intersection(meta.index)
    return cpm[shared], meta.loc[shared]


# ---------------------------------------------------------------------------
# 3.  Gene filtering (detection & abundance criteria)
# ---------------------------------------------------------------------------
def filter_genes(cpm: pd.DataFrame, meta: pd.DataFrame,
                 min_samples: int = 12, min_cpm: float = 4.0) -> pd.DataFrame:
    """
    Retain only genes that meet ALL selection criteria:
      (i)  detected (CPM > 0) in ≥ min_samples metagenomes
      (ii) mean CPM ≥ min_cpm in at least one conservation state

    Parameters
    ----------
    cpm         : gene × samples CPM matrix
    meta        : sample metadata (must contain 'Degradation')
    min_samples : detection threshold (default 12 / 14)
    min_cpm     : minimum mean CPM in ≥ 1 group (default 4)

    Returns
    -------
    cpm_filtered : gene × samples, only genes passing all criteria
    """
    pres_cols = meta.index[meta["Degradation"] == "Preserved"].tolist()
    deg_cols  = meta.index[meta["Degradation"] == "Degraded"].tolist()

    detected = (cpm > 0).sum(axis=1) >= min_samples

    mean_pres = cpm[pres_cols].mean(axis=1)
    mean_deg  = cpm[deg_cols].mean(axis=1)
    abundant  = (mean_pres >= min_cpm) | (mean_deg >= min_cpm)

    # Also intersect with the canonical gene list
    canonical = cpm.index.isin(ALL_GENES)

    mask = detected & abundant & canonical
    cpm_filtered = cpm[mask]
    print(f"  Genes after filtering: {cpm_filtered.shape[0]} "
          f"(from {cpm.shape[0]} total)")
    return cpm_filtered


# ---------------------------------------------------------------------------
# 4.  Spearman correlations (PI × gene)
# ---------------------------------------------------------------------------
def spearman_pi(cpm: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman ρ and p-value between PI and each gene's CPM.

    Returns
    -------
    df_sp : pd.DataFrame – Gene, rho, p_value
    """
    pi = meta["PI"].values
    rows = []
    for gene in cpm.index:
        r, p = stats.spearmanr(cpm.loc[gene], pi)
        rows.append({"Gene": gene, "rho": r, "p_value": p})
    df_sp = pd.DataFrame(rows).sort_values("rho")
    return df_sp


# ---------------------------------------------------------------------------
# 5.  Gaussian GLM: CPM ~ β₀ + β₁ × PI  (per gene)
# ---------------------------------------------------------------------------
def glm_per_gene(cpm:  pd.DataFrame,
                 meta: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a Gaussian GLM (identity link = OLS) for each marker gene.
    Reports β₁, pseudo-R², p-value, and Bonferroni-corrected significance.

    Returns
    -------
    df_glm : pd.DataFrame
    """
    pi = meta["PI"].values
    X  = sm.add_constant(pi)
    rows = []

    for gene in cpm.index:
        y = cpm.loc[gene].values
        model  = sm.OLS(y, X).fit()
        beta0  = model.params[0]
        beta1  = model.params[1]
        r2     = model.rsquared
        p_val  = model.pvalues[1]
        sig    = p_val < ALPHA_BONF

        rows.append({
            "Gene":        gene,
            "beta0":       round(beta0, 4),
            "beta1 (PI)":  round(beta1, 4),
            "R2":          round(r2, 4),
            "p_value":     p_val,
            "Bonf_sig":    sig
        })
        print(f"  {gene:<20} β₁={beta1:+.2f}  R²={r2:.3f}  "
              f"p={'< 0.00278' if sig else p_val:.4f}  "
              f"{'*' if sig else ''}")

    df_glm = pd.DataFrame(rows)
    fname  = os.path.join(TABLE_DIR, "GLM_biogeochem_results.csv")
    df_glm.to_csv(fname, index=False)
    print(f"\n  Saved: {fname}")
    return df_glm


# ---------------------------------------------------------------------------
# 6.  Functional indices (FII, AAR, GGI)
# ---------------------------------------------------------------------------
def _get_gene(cpm: pd.DataFrame, gene: str) -> pd.Series:
    """Return gene CPM or zeros if gene not present."""
    if gene in cpm.index:
        return cpm.loc[gene]
    return pd.Series(np.zeros(cpm.shape[1]), index=cpm.columns)


def compute_indices(cpm: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate FII, AAR, and GGI for each sample.

    Returns
    -------
    df_idx : pd.DataFrame – SampleID × {FII, AAR, GGI}
    """
    eps = 1e-10   # avoid division by zero

    # FII
    beneficial  = sum(_get_gene(cpm, g) for g in BENEFICIAL)
    detrimental = sum(_get_gene(cpm, g) for g in DETRIMENTAL)
    FII = beneficial / (detrimental + eps)

    # AAR
    aerobic   = sum(_get_gene(cpm, g) for g in AEROBIC)
    anaerobic = sum(_get_gene(cpm, g) for g in ANAEROBIC)
    AAR = aerobic / (anaerobic + eps)

    # GGI  = (28 × mcrA) + (265 × norB) − (28 × pmoA-B) − (265 × nosZ)
    mcrA  = _get_gene(cpm, "mcrA")
    norB  = _get_gene(cpm, "norB")
    pmoAB = _get_gene(cpm, "pmoA-B")
    nosZ  = _get_gene(cpm, "nosZ")
    GGI   = GWP_CH4 * mcrA + GWP_N2O * norB - GWP_CH4 * pmoAB - GWP_N2O * nosZ

    df_idx = pd.DataFrame({"FII": FII, "AAR": AAR, "GGI": GGI})
    return df_idx


def regress_index(y: pd.Series, pi: pd.Series,
                  index_name: str) -> dict:
    """OLS regression for a functional index on PI."""
    X     = sm.add_constant(pi.values)
    model = sm.OLS(y.values, X).fit()
    beta0, beta1 = model.params
    r2           = model.rsquared
    p_val        = model.pvalues[1]

    # Equilibrium threshold  (FII=1, AAR=1, GGI=0)
    target = {"FII": 1.0, "AAR": 1.0, "GGI": 0.0}.get(index_name, 0.0)
    pi_eq  = (target - beta0) / beta1 if beta1 != 0 else np.nan

    print(f"  {index_name}: R²={r2:.3f}  β₁={beta1:+.4f}  "
          f"p={p_val:.2e}  PI_eq={pi_eq:.2f}")
    return {
        "Index":    index_name,
        "beta0":    round(beta0, 4),
        "beta1":    round(beta1, 4),
        "R2":       round(r2, 4),
        "p_value":  p_val,
        "PI_eq":    round(pi_eq, 2)
    }


# ---------------------------------------------------------------------------
# 7.  Figure 4 – Coordinated biogeochemical cycle shifts
# ---------------------------------------------------------------------------
def plot_figure4(cpm:    pd.DataFrame,
                 meta:   pd.DataFrame,
                 df_glm: pd.DataFrame) -> None:
    """
    One scatter + regression panel per marker gene, grouped by cycle,
    coloured by conservation state.  Points with Bonferroni-significant
    GLMs are highlighted with a filled marker.
    """
    pi     = meta["PI"].values
    is_deg = meta["Degradation"] == "Degraded"

    # Determine layout: ceil(n_genes / 4) rows × 4 columns
    genes  = cpm.index.tolist()
    n_cols = 4
    n_rows = int(np.ceil(len(genes) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.5, n_rows * 3.2))
    axes = axes.flatten()

    for ax, gene in zip(axes, genes):
        y   = cpm.loc[gene].values
        row = df_glm[df_glm["Gene"] == gene]

        # Scatter
        ax.scatter(pi[is_deg],  y[is_deg],  color=COL_DEG,
                   s=40, zorder=3, label="Degraded")
        ax.scatter(pi[~is_deg], y[~is_deg], color=COL_PRES,
                   s=40, zorder=3, label="Preserved")

        # Regression line + 95% CI
        pi_range = np.linspace(pi.min(), pi.max(), 100)
        if not row.empty:
            b0  = row["beta0"].values[0]
            b1  = row["beta1 (PI)"].values[0]
            y_hat = b0 + b1 * pi_range
            ax.plot(pi_range, y_hat, "k-", lw=1.2, zorder=2)

            # Annotate R² and β₁
            r2  = row["R2"].values[0]
            sig = row["Bonf_sig"].values[0]
            ax.set_title(
                f"$\\it{{{gene.replace('/', '_')}}}$\n"
                f"β₁={b1:+.2f}  R²={r2:.3f}{'*' if sig else ''}",
                fontsize=8
            )
        else:
            ax.set_title(f"$\\it{{{gene}}}$", fontsize=8)

        ax.set_xlabel("Preservation Index", fontsize=7)
        ax.set_ylabel("CPM", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused axes
    for ax in axes[len(genes):]:
        ax.set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_DEG,
               markersize=7, label="Degraded"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_PRES,
               markersize=7, label="Preserved"),
    ]
    fig.legend(handles=legend_elements, loc="lower right",
               fontsize=8, frameon=False, ncol=2)

    fig.suptitle(
        "Figure 4 – Coordinated shifts in biogeochemical marker genes\n"
        "(* Bonferroni-significant, α = 0.00278)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(FIG_DIR, "fig4_biogeochem_cycles.tiff")
    fig.savefig(fname, dpi=DPI, compression="tiff_lzw")
    print(f"  Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8.  Figure S6 – R² summary barplot
# ---------------------------------------------------------------------------
def plot_r2_summary(df_glm: pd.DataFrame) -> None:
    """Horizontal barplot of R² per gene, coloured by Bonferroni significance."""
    df_sorted = df_glm.sort_values("R2")
    colours   = [COL_DEG if s else "grey" for s in df_sorted["Bonf_sig"]]

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.barh(df_sorted["Gene"], df_sorted["R2"],
                   color=colours, edgecolor="white")
    ax.axvline(0.90, ls="--", lw=0.8, color="black", alpha=0.5,
               label="R² = 0.90")
    ax.set_xlabel("R² (GLM: CPM ~ PI)", fontsize=10)
    ax.set_title("Figure S6 – Predictability of biogeochemical genes by PI",
                 fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_yticklabels(
        [f"$\\it{{{g}}}$" for g in df_sorted["Gene"]], fontsize=7
    )
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate R² values
    for bar, val in zip(bars, df_sorted["R2"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=6)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "figS6_R2_summary.tiff")
    fig.savefig(fname, dpi=DPI, compression="tiff_lzw")
    print(f"  Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9.  Figure S7 – Functional indices vs PI
# ---------------------------------------------------------------------------
def plot_functional_indices(df_idx:   pd.DataFrame,
                            meta:     pd.DataFrame,
                            idx_regs: list) -> None:
    """
    Three-panel figure: FII, AAR, GGI vs PI with regression line and
    equilibrium threshold annotated.
    """
    pi     = meta["PI"].values
    is_deg = meta["Degradation"] == "Degraded"

    thresholds = {"FII": 1.0, "AAR": 1.0, "GGI": 0.0}
    ylabels    = {
        "FII": "Functional Integrity Index",
        "AAR": "Aerobiosis/Anaerobiosis Ratio",
        "GGI": "Greenhouse-gas Gene Enrichment Index"
    }
    ylogs = {"FII": True, "AAR": True, "GGI": False}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, reg in zip(axes, idx_regs):
        idx_name = reg["Index"]
        y        = df_idx[idx_name].values

        # Apply log for FII and AAR (positive-valued ratios)
        if ylogs[idx_name]:
            y_plot = np.log(y + 1e-10)
            ylabel = f"log({ylabels[idx_name]})"
        else:
            y_plot = y
            ylabel = ylabels[idx_name]

        ax.scatter(pi[is_deg],  y_plot[is_deg],  color=COL_DEG,
                   s=45, zorder=3, label="Degraded")
        ax.scatter(pi[~is_deg], y_plot[~is_deg], color=COL_PRES,
                   s=45, zorder=3, label="Preserved")

        # Regression line
        pi_range = np.linspace(pi.min(), pi.max(), 200)
        y_hat    = reg["beta0"] + reg["beta1"] * pi_range
        ax.plot(pi_range, y_hat, "k-", lw=1.5)

        # Equilibrium line
        thr = thresholds[idx_name]
        if ylogs[idx_name]:
            thr_plot = np.log(thr + 1e-10)
        else:
            thr_plot = thr
        ax.axhline(thr_plot, ls="--", lw=0.9, color="grey")

        # Annotate PI threshold
        pi_eq = reg["PI_eq"]
        ax.axvline(pi_eq, ls=":", lw=0.8, color="grey", alpha=0.7)
        ax.text(pi_eq + 0.05, ax.get_ylim()[0],
                f"PI={pi_eq:.2f}", fontsize=7, color="grey", rotation=90,
                va="bottom")

        ax.set_xlabel("Preservation Index", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            f"{idx_name}\nR²={reg['R2']:.3f}  β₁={reg['beta1']:+.4f}",
            fontsize=10
        )
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Figure S7 – Functional integrity indices along the PI gradient",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "figS7_functional_indices.tiff")
    fig.savefig(fname, dpi=DPI, compression="tiff_lzw")
    print(f"  Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10.  Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\nLoading data …")
    cpm, meta = load_data()
    print(f"  Raw genes  : {cpm.shape[0]}")
    print(f"  Samples    : {cpm.shape[1]}")

    print("\n[Filter] Applying gene selection criteria …")
    cpm_filt = filter_genes(cpm, meta)

    print("\n[Spearman] PI × gene correlations …")
    df_sp = spearman_pi(cpm_filt, meta)
    print(df_sp.to_string(index=False))

    print("\n[GLM] Gaussian GLMs (CPM ~ PI), Bonferroni correction …")
    df_glm = glm_per_gene(cpm_filt, meta)

    print("\n[Figure 4] Plotting biogeochemical cycle panels …")
    plot_figure4(cpm_filt, meta, df_glm)

    print("\n[Figure S6] R² summary barplot …")
    plot_r2_summary(df_glm)

    print("\n[Indices] Computing FII, AAR, GGI …")
    df_idx = compute_indices(cpm_filt)
    pi_ser = meta["PI"]

    idx_regs = []
    for idx_name in ("FII", "AAR", "GGI"):
        y = df_idx[idx_name]
        if idx_name in ("FII", "AAR"):
            y = np.log(y + 1e-10)
        reg = regress_index(pd.Series(y, index=meta.index), pi_ser, idx_name)
        idx_regs.append(reg)

    # Save indices
    df_idx["PI"]          = meta["PI"].values
    df_idx["Degradation"] = meta["Degradation"].values
    fname_idx = os.path.join(TABLE_DIR, "functional_indices.csv")
    df_idx.to_csv(fname_idx)
    print(f"\n  Saved: {fname_idx}")

    print("\n[Figure S7] Functional indices vs PI …")
    plot_functional_indices(df_idx, meta, idx_regs)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
