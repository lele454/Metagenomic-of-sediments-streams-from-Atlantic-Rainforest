"""
#Script 02 - ARG Profiles in Preserved vs. Degraded Streams


#Description
-----------
  1. Stacked bar chart of raw ARG composition (RPKM) per stream
  2. Z-score heatmap of ARG classes ordered by Preservation Index (PI)
  3. Spearman correlation heatmap (ARG classes x limnological variables)
     with Benjamini-Hochberg FDR correction
  4. Quantile regression (tau = 0.75) for the strongest ARG-environment pairs
  5. GLM with quasipoisson distribution + stepwise backward AIC selection
     for each ARG class

ARG annotation was performed with DeepARG v2.0 (protein-long mode,
probability >= 0.80).  Abundances were normalized as RPKM (reads per
kilobase per million).

#Input files
-----------
  - data/ARG_RPKM.csv         : ARG class x samples matrix (RPKM values)
  - data/metadata.csv         : SampleID, Degradation, PI, HFP,
                                 + all limnological variables (ammonia,
                                 CDOM, DO, chlorophyll, turbidity, pH,
                                 conductivity, temperature, TDS)



#Dependencies
------------
  Python 3.12 | pandas 2.x | numpy 1.x | scipy 1.11+ | matplotlib 3.8
  seaborn 0.13 | statsmodels 0.14

#Usage
-----
  python 02_ARG_profiles.py


=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

# Limnological variables used as predictors in GLM and correlation analyses
LIMNO_VARS = [
    "Ammonia", "CDOM", "Dissolved_oxygen", "Chlorophyll",
    "Turbidity", "pH", "Conductivity", "Temperature", "TDS"
]

# ---------------------------------------------------------------------------
# 1.  Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ARG RPKM table and sample metadata.

    Returns
    -------
    arg_rpkm : pd.DataFrame  - ARG class x samples (RPKM)
    meta     : pd.DataFrame  - samples x variables
    """
    arg_rpkm = pd.read_csv(
        os.path.join(DATA_DIR, "ARG_RPKM.csv"), index_col=0
    )
    meta = pd.read_csv(
        os.path.join(DATA_DIR, "metadata.csv"), index_col="SampleID"
    )
    shared = arg_rpkm.columns.intersection(meta.index)
    return arg_rpkm[shared], meta.loc[shared]


# ---------------------------------------------------------------------------
# 2.  Preprocessing helpers
# ---------------------------------------------------------------------------
def zscore_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise Z-score: each ARG class is standardised across all samples."""
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def clr_transform(df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """
    Centred log-ratio transformation for compositional data.
    Adds pseudocount before log10 to handle zeros.
    """
    df_ps  = df + pseudocount
    log_df = np.log(df_ps)
    return log_df.sub(log_df.mean(axis=0), axis=1)   # centre per sample


# ---------------------------------------------------------------------------
# 3.  Figure 3a - Stacked bar chart (RPKM)
# ---------------------------------------------------------------------------
def plot_stacked_bars(arg_rpkm: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Stacked horizontal bar chart showing ARG class composition per stream,
    ordered by PI (most degraded -> most preserved).
    """
    # Order samples by PI
    order     = meta["PI"].sort_values().index
    rpkm_ord  = arg_rpkm[order]

    # Normalise to 100 % for display
    rpkm_pct = rpkm_ord.div(rpkm_ord.sum(axis=0), axis=1) * 100

    palette = sns.color_palette("tab20", n_colors=len(rpkm_pct.index))
    cmap    = dict(zip(rpkm_pct.index, palette))

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom  = np.zeros(rpkm_pct.shape[1])

    for cls in rpkm_pct.index:
        vals = rpkm_pct.loc[cls].values
        ax.bar(range(len(order)), vals, bottom=bottom,
               color=cmap[cls], label=cls, width=0.8)
        bottom += vals

    # Add degradation labels on x-axis ticks
    deg_map = meta.loc[order, "Degradation"].tolist()
    colours = [COL_DEG if d == "Degraded" else COL_PRES for d in deg_map]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    for tick, col in zip(ax.get_xticklabels(), colours):
        tick.set_color(col)

    ax.set_ylabel("ARG class composition (%)", fontsize=10)
    ax.set_title("Figure 3a - ARG class composition per stream", fontsize=11)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, frameon=False, title="ARG class", title_fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Degradation bar on top
    for i, (col, deg) in enumerate(zip(colours, deg_map)):
        ax.annotate("", xy=(i, 102), xytext=(i, 101),
                    arrowprops=dict(arrowstyle="-", color=col, lw=3))

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "fig3a_ARG_stacked_bars.tiff")
    plt.gcf().savefig(fname, dpi=DPI, bbox_inches='tight')  # PNG buffer
    _save_tiff_lzw(fname, DPI)
    print(f"  Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4.  Figure 3b - Z-score heatmap
# ---------------------------------------------------------------------------
def plot_zscore_heatmap(arg_rpkm: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Heatmap of Z-scores (per ARG class) with samples ordered by PI.
    No hierarchical clustering is applied.
    """
    order    = meta["PI"].sort_values().index
    zscores  = zscore_by_class(arg_rpkm[order])

    # Annotation bar showing Degradation status
    deg_colours = [COL_DEG if meta.loc[s, "Degradation"] == "Degraded"
                   else COL_PRES for s in order]

    fig, (ax_bar, ax_hm) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [0.04, 1], "hspace": 0.02}
    )

    # Degradation annotation bar
    for i, col in enumerate(deg_colours):
        ax_bar.add_patch(plt.Rectangle((i, 0), 1, 1, color=col))
    ax_bar.set_xlim(0, len(order))
    ax_bar.set_ylim(0, 1)
    ax_bar.axis("off")
    ax_bar.set_title("Figure 3b - ARG Z-score heatmap (ordered by PI)",
                      fontsize=11, pad=4)

    # Heatmap
    sns.heatmap(
        zscores,
        ax=ax_hm,
        cmap="RdBu_r",
        center=0,
        vmin=-2, vmax=2,
        linewidths=0.3,
        linecolor="white",
        xticklabels=order,
        yticklabels=zscores.index,
        cbar_kws={"label": "Z-score", "shrink": 0.6}
    )
    ax_hm.set_xticklabels(
        ax_hm.get_xticklabels(), rotation=45, ha="right", fontsize=8
    )
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), fontsize=8)
    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("ARG class", fontsize=10)

    # Colour x-tick labels by degradation
    for tick, col in zip(ax_hm.get_xticklabels(), deg_colours):
        tick.set_color(col)

    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "fig3b_ARG_zscore_heatmap.tiff")
    plt.gcf().savefig(fname, dpi=DPI, bbox_inches='tight')  # PNG buffer
    _save_tiff_lzw(fname, DPI)
    print(f"  Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5.  Figure 3c - Spearman correlation heatmap (ARG x limnological vars)
# ---------------------------------------------------------------------------
def spearman_correlations(arg_rpkm: pd.DataFrame,
                          meta:     pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman rho between log10(RPKM + 1) of each ARG class and each
    limnological variable. Applies Benjamini-Hochberg FDR correction.

    Returns
    -------
    df_rho : pd.DataFrame  - matrix of rho values (ARG x variable)
    df_fdr : pd.DataFrame  - matrix of FDR values
    """
    log_arg = np.log10(arg_rpkm + 1)
    vars_avail = [v for v in LIMNO_VARS if v in meta.columns]

    rho_data = {}
    p_data   = {}
    for var in vars_avail:
        rhos, ps = [], []
        for cls in log_arg.index:
            r, p = spearmanr(log_arg.loc[cls], meta[var])
            rhos.append(r)
            ps.append(p)
        rho_data[var] = rhos
        p_data[var]   = ps

    df_rho = pd.DataFrame(rho_data, index=log_arg.index)
    df_p   = pd.DataFrame(p_data,   index=log_arg.index)

    # FDR correction (all p-values flattened, then reshaped)
    p_flat = df_p.values.flatten()
    _, fdr_flat, _, _ = multipletests(p_flat, method="fdr_bh")
    df_fdr = pd.DataFrame(
        fdr_flat.reshape(df_p.shape), index=df_p.index, columns=df_p.columns
    )

    # Save table
    out = pd.concat([df_rho.add_suffix("_rho"), df_fdr.add_suffix("_FDR")],
                    axis=1)
    fname = os.path.join(TABLE_DIR, "ARG_spearman_correlations.csv")
    out.to_csv(fname)
    print(f"  Saved: {fname}")

    return df_rho, df_fdr


def plot_spearman_heatmap(df_rho: pd.DataFrame,
                          df_fdr: pd.DataFrame) -> None:
    """
    Clustered heatmap of Spearman rho with significance asterisks.
    """
    annot = df_rho.map(lambda x: f"{x:.2f}")
    # Add asterisk where FDR < 0.01
    mask_sig = df_fdr < 0.01
    for r in df_rho.index:
        for c in df_rho.columns:
            if mask_sig.loc[r, c]:
                annot.loc[r, c] += "*"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df_rho, ax=ax, annot=annot, fmt="",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Spearman rho", "shrink": 0.6}
    )
    ax.set_title(
        "Figure 3c - Spearman rho: ARG classes x environmental variables\n"
        "(* FDR < 0.01)",
        fontsize=11
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "fig3c_ARG_spearman_heatmap.tiff")
    plt.gcf().savefig(fname, dpi=DPI, bbox_inches='tight')  # PNG buffer
    _save_tiff_lzw(fname, DPI)
    print(f"  Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.  Quantile regression (tau = 0.75) for top ARG-variable pairs
# ---------------------------------------------------------------------------
def quantile_regression_top_pairs(arg_rpkm: pd.DataFrame,
                                  meta:     pd.DataFrame,
                                  df_rho:   pd.DataFrame,
                                  df_fdr:   pd.DataFrame,
                                  top_n:    int = 5) -> None:
    """
    Fit quantile regression (tau = 0.75) for the top_n ARG-variable pairs
    with the highest |rho| and FDR < 0.01.
    Prints slope, p-value (Wald test), and pseudo-R2 for each pair.
    """
    log_arg    = np.log10(arg_rpkm + 1)
    vars_avail = [v for v in LIMNO_VARS if v in meta.columns]

    # Identify top pairs
    pairs = []
    for cls in df_rho.index:
        for var in vars_avail:
            if var in df_rho.columns and df_fdr.loc[cls, var] < 0.01:
                pairs.append((abs(df_rho.loc[cls, var]), cls, var))
    pairs.sort(reverse=True)

    print(f"\n  Quantile regression (tau=0.75) - top {top_n} pairs:")
    for _, cls, var in pairs[:top_n]:
        y   = log_arg.loc[cls].values
        x   = meta[var].values
        df_qr = pd.DataFrame({"y": y, "x": x})
        try:
            model = smf.quantreg("y ~ x", df_qr).fit(q=0.75, max_iter=2000)
            print(f"    {cls} ~ {var}: slope={model.params['x']:.4f}, "
                  f"p={model.pvalues['x']:.3e}, "
                  f"pseudo-R2={model.prsquared:.3f}")
        except Exception as e:
            print(f"    {cls} ~ {var}: QR failed ({e})")


# ---------------------------------------------------------------------------
# 7.  GLM - quasipoisson per ARG class
# ---------------------------------------------------------------------------
def run_glm_all_classes(arg_rpkm: pd.DataFrame,
                        meta:     pd.DataFrame) -> pd.DataFrame:
    """
    Fit a GLM with quasipoisson family (log link) for each ARG class.
    Predictor selection: stepwise backward AIC.

    Returns
    -------
    df_results : pd.DataFrame with columns:
        ARG_class, selected_predictors, n_predictors,
        deviance_explained_pct, dispersion_pearson,
        AIC_final
    """
    vars_avail = [v for v in LIMNO_VARS if v in meta.columns]
    rows       = []

    for cls in arg_rpkm.index:
        y = arg_rpkm.loc[cls].values.astype(float)

        # Stepwise backward AIC
        current_predictors = vars_avail.copy()
        current_aic        = _fit_quasipoisson_aic(y, meta, current_predictors)

        improved = True
        while improved and len(current_predictors) > 1:
            improved = False
            best_aic  = current_aic
            best_drop = None
            for pred in current_predictors:
                candidate = [p for p in current_predictors if p != pred]
                try:
                    aic = _fit_quasipoisson_aic(y, meta, candidate)
                    if aic < best_aic:
                        best_aic  = aic
                        best_drop = pred
                except Exception:
                    pass
            if best_drop is not None:
                current_predictors.remove(best_drop)
                current_aic = best_aic
                improved    = True

        # Final model
        try:
            result, disp, dev_exp = _fit_quasipoisson_full(
                y, meta, current_predictors
            )
            rows.append({
                "ARG_class":            cls,
                "selected_predictors":  ", ".join(current_predictors),
                "n_predictors":         len(current_predictors),
                "deviance_explained_%": round(dev_exp * 100, 1),
                "Pearson_dispersion":   round(disp, 3),
                "AIC_final":            round(current_aic, 2)
            })
            print(f"  {cls:<30} DE={dev_exp*100:.1f}%  "
                  f"disp={disp:.2f}  preds={len(current_predictors)}")
        except Exception as e:
            print(f"  {cls}: GLM failed ({e})")

    df_results = pd.DataFrame(rows)
    fname      = os.path.join(TABLE_DIR, "ARG_GLM_results.csv")
    df_results.to_csv(fname, index=False)
    print(f"\n  Saved: {fname}")
    return df_results


def _fit_quasipoisson_aic(y:          np.ndarray,
                          meta:       pd.DataFrame,
                          predictors: list) -> float:
    """Fit quasi-Poisson GLM and return AIC (approximated as for Poisson)."""
    X = sm.add_constant(meta[predictors].values.astype(float))
    # Use Poisson family for AIC comparison; dispersion does not affect AIC rank
    model = sm.GLM(y, X, family=sm.families.Poisson(
        link=sm.families.links.Log())
    ).fit(disp=1)
    return model.aic


def _fit_quasipoisson_full(y:          np.ndarray,
                           meta:       pd.DataFrame,
                           predictors: list
                           ) -> tuple[object, float, float]:
    """
    Fit quasi-Poisson GLM and compute:
      - Pearson dispersion ratio
      - Deviance explained = 1 - (residual deviance / null deviance)
    """
    X = sm.add_constant(meta[predictors].values.astype(float))
    model = sm.GLM(
        y, X,
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit()

    null_model = sm.GLM(
        y, np.ones((len(y), 1)),
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit()

    # Pearson dispersion
    pearson_chi2 = ((y - model.fittedvalues)**2 / model.fittedvalues).sum()
    dispersion   = pearson_chi2 / model.df_resid

    dev_explained = 1 - model.deviance / null_model.deviance
    return model, dispersion, dev_explained


# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\nLoading data …")
    arg_rpkm, meta = load_data()
    print(f"  ARG classes : {arg_rpkm.shape[0]}")
    print(f"  Samples     : {arg_rpkm.shape[1]}")

    print("\n[3a] Stacked bar chart …")
    plot_stacked_bars(arg_rpkm, meta)

    print("\n[3b] Z-score heatmap …")
    plot_zscore_heatmap(arg_rpkm, meta)

    print("\n[3c] Spearman correlation heatmap …")
    df_rho, df_fdr = spearman_correlations(arg_rpkm, meta)
    plot_spearman_heatmap(df_rho, df_fdr)

    print("\n[QR] Quantile regression (tau=0.75) …")
    quantile_regression_top_pairs(arg_rpkm, meta, df_rho, df_fdr)

    print("\n[GLM] Quasi-Poisson GLMs …")
    run_glm_all_classes(arg_rpkm, meta)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
