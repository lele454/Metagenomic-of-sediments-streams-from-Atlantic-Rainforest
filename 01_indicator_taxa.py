#Script 01 – Assessment of Indicator Taxa of Environmental Degradation

#Description
-----------

  1. Differential abundance analysis (Mann-Whitney U + FDR correction)
  2. Pearson correlation with the continuous Preservation Index (PI)
  3. Random Forest Classifier  (binary: Degraded vs. Preserved)
  4. Random Forest Regressor   (continuous: PI as response variable)


#Microbial groups
----------------
  bacteria : 7205 TaxIDs; Kraken2/Bracken taxonomy; relative abundance
  fungi    :   82 TaxIDs; Kraken2/Bracken taxonomy; relative abundance
  protists :  347 TaxIDs; Kaiju (nr_euk) taxonomy;  relative abundance


#Input files
-----------
  - bacteria_abundance.csv   : Relative abundance table (genera × samples)
  - fungi_abundance.csv      : Relative abundance table (genera × samples)
  - protist_abundance.csv    : Relative abundance table (genera × samples)
  - metadata.csv             : Per-sample metadata including PI and
                               degradation status (Preserved / Degraded)
    rows = genera (TaxID labels)
    columns = sample IDs (Pres01–Pres07, Deg01–Deg07)

    SampleID, Degradation (Preserved/Degraded), PI (numeric),
    HFP (numeric), [+ any limnological variables]


#Dependencies
------------
  Python 3.12 | pandas 2.x | numpy 1.x | scipy 1.11+ | scikit-learn 1.4+
  matplotlib 3.8 | statsmodels 0.14

#Usage
-----
  python 01_indicator_taxa.py


# ---------------------------------------------------------------------------

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, r2_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  Paths & global settings
# ---------------------------------------------------------------------------
DATA_DIR   = "data"
FIG_DIR    = "figures"
TABLE_DIR  = "tables"

os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

COL_PRES = "#2E8B57"   # SeaGreen  – Preserved
COL_DEG  = "#CC3333"   # Crimson   – Degraded

DPI = 300

# Random Forest hyper-parameters
RF_PARAMS = dict(n_estimators=1000, max_depth=4, random_state=42,
                 n_jobs=-1)

# ---------------------------------------------------------------------------
# 1.  Data loading
# ---------------------------------------------------------------------------
def load_data(group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load abundance table and metadata for a given microbial group.

    Parameters
    ----------
    group : str
        One of 'bacteria', 'fungi', or 'protist'.

    Returns
    -------
    abund : pd.DataFrame
        Genera (rows) × samples (columns), relative abundances.
    meta  : pd.DataFrame
        Samples (rows) with columns SampleID, Degradation, PI, HFP.
    """
    abund = pd.read_csv(
        os.path.join(DATA_DIR, f"{group}_abundance.csv"), index_col=0
    )
    meta  = pd.read_csv(
        os.path.join(DATA_DIR, "metadata.csv"), index_col="SampleID"
    )
    # Align sample order
    shared = abund.columns.intersection(meta.index)
    abund  = abund[shared]
    meta   = meta.loc[shared]
    return abund, meta


# ---------------------------------------------------------------------------
# 2.  Alpha-diversity (Shannon H') comparison
# ---------------------------------------------------------------------------
def shannon_index(abund: pd.DataFrame) -> pd.Series:
    """
    Compute Shannon H' for each sample.

    H' = -Σ (p_i × log(p_i))  where p_i are relative abundances > 0.
    """
    def _h(col):
        p = col[col > 0]
        return -np.sum(p * np.log(p))
    return abund.apply(_h)


def compare_alpha_diversity(abund: pd.DataFrame,
                             meta:  pd.DataFrame,
                             group: str) -> dict:
    """
    Compare Shannon H' between Preserved and Degraded streams.
    Uses Student's t-test when normality (Shapiro-Wilk) and
    homoscedasticity (Levene) are met; otherwise Mann-Whitney U.
    Effect size: Cohen's d.

    Returns
    -------
    result : dict with keys H_pres, H_deg, test, statistic, p_value, cohens_d
    """
    from scipy.stats import shapiro, levene, ttest_ind

    H = shannon_index(abund)
    pres_vals = H[meta.index[meta["Degradation"] == "Preserved"]].values
    deg_vals  = H[meta.index[meta["Degradation"] == "Degraded"]].values

    # Normality
    _, p_norm_pres = shapiro(pres_vals)
    _, p_norm_deg  = shapiro(deg_vals)
    normal = (p_norm_pres > 0.05) and (p_norm_deg > 0.05)

    # Homoscedasticity
    _, p_levene = levene(pres_vals, deg_vals)
    homoscedastic = p_levene > 0.05

    if normal and homoscedastic:
        stat, p_val = ttest_ind(pres_vals, deg_vals)
        test = "t-test"
    else:
        from scipy.stats import mannwhitneyu as mwu
        stat, p_val = mwu(pres_vals, deg_vals, alternative="two-sided")
        test = "Mann-Whitney U"

    # Cohen's d
    pooled_sd = np.sqrt(
        ((len(pres_vals) - 1) * pres_vals.std(ddof=1)**2 +
         (len(deg_vals)  - 1) * deg_vals.std(ddof=1)**2) /
        (len(pres_vals) + len(deg_vals) - 2)
    )
    cohens_d = abs(pres_vals.mean() - deg_vals.mean()) / (pooled_sd + 1e-10)

    result = {
        "Group":    group,
        "H_pres":   f"{pres_vals.mean():.3f} ± {pres_vals.std(ddof=1):.3f}",
        "H_deg":    f"{deg_vals.mean():.3f} ± {deg_vals.std(ddof=1):.3f}",
        "test":     test,
        "statistic": round(stat, 3),
        "p_value":  round(p_val, 3),
        "cohens_d": round(cohens_d, 3)
    }
    print(f"  Alpha-diversity ({group}): H'_pres={result['H_pres']}  "
          f"H'_deg={result['H_deg']}  {test} stat={stat:.3f}  "
          f"p={p_val:.3f}  d={cohens_d:.3f}")
    return result


def plot_alpha_diversity(all_results: list,
                          abund_dict:  dict,
                          meta:        pd.DataFrame) -> None:
    """
    Boxplot of Shannon H' per microbial group and conservation state,
    with significance annotation.
    """
    groups = list(abund_dict.keys())
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 5),
                              sharey=False)
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        H     = shannon_index(abund_dict[group])
        pres  = H[meta.index[meta["Degradation"] == "Preserved"]].values
        deg   = H[meta.index[meta["Degradation"] == "Degraded"]].values

        bp = ax.boxplot(
            [pres, deg],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", lw=2)
        )
        bp["boxes"][0].set_facecolor(COL_PRES)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(COL_DEG)
        bp["boxes"][1].set_alpha(0.7)

        # Overlay individual points
        for vals, x, col in [(pres, 1, COL_PRES), (deg, 2, COL_DEG)]:
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(x + jitter, vals, color=col, s=25, zorder=3, alpha=0.85)

        # Significance bracket
        res = next(r for r in all_results if r["Group"] == group)
        p   = res["p_value"]
        sig_label = ("n.s." if p > 0.05
                     else "*" if p > 0.01
                     else "**" if p > 0.001
                     else "***")
        y_max = max(np.concatenate([pres, deg])) * 1.08
        ax.plot([1, 1, 2, 2], [y_max * 0.97, y_max, y_max, y_max * 0.97],
                lw=1, color="black")
        ax.text(1.5, y_max * 1.01, sig_label, ha="center", fontsize=10)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Preserved", "Degraded"], fontsize=9)
        ax.set_ylabel("Shannon H'", fontsize=10)
        ax.set_title(group.capitalize(), fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Alpha-diversity (Shannon H') by conservation state",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "alpha_diversity_comparison.tiff")
    fig.savefig(fname, dpi=DPI, compression="tiff_lzw")
    print(f"  Saved: {fname}")
    plt.close(fig)



# ---------------------------------------------------------------------------
# 3.  Differential abundance (Mann-Whitney U + FDR)
# ---------------------------------------------------------------------------
def differential_abundance(abund: pd.DataFrame,
                            meta:  pd.DataFrame) -> pd.DataFrame:
    """
    Identify differentially abundant genera between Preserved and Degraded
    using a two-tailed Mann-Whitney U test with Benjamini-Hochberg FDR
    correction.

    Returns
    -------
    df_da : pd.DataFrame
        Genus, log2FC, U-statistic, p-value, FDR, significant (bool).
    """
    pres_cols = meta.index[meta["Degradation"] == "Preserved"].tolist()
    deg_cols  = meta.index[meta["Degradation"] == "Degraded"].tolist()

    rows = []
    for genus in abund.index:
        x_pres = abund.loc[genus, pres_cols].values + 1e-10
        x_deg  = abund.loc[genus, deg_cols].values  + 1e-10
        u_stat, p_val = mannwhitneyu(x_deg, x_pres, alternative="two-sided")
        log2fc = np.log2(x_deg.mean() / x_pres.mean())
        rows.append({"Genus": genus, "log2FC": log2fc,
                     "U": u_stat, "p_value": p_val})

    df = pd.DataFrame(rows)
    _, fdr, _, _ = multipletests(df["p_value"], method="fdr_bh")
    df["FDR"]         = fdr
    df["significant"] = (df["FDR"] < 0.05) & (df["log2FC"].abs() > 1)
    return df.sort_values("FDR")


# ---------------------------------------------------------------------------
# 3.  Pearson correlation with Preservation Index
# ---------------------------------------------------------------------------
def pearson_with_pi(abund: pd.DataFrame,
                    meta:  pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between log₁₀(abundance + 1) of each genus
    and the continuous Preservation Index (PI).

    Returns
    -------
    df_cor : pd.DataFrame
        Genus, r, p_value, significant (bool).
    """
    pi_values = meta["PI"].values
    rows = []
    for genus in abund.index:
        x = np.log10(abund.loc[genus].values + 1)
        r, p = stats.pearsonr(x, pi_values)
        rows.append({"Genus": genus, "r": r, "p_value": p})

    df = pd.DataFrame(rows)
    df["significant"] = df["p_value"] < 0.05
    return df.sort_values("r", ascending=False)


# ---------------------------------------------------------------------------
# 4.  Random Forest – binary classification (LOO-CV)
# ---------------------------------------------------------------------------
def rf_classifier(abund: pd.DataFrame,
                  meta:  pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Train a Random Forest Classifier (Degraded vs. Preserved) with
    Leave-One-Out Cross-Validation.

    Returns
    -------
    importance_df : pd.DataFrame  – genera ranked by Mean Decrease in Impurity
    loo_accuracy  : float         – LOO-CV overall accuracy
    """
    X = abund.T.values            # samples × genera
    y = (meta["Degradation"] == "Degraded").astype(int).values

    loo    = LeaveOneOut()
    y_pred = np.empty_like(y)

    # Fit a fresh RF for each LOO fold
    for train_idx, test_idx in loo.split(X):
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = rf.predict(X[test_idx])

    accuracy = accuracy_score(y, y_pred)

    # Full-data RF for feature importance
    rf_full = RandomForestClassifier(**RF_PARAMS)
    rf_full.fit(X, y)
    importance_df = pd.DataFrame({
        "Genus":      abund.index,
        "RF_Class_Imp": rf_full.feature_importances_
    }).sort_values("RF_Class_Imp", ascending=False)

    return importance_df, accuracy


# ---------------------------------------------------------------------------
# 5.  Random Forest – continuous regression (LOO-CV)
# ---------------------------------------------------------------------------
def rf_regressor(abund: pd.DataFrame,
                 meta:  pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Train a Random Forest Regressor with PI as the continuous response and
    Leave-One-Out Cross-Validation.

    Returns
    -------
    importance_df : pd.DataFrame  – genera ranked by Mean Decrease in Impurity
    loo_r2        : float         – LOO-CV R²
    loo_p         : float         – one-tailed p-value (permutation, 999 perm)
    """
    X      = abund.T.values
    y      = meta["PI"].values
    loo    = LeaveOneOut()
    y_pred = np.empty_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = rf.predict(X[test_idx])

    loo_r2 = r2_score(y, y_pred)

    # Permutation test for significance of LOO-R²
    n_perm = 999
    r2_null = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        y_shuf = rng.permutation(y)
        pred_shuf = np.empty_like(y_shuf, dtype=float)
        for tr, te in loo.split(X):
            rf_p = RandomForestRegressor(**RF_PARAMS)
            rf_p.fit(X[tr], y_shuf[tr])
            pred_shuf[te] = rf_p.predict(X[te])
        r2_null.append(r2_score(y_shuf, pred_shuf))

    loo_p = (np.sum(np.array(r2_null) >= loo_r2) + 1) / (n_perm + 1)

    # Full-data RF for importance
    rf_full = RandomForestRegressor(**RF_PARAMS)
    rf_full.fit(X, y)
    importance_df = pd.DataFrame({
        "Genus":      abund.index,
        "RF_Reg_Imp": rf_full.feature_importances_
    }).sort_values("RF_Reg_Imp", ascending=False)

    return importance_df, loo_r2, loo_p


# ---------------------------------------------------------------------------
# 6.  Volcano plot
# ---------------------------------------------------------------------------
def plot_volcano(df_da: pd.DataFrame, group: str, ax=None) -> None:
    """
    Volcano plot: log₂FC (x) vs −log₁₀(FDR) (y).
    Significant genera (FDR < 0.05, |log2FC| > 1) are coloured.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    non_sig = df_da[~df_da["significant"]]
    sig_up  = df_da[ df_da["significant"] & (df_da["log2FC"] > 0)]  # enriched in Degraded
    sig_dn  = df_da[ df_da["significant"] & (df_da["log2FC"] < 0)]  # enriched in Preserved

    ax.scatter(non_sig["log2FC"], -np.log10(non_sig["FDR"]),
               color="grey", alpha=0.5, s=18, label="Not significant")
    ax.scatter(sig_up["log2FC"], -np.log10(sig_up["FDR"]),
               color=COL_DEG, s=40, zorder=3, label="Enriched in Degraded")
    ax.scatter(sig_dn["log2FC"], -np.log10(sig_dn["FDR"]),
               color=COL_PRES, s=40, zorder=3, label="Enriched in Preserved")

    # Label top 10 significant
    for _, row in df_da[df_da["significant"]].head(10).iterrows():
        ax.annotate(row["Genus"],
                    xy=(row["log2FC"], -np.log10(row["FDR"])),
                    xytext=(3, 2), textcoords="offset points",
                    fontsize=6, fontstyle="italic")

    # Threshold lines
    ax.axhline(-np.log10(0.05), ls="--", lw=0.8, color="black", alpha=0.6)
    ax.axvline(-1, ls=":",       lw=0.8, color="black", alpha=0.6)
    ax.axvline( 1, ls=":",       lw=0.8, color="black", alpha=0.6)

    ax.set_xlabel("log₂ Fold Change (Degraded / Preserved)", fontsize=9)
    ax.set_ylabel("−log₁₀(FDR)", fontsize=9)
    ax.set_title(f"Differential abundance – {group.capitalize()}", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# 7.  RF importance barplot
# ---------------------------------------------------------------------------
def plot_rf_importance(imp_class: pd.DataFrame,
                       imp_reg:   pd.DataFrame,
                       group:     str,
                       top_n:     int = 20) -> None:
    """
    Side-by-side horizontal barplots of RF Classifier and RF Regressor
    feature importance for the top_n genera.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, df, label, col in zip(
        axes,
        [imp_class.head(top_n), imp_reg.head(top_n)],
        ["RF Classifier (Importance)", "RF Regressor (Importance)"],
        ["#4C72B0", "#DD8452"]
    ):
        imp_col = [c for c in df.columns if "Imp" in c][0]
        ax.barh(df["Genus"][::-1], df[imp_col][::-1], color=col, edgecolor="white")
        ax.set_xlabel("Mean Decrease in Impurity", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        # Italicise genus labels
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(
            [f"$\\it{{{g}}}$" for g in df["Genus"][::-1]], fontsize=7
        )
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Random Forest importance – {group.capitalize()}",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, f"rf_importance_{group}.tiff")
    fig.savefig(fname, dpi=DPI, compression="tiff_lzw")
    print(f"  Saved: {fname}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8.  Combined indicator table
# ---------------------------------------------------------------------------
def build_indicator_table(df_da:    pd.DataFrame,
                          df_cor:   pd.DataFrame,
                          imp_class: pd.DataFrame,
                          imp_reg:   pd.DataFrame) -> pd.DataFrame:
    """
    Merge all four analytical approaches into a single ranked table.
    Combined score = RF_Reg_Imp × |Pearson r|.
    """
    df = (df_da[["Genus", "log2FC", "FDR", "significant"]]
          .merge(df_cor[["Genus", "r", "p_value"]],
                 on="Genus", suffixes=("_da", "_cor"))
          .merge(imp_class[["Genus", "RF_Class_Imp"]], on="Genus", how="left")
          .merge(imp_reg[["Genus", "RF_Reg_Imp"]],    on="Genus", how="left"))

    df["Combined_score"] = df["RF_Reg_Imp"].fillna(0) * df["r"].abs()
    df.sort_values("Combined_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# 9.  Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Load all three groups first (needed for joint alpha-diversity figure)
    groups     = ("bacteria", "fungi", "protist")
    abund_dict = {}
    meta_ref   = None

    for group in groups:
        abund, meta = load_data(group)
        abund_dict[group] = abund
        if meta_ref is None:
            meta_ref = meta   # metadata is the same for all groups

    # ------------------------------------------------------------------
    # Alpha-diversity comparison (all three groups together)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Alpha-diversity (Shannon H') – all groups")
    print("="*60)
    alpha_results = []
    for group in groups:
        res = compare_alpha_diversity(abund_dict[group], meta_ref, group)
        alpha_results.append(res)

    # Save alpha-diversity table
    df_alpha = pd.DataFrame(alpha_results)
    fname_alpha = os.path.join(TABLE_DIR, "alpha_diversity_summary.csv")
    df_alpha.to_csv(fname_alpha, index=False)
    print(f"  Saved: {fname_alpha}")

    # Joint boxplot
    plot_alpha_diversity(alpha_results, abund_dict, meta_ref)

    # ------------------------------------------------------------------
    # Per-group indicator analysis
    # ------------------------------------------------------------------
    for group in groups:
        print(f"\n{'='*60}")
        print(f"  Processing indicator taxa: {group.upper()}")
        print(f"{'='*60}")

        abund = abund_dict[group]
        meta  = meta_ref
        print(f"  Genera loaded : {abund.shape[0]}")
        print(f"  Samples loaded: {abund.shape[1]}")

        # Note for fungi: low separation expected (PERMANOVA p=0.450)
        if group == "fungi":
            print("  [Note] Fungal community showed no significant PERMANOVA "
                  "separation (p=0.450, R²=0.075). Results reported for "
                  "completeness.")

        # -- Differential abundance ------------------------------------------
        print("  Running differential abundance analysis …")
        df_da = differential_abundance(abund, meta)
        n_sig = df_da["significant"].sum()
        print(f"  Significant genera (FDR<0.05, |log2FC|>1): {n_sig}")

        # -- Pearson correlation ---------------------------------------------
        print("  Computing Pearson correlations with PI …")
        df_cor = pearson_with_pi(abund, meta)

        # -- RF Classifier ---------------------------------------------------
        print("  Training RF Classifier (LOO-CV) …")
        imp_class, acc = rf_classifier(abund, meta)
        print(f"  LOO-CV accuracy: {acc*100:.1f}%")

        # -- RF Regressor ----------------------------------------------------
        print("  Training RF Regressor (LOO-CV) …")
        imp_reg, r2, p_val = rf_regressor(abund, meta)
        print(f"  LOO-CV R²={r2:.3f}, p={p_val:.3f}")

        # -- Volcano plot ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_volcano(df_da, group, ax=ax)
        fig.tight_layout()
        fname_vol = os.path.join(FIG_DIR, f"volcano_{group}.tiff")
        fig.savefig(fname_vol, dpi=DPI, compression="tiff_lzw")
        print(f"  Saved: {fname_vol}")
        plt.close(fig)

        # -- RF importance barplot -------------------------------------------
        plot_rf_importance(imp_class, imp_reg, group)

        # -- Combined indicator table ----------------------------------------
        df_indicator = build_indicator_table(df_da, df_cor, imp_class, imp_reg)
        fname_tab = os.path.join(TABLE_DIR, f"indicator_taxa_{group}.csv")
        df_indicator.to_csv(fname_tab, index=False)
        print(f"  Saved: {fname_tab}")

        # Print top 10
        print(f"\n  Top 10 indicator genera ({group}):")
        print(df_indicator[["Genus", "log2FC", "FDR", "r",
                             "RF_Class_Imp", "RF_Reg_Imp",
                             "Combined_score"]].head(10).to_string(index=False))

    print("\nDone.\n")



if __name__ == "__main__":
    main()
