"""Stage `explain`: explainability (XAI) and bias analysis of the trained EBM.

Methods
  1. EBM global term importance            (built-in, glass box)
  2. EBM shape functions + interaction     (built-in, glass box)
  3. EBM local explanation of one sample   (built-in, exact)
  4. SHAP (permutation, model-agnostic)    -> cross-checks the EBM's own explanation
  5. LIME on the same sample               -> a second local, model-agnostic view
  6. Partial dependence + ICE (Colombo)
  7. Permutation importance on the test set
  8. Bias / fairness: error by port, by year and on high-volume days

Usage:  python -m src.explain     Writes plots/xai/*.png, metrics/xai_summary.json, fairness.json
"""

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from .config import FEATURE_LABELS, FEATURES, METRICS, PLOTS, PORTS, TARGET, VESSEL_TYPES
from .data import xy
from .evaluate import load_model, per_port, predict
from .metrics import regression_metrics
from .train import load_split

warnings.filterwarnings("ignore")
OUT = PLOTS / "xai"
BLUE, RED, TEAL, GREY = "#2563eb", "#dc2626", "#0f766e", "#9aa5b1"
RNG = 42


def label(name: str) -> str:
    return " x ".join(FEATURE_LABELS.get(p, p) for p in name.split(" & "))


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Numeric wrapper: SHAP and LIME need an all-numeric matrix, the EBM wants `port` as text
# --------------------------------------------------------------------------- #
def encode(X: pd.DataFrame) -> np.ndarray:
    A = X.copy()
    A["port"] = A["port"].map({p: i for i, p in enumerate(PORTS)})
    return A.to_numpy(float)


def numeric_predict(model):
    def f(A):
        df = pd.DataFrame(np.atleast_2d(A), columns=FEATURES)
        df["port"] = [PORTS[int(round(i))] for i in df["port"]]
        for c in FEATURES[1:]:
            df[c] = df[c].astype(float)
        return model.predict(df)

    return f


# --------------------------------------------------------------------------- #
def ebm_global(model):
    imp = pd.Series(model.term_importances(), index=model.term_names_).sort_values()
    top = imp.tail(15)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(
        [label(n) for n in top.index],
        top.values,
        color=[TEAL if " & " not in n else BLUE for n in top.index],
    )
    ax.set_xlabel("Mean |contribution| to the prediction (tonnes/day)")
    ax.set_title("EBM global importance (teal = single feature, blue = interaction)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "ebm_global_importance.png")
    return imp.sort_values(ascending=False)


def ebm_shapes(model, imp):
    g = model.explain_global()
    names = list(model.term_names_)
    mains = [n for n in imp.index if " & " not in n][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, term in zip(axes.flat, mains, strict=False):
        d = g.data(names.index(term))
        if term == "port":
            ax.bar(
                d["names"],
                d["scores"],
                color=TEAL,
                yerr=np.array(d["upper_bounds"]) - np.array(d["scores"]),
                capsize=3,
            )
        else:
            edges = np.asarray(d["names"], float)
            s, lo, hi = (
                np.asarray(d[k], float) for k in ("scores", "lower_bounds", "upper_bounds")
            )
            ax.step(edges[:-1], s, where="post", color=TEAL, lw=1.8)
            ax.fill_between(edges[:-1], lo, hi, step="post", color=TEAL, alpha=0.2)
            if term == "import_roll7":
                ax.set_xscale("symlog", linthresh=1000)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(label(term), fontsize=10)
        ax.set_ylabel("Contribution (t/day)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "EBM shape functions: how each feature changes the estimate "
        "(band = uncertainty across bags)"
    )
    _save(fig, "ebm_shape_functions.png")

    inter = [n for n in imp.index if " & " in n][0]
    d = g.data(names.index(inter))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(np.asarray(d["scores"], float).T, aspect="auto", cmap="RdBu_r", origin="lower")
    left = [str(v) for v in d["left_names"]]
    right = np.asarray(d["right_names"], float)
    ax.set_xticks(range(len(d["scores"])))
    ax.set_xticklabels(
        left
        if len(left) == len(d["scores"])
        else [f"{v:.0f}" for v in np.asarray(left, float)[:-1]]
    )
    ax.set_yticks(range(len(right) - 1))
    ax.set_yticklabels([f"{a:g}-{b:g}" for a, b in zip(right[:-1], right[1:], strict=True)])
    a, b = inter.split(" & ")
    ax.set_xlabel(label(a))
    ax.set_ylabel(label(b))
    fig.colorbar(im, ax=ax, label="Contribution (t/day)")
    ax.set_title(f"Strongest interaction: {label(inter)}")
    _save(fig, "ebm_top_interaction.png")
    return mains, inter


def ebm_local(model, row: pd.DataFrame) -> dict:
    contrib = pd.Series(model.eval_terms(row)[0], index=model.term_names_)
    pred = float(model.intercept_ + contrib.sum())
    top = contrib.reindex(contrib.abs().sort_values().index).tail(10)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(
        [label(n) for n in top.index],
        top.values,
        color=[BLUE if v > 0 else RED for v in top.values],
    )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Contribution (tonnes)")
    fig.suptitle(
        f"EBM local explanation, Colombo {row.index.name or ''}\n"
        f"average port-day {model.intercept_:,.0f} t -> prediction {pred:,.0f} t"
    )
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "ebm_local_holdout.png")
    return {
        "intercept": float(model.intercept_),
        "prediction": pred,
        "contributions": {
            k: round(float(v), 1) for k, v in contrib.sort_values(key=abs, ascending=False).items()
        },
    }


def shap_check(model, X: pd.DataFrame, ebm_imp: pd.Series) -> dict:
    f = numeric_predict(model)
    bg = encode(X.sample(100, random_state=RNG))
    sample = X.sample(300, random_state=RNG + 1)
    explainer = shap.PermutationExplainer(
        f,
        shap.maskers.Independent(bg, max_samples=100),
        feature_names=[FEATURE_LABELS[c] for c in FEATURES],
    )
    sv = explainer(encode(sample), max_evals=2 * len(FEATURES) + 1, silent=True)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("SHAP values of the EBM (300 test port-days)")
    plt.gcf().set_size_inches(8, 5.5)
    plt.tight_layout()
    plt.savefig(OUT / "shap_beeswarm.png", dpi=130)
    plt.close()

    shap_imp = pd.Series(np.abs(sv.values).mean(0), index=FEATURES).sort_values(ascending=False)
    # EBM main effects only (interactions are spread over their two features by SHAP)
    ebm_main = ebm_imp[[n for n in ebm_imp.index if " & " not in n]].reindex(FEATURES)
    rho = spearmanr(ebm_main.values, shap_imp.reindex(FEATURES).values).statistic
    return {"shap_mean_abs": shap_imp.round(1).to_dict(), "spearman_ebm_vs_shap": float(rho)}


def lime_check(model, X_train: pd.DataFrame, row: pd.DataFrame) -> list:
    # Vessel counts are small integers (mostly 0): LIME's quartile bins collapse on them, so
    # they are treated as categorical (sampled from their observed values) like `port`.
    data = encode(X_train.dropna().sample(3000, random_state=RNG))
    cat_idx = [FEATURES.index(c) for c in ["port", *VESSEL_TYPES, "portcalls_total"]]
    cat_names = {0: PORTS}
    for j in cat_idx[1:]:
        cat_names[j] = [str(v) for v in range(61)]  # counts 0..60 (API upper bound)
    explainer = LimeTabularExplainer(
        data,
        feature_names=[FEATURE_LABELS[c] for c in FEATURES],
        categorical_features=cat_idx,
        categorical_names=cat_names,
        mode="regression",
        discretize_continuous=True,
        random_state=RNG,
    )
    exp = explainer.explain_instance(
        encode(row)[0], numeric_predict(model), num_features=8, num_samples=4000
    )
    items = exp.as_list()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names, vals = zip(*reversed(items), strict=True)
    ax.barh(names, vals, color=[BLUE if v > 0 else RED for v in vals])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("LIME weight (tonnes)")
    ax.set_title("LIME explanation of the same hold-out sample")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "lime_holdout.png")
    return [(n, round(float(v), 1)) for n, v in items]


def pdp(model, X: pd.DataFrame):
    col = X[X["port"] == "Colombo"].astype({c: float for c in FEATURES[1:]})
    feats = ["portcalls_container", "portcalls_dry_bulk", "portcalls_tanker", "import_roll7"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    PartialDependenceDisplay.from_estimator(
        model,
        col,
        feats,
        kind="both",
        subsample=120,
        random_state=RNG,
        ax=axes,
        grid_resolution=15,
        ice_lines_kw={"color": GREY, "alpha": 0.25},
        pd_line_kw={"color": TEAL, "lw": 2.5},
    )
    for ax, f in zip(axes, feats, strict=True):
        ax.set_xlabel(FEATURE_LABELS[f])
    fig.suptitle("Partial dependence (teal) and ICE curves (grey), Colombo test days")
    _save(fig, "pdp_ice_colombo.png")


def perm_importance(model, X, y) -> dict:
    r = permutation_importance(
        model, X, y, scoring="neg_mean_absolute_error", n_repeats=5, random_state=RNG, n_jobs=1
    )
    s = pd.Series(r.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(
        [FEATURE_LABELS[c] for c in s.index],
        s.values,
        xerr=pd.Series(r.importances_std, index=FEATURES)[s.index],
        color=TEAL,
    )
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (tonnes/day)")
    ax.set_title("Permutation importance (model-agnostic)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "permutation_importance.png")
    return s.sort_values(ascending=False).round(1).to_dict()


def fairness(test: pd.DataFrame, pred: np.ndarray) -> dict:
    ports = per_port(test, pred)
    years = {}
    for yr, idx in test.groupby(test["date"].dt.year).groups.items():
        m = test.index.get_indexer(idx)
        years[int(yr)] = regression_metrics(test[TARGET].to_numpy()[m], pred[m])
    y = test[TARGET].to_numpy()
    big = y > np.quantile(y[y > 0], 0.9)
    extremes = {
        "high_volume_days": int(big.sum()),
        "bias_on_high_volume_days": float((pred[big] - y[big]).mean()),
        "bias_on_other_days": float((pred[~big] - y[~big]).mean()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(ports)
    axes[0].bar(names, [ports[p]["wape"] or 0 for p in names], color=TEAL)
    axes[0].set_title("Relative error by port (WAPE %; Galle: no cargo, undefined)")
    axes[1].bar(
        names,
        [ports[p]["bias"] for p in names],
        color=[BLUE if ports[p]["bias"] > 0 else RED for p in names],
    )
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("Mean bias by port (t/day; + = over-predicts)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "bias_by_port.png")
    return {"by_port": ports, "by_year": years, "extremes": extremes}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    train, test, hold = load_split("train"), load_split("test"), load_split("holdout")
    X, y = xy(test)
    row = xy(hold[hold["port"] == "Colombo"])[0]
    row.index.name = str(hold["date"].iloc[0].date())

    imp = ebm_global(model)
    mains, inter = ebm_shapes(model, imp)
    local = ebm_local(model, row)
    shap_res = shap_check(model, X, imp)
    lime_res = lime_check(model, xy(train)[0], row)
    pdp(model, X)
    perm = perm_importance(model, X, y)
    fair = fairness(test, predict(model, X))

    summary = {
        "ebm_term_importance": imp.round(1).to_dict(),
        "top_interaction": inter,
        "holdout_sample": {
            "date": row.index.name,
            "port": "Colombo",
            "actual": float(hold.loc[hold["port"] == "Colombo", TARGET].iloc[0]),
            **local,
        },
        **shap_res,
        "lime_holdout": lime_res,
        "permutation_importance": perm,
    }
    METRICS.mkdir(exist_ok=True)
    (METRICS / "xai_summary.json").write_text(json.dumps(summary, indent=2))
    (METRICS / "fairness.json").write_text(json.dumps(fair, indent=2))
    print(
        json.dumps({k: summary[k] for k in ["top_interaction", "spearman_ebm_vs_shap"]}, indent=2)
    )
    print("EBM top terms:", list(imp.index[:6]))
    print("permutation:", list(perm)[:6])
    print("LIME:", lime_res[:5])


if __name__ == "__main__":
    main()
