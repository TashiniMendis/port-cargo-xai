"""Front end.  Run:  make app   (or: uv run streamlit run app/streamlit_app.py)"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import FEATURE_LABELS, METRICS, PLOTS, PORTS, VESSEL_TYPES  # noqa: E402
from src.data import recent_history  # noqa: E402
from src.serving import Predictor  # noqa: E402

st.set_page_config(page_title="Port Cargo XAI", page_icon="🤖", layout="wide")
TEAL, BLUE, RED = "#0f766e", "#2563eb", "#dc2626"


@st.cache_resource
def get_predictor():
    return Predictor()


if not (ROOT / "models" / "model.pkl").exists():
    st.error("No trained model. Run `make repro` (or `uv run dvc pull`) first.")
    st.stop()

P = get_predictor()
hist = P.history
last_day = hist["date"].max().date()

st.title("🤖 Sri Lanka Port Cargo Estimator")
st.caption(
    "Explainable Boosting Machine (EBM) · estimates the daily import cargo of a port from "
    f"the expected vessel calls · model version `{P.version}`"
)

tab_pred, tab_global, tab_perf, tab_xai, tab_about = st.tabs(
    [
        "🔮 Predict & explain",
        "🧠 How the model thinks",
        "📊 Performance",
        "🔍 XAI evidence",
        "ℹ️ About & ethics",
    ]
)

# --------------------------------------------------------------------------- #
with tab_pred:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        c1, c2 = st.columns(2)
        port = c1.selectbox("Port", PORTS)
        day = c2.date_input("Date", value=last_day + timedelta(days=1), min_value=date(2019, 1, 8))
        rec = P.recorded(port, day)
        st.markdown("**Vessel calls expected that day**")
        if rec:
            st.caption("This date is in the dataset: the boxes show the recorded calls.")
        calls, cols = {}, st.columns(2)
        for i, v in enumerate(VESSEL_TYPES):
            calls[v] = cols[i % 2].number_input(
                FEATURE_LABELS[v], 0, 60, rec[v] if rec else 0, key=f"{v}_{port}_{day}"
            )
        h = recent_history(hist, port, day)
        with st.expander("Recent activity (previous 7 days)"):
            st.caption(f"Auto-filled from {h['from']} to {h['to']}; edit if you know better.")
            imp7 = st.number_input(
                "Mean daily import (t)",
                0.0,
                value=float(round(h["import_roll7"])),
                step=1000.0,
                key=f"i7_{port}_{day}",
            )
            calls7 = st.number_input(
                "Mean daily vessel calls",
                0.0,
                value=float(round(h["portcalls_roll7"], 2)),
                step=0.5,
                key=f"c7_{port}_{day}",
            )
        run = st.button("Estimate import cargo", type="primary")

    with right:
        if not run:
            st.info("Enter the expected vessel calls and click **Estimate import cargo**.")
        else:
            r = P.predict(port, day, calls, imp7, calls7)
            m1, m2 = st.columns(2)
            m1.metric(
                f"Estimated import, {port}, {day:%d %b %Y}", f"{r['prediction_tonnes']:,.0f} t"
            )
            if r["typical_error_tonnes"]:
                m2.metric(
                    "Typical error at this port (test MAE)", f"± {r['typical_error_tonnes']:,.0f} t"
                )
            if rec:
                st.info(f"Recorded value for this day: **{rec['import']:,.0f} t**")
            if sum(calls.values()) == 0:
                st.warning("No vessel calls entered: imports were never recorded on such days.")

            top = [t for t in r["contributions"] if abs(t["value"]) >= 1][:8]
            rest = r["raw_prediction"] - r["intercept"] - sum(t["value"] for t in top)
            # Horizontal waterfall: labels stay readable; read top (average day) to bottom
            # (estimate). Blue raises the estimate, red lowers it.
            values = [r["intercept"]] + [t["value"] for t in top] + [rest, 0]
            fig = go.Figure(
                go.Waterfall(
                    orientation="h",
                    y=["Average port-day"]
                    + [t["label"] for t in top]
                    + ["Other terms", "Estimate"],
                    measure=["absolute"] + ["relative"] * (len(top) + 1) + ["total"],
                    x=values,
                    text=[f"{r['intercept']:,.0f}"]
                    + [f"{v:+,.0f}" for v in values[1:-1]]
                    + [f"{r['prediction_tonnes']:,.0f}"],
                    textposition="outside",
                    increasing={"marker": {"color": BLUE}},
                    decreasing={"marker": {"color": RED}},
                    totals={"marker": {"color": TEAL}},
                    connector={"line": {"color": "#cbd5e1"}},
                )
            )
            fig.update_layout(
                title="Why this estimate? Exact EBM contributions (tonnes)",
                height=120 + 38 * len(values),
                showlegend=False,
                margin={"t": 50, "b": 10, "l": 10, "r": 40},
                yaxis={"autorange": "reversed"},
                xaxis={"title": "tonnes", "rangemode": "tozero"},
            )
            st.plotly_chart(fig)
            pos = [t for t in top if t["value"] > 0][:2]
            neg = [t for t in top if t["value"] < 0][:1]
            text = f"Starting from the average port-day ({r['intercept']:,.0f} t), "
            text += ", ".join(f"**{t['label']}** adds {t['value']:,.0f} t" for t in pos)
            if neg:
                text += f", while **{neg[0]['label']}** removes {abs(neg[0]['value']):,.0f} t"
            st.markdown(text + ".")
            if r["raw_prediction"] < 0:
                st.caption("The additive score was slightly below zero and is reported as 0 t.")

# --------------------------------------------------------------------------- #
with tab_global:
    st.markdown(
        "The EBM is a **glass-box** model: the prediction is a sum of one learned curve "
        "per feature plus a few pairwise interactions. These are the model itself, "
        "not an approximation."
    )
    imp = pd.DataFrame(P.global_importance()).head(15).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=imp["importance"],
            y=imp["label"],
            orientation="h",
            marker_color=[BLUE if " x " in s else TEAL for s in imp["label"]],
        )
    )
    fig.update_layout(
        title="Global importance: mean |contribution| (t/day); blue = interaction",
        height=480,
        margin={"t": 50},
    )
    st.plotly_chart(fig)

    mains = [t for t in P.model.term_names_ if " & " not in t]
    feat = st.selectbox(
        "Shape function of",
        mains,
        index=mains.index("portcalls_dry_bulk"),
        format_func=lambda f: FEATURE_LABELS[f],
    )
    d = P.shape_function(feat)
    fig = go.Figure()
    if feat == "port":
        fig.add_bar(x=d["names"], y=d["scores"], marker_color=TEAL)
    else:
        x = d["names"][:-1]
        fig.add_scatter(
            x=x,
            y=d["upper_bounds"],
            line={"shape": "hv", "width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_scatter(
            x=x,
            y=d["lower_bounds"],
            line={"shape": "hv", "width": 0},
            fill="tonexty",
            fillcolor="rgba(15,118,110,0.18)",
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_scatter(
            x=x, y=d["scores"], line={"shape": "hv", "color": TEAL, "width": 3}, name="contribution"
        )
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"How '{FEATURE_LABELS[feat]}' changes the estimate (t/day)",
        xaxis_title=FEATURE_LABELS[feat],
        yaxis_title="Contribution (t/day)",
        height=400,
    )
    st.plotly_chart(fig)

# --------------------------------------------------------------------------- #
with tab_perf:
    exp_path = METRICS / "experiments.csv"
    if exp_path.exists():
        exp = pd.read_csv(exp_path)
        st.subheader("EBM vs standard models (test set 2025-2026)")
        st.dataframe(
            exp.set_index("model")[["mae", "rmse", "r2", "wape"]].style.format(
                {"mae": "{:,.0f}", "rmse": "{:,.0f}", "r2": "{:.3f}", "wape": "{:.1f}"}
            )
        )
    pp = P.metrics["per_port"]
    if pp:
        st.subheader("EBM error by port")
        st.dataframe(
            pd.DataFrame(pp)
            .T[["mae", "wape", "bias", "mean_actual", "rows"]]
            .style.format("{:,.1f}")
        )
    for f in ["model_comparison.png", "actual_vs_predicted.png", "residuals.png"]:
        if (PLOTS / f).exists():
            st.image(str(PLOTS / f))

# --------------------------------------------------------------------------- #
with tab_xai:
    captions = {
        "ebm_shape_functions.png": "EBM shape functions (glass box)",
        "ebm_top_interaction.png": "Strongest pairwise interaction",
        "ebm_local_holdout.png": "Exact local explanation of the left-out day",
        "shap_beeswarm.png": "SHAP (model-agnostic cross-check)",
        "lime_holdout.png": "LIME on the same left-out day",
        "pdp_ice_colombo.png": "Partial dependence + ICE (Colombo)",
        "permutation_importance.png": "Permutation importance",
        "bias_by_port.png": "Bias and error by port",
    }
    for f, cap in captions.items():
        if (PLOTS / "xai" / f).exists():
            st.markdown(f"**{cap}**")
            st.image(str(PLOTS / "xai" / f))

# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(f"""
**Problem.** Sri Lankan ports plan cranes, berths, trucks and customs staff before ships arrive.
Vessel calls are scheduled in advance, but cargo tonnage is only known after unloading.

**Data.** IMF PortWatch daily port activity for Sri Lanka, derived from satellite AIS
vessel positions. {len(hist):,} port-days, {hist["date"].min():%d %b %Y} to {last_day:%d %b %Y}.
Aggregated per port and day: **no personal or commercially sensitive data**.

**Model.** Explainable Boosting Machine (a GA²M), trained on 2019-2024, tested on 2025-2026.

**Use responsibly.** The estimate is a planning aid, not a measurement. It is least accurate
on days with unusually large vessels and at Colombo, where ship size (not in the data) matters
most. Do not use it for customs, tax or commercial decisions about individual shipments.
""")
