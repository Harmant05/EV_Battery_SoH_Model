from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="EV Battery SoH Research Dashboard", page_icon="🔋", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEG_FILE = DATA / "ev_battery_degradation.csv"
TEL_FILE = DATA / "ev_battery_failure_prediction.csv"
SEED = 42

st.title("🔋 EV Battery State of Health (SoH) Research Dashboard")
st.caption("Interactive exploration and machine-learning demonstration based on the two supplied CSV datasets and the project notebook.")

@st.cache_data
def load_data():
    deg = pd.read_csv(DEG_FILE)
    tel = pd.read_csv(TEL_FILE)
    deg["SoH_Percent"] = pd.to_numeric(deg["SoH_Percent"], errors="coerce")
    tel["state_of_health"] = pd.to_numeric(tel["state_of_health"], errors="coerce")
    return deg, tel

deg_df, tel_df = load_data()

DATASETS = {
    "Degradation dataset": {
        "df": deg_df, "target": "SoH_Percent", "id": "Vehicle_ID",
        "drop": {"SoH_Percent", "Battery_Status", "Vehicle_ID", "Condition_Label"},
        "label": "SoH_Percent"
    },
    "Battery telemetry dataset": {
        "df": tel_df, "target": "state_of_health", "id": "vehicle_id",
        "drop": {
            "state_of_health", "vehicle_id", "battery_serial", "battery_failure",
            "battery_health_percent", "capacity_loss_percent", "remaining_capacity",
            "aging_score", "thermal_health_score", "predicted_remaining_life_cycles",
            "Condition_Label"
        },
        "label": "state_of_health"
    }
}

def feature_columns(info):
    return [c for c in info["df"].columns if c not in info["drop"]]

def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ], remainder="drop")

@st.cache_resource
def fit_models(dataset_name):
    info = DATASETS[dataset_name]
    df, target = info["df"].copy(), info["target"]
    features = feature_columns(info)
    work = df.loc[df[target].notna()].copy()
    X, y = work[features].copy(), work[target].astype(float)
    # Match notebook intent: grouped split if ID is repeated, otherwise a reproducible random split.
    id_col = info["id"]
    if id_col in work.columns and work[id_col].nunique() < len(work) and work[id_col].nunique() > 1:
        from sklearn.model_selection import GroupShuffleSplit
        split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
        tr, te = next(split.split(X, y, groups=work[id_col].astype(str)))
        split_note = f"Group split by {id_col}"
        X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
        split_note = "Random split (no repeated group ID)"
    estimators = {
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    }
    results, rows = {}, []
    for name, estimator in estimators.items():
        pipe = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "MAE (percentage points)": mean_absolute_error(y_test, pred),
            "RMSE (percentage points)": mean_squared_error(y_test, pred) ** 0.5,
            "R²": r2_score(y_test, pred)
        })
        results[name] = {"pipeline": pipe, "pred": pred, "X_test": X_test, "y_test": y_test}
    metrics = pd.DataFrame(rows).sort_values("MAE (percentage points)").reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    return {"metrics": metrics, "results": results, "best_name": best_name,
            "features": features, "X_train": X_train, "y_train": y_train,
            "split_note": split_note, "rows_used": len(work), "rows_dropped": int(df[target].isna().sum())}

with st.sidebar:
    st.header("Dashboard controls")
    dataset_name = st.selectbox("Choose dataset", list(DATASETS.keys()))
    view = st.radio("Section", ["Dataset overview", "Explore changes", "Model evaluation", "Predict SoH"])
    st.markdown("---")
    st.caption("The datasets are analyzed separately because their schemas and target definitions differ.")

info = DATASETS[dataset_name]
df, target = info["df"], info["target"]
numeric_df = df.select_dtypes(include=np.number)
target_values = pd.to_numeric(df[target], errors="coerce").dropna()

if view == "Dataset overview":
    st.subheader(f"{dataset_name}: overview")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Rows with SoH", f"{target_values.notna().sum():,}")
    d.metric("Missing SoH", f"{df[target].isna().sum():,}")
    st.markdown("### Data preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.download_button(
        "Download selected dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=dataset_name.lower().replace(" ", "_") + ".csv",
        mime="text/csv"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("### Target distribution")
        fig = px.histogram(df, x=target, nbins=35, title=f"{target} distribution", labels={target: "SoH (%)"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Missing values by column")
        miss = (df.isna().mean() * 100).sort_values(ascending=False).head(20).reset_index()
        miss.columns = ["Column", "Missing (%)"]
        fig = px.bar(miss.sort_values("Missing (%)"), x="Missing (%)", y="Column", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Battery condition overview")
    st.caption("This section summarizes observed dataset SoH values. The bands are illustrative screening ranges, not universal battery replacement rules.")
    soh_valid = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(soh_valid):
        mean_soh = float(soh_valid.mean())
        median_soh = float(soh_valid.median())
        low_share = float((soh_valid < 80).mean() * 100)
        mid_share = float(((soh_valid >= 80) & (soh_valid < 90)).mean() * 100)
        high_share = float((soh_valid >= 90).mean() * 100)
        q1, q3 = float(soh_valid.quantile(0.25)), float(soh_valid.quantile(0.75))
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Mean SoH", f"{mean_soh:.2f}%")
        k2.metric("Median SoH", f"{median_soh:.2f}%")
        k3.metric("Below 80%", f"{low_share:.1f}%")
        k4.metric("80% to <90%", f"{mid_share:.1f}%")
        k5.metric("90% or higher", f"{high_share:.1f}%")
        band_df = pd.DataFrame({
            "Illustrative SoH band": ["Below 80%", "80% to <90%", "90% or higher"],
            "Records (%)": [low_share, mid_share, high_share],
            "Records": [
                int((soh_valid < 80).sum()),
                int(((soh_valid >= 80) & (soh_valid < 90)).sum()),
                int((soh_valid >= 90).sum())
            ]
        })
        fig = px.bar(band_df, x="Illustrative SoH band", y="Records (%)", text="Records",
                     title="How records fall across illustrative SoH bands",
                     color="Illustrative SoH band")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        if low_share >= 20:
            st.warning(f"Review signal: {low_share:.1f}% of records are below 80% SoH. Inspect individual records and validate the dataset before drawing a battery-condition conclusion.")
        elif mid_share >= 30:
            st.info(f"Review signal: {mid_share:.1f}% of records are between 80% and 90% SoH. Examine usage and charging patterns and compare with manufacturer guidance.")
        else:
            st.success(f"Most records are in the 90%+ band ({high_share:.1f}%). This describes the dataset distribution only, not the condition of a specific vehicle.")
        st.caption(f"Interquartile range: {q1:.2f}% to {q3:.2f}%. A dataset-level chart cannot diagnose an individual battery.")
    st.markdown("### Data quality")
    quality = pd.DataFrame({
        "Data type": df.dtypes.astype(str),
        "Missing count": df.isna().sum(),
        "Missing (%)": (df.isna().mean() * 100).round(2),
        "Unique values": df.nunique(dropna=True)
    }).sort_values("Missing (%)", ascending=False)
    st.dataframe(quality, use_container_width=True)

elif view == "Explore changes":
    st.subheader("Explore how dataset variables relate to SoH")
    st.write("Choose a numeric field to view its observed values against the target. These are associations in the supplied data, not proof of causation.")
    candidate_features = [c for c in numeric_df.columns if c != target and numeric_df[c].nunique(dropna=True) > 1]
    default_feature = candidate_features[0] if candidate_features else None
    if default_feature:
        selected = st.selectbox("Numeric variable", candidate_features, index=0)
        plot_df = df[[selected, target]].copy().dropna()
        # Avoid optional statsmodels dependency: scatter plot works with Plotly alone.
        fig = px.scatter(plot_df, x=selected, y=target, opacity=0.55,
                         title=f"{target} versus {selected}",
                         labels={target: "SoH (%)"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Group comparison")
        cat_candidates = [c for c in df.select_dtypes(exclude=np.number).columns if c not in info["drop"] and df[c].nunique(dropna=True) <= 30]
        if cat_candidates:
            cat = st.selectbox("Categorical field", cat_candidates)
            grouped = df[[cat, target]].dropna().groupby(cat)[target].agg(["count", "mean", "median"]).reset_index()
            grouped.columns = [cat, "Record count", "Mean SoH (%)", "Median SoH (%)"]
            fig = px.bar(grouped, x=cat, y="Mean SoH (%)", hover_data=["Record count", "Median SoH (%)"],
                         title=f"Mean SoH by {cat}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(grouped.sort_values("Mean SoH (%)", ascending=False), use_container_width=True)
    st.markdown("### Correlation heatmap")
    corr_cols = [c for c in numeric_df.columns if numeric_df[c].nunique(dropna=True) > 1]
    if len(corr_cols) > 1:
        corr = df[corr_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Numeric-feature correlation matrix")
        st.plotly_chart(fig, use_container_width=True)

elif view == "Model evaluation":
    st.subheader("Train and compare machine-learning models")
    st.info("The app fits Extra Trees and Random Forest regressors on an 80/20 train-test split. Training is cached for this session.")
    with st.spinner("Training/evaluating models on the selected dataset..."):
        fitted = fit_models(dataset_name)
    m1, m2, m3 = st.columns(3)
    bestrow = fitted["metrics"].iloc[0]
    m1.metric("Best model by test MAE", fitted["best_name"])
    m2.metric("Test MAE", f'{bestrow["MAE (percentage points)"]:.3f} pp')
    m3.metric("Test R²", f'{bestrow["R²"]:.4f}')
    st.caption(f'{fitted["split_note"]}. Target-missing rows excluded: {fitted["rows_dropped"]:,}.')
    mae_value = float(bestrow["MAE (percentage points)"])
    st.markdown("### Model quality interpretation")
    if mae_value < 1.0:
        st.success(f"Low test MAE ({mae_value:.3f} SoH percentage points) on this dataset split. Check target leakage, synthetic-data rules, and an independent validation set before calling the model reliable.")
    elif mae_value < 3.0:
        st.info(f"Test MAE is {mae_value:.3f} SoH percentage points. This can be useful for a research prototype, but review residuals and validate on truly unseen vehicles before deployment.")
    else:
        st.warning(f"Test MAE is {mae_value:.3f} SoH percentage points. Review the features, missing data, target definition, and split strategy before relying on predictions.")
    st.caption("These are research-review prompts, not universal acceptance thresholds. R² and MAE alone do not prove the model is safe for real-world battery decisions.")
    st.markdown("### Model comparison")
    st.dataframe(fitted["metrics"].style.format({
        "MAE (percentage points)": "{:.4f}", "RMSE (percentage points)": "{:.4f}", "R²": "{:.4f}"
    }), use_container_width=True)
    chosen = st.selectbox("Show diagnostic plots for", list(fitted["results"].keys()), index=list(fitted["results"].keys()).index(fitted["best_name"]))
    res = fitted["results"][chosen]
    actual = np.asarray(res["y_test"])
    pred = np.asarray(res["pred"])
    left, right = st.columns(2)
    with left:
        lo = float(min(actual.min(), pred.min()))
        hi = float(max(actual.max(), pred.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual, y=pred, mode="markers", marker={"opacity": 0.45}, name="Test records"))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal prediction"))
        fig.update_layout(title="Actual vs predicted SoH", xaxis_title="Actual SoH (%)", yaxis_title="Predicted SoH (%)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        residual = actual - pred
        fig = px.scatter(x=pred, y=residual, opacity=0.45, labels={"x": "Predicted SoH (%)", "y": "Residual (actual − predicted)"},
                         title="Residual analysis")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Permutation feature importance")
    if st.button("Calculate feature importance"):
        with st.spinner("Calculating permutation importance..."):
            perm = permutation_importance(res["pipeline"], res["X_test"], res["y_test"],
                                          scoring="neg_mean_absolute_error", n_repeats=3,
                                          random_state=SEED, n_jobs=-1)
        imp = pd.DataFrame({"Feature": res["X_test"].columns,
                            "Importance mean": perm.importances_mean,
                            "Importance std": perm.importances_std}).sort_values("Importance mean", ascending=False).head(20)
        fig = px.bar(imp.sort_values("Importance mean"), x="Importance mean", y="Feature", orientation="h",
                     error_x="Importance std", title="Top 20 input features by permutation importance")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(imp, use_container_width=True)

else:
    st.subheader("Predict SoH for a custom vehicle record")
    st.warning("This is a research estimate based on the selected dataset, not a certified battery diagnosis. For telemetry records, the model excludes health/failure fields that could leak the target.")
    with st.spinner("Preparing the trained model..."):
        fitted = fit_models(dataset_name)
    model_name = st.selectbox("Prediction model", list(fitted["results"].keys()), index=list(fitted["results"].keys()).index(fitted["best_name"]))
    pipe = fitted["results"][model_name]["pipeline"]
    features = fitted["features"]
    trainX = fitted["X_train"]
    st.write("Enter available feature values. Defaults use typical values from the training dataset (median for numeric fields, most common value for categorical fields).")
    custom = {}
    with st.form("prediction_form"):
        cols = st.columns(3)
        for i, col in enumerate(features):
            series = trainX[col]
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(series):
                    med = pd.to_numeric(series, errors="coerce").median()
                    if pd.isna(med): med = 0.0
                    minv = pd.to_numeric(series, errors="coerce").min()
                    maxv = pd.to_numeric(series, errors="coerce").max()
                    custom[col] = st.number_input(col, value=float(med), min_value=float(minv) if pd.notna(minv) else None,
                                                  max_value=float(maxv) if pd.notna(maxv) else None)
                else:
                    opts = series.dropna().astype(str).value_counts().index.tolist()
                    if not opts: opts = ["Unknown"]
                    custom[col] = st.selectbox(col, opts, index=0)
        submitted = st.form_submit_button("Estimate battery SoH")
    if submitted:
        pred = float(pipe.predict(pd.DataFrame([custom], columns=features))[0])
        st.metric("Estimated State of Health", f"{pred:.2f}%")
        if pred >= 90:
            st.write("Interpretation: the estimate is in the higher part of the SoH scale. Check the vehicle manufacturer's guidance and measurement uncertainty.")
        elif pred >= 80:
            st.write("Interpretation: the estimate is in a middle range. Review real battery diagnostics, operating history, and manufacturer guidance.")
        else:
            st.write("Interpretation: the estimate is relatively low. Consider a qualified battery health assessment before making decisions.")
        st.caption("The explanatory ranges above are only interface guidance, not universal replacement thresholds.")

st.markdown("---")
st.caption("Research note: the supplied notebook reports Extra Trees as the lowest-test-MAE model for both datasets. The dashboard retrains the models from the bundled CSVs; values can differ slightly if the environment or package versions differ.")
