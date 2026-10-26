import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="EV Battery Health Predictor", page_icon="🔋", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

LABELS = {
    "avg_voltage": "Average voltage (V)",
    "min_voltage": "Minimum voltage (V)",
    "avg_current": "Average current (A)",
    "avg_temp": "Average temperature (°C)",
    "max_temp": "Maximum temperature (°C)",
    "discharge_time_s": "Discharge time (s)",
}


@st.cache_resource
def load_models():
    capacity_model = XGBRegressor()
    capacity_model.load_model(MODEL_DIR / "capacity_model.json")
    rul_model = XGBRegressor()
    rul_model.load_model(MODEL_DIR / "rul_model.json")
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    return capacity_model, rul_model, meta


def predict(frame: pd.DataFrame, start_capacity: float):
    """Return capacity (Ah), SOH (% of the starting capacity) and RUL (>= 0 cycles) for each row."""
    X = frame[FEATURES].astype(float)
    capacity = np.clip(capacity_model.predict(X), 0, None)
    soh = capacity / start_capacity * 100
    rul = np.clip(rul_model.predict(X), 0, None)
    return capacity, soh, rul


def health_status(soh: float) -> str:
    if soh >= 90:
        return "Healthy"
    if soh >= meta["eol_soh"]:
        return "Ageing, still above the end-of-life line"
    return "At or past end of life"


try:
    capacity_model, rul_model, meta = load_models()
except Exception as e:
    st.error(
        "Could not load the saved models. Run the Step 6 notebook first "
        "so the `models/` folder exists."
    )
    st.exception(e)
    st.stop()

FEATURES = meta["features"]
EOL = meta["eol_soh"]
RATED = meta["rated_capacity"]
START_HELP = (
    "SOH is measured against this number. Use the capacity the battery had on its first cycle, "
    f"or its rated capacity ({RATED} Ah for these cells) if you do not have it."
)

# ------------------------------------------------------------------ header
st.title("🔋 EV Battery Health Predictor")
st.write(
    "Estimates a lithium-ion battery's **capacity**, **State of Health (SOH)** and "
    "**Remaining Useful Life (RUL)** from measurements taken during one discharge cycle."
)
st.caption(
    f"Models trained on NASA battery aging data ({', '.join(meta['trained_on'])}). "
    f"SOH = predicted capacity ÷ starting capacity. End of life is defined as SOH ≤ {EOL}%."
)

tab_manual, tab_upload, tab_about = st.tabs(["Single cycle", "Upload many cycles", "About"])

# ------------------------------------------------------------------ tab 1
with tab_manual:
    st.subheader("Enter the measurements from one discharge cycle")

    preset = st.selectbox("Start from an example", list(meta["examples"].keys()))
    example = meta["examples"][preset]
    st.caption(
        f"Example from battery {meta['example_battery']}, cycle {example['cycle']} "
        f"(measured capacity {example['actual_capacity']:.3f} Ah, which is "
        f"{example['actual_soh']:.1f}% of its first-cycle capacity). Change any value below."
    )

    start_capacity = st.number_input(
        "Starting capacity of this battery (Ah)",
        min_value=0.5,
        max_value=5.0,
        value=float(example["start_capacity"]),
        step=0.01,
        format="%.3f",
        key=f"start_{preset}",
        help=START_HELP,
    )

    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            values[feat] = st.number_input(
                LABELS.get(feat, feat),
                value=float(example["values"][feat]),
                format="%.4f",
                key=f"{feat}_{preset}",
            )

    out_of_range = [
        LABELS.get(f, f)
        for f in FEATURES
        if not (meta["ranges"][f]["min"] <= values[f] <= meta["ranges"][f]["max"])
    ]
    if out_of_range:
        st.warning(
            "These values are outside the range the model was trained on, so the prediction "
            "may be unreliable: " + ", ".join(out_of_range)
        )

    cap_arr, soh_arr, rul_arr = predict(pd.DataFrame([values]), start_capacity)
    cap, soh, rul = float(cap_arr[0]), float(soh_arr[0]), float(rul_arr[0])

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted capacity", f"{cap:.3f} Ah")
    c2.metric("Predicted SOH", f"{soh:.1f} %")
    c3.metric("Predicted remaining life", f"{rul:.0f} cycles")
    c4.metric("Status", health_status(soh))
    st.progress(int(round(min(max(soh, 0), 100))))

    if soh > 100:
        st.info(
            "SOH is above 100% because the predicted capacity is higher than the starting "
            "capacity you entered. Check that starting capacity."
        )

# ------------------------------------------------------------------ tab 2
with tab_upload:
    st.subheader("Predict for many cycles at once")
    st.write(
        "Upload a CSV with one row per discharge cycle. Required columns: "
        + ", ".join(f"`{f}`" for f in FEATURES)
        + ". An optional `cycle` column is used for the chart axis."
    )

    uploaded = st.file_uploader("CSV file", type="csv")
    use_sample = st.checkbox(f"Use the built-in sample (battery {meta['example_battery']})")
    using_sample = use_sample and uploaded is None

    start_upload = st.number_input(
        "Starting capacity of this battery (Ah)",
        min_value=0.5,
        max_value=5.0,
        value=float(meta["examples"]["Early life"]["start_capacity"]) if using_sample else float(RATED),
        step=0.01,
        format="%.3f",
        key=f"start_upload_{using_sample}",
        help=START_HELP,
    )

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        sample_path = BASE_DIR / "sample_input.csv"
        if sample_path.exists():
            data = pd.read_csv(sample_path)
        else:
            st.error("sample_input.csv was not found next to app.py.")

    if data is not None:
        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            data = data.dropna(subset=FEATURES).reset_index(drop=True)
            if data.empty:
                st.error("No complete rows were found in the file.")
            else:
                cap_arr, soh_arr, rul_arr = predict(data, start_upload)

                result = data.copy()
                result["Predicted capacity (Ah)"] = cap_arr.round(4)
                result["Predicted SOH (%)"] = soh_arr.round(2)
                result["Predicted RUL (cycles)"] = rul_arr.round(1)
                x_col = "cycle" if "cycle" in result.columns else None
                plot_df = result.set_index(x_col) if x_col else result

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Cycles", len(result))
                m2.metric("Capacity at last cycle", f"{cap_arr[-1]:.3f} Ah")
                m3.metric("SOH at last cycle", f"{soh_arr[-1]:.1f} %")
                m4.metric("RUL at last cycle", f"{rul_arr[-1]:.0f} cycles")

                left, right = st.columns(2)
                with left:
                    st.markdown("**Predicted SOH (%)**")
                    st.line_chart(plot_df["Predicted SOH (%)"])
                with right:
                    st.markdown("**Predicted RUL (cycles)**")
                    st.line_chart(plot_df["Predicted RUL (cycles)"])

                st.dataframe(result, width="stretch")
                st.download_button(
                    "Download predictions as CSV",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="battery_predictions.csv",
                    mime="text/csv",
                )

# ------------------------------------------------------------------ tab 3
with tab_about:
    st.subheader("How this works")
    st.markdown(
        f"""
**Data.** NASA Prognostics Center of Excellence Li-ion battery aging dataset: batteries
charged and discharged in a lab until they wore out.

**Inputs.** Six numbers summarising one discharge cycle (voltage, current, temperature, discharge time),
plus the battery's starting capacity.

**Models.** Two XGBoost regressors. One predicts the battery's **capacity in Ah**, and SOH is that
capacity divided by the starting capacity you enter. The other predicts the number of discharge cycles
left until SOH reaches {EOL}%.

**Why capacity and not SOH directly?** The batteries in the dataset start at slightly different
capacities. Predicting SOH directly forces every battery to start at 100%, which blurs that difference
and made the model under-predict early life. Predicting capacity first worked better in testing.

**Limits you should know about**
- Only four batteries were used, all from one lab, so this is a **proof of concept**,
  not a tool for real vehicles.
- SOH depends on the starting capacity you enter. Using the rated capacity instead of the real
  first-cycle capacity shifts every SOH value by the difference between them.
- Discharge time is almost the same information as capacity, so the model works best when a
  full discharge was measured. A real vehicle rarely gives you one.
- The remaining-life prediction is much less accurate than the capacity prediction, especially
  early in a battery's life.
- Predictions outside the training ranges are unreliable (the app warns you).
- The {EOL}% end-of-life line is a common convention, not a universal rule.
"""
    )
