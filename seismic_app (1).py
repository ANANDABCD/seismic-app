
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load models
reg_model = joblib.load("regression_model.pkl")
cls_model = joblib.load("classification_model.pkl")

st.set_page_config(page_title="Seismic Response Predictor", layout="wide")
st.title("🏗️ Seismic Response Predictor – IS 1893 Based")

st.markdown("Provide building details to get predictions for base shear, displacements, drift, compliance, and more.")

# Input form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        btype = st.selectbox("Building Type", ["Residential", "Commercial", "Office"])
        cgrade = st.selectbox("Concrete Grade", ["M20", "M25"])
    with col2:
        floors = st.number_input("No. of Floors", min_value=1, max_value=50, value=5)
        sgrade = st.selectbox("Steel Grade", ["Fe415", "Fe500", "Fe550"])
    with col3:
        height = st.number_input("Total Height (m)", min_value=3.0, max_value=100.0, value=15.0)
        aspect = st.number_input("Plan Aspect Ratio", min_value=0.5, max_value=3.0, value=1.5)

    dead = st.slider("Dead Load (kN/m²)", 2.0, 10.0, 5.0)
    live = st.slider("Live Load (kN/m²)", 1.0, 5.0, 3.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Building Type": btype,
        "No. of Floors": floors,
        "Total Height (m)": height,
        "Plan Aspect Ratio": aspect,
        "Concrete Grade": cgrade,
        "Steel Grade": sgrade,
        "Dead Load (kN/m²)": dead,
        "Live Load (kN/m²)": live
    }])

    # Regression Predictions
    reg_output = reg_model.predict(input_df)[0]
    base_shear, roof_disp, max_drift = reg_output
    drift_ratio = max_drift / (height if height else 1)
    time_period = 0.075 * (height ** 0.75)

    # Classification
    cls_output = cls_model.predict(input_df)[0]
    risk = min(100, drift_ratio / 0.004 * 100)
    compliant = "✅ Yes" if cls_output == "Pass" else "❌ No"

    # Display
    st.subheader("📊 Predicted Outputs")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Base Shear (kN)", f"{base_shear:.2f}")
        st.metric("Roof Displacement (mm)", f"{roof_disp:.2f}")
        st.metric("Max Storey Drift (mm)", f"{max_drift:.3f}")
    with col2:
        st.metric("Interstorey Drift Ratio", f"{drift_ratio:.5f}")
        st.metric("Time Period (s)", f"{time_period:.2f}")
        st.metric("IS 1893 Compliant?", compliant)

    st.subheader("📉 Interstorey Drift Plot")
    drifts = np.linspace(drift_ratio * 0.5, drift_ratio, int(floors))
    plt.figure(figsize=(6, 3))
    plt.plot(range(1, int(floors)+1), drifts, marker='o')
    plt.xlabel("Storey")
    plt.ylabel("Drift Ratio")
    plt.grid(True)
    st.pyplot(plt)

    st.subheader("📟 Risk of Failure Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 60], 'color': "green"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 90}
        },
        title={'text': "Risk of Failure (%)"}
    ))
    st.plotly_chart(fig)
