import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit config
st.set_page_config(page_title="Lung Cancer Survival Predictor", page_icon="🩺", layout="centered")

# Title & subtitle
st.title("Lung Cancer Survival Predictor")
st.markdown("Provide the patient details below to check their **survival prediction**.")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Tabs
tab1, tab2 = st.tabs(["Predict", "History"])

with tab1:
    st.subheader("Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 10, 100, 60)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        country = st.selectbox("Country", ['India', 'USA', 'UK', 'Other'])
        cancer_stage = st.selectbox("Cancer Stage", ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])
        family_history = st.selectbox("Family History of Cancer", ['Yes', 'No'])
        smoking_status = st.selectbox("Smoking Status", ['Never Smoked', 'Former Smoker', 'Current Smoker', 'Passive Smoker'])

    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
        cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
        hypertension = st.selectbox("Hypertension", ['Yes', 'No'])
        asthma = st.selectbox("Asthma", ['Yes', 'No'])
        cirrhosis = st.selectbox("Cirrhosis", ['Yes', 'No'])
        other_cancer = st.selectbox("Other Cancer History", ['Yes', 'No'])
        treatment_type = st.selectbox("Treatment Type", ['Surgery', 'Chemotherapy', 'Radiation', 'Combined'])

    # Encoding maps
    gender_map = {'Male': 1, 'Female': 0}
    country_map = {'India': 0, 'USA': 1, 'UK': 2, 'Other': 3}
    stage_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
    bool_map = {'Yes': 1, 'No': 0}
    smoke_map = {'Never Smoked': 0, 'Former Smoker': 1, 'Current Smoker': 2, 'Passive Smoker': 3}
    treat_map = {'Surgery': 0, 'Chemotherapy': 1, 'Radiation': 2, 'Combined': 3}

    # Input vector
    input_data = np.array([[
        age,
        gender_map[gender],
        country_map[country],
        stage_map[cancer_stage],
        bool_map[family_history],
        smoke_map[smoking_status],
        bmi,
        cholesterol,
        bool_map[hypertension],
        bool_map[asthma],
        bool_map[cirrhosis],
        bool_map[other_cancer],
        treat_map[treatment_type]
    ]])

    if st.button("Predict Survival"):
        # Progress bar
        progress_text = "Running prediction..."
        progress_bar = st.progress(0, text=progress_text)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1, text=progress_text)

        # Scale & predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        confidence = model.predict_proba(scaled_input)[0][prediction] * 100

        # Display result
        if prediction == 1:
            st.success(f"The patient is **likely to SURVIVE**.\n\nConfidence: **{confidence:.2f}%**")
        else:
            st.error(f"The patient is **unlikely to survive**.\n\nConfidence: **{confidence:.2f}%**")

        # Save to history
        result = {
            "Age": age,
            "Gender": gender,
            "Country": country,
            "Stage": cancer_stage,
            "Smoking": smoking_status,
            "Treatment": treatment_type,
            "Prediction": "Survived" if prediction == 1 else "Not Survived",
            "Confidence (%)": f"{confidence:.2f}"
        }
        st.session_state.history.append(result)

with tab2:
    if st.session_state.history:
        st.markdown("###Prediction History")
        df_history = pd.DataFrame(st.session_state.history).iloc[::-1]
        st.dataframe(df_history, use_container_width=True)

        csv = df_history.to_csv(index=False).encode("utf-8")
        st.download_button("Download as CSV", csv, "survival_predictions.csv", "text/csv")

        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared!")