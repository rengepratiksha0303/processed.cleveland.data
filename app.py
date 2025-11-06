import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open("NB_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.set_page_config(page_title="Heart Disease Prediction App", page_icon="💓", layout="wide")

st.title("💓 Heart Disease Prediction using Machine Learning")
st.write("This app predicts whether a person is likely to have heart disease based on medical attributes.")

# Sidebar input fields
st.sidebar.header("Enter Patient Details")

def user_input_features():
    age = st.sidebar.number_input("Age", 20, 100, 40)
    sex = st.sidebar.selectbox("Sex (1 = male, 0 = female)", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)", [1, 0])
    restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [1, 0])
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display input data
st.subheader("Patient Details")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error("⚠ The model predicts that this patient is likely to have heart disease.")
else:
    st.success("✅ The model predicts that this patient is not likely to have heart disease.")

# Probability chart
# Probability chart
st.subheader("Prediction Probability")

labels = ["No Disease", "Disease"]

# Flatten the probabilities properly
prob_values = np.ravel(prediction_proba)[0:2]  # ensures exactly 2 values

# Make sure lengths match
if len(prob_values) == len(labels):
    prob_df = pd.DataFrame({"Condition": labels, "Probability": prob_values})
    fig, ax = plt.subplots()
    sns.barplot(x="Condition", y="Probability", data=prob_df, ax=ax)
    ax.set_ylim(0, 1)
    st.pyplot(fig)
else:
    st.warning("Model output shape mismatch — probability chart not displayed.")



st.markdown("---")
st.caption("Built with ❤ using Streamlit and Naive Bayes Model (NB_model.pkl)")
