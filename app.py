import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load model
with open("stacking_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# EXACT feature list used during training
ALL_COLUMNS = model.named_steps["preprocessor"].transformers_[0][2]

def predict_cancer(
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean,
    smoothness_mean,
    concave_points_mean
):
    # 1️⃣ Empty row with all features
    data = {col: np.nan for col in ALL_COLUMNS}

    # 2️⃣ User provided values
    data["radius_mean"] = radius_mean
    data["texture_mean"] = texture_mean
    data["perimeter_mean"] = perimeter_mean
    data["area_mean"] = area_mean
    data["smoothness_mean"] = smoothness_mean
    data["concave points_mean"] = concave_points_mean

    # 3️⃣ Feature engineering (same as training)
    data["area_perimeter_ratio"] = (
        area_mean / perimeter_mean if perimeter_mean else 0
    )
    data["area_concave_points_ratio"] = (
        area_mean / concave_points_mean if concave_points_mean else 0
    )
    data["texture_smoothness_prod"] = texture_mean * smoothness_mean

    # 4️⃣ Bin features
    data["radius_bin"] = 0 if radius_mean < 12 else 1 if radius_mean < 18 else 2
    data["texture_bin"] = 0 if texture_mean < 15 else 1 if texture_mean < 25 else 2

    # 5️⃣ DataFrame
    input_df = pd.DataFrame([data])

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    result = "Malignant (Cancer Detected)" if pred == 1 else "Benign (No Cancer)"
    return f"Prediction: {result}\nCancer Probability: {prob:.2f}"

inputs = [
    gr.Number(label="Radius Mean"),
    gr.Number(label="Texture Mean"),
    gr.Number(label="Perimeter Mean"),
    gr.Number(label="Area Mean"),
    gr.Number(label="Smoothness Mean"),
    gr.Number(label="Concave Points Mean"),
]

app = gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Prediction System",
    description="Deployment with SAME feature schema as training model"
)

app.launch(share=True)
