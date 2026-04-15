import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ====================
# Page Config
# ====================

st.set_page_config(page_title="PCOS Classifier", page_icon="🧬", layout="wide")

# ---- bigger base font ----

st.markdown("""
<style>
p, li, td, th, div[data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem !important;
    line-height: 1.7 !important;
}
</style>
""", unsafe_allow_html=True)

# ====================
# Sidebar Navigation
# ====================

with st.sidebar:
    st.markdown("## 🧬 PCOS Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Project Overview", "📊 Interactive Results"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Gabriela Nirmal · Udita Shah · Praveen Sinha")


# ====================
# Page 1 - Overview
# ====================

if page == "🏠 Project Overview":

    st.title("Predicting PCOS with Attention-Weighted Neural Networks")

    st.markdown("""
**Polycystic Ovary Syndrome (PCOS)** is one of the most common hormonal disorders
among women of reproductive age, affecting **6–10% of this population worldwide**.
Diagnosis requires at least two of three symptoms under the Rotterdam criteria:
irregular ovulation, elevated male hormone levels, and cysts visible on ultrasound.

Despite its prevalence, up to **70% of cases go undiagnosed** because symptoms vary
widely and overlap with other conditions. Left untreated, PCOS significantly increases
the risk of type 2 diabetes, heart disease, and infertility.
""")

    st.markdown("---")

    # ---- Why ML ----

    st.subheader("Why Machine Learning?")

    st.markdown("""
Machine learning offers a promising path toward faster and more consistent diagnosis.
Prior work has shown that clinical features such as **follicle counts, LH/FSH ratio,
and AMH levels** carry strong predictive signal, with models consistently achieving
**80–95% accuracy**.

This project applies three ML methods to a publicly available dataset of 541 PCOS
patients, comparing each model's performance and interpretability on the same data:

| Model | Description |
|---|---|
| **MLP with Attention** | A manually implemented multi-layer perceptron with a custom attention layer that learns which features matter most |
| **1D CNN** | A convolutional network designed to prioritize clinically established PCOS markers by grouping follicle counts and hormone levels together in the input sequence |
| **Bayesian Logistic Regression** | A probabilistic model that provides uncertainty estimates alongside predictions |
""")

    st.markdown("---")

    # ---- Dataset + Pie Chart ----

    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        st.subheader("The Dataset")
        st.markdown("""
The model is trained on the publicly available **PCOS Dataset** (Kaggle), which
contains **541 patient records** from fertility clinics in Kerala, India.
Each record includes 41 clinical and lifestyle features:

- **Hormonal markers** — FSH, LH, AMH, TSH, Prolactin, Progesterone
- **Physical measurements** — BMI, waist-hip ratio, blood pressure
- **Reproductive indicators** — follicle counts, cycle regularity, pregnancy history
- **Lifestyle factors** — fast food consumption, exercise frequency

The target variable is a binary PCOS diagnosis (Yes / No).
""")

    with col_r:
        st.subheader("Class Balance")
        fig, ax = plt.subplots(figsize=(3, 3))
        wedges, texts, autotexts = ax.pie(
            [364, 177],
            labels=["No PCOS", "PCOS"],
            autopct="%1.0f%%",
            startangle=90,
            colors=["#c8b8f0", "#7c5cbf"],
            wedgeprops=dict(edgecolor="white", linewidth=2),
            textprops=dict(fontsize=11),
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
        ax.set_title("Dataset Composition (n=541)", fontsize=11, pad=8)
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # ---- Model Architecture ----

    st.subheader("MLP Model Architecture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
**Step 1 — Preprocessing**
- Rename & drop extraneous columns
- Coerce all values to numeric
- Impute nulls with column medians
- Min-Max scale to [0, 1]
""")

    with col2:
        st.markdown("""
**Step 2 — Attention Layer**
A custom `FeatureAttention` layer learns a softmax-normalised weight for each
of the 41 input features, amplifying the most diagnostically relevant signals
before passing data to the dense layers.
""")

    with col3:
        st.markdown("""
**Step 3 — MLP Classifier**
```
Input (41)
  → FeatureAttention
  → Dense(64, ReLU) + Dropout(0.3)
  → Dense(32, ReLU) + Dropout(0.2)
  → Dense(1, Sigmoid)
```
Trained with *Adam* + *binary cross-entropy*;
early stopping on validation loss (patience = 40).
""")

    st.markdown("---")

    # ---- Key Results ----

    st.subheader("Key Results")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Test Accuracy",   "89.9%")
    r2.metric("Test Loss",       "0.2475")
    r3.metric("AUC-ROC",         "0.9509")
    r4.metric("F1 (PCOS class)", "0.81")

    st.markdown(
        "> The model correctly identifies **97% of true negatives** (No PCOS) and "
        "**72% of true positives** (PCOS), with an AUC-ROC of **0.95** — "
        "indicating strong discriminative ability even on imbalanced data."
    )


# ====================
# Page 2 - Interactive
# ====================

else:

    st.title("Try the Model")
    st.markdown("""
Enter values for the most important clinical features identified by the MLP model.
All other features will be filled with dataset median values. Click **Predict** to
see the model's PCOS diagnosis.
""")

    st.markdown("---")

    # ---- Load model (lazy) ----

    @st.cache_resource
    def load_model():
        import tensorflow as tf
        from keras import layers
        import keras

        class FeatureAttention(layers.Layer):
            def __init__(self, input_dim, **kwargs):
                super().__init__(**kwargs)
                self.attention_weights = self.add_weight(
                    shape=(input_dim,),
                    initializer="ones",
                    trainable=True,
                    name="attention_weights",
                )
            def call(self, x):
                return x * tf.nn.softmax(self.attention_weights)
            def get_config(self):
                config = super().get_config()
                config.update({"input_dim": self.attention_weights.shape[0]})
                return config

        import os
        model_path = os.path.join(os.path.dirname(__file__), "pcos_mlp_model.keras")
        model = keras.models.load_model(
            model_path,
            custom_objects={"FeatureAttention": FeatureAttention},
        )
        return model

    # ---- Dataset medians (pre-computed from training data) ----
    # These are the median values for all 41 features used to fill
    # in anything the user doesn't enter directly.

    MEDIANS = {
        "Age (yrs)": 28.0,
        "Weight (Kg)": 58.0,
        "Height(Cm)": 160.0,
        "BMI": 22.6,
        "Blood Group": 11.0,
        "Pulse rate(bpm) ": 74.0,
        "RR (breaths/min)": 18.0,
        "Hb(g/dl)": 12.2,
        "Cycle(R/I)": 2.0,
        "Cycle length(days)": 4.0,
        "Marriage Status (Yrs)": 3.0,
        "Pregnant(Y/N)": 0.0,
        "No. of abortions": 0.0,
        "beta-HCG(mIU/mL) 1": 1.0,
        "beta-HCG(mIU/mL) 2": 1.0,
        "FSH(mIU/mL)": 6.3,
        "LH(mIU/mL)": 5.2,
        "FSH/LH": 1.17,
        "Hip(inch)": 36.0,
        "Waist(inch)": 30.0,
        "Waist:Hip Ratio": 0.83,
        "TSH (mIU/L)": 2.17,
        "AMH(ng/mL)": 3.39,
        "PRL(ng/mL)": 14.3,
        "Vit D3 (ng/mL)": 18.4,
        "PRG(ng/mL)": 0.7,
        "RBS(mg/dl)": 90.0,
        "Weight gain(Y/N)": 0.0,
        "hair growth(Y/N)": 0.0,
        "Skin darkening (Y/N)": 0.0,
        "Hair loss(Y/N)": 0.0,
        "Pimples(Y/N)": 0.0,
        "Fast food (Y/N)": 1.0,
        "Reg.Exercise(Y/N)": 0.0,
        "BP _Systolic (mmHg)": 110.0,
        "BP _Diastolic (mmHg)": 70.0,
        "Follicle No. (L)": 5.0,
        "Follicle No. (R)": 5.0,
        "Avg. F size (L) (mm)": 18.0,
        "Avg. F size (R) (mm)": 18.0,
        "Endometrium (mm)": 8.5,
    }

    # Min/Max for scaling (pre-computed from training data)
    MINS = {
        "Age (yrs)": 19.0, "Weight (Kg)": 29.9, "Height(Cm)": 130.0,
        "BMI": 13.0, "Blood Group": 11.0, "Pulse rate(bpm) ": 60.0,
        "RR (breaths/min)": 12.0, "Hb(g/dl)": 6.9, "Cycle(R/I)": 2.0,
        "Cycle length(days)": 2.0, "Marriage Status (Yrs)": 0.0,
        "Pregnant(Y/N)": 0.0, "No. of abortions": 0.0,
        "beta-HCG(mIU/mL) 1": 0.1, "beta-HCG(mIU/mL) 2": 0.1,
        "FSH(mIU/mL)": 1.25, "LH(mIU/mL)": 0.17, "FSH/LH": 0.07,
        "Hip(inch)": 28.0, "Waist(inch)": 23.0, "Waist:Hip Ratio": 0.65,
        "TSH (mIU/L)": 0.01, "AMH(ng/mL)": 0.1, "PRL(ng/mL)": 2.5,
        "Vit D3 (ng/mL)": 2.23, "PRG(ng/mL)": 0.1, "RBS(mg/dl)": 68.0,
        "Weight gain(Y/N)": 0.0, "hair growth(Y/N)": 0.0,
        "Skin darkening (Y/N)": 0.0, "Hair loss(Y/N)": 0.0,
        "Pimples(Y/N)": 0.0, "Fast food (Y/N)": 0.0,
        "Reg.Exercise(Y/N)": 0.0, "BP _Systolic (mmHg)": 90.0,
        "BP _Diastolic (mmHg)": 60.0, "Follicle No. (L)": 1.0,
        "Follicle No. (R)": 1.0, "Avg. F size (L) (mm)": 6.0,
        "Avg. F size (R) (mm)": 6.0, "Endometrium (mm)": 3.0,
    }

    MAXS = {
        "Age (yrs)": 48.0, "Weight (Kg)": 120.0, "Height(Cm)": 186.0,
        "BMI": 60.0, "Blood Group": 15.0, "Pulse rate(bpm) ": 105.0,
        "RR (breaths/min)": 24.0, "Hb(g/dl)": 16.3, "Cycle(R/I)": 5.0,
        "Cycle length(days)": 14.0, "Marriage Status (Yrs)": 18.0,
        "Pregnant(Y/N)": 1.0, "No. of abortions": 5.0,
        "beta-HCG(mIU/mL) 1": 2200.0, "beta-HCG(mIU/mL) 2": 2200.0,
        "FSH(mIU/mL)": 22.74, "LH(mIU/mL)": 48.27, "FSH/LH": 15.6,
        "Hip(inch)": 52.0, "Waist(inch)": 52.0, "Waist:Hip Ratio": 1.37,
        "TSH (mIU/L)": 10.0, "AMH(ng/mL)": 19.2, "PRL(ng/mL)": 82.0,
        "Vit D3 (ng/mL)": 74.0, "PRG(ng/mL)": 14.0, "RBS(mg/dl)": 230.0,
        "Weight gain(Y/N)": 1.0, "hair growth(Y/N)": 1.0,
        "Skin darkening (Y/N)": 1.0, "Hair loss(Y/N)": 1.0,
        "Pimples(Y/N)": 1.0, "Fast food (Y/N)": 1.0,
        "Reg.Exercise(Y/N)": 1.0, "BP _Systolic (mmHg)": 140.0,
        "BP _Diastolic (mmHg)": 90.0, "Follicle No. (L)": 22.0,
        "Follicle No. (R)": 23.0, "Avg. F size (L) (mm)": 30.0,
        "Avg. F size (R) (mm)": 30.0, "Endometrium (mm)": 18.0,
    }

    FEATURE_ORDER = list(MEDIANS.keys())

    def scale_value(val, feature):
        mn = MINS[feature]
        mx = MAXS[feature]
        if mx == mn:
            return 0.0
        return (val - mn) / (mx - mn)

    # ---- Input Form ----

    st.subheader("Enter Patient Values")
    st.markdown("Adjust the sliders for the key features. Everything else uses typical median values from the dataset.")

    col1, col2 = st.columns(2)

    # ---- PCOS default note ----
    st.info("💡 Default values are set to a **typical PCOS-positive profile** — high follicle counts, irregular cycle, elevated AMH and LH. Hit Predict to see the result, then adjust values to explore.")

    with col1:
        st.markdown("**Follicle Counts**")
        follicle_r = st.slider("Follicle No. (R) — right ovary",    1, 23, 14)
        follicle_l = st.slider("Follicle No. (L) — left ovary",     1, 22, 13)

        st.markdown("**Cycle**")
        cycle = st.selectbox("Cycle (R=Regular / I=Irregular)", options=[2, 5], index=1, format_func=lambda x: "Regular (R)" if x == 2 else "Irregular (I)")

        st.markdown("**Symptoms**")
        hair_growth = st.selectbox("Excess hair growth (Y/N)",       [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
        weight_gain = st.selectbox("Unexplained weight gain (Y/N)",  [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.markdown("**Hormones**")
        amh  = st.slider("AMH (ng/mL)",    0.1, 19.2, 8.5)
        lh   = st.slider("LH (mIU/mL)",    0.2, 48.3, 12.0)
        fsh  = st.slider("FSH (mIU/mL)",   1.3, 22.7, 5.0)

        st.markdown("**Follicle Size**")
        avg_f_r = st.slider("Avg. Follicle Size Right (mm)", 6, 30, 20)
        avg_f_l = st.slider("Avg. Follicle Size Left (mm)",  6, 30, 20)

    st.markdown("---")

    # ---- Predict ----

    if st.button("🔍 Predict"):

        with st.spinner("Loading model…"):
            model = load_model()

        # Build input row from medians, then override with user values
        row = dict(MEDIANS)
        row["Follicle No. (R)"]       = float(follicle_r)
        row["Follicle No. (L)"]       = float(follicle_l)
        row["Cycle(R/I)"]             = float(cycle)
        row["hair growth(Y/N)"]       = float(hair_growth)
        row["Weight gain(Y/N)"]       = float(weight_gain)
        row["AMH(ng/mL)"]             = float(amh)
        row["LH(mIU/mL)"]             = float(lh)
        row["FSH(mIU/mL)"]            = float(fsh)
        row["Avg. F size (R) (mm)"]   = float(avg_f_r)
        row["Avg. F size (L) (mm)"]   = float(avg_f_l)

        # Scale each feature
        scaled = np.array([scale_value(row[f], f) for f in FEATURE_ORDER], dtype=np.float32)
        scaled = scaled.reshape(1, -1)

        prob = float(model.predict(scaled, verbose=0)[0][0])
        label = "PCOS Detected" if prob >= 0.5 else "No PCOS Detected"
        color = "#7c5cbf" if prob >= 0.5 else "#2ecc71"

        st.markdown("---")
        st.subheader("Prediction Result")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(
                f"<div style='background:{color};padding:24px;border-radius:12px;text-align:center;'>"
                f"<span style='color:white;font-size:1.5rem;font-weight:700;'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confidence:** {prob*100:.1f}% probability of PCOS")

        with res_col2:
            # Confidence bar
            fig, ax = plt.subplots(figsize=(5, 1.2))
            ax.barh([""], [prob],        color="#7c5cbf", height=0.4)
            ax.barh([""], [1 - prob],    color="#e8e0f5", height=0.4, left=[prob])
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_xlabel("Probability")
            ax.set_title("PCOS Probability", fontsize=10)
            ax.spines[["top", "right", "left"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Values Used for Prediction")
        input_df = pd.DataFrame({
            "Feature": ["Follicle No. (R)", "Follicle No. (L)", "Cycle", "Hair Growth",
                        "Weight Gain", "AMH", "LH", "FSH", "Avg. F Size (R)", "Avg. F Size (L)"],
            "Value Entered": [follicle_r, follicle_l,
                              "Irregular" if cycle == 5 else "Regular",
                              "Yes" if hair_growth else "No",
                              "Yes" if weight_gain else "No",
                              amh, lh, fsh, avg_f_r, avg_f_l],
        })
        st.dataframe(input_df, use_container_width=True)
        st.caption("All other features were filled with dataset median values.")