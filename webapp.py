# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import pandas as pd

# ---------- Load model ----------
MODEL_PATH = "C:/Users/shash/Desktop/Hate Speech Detection/hsd_model.sav"
with open(MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)

positive_words = {"love", "nice", "good", "happy", "great", "friend", "beautiful", "like", "family" , "ladies"}

def custom_predict(text: str):
    base_pred = loaded_model.predict([text])[0]
    proba = loaded_model.predict_proba([text])[0]
    if any(w in text.lower() for w in positive_words):
        not_off_idx = list(loaded_model.classes_).index("Not Offensive")
        if proba[not_off_idx] < 0.6:
            base_pred = "Not Offensive"
    return base_pred, dict(zip(loaded_model.classes_, proba))

# ---------- Page setup ----------
st.set_page_config(page_title="🛡️ Hate Speech Detector", page_icon="🛡️", layout="centered")

# Optional: slightly lighter page bg so the box stands out
st.markdown("""
<style>
/* limit overall content width */
.main > div { max-width: 2600px; margin: 0 auto; }

/* make the bordered container look like a card */
div[data-testid="stVerticalBlock"][data-border="true"]{
    border: 6px solid #ffffff;
    border-radius: 14px;
    background: #3a3a3a;
    padding: 24px;
    font-family: 'Segoe UI', sans-serif; 
}
</style>
""", unsafe_allow_html=True)

# ---------- ONE bordered box with everything inside ----------
col_left, col_center, col_right = st.columns([0.002, 6, 0.002])
with col_center:
    with st.container(border=True):
        st.markdown(
            """
            <h1 style="text-align:center; color:#B3C6E3;">🛡️ Hate Speech Detection</h1>
            <p style="text-align:center; font-size:17px;">
            Enter a phrase and our AI model will classify it as
            <b style="color:#88D180;">Not Offensive</b> or
            <b style="color:#DE2C3C;">Offensive / Hate Speech</b>.
            </p>
            """,
            unsafe_allow_html=True
        )

        user_input = st.text_area("✍️ Enter text:", placeholder="Type something here...", height=140)

        analyze = st.button("🔍 Analyze", use_container_width=True)

        if analyze:
            if user_input.strip():
                prediction, probabilities = custom_predict(user_input)

                st.markdown("### 📊 Prediction Result:")
                if prediction == "Not Offensive":
                    st.success("✅ The text is classified as **Not Offensive**")
                else:
                    st.error("⚠️ The text is classified as **Offensive / Hate Speech**")

                st.markdown("### 🔢 Confidence Scores:")
                prob_df = pd.DataFrame(list(probabilities.items()), columns=["Class", "Confidence"])
                prob_df["Confidence"] = prob_df["Confidence"].round(4)
                st.bar_chart(prob_df.set_index("Class"))

                with st.expander("🔎 Detailed Confidence Values"):
                    st.table(prob_df)
            else:
                st.warning("⚠️ Please enter some text to analyze.")
