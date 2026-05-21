import streamlit as st
import requests

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- TITLE ---------------- #

st.title("🛡️ Hate Speech Detection System")

st.markdown(
    "Detect hate speech, offensive language, severity level, and generate neutralized text using ML + GenAI."
)

# ---------------- INPUT ---------------- #

text = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Type something here..."
)

# ---------------- BUTTON ---------------- #

if st.button("Analyze Text"):

    if text.strip() == "":

        st.warning("Please enter some text.")

    else:

        url = "http://127.0.0.1:8000/predict"

        payload = {
            "text": text
        }

        with st.spinner("Analyzing..."):

            response = requests.post(url, json=payload)

            result = response.json()

        # ---------------- RESULTS ---------------- #

        st.divider()

        st.subheader("📊 Prediction Result")

        prediction = result.get("prediction")
        confidence = result.get("confidence")
        severity = result.get("severity")
        toxic_words = result.get("toxic_words")
        explanation = result.get("explanation")
        neutralized = result.get("neutralized_text")

        # prediction color
        if prediction == "Hate Speech":
            st.error(f"Prediction: {prediction}")

        elif prediction == "Offensive Language":
            st.warning(f"Prediction: {prediction}")

        else:
            st.success(f"Prediction: {prediction}")

        # metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Confidence", f"{confidence}%")

        with col2:
            st.metric("Severity", severity)

        # toxic words
        st.subheader("🚨 Toxic Words")

        if toxic_words:
            st.write(", ".join(toxic_words))
        else:
            st.write("No toxic words detected.")

        # explanation
        st.subheader("🧠 Explanation")

        st.info(explanation)

        # neutralized text
        if neutralized:

            st.subheader("🤖 Neutralized Text")

            st.success(neutralized)