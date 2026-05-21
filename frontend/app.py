import streamlit as st
import requests

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- BACKEND URL ---------------- #

API_URL = "https://hate-speech-5.onrender.com/predict"

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

    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    payload = {"text": text}

    try:
        with st.spinner("Analyzing..."):

            response = requests.post(API_URL, json=payload, timeout=10)
            result = response.json()

    except Exception:
        st.error("❌ Backend is not reachable. Try again later.")
        st.stop()

    # ---------------- RESULTS ---------------- #

    st.divider()
    st.subheader("📊 Prediction Result")

    prediction = result.get("prediction", "N/A")
    confidence = result.get("confidence", 0)
    severity = result.get("severity", "N/A")
    toxic_words = result.get("toxic_words", [])
    explanation = result.get("explanation", "N/A")
    neutralized = result.get("neutralized_text")

    # Prediction display
    if prediction == "Hate Speech":
        st.error(f"Prediction: {prediction}")
    elif prediction == "Offensive Language":
        st.warning(f"Prediction: {prediction}")
    else:
        st.success(f"Prediction: {prediction}")

    # Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Confidence", f"{confidence}%")

    with col2:
        st.metric("Severity", severity)

    # Toxic words
    st.subheader("🚨 Toxic Words")

    if toxic_words:
        st.write(", ".join(toxic_words))
    else:
        st.write("No toxic words detected.")

    # Explanation
    st.subheader("🧠 Explanation")
    st.info(explanation)

    # Neutralized text
    if neutralized:
        st.subheader("🤖 Neutralized Text")
        st.success(neutralized)
