import streamlit as st
import pickle

model = pickle.load(open("fake_job_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("🔍 AI-Powered Fake Job Detector")

st.markdown(
    "Detect fraudulent job and internship postings using Machine Learning."
)

job_text = st.text_area(
    "Paste Job Description",
    height=250
)

if st.button("Analyze Job"):

    vector = vectorizer.transform([job_text])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    confidence = max(probability) * 100

    st.subheader("Analysis Result")

    if prediction == 1:
        st.error("⚠️ Fake Job Posting Detected")
    else:
        st.success("✅ Real Job Posting")

   

    st.progress(int(confidence))

    st.info(
        f"Model Confidence: {confidence:.2f}%"
    )

    if confidence >= 80:
        st.error("🔴 High Risk")
    elif confidence >= 60:
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")