import streamlit as st
import requests


# Page Config

st.set_page_config(
    page_title="Student Enrollment Prediction",
    page_icon="🎓",
    layout="centered"
)


# FastAPI URL

API_URL = "http://127.0.0.1:8000"


# Fixed Dropdown Options — Only Main Courses

COURSE_OPTIONS = [
    "Data Science",
    "Web Development",
    "Python",
    "Java",
    "Full Stack",
    "Digital Marketing",
    "Software Testing",
    "Data Analytics",
    "Cyber Security",
    "Android Development",
    "AWS",
    "Salesforce",
    "HR Training",
    "Internship",
    "Placement",
    "Other"
]


# Header with Logo

col1, col2 = st.columns([1, 4])
with col1:
    try:
        st.image("percept logo.png", width=100)
    except:
        st.write("🎓")
with col2:
    st.title("Student Enrollment Prediction")
    st.markdown("**Percept Infosystem Consultancy**")

st.divider()


# Tabs

tab1, tab2 = st.tabs(["Single Prediction", " Batch Prediction"])



# TAB 1 - Single Prediction

with tab1:
    st.subheader("Single Student Prediction")

    # Input method
    input_method = st.radio(
        "Choose input method:",
        ["Select from dropdown", "Type manually"],
        horizontal=True
    )

    if input_method == "Select from dropdown":
        # Fixed clean dropdown — only main courses
        purpose = st.selectbox("Select Course Interest:", COURSE_OPTIONS)
    else:
        purpose = st.text_input(
            "Enter Course Interest:",
            placeholder="e.g. data science, python, web development"
        )

    # Predict Button
    if st.button("🔮 Predict", use_container_width=True):
        if not purpose:
            st.warning("Please enter a course interest!")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"purpose": purpose},
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()

                    st.divider()

                    # Show Result
                    if result["prediction"] == "registered":
                        st.success("✅ Prediction: **REGISTERED**")
                    else:
                        st.error("❌ Prediction: **NOT REGISTERED**")

                    # Show cleaned purpose
                    st.info(f"🧹 Cleaned Input: `{result['cleaned_purpose']}`")

                    # Show probabilities
                    st.subheader("📊 Probability Scores")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            label="✅ Registered",
                            value=f"{result['probability_registered'] * 100:.2f}%"
                        )
                    with col2:
                        st.metric(
                            label="❌ Not Registered",
                            value=f"{result['probability_not_registered'] * 100:.2f}%"
                        )

                    # Progress bar
                    st.progress(result["probability_registered"])

                else:
                    st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API!\n\nMake sure FastAPI is running:\n```\nuvicorn main:app --reload\n```")



# TAB 2 - Batch Prediction

with tab2:
    st.subheader("Batch Prediction")
    st.markdown("Enter multiple course interests — **one per line:**")

    batch_input = st.text_area(
        "Course Interests:",
        placeholder="data science\npython\nweb development\njava",
        height=150
    )

    if st.button("🔮 Predict All", use_container_width=True):
        if not batch_input.strip():
            st.warning("Please enter at least one course interest!")
        else:
            purposes = [p.strip() for p in batch_input.strip().split("\n") if p.strip()]

            try:
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    json={"students": [{"purpose": p} for p in purposes]},
                    timeout=10
                )

                if response.status_code == 200:
                    results = response.json()["results"]

                    st.divider()
                    st.subheader(f"📋 Results for {len(results)} Students")

                    for i, result in enumerate(results, 1):
                        with st.expander(f"Student {i}: {result['purpose']}"):
                            if result["prediction"] == "registered":
                                st.success("✅ Prediction: **REGISTERED**")
                            else:
                                st.error("❌ Prediction: **NOT REGISTERED**")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("✅ Registered", f"{result['probability_registered'] * 100:.2f}%")
                            with col2:
                                st.metric("❌ Not Registered", f"{result['probability_not_registered'] * 100:.2f}%")

                            st.caption(f"🧹 Cleaned: `{result['cleaned_purpose']}`")

                else:
                    st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API!\n\nMake sure FastAPI is running:\n```\nuvicorn main:app --reload\n```")



# Sidebar

with st.sidebar:
    try:
        st.image("logo.jpeg", width=150)
    except:
        pass

    st.divider()
    st.title("⚙️ API Status")

    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ API is Running")
            st.success("✅ Model is Loaded")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API is Offline")
        st.code("uvicorn main:app --reload", language="bash")

    st.divider()
    st.markdown("### 📌 Quick Links")
    st.markdown(f"[📖 Swagger UI]({API_URL}/docs)")
    st.markdown(f"[❤️ Health Check]({API_URL}/health)")

    st.divider()
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. Select course interest
    2. Click Predict
    3. ML Model predicts
    4. Result shown here
    """)