from matplotlib.axis import Tick
import streamlit as st
import pandas as pd

import joblib
import numpy as np

model = joblib.load("/Users/jein/env/env/logistic_job_acceptance_model.pkl")
scaler = joblib.load("/Users/jein/env/env/scaler.pkl")
feature_columns = joblib.load("/Users/jein/env/env/feature_columns.pkl")


st.title("Job Acceptance Prediction Dashboard")
st.subheader("Enter Candidate Details")

# ---------- Session storage ----------
if "records" not in st.session_state:
    st.session_state.records = []

# ---------- Candidate Input Layout ----------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 60, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc = st.slider("SSC %", 0, 100, 70)
    degree = st.slider("Degree %", 0, 100, 75)

with col2:
    tech = st.slider("Technical Score", 0, 100, 70)
    apt = st.slider("Aptitude Score", 0, 100, 70)
    comm = st.slider("Communication Score", 0, 100, 70)
    skills = st.slider("Skills Match %", 0, 100, 75)

with col3:
    certs = st.number_input("Certifications", 0, 20, 2)
    experience = st.slider("Experience (years)", 0, 10, 2)
    gap_years = st.number_input("Employment Gap (years)", 0.0, 5.0, 0.0)

gap_months = gap_years * 12

internship = st.selectbox("Internship Experience", ["Yes", "No"])
career_switch = st.selectbox("Career Switch?", ["Yes", "No"])
relevant = st.selectbox("Relevant Experience?", ["Yes", "No"])
bond = st.selectbox("Bond Requirement", ["Yes", "No"])
layoff = st.selectbox("Layoff History", ["Yes", "No"])
relocation = st.selectbox("Relocation?", ["Yes", "No"])

specialization = st.selectbox("Degree Specialization",
                              ["IT", "CSE", "ECE", "Mechanical", "Other"])
job_role = st.selectbox("Job Role Match", ["High", "Medium", "Low"])
company = st.selectbox("Company Tier", ["Tier1", "Tier2", "Tier3"])
competition = st.selectbox("Competition Level", ["High", "Medium", "Low"])

prev_ctc = st.number_input("Previous CTC", 0.0, 50.0, 3.0)
exp_ctc = st.number_input("Expected CTC", 0.0, 50.0, 5.0)
notice = st.slider("Notice Period", 0, 120, 30)

# ---------- Prediction ----------
if st.button("Predict & Add Candidate"):

    yes_no = lambda x: 1 if x == "Yes" else 0

    data = {
        "age_years": age,
        "gender": 1 if gender == "Male" else 0,
        "ssc_percentage": ssc,
        "degree_percentage": degree,
        "technical_score": tech,
        "aptitude_score": apt,
        "communication_score": comm,
        "skills_match_percentage": skills,
        "certifications_count": certs,
        "internship_experience": yes_no(internship),
        "years_of_experience": experience,
        "career_switch_willingness": yes_no(career_switch),
        "relevant_experience": yes_no(relevant),
        "previous_ctc_lpa": prev_ctc,
        "expected_ctc_lpa": exp_ctc,
        "bond_requirement": yes_no(bond),
        "notice_period_days": notice,
        "layoff_history": yes_no(layoff),
        "employment_gap_months": gap_months,
        "relocation_willingness": yes_no(relocation),
        "degree_specialization": specialization,
        "job_role_match": job_role,
        "company_tier": company,
        "competition_level": competition
    }

    df = pd.DataFrame([data])

    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(df_encoded)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    data["prediction"] = pred
    data["confidence"] = prob

    st.session_state.records.append(data)

    if pred == 1:
        st.success(f"✅ Likely ACCEPT ({prob:.2f})")
    else:
        st.error(f"❌ Likely REJECT ({1-prob:.2f})")

# ---------- KPI Dashboard ----------
if st.session_state.records:

    df_hist = pd.DataFrame(st.session_state.records)

    total = len(df_hist)
    accept_rate = (df_hist["prediction"] == 1).mean() * 100
    dropout_rate = (df_hist["prediction"] == 0).mean() * 100
    avg_interview = df_hist[
        ["technical_score", "aptitude_score", "communication_score"]
    ].mean().mean()
    avg_skills = df_hist["skills_match_percentage"].mean()
    high_risk = dropout_rate

    st.subheader("Recruitment KPIs")

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric("Placement Rate %", f"{accept_rate:.1f}")
    c2.metric("Job Acceptance Rate %", f"{accept_rate:.1f}")
    c3.metric("Offer Dropout Rate %", f"{dropout_rate:.1f}")

    c4.metric("Avg Interview Score %", f"{avg_interview:.1f}")
    c5.metric("Avg Skills Match %", f"{avg_skills:.1f}")
    c6.metric("High Risk Candidate %", f"{high_risk:.1f}")

    st.subheader("Candidate History")
    st.dataframe(df_hist)
