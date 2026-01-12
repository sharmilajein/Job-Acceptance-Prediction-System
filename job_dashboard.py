
import streamlit as st
import pandas as pd

# Basic counts

df_new = pd.read_csv("/Users/jein/env/env/job_acceptance_model_data.csv")
df_new.columns = df_new.columns.str.strip()
total_candidates = len(df_new)

placed = df_new[df_new['status'] == 1]        # 1 = Placed 
not_placed = df_new[df_new['status'] == 0]    # 0 = Not Placed 

# Rates
placement_rate = (len(placed) / total_candidates) * 100
job_acceptance_rate = (len(placed) / total_candidates) * 100
offer_dropout_rate = (len(not_placed) / total_candidates) * 100

# Interview & skills KPIs
avg_interview_score = (
    df_new['technical_score'] +
    df_new['aptitude_score'] +
    df_new['communication_score']
).mean() / 3

avg_skills_match = df_new['skills_match_percentage'].mean()


# High-risk candidates 
# predicted_status: 1 = likely accept, 0 = high risk / likely reject

df_new['predicted_label'] = df_new['predicted_status'].map({
    1: 'Likely Accept',
    0: 'High Risk Reject'
})
high_risk_pct = (df_new['predicted_status'] == 0).mean() * 100


print(df_new['predicted_status'].value_counts())

# Streamlit layout--->


st.subheader("📌 Recruitment Performance KPIs")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Candidates", total_candidates)
col2.metric("Placement Rate (%)", f"{placement_rate:.2f}")
col3.metric("Job Acceptance Rate (%)", f"{job_acceptance_rate:.2f}")
col4.metric("Offer Dropout Rate (%)", f"{offer_dropout_rate:.2f}")

col5, col6, col7 = st.columns(3)
col5.metric("Avg Interview Score", f"{avg_interview_score:.2f}")
col6.metric("Avg Skills Match (%)", f"{avg_skills_match:.2f}")
col7.metric("High-Risk Candidates (%)", f"{high_risk_pct:.2f}")





