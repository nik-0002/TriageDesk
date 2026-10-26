import os

import pandas as pd
import streamlit as st

from analyzer import DEPARTMENTS, URGENCIES, analyze_all
from data import COMPLAINTS

st.set_page_config(page_title="Grievance Analyzer", layout="wide")
st.title("AI Grievance & Service Request Analyzer")
st.info("AI recommends, a human decides. Synthetic data only.")

if "texts" not in st.session_state:
    st.session_state.texts = list(COMPLAINTS)
    st.session_state.decisions = {}

with st.sidebar:
    st.caption("Mode: " + ("Claude AI" if os.environ.get("ANTHROPIC_API_KEY") else "Keyword fallback (no API key)"))
    new = st.text_area("Submit a new complaint")
    if st.button("Submit") and new.strip():
        st.session_state.texts.append(new.strip())
    if st.button("Reset to synthetic dataset"):
        st.session_state.texts = list(COMPLAINTS)
        st.session_state.decisions = {}


@st.cache_data(show_spinner="Analyzing complaints...")
def run(texts):
    return pd.DataFrame(analyze_all(list(texts)))


df = run(tuple(st.session_state.texts))
df["rank"] = df["urgency"].map({u: i for i, u in enumerate(URGENCIES)})
df = df.sort_values(["rank", "id"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", len(df))
c2.metric("Critical", int((df.urgency == "Critical").sum()))
c3.metric("Duplicate clusters", int(df.cluster[df.cluster > 0].nunique()))
c4.metric("Reviewed", len(st.session_state.decisions))

COLORS = {"Critical": "#d9534f", "High": "#f0ad4e", "Medium": "#5bc0de", "Low": "#5cb85c"}
st.subheader("Priority queue")
st.dataframe(
    df[["id", "urgency", "category", "department", "next_action", "cluster", "confidence", "text"]]
    .style.map(lambda u: f"background-color:{COLORS.get(u, '')};color:white", subset=["urgency"]),
    width="stretch", hide_index=True,
)

st.subheader("Duplicate / related complaints")
for cid, g in df[df.cluster > 0].groupby("cluster"):
    with st.expander(f"Cluster {cid}: {len(g)} complaints, {g.iloc[0].category}"):
        for _, r in g.iterrows():
            st.write(f"#{r.id}: {r.text}")

st.subheader("Human review")
for _, r in df.iterrows():
    done = st.session_state.decisions.get(r.id)
    label = f"#{r.id} [{r.urgency}] {r.text[:70]}" + (f" ✅ {done['decision']}" if done else "")
    with st.expander(label):
        st.write(f"**AI suggestion:** {r.department} / {r.urgency}: {r.next_action}")
        st.caption(f"Reason: {r.reason} (confidence {r.confidence}, source: {r.source})")
        a, b = st.columns(2)
        dept = a.selectbox("Department", DEPARTMENTS, DEPARTMENTS.index(r.department), key=f"d{r.id}")
        urg = b.selectbox("Urgency", URGENCIES, URGENCIES.index(r.urgency), key=f"u{r.id}")
        x, y, z = st.columns(3)
        for col, name in ((x, "Approved"), (y, "Overridden"), (z, "Rejected")):
            if col.button(name[:-1] if name != "Overridden" else "Override", key=f"{name}{r.id}"):
                st.session_state.decisions[r.id] = {
                    "id": r.id, "text": r.text, "ai_department": r.department, "ai_urgency": r.urgency,
                    "final_department": dept, "final_urgency": urg, "decision": name,
                }
                st.rerun()

if st.session_state.decisions:
    out = pd.DataFrame(st.session_state.decisions.values())
    st.download_button("Download reviewed queue (CSV)", out.to_csv(index=False), "reviewed.csv")
