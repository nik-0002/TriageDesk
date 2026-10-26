import os

import pandas as pd
import streamlit as st

import providers
from analyzer import DEPARTMENTS, URGENCIES, analyze_all, classify
from data import COMPLAINTS

st.set_page_config(page_title="TriageDesk", layout="wide")
st.title("TriageDesk: AI Grievance & Service Request Analyzer")
st.info("AI recommends, a human decides. Synthetic data only.")

if "texts" not in st.session_state:
    st.session_state.texts = list(COMPLAINTS)
    st.session_state.decisions = {}

names = list(providers.PROVIDERS)
ready = {n: providers.available(n) for n in names}

with st.sidebar:
    st.header("AI model")
    provider = st.selectbox(
        "Triage with", names,
        format_func=lambda n: f"{'🟢' if ready[n] else '⚪'} {providers.PROVIDERS[n]}"
                              + (" · open source" if n in providers.OPEN_SOURCE else ""),
    )
    st.caption(f"Model: `{providers.model_for(provider)}`"
               + ("" if ready[provider] else " · not configured, keyword rules will be used"))
    vote = st.toggle("Ensemble vote (all configured models)")
    vote_with = tuple(n for n in names if ready[n] and n != "rules") if vote else ()
    if vote:
        st.caption("Voting: " + (", ".join(vote_with) or "no models configured, using rules"))
    with st.expander("Provider status"):
        for n in names:
            st.write(f"{'🟢' if ready[n] else '⚪'} {providers.PROVIDERS[n]}")

    st.header("Complaints")
    new = st.text_area("Submit a new complaint")
    if st.button("Submit") and new.strip():
        st.session_state.texts.append(new.strip())
    if st.button("Reset to synthetic dataset"):
        st.session_state.texts = list(COMPLAINTS)
        st.session_state.decisions = {}


@st.cache_data(show_spinner="Analyzing complaints...")
def run(texts, provider, vote_with):
    return pd.DataFrame(analyze_all(list(texts), provider, list(vote_with) or None))


df = run(tuple(st.session_state.texts), provider, vote_with)
st.caption(f"Duplicate detection: {df.dedup.iloc[0]}")
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

st.subheader("Compare models")
sample = st.selectbox("Complaint", df.text.tolist(), key="cmp_text")
picked = st.multiselect("Models", names, default=[n for n in names if ready[n]][:4] or ["rules"],
                        format_func=lambda n: providers.PROVIDERS[n])
if st.button("Run comparison"):
    st.dataframe(pd.DataFrame([
        {"model": providers.PROVIDERS[p], **{k: v for k, v in classify(sample, p).items()}} for p in picked
    ]), width="stretch", hide_index=True)

st.subheader("Human review")
for _, r in df.iterrows():
    done = st.session_state.decisions.get(r.id)
    label = f"#{r.id} [{r.urgency}] {r.text[:70]}" + (f" ✅ {done['decision']}" if done else "")
    with st.expander(label):
        st.write(f"**AI suggestion:** {r.department} / {r.urgency}: {r.next_action}")
        st.caption(f"Reason: {r.reason} (confidence {r.confidence}, source: {r.source})")
        if os.environ.get("ELEVENLABS_API_KEY") and st.button("🔊 Read aloud", key=f"tts{r.id}"):
            try:
                st.audio(providers.speak(
                    f"Complaint {r.id}. {r.text}. Suggested: {r.urgency} priority, "
                    f"route to {r.department}. Next step: {r.next_action}."), format="audio/mp3")
            except Exception as e:
                st.error(f"ElevenLabs failed: {e}")
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
