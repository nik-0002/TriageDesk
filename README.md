# TriageDesk: AI Grievance & Service Request Analyzer

Reads complaints, picks out the issue and urgency, groups duplicate/related complaints, and recommends a department and next action. A human reviewer makes every final decision. Synthetic data only.

## Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # optional; keyword rules are used without it
streamlit run app.py
```

## Stack
Python, Streamlit, Claude API, scikit-learn (TF-IDF duplicate detection), pandas
