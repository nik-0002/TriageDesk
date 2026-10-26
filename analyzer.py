import json
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

URGENCIES = ["Critical", "High", "Medium", "Low"]
DEPARTMENTS = [
    "Electricity", "Water Supply", "Sanitation", "Roads & Transport",
    "Campus IT", "Hostel & Mess", "Public Safety", "Health", "Parks", "General",
]

PROMPT = """You triage public grievances. Read the complaint and reply with ONLY a JSON object:
{{"category": short issue label,
 "urgency": one of {urg},
 "department": one of {dept},
 "next_action": one concrete next step,
 "reason": one sentence explaining urgency,
 "confidence": number 0-1}}
Critical = immediate risk to life/safety. High = injury, illness, or essential service down for many. Medium = service degraded. Low = minor/cosmetic.

Complaint: {text}"""

# Keyword fallback: (keywords, department, category)
RULES = [
    (["wire", "spark", "power", "electric", "streetlight", "street light", "outage"], "Electricity", "Electrical fault"),
    (["water", "leak", "pipe", "tap"], "Water Supply", "Water issue"),
    (["garbage", "toilet", "bin", "smell"], "Sanitation", "Sanitation"),
    (["pothole", "road", "speed breaker"], "Roads & Transport", "Road damage"),
    (["wifi", "wi-fi", "internet"], "Campus IT", "Network outage"),
    (["hostel", "mess"], "Hostel & Mess", "Hostel issue"),
    (["fire", "smoke", "dog", "noise"], "Public Safety", "Safety hazard"),
    (["park", "bench"], "Parks", "Park maintenance"),
]
CRITICAL = ["fire", "smoke", "spark", "live wire", "electrocut", "bit a child"]
HIGH = ["injur", "accident", "sick", "no water", "3 days", "three days", "exam", "flood"]
LOW = ["minor", "bench", "request"]


def _fallback(text):
    t = text.lower()
    dept, cat = "General", "Other"
    for kws, d, c in RULES:
        if any(k in t for k in kws):
            dept, cat = d, c
            break
    if any(k in t for k in CRITICAL):
        urg = "Critical"
    elif any(k in t for k in HIGH):
        urg = "High"
    elif any(k in t for k in LOW):
        urg = "Low"
    else:
        urg = "Medium"
    return {
        "category": cat, "urgency": urg, "department": dept,
        "next_action": f"Assign to {dept} field team",
        "reason": "Keyword rules (AI unavailable)", "confidence": 0.5,
        "source": "rules",
    }


def classify(text):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _fallback(text)
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-haiku-5-5",
            max_tokens=300,
            messages=[{"role": "user", "content": PROMPT.format(
                urg=URGENCIES, dept=DEPARTMENTS, text=text)}],
        )
        raw = msg.content[0].text
        out = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        if out.get("urgency") not in URGENCIES:
            out["urgency"] = "Medium"
        if out.get("department") not in DEPARTMENTS:
            out["department"] = "General"
        out["source"] = "claude"
        return out
    except Exception:
        return _fallback(text)


def find_duplicates(texts, threshold=0.2):
    """Union-find over TF-IDF cosine pairs above threshold. Returns cluster id per text (-1 = unique)."""
    if len(texts) < 2:
        return [-1] * len(texts)
    norm = [re.sub(r"wi-?fi", "wifi", t.lower()) for t in texts]
    sim = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(norm))
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sim[i, j] >= threshold:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(texts))]
    sizes = {r: roots.count(r) for r in roots}
    ids, out = {}, []
    for r in roots:
        if sizes[r] == 1:
            out.append(-1)
        else:
            out.append(ids.setdefault(r, len(ids) + 1))
    return out


def analyze_all(texts):
    rows = []
    clusters = find_duplicates(texts)
    for i, (t, c) in enumerate(zip(texts, clusters)):
        r = classify(t)
        r.update(id=i + 1, text=t, cluster=c)
        rows.append(r)
    return rows
