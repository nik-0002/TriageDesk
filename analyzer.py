import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import providers

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


def classify(text, provider="claude"):
    """Ask one provider to triage a complaint. Falls back to keyword rules on any failure."""
    if provider == "rules" or not providers.available(provider):
        return _fallback(text)
    try:
        out = providers.parse_json(providers.complete(
            provider, PROMPT.format(urg=URGENCIES, dept=DEPARTMENTS, text=text)))
        if out.get("urgency") not in URGENCIES:
            out["urgency"] = "Medium"
        if out.get("department") not in DEPARTMENTS:
            out["department"] = "General"
        out["source"] = provider
        return out
    except Exception as e:
        r = _fallback(text)
        r["reason"] = f"{provider} failed ({type(e).__name__}), used keyword rules"
        return r


def consensus(text, provider_list):
    """Run several providers and take a majority vote on urgency and department."""
    votes = [classify(text, p) for p in provider_list]
    pick = lambda field: Counter(v[field] for v in votes).most_common(1)[0][0]
    urg, dept = pick("urgency"), pick("department")
    best = next(v for v in votes if v["urgency"] == urg and v["department"] == dept) \
        if any(v["urgency"] == urg and v["department"] == dept for v in votes) else votes[0]
    agree = sum(v["urgency"] == urg for v in votes)
    return {**best, "urgency": urg, "department": dept,
            "confidence": round(agree / len(votes), 2),
            "source": "vote:" + "+".join(v["source"] for v in votes),
            "votes": votes}


_EMBEDDER = None


def _embed_similarity(texts):
    """Open-source sentence embeddings (all-MiniLM-L6-v2). Returns None if unavailable."""
    global _EMBEDDER
    try:
        if _EMBEDDER is None:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer(os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        return cosine_similarity(_EMBEDDER.encode(texts))
    except Exception:
        _EMBEDDER = False
        return None


def find_duplicates(texts, threshold=0.2, embed_threshold=0.6, use_embeddings=True):
    """Union-find over similar pairs. Returns (cluster id per text, method); -1 = unique.

    Uses sentence embeddings when available, otherwise TF-IDF cosine similarity.
    """
    if len(texts) < 2:
        return [-1] * len(texts), "none"
    sim = _embed_similarity(texts) if use_embeddings and _EMBEDDER is not False else None
    if sim is not None:
        method, threshold = "embeddings", embed_threshold
    else:
        method = "tfidf"
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
    return out, method


def analyze_all(texts, provider="claude", vote_with=None):
    """Classify every complaint and attach duplicate clusters.

    vote_with: optional list of providers to run as a majority-vote ensemble.
    """
    clusters, method = find_duplicates(texts)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        if vote_with:
            results = list(pool.map(lambda t: consensus(t, vote_with), texts))
        else:
            results = list(pool.map(lambda t: classify(t, provider), texts))
    for i, (t, c, r) in enumerate(zip(texts, clusters, results)):
        r.pop("votes", None)
        r.update(id=i + 1, text=t, cluster=c, dedup=method)
        rows.append(r)
    return rows
