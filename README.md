<h1 align="center">
  🗂️ TriageDesk
</h1>

<p align="center">
  <b>AI-powered Grievance & Service Request Analyzer</b><br/>
  <i>AI recommends — a human decides. Built for transparency, speed, and flexibility.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/providers-11-6e40c9" />
  <img src="https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/tests-pytest-green?logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

---

## ✨ What is TriageDesk?

TriageDesk is a Streamlit application that reads public complaints or service requests, automatically:

- 🏷️ **Categorises** the issue and assigns it to the right department  
- 🚨 **Prioritises** urgency: Critical → High → Medium → Low  
- 🔁 **Detects duplicates** and groups related complaints into clusters  
- 🤖 **Supports 11 AI providers** — local open-source models or cloud APIs  
- 🗳️ **Ensemble voting** — run multiple models and let the majority decide  
- 👨‍⚖️ **Human-in-the-loop** — every AI recommendation requires a human to Approve, Override, or Reject  
- 📥 **Export** reviewed decisions to CSV

> **All complaint data is synthetic. No real people or places are involved.**

---

## 🖥️ Demo

| Priority Queue | Human Review | Model Comparison |
|---|---|---|
| Colour-coded urgency table sorted by priority | Per-complaint approve / override / reject workflow | Run any set of models side-by-side on one complaint |

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone
git clone https://github.com/nik-0002/TriageDesk.git
cd TriageDesk

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure providers (all optional — keyword rules work offline)
cp .env.example .env
# Edit .env and add whichever API keys you have

# 4. Load env vars and run
set -a; source .env; set +a
streamlit run app.py
```

### 🆓 Free, Fully Open-Source (No API Keys)

Run with local models through [Ollama](https://ollama.com) — no cloud account needed:

```bash
# Install Ollama from https://ollama.com, then:
ollama pull gemma4
ollama pull llama3.2

streamlit run app.py
# Select "Gemma 4 (Ollama, local)" or "Llama (Ollama, local)" in the sidebar
```

---

## 🐳 One Command with Docker

Starts the app **and** a local Ollama server with Gemma 4 + Llama pulled automatically:

```bash
cp .env.example .env        # add any cloud keys you want (optional)
docker compose up           # app available at http://localhost:8501
```

---

## 🤖 Supported Providers

| Sidebar Option | Open Source | Requires |
|---|:---:|---|
| Gemma 4 (Ollama, local) | ✅ | Ollama running · `ollama pull gemma4` |
| Llama (Ollama, local) | ✅ | Ollama running · `ollama pull llama3.2` |
| Hugging Face (Gemma) | ✅ | `HF_TOKEN` |
| OpenRouter (Llama) | ✅ | `OPENROUTER_API_KEY` |
| DeepSeek | ✅ | `DEEPSEEK_API_KEY` |
| Mistral | ✅ | `MISTRAL_API_KEY` |
| Claude (Anthropic) | ❌ | `ANTHROPIC_API_KEY` |
| OpenAI | ❌ | `OPENAI_API_KEY` |
| Gemini | ❌ | `GEMINI_API_KEY` |
| Grok (xAI) | ❌ | `XAI_API_KEY` |
| ElevenLabs (voice read-aloud) | ❌ | `ELEVENLABS_API_KEY` |
| Keyword Rules | ✅ | Nothing — always available |

> Every model ID can be overridden with an env var. See `.env.example` for the full list.

---

## 🗳️ Ensemble Mode

Enable **"Ensemble vote"** in the sidebar to query every configured model simultaneously and take a **majority vote** on urgency and department. Agreement among models becomes the confidence score.

```
Urgency: Critical  (3/3 models agreed → confidence 1.0)
Source:  vote:claude+gemma+openai
```

---

## 🔍 Duplicate Detection

TriageDesk clusters similar complaints automatically:

1. **Sentence embeddings** (`all-MiniLM-L6-v2`) when `sentence-transformers` is installed  
2. **TF-IDF cosine similarity** as a lightweight fallback

Duplicate clusters are shown in a collapsible section below the main queue.

---

## 📁 Project Structure

```
TriageDesk/
├── app.py               # Streamlit UI — layout, sidebar, review workflow
├── analyzer.py          # Triage prompt, keyword rules, ensemble vote, dedup
├── providers.py         # Unified interface for all AI providers + ElevenLabs
├── data.py              # 25 synthetic complaints (no real data)
├── tests/
│   └── test_analyzer.py # pytest suite with a fake OpenAI-compatible server
├── Dockerfile           # Production Docker image
├── docker-compose.yml   # App + Ollama + model pulls in one command
├── pytest.ini           # Test configuration
├── requirements.txt     # Python dependencies
└── .env.example         # Template for API keys and model overrides
```

---

## 🧪 Tests

```bash
pytest
```

The test suite spins up a **fake OpenAI-compatible HTTP server** to exercise the real HTTP path for every provider without making live API calls:

- Keyword rule fallback
- Unconfigured provider fallback
- JSON parsing with model chatter
- Every OpenAI-compatible provider (parametrized)
- Claude path
- Bad values clamped to valid options
- Provider errors fall back to rules
- Ensemble consensus vote
- Duplicate clustering
- ElevenLabs TTS
- `analyze_all` end-to-end

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in the keys you want:

```env
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
XAI_API_KEY=
DEEPSEEK_API_KEY=
MISTRAL_API_KEY=
OPENROUTER_API_KEY=
HF_TOKEN=
ELEVENLABS_API_KEY=

# Optional model overrides
CLAUDE_MODEL=claude-haiku-5-5
OPENAI_MODEL=gpt-4o-mini
GEMINI_MODEL=gemini-2.5-flash
GEMMA_MODEL=gemma4
LLAMA_MODEL=llama3.2
OLLAMA_URL=http://localhost:11434
```

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| AI (cloud) | Claude, OpenAI, Gemini, Grok, DeepSeek, Mistral, Hugging Face, OpenRouter |
| AI (local) | Gemma 4, Llama via Ollama |
| Duplicate detection | sentence-transformers · scikit-learn (TF-IDF) |
| Data | pandas |
| Voice | ElevenLabs |
| Tests | pytest |
| Containers | Docker · Docker Compose |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">Made with ❤️ · <i>AI recommends, a human decides.</i></p>
