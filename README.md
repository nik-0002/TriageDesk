# TriageDesk: AI Grievance & Service Request Analyzer

Reads complaints, picks out the issue and urgency, groups duplicate or related complaints, and recommends a department and next action. A human reviewer makes every final decision. Synthetic data only.

## Features
- Triage with any of 11 options: Gemma 4 and Llama running locally through Ollama, Claude, OpenAI, Gemini, Grok, DeepSeek, Mistral, OpenRouter, Hugging Face, or offline keyword rules
- Ensemble mode: every configured model votes, majority wins, and agreement becomes the confidence score
- Compare models side by side on one complaint
- Duplicate detection with open source sentence embeddings (all MiniLM L6 v2), falling back to TF IDF
- ElevenLabs voice read out of each triage result
- Human review: approve, override or reject, then export to CSV
- Never breaks: any model error falls back to keyword rules

## Quick start (local)
```bash
pip install -r requirements.txt
cp .env.example .env        # add whichever keys you have; all are optional
set -a; source .env; set +a
streamlit run app.py
```

### Free, fully open source (no API keys)
```bash
# install Ollama from https://ollama.com, then:
ollama pull gemma4
ollama pull llama3.2
streamlit run app.py        # pick "Gemma 4 (Ollama, local)" in the sidebar
```

## One command with Docker
```bash
cp .env.example .env
docker compose up           # starts Ollama, pulls Gemma 4 + Llama, runs the app on :8501
```

## Providers
| Sidebar option | Open source | Needs |
|---|---|---|
| Gemma 4 (Ollama, local) | yes | Ollama running, `ollama pull gemma4` |
| Llama (Ollama, local) | yes | Ollama running, `ollama pull llama3.2` |
| Hugging Face (Gemma) | yes | `HF_TOKEN` |
| OpenRouter (Llama) | yes | `OPENROUTER_API_KEY` |
| DeepSeek | yes | `DEEPSEEK_API_KEY` |
| Mistral | yes | `MISTRAL_API_KEY` |
| Claude | no | `ANTHROPIC_API_KEY` |
| OpenAI | no | `OPENAI_API_KEY` |
| Gemini | no | `GEMINI_API_KEY` |
| Grok (xAI) | no | `XAI_API_KEY` |
| ElevenLabs (voice) | no | `ELEVENLABS_API_KEY` |

Every model ID can be changed with an env var (see `.env.example`). Check each provider's current model list if a default ID has been retired.

## Tests
```bash
pytest
```
A fake OpenAI compatible server tests the real HTTP path for every provider.

## Files
- `app.py`: Streamlit UI
- `analyzer.py`: triage prompt, keyword rules, ensemble vote, duplicate detection
- `providers.py`: every AI provider plus ElevenLabs
- `data.py`: 25 synthetic complaints
- `tests/`: pytest suite

## Stack
Python, Streamlit, scikit learn, pandas, sentence transformers, Claude, OpenAI, Gemini, Gemma, Llama, Hugging Face, Ollama, OpenRouter, DeepSeek, Grok, Mistral, ElevenLabs
