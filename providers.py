"""One classify() interface over every supported AI provider.

Most providers expose an OpenAI-compatible /chat/completions endpoint, so they share
one HTTP client. Claude uses the Anthropic SDK. Every model ID and base URL can be
overridden with environment variables.
"""
import json
import os
import re

import requests

TIMEOUT = 60

# name: (label, base_url, key_env, model_env, default_model, open_source)
OPENAI_COMPATIBLE = {
    "gemma": ("Gemma 4 (Ollama, local)", os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/v1",
              None, "GEMMA_MODEL", "gemma4", True),
    "llama": ("Llama (Ollama, local)", os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/v1",
              None, "LLAMA_MODEL", "llama3.2", True),
    "huggingface": ("Hugging Face (Gemma)", "https://router.huggingface.co/v1",
                    "HF_TOKEN", "HF_MODEL", "google/gemma-3-27b-it", True),
    "openrouter": ("OpenRouter (Llama)", "https://openrouter.ai/api/v1",
                   "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct", True),
    "deepseek": ("DeepSeek", "https://api.deepseek.com/v1",
                 "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat", True),
    "mistral": ("Mistral", "https://api.mistral.ai/v1",
                "MISTRAL_API_KEY", "MISTRAL_MODEL", "mistral-small-latest", True),
    "openai": ("OpenAI", "https://api.openai.com/v1",
               "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini", False),
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
               "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash", False),
    "grok": ("Grok (xAI)", "https://api.x.ai/v1",
             "XAI_API_KEY", "GROK_MODEL", "grok-3-mini", False),
}

PROVIDERS = {"claude": "Claude (Anthropic)", **{k: v[0] for k, v in OPENAI_COMPATIBLE.items()},
             "rules": "Keyword rules (offline)"}
OPEN_SOURCE = {k for k, v in OPENAI_COMPATIBLE.items() if v[5]}


def model_for(name):
    if name == "claude":
        return os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")
    if name in OPENAI_COMPATIBLE:
        _, _, _, model_env, default, _ = OPENAI_COMPATIBLE[name]
        return os.environ.get(model_env, default)
    return "rules"


def ollama_up():
    try:
        return requests.get(os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/tags", timeout=2).ok
    except requests.RequestException:
        return False


def available(name):
    """True if the provider is configured (key set, or local server reachable)."""
    if name == "rules":
        return True
    if name == "claude":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    key_env = OPENAI_COMPATIBLE[name][2]
    return ollama_up() if key_env is None else bool(os.environ.get(key_env))


def parse_json(raw):
    return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))


def _claude(prompt):
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=model_for("claude"), max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _openai_compatible(name, prompt):
    _, base, key_env, _, _, _ = OPENAI_COMPATIBLE[name]
    headers = {"Content-Type": "application/json"}
    if key_env:
        headers["Authorization"] = f"Bearer {os.environ[key_env]}"
    r = requests.post(f"{base}/chat/completions", headers=headers, timeout=TIMEOUT, json={
        "model": model_for(name),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 400,
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def complete(name, prompt):
    """Send a prompt to a provider and return the raw text reply."""
    if name == "claude":
        return _claude(prompt)
    return _openai_compatible(name, prompt)


def speak(text):
    """ElevenLabs text-to-speech. Returns MP3 bytes, or None if not configured."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        return None
    voice = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        json={"text": text, "model_id": os.environ.get("ELEVENLABS_MODEL", "eleven_multilingual_v2")},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.content
