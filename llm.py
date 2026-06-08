"""Thin wrapper around the OpenRouter chat API (OpenAI-compatible)."""

import requests

import config


def chat(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Send a chat completion request and return the assistant's text."""
    config.require("OPENROUTER_API_KEY")

    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = requests.post(
        config.OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
