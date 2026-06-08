"""Central config: loads settings from the .env file once."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def require(*names: str) -> None:
    """Exit with a helpful message if any required setting is missing."""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        print(f"Missing config values: {', '.join(missing)}")
        print("Copy .env.example to .env and fill them in.")
        sys.exit(1)
