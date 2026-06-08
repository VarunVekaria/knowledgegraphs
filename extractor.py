"""Stage 1: turn raw text into enriched triples via the LLM.

Each extracted triple carries, beyond the core (subject, relation, object):
  - subject_type / object_type : a coarse entity type (Person, Organization, ...)
  - source_sentence            : the exact sentence the fact came from (provenance)
"""

import json
import re

import llm

SYSTEM_PROMPT = """You are an information extraction engine that builds knowledge graphs.
Read the text and extract EVERY factual relationship you can find, as triples.

Each triple is a JSON object with these fields:
  - "subject": the entity the fact is about
  - "subject_type": one of Person, Organization, Place, Product, Concept, Other
  - "relation": a short verb phrase in snake_case (e.g. founded, works_at,
    located_in, acquired, subject_to)
  - "object": the related entity or value
  - "object_type": one of Person, Organization, Place, Product, Concept, Other
  - "source_sentence": the sentence (or closest sentence) the fact came from,
    copied from the text when possible. If a fact is spread across multiple
    sentences or only loosely supported, give the most relevant sentence anyway.
    Use an empty string only if truly no sentence applies.

Goal: MAXIMIZE RECALL. Capture all facts, whether strongly or weakly stated.
- Extract every relationship, including secondary, minor, and implied ones.
- Break compound sentences into multiple triples (one fact each).
- Include facts even if the citation is weak or approximate.
- Normalize entity names: use the fullest form of a name consistently
  (e.g. always "Acme Corp", not "Acme" in one triple and "the company" in another).
- Do not fabricate entities or relationships that the text does not support.
- Return ONLY a JSON array of these objects. No prose, no markdown fences.
"""

_CORE_FIELDS = ("subject", "relation", "object")

# Per-chunk extraction settings. High recall means lots of output, so we chunk
# the input small enough that each response fits comfortably under the cap.
CHUNK_CHARS = 3500
MAX_OUTPUT_TOKENS = 8000


def _chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Split text into sentence-aligned chunks of at most ~max_chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _salvage_objects(raw: str) -> list[dict]:
    """Recover complete JSON objects from a (possibly truncated) reply.

    The model can hit its output limit mid-array, leaving invalid JSON. Rather
    than lose the whole chunk, we pull out each well-formed {...} object.
    """
    objects = []
    for match in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            objects.append(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
    return objects


def _parse_json_array(raw: str) -> list[dict]:
    """Parse the model's reply into a list of dicts, tolerating truncation."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fall back to salvaging individual objects from a truncated/messy reply.
    return _salvage_objects(raw)


def _normalize(triples: list[dict]) -> list[dict]:
    """Keep well-formed triples and standardize their fields."""
    clean = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        if not all(k in t and str(t[k]).strip() for k in _CORE_FIELDS):
            continue
        clean.append(
            {
                "subject": str(t["subject"]).strip(),
                "subject_type": str(t.get("subject_type", "Other")).strip() or "Other",
                "relation": str(t["relation"]).strip().lower().replace(" ", "_"),
                "object": str(t["object"]).strip(),
                "object_type": str(t.get("object_type", "Other")).strip() or "Other",
                "source_sentence": str(t.get("source_sentence", "")).strip(),
            }
        )
    return clean


def _extract_chunk(chunk: str) -> list[dict]:
    reply = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return _normalize(_parse_json_array(reply))


def extract_triples(text: str, verbose: bool = False) -> list[dict]:
    """Chunk the text, extract triples per chunk, then merge and dedup."""
    chunks = _chunk_text(text)
    if verbose:
        print(f"Split into {len(chunks)} chunk(s).")

    seen, merged = set(), []
    for i, chunk in enumerate(chunks, 1):
        triples = _extract_chunk(chunk)
        if verbose:
            print(f"  chunk {i}/{len(chunks)}: {len(triples)} triples")
        for t in triples:
            # Dedup on the core fact; first source sentence wins.
            key = (t["subject"].lower(), t["relation"], t["object"].lower())
            if key not in seen:
                seen.add(key)
                merged.append(t)
    return merged
