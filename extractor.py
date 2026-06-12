"""Stage 1: turn a piece of text into precise, well-cited triples via the LLM.

This is the conservative (high-precision) extractor: it only emits facts that
are explicitly stated and can be tied to a verbatim source sentence. It runs on
ONE chunk of text at a time; chunking lives in chunker.py.

Each triple carries:
  - subject / object           : the entities
  - subject_type / object_type : a coarse entity type (Person, Organization, ...)
  - relation                   : a snake_case verb phrase
  - source_sentence            : the exact sentence the fact came from
"""

import json
import re

import llm

SYSTEM_PROMPT = """You are an information extraction engine that builds knowledge graphs.
Read the text and extract factual relationships as triples.

Each triple is a JSON object with these fields:
  - "subject": the entity the fact is about
  - "subject_type": one of Person, Organization, Place, Product, Concept, Other
  - "relation": a verb phrase in snake_case carrying the meaning of the fact
    (e.g. founded, acquired, headquartered_in, exposed_to_risk_from)
  - "object": the related entity or value
  - "object_type": one of Person, Organization, Place, Product, Concept, Other
  - "source_sentence": the exact sentence from the text this fact was taken from,
    copied verbatim (used as evidence/provenance)

ENTITIES (subject and object) MUST be short, reusable names — the kind of thing
that could appear in many facts. This is the most important rule.
  - Prefer a named thing: a company, person, place, product, law, standard, or a
    well-defined concept (e.g. "Amcor", "Bemis", "Jersey", "LIBOR", "COVID-19",
    "raw materials", "credit rating").
  - Keep entities to roughly 1-4 words. NEVER put a clause or full description in
    the subject/object — move that meaning into the relation instead.
  - Strip leading articles and possessives ("the", "our", "certain"). Use the
    canonical noun, not a sentence fragment.

Good:  {"subject":"Amcor","relation":"acquired","object":"Bemis"}
Good:  {"subject":"Amcor","relation":"exposed_to_risk_from","object":"foreign exchange rates"}
Bad:   {"subject":"sales of our products and services","relation":"depend_on",
        "object":"volume of sales made by our customers to consumers"}
       -> instead: {"subject":"Amcor","relation":"sales_depend_on","object":"consumer demand"}
Bad:   {"subject":"changing preferences for products and packaging formats",...}
       -> instead: {"subject":"Amcor","relation":"exposed_to_risk_from","object":"changing consumer preferences"}

Rules:
- Only extract clear, important facts explicitly stated in the text. Do not infer
  or invent, and skip vague or hedged statements.
- Use the fullest canonical form of a name consistently (always "Amcor", never
  "the company" or "we").
- Avoid Concept/Other entities unless they are a concise, named concept; if the
  "object" would be a long phrase, shorten it to its core noun.
- "source_sentence" must be copied word-for-word from the input.
- Return ONLY a JSON array of these objects. No prose, no markdown fences.
"""

_CORE_FIELDS = ("subject", "relation", "object")
MAX_OUTPUT_TOKENS = 4000


def _salvage_objects(raw: str) -> list[dict]:
    """Recover complete JSON objects from a (possibly truncated) reply."""
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


def _subject_instruction(subject: str) -> str:
    """Extra prompt rule telling the model to resolve the narrator to `subject`."""
    return (
        f'\nThis document is written by/about "{subject}". Resolve all first-person '
        f'and generic self-references — "we", "us", "our", "the company", '
        f'"the Company", "the Group" — to "{subject}". Use "{subject}" as the '
        f"entity name in those cases, never the pronoun or generic phrase."
    )


def extract_triples(text: str, subject: str | None = None) -> list[dict]:
    """Extract precise triples from a single piece (chunk) of text.

    If `subject` is given, the narrator's self-references are canonicalized to it
    (coreference resolution), so facts attach to one consistent entity node.
    """
    if not text.strip():
        return []
    system = SYSTEM_PROMPT + (_subject_instruction(subject) if subject else "")
    reply = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return _normalize(_parse_json_array(reply))
