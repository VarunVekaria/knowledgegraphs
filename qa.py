"""Stage 4: answer a question using only facts retrieved from the graph.

Retrieval is two-stage:
  1. Candidate generation - traverse the graph from the question's entities
     (broad; a hub entity like "Amcor" pulls in most of the graph).
  2. Rank & trim - score each candidate by relevance to the question and keep
     only the top few. Question entities are weighted by SPECIFICITY (inverse
     document frequency), so a rare term ("COVID-19") outweighs a hub ("Amcor"),
     and an "Amcor + COVID-19" question surfaces COVID facts, not all Amcor facts.
"""

import json
import math
import re

import llm
from graph_store import GraphStore

# How many facts / chunks survive ranking and reach the LLM.
TOP_FACTS = 40
TOP_CHUNKS = 5

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "for", "is", "are",
    "was", "were", "how", "does", "do", "did", "what", "which", "who", "that",
    "this", "these", "those", "by", "with", "as", "at", "from", "be", "it",
    "its", "our", "we", "us", "their", "about", "into", "than", "then",
}

ENTITY_PROMPT = """Extract the named entities (people, organizations, places,
products, concepts) mentioned in the question. Return ONLY a JSON array of
strings, e.g. ["Bob", "Berlin"]. No other text."""

ANSWER_PROMPT = """You answer questions using ONLY the provided context.
You are given two things:
  1. FACTS: triples "subject -> relation -> object" with their source sentence.
  2. PASSAGES: the raw text chunks those facts came from (fuller context).

- Prefer the precise FACTS; use the PASSAGES to fill in detail or when the
  facts alone are insufficient.
- Quote or cite the relevant source sentence as evidence when helpful.
- If neither contains the answer, say you don't have enough information.
- Do not use outside knowledge or guess."""


def _question_entities(question: str) -> list[str]:
    reply = llm.chat(
        [
            {"role": "system", "content": ENTITY_PROMPT},
            {"role": "user", "content": question},
        ]
    )
    match = re.search(r"\[.*\]", reply, re.DOTALL)
    return json.loads(match.group(0)) if match else []


def _facts_to_text(triples: list[dict]) -> str:
    lines = []
    for t in triples:
        line = f"{t['subject']} -> {t['relation']} -> {t['object']}"
        src = t.get("source_sentence")
        if src:
            line += f'\n    (source: "{src}")'
        lines.append(line)
    return "\n".join(lines)


def _chunks_to_text(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['chunk_id']} | lines {c['start_line']}-{c['end_line']}]\n{c['text']}"
        for c in chunks
    )


# --- ranking helpers ------------------------------------------------------

def _words(text: str) -> set[str]:
    """Content words of a string (lowercased, stopwords/short tokens removed)."""
    return {
        w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    }


def _fact_text(fact: dict) -> str:
    return " ".join(
        str(fact.get(k, "")) for k in ("subject", "relation", "object", "source_sentence")
    ).lower()


def _entity_weights(entities: list[str], facts: list[dict]) -> dict[str, float]:
    """Inverse-frequency weight per question entity: rarer => higher weight.

    A hub entity that appears in most candidate facts gets a near-zero weight;
    a specific entity that appears in only a few facts gets a high weight.
    """
    n = len(facts) or 1
    weights = {}
    for e in entities:
        el = e.lower()
        df = sum(1 for f in facts if el in _fact_text(f))
        weights[el] = math.log((n + 1) / (df + 1)) + 0.05
    return weights


def _score(text: str, weights: dict[str, float], q_words: set[str]) -> float:
    """Relevance = specificity-weighted entity hits + question-word overlap."""
    ent_score = sum(w for e, w in weights.items() if e in text)
    word_score = 0.1 * len(q_words & _words(text))
    return ent_score + word_score


def _rank(items, text_of, weights, q_words, top_k):
    """Sort items by relevance score, keep the top_k that score above zero."""
    scored = [(_score(text_of(it), weights, q_words), it) for it in items]
    scored.sort(key=lambda p: p[0], reverse=True)
    relevant = [it for s, it in scored if s > 0][:top_k]
    # If nothing matched (e.g. no entities), fall back to the first top_k.
    return relevant or [it for _, it in scored[:top_k]]


def answer(question: str, store: GraphStore, hops: int = 2, verbose: bool = False) -> str:
    entities = _question_entities(question)

    # Stage 1: broad candidate generation from the graph.
    candidates = store.subgraph_for_entities(entities, hops=hops) if entities else []
    if not candidates:
        candidates = store.all_triples()
    candidate_chunks = store.chunks_for_entities(entities) if entities else []

    # Stage 2: rank by relevance to the question and trim.
    q_words = _words(question)
    weights = _entity_weights(entities, candidates)
    triples = _rank(candidates, _fact_text, weights, q_words, TOP_FACTS)
    chunks = _rank(
        candidate_chunks, lambda c: str(c.get("text", "")).lower(),
        weights, q_words, TOP_CHUNKS,
    )

    if verbose:
        print(f"\n[entities] {entities}")
        print(
            f"[candidates: {len(candidates)} facts, {len(candidate_chunks)} chunks "
            f"-> kept {len(triples)} facts, {len(chunks)} chunks]"
        )
        print("[entity weights] " + ", ".join(
            f"{e}={w:.2f}" for e, w in weights.items()
        ))
        print(_facts_to_text(triples) or "  (no facts)")
        print()

    if not triples and not chunks:
        return "The graph is empty - ingest some text first."

    facts = _facts_to_text(triples) or "(none)"
    passages = _chunks_to_text(chunks) or "(none)"
    return llm.chat(
        [
            {"role": "system", "content": ANSWER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"FACTS:\n{facts}\n\nPASSAGES:\n{passages}\n\n"
                    f"Question: {question}"
                ),
            },
        ]
    )
