"""Stage 4: answer a question using only facts retrieved from the graph."""

import json
import re

import llm
from graph_store import GraphStore

ENTITY_PROMPT = """Extract the named entities (people, organizations, places,
products, concepts) mentioned in the question. Return ONLY a JSON array of
strings, e.g. ["Bob", "Berlin"]. No other text."""

ANSWER_PROMPT = """You answer questions using ONLY the facts provided.
Each fact is a triple "subject -> relation -> object" followed by the source
sentence it was extracted from.

- If the facts contain the answer, state it clearly and concisely.
- Quote or cite the relevant source sentence as evidence when helpful.
- If the facts are insufficient, say you don't have enough information.
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


def answer(question: str, store: GraphStore, hops: int = 2, verbose: bool = False) -> str:
    entities = _question_entities(question)

    triples = store.subgraph_for_entities(entities, hops=hops) if entities else []
    if not triples:
        # Fall back to the whole graph if entity matching found nothing.
        triples = store.all_triples()

    if verbose:
        print(f"\n[entities] {entities}")
        print(f"[retrieved {len(triples)} facts]")
        print(_facts_to_text(triples) or "  (none)")
        print()

    if not triples:
        return "The graph is empty - ingest some text first."

    facts = _facts_to_text(triples)
    return llm.chat(
        [
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"Facts:\n{facts}\n\nQuestion: {question}"},
        ]
    )
