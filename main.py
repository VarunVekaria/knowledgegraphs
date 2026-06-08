"""Command-line entry point for the knowledge-graph pipeline.

Usage:
  python main.py ingest <file.txt>     Extract triples from a file and load them
  python main.py ingest --text "..."   Extract triples from inline text
  python main.py ask "your question"   Ask a question against the graph
  python main.py show                  Print every triple in the graph
  python main.py clear                 Delete everything in the graph
"""

import sys

import extractor
import qa
from graph_store import GraphStore


def cmd_ingest(args: list[str]):
    if args and args[0] == "--text":
        text = " ".join(args[1:])
        source = "inline-text"
    elif args:
        with open(args[0], encoding="utf-8") as f:
            text = f.read()
        source = args[0]
    else:
        print('Provide a file path or --text "...".')
        return

    print("Extracting triples via the LLM...")
    triples = extractor.extract_triples(text, verbose=True)
    print(f"Extracted {len(triples)} triples:")
    for t in triples:
        print(f"  ({t['subject_type']}) {t['subject']} -[{t['relation']}]-> "
              f"{t['object']} ({t['object_type']})")
        if t["source_sentence"]:
            print(f'      source: "{t["source_sentence"]}"')

    store = GraphStore()
    try:
        store.setup()
        store.add_triples(triples, source=source)
        print(f"\nLoaded into Neo4j (source={source}). "
              f"Total triples now: {len(store.all_triples())}")
    finally:
        store.close()


def cmd_ask(args: list[str]):
    if not args:
        print("Provide a question.")
        return
    question = " ".join(args)
    store = GraphStore()
    try:
        result = qa.answer(question, store, verbose=True)
        print("Answer:")
        print(result)
    finally:
        store.close()


def cmd_show(_args):
    store = GraphStore()
    try:
        triples = store.all_triples()
        if not triples:
            print("Graph is empty.")
        for t in triples:
            print(f"  ({t.get('subject_type')}) {t['subject']} -[{t['relation']}]-> "
                  f"{t['object']} ({t.get('object_type')})")
            if t.get("source_sentence"):
                print(f'      source: "{t["source_sentence"]}"')
    finally:
        store.close()


def cmd_clear(_args):
    store = GraphStore()
    try:
        store.clear()
        print("Graph cleared.")
    finally:
        store.close()


COMMANDS = {"ingest": cmd_ingest, "ask": cmd_ask, "show": cmd_show, "clear": cmd_clear}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
