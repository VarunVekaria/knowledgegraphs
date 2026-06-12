"""Command-line entry point for the knowledge-graph pipeline.

Usage:
  python main.py ingest <file.txt> [--subject "Amcor"]
                                       Chunk text, extract facts, load both layers.
                                       --subject canonicalizes self-references
                                       ("we"/"us"/"the company") to that name.
  python main.py ingest --text "..." [--subject "Amcor"]   Same, from inline text
  python main.py ask "your question"   Ask a question against the graph
  python main.py show                  Print every fact (domain layer)
  python main.py chunks                Print every chunk (lexical layer)
  python main.py clear                 Delete everything in the graph
"""

import sys

import chunker
import extractor
import qa
from graph_store import GraphStore


def cmd_ingest(args: list[str]):
    # Pull out the optional --subject "Name" flag.
    subject = None
    if "--subject" in args:
        i = args.index("--subject")
        subject = args[i + 1] if i + 1 < len(args) else None
        args = args[:i] + args[i + 2:]

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

    # Lexical layer: split the raw text into contiguous chunks.
    chunks = chunker.chunk_document(text, source)
    print(f"Split into {len(chunks)} contiguous chunk(s).")
    if subject:
        print(f'Canonicalizing self-references to "{subject}".')

    store = GraphStore()
    try:
        store.setup()
        store.add_chunks(chunks)

        # Domain layer: extract precise facts per chunk and link them back.
        total = 0
        for c in chunks:
            triples = extractor.extract_triples(c["text"], subject=subject)
            store.add_triples(triples, chunk_id=c["chunk_id"], source=source)
            total += len(triples)
            print(
                f"  chunk {c['index']} (lines {c['start_line']}-{c['end_line']}): "
                f"{len(triples)} facts"
            )
            for t in triples:
                print(
                    f"      ({t['subject_type']}) {t['subject']} "
                    f"-[{t['relation']}]-> {t['object']} ({t['object_type']})"
                )

        print(
            f"\nLoaded into Neo4j (source={source}). "
            f"{len(chunks)} chunks, {total} facts."
        )
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
            print("No facts in the graph.")
        for t in triples:
            print(
                f"  ({t.get('subject_type')}) {t['subject']} -[{t['relation']}]-> "
                f"{t['object']} ({t.get('object_type')})  [{t.get('chunk_id')}]"
            )
            if t.get("source_sentence"):
                print(f'      source: "{t["source_sentence"]}"')
    finally:
        store.close()


def cmd_chunks(_args):
    store = GraphStore()
    try:
        chunks = store.all_chunks()
        if not chunks:
            print("No chunks in the graph.")
        for c in chunks:
            preview = c["text"][:120].replace("\n", " ")
            print(
                f"  [{c['index']}] {c['chunk_id']} "
                f"(lines {c['start_line']}-{c['end_line']}): {preview}..."
            )
    finally:
        store.close()


def cmd_clear(_args):
    store = GraphStore()
    try:
        store.clear()
        print("Graph cleared.")
    finally:
        store.close()


COMMANDS = {
    "ingest": cmd_ingest,
    "ask": cmd_ask,
    "show": cmd_show,
    "chunks": cmd_chunks,
    "clear": cmd_clear,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
