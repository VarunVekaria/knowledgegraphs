# Knowledge Graph from Text (OpenRouter + Neo4j)

Turn plain text into a knowledge graph, then answer questions by retrieving the
relevant subgraph and letting an LLM phrase the answer from those facts.

## Pipeline

```
Text
  -> extractor.py   LLM (OpenRouter) extracts (subject, relation, object) triples
  -> graph_store.py triples upserted into Neo4j with Cypher MERGE  (stage 2: build)
  -> graph_store.py question entities -> matching subgraph pulled out (stage 3: retrieve)
  -> qa.py          LLM phrases an answer using only the retrieved facts (stage 4)
```

## Setup

1. Create a free Neo4j AuraDB instance at https://console.neo4j.io
   (save the generated password - it's shown only once).
2. Get an OpenRouter API key at https://openrouter.ai/keys
3. Install dependencies and configure credentials:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env   # then edit .env with your keys/URI
   ```

## Usage

```powershell
python main.py ingest sample.txt          # build the graph from a file
python main.py ingest --text "Alice founded Acme."
python main.py ask "Who does Carla Diaz report to?"
python main.py show                        # list every triple
python main.py clear                       # wipe the graph
```

`ask` runs with verbose output so you can see which entities were detected and
which facts were retrieved before the answer is composed.

## Files

| File | Role |
|------|------|
| `config.py`      | loads settings from `.env` |
| `llm.py`         | OpenRouter chat wrapper |
| `extractor.py`   | stage 1: text -> triples |
| `graph_store.py` | stages 2 & 3: Neo4j build + retrieve |
| `qa.py`          | stage 4: question -> answer |
| `main.py`        | CLI tying the stages together |

## Inspecting the graph

Open the Neo4j Browser from the AuraDB console and run:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m
```

to see the whole graph visually.
