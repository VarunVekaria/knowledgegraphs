"""Stages 2 & 3: write triples into Neo4j and query subgraphs back out.

Stored data model:
  (:Entity {name, type})-[:REL {type, source_sentence, source, extracted_at}]->(:Entity)

- Node properties hold facts about the *entity* (its canonical name + type).
- Relationship properties hold provenance of the *fact* (the exact source
  sentence, which document it came from, and when it was extracted).
"""

from datetime import datetime, timezone

from neo4j import GraphDatabase

import config


class GraphStore:
    def __init__(self):
        config.require("NEO4J_URI", "NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )

    def close(self):
        self.driver.close()

    # --- setup -------------------------------------------------------------

    def setup(self):
        """Create a uniqueness constraint so entities don't duplicate."""
        with self.driver.session() as s:
            s.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def clear(self):
        """Wipe the graph (handy while experimenting)."""
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    # --- stage 2: build ----------------------------------------------------

    def add_triples(self, triples: list[dict], source: str = "unknown"):
        """Upsert each triple with its metadata. MERGE avoids duplicates.

        `source` records which document/file these triples came from and is
        stored on every relationship for provenance.
        """
        extracted_at = datetime.now(timezone.utc).isoformat()
        query = """
        UNWIND $triples AS t
        MERGE (s:Entity {name: t.subject})
          SET s.type = coalesce(t.subject_type, s.type)
        MERGE (o:Entity {name: t.object})
          SET o.type = coalesce(t.object_type, o.type)
        MERGE (s)-[r:REL {type: t.relation}]->(o)
          SET r.source_sentence = t.source_sentence,
              r.source = $source,
              r.extracted_at = $extracted_at
        """
        with self.driver.session() as s:
            s.run(query, triples=triples, source=source, extracted_at=extracted_at)

    # --- stage 3: retrieve -------------------------------------------------

    def subgraph_for_entities(self, names: list[str], hops: int = 2) -> list[dict]:
        """Return triples (with metadata) within `hops` of any matched entity.

        Matching is case-insensitive and substring-based so the question's
        wording ('acme') still finds the node ('Acme Corp').
        """
        # Variable-length bounds can't be parameterized in Cypher, so the
        # validated int is inlined directly into the query string.
        hops = max(1, int(hops))
        query = f"""
        MATCH (start:Entity)
        WHERE any(n IN $names WHERE toLower(start.name) CONTAINS toLower(n))
        MATCH path = (start)-[*1..{hops}]-(other:Entity)
        UNWIND relationships(path) AS r
        WITH startNode(r) AS s, r, endNode(r) AS o
        RETURN DISTINCT
            s.name AS subject, s.type AS subject_type,
            r.type AS relation,
            o.name AS object, o.type AS object_type,
            r.source_sentence AS source_sentence, r.source AS source
        LIMIT 200
        """
        with self.driver.session() as s:
            result = s.run(query, names=names)
            return [dict(record) for record in result]

    def all_triples(self) -> list[dict]:
        """Dump the whole graph as triples with metadata (inspection/fallback)."""
        query = """
        MATCH (s:Entity)-[r:REL]->(o:Entity)
        RETURN s.name AS subject, s.type AS subject_type,
               r.type AS relation,
               o.name AS object, o.type AS object_type,
               r.source_sentence AS source_sentence, r.source AS source
        """
        with self.driver.session() as s:
            return [dict(record) for record in s.run(query)]
