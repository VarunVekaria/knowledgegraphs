"""Stages 2 & 3: write the two-layer graph into Neo4j and query it back out.

Data model (one graph, two layers):

  Lexical layer (the raw text, tiled contiguously):
    (:Chunk {chunk_id, index, source, start_line, end_line, text})
    (:Chunk)-[:NEXT]->(:Chunk)                 reading order

  Domain layer (precise facts):
    (:Entity {name, type})
    (:Entity)-[:REL {type, source_sentence, chunk_id, source, extracted_at}]->(:Entity)

  Links between layers:
    (:Chunk)-[:MENTIONS]->(:Entity)            which entities a chunk contains
    (:Entity)-[:REL {chunk_id}]->(:Entity)     each fact points back to its chunk
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
        """Create uniqueness constraints so nodes don't duplicate."""
        with self.driver.session() as s:
            s.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            s.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            )

    def clear(self):
        """Wipe the graph (handy while experimenting)."""
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    # --- stage 2: build the lexical layer ----------------------------------

    def add_chunks(self, chunks: list[dict]):
        """Upsert chunk nodes and chain them in reading order with :NEXT."""
        upsert = """
        UNWIND $chunks AS c
        MERGE (n:Chunk {chunk_id: c.chunk_id})
          SET n.index = c.index, n.source = c.source,
              n.start_line = c.start_line, n.end_line = c.end_line, n.text = c.text
        """
        chain = """
        UNWIND $pairs AS p
        MATCH (a:Chunk {chunk_id: p.prev}), (b:Chunk {chunk_id: p.next})
        MERGE (a)-[:NEXT]->(b)
        """
        pairs = [
            {"prev": chunks[i]["chunk_id"], "next": chunks[i + 1]["chunk_id"]}
            for i in range(len(chunks) - 1)
        ]
        with self.driver.session() as s:
            s.run(upsert, chunks=chunks)
            if pairs:
                s.run(chain, pairs=pairs)

    # --- stage 2: build the domain layer -----------------------------------

    def add_triples(self, triples: list[dict], chunk_id: str, source: str = "unknown"):
        """Upsert entities + facts for one chunk, and link the chunk to them.

        Each fact records the chunk_id it came from, and the chunk is linked to
        every entity it mentions via :MENTIONS.
        """
        if not triples:
            return
        extracted_at = datetime.now(timezone.utc).isoformat()
        query = """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        UNWIND $triples AS t
        MERGE (s:Entity {name: t.subject})
          SET s.type = coalesce(t.subject_type, s.type)
        MERGE (o:Entity {name: t.object})
          SET o.type = coalesce(t.object_type, o.type)
        MERGE (s)-[r:REL {type: t.relation}]->(o)
          SET r.source_sentence = t.source_sentence,
              r.chunk_id = $chunk_id,
              r.source = $source,
              r.extracted_at = $extracted_at
        MERGE (c)-[:MENTIONS]->(s)
        MERGE (c)-[:MENTIONS]->(o)
        """
        with self.driver.session() as s:
            s.run(
                query,
                triples=triples,
                chunk_id=chunk_id,
                source=source,
                extracted_at=extracted_at,
            )

    # --- stage 3: retrieve -------------------------------------------------

    def subgraph_for_entities(self, names: list[str], hops: int = 2) -> list[dict]:
        """Return facts (with provenance) within `hops` of any matched entity."""
        # Variable-length bounds can't be parameterized in Cypher, so the
        # validated int is inlined directly into the query string.
        hops = max(1, int(hops))
        query = f"""
        MATCH (start:Entity)
        WHERE any(n IN $names WHERE toLower(start.name) CONTAINS toLower(n))
        MATCH path = (start)-[:REL*1..{hops}]-(other:Entity)
        UNWIND relationships(path) AS r
        WITH startNode(r) AS s, r, endNode(r) AS o
        RETURN DISTINCT
            s.name AS subject, s.type AS subject_type,
            r.type AS relation,
            o.name AS object, o.type AS object_type,
            r.source_sentence AS source_sentence,
            r.chunk_id AS chunk_id, r.source AS source
        LIMIT 200
        """
        with self.driver.session() as s:
            return [dict(record) for record in s.run(query, names=names)]

    def chunks_for_entities(self, names: list[str]) -> list[dict]:
        """Return the raw-text chunks that mention any matched entity."""
        query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE any(n IN $names WHERE toLower(e.name) CONTAINS toLower(n))
        RETURN DISTINCT c.chunk_id AS chunk_id, c.index AS index,
               c.start_line AS start_line, c.end_line AS end_line, c.text AS text
        ORDER BY index
        """
        with self.driver.session() as s:
            return [dict(record) for record in s.run(query, names=names)]

    def all_triples(self) -> list[dict]:
        """Dump the domain layer as triples with provenance (inspection)."""
        query = """
        MATCH (s:Entity)-[r:REL]->(o:Entity)
        RETURN s.name AS subject, s.type AS subject_type,
               r.type AS relation,
               o.name AS object, o.type AS object_type,
               r.source_sentence AS source_sentence,
               r.chunk_id AS chunk_id, r.source AS source
        """
        with self.driver.session() as s:
            return [dict(record) for record in s.run(query)]

    def all_chunks(self) -> list[dict]:
        """Dump the lexical layer (chunk nodes) in reading order."""
        query = """
        MATCH (c:Chunk)
        RETURN c.chunk_id AS chunk_id, c.index AS index, c.source AS source,
               c.start_line AS start_line, c.end_line AS end_line, c.text AS text
        ORDER BY c.index
        """
        with self.driver.session() as s:
            return [dict(record) for record in s.run(query)]
