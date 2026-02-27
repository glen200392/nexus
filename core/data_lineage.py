"""
NEXUS Data Lineage Tracker
Tracks data provenance as a directed graph: sources → transformations → outputs.

Dual-mode operation:
  Primary  — Neo4j (py2neo / neo4j-driver) for full graph query capability
  Fallback — NetworkX in-memory + JSON persistence (zero external deps)

Graph Schema
────────────
Nodes (node_type):
  data_source    — original files, DB tables, API responses
  agent_task     — a NEXUS task execution
  transformation — data processing step (imputation, chunking, embedding)
  model          — trained ML model artifact
  output         — final result (answer text, chart, report)
  knowledge_doc  — document ingested into RAG knowledge base

Edges (edge_type):
  PRODUCED       — agent_task → output
  CONSUMED       — agent_task → data_source (reads input)
  DERIVED_FROM   — transformation → data_source
  TRAINED_ON     — model → data_source
  STORED_IN      — output → knowledge_doc (when distilled to memory)
  TRIGGERED_BY   — agent_task → agent_task (sub-task chain)

Usage:
    tracker = get_tracker()
    tracker.record_node("task_abc123", "agent_task", {"agent": "code_agent"})
    tracker.record_node("file_data.csv", "data_source", {"path": "/data/data.csv"})
    tracker.record_edge("task_abc123", "file_data.csv", "CONSUMED")
    lineage = tracker.get_lineage("file_data.csv")
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("nexus.lineage")

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "talentai123")
LINEAGE_FILE   = Path(os.environ.get("NEXUS_LINEAGE_FILE", "data/lineage_graph.json"))


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class LineageNode:
    node_id:    str
    node_type:  str    # data_source | agent_task | transformation | model | output | knowledge_doc
    properties: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "node_type": self.node_type,
                "properties": self.properties, "created_at": self.created_at}


@dataclass
class LineageEdge:
    from_id:    str
    to_id:      str
    edge_type:  str    # PRODUCED | CONSUMED | DERIVED_FROM | TRAINED_ON | STORED_IN | TRIGGERED_BY
    properties: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"from_id": self.from_id, "to_id": self.to_id,
                "edge_type": self.edge_type, "properties": self.properties,
                "created_at": self.created_at}


# ── Neo4j Backend ──────────────────────────────────────────────────────────────

class Neo4jLineageBackend:
    def __init__(self):
        self._driver = None
        self._connect()

    def _connect(self):
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            # Verify connectivity
            with self._driver.session() as s:
                s.run("RETURN 1")
            logger.info("Data lineage: connected to Neo4j at %s", NEO4J_URI)
        except Exception as exc:
            logger.warning("Neo4j unavailable for lineage tracking: %s", exc)
            self._driver = None

    @property
    def available(self) -> bool:
        return self._driver is not None

    def record_node(self, node: LineageNode) -> bool:
        if not self._driver:
            return False
        cypher = (
            f"MERGE (n:{node.node_type} {{node_id: $node_id}}) "
            "SET n += $props, n.created_at = $created_at"
        )
        props = dict(node.properties)
        props["node_id"] = node.node_id
        with self._driver.session() as s:
            s.run(cypher, node_id=node.node_id, props=props, created_at=node.created_at)
        return True

    def record_edge(self, edge: LineageEdge) -> bool:
        if not self._driver:
            return False
        cypher = (
            "MATCH (a {node_id: $from_id}), (b {node_id: $to_id}) "
            f"MERGE (a)-[r:{edge.edge_type}]->(b) "
            "SET r += $props, r.created_at = $created_at"
        )
        with self._driver.session() as s:
            s.run(cypher, from_id=edge.from_id, to_id=edge.to_id,
                  props=edge.properties, created_at=edge.created_at)
        return True

    def get_lineage(self, node_id: str, depth: int = 3) -> dict:
        if not self._driver:
            return {}
        cypher = (
            "MATCH path = (n {node_id: $node_id})-[*0.." + str(depth) + "]-(m) "
            "RETURN path"
        )
        nodes_seen: dict[str, dict] = {}
        edges_seen: list[dict] = []
        with self._driver.session() as s:
            result = s.run(cypher, node_id=node_id)
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    nid = node.get("node_id", str(node.id))
                    nodes_seen[nid] = dict(node)
                for rel in path.relationships:
                    edges_seen.append({
                        "from": rel.start_node.get("node_id"),
                        "to":   rel.end_node.get("node_id"),
                        "type": rel.type,
                    })
        return {"root": node_id, "nodes": list(nodes_seen.values()), "edges": edges_seen}

    def trace_back(self, node_id: str, depth: int = 5) -> list[dict]:
        """Trace upstream data sources for a given node."""
        if not self._driver:
            return []
        cypher = (
            "MATCH path = (n {node_id: $node_id})<-[*1.." + str(depth) + "]-(src) "
            "WHERE src.node_type IN ['data_source', 'knowledge_doc'] "
            "RETURN src"
        )
        sources = []
        with self._driver.session() as s:
            for record in s.run(cypher, node_id=node_id):
                sources.append(dict(record["src"]))
        return sources

    def get_downstream(self, node_id: str, depth: int = 5) -> list[dict]:
        """Get all nodes that depend on a given source."""
        if not self._driver:
            return []
        cypher = (
            "MATCH path = (n {node_id: $node_id})-[*1.." + str(depth) + "]->(downstream) "
            "RETURN DISTINCT downstream"
        )
        results = []
        with self._driver.session() as s:
            for record in s.run(cypher, node_id=node_id):
                results.append(dict(record["downstream"]))
        return results

    def export_dot(self) -> str:
        """Export entire graph as Graphviz DOT format."""
        if not self._driver:
            return "digraph NEXUS {}"
        lines = ["digraph NEXUS_Lineage {", "  rankdir=LR;", "  node [shape=box];"]
        with self._driver.session() as s:
            for record in s.run("MATCH (n) RETURN n"):
                n    = record["n"]
                nid  = n.get("node_id", str(n.id)).replace("-", "_")
                ntype = list(n.labels)[0] if n.labels else "unknown"
                lines.append(f'  {nid} [label="{nid}\\n({ntype})"];')
            for record in s.run("MATCH (a)-[r]->(b) RETURN a.node_id, type(r), b.node_id"):
                a = (record["a.node_id"] or "").replace("-", "_")
                b = (record["b.node_id"] or "").replace("-", "_")
                t = record["type(r)"]
                lines.append(f"  {a} -> {b} [label=\"{t}\"];")
        lines.append("}")
        return "\n".join(lines)


# ── NetworkX Fallback Backend ──────────────────────────────────────────────────

class NetworkXLineageBackend:
    def __init__(self):
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge]      = []
        self._load()

    def _load(self) -> None:
        if not LINEAGE_FILE.exists():
            return
        try:
            raw = json.loads(LINEAGE_FILE.read_text(encoding="utf-8"))
            for n in raw.get("nodes", []):
                node = LineageNode(**n)
                self._nodes[node.node_id] = node
            for e in raw.get("edges", []):
                self._edges.append(LineageEdge(**e))
        except Exception as exc:
            logger.warning("Could not load lineage file: %s", exc)

    def _save(self) -> None:
        LINEAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }
        LINEAGE_FILE.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def available(self) -> bool:
        return True

    def record_node(self, node: LineageNode) -> bool:
        self._nodes[node.node_id] = node
        self._save()
        return True

    def record_edge(self, edge: LineageEdge) -> bool:
        # Avoid duplicate edges
        for e in self._edges:
            if e.from_id == edge.from_id and e.to_id == edge.to_id and e.edge_type == edge.edge_type:
                return True
        self._edges.append(edge)
        self._save()
        return True

    def get_lineage(self, node_id: str, depth: int = 3) -> dict:
        visited: set[str] = set()
        result_nodes: list[dict] = []
        result_edges: list[dict] = []

        def _traverse(nid: str, d: int) -> None:
            if d < 0 or nid in visited:
                return
            visited.add(nid)
            if nid in self._nodes:
                result_nodes.append(self._nodes[nid].to_dict())
            for edge in self._edges:
                if edge.from_id == nid or edge.to_id == nid:
                    result_edges.append(edge.to_dict())
                    neighbor = edge.to_id if edge.from_id == nid else edge.from_id
                    _traverse(neighbor, d - 1)

        _traverse(node_id, depth)
        return {"root": node_id, "nodes": result_nodes, "edges": result_edges}

    def trace_back(self, node_id: str, depth: int = 5) -> list[dict]:
        """BFS upstream."""
        sources: list[dict] = []
        queue = [(node_id, 0)]
        visited: set[str] = set()
        while queue:
            nid, d = queue.pop(0)
            if d >= depth or nid in visited:
                continue
            visited.add(nid)
            for edge in self._edges:
                if edge.to_id == nid:
                    upstream = edge.from_id
                    n = self._nodes.get(upstream)
                    if n and n.node_type in ("data_source", "knowledge_doc"):
                        sources.append(n.to_dict())
                    queue.append((upstream, d + 1))
        return sources

    def get_downstream(self, node_id: str, depth: int = 5) -> list[dict]:
        results: list[dict] = []
        queue = [(node_id, 0)]
        visited: set[str] = set()
        while queue:
            nid, d = queue.pop(0)
            if d >= depth or nid in visited:
                continue
            visited.add(nid)
            for edge in self._edges:
                if edge.from_id == nid:
                    n = self._nodes.get(edge.to_id)
                    if n:
                        results.append(n.to_dict())
                    queue.append((edge.to_id, d + 1))
        return results

    def export_dot(self) -> str:
        lines = ["digraph NEXUS_Lineage {", "  rankdir=LR;", "  node [shape=box];"]
        for node in self._nodes.values():
            safe = node.node_id.replace("-", "_").replace(".", "_")
            lines.append(f'  {safe} [label="{node.node_id}\\n({node.node_type})"];')
        for edge in self._edges:
            a = edge.from_id.replace("-", "_").replace(".", "_")
            b = edge.to_id.replace("-", "_").replace(".", "_")
            lines.append(f'  {a} -> {b} [label="{edge.edge_type}"];')
        lines.append("}")
        return "\n".join(lines)


# ── DataLineageTracker (unified API) ──────────────────────────────────────────

class DataLineageTracker:
    """
    Unified data lineage tracking API.
    Automatically selects Neo4j or NetworkX backend based on availability.
    """

    def __init__(self):
        neo4j_backend = Neo4jLineageBackend()
        if neo4j_backend.available:
            self._backend = neo4j_backend
            self._backend_name = "neo4j"
        else:
            self._backend = NetworkXLineageBackend()
            self._backend_name = "networkx"
        logger.info("Data lineage backend: %s", self._backend_name)

    @property
    def backend(self) -> str:
        return self._backend_name

    def record_node(self, node_id: str, node_type: str, properties: Optional[dict] = None) -> None:
        node = LineageNode(node_id=node_id, node_type=node_type, properties=properties or {})
        self._backend.record_node(node)

    def record_edge(self, from_id: str, to_id: str, edge_type: str,
                    properties: Optional[dict] = None) -> None:
        edge = LineageEdge(from_id=from_id, to_id=to_id, edge_type=edge_type,
                           properties=properties or {})
        self._backend.record_edge(edge)

    def record_task(
        self,
        task_id: str,
        agent_id: str,
        input_ids: list[str],
        output_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Convenience method: record a complete task execution in the lineage graph."""
        props = {"agent_id": agent_id, **(metadata or {})}
        self.record_node(task_id, "agent_task", props)
        for inp in input_ids:
            self.record_edge(task_id, inp, "CONSUMED")
        if output_id:
            self.record_node(output_id, "output", {"task_id": task_id})
            self.record_edge(task_id, output_id, "PRODUCED")

    def get_lineage(self, node_id: str, depth: int = 3) -> dict:
        return self._backend.get_lineage(node_id, depth)

    def trace_back(self, node_id: str, depth: int = 5) -> list[dict]:
        """Return all upstream data sources for a node."""
        return self._backend.trace_back(node_id, depth)

    def get_downstream(self, node_id: str, depth: int = 5) -> list[dict]:
        """Return all nodes that depend on this source."""
        return self._backend.get_downstream(node_id, depth)

    def export_dot(self) -> str:
        return self._backend.export_dot()

    def status(self) -> dict:
        return {"backend": self._backend_name, "neo4j_uri": NEO4J_URI}


# ── Singleton ──────────────────────────────────────────────────────────────────
_tracker: Optional[DataLineageTracker] = None


def get_tracker() -> DataLineageTracker:
    global _tracker
    if _tracker is None:
        _tracker = DataLineageTracker()
    return _tracker
