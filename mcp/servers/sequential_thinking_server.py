"""
NEXUS MCP Sequential Thinking Server
Enables structured chain-of-thought reasoning as a tool.

Based on Anthropic's official sequential-thinking reference server pattern.
An agent calls think() multiple times to build up a reasoning chain,
then calls conclude() to get the final synthesized answer.

This server maintains thought state in memory for the session lifetime.
Each thought chain has an ID; multiple chains can run in parallel.

Tools:
  think         — Add a thought step to the chain
  branch_thought — Create an alternative reasoning path
  revise_thought — Go back and revise an earlier thought
  conclude      — Synthesize the chain into a final answer
  get_chain     — Retrieve the full thought chain
  reset_chain   — Start fresh
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class Thought:
    id:              str
    content:         str
    thought_number:  int
    total_thoughts:  int
    is_revision:     bool = False
    revises:         Optional[str] = None   # thought id being revised
    branch_from:     Optional[str] = None   # thought id this branched from
    branch_id:       Optional[str] = None   # which branch this belongs to
    needs_more:      bool = False
    timestamp:       float = field(default_factory=time.time)


@dataclass
class ThoughtChain:
    chain_id:  str
    thoughts:  list[Thought] = field(default_factory=list)
    concluded: bool = False
    conclusion: str = ""
    created_at: float = field(default_factory=time.time)


# ── In-memory store ────────────────────────────────────────────────────────────
_chains: dict[str, ThoughtChain] = {}
_default_chain_id = "default"


def _get_or_create_chain(chain_id: str) -> ThoughtChain:
    if chain_id not in _chains:
        _chains[chain_id] = ThoughtChain(chain_id=chain_id)
    return _chains[chain_id]


def _ok(data: Any) -> dict:
    text = json.dumps(data, indent=2, default=str)
    return {"content": [{"type": "text", "text": text}]}

def _err(msg: str) -> dict:
    return {"content": [{"type": "text", "text": f"ERROR: {msg}"}], "isError": True}


# ── Tool implementations ───────────────────────────────────────────────────────

def think(
    thought:         str,
    thought_number:  int,
    total_thoughts:  int,
    needs_more:      bool = False,
    chain_id:        str  = "default",
) -> dict:
    """
    Add a thought step to the reasoning chain.

    Use this to think step-by-step about a problem:
    - thought_number: current step (1-indexed)
    - total_thoughts: estimated total steps (can increase as reasoning deepens)
    - needs_more: set True if you realize you need more steps than estimated
    """
    chain = _get_or_create_chain(chain_id)
    if chain.concluded:
        return _err(f"Chain '{chain_id}' is already concluded. Use reset_chain to start over.")

    t = Thought(
        id=str(uuid.uuid4())[:8],
        content=thought,
        thought_number=thought_number,
        total_thoughts=total_thoughts,
        needs_more=needs_more,
    )
    chain.thoughts.append(t)

    # Show the current reasoning context
    context = {
        "thought_id":     t.id,
        "thought_number": thought_number,
        "total_thoughts": total_thoughts,
        "chain_id":       chain_id,
        "thoughts_so_far": thought_number,
        "needs_more":     needs_more,
        "status":         "in_progress",
        "next_action":    (
            "Call think() again with thought_number+1"
            if thought_number < total_thoughts or needs_more
            else "Call conclude() to synthesize the answer, or branch_thought() to explore alternatives"
        ),
    }
    return _ok(context)


def branch_thought(
    thought:    str,
    from_thought_id: str,
    branch_label: str = "alternative",
    chain_id:   str = "default",
) -> dict:
    """
    Create an alternative reasoning path from an earlier thought.
    Useful for exploring different approaches to the same problem.
    """
    chain = _get_or_create_chain(chain_id)
    parent = next((t for t in chain.thoughts if t.id == from_thought_id), None)
    if parent is None:
        return _err(f"Thought '{from_thought_id}' not found in chain '{chain_id}'")

    branch_id = str(uuid.uuid4())[:6]
    t = Thought(
        id=str(uuid.uuid4())[:8],
        content=thought,
        thought_number=parent.thought_number + 1,
        total_thoughts=parent.total_thoughts,
        branch_from=from_thought_id,
        branch_id=branch_id,
    )
    chain.thoughts.append(t)
    return _ok({
        "thought_id":   t.id,
        "branch_id":    branch_id,
        "branched_from": from_thought_id,
        "label":        branch_label,
        "chain_id":     chain_id,
    })


def revise_thought(
    revised_content: str,
    revises_thought_id: str,
    chain_id: str = "default",
) -> dict:
    """
    Revise an earlier thought when you realize it was wrong.
    The original thought is kept for traceability.
    """
    chain = _get_or_create_chain(chain_id)
    original = next((t for t in chain.thoughts if t.id == revises_thought_id), None)
    if original is None:
        return _err(f"Thought '{revises_thought_id}' not found")

    t = Thought(
        id=str(uuid.uuid4())[:8],
        content=revised_content,
        thought_number=original.thought_number,
        total_thoughts=original.total_thoughts,
        is_revision=True,
        revises=revises_thought_id,
    )
    chain.thoughts.append(t)
    return _ok({
        "thought_id":   t.id,
        "revises":      revises_thought_id,
        "chain_id":     chain_id,
        "status":       "revised",
    })


def conclude(chain_id: str = "default", final_answer: str = "") -> dict:
    """
    Conclude the thought chain.
    If final_answer is not provided, returns the full chain for synthesis.
    """
    chain = _get_or_create_chain(chain_id)
    if not chain.thoughts:
        return _err(f"No thoughts in chain '{chain_id}'. Call think() first.")

    if final_answer:
        chain.concluded  = True
        chain.conclusion = final_answer
        return _ok({
            "chain_id":      chain_id,
            "concluded":     True,
            "final_answer":  final_answer,
            "thought_count": len(chain.thoughts),
        })

    # Return chain for the LLM to synthesize
    return _ok({
        "chain_id":    chain_id,
        "thoughts":    [
            {
                "number":      t.thought_number,
                "content":     t.content,
                "is_revision": t.is_revision,
                "branch":      t.branch_id,
            }
            for t in chain.thoughts
        ],
        "instruction": "Review the thought chain above and provide your final synthesized answer.",
    })


def get_chain(chain_id: str = "default") -> dict:
    """Retrieve the full thought chain with all steps."""
    chain = _chains.get(chain_id)
    if chain is None:
        return _ok({"chain_id": chain_id, "thoughts": [], "status": "not_started"})
    return _ok({
        "chain_id":      chain_id,
        "thoughts":      [asdict(t) for t in chain.thoughts],
        "concluded":     chain.concluded,
        "conclusion":    chain.conclusion,
        "thought_count": len(chain.thoughts),
    })


def reset_chain(chain_id: str = "default") -> dict:
    """Clear a thought chain and start fresh."""
    _chains.pop(chain_id, None)
    return _ok({"chain_id": chain_id, "status": "reset"})


TOOLS = {
    "think": {
        "fn": think,
        "description": (
            "Add a step to the sequential thought chain. Use for complex multi-step reasoning. "
            "Call repeatedly, incrementing thought_number each time. "
            "Set needs_more=true if you realize you need more steps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "thought":        {"type": "string", "description": "Your reasoning for this step"},
                "thought_number": {"type": "integer", "description": "Current step number (1-indexed)"},
                "total_thoughts": {"type": "integer", "description": "Estimated total steps needed"},
                "needs_more":     {"type": "boolean", "default": False},
                "chain_id":       {"type": "string",  "default": "default"},
            },
            "required": ["thought", "thought_number", "total_thoughts"],
        },
    },
    "branch_thought": {
        "fn": branch_thought,
        "description": "Create an alternative reasoning branch from an earlier thought",
        "inputSchema": {
            "type": "object",
            "properties": {
                "thought":          {"type": "string"},
                "from_thought_id":  {"type": "string", "description": "ID of the thought to branch from"},
                "branch_label":     {"type": "string", "default": "alternative"},
                "chain_id":         {"type": "string", "default": "default"},
            },
            "required": ["thought", "from_thought_id"],
        },
    },
    "revise_thought": {
        "fn": revise_thought,
        "description": "Revise an earlier thought that turned out to be incorrect",
        "inputSchema": {
            "type": "object",
            "properties": {
                "revised_content":    {"type": "string"},
                "revises_thought_id": {"type": "string"},
                "chain_id":           {"type": "string", "default": "default"},
            },
            "required": ["revised_content", "revises_thought_id"],
        },
    },
    "conclude": {
        "fn": conclude,
        "description": "Synthesize the thought chain into a final answer",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chain_id":     {"type": "string", "default": "default"},
                "final_answer": {"type": "string", "default": ""},
            },
        },
    },
    "get_chain": {
        "fn": get_chain,
        "description": "Retrieve the full thought chain with all steps",
        "inputSchema": {
            "type": "object",
            "properties": {"chain_id": {"type": "string", "default": "default"}},
        },
    },
    "reset_chain": {
        "fn": reset_chain,
        "description": "Clear a thought chain and start reasoning from scratch",
        "inputSchema": {
            "type": "object",
            "properties": {"chain_id": {"type": "string", "default": "default"}},
        },
    },
}


def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", ""); rpc_id = msg.get("id")
    if method == "initialize":
        return {"jsonrpc":"2.0","id":rpc_id,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"nexus-sequential-thinking","version":"1.0.0"}}}
    if method == "tools/list":
        return {"jsonrpc":"2.0","id":rpc_id,"result":{"tools":[{"name":n,"description":s["description"],"inputSchema":s["inputSchema"]} for n,s in TOOLS.items()]}}
    if method == "tools/call":
        params=msg.get("params",{}); name=params.get("name",""); args=params.get("arguments",{})
        if name not in TOOLS: return {"jsonrpc":"2.0","id":rpc_id,"result":_err(f"Unknown: {name}")}
        try: return {"jsonrpc":"2.0","id":rpc_id,"result":TOOLS[name]["fn"](**args)}
        except Exception as exc: return {"jsonrpc":"2.0","id":rpc_id,"result":_err(str(exc))}
    if method.startswith("notifications/"): return None
    return {"jsonrpc":"2.0","id":rpc_id,"error":{"code":-32601,"message":f"Unknown: {method}"}}


def main():
    for line in sys.stdin:
        line=line.strip()
        if not line: continue
        try: msg=json.loads(line)
        except: continue
        resp=handle_message(msg)
        if resp: sys.stdout.write(json.dumps(resp)+"\n"); sys.stdout.flush()

if __name__=="__main__": main()
