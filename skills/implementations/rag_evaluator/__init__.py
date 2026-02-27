"""
RAG Evaluator Skill
Evaluate retrieval and generation quality of the NEXUS RAG pipeline.
Computes standard IR metrics with pure Python (zero ML dependencies).

Retrieval metrics:
  Precision@k, Recall@k, F1@k, NDCG@k, MRR, Hit Rate

Generation metrics:
  ROUGE-L (pure Python), Exact Match, Contains Answer

Operations:
  add_ground_truth    — add annotated (query, relevant_doc_ids) pairs
  evaluate_retrieval  — compute Precision/Recall/NDCG against ground truth
  evaluate_generation — compute ROUGE-L / EM for LLM outputs
  run_benchmark       — full eval suite on all stored ground truth pairs
  compare_configs     — compare two RAG configs (e.g. top_k=5 vs top_k=10)
  get_report          — get the latest benchmark report
  list_benchmarks     — list all saved benchmark runs
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Optional
import os

from nexus.skills.registry import BaseSkill, SkillMeta

SKILL_META = SkillMeta(
    name="rag_evaluator",
    description=(
        "Evaluate NEXUS RAG pipeline quality using standard IR metrics "
        "(Precision@k, NDCG@k, MRR, ROUGE-L). Stores ground truth Q&A pairs "
        "and generates benchmarking reports."
    ),
    version="1.0.0",
    domains=["engineering", "analysis", "governance"],
    triggers=["rag eval", "retrieval quality", "ndcg", "precision recall",
              "rag benchmark", "檢索評估", "RAG品質", "向量搜尋評估"],
    requires=[],
    is_local=True,
)

_EVAL_DIR = Path(os.environ.get("NEXUS_EVAL_DIR", "data/rag_evals"))


def _precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / k if k > 0 else 0.0


def _recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / len(relevant)


def _ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    def dcg(results: list) -> float:
        score = 0.0
        for i, r in enumerate(results[:k]):
            if r in relevant:
                score += 1.0 / math.log2(i + 2)
        return score
    actual_dcg  = dcg(retrieved)
    ideal_dcg   = dcg(list(relevant))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def _mrr(retrieved: list, relevant: set) -> float:
    """Mean Reciprocal Rank (single query)."""
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 — LCS-based, pure Python."""
    def _lcs_len(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp   = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    h_tokens = hypothesis.lower().split()
    r_tokens = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0
    lcs   = _lcs_len(h_tokens, r_tokens)
    prec  = lcs / len(h_tokens)
    rec   = lcs / len(r_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Skill(BaseSkill):
    meta = SKILL_META

    def _gt_path(self) -> Path:
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        return _EVAL_DIR / "ground_truth.jsonl"

    def _bench_path(self, run_id: str) -> Path:
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        return _EVAL_DIR / f"bench_{run_id}.json"

    def _load_ground_truth(self) -> list[dict]:
        fp = self._gt_path()
        if not fp.exists():
            return []
        return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]

    async def run(
        self,
        operation: str,
        # add_ground_truth
        query: str = "",
        relevant_doc_ids: Optional[list[str]] = None,
        expected_answer: str = "",
        query_id: Optional[str] = None,
        # evaluate_retrieval
        retrieved_doc_ids: Optional[list[str]] = None,
        k: int = 5,
        # evaluate_generation
        generated_answer: str = "",
        reference_answer: str = "",
        # compare_configs
        config_a: Optional[dict] = None,
        config_b: Optional[dict] = None,
        results_a: Optional[list[dict]] = None,
        results_b: Optional[list[dict]] = None,
        **kwargs,
    ) -> Any:
        op = operation.lower()

        # ── add_ground_truth ──────────────────────────────────────────────────
        if op == "add_ground_truth":
            if not query or not relevant_doc_ids:
                return {"error": "query and relevant_doc_ids required"}
            entry = {
                "query_id":         query_id or f"q_{int(time.time()*1000)}",
                "query":            query,
                "relevant_doc_ids": relevant_doc_ids,
                "expected_answer":  expected_answer,
                "added_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with self._gt_path().open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return {"added": True, "query_id": entry["query_id"]}

        # ── evaluate_retrieval ────────────────────────────────────────────────
        if op == "evaluate_retrieval":
            if not retrieved_doc_ids:
                return {"error": "retrieved_doc_ids required"}
            # Find ground truth for this query
            if not relevant_doc_ids:
                if not query:
                    return {"error": "query or relevant_doc_ids required"}
                gt_entries = self._load_ground_truth()
                match = next((e for e in gt_entries if e["query"] == query), None)
                if match is None:
                    return {"error": f"No ground truth found for query: '{query[:80]}'"}
                relevant_doc_ids = match["relevant_doc_ids"]
            rel_set = set(relevant_doc_ids)
            metrics = {
                f"precision@{k}":   round(_precision_at_k(retrieved_doc_ids, rel_set, k), 4),
                f"recall@{k}":      round(_recall_at_k(retrieved_doc_ids, rel_set, k), 4),
                f"ndcg@{k}":        round(_ndcg_at_k(retrieved_doc_ids, rel_set, k), 4),
                "mrr":              round(_mrr(retrieved_doc_ids, rel_set), 4),
                "hit_rate":         1 if any(r in rel_set for r in retrieved_doc_ids[:k]) else 0,
                "retrieved_count":  len(retrieved_doc_ids),
                "relevant_count":   len(rel_set),
            }
            f1 = 0.0
            p = metrics[f"precision@{k}"]
            r = metrics[f"recall@{k}"]
            if p + r > 0:
                f1 = round(2 * p * r / (p + r), 4)
            metrics[f"f1@{k}"] = f1
            return {"query": query, "metrics": metrics}

        # ── evaluate_generation ───────────────────────────────────────────────
        if op == "evaluate_generation":
            if not generated_answer or not reference_answer:
                return {"error": "generated_answer and reference_answer required"}
            gen_tokens = _tokenize(generated_answer)
            ref_tokens = _tokenize(reference_answer)
            exact_match = int(generated_answer.strip().lower() == reference_answer.strip().lower())
            contains    = int(reference_answer.strip().lower() in generated_answer.lower())
            rouge       = round(_rouge_l(generated_answer, reference_answer), 4)
            # Token overlap F1
            gen_set = set(gen_tokens)
            ref_set = set(ref_tokens)
            common  = gen_set & ref_set
            tok_p   = len(common) / len(gen_set) if gen_set else 0.0
            tok_r   = len(common) / len(ref_set) if ref_set else 0.0
            tok_f1  = round(2 * tok_p * tok_r / (tok_p + tok_r), 4) if (tok_p + tok_r) > 0 else 0.0
            return {
                "metrics": {
                    "rouge_l":       rouge,
                    "token_f1":      tok_f1,
                    "exact_match":   exact_match,
                    "contains_ref":  contains,
                }
            }

        # ── run_benchmark ─────────────────────────────────────────────────────
        if op == "run_benchmark":
            gt_entries = self._load_ground_truth()
            if not gt_entries:
                return {"error": "No ground truth entries found. Add some with add_ground_truth first."}
            # For benchmark we need actual retrieved results; accept them in kwargs
            all_results_raw = kwargs.get("all_results", [])
            # If no actual results provided, just report ground truth stats
            if not all_results_raw:
                return {
                    "benchmark_ready": True,
                    "ground_truth_count": len(gt_entries),
                    "note": "Provide all_results=[{query_id, retrieved_doc_ids, generated_answer}] to run full benchmark",
                }

            # Compute metrics for each query
            per_query = []
            for gt in gt_entries:
                result = next((r for r in all_results_raw if r.get("query_id") == gt["query_id"]), None)
                if result is None:
                    continue
                rel_set  = set(gt["relevant_doc_ids"])
                ret_ids  = result.get("retrieved_doc_ids", [])
                gen_ans  = result.get("generated_answer", "")
                per_query.append({
                    "query_id":   gt["query_id"],
                    f"ndcg@{k}":  round(_ndcg_at_k(ret_ids, rel_set, k), 4),
                    f"precision@{k}": round(_precision_at_k(ret_ids, rel_set, k), 4),
                    "mrr":        round(_mrr(ret_ids, rel_set), 4),
                    "rouge_l":    round(_rouge_l(gen_ans, gt["expected_answer"]), 4) if gen_ans else None,
                })

            if not per_query:
                return {"error": "No matching query_ids between ground_truth and all_results"}

            # Aggregate
            def _avg(key: str) -> float:
                vals = [r[key] for r in per_query if r.get(key) is not None]
                return round(sum(vals) / len(vals), 4) if vals else 0.0

            run_id  = time.strftime("%Y%m%d_%H%M%S")
            summary = {
                "run_id":       run_id,
                "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "query_count":  len(per_query),
                "aggregate": {
                    f"avg_ndcg@{k}":      _avg(f"ndcg@{k}"),
                    f"avg_precision@{k}": _avg(f"precision@{k}"),
                    "avg_mrr":            _avg("mrr"),
                    "avg_rouge_l":        _avg("rouge_l"),
                },
                "per_query": per_query,
            }
            self._bench_path(run_id).write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            return summary

        # ── compare_configs ───────────────────────────────────────────────────
        if op == "compare_configs":
            if not results_a or not results_b:
                return {"error": "results_a and results_b required (list of {query_id, retrieved_doc_ids})"}
            gt_entries = self._load_ground_truth()
            gt_map     = {e["query_id"]: set(e["relevant_doc_ids"]) for e in gt_entries}

            def _score(results: list) -> dict:
                ndcgs, precs, mrrs = [], [], []
                for r in results:
                    rel = gt_map.get(r.get("query_id"), set())
                    if not rel:
                        continue
                    ret = r.get("retrieved_doc_ids", [])
                    ndcgs.append(_ndcg_at_k(ret, rel, k))
                    precs.append(_precision_at_k(ret, rel, k))
                    mrrs.append(_mrr(ret, rel))
                avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0
                return {f"ndcg@{k}": avg(ndcgs), f"precision@{k}": avg(precs), "mrr": avg(mrrs)}

            score_a = _score(results_a)
            score_b = _score(results_b)
            winner  = "A" if score_a[f"ndcg@{k}"] >= score_b[f"ndcg@{k}"] else "B"
            return {
                "config_a":      config_a or {},
                "config_b":      config_b or {},
                "scores_a":      score_a,
                "scores_b":      score_b,
                "winner":        winner,
                "ndcg_delta":    round(score_b[f"ndcg@{k}"] - score_a[f"ndcg@{k}"], 4),
            }

        # ── get_report ────────────────────────────────────────────────────────
        if op == "get_report":
            run_id = kwargs.get("run_id", "")
            if run_id:
                fp = self._bench_path(run_id)
                if not fp.exists():
                    return {"error": f"Run {run_id} not found"}
                return json.loads(fp.read_text(encoding="utf-8"))
            # Return latest
            _EVAL_DIR.mkdir(parents=True, exist_ok=True)
            reports = sorted(_EVAL_DIR.glob("bench_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not reports:
                return {"error": "No benchmark reports found. Run run_benchmark first."}
            return json.loads(reports[0].read_text(encoding="utf-8"))

        # ── list_benchmarks ───────────────────────────────────────────────────
        if op == "list_benchmarks":
            _EVAL_DIR.mkdir(parents=True, exist_ok=True)
            reports = sorted(_EVAL_DIR.glob("bench_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            items   = []
            for fp in reports:
                try:
                    d = json.loads(fp.read_text(encoding="utf-8"))
                    items.append({
                        "run_id":      d.get("run_id", fp.stem),
                        "timestamp":   d.get("timestamp", ""),
                        "query_count": d.get("query_count", 0),
                        "aggregate":   d.get("aggregate", {}),
                    })
                except Exception:
                    pass
            return {"benchmarks": items, "total": len(items)}

        return {"error": f"Unknown operation: {op}"}
