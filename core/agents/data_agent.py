"""
NEXUS Data Agent — Pandas / Matplotlib Analysis
Loads structured data, runs statistical analysis, generates charts,
and uses LLM to extract business insights.

Supported inputs:
  - File paths: CSV, Excel, JSON, Parquet, TSV
  - Raw data: list of dicts or 2D list passed via context.metadata["data"]
  - SQL query result (from sqlite_server) via context.metadata["sql_result"]

Output: structured report with stats, key findings, and chart artifacts.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.data")


class DataAgent(BaseAgent):
    agent_id   = "data_agent"
    agent_name = "Data Analysis Agent"
    description = (
        "Loads structured data (CSV/Excel/JSON/Parquet), runs statistical analysis "
        "with pandas, generates matplotlib charts, and extracts business insights with LLM"
    )
    domain     = TaskDomain.ANALYSIS
    default_complexity = TaskComplexity.MEDIUM

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are an expert data analyst. Given a statistical summary of a dataset, "
            "provide clear, actionable insights.\n\n"
            "Structure your response as JSON:\n"
            "{\n"
            '  "executive_summary": "2-3 sentences on the most important findings",\n'
            '  "key_findings": ["finding 1", "finding 2", ...],\n'
            '  "anomalies": ["anything unusual or concerning"],\n'
            '  "recommendations": ["actionable next step 1", ...],\n'
            '  "follow_up_questions": ["what to investigate next"]\n'
            "}\n\n"
            "Be specific with numbers. Avoid vague statements like 'the data shows trends'."
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        task = context.user_message

        # Step 1: Load data
        df, load_error = await self._load_data(context)
        if df is None:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error=load_error or "Could not load data",
            )

        self._logger.info(
            "Loaded data: %d rows × %d cols", len(df), len(df.columns)
        )

        # Step 2: Statistical summary
        stats = self._compute_stats(df)

        # Step 3: Generate charts
        charts = self._generate_charts(df, context)

        # Step 4: LLM interprets findings
        decision = self.route_llm(context)
        stats_text = json.dumps(stats, indent=2, default=str)[:4000]

        llm_resp = await self._llm.chat(
            messages=[
                Message("user",
                    f"Task: {task}\n\nDataset statistics:\n{stats_text}\n\n"
                    "Provide data insights."
                )
            ],
            model=decision.primary,
            system=self._build_system_prompt(context),
            privacy_tier=context.privacy_tier,
        )

        insights = self._parse_json(llm_resp.content)

        # Step 5: Store summary to memory
        summary = f"Data analysis: {task}\nShape: {stats['shape']}\n{insights.get('executive_summary', '')}"
        await self.remember(
            content=summary, context=context,
            doc_type=DocumentType.SUMMARY,
            tags=["data_analysis", "pandas", context.domain.value],
        )

        return AgentResult(
            agent_id=self.agent_id,
            task_id=context.task_id,
            success=True,
            output={
                "insights":          insights,
                "statistics":        stats,
                "shape":             stats["shape"],
                "columns":           stats["columns"],
            },
            quality_score=0.8,
            tokens_used=llm_resp.tokens_in + llm_resp.tokens_out,
            cost_usd=llm_resp.cost_usd,
            llm_used=decision.primary.display_name,
            artifacts=charts,
        )

    # ── Data Loading ──────────────────────────────────────────────────────────

    async def _load_data(self, context: AgentContext):
        """Try multiple sources to load a DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            return None, "pandas not installed. Run: pip install pandas"

        # Source 1: file path from metadata or message
        file_path = context.metadata.get("file_path", "")
        if not file_path:
            # Try extracting path from message
            matches = re.findall(r"[\"']?(/[^\s\"']+\.(csv|xlsx?|json|parquet|tsv))[\"']?",
                                 context.user_message, re.I)
            if matches:
                file_path = matches[0][0]

        if file_path and Path(file_path).exists():
            ext = Path(file_path).suffix.lower()
            try:
                loaders = {
                    ".csv":     lambda p: pd.read_csv(p),
                    ".tsv":     lambda p: pd.read_csv(p, sep="\t"),
                    ".json":    lambda p: pd.read_json(p),
                    ".parquet": lambda p: pd.read_parquet(p),
                    ".xlsx":    lambda p: pd.read_excel(p),
                    ".xls":     lambda p: pd.read_excel(p),
                }
                loader = loaders.get(ext)
                if loader:
                    return loader(file_path), ""
            except Exception as exc:
                return None, f"Failed to load {file_path}: {exc}"

        # Source 2: raw data in metadata
        raw_data = context.metadata.get("data")
        if raw_data:
            try:
                if isinstance(raw_data, list):
                    return pd.DataFrame(raw_data), ""
                elif isinstance(raw_data, dict):
                    return pd.DataFrame([raw_data]), ""
            except Exception as exc:
                return None, f"Failed to create DataFrame from metadata: {exc}"

        # Source 3: SQL query result
        sql_result = context.metadata.get("sql_result", {})
        if sql_result and "rows" in sql_result:
            try:
                df = pd.DataFrame(
                    sql_result["rows"],
                    columns=sql_result.get("columns", []),
                )
                return df, ""
            except Exception as exc:
                return None, f"Failed to load SQL result: {exc}"

        return None, (
            "No data source found. Provide file_path in metadata, "
            "or mention a CSV/Excel/JSON file path in your message."
        )

    # ── Statistics ────────────────────────────────────────────────────────────

    def _compute_stats(self, df) -> dict:
        """Compute comprehensive statistics without assuming data types."""
        import pandas as pd

        stats: dict[str, Any] = {
            "shape":   {"rows": len(df), "cols": len(df.columns)},
            "columns": list(df.columns),
            "dtypes":  {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": {col: int(df[col].isna().sum()) for col in df.columns},
        }

        # Numeric stats
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            desc = df[numeric_cols].describe().round(4).to_dict()
            stats["numeric_summary"] = desc

        # Categorical stats (top values)
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_stats = {}
        for col in cat_cols[:10]:   # limit to first 10 cat cols
            vc = df[col].value_counts()
            cat_stats[col] = {
                "unique":    int(df[col].nunique()),
                "top_5":     vc.head(5).to_dict(),
                "null_count": int(df[col].isna().sum()),
            }
        if cat_stats:
            stats["categorical_summary"] = cat_stats

        # Correlations for numeric cols
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().round(3)
            # Get top 5 most correlated pairs
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    pairs.append({
                        "col_a": corr.columns[i],
                        "col_b": corr.columns[j],
                        "r":     corr.iloc[i, j],
                    })
            pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
            stats["top_correlations"] = pairs[:5]

        return stats

    # ── Chart Generation ──────────────────────────────────────────────────────

    def _generate_charts(self, df, context: AgentContext) -> list[dict]:
        """Generate relevant charts based on data types. Returns artifact dicts."""
        try:
            import matplotlib
            matplotlib.use("Agg")   # Non-interactive backend
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            logger.warning("matplotlib not installed; charts disabled")
            return []

        artifacts = []
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()

        def _save_fig(name: str) -> dict:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            return {"type": "chart", "name": name, "format": "png", "data": b64}

        # Chart 1: Distribution of first numeric column
        if numeric_cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                df[numeric_cols[0]].dropna().hist(bins=30, ax=ax, color="#366092", edgecolor="white")
                ax.set_title(f"Distribution: {numeric_cols[0]}")
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel("Count")
                plt.tight_layout()
                artifacts.append(_save_fig(f"dist_{numeric_cols[0]}"))
            except Exception as exc:
                logger.debug("Histogram failed: %s", exc)

        # Chart 2: Top category counts
        if cat_cols:
            try:
                col = cat_cols[0]
                top_n = df[col].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                top_n.plot(kind="bar", ax=ax, color="#2ea043", edgecolor="white")
                ax.set_title(f"Top Values: {col}")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                artifacts.append(_save_fig(f"bar_{col}"))
            except Exception as exc:
                logger.debug("Bar chart failed: %s", exc)

        # Chart 3: Correlation heatmap (if enough numeric cols)
        if len(numeric_cols) >= 3:
            try:
                import numpy as np
                corr = df[numeric_cols[:8]].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_yticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(corr.columns, fontsize=8)
                plt.colorbar(im, ax=ax)
                ax.set_title("Correlation Matrix")
                # Add values
                for i in range(len(corr)):
                    for j in range(len(corr.columns)):
                        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                                fontsize=7, color="black" if abs(corr.iloc[i, j]) < 0.7 else "white")
                plt.tight_layout()
                artifacts.append(_save_fig("correlation_heatmap"))
            except Exception as exc:
                logger.debug("Heatmap failed: %s", exc)

        return artifacts

    def _parse_json(self, raw: str) -> dict:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return {"executive_summary": raw[:500], "key_findings": [], "anomalies": [], "recommendations": []}
