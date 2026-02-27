"""
Notebook Executor Skill
Execute Jupyter notebooks with parameter injection and capture outputs.

Uses papermill (primary) with nbformat fallback for cell-by-cell execution.
Output: executed notebook HTML, extracted metrics, inline charts.

Operations:
  execute       — run a notebook with parameters, return results
  list          — list available notebooks in configured dirs
  get_outputs   — parse outputs from a previously executed notebook
  validate      — check if a notebook is syntactically valid

Environment:
  NEXUS_NOTEBOOKS_DIR    — directory to search for notebooks (default: notebooks/)
  NEXUS_NOTEBOOKS_OUT    — output directory (default: data/notebooks/executed/)
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from nexus.skills.registry import BaseSkill, SkillMeta

SKILL_META = SkillMeta(
    name="notebook_executor",
    description=(
        "Execute Jupyter notebooks (.ipynb) with parameter injection. "
        "Supports papermill for parameterized execution. "
        "Returns executed notebook HTML, extracted metrics, and chart artifacts."
    ),
    version="1.0.0",
    domains=["analysis", "engineering", "research"],
    triggers=["jupyter", "notebook", "ipynb", "papermill",
              "run notebook", "execute notebook", "執行筆記本", "Jupyter"],
    requires=["nbformat"],
    is_local=True,
)

_NOTEBOOKS_DIR = Path(os.environ.get("NEXUS_NOTEBOOKS_DIR", "notebooks"))
_OUTPUT_DIR    = Path(os.environ.get("NEXUS_NOTEBOOKS_OUT", "data/notebooks/executed"))
_DEFAULT_TIMEOUT_S = 300   # 5 minutes


class Skill(BaseSkill):
    meta = SKILL_META

    async def run(
        self,
        operation: str = "execute",
        notebook_path: str = "",
        parameters: Optional[dict] = None,
        timeout_seconds: int = _DEFAULT_TIMEOUT_S,
        kernel_name: str = "python3",
        output_format: str = "html",    # html | notebook | both
        **kwargs,
    ) -> Any:
        op = operation.lower()

        # ── list ──────────────────────────────────────────────────────────────
        if op == "list":
            return self._list_notebooks()

        # ── validate ──────────────────────────────────────────────────────────
        if op == "validate":
            return self._validate(notebook_path)

        # ── get_outputs ───────────────────────────────────────────────────────
        if op == "get_outputs":
            return self._get_outputs(notebook_path)

        # ── execute ───────────────────────────────────────────────────────────
        if not notebook_path:
            return {"error": "notebook_path is required for execute"}

        path = Path(notebook_path)
        if not path.exists():
            # Try relative to notebooks dir
            path = _NOTEBOOKS_DIR / notebook_path
        if not path.exists():
            return {"error": f"Notebook not found: {notebook_path}"}

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp   = time.strftime("%Y%m%d_%H%M%S")
        stem        = path.stem
        out_nb_path = _OUTPUT_DIR / f"{stem}_executed_{timestamp}.ipynb"
        out_html    = _OUTPUT_DIR / f"{stem}_executed_{timestamp}.html"

        # ── Try papermill (preferred) ──────────────────────────────────────────
        papermill_result = None
        try:
            import papermill as pm
            start = time.time()
            pm.execute_notebook(
                str(path),
                str(out_nb_path),
                parameters=parameters or {},
                kernel_name=kernel_name,
                execution_timeout=timeout_seconds,
                report_mode=True,
            )
            elapsed = time.time() - start
            papermill_result = {"engine": "papermill", "elapsed_s": round(elapsed, 1)}
        except ImportError:
            pass    # Fall back to nbformat
        except Exception as exc:
            return {"error": f"papermill execution failed: {exc}", "notebook_path": str(path)}

        # ── Fallback: nbformat direct execution ────────────────────────────────
        if papermill_result is None:
            result = self._run_with_nbformat(path, out_nb_path, parameters, timeout_seconds, kernel_name)
            if "error" in result:
                return result
            papermill_result = result

        # ── Parse outputs from executed notebook ───────────────────────────────
        outputs = self._get_outputs(str(out_nb_path))

        # ── Export to HTML ────────────────────────────────────────────────────
        html_artifact = None
        if output_format in ("html", "both") and out_nb_path.exists():
            try:
                import subprocess
                subprocess.run(
                    ["jupyter", "nbconvert", "--to", "html",
                     str(out_nb_path), "--output", str(out_html)],
                    capture_output=True, timeout=30,
                )
                if out_html.exists():
                    html_bytes = out_html.read_bytes()
                    html_artifact = {
                        "type": "html", "name": out_html.name,
                        "data": base64.b64encode(html_bytes).decode(),
                        "path": str(out_html),
                    }
            except Exception as exc:
                pass   # HTML export optional

        result = {
            "executed_notebook": str(out_nb_path),
            "engine":            papermill_result.get("engine", "nbformat"),
            "elapsed_s":         papermill_result.get("elapsed_s", 0),
            "cell_count":        outputs.get("cell_count", 0),
            "metrics":           outputs.get("metrics", {}),
            "text_outputs":      outputs.get("text_outputs", [])[:5],
            "error_cells":       outputs.get("error_cells", []),
            "html_exported":     html_artifact is not None,
            "html_path":         str(out_html) if html_artifact else None,
        }
        artifacts = outputs.get("image_artifacts", [])
        if html_artifact:
            artifacts.append(html_artifact)
        if artifacts:
            result["artifacts"] = artifacts
        return result

    def _run_with_nbformat(
        self,
        nb_path: Path,
        out_path: Path,
        parameters: Optional[dict],
        timeout: int,
        kernel_name: str,
    ) -> dict:
        """Execute notebook cells using nbformat + nbconvert ExecutePreprocessor."""
        try:
            import nbformat
            from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
        except ImportError:
            return {"error": "nbformat/nbconvert not installed. Run: pip install nbformat nbconvert"}

        try:
            nb = nbformat.read(str(nb_path), as_version=4)

            # Inject parameters into first code cell tagged 'parameters'
            if parameters:
                param_code = "\n".join(f"{k} = {json.dumps(v)}" for k, v in parameters.items())
                for cell in nb.cells:
                    if cell.cell_type == "code" and "parameters" in cell.get("metadata", {}).get("tags", []):
                        cell.source = param_code
                        break

            start = time.time()
            ep    = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
            ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
            elapsed = time.time() - start

            nbformat.write(nb, str(out_path))
            return {"engine": "nbformat", "elapsed_s": round(elapsed, 1)}
        except Exception as exc:
            return {"error": str(exc)}

    def _get_outputs(self, notebook_path: str) -> dict:
        """Extract text outputs, metrics, and image artifacts from an executed notebook."""
        try:
            import nbformat
        except ImportError:
            return {"error": "nbformat not installed"}

        path = Path(notebook_path)
        if not path.exists():
            return {"error": f"Notebook not found: {notebook_path}"}

        try:
            nb = nbformat.read(str(path), as_version=4)
        except Exception as exc:
            return {"error": f"Could not read notebook: {exc}"}

        text_outputs:   list[str] = []
        image_artifacts: list[dict] = []
        error_cells:    list[dict] = []
        metrics: dict = {}

        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                otype = output.get("output_type", "")

                if otype in ("stream", "execute_result", "display_data"):
                    # Text
                    text = output.get("text", "") or "".join(output.get("data", {}).get("text/plain", []))
                    if isinstance(text, list):
                        text = "".join(text)
                    if text.strip():
                        text_outputs.append(text.strip()[:500])
                        # Try to extract metrics from text (JSON lines)
                        try:
                            data = json.loads(text.strip())
                            if isinstance(data, dict):
                                metrics.update(data)
                        except Exception:
                            pass

                    # Images (PNG/SVG)
                    data_dict = output.get("data", {})
                    if "image/png" in data_dict:
                        image_artifacts.append({
                            "type":   "chart",
                            "name":   f"cell_{i}_output",
                            "format": "png",
                            "data":   data_dict["image/png"],
                        })

                elif otype == "error":
                    error_cells.append({
                        "cell_index": i,
                        "ename":      output.get("ename", ""),
                        "evalue":     output.get("evalue", "")[:200],
                    })

        return {
            "cell_count":      len(nb.cells),
            "text_outputs":    text_outputs,
            "image_artifacts": image_artifacts,
            "error_cells":     error_cells,
            "metrics":         metrics,
        }

    def _list_notebooks(self) -> dict:
        _NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
        notebooks = []
        for fp in sorted(_NOTEBOOKS_DIR.rglob("*.ipynb")):
            try:
                import nbformat
                nb    = nbformat.read(str(fp), as_version=4)
                count = len(nb.cells)
            except Exception:
                count = -1
            notebooks.append({
                "path":       str(fp),
                "name":       fp.name,
                "cell_count": count,
            })
        return {"notebooks": notebooks, "count": len(notebooks)}

    def _validate(self, notebook_path: str) -> dict:
        if not notebook_path:
            return {"error": "notebook_path required"}
        path = Path(notebook_path)
        if not path.exists():
            path = _NOTEBOOKS_DIR / notebook_path
        if not path.exists():
            return {"error": f"Notebook not found: {notebook_path}"}
        try:
            import nbformat
            nb = nbformat.read(str(path), as_version=4)
            nbformat.validate(nb)
            return {
                "valid":      True,
                "cell_count": len(nb.cells),
                "nbformat":   nb.nbformat,
                "path":       str(path),
            }
        except Exception as exc:
            return {"valid": False, "error": str(exc)}
