"""
NEXUS ML Pipeline Agent
Orchestrates scikit-learn machine learning pipelines end-to-end.

Operations (context.metadata["operation"]):
  train          — train a model on provided data, return metrics + artifact
  evaluate       — evaluate an existing model on new data
  predict        — run inference on new samples
  compare_models — train multiple models, compare cross-val scores
  feature_importance — extract and rank feature importances
  auto_ml        — auto-select best model from a candidate list

Input (context.metadata):
  file_path       — CSV/Excel/JSON file path (or use data_agent output)
  sql_result      — dict with {rows, columns} from sqlite_server
  data            — list of dicts (inline data)
  target_column   — name of the label column
  pipeline_config — {
      problem_type:  "classification" | "regression",
      model:         "auto" | "random_forest" | "gradient_boosting" | "logistic" | "linear",
      test_size:     0.2,
      cv_folds:      5,
      scale_features: true,
      handle_missing: "mean" | "median" | "most_frequent" | "drop",
  }

Output:
  metrics, feature_importances, model_artifact (base64 pickle), charts
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any, Optional

from nexus.core.agents.base import AgentContext, AgentResult, BaseAgent
from nexus.core.llm.client import LLMClient, Message, get_client
from nexus.core.llm.router import TaskDomain, TaskComplexity
from nexus.knowledge.rag.schema import DocumentType

logger = logging.getLogger("nexus.agents.ml_pipeline")

_MODELS_DIR = Path("data/models")

# ── Model registry ─────────────────────────────────────────────────────────────

def _build_classifier(model_type: str, random_state: int = 42):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    choices = {
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=random_state),
        "gradient_boosting":   GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "logistic":            LogisticRegression(max_iter=500, random_state=random_state),
        "svm":                 SVC(probability=True, random_state=random_state),
    }
    return choices.get(model_type, choices["random_forest"])


def _build_regressor(model_type: str, random_state: int = 42):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    choices = {
        "random_forest":     RandomForestRegressor(n_estimators=100, random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        "linear":            LinearRegression(),
        "ridge":             Ridge(random_state=random_state),
    }
    return choices.get(model_type, choices["random_forest"])


# ── Training logic (runs in executor to avoid blocking event loop) ─────────────

def _train_sync(
    df,
    target_column: str,
    config: dict,
) -> dict:
    """Synchronous training — called via run_in_executor."""
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            mean_squared_error, r2_score, mean_absolute_error,
            classification_report,
        )
    except ImportError:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}

    problem_type   = config.get("problem_type", "classification")
    model_type     = config.get("model", "random_forest")
    test_size      = float(config.get("test_size", 0.2))
    cv_folds       = int(config.get("cv_folds", 5))
    scale_features = config.get("scale_features", True)
    missing_strat  = config.get("handle_missing", "mean")

    # ── Data preparation ───────────────────────────────────────────────────────
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found. Available: {list(df.columns)}"}

    y_raw = df[target_column]
    X     = df.drop(columns=[target_column])

    # Keep only numeric columns (convert what we can)
    X_numeric = X.select_dtypes(include=["number"])
    dropped_cols = [c for c in X.columns if c not in X_numeric.columns]

    # Encode target for classification
    le = None
    if problem_type == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(y_raw.astype(str))
    else:
        y  = y_raw.values

    # ── Pipeline ───────────────────────────────────────────────────────────────
    steps = [("imputer", SimpleImputer(strategy=missing_strat if missing_strat != "drop" else "mean"))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    if problem_type == "classification":
        steps.append(("model", _build_classifier(model_type)))
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        steps.append(("model", _build_regressor(model_type)))
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    pipe = Pipeline(steps)

    # ── Train / test split ─────────────────────────────────────────────────────
    if problem_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42
        )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # ── Metrics ────────────────────────────────────────────────────────────────
    if problem_type == "classification":
        scoring   = "f1_weighted"
        cv_scores = cross_val_score(pipe, X_numeric, y, cv=cv, scoring=scoring)
        metrics   = {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
            "cv_mean":   round(float(cv_scores.mean()), 4),
            "cv_std":    round(float(cv_scores.std()), 4),
            "test_size": len(y_test),
            "train_size": len(y_train),
        }
        if len(np.unique(y)) == 2:
            try:
                metrics["auc_roc"] = round(float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])), 4)
            except Exception:
                pass
    else:
        scoring   = "r2"
        cv_scores = cross_val_score(pipe, X_numeric, y, cv=cv, scoring=scoring)
        metrics   = {
            "r2":      round(float(r2_score(y_test, y_pred)), 4),
            "mae":     round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse":    round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std":  round(float(cv_scores.std()), 4),
        }

    # ── Feature importance ─────────────────────────────────────────────────────
    feature_names  = list(X_numeric.columns)
    model_step     = pipe.named_steps["model"]
    importances    = []
    if hasattr(model_step, "feature_importances_"):
        imp = model_step.feature_importances_
        pairs = sorted(zip(feature_names, imp.tolist()), key=lambda x: x[1], reverse=True)
        importances = [{"feature": f, "importance": round(i, 4)} for f, i in pairs]
    elif hasattr(model_step, "coef_"):
        coef = model_step.coef_.flatten() if model_step.coef_.ndim > 1 else model_step.coef_
        pairs = sorted(zip(feature_names, abs(coef).tolist()), key=lambda x: x[1], reverse=True)
        importances = [{"feature": f, "importance": round(i, 4)} for f, i in pairs]

    # ── Serialize model ────────────────────────────────────────────────────────
    buf = io.BytesIO()
    pickle.dump(pipe, buf)
    model_b64 = base64.b64encode(buf.getvalue()).decode()

    # ── Learning curve chart ───────────────────────────────────────────────────
    charts = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X_numeric, y, cv=cv, n_jobs=-1,
            train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_sizes, train_scores.mean(axis=1), label="Train", color="#2ea043")
        ax.plot(train_sizes, val_scores.mean(axis=1), label="Validation", color="#366092")
        ax.fill_between(train_sizes,
                        train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color="#2ea043")
        ax.fill_between(train_sizes,
                        val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color="#366092")
        ax.set_xlabel("Training samples")
        ax.set_ylabel(scoring)
        ax.set_title(f"Learning Curve — {model_type}")
        ax.legend()
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf2.seek(0)
        charts.append({
            "type": "chart", "name": "learning_curve",
            "format": "png", "data": base64.b64encode(buf2.read()).decode(),
        })
    except Exception as exc:
        logger.debug("Learning curve chart failed: %s", exc)

    return {
        "model_type":          model_type,
        "problem_type":        problem_type,
        "features_used":       feature_names,
        "features_dropped":    dropped_cols,
        "metrics":             metrics,
        "feature_importances": importances[:15],
        "model_artifact":      model_b64,
        "model_size_kb":       round(len(model_b64) * 0.75 / 1024, 1),
        "charts":              charts,
        "label_classes":       le.classes_.tolist() if le is not None else None,
    }


class MLPipelineAgent(BaseAgent):
    agent_id   = "ml_pipeline_agent"
    agent_name = "ML Pipeline Agent"
    description = (
        "Orchestrates scikit-learn ML pipelines: data loading, imputation, "
        "scaling, training, cross-validation, and feature importance. "
        "Returns trained model artifact, metrics, and learning curves."
    )
    domain             = TaskDomain.ANALYSIS
    default_complexity = TaskComplexity.HIGH

    def __init__(self, llm_client: Optional[LLMClient] = None, **kwargs):
        super().__init__(**kwargs)
        self._llm = llm_client or get_client()

    def _build_system_prompt(self, context: AgentContext) -> str:
        return (
            "You are an ML pipeline agent. Train, evaluate, and interpret machine learning "
            "models using scikit-learn. Select the best model via cross-validation and "
            "return structured metrics with feature importances."
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        operation = context.metadata.get("operation", "train")

        if operation in ("train", "compare_models", "auto_ml"):
            return await self._train_pipeline(context, operation)
        if operation == "feature_importance":
            return await self._feature_importance(context)
        if operation == "predict":
            return await self._predict(context)

        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=False, output=None,
            error=f"Unknown operation: {operation}",
        )

    async def _train_pipeline(self, context: AgentContext, mode: str) -> AgentResult:
        """Load data, train, evaluate. Runs scikit-learn in executor."""
        df, err = await self._load_data(context)
        if df is None:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None, error=err,
            )

        config        = context.metadata.get("pipeline_config", {})
        target_column = context.metadata.get("target_column", "")
        if not target_column:
            # Try to infer last column as target
            target_column = df.columns[-1]
            logger.info("No target_column specified; using last column: %s", target_column)

        if mode == "compare_models":
            model_types = ["random_forest", "gradient_boosting", "logistic"]
        elif mode == "auto_ml":
            model_types = ["random_forest", "gradient_boosting", "logistic", "ridge"]
        else:
            model_types = [config.get("model", "random_forest")]

        loop    = asyncio.get_event_loop()
        results = []
        for model_type in model_types:
            cfg_copy = dict(config)
            cfg_copy["model"] = model_type
            result = await loop.run_in_executor(
                None, _train_sync, df, target_column, cfg_copy
            )
            results.append(result)

        if mode in ("compare_models", "auto_ml"):
            # Pick best by primary metric
            def _primary(r: dict) -> float:
                m = r.get("metrics", {})
                return m.get("f1_weighted") or m.get("r2") or 0.0
            best = max(results, key=_primary)
            output = {
                "mode":           mode,
                "target_column":  target_column,
                "best_model":     best["model_type"],
                "best_metrics":   best["metrics"],
                "comparison":     [{"model": r["model_type"], "metrics": r["metrics"]} for r in results],
                "model_artifact": best["model_artifact"],
                "charts":         best.get("charts", []),
            }
        else:
            output = results[0]
            output["target_column"] = target_column

        if "error" in output:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None, error=output["error"],
            )

        # Save model to disk
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_name = f"{output.get('model_type', 'model')}_{int(time.time())}.pkl"
        model_path = _MODELS_DIR / model_name
        model_bytes = base64.b64decode(output["model_artifact"])
        model_path.write_bytes(model_bytes)
        output["model_path"] = str(model_path)

        # LLM interprets metrics
        decision = self.route_llm(context)
        metrics_txt = json.dumps(output.get("metrics", {}))
        llm_resp = await self._llm.chat(
            messages=[Message("user",
                f"ML model training complete.\n"
                f"Task: {context.user_message}\n"
                f"Model: {output.get('model_type')}, Target: {target_column}\n"
                f"Metrics: {metrics_txt}\n"
                f"Top features: {json.dumps(output.get('feature_importances', [])[:5])}\n\n"
                "Interpret these results in 2-3 sentences for a business audience."
            )],
            model=decision.primary,
            system="You are a data scientist summarizing ML model results. Be concise and actionable.",
            privacy_tier=context.privacy_tier,
        )
        output["interpretation"] = llm_resp.content[:500]

        await self.remember(
            content=f"ML pipeline: {output.get('model_type')} on {target_column}\n{metrics_txt}",
            context=context,
            doc_type=DocumentType.SUMMARY,
            tags=["ml_pipeline", output.get("model_type", ""), "training"],
        )

        artifacts = output.pop("charts", [])
        return AgentResult(
            agent_id=self.agent_id, task_id=context.task_id,
            success=True, output=output,
            quality_score=min(output.get("metrics", {}).get("cv_mean", 0.7) + 0.1, 1.0),
            artifacts=artifacts,
        )

    async def _feature_importance(self, context: AgentContext) -> AgentResult:
        """Load a saved model and return feature importances."""
        model_path = context.metadata.get("model_path", "")
        if not model_path or not Path(model_path).exists():
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None,
                error=f"Model file not found: {model_path}",
            )
        try:
            pipe = pickle.loads(Path(model_path).read_bytes())
            model_step = pipe.named_steps.get("model")
            if hasattr(model_step, "feature_importances_"):
                imp = model_step.feature_importances_.tolist()
            else:
                imp = []
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=True,
                output={"importances": imp, "model_path": model_path},
            )
        except Exception as exc:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None, error=str(exc),
            )

    async def _predict(self, context: AgentContext) -> AgentResult:
        """Run inference using a saved model."""
        model_path = context.metadata.get("model_path", "")
        if not model_path or not Path(model_path).exists():
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None, error=f"Model file not found: {model_path}",
            )
        df, err = await self._load_data(context)
        if df is None:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None, error=err,
            )
        try:
            pipe = pickle.loads(Path(model_path).read_bytes())
            preds = pipe.predict(df.select_dtypes(include=["number"])).tolist()
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=True,
                output={"predictions": preds, "count": len(preds)},
            )
        except Exception as exc:
            return AgentResult(
                agent_id=self.agent_id, task_id=context.task_id,
                success=False, output=None, error=str(exc),
            )

    # ── Data loading (reuse data_agent logic) ─────────────────────────────────

    async def _load_data(self, context: AgentContext):
        try:
            import pandas as pd
        except ImportError:
            return None, "pandas not installed. Run: pip install pandas"

        file_path = context.metadata.get("file_path", "")
        if not file_path:
            import re as _re
            matches = _re.findall(
                r'["\']?(/[^\s"\']+\.(?:csv|xlsx?|json|parquet|tsv))["\']?',
                context.user_message, _re.I,
            )
            if matches:
                file_path = matches[0]

        if file_path and Path(file_path).exists():
            ext = Path(file_path).suffix.lower()
            loaders = {
                ".csv": pd.read_csv, ".tsv": lambda p: pd.read_csv(p, sep="\t"),
                ".json": pd.read_json, ".parquet": pd.read_parquet,
                ".xlsx": pd.read_excel, ".xls": pd.read_excel,
            }
            fn = loaders.get(ext)
            if fn:
                try:
                    return fn(file_path), ""
                except Exception as exc:
                    return None, str(exc)

        raw_data = context.metadata.get("data")
        if raw_data and isinstance(raw_data, list):
            return pd.DataFrame(raw_data), ""

        sql_result = context.metadata.get("sql_result", {})
        if sql_result and "rows" in sql_result:
            return pd.DataFrame(sql_result["rows"], columns=sql_result.get("columns", [])), ""

        return None, "No data source found. Provide file_path, data, or sql_result in metadata."
