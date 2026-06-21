"""
Logs each sleep analysis to Arize Phoenix for ML observability.
Gracefully no-ops if ARIZE_API_KEY / ARIZE_SPACE_KEY are not set.
"""
import os

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key   = os.environ.get("ARIZE_API_KEY")
        space_key = os.environ.get("ARIZE_SPACE_KEY")
        if api_key and space_key:
            from arize.api import Client
            _client = Client(space_key=space_key, api_key=api_key)
    return _client


def log_prediction(session_id: str, analysis: dict) -> None:
    client = _get_client()
    if not client:
        return

    try:
        import pandas as pd
        from arize.utils.types import ModelTypes, Environments, Schema

        summary = analysis.get("sleep_summary", {})
        meta    = analysis.get("metadata", {})
        pct     = summary.get("pct_in_stage", {})

        features = {
            "analysis_type":  meta.get("analysis_type", "unknown"),
            "n_epochs":       int(meta.get("n_epochs", 0)),
            "efficiency_pct": float(summary.get("efficiency_pct", 0)),
            "latency_min":    float(summary.get("latency_min", 0)),
            "awakenings":     int(summary.get("awakenings", 0)),
            "rem_pct":        float(pct.get("REM", 0) if isinstance(pct, dict) else 0),
            "deep_pct":       float(pct.get("N3", 0) if isinstance(pct, dict) else 0),
        }

        schema = Schema(
            prediction_id_column_name="id",
            prediction_label_column_name="grade",
            prediction_score_column_name="score",
            feature_column_names=list(features.keys()),
        )

        df = pd.DataFrame([{
            "id":    session_id,
            "grade": summary.get("quality_grade", "?"),
            "score": float(summary.get("quality_score", 0)) / 100.0,
            **features,
        }])

        client.log(
            dataframe=df,
            schema=schema,
            model_id="sleepsense-ai",
            model_version="1.0",
            model_type=ModelTypes.SCORE_CATEGORICAL,
            environment=Environments.PRODUCTION,
        )
    except Exception as e:
        print(f"[Arize] Logging failed (non-fatal): {e}")
