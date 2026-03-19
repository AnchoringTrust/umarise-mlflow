"""
umarise-mlflow: Anchor MLflow artifacts to Bitcoin.

Usage:
    import umarise_mlflow

    # Option 1: Anchor a specific artifact
    umarise_mlflow.anchor_artifact("model.pt", run_id=run.info.run_id)

    # Option 2: Auto-anchor all artifacts in a run
    with umarise_mlflow.auto_anchor():
        mlflow.log_artifact("model.pt")
"""

from umarise_mlflow.anchor import anchor_artifact, anchor_run_artifacts
from umarise_mlflow.autolog import auto_anchor

__version__ = "0.1.0"
__all__ = ["anchor_artifact", "anchor_run_artifacts", "auto_anchor"]
