"""Core anchoring functions for MLflow artifacts."""

import hashlib
import os
from typing import Optional

import mlflow
from umarise import UmariseCore


def _hash_file(path: str) -> str:
    """Compute SHA-256 hash of a file. Bytes never leave the machine."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def anchor_artifact(
    path: str,
    run_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Anchor a single artifact file to Bitcoin.

    Args:
        path: Path to the artifact file.
        run_id: MLflow run ID. If provided, writes origin_id back as a run param.
        api_key: Umarise API key. Falls back to UMARISE_API_KEY env var.

    Returns:
        dict with origin_id, hash, captured_at, proof_status.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")

    client = UmariseCore(api_key=api_key)
    file_hash = _hash_file(path)
    result = client.attest(hash=file_hash)

    # Write metadata back to the MLflow run
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_param("umarise_origin_id", result.get("origin_id"))
            mlflow.log_param("umarise_hash", file_hash)
            mlflow.set_tag("umarise.anchored", "true")
            mlflow.set_tag("umarise.proof_status", result.get("proof_status", "pending"))

    return result


def anchor_run_artifacts(
    run_id: str,
    artifact_path: str = "",
    api_key: Optional[str] = None,
) -> list:
    """
    Anchor all artifacts in an MLflow run.

    Args:
        run_id: MLflow run ID.
        artifact_path: Subdirectory within the run's artifact store.
        api_key: Umarise API key.

    Returns:
        List of anchor results.
    """
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, artifact_path)
    results = []

    for artifact in artifacts:
        if artifact.is_dir:
            # Recurse into subdirectories
            results.extend(
                anchor_run_artifacts(run_id, artifact.path, api_key)
            )
        else:
            local_path = client.download_artifacts(run_id, artifact.path)
            result = anchor_artifact(local_path, run_id=None, api_key=api_key)
            result["artifact_path"] = artifact.path
            results.append(result)

    # Log summary to run
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("umarise.anchored", "true")
        mlflow.set_tag("umarise.artifact_count", str(len(results)))

    return results
