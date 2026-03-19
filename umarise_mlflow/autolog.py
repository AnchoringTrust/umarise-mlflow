"""Auto-anchor context manager for MLflow runs."""

import atexit
import threading
from contextlib import contextmanager
from typing import Optional

import mlflow
from umarise_mlflow.anchor import anchor_artifact


class _AnchorTracker:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._original_log_artifact = None
        self._original_log_artifacts = None
        self._logged_paths: list = []
        self._lock = threading.Lock()

    def _patched_log_artifact(self, local_path, artifact_path=None):
        self._original_log_artifact(local_path, artifact_path)
        with self._lock:
            self._logged_paths.append(local_path)

    def _patched_log_artifacts(self, local_dir, artifact_path=None):
        import os
        self._original_log_artifacts(local_dir, artifact_path)
        with self._lock:
            for root, _, files in os.walk(local_dir):
                for f in files:
                    self._logged_paths.append(os.path.join(root, f))

    def start(self):
        self._original_log_artifact = mlflow.log_artifact
        self._original_log_artifacts = mlflow.log_artifacts
        mlflow.log_artifact = self._patched_log_artifact
        mlflow.log_artifacts = self._patched_log_artifacts

    def stop(self):
        if self._original_log_artifact:
            mlflow.log_artifact = self._original_log_artifact
        if self._original_log_artifacts:
            mlflow.log_artifacts = self._original_log_artifacts

    def anchor_all(self):
        run = mlflow.active_run()
        run_id = run.info.run_id if run else None
        with self._lock:
            paths = list(self._logged_paths)
            self._logged_paths.clear()
        results = []
        for path in paths:
            try:
                result = anchor_artifact(path, run_id=run_id, api_key=self.api_key)
                results.append(result)
            except Exception:
                pass
        return results


@contextmanager
def auto_anchor(api_key: Optional[str] = None):
    tracker = _AnchorTracker(api_key=api_key)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.anchor_all()
        tracker.stop()
