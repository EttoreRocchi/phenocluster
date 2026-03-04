"""
Artifact Cache Module
=====================

Per-step caching with cascade invalidation for the PhenoCluster pipeline.
Uses joblib for serialization and SHA-256 hashing for config/data fingerprints.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# Pipeline step execution order (topological sort of the DAG).
PIPELINE_STEPS: List[str] = [
    "preprocess",
    "feature_select",
    "train_model",
    "evaluate_model",
    "stability",
    "run_analyses",
]

# DAG edges: each step lists its parent steps.
STEP_DEPENDENCIES: Dict[str, List[str]] = {
    "preprocess": [],
    "feature_select": ["preprocess"],
    "train_model": ["feature_select"],
    "evaluate_model": ["train_model"],
    "stability": ["evaluate_model"],
    "run_analyses": ["evaluate_model"],
}

# Config sections that each step depends on.
# A change in any of these sections invalidates the step.
# Keys match the top-level sections in PhenoClusterConfig.to_dict().
STEP_CONFIG_KEYS: Dict[str, List[str]] = {
    "preprocess": [
        "data",
        "preprocessing",
        "global",
    ],
    "feature_select": ["preprocessing"],
    "train_model": ["model", "global"],
    "evaluate_model": ["reference_phenotype"],
    "stability": ["stability"],
    "run_analyses": ["survival", "multistate", "inference"],
}


def _compute_data_fingerprint(data: pd.DataFrame) -> str:
    """Compute a lightweight SHA-256 fingerprint of a DataFrame.

    Uses shape, column names, dtypes, and a sample of rows (first/last 10)
    to detect dataset changes without hashing every cell.
    """
    h = hashlib.sha256()
    h.update(f"shape={data.shape}".encode())
    h.update(f"columns={list(data.columns)}".encode())
    h.update(f"dtypes={list(data.dtypes.astype(str))}".encode())

    # Hash a deterministic sample covering more of the data
    n = len(data)
    if n > 0:
        sample_size = min(max(20, n // 10), min(n, 1000))
        sample = data.sample(n=sample_size, random_state=0)
        h.update(sample.to_csv(index=False).encode())

    return h.hexdigest()


class ArtifactCache:
    """Per-step artifact cache with cascade invalidation.

    Parameters
    ----------
    artifacts_dir : Path
        Directory for storing cached artifacts and manifest.
    config_dict : dict
        Full pipeline config as a dict (from ``PhenoClusterConfig.to_dict()``).
    compress_level : int
        joblib compression level (0-9). Default 3.
    """

    MANIFEST_FILE = "cache_manifest.json"
    PIPELINE_VERSION = "1.0"

    def __init__(
        self,
        artifacts_dir: Path,
        config_dict: dict,
        compress_level: int = 3,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config_dict = config_dict
        self.compress_level = compress_level
        self._manifest = self._load_manifest()
        self._step_hashes: Dict[str, str] = {}

    # Manifest I/O

    def _load_manifest(self) -> dict:
        manifest_path = self.artifacts_dir / self.MANIFEST_FILE
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt cache manifest - starting fresh")
        return {"pipeline_version": self.PIPELINE_VERSION, "data_hash": None, "steps": {}}

    def _save_manifest(self) -> None:
        manifest_path = self.artifacts_dir / self.MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    # Config hashing

    def _hash_config_sections(self, keys: List[str]) -> str:
        """SHA-256 of the config sections for a set of keys."""
        h = hashlib.sha256()
        for key in sorted(keys):
            value = self.config_dict.get(key)
            h.update(f"{key}={json.dumps(value, sort_keys=True, default=str)}".encode())
        return h.hexdigest()

    def compute_step_hash(self, step_name: str) -> str:
        """Compute the composite hash for a step: own config + parent hashes."""
        if step_name in self._step_hashes:
            return self._step_hashes[step_name]

        h = hashlib.sha256()

        # Chain parent hashes
        for parent in STEP_DEPENDENCIES.get(step_name, []):
            parent_hash = self.compute_step_hash(parent)
            h.update(f"parent:{parent}={parent_hash}".encode())

        # Own config sections
        own_keys = STEP_CONFIG_KEYS.get(step_name, [])
        if own_keys:
            h.update(self._hash_config_sections(own_keys).encode())

        result = h.hexdigest()
        self._step_hashes[step_name] = result
        return result

    # Validity checks

    def is_step_valid(self, step_name: str, data_hash: str) -> bool:
        """Check whether a step's cached artifacts are still valid."""
        # Data must match
        if self._manifest.get("data_hash") != data_hash:
            return False

        # Pipeline version must match
        if self._manifest.get("pipeline_version") != self.PIPELINE_VERSION:
            return False

        step_info = self._manifest.get("steps", {}).get(step_name)
        if not step_info or not step_info.get("valid", False):
            return False

        # Config hash must match
        current_hash = self.compute_step_hash(step_name)
        if step_info.get("config_hash") != current_hash:
            return False

        # All artifact files must exist
        for artifact_file in step_info.get("artifacts", []):
            if not (self.artifacts_dir / artifact_file).exists():
                return False

        return True

    # Save / Load

    def _artifact_filename(self, step_name: str) -> str:
        return f"cache_{step_name}.joblib"

    def save_step_artifacts(
        self,
        step_name: str,
        artifacts: Dict[str, Any],
        data_hash: str,
    ) -> None:
        """Save artifacts for a pipeline step and update the manifest."""
        filename = self._artifact_filename(step_name)
        filepath = self.artifacts_dir / filename

        joblib.dump(artifacts, filepath, compress=self.compress_level)

        # Update manifest
        self._manifest["data_hash"] = data_hash
        self._manifest.setdefault("steps", {})[step_name] = {
            "config_hash": self.compute_step_hash(step_name),
            "artifacts": [filename],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "valid": True,
        }
        self._save_manifest()
        logger.info(f"CACHE SAVE: {step_name} -> {filename}")

    def load_step_artifacts(self, step_name: str) -> Dict[str, Any]:
        """Load cached artifacts for a pipeline step."""
        filename = self._artifact_filename(step_name)
        filepath = self.artifacts_dir / filename
        return joblib.load(filepath)

    # Invalidation

    def _get_downstream(self, step_name: str) -> List[str]:
        """Return all steps downstream of (and including) ``step_name``."""
        downstream = {step_name}
        changed = True
        while changed:
            changed = False
            for s, parents in STEP_DEPENDENCIES.items():
                if s not in downstream and any(p in downstream for p in parents):
                    downstream.add(s)
                    changed = True
        return list(downstream)

    def invalidate_from(self, step_name: str) -> None:
        """Mark a step and all its downstream dependents as invalid."""
        to_invalidate = self._get_downstream(step_name)
        for s in to_invalidate:
            if s in self._manifest.get("steps", {}):
                self._manifest["steps"][s]["valid"] = False
        self._save_manifest()
        logger.info(f"CACHE INVALIDATE: {to_invalidate}")
