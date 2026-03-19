"""
models/versioning.py
====================
Model versioning system for AlphaForge.

Instead of overwriting xgb_model.pkl and hmm_model.pkl on each retrain,
this module saves timestamped copies and maintains a symlink-like pointer
to the "current" version. Supports rollback to any previous version.

Why this matters:
    If a retrain produces a worse model (which WILL happen eventually),
    there's no way to recover without versioning. A bad model silently
    produces wrong predictions with no error message — the worst kind
    of bug in a trading system.

Usage:
    from models.versioning import save_model, load_model, rollback_model

    # Save a new version
    save_model(model_obj, "xgb_model", metadata={"auc": 0.83, "fold": 18})

    # Load the current version
    model_obj, metadata = load_model("xgb_model")

    # Rollback to previous version
    rollback_model("xgb_model")
"""

import os
import sys
import glob
import json
import shutil
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODELS_DIR, MODEL_VERSIONS_DIR, MAX_MODEL_VERSIONS


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODEL_VERSIONS_DIR, exist_ok=True)


def save_model(model_obj, model_name: str, metadata: dict = None) -> str:
    """
    Save a model with timestamp versioning.

    Creates:
        models/versions/{model_name}_20260320_093000.pkl  — versioned copy
        models/{model_name}.pkl                           — current pointer

    Args:
        model_obj:  The object to pickle (dict, model, etc.)
        model_name: Base name without extension (e.g. "xgb_model")
        metadata:   Optional dict of training metrics to store alongside

    Returns:
        Path to the versioned file.
    """
    _ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_filename = f"{model_name}_{timestamp}.pkl"
    version_path = os.path.join(MODEL_VERSIONS_DIR, version_filename)

    # Save the versioned copy
    bundle = {
        "model_data": model_obj,
        "metadata": metadata or {},
        "timestamp": timestamp,
        "model_name": model_name,
    }
    with open(version_path, "wb") as f:
        pickle.dump(bundle, f)

    # Save metadata as JSON sidecar for easy inspection
    meta_path = version_path.replace(".pkl", "_meta.json")
    meta_record = {
        "timestamp": timestamp,
        "model_name": model_name,
        **(metadata or {}),
    }
    with open(meta_path, "w") as f:
        json.dump(meta_record, f, indent=2, default=str)

    # Update the "current" pointer — overwrite models/{model_name}.pkl
    current_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(current_path, "wb") as f:
        pickle.dump(model_obj, f)

    # Write a pointer file so we know which version is current
    pointer_path = os.path.join(MODELS_DIR, f"{model_name}_current.txt")
    with open(pointer_path, "w") as f:
        f.write(version_filename)

    # Prune old versions beyond MAX_MODEL_VERSIONS
    _prune_old_versions(model_name)

    return version_path


def load_model(model_name: str):
    """
    Load the current model version.

    Returns:
        (model_obj, metadata) tuple.
        model_obj is whatever was passed to save_model().
        metadata is the dict of training metrics, or {} if none.
    """
    current_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(current_path):
        raise FileNotFoundError(f"No model found at {current_path}")

    with open(current_path, "rb") as f:
        model_obj = pickle.load(f)

    # Try to load metadata from the current version's sidecar
    pointer_path = os.path.join(MODELS_DIR, f"{model_name}_current.txt")
    metadata = {}
    if os.path.exists(pointer_path):
        with open(pointer_path) as f:
            version_filename = f.read().strip()
        meta_path = os.path.join(
            MODEL_VERSIONS_DIR,
            version_filename.replace(".pkl", "_meta.json")
        )
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

    return model_obj, metadata


def rollback_model(model_name: str, steps_back: int = 1) -> str:
    """
    Rollback to a previous model version.

    Args:
        model_name: Base name (e.g. "xgb_model")
        steps_back: How many versions to go back (default 1 = previous)

    Returns:
        The version filename that was restored.
    """
    versions = list_versions(model_name)
    if len(versions) < steps_back + 1:
        raise ValueError(
            f"Cannot rollback {steps_back} steps — only {len(versions)} "
            f"versions available for '{model_name}'"
        )

    target_version = versions[-(steps_back + 1)]  # -1 is current, -2 is previous
    target_path = os.path.join(MODEL_VERSIONS_DIR, target_version)

    with open(target_path, "rb") as f:
        bundle = pickle.load(f)

    # Restore the model to the current pointer
    current_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(current_path, "wb") as f:
        pickle.dump(bundle["model_data"], f)

    # Update pointer
    pointer_path = os.path.join(MODELS_DIR, f"{model_name}_current.txt")
    with open(pointer_path, "w") as f:
        f.write(target_version)

    return target_version


def list_versions(model_name: str) -> list:
    """
    List all saved versions of a model, sorted by timestamp (oldest first).

    Returns:
        List of version filenames.
    """
    _ensure_dirs()
    pattern = os.path.join(MODEL_VERSIONS_DIR, f"{model_name}_*.pkl")
    files = glob.glob(pattern)
    # Filter out metadata JSON files
    pkl_files = [os.path.basename(f) for f in files if f.endswith(".pkl")]
    return sorted(pkl_files)


def _prune_old_versions(model_name: str):
    """Remove versions beyond MAX_MODEL_VERSIONS, keeping the most recent."""
    versions = list_versions(model_name)
    if len(versions) <= MAX_MODEL_VERSIONS:
        return

    to_remove = versions[:len(versions) - MAX_MODEL_VERSIONS]
    for filename in to_remove:
        pkl_path = os.path.join(MODEL_VERSIONS_DIR, filename)
        meta_path = pkl_path.replace(".pkl", "_meta.json")
        if os.path.exists(pkl_path):
            os.remove(pkl_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)
