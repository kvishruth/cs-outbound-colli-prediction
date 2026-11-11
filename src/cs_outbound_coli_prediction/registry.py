"""Saving versions of the trained models, with what is needed to reproduce and check them."""
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version as library_version

import joblib

from . import config


def save_version(models, input_path, deliveries, metrics, models_dir, tables=None):
    """Write the models, a metadata.json and the backtest tables to a new folder named after the training time.

    Returns the version.
    """
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    folder = models_dir / version
    folder.mkdir(parents=True)
    joblib.dump(models, folder / "models.joblib")

    metadata = {
        "version": version,
        "git_commit": _git_commit(),
        "data": {
            "file": input_path.name,
            "sha256": _file_hash(input_path),
            "rows": len(deliveries),
            "first_week": str(deliveries["Monday_of_week"].min().date()),
            "last_week": str(deliveries["Monday_of_week"].max().date()),
        },
        "features": config.feature_columns,
        "lgb_params": config.lgb_params,
        "xgb_params": config.xgb_params,
        "metrics": metrics,
        "libraries": _installed_libraries() | {"python": platform.python_version()},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for name, table in (tables or {}).items():
        (folder / "backtest").mkdir(exist_ok=True)
        table.to_csv(folder / "backtest" / f"by_{name}.csv")
    return version


def promote(version, models_dir):
    """Make `version` the one the forecast uses. To roll back, promote an older version."""
    (models_dir / "current.txt").write_text(version)


def load_current(models_dir):
    """The models and the metadata of the promoted version.

    Raises a ValueError, before the models are read, if a library differs from the one the version was trained with.
    """
    version = (models_dir / "current.txt").read_text().strip()
    metadata = json.loads((models_dir / version / "metadata.json").read_text())
    _check_libraries(metadata["libraries"], version)
    models = joblib.load(models_dir / version / "models.joblib")
    return models, metadata


def _installed_libraries():
    return {name: library_version(name) for name in ("lightgbm", "xgboost", "pandas", "numpy", "joblib")}


def _check_libraries(trained_with, version):
    """A saved model can predict differently, or fail to load, with other library versions."""
    installed = _installed_libraries()
    different = [f"{name}: trained with {trained_with.get(name)}, installed {found}"
                 for name, found in installed.items() if trained_with.get(name) != found]
    if different:
        raise ValueError(f"version {version} was trained with other libraries (retrain, or install these):\n- "
                         + "\n- ".join(different))


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit():
    """Current commit, with '-dirty' if there are uncommitted changes; 'unknown' without git history."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=config.root)
        changes = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=config.root)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit.stdout.strip() + ("-dirty" if changes.stdout.strip() else "")
