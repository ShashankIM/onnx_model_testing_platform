from pathlib import Path
from typing import List, Dict


# Base dir = project_root/app/.. (same logic as in main.py)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DET_MODELS_DIR = MODELS_DIR / "detection"
CLS_MODELS_DIR = MODELS_DIR / "classification"

DET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
CLS_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _list_models_in_dir(models_dir: Path) -> List[str]:
    """
    List model base names in a directory.

    A "model" is:
      - any non-.txt file (onnx, pt, etc.)
      - for which a <basename>.txt labels file also exists
    """
    names = set()

    for path in models_dir.glob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".txt":
            continue  # skip pure label files

        base = path.stem
        labels_path = models_dir / (base + ".txt")
        if labels_path.exists():
            names.add(base)

    return sorted(names)


def list_detection_models() -> List[str]:
    """Return list of detection model names (base names)."""
    return _list_models_in_dir(DET_MODELS_DIR)


def list_classification_models() -> List[str]:
    """Return list of classification model names (base names)."""
    return _list_models_in_dir(CLS_MODELS_DIR)


def _get_model_paths(models_dir: Path, name: str) -> Dict[str, str]:
    """
    Find model file and labels file for a given base name.
    Returns dict with 'model_path' and 'labels_path' (as strings).
    """
    # find any non-txt file whose stem == name
    model_path = None
    for p in models_dir.glob(name + ".*"):
        if p.suffix.lower() == ".txt":
            continue
        model_path = p
        break

    if model_path is None:
        raise FileNotFoundError("Model file for '%s' not found in %s" % (name, models_dir))

    labels_path = models_dir / (name + ".txt")
    if not labels_path.exists():
        raise FileNotFoundError("Labels file '%s.txt' not found in %s" % (name, models_dir))

    return {
        "model_path": str(model_path),
        "labels_path": str(labels_path),
    }


def get_detection_model_config(name: str) -> Dict[str, str]:
    """Return paths for a detection model by name."""
    return _get_model_paths(DET_MODELS_DIR, name)


def get_classification_model_config(name: str) -> Dict[str, str]:
    """Return paths for a classification model by name."""
    return _get_model_paths(CLS_MODELS_DIR, name)


def _read_labels(labels_path: Path) -> List[str]:
    """Read labels from a .txt file, stripping whitespace and skipping empty lines."""
    labels: List[str] = []
    try:
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                labels.append(line)
    except FileNotFoundError:
        pass
    return labels


def get_detection_labels() -> Dict[str, List[str]]:
    """
    Return {model_name: [label1, label2, ...]} for all detection models.
    """
    result: Dict[str, List[str]] = {}
    for name in list_detection_models():
        labels_path = DET_MODELS_DIR / (name + ".txt")
        result[name] = _read_labels(labels_path)
    return result


def get_classification_labels() -> Dict[str, List[str]]:
    """
    Return {model_name: [label1, label2, ...]} for all classification models.
    """
    result: Dict[str, List[str]] = {}
    for name in list_classification_models():
        labels_path = CLS_MODELS_DIR / (name + ".txt")
        result[name] = _read_labels(labels_path)
    return result
