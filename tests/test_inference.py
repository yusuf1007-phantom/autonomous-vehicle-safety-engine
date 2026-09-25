import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    analyze_scenario,
    get_model_info
)


def main():

    info = get_model_info()

    print("\nAV SAFETY ENGINE")
    print("=" * 60)

    print("Model:", info["name"])
    print("Dataset:", info["dataset"])
    print("Threshold:", info["threshold"])
    print("ROC-AUC:", info["roc_auc"])
    print("PR-AUC:", info["pr_auc"])

    print("\nModel loaded successfully.")


if __name__ == "__main__":
    main()