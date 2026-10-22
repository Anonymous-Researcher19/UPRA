from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from upra.metrics import audiobox_score, cosine_rows, frechet_distance, kl_divergence, preference_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the paper metrics from extracted model features")
    parser.add_argument("--features", required=True, help="NPZ; see data/README.md")
    parser.add_argument("--output", default="outputs/metrics.json")
    args = parser.parse_args()
    data = np.load(args.features)
    metrics = {
        "CLAP-T": float(cosine_rows(data["clap_generated"], data["clap_text"]).mean()),
        "CLAP-P": float(preference_similarity(data["clap_generated"], data["clap_history"]).mean()),
        "PaSST-P": float(preference_similarity(data["passt_generated"], data["passt_history"]).mean()),
        "KL-Div": float(kl_divergence(data["passt_generated_posterior"], data["passt_history_posteriors"]).mean()),
        "FAD": frechet_distance(data["vggish_generated"], data["vggish_targets"]),
        "AudBox": float(audiobox_score(data["production_quality"], data["content_enjoyment"], data["content_usefulness"]).mean()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
