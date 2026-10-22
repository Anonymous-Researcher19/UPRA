from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def absolute(base: Path, path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else (base / value).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract every Appendix-B.2 metric feature from a generation manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--factory", required=True, help="Evaluation backend module:function")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("generation manifest is empty")
    module_name, function_name = args.factory.split(":", 1)
    backend = getattr(importlib.import_module(module_name), function_name)()
    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "clap_generated", "clap_text", "clap_history", "passt_generated", "passt_history",
            "passt_generated_posterior", "passt_history_posteriors", "vggish_generated",
            "vggish_targets", "production_quality", "content_enjoyment", "content_usefulness",
        )
    }
    with torch.no_grad():
        for row in rows:
            generated = absolute(manifest.parent, row["output_audio"])
            audio_root = (manifest.parent / row.get("audio_root", ".")).resolve()
            history = [absolute(audio_root, path) for path in row["history_paths"]]
            target = absolute(audio_root, row["target_path"])
            collected["clap_generated"].append(backend.clap_audio([generated])[0].cpu().numpy())
            collected["clap_text"].append(backend.clap_text([row["query"]])[0].cpu().numpy())
            collected["clap_history"].append(backend.clap_audio(history).mean(0, keepdim=True).cpu().numpy())
            collected["passt_generated"].append(backend.passt_embedding([generated])[0].cpu().numpy())
            collected["passt_history"].append(backend.passt_embedding(history).mean(0, keepdim=True).cpu().numpy())
            collected["passt_generated_posterior"].append(backend.passt_posterior([generated])[0].cpu().numpy())
            collected["passt_history_posteriors"].append(backend.passt_posterior(history).mean(0, keepdim=True).cpu().numpy())
            collected["vggish_generated"].append(backend.vggish([generated])[0].cpu().numpy())
            collected["vggish_targets"].append(backend.vggish([target])[0].cpu().numpy())
            scores = backend.audiobox([generated])
            for key in ("production_quality", "content_enjoyment", "content_usefulness"):
                collected[key].append(np.asarray(float(scores[key][0])))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{key: np.stack(values) for key, values in collected.items()})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
