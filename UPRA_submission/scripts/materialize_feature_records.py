from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upra.library import RetrievalLibrary
from upra.retrieval import hybrid_intent_retrieval
from upra.text import semantic_tokens


def cpu_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return value


def serial_track(track_id: str, item: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "track_id": str(track_id), "source": source, "path": item.get("path"),
        "tags": item["tags"], "text_embed": cpu_tensor(item["text_embed"]),
        "audio_embed": cpu_tensor(item["audio_embed"]), "chunks": cpu_tensor(item["chunks"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the complete, self-contained UPRA feature-record file")
    parser.add_argument("--benchmark", required=True, help="Appendix-C enriched benchmark JSONL")
    parser.add_argument("--mtg-features", required=True)
    parser.add_argument("--fma-features", required=True)
    parser.add_argument("--clap-checkpoint", required=True)
    parser.add_argument("--output", default="data/processed/feature_records.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lexical-weight", type=float, default=0.5)
    parser.add_argument("--candidate-source", choices=["both", "history_only", "external_only"], default="both")
    args = parser.parse_args()
    try:
        import laion_clap
    except ImportError as error:
        raise SystemExit("Install laion-clap before materializing publication features") from error
    device = args.device if torch.cuda.is_available() else "cpu"
    mtg = torch.load(args.mtg_features, map_location="cpu", weights_only=False)
    fma = torch.load(args.fma_features, map_location="cpu", weights_only=False)
    library = RetrievalLibrary(fma, device=device)
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    clap.load_ckpt(ckpt=args.clap_checkpoint); clap.eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with Path(args.benchmark).open("r", encoding="utf-8") as source, torch.no_grad():
        for line in source:
            profile = json.loads(line)
            history = []
            history_candidates = []
            for track in profile["history"]:
                track_id = str(track["track_id"])
                item = serial_track(track_id, mtg[track_id], "history")
                history.append(item)
                history_candidates.append(item)
            profile_embedding = clap.get_text_embedding([profile["history_profile"]], use_tensor=True)[0].cpu()
            queries = [target["query"] for target in profile["targets"]]
            query_embeddings = clap.get_text_embedding(queries, use_tensor=True).cpu()
            for target, query_embedding in zip(profile["targets"], query_embeddings):
                if args.candidate_source == "history_only":
                    history_audio = torch.stack([torch.as_tensor(item["audio_embed"]) for item in history_candidates])
                    local = hybrid_intent_retrieval(
                        query_embedding, history_audio, semantic_tokens(target["query"]),
                        [item["tags"] for item in history_candidates], args.top_k, args.lexical_weight,
                    )
                    selected_items = [history_candidates[index] for index in local.indices.tolist()]
                    trace = {
                        "selected_global_indices": local.indices.tolist(), "scores": local.scores.tolist(),
                        "implicit": local.implicit[local.indices].tolist(),
                        "explicit": local.explicit[local.indices].tolist(),
                        "sources": ["history"] * len(local.indices), "library_size": 0,
                        "history_candidates": len(history_candidates),
                    }
                else:
                    extras = [] if args.candidate_source == "external_only" else history_candidates
                    result = library.query(
                        query_embedding, semantic_tokens(target["query"]), extras,
                        top_k=args.top_k, lexical_weight=args.lexical_weight,
                    )
                    selected_items, trace = result.candidates, result.trace
                target_id = str(target["track_id"])
                selected = []
                for item, source_name in zip(selected_items, trace["sources"]):
                    selected.append(serial_track(str(item["track_id"]), item, source_name))
                record = {
                    "schema_version": 1, "user_id": str(profile["user_id"]), "split": profile["split"],
                    "target_id": target_id, "query": target["query"],
                    "query_tokens": semantic_tokens(target["query"]),
                    "target_path": mtg[target_id].get("path"),
                    "query_embed": cpu_tensor(query_embedding), "history_profile": profile["history_profile"],
                    "history_profile_embed": cpu_tensor(profile_embedding),
                    "target_audio_embed": cpu_tensor(mtg[target_id]["audio_embed"]),
                    "history": history, "retrieved_candidates": selected, "retrieval": trace,
                }
                records.append(record)
    torch.save({"schema_version": 1, "records": records}, output)
    print(f"wrote feature records to {output}")


if __name__ == "__main__":
    main()
