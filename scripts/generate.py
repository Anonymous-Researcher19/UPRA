from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upra.backbones import PreferenceConditioningAdapter
from upra.config import load_backbone_config, load_config, resolve_path
from upra.data import FeatureDataset, collate_feature_records, move_batch
from upra.model import UPRA
from upra.strategies import prepare_strategy
from upra.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-comparison audio with fixed seeds")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--backbone-config", default=None,
        help="Optional configs/backbones/*.yaml fragment overlaid on --config",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--strategy", required=True, choices=["text_only", "llm_profile", "vanilla_rag", "re_audioldm", "upra"])
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backbone_config:
        cfg = load_backbone_config(cfg, args.backbone_config)
    factory_path = cfg["backbone"].get("factory")
    if not factory_path:
        raise SystemExit("Missing backbone.factory (expected module:function in the config).")
    module_name, function_name = factory_path.split(":", 1)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    backbone = getattr(importlib.import_module(module_name), function_name)(cfg=cfg, device=device)
    dim, native_dim = int(cfg["model"]["feature_dim"]), int(cfg["backbone"]["native_conditioning_dim"])
    upra = UPRA(dim, int(cfg["model"]["num_heads"]), float(cfg["model"]["beta"])).to(device).eval()
    adapter = PreferenceConditioningAdapter(dim, native_dim).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    upra.load_state_dict(state["upra"]); adapter.load_state_dict(state["adapter"])
    dataset = FeatureDataset(resolve_path(cfg, cfg["data"]["records"]), args.split)
    collate = partial(collate_feature_records, top_k=int(cfg["retrieval"]["top_k"]), lexical_weight=float(cfg["retrieval"]["lexical_weight"]))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    output_root = resolve_path(cfg, cfg["output_dir"]) / "audio" / cfg["backbone"]["name"] / args.strategy
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root.parent.parent / f"manifest_{cfg['backbone']['name']}_{args.strategy}.jsonl"
    seed = int(cfg["seed"])
    seed_everything(seed)
    with manifest_path.open("w", encoding="utf-8") as manifest, torch.no_grad():
        for raw in loader:
            batch = move_batch(raw, device)
            output = None
            if args.strategy == "upra":
                output = upra(
                    batch["query"], batch["consolidated_history_text"], batch["history_text"],
                    batch["history_chunks"], batch["history_mask"], batch["history_chunk_mask"],
                    batch["retrieved_chunks"], batch["retrieved_chunk_mask"], variant="full",
                )
            target_id = batch["target_ids"][0] or "target"
            output_path = output_root / f"{batch['user_ids'][0]}__{target_id}__seed{seed}.wav"
            if args.strategy == "re_audioldm":
                backbone.generate_re_audio_ldm(batch, [seed], [str(output_path)], cfg.get("generation", {}))
            else:
                strategy = prepare_strategy(args.strategy, batch, output, float(cfg["model"]["eta"]))
                native = backbone.native_conditioning_texts(strategy.prompts)
                conditioned = native if strategy.preference is None else adapter(native, strategy.preference, strategy.eta)
                backbone.generate(conditioned, batch, [seed], [str(output_path)], cfg.get("generation", {}))
            record = {
                "user_id": batch["user_ids"][0], "target_id": target_id,
                "query": batch["query_texts"][0], "strategy": args.strategy,
                "backbone": cfg["backbone"]["name"], "seed": seed,
                "output_audio": str(output_path.relative_to(manifest_path.parent)),
                "target_path": batch["target_paths"][0],
                "history_paths": batch["history_paths"][0],
                "audio_root": os.path.relpath(resolve_path(cfg, cfg["data"]["audio_root"]), manifest_path.parent),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
