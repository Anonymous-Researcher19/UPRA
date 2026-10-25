from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from upra.backbones import PreferenceConditioningAdapter, total_objective
from upra.config import load_backbone_config, load_config, resolve_path, with_overrides
from upra.data import FeatureDataset, collate_feature_records, move_batch
from upra.model import UPRA
from upra.protocols import FrozenBackboneProtocol
from upra.utils import seed_everything


def _distributed_device(cfg: dict[str, Any]) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=cfg["training"].get("distributed_backend", "nccl"))
        return torch.device("cuda", local_rank), rank, local_rank, world_size
    requested = str(cfg.get("device", "cuda"))
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    return device, rank, local_rank, world_size


def _load_backbone(cfg: dict[str, Any], device: torch.device) -> FrozenBackboneProtocol:
    factory_path = cfg["backbone"].get("factory")
    if not factory_path:
        raise ValueError(
            "backbone.factory is required (module:function); it must implement "
            "FrozenBackboneProtocol for the selected upstream checkout"
        )
    module_name, function_name = factory_path.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    backbone = factory(cfg=cfg, device=device)
    required = ("native_conditioning_dim", "native_conditioning", "native_loss")
    missing = [name for name in required if not hasattr(backbone, name)]
    if missing:
        raise TypeError(f"{factory_path} is missing backbone interface members: {missing}")
    if int(backbone.native_conditioning_dim) != int(cfg["backbone"]["native_conditioning_dim"]):
        raise ValueError("Configured native_conditioning_dim does not match the backbone wrapper")
    if isinstance(backbone, nn.Module):
        backbone.to(device).eval()
        trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
        if trainable:
            raise ValueError(f"The generative backbone must be frozen; trainable parameters: {trainable[:5]}")
    checkpointing = getattr(backbone, "set_gradient_checkpointing", None)
    if cfg["training"].get("gradient_checkpointing", False) and callable(checkpointing):
        checkpointing(True)
    return backbone


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UPRA with a frozen music-generation backbone")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--backbone-config", default=None,
        help="Optional configs/backbones/*.yaml fragment overlaid on --config",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backbone_config:
        cfg = load_backbone_config(cfg, args.backbone_config)
    cfg = with_overrides(cfg, args.set)
    device, rank, local_rank, world_size = _distributed_device(cfg)
    seed_everything(int(cfg["seed"]) + rank)

    dataset = FeatureDataset(resolve_path(cfg, cfg["data"]["records"]), cfg["data"].get("split", "train"))
    sampler = DistributedSampler(dataset, shuffle=True, seed=int(cfg["seed"])) if world_size > 1 else None
    collate = partial(
        collate_feature_records,
        top_k=int(cfg["retrieval"]["top_k"]),
        lexical_weight=float(cfg["retrieval"]["lexical_weight"]),
    )
    per_device_batch = int(cfg["training"]["per_device_batch_size"])
    accumulation = int(cfg["training"]["gradient_accumulation"])
    effective_batch = per_device_batch * world_size * accumulation
    expected_batch = int(cfg["training"]["effective_batch_size"])
    if effective_batch != expected_batch:
        raise ValueError(
            "per_device_batch_size * WORLD_SIZE * gradient_accumulation must equal "
            f"effective_batch_size ({per_device_batch} * {world_size} * {accumulation} "
            f"= {effective_batch}, expected {expected_batch})"
        )
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(cfg["training"].get("workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    dim = int(cfg["model"]["feature_dim"])
    native_dim = int(cfg["backbone"]["native_conditioning_dim"])
    upra: nn.Module = UPRA(dim, int(cfg["model"]["num_heads"]), float(cfg["model"]["beta"])).to(device)
    adapter: nn.Module = PreferenceConditioningAdapter(dim, native_dim).to(device)
    backbone = _load_backbone(cfg, device)
    if world_size > 1:
        has_ablation = cfg.get("experiment", {}).get("variant", "full") != "full"
        upra = DistributedDataParallel(
            upra, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=has_ablation,
        )
        adapter = DistributedDataParallel(
            adapter, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=has_ablation,
        )

    optimizer = torch.optim.AdamW(
        list(upra.parameters()) + list(adapter.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        betas=tuple(float(value) for value in cfg["training"].get("adam_betas", (0.9, 0.999))),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    max_steps = int(cfg["training"]["steps"])
    eta = float(cfg["model"]["eta"])
    lambda_align = float(cfg["model"]["lambda_align"])
    use_bfloat16 = cfg["training"].get("precision") == "bfloat16" and device.type == "cuda"
    upra.train()
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    update_step = micro_step = epoch = 0
    logs: list[dict[str, float | int]] = []

    try:
        while update_step < max_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                will_update = (micro_step + 1) % accumulation == 0
                with ExitStack() as stack:
                    if world_size > 1 and not will_update:
                        stack.enter_context(upra.no_sync())
                        stack.enter_context(adapter.no_sync())
                    stack.enter_context(
                        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16)
                    )
                    output = upra(
                        batch["query"], batch["consolidated_history_text"], batch["history_text"],
                        batch["history_chunks"], batch["history_mask"], batch["history_chunk_mask"],
                        batch["retrieved_chunks"], batch["retrieved_chunk_mask"],
                        variant=cfg.get("experiment", {}).get("variant", "full"),
                    )
                    native = backbone.native_conditioning(batch)
                    if native.ndim != 3 or native.shape[0] != batch["query"].shape[0]:
                        raise ValueError("native_conditioning must return [batch, sequence, width]")
                    if native.shape[-1] != native_dim:
                        raise ValueError(
                            f"native conditioning width {native.shape[-1]} does not match {native_dim}"
                        )
                    conditioned = adapter(native, output.personalized, eta)
                    generation_loss = backbone.native_loss(conditioned, batch)
                    loss = total_objective(generation_loss, output.alignment_loss, lambda_align)
                    (loss / accumulation).backward()
                micro_step += 1
                if not will_update:
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                record = {
                    "step": update_step,
                    "micro_step": micro_step,
                    "loss": float(loss.detach()),
                    "generation_loss": float(generation_loss.detach()),
                    "alignment_loss": float(output.alignment_loss.detach()),
                }
                logs.append(record)
                if rank == 0:
                    print(json.dumps(record), flush=True)
                if update_step >= max_steps:
                    break
            epoch += 1

        if rank == 0:
            output_dir = resolve_path(cfg, cfg["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "upra": _unwrap(upra).state_dict(),
                    "adapter": _unwrap(adapter).state_dict(),
                    "config": cfg,
                    "world_size": world_size,
                },
                output_dir / "last.pt",
            )
            with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as handle:
                for record in logs:
                    handle.write(json.dumps(record) + "\n")
            print(f"saved checkpoint to {output_dir / 'last.pt'}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
