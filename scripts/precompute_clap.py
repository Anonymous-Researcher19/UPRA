from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def clean_tags(tags: list[str]) -> str:
    prefixes = ("genre---", "instrument---", "mood/theme---", "theme---", "mood---")
    values = []
    for tag in tags:
        for prefix in prefixes:
            tag = tag.replace(prefix, "")
        values.append(tag)
    return ", ".join(values)


def chunks(path: str, sample_rate: int, duration: float, stride: float) -> list[np.ndarray]:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True, duration=30.0)
    excerpt_width = int(30.0 * sample_rate)
    if len(audio) < excerpt_width:
        audio = np.pad(audio, (0, excerpt_width - len(audio)))
    else:
        audio = audio[:excerpt_width]
    width, hop = int(duration * sample_rate), int(stride * sample_rate)
    return [audio[start:start + width] for start in range(0, len(audio) - width + 1, hop)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the frozen Microsoft CLAP 2023 features")
    parser.add_argument("--tracks", required=True, help="JSONL with track_id,path,tags")
    parser.add_argument("--checkpoint", required=True, help="Path to CLAP_weights_2023.pth")
    parser.add_argument("--output", default="data/processed/clap_features.pt")
    parser.add_argument("--audio-root", default=None, help="Root used to resolve and store relative audio paths")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()
    try:
        import torch
        import laion_clap
    except ImportError as error:
        raise SystemExit("Install torch, librosa, and laion-clap for audio preprocessing") from error
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=args.device)
    model.load_ckpt(ckpt=args.checkpoint)
    model.eval()
    features = {}
    metadata_path = Path(args.tracks).resolve()
    audio_root = Path(args.audio_root).resolve() if args.audio_root else metadata_path.parent
    with Path(args.tracks).open("r", encoding="utf-8") as handle, torch.no_grad():
        for line in handle:
            row = json.loads(line)
            declared = Path(row["path"])
            actual = declared if declared.is_absolute() else audio_root / declared
            try:
                stored_path = actual.resolve().relative_to(audio_root).as_posix()
            except ValueError as error:
                raise ValueError(f"Audio path must be under --audio-root: {actual}") from error
            audio_chunks = chunks(str(actual), args.sample_rate, duration=5.0, stride=2.5)
            if not audio_chunks:
                continue
            chunk_embeddings = model.get_audio_embedding_from_data(x=audio_chunks, use_tensor=True).cpu()
            text_embedding = model.get_text_embedding([clean_tags(row["tags"])], use_tensor=True)[0].cpu()
            features[str(row["track_id"])] = {
                "path": stored_path, "tags": row["tags"], "text_embed": text_embedding,
                "chunks": chunk_embeddings, "audio_embed": chunk_embeddings.mean(0),
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, output)
    print(f"saved {len(features)} tracks to {output}")


if __name__ == "__main__":
    main()
