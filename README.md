# UPRA

Code for **UPRA: User-History-Based Personalized Text-to-Music Generation via User Preference
Retrieval and Alignment**. Pretrained generator weights, CLAP/evaluation checkpoints, raw datasets,
generated audio, and trained UPRA checkpoints are not included.

## Repository layout

```text
configs/                         paper hyperparameters and backbone settings
scripts/prepare_benchmark.py     construct the user-disjoint proxy benchmark
scripts/validate_benchmark.py    validate split integrity and paper counts
scripts/build_semantic_text.py   create history summaries and target queries
scripts/precompute_clap.py       extract track/chunk CLAP representations
scripts/materialize_feature_records.py  run HIR and assemble training records
scripts/train.py                 train MPP, IPD, PSA, and conditioning adapters
scripts/generate.py              generate Text-Only/baseline/UPRA outputs
scripts/extract_evaluation_features.py  run frozen evaluation encoders
scripts/evaluate_features.py     compute the reported automatic metrics
upra/                            method, retrieval, data, and metric implementation
```

All paths in the YAML files are relative to the YAML file. External repositories and weights may
therefore be stored beside this repository, for example:

```text
workspace/
├── UPRA_submission/
└── external_models/
    ├── AudioLDM2/
    ├── audiocraft/
    ├── ACE-Step-1.5/
    └── checkpoints/
        ├── audioldm2-music/
        ├── musicgen-small/
        └── acestep-v15-xl-sft/
```

The expected source revisions, model IDs, weight sizes, and SHA-256 values are recorded in
`backbone_manifest.json`. Check local files with:

```bash
python scripts/verify_backbones.py
python scripts/verify_backbones.py --hash
```

The backbone files are fragments over the full experiment configuration. Select one without
editing the default YAML:

```bash
python scripts/train.py --config configs/default.yaml \
  --backbone-config configs/backbones/musicgen_small.yaml
```

Use `audioldm2_music.yaml` or `ace_step_1_5_xl_sft.yaml` for the other paper backbones.

The upstream repositories are kept outside this submission tree. Before training or generation,
set `backbone.factory` (for example with `--set backbone.factory=your_wrapper:create_backbone`) to
the wrapper that exposes the `FrozenBackboneProtocol` in `upra/protocols.py`; the scripts fail fast
if that external integration or the omitted data package is not present.

## Environment

```bash
pip install -r requirements.txt
pip install -e .
```

Audio preprocessing requires `requirements-audio.txt`. AudioLDM2 and MusicGen dependencies are
pinned in `requirements-backbones.txt`. ACE-Step 1.5 uses the `uv.lock` from its own checkout and
should be kept in a separate Python 3.11/3.12 environment.

## Data preparation

The inputs are Music4All interaction metadata, MTG-Jamendo metadata/audio, and FMA-large
metadata/audio. Audio paths are relative to the metadata file or configured `audio_root`.

1. Construct and validate the controlled benchmark:

   ```bash
   python scripts/prepare_benchmark.py \
     --music4all data/raw/music4all.jsonl \
     --mtg-jamendo data/raw/mtg_jamendo.jsonl \
     --output data/processed/benchmark.jsonl

   python scripts/validate_benchmark.py \
     --input data/processed/benchmark.jsonl --paper-counts
   ```

2. Build fixed semantic text from history and target annotations:

   ```bash
   python scripts/build_semantic_text.py \
     --input data/processed/benchmark.jsonl \
     --output data/processed/benchmark_text.jsonl
   ```

3. Precompute Microsoft CLAP 2023 features. Audio is resampled to 44.1 kHz; every track is a
   zero-padded 30-second excerpt split into eleven 5-second chunks with a 2.5-second stride.

   ```bash
   python scripts/precompute_clap.py \
     --tracks data/raw/mtg_jamendo.jsonl \
     --checkpoint assets/checkpoints/CLAP_weights_2023.pth \
     --output data/processed/mtg_clap_features.pt

   python scripts/precompute_clap.py \
     --tracks data/raw/fma_large.jsonl \
     --checkpoint assets/checkpoints/CLAP_weights_2023.pth \
     --output data/processed/fma_clap_features.pt
   ```

4. Materialize top-5 HIR evidence and the MPP/IPD inputs:

   ```bash
   python scripts/materialize_feature_records.py \
     --benchmark data/processed/benchmark_text.jsonl \
     --mtg-features data/processed/mtg_clap_features.pt \
     --fma-features data/processed/fma_clap_features.pt \
     --clap-checkpoint assets/checkpoints/CLAP_weights_2023.pth \
     --output data/processed/feature_records.pt
   ```

The PyTorch feature package has `schema_version: 1` and a `records` list. Each record contains the
user/split identifiers, query, CLAP query/profile/target embeddings, history tracks, HIR candidates,
chunk embeddings, source provenance, and target/history audio paths.

## Backbone interface

The frozen generator is provided through `backbone.factory` in `configs/default.yaml` as
`module:function`. The factory receives `cfg` and `device` and returns an object implementing
`FrozenBackboneProtocol` from `upra/protocols.py`:

- `native_conditioning(batch) -> [batch, sequence, width]`;
- `native_loss(conditioned, batch) -> scalar`, using the native diffusion, autoregressive, or
  flow-matching objective;
- `native_conditioning_dim`, matching the selected backbone configuration.

For generation it additionally implements `native_conditioning_texts(...)` and `generate(...)`.
Backbone parameters must have `requires_grad=False`; gradients still propagate through the frozen
network to the appended preference token. The trainer rejects an unfrozen backbone. Conditioning
dimensions and inference settings are under `configs/backbones/`.

## Training

The paper configuration uses AdamW for 100,000 optimizer steps, learning rate `1e-5`, weight decay
`1e-2`, betas `(0.9, 0.999)`, BF16, `k=5`, HIR weight `0.5`, IPD `beta=0.5`, alignment weight `0.1`,
and preference strength `eta=0.6`. All generator parameters remain frozen.

Launch the four-L40 configuration with PyTorch DDP:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/default.yaml
```

The global batch calculation is checked at startup:

```text
per-device batch 1 × 4 GPUs × 16 accumulation steps = effective batch 64
```

For one GPU, preserve the same global batch with:

```bash
python scripts/train.py --config configs/default.yaml \
  --set training.gradient_accumulation=64
```

ACE-Step XL-SFT should use per-device batch 1 and gradient checkpointing. Missing data, weights, or
a factory cause an explicit error rather than a fallback model.

## Inference

```bash
python scripts/generate.py \
  --config configs/default.yaml \
  --checkpoint outputs/default/last.pt \
  --strategy upra
```

`--strategy` accepts `text_only`, `llm_profile`, `vanilla_rag`, `re_audioldm`, and `upra`. Output
WAV files and a JSONL provenance manifest are written under `output_dir/audio/`.

## Evaluation

```bash
python scripts/extract_evaluation_features.py \
  --manifest outputs/default/audio/manifest_musicgen_upra.jsonl \
  --factory your_evaluation_module:create_backend \
  --output outputs/default/features_musicgen_upra.npz

python scripts/evaluate_features.py \
  --features outputs/default/features_musicgen_upra.npz
```

The implementation reports CLAP-T, CLAP-P, PaSST-P, KL divergence, FAD inputs, and the Audiobox
aggregate defined in the paper. Experiment variants are listed in `configs/experiments/`.
