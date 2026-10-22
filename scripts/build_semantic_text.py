from __future__ import annotations

import argparse
import json
from pathlib import Path


HISTORY_INSTRUCTIONS = "You summarize semantic annotations from a user's historical music collection."
HISTORY_TASK = (
    "Given track-level genre, mood, instrument, and theme tags, return a compact comma-separated "
    "list of representative user history tags. Use only supported attributes; prioritize recurring "
    "attributes; merge duplicates and obvious synonyms; retain distinct recurring preferences; infer "
    "no demographics or unsupported details; return only the final list.\n\n"
)
QUERY_INSTRUCTIONS = "You format annotation tags as text-to-music generation prompts."
QUERY_TASK = (
    "Convert the provided genre, mood, instrument, or theme tags into one concise and grammatically "
    "complete English music description. Preserve their meaning, add no unsupported musical details, "
    "use at most 20 words, and return only the final description.\n\nTags: "
)


def call(client, model: str, instructions: str, text: str) -> str:
    response = client.responses.create(
        model=model, instructions=instructions, input=text,
        reasoning={"effort": "none"}, temperature=0, max_output_tokens=128,
    )
    return response.output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the two fixed Appendix-C formatting operations")
    parser.add_argument("--input", required=True, help="Benchmark JSONL from prepare_benchmark.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.2-2025-12-11")
    args = parser.parse_args()
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Install the optional 'openai' package before running this script") from error
    client = OpenAI()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open("r", encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for line in source:
            row = json.loads(line)
            history_lines = [f"Track {i + 1}: {', '.join(track['tags'])}" for i, track in enumerate(row["history"])]
            row["history_profile"] = call(client, args.model, HISTORY_INSTRUCTIONS, HISTORY_TASK + "\n".join(history_lines))
            for target in row["targets"]:
                target["query"] = call(client, args.model, QUERY_INSTRUCTIONS, QUERY_TASK + ", ".join(target["tags"]))
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
