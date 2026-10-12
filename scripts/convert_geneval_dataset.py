#!/usr/bin/env python3
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert DiffusionNFT GenEval dataset to Flow-Factory format.

Input format (DiffusionNFT):
    {"tag": "...", "include": [...], "prompt": "...", "exclude": [...]}

Output format (Flow-Factory):
    {"prompt": "...", "include": "<JSON string>", "exclude": "<JSON string>", "tag": "..."}

The include/exclude fields are stored as JSON strings to avoid Arrow
serialization issues with heterogeneous nested structs.

Usage:
    python scripts/convert_geneval_dataset.py
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SCRATCH_DIR = os.path.join(REPO_ROOT, ".scratch")
OUTPUT_DIR = os.path.join(REPO_ROOT, "dataset", "geneval")


def process_geneval(input_path: str, output_path: str) -> tuple:
    """Deduplicate by prompt and convert to Flow-Factory format."""
    seen_prompts = set()
    unique_records = []
    duplicates = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record["prompt"]
            if prompt in seen_prompts:
                duplicates += 1
                continue
            seen_prompts.add(prompt)
            # Store include/exclude as JSON strings (Arrow-safe)
            output_record = {
                "prompt": prompt,
                "include": json.dumps(record["include"], ensure_ascii=False),
                "exclude": json.dumps(record.get("exclude"), ensure_ascii=False),
                "tag": record.get("tag", "unknown"),
            }
            unique_records.append(output_record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for rec in unique_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(unique_records), duplicates


def main():
    train_input = os.path.join(SCRATCH_DIR, "geneval_train_raw.jsonl")
    test_input = os.path.join(SCRATCH_DIR, "geneval_test_raw.jsonl")
    train_output = os.path.join(OUTPUT_DIR, "train.jsonl")
    test_output = os.path.join(OUTPUT_DIR, "test.jsonl")

    if not os.path.exists(train_input):
        print(f"ERROR: {train_input} not found.")
        print("Download first with:")
        print(
            "  curl -sL https://raw.githubusercontent.com/NVlabs/DiffusionNFT/main/dataset/geneval/train_metadata.jsonl"
            f" -o {train_input}"
        )
        return

    print("Processing GenEval dataset...")

    train_unique, train_dups = process_geneval(train_input, train_output)
    print(f"  Train: {train_unique} unique prompts, {train_dups} duplicates removed")

    if os.path.exists(test_input):
        test_unique, test_dups = process_geneval(test_input, test_output)
        print(f"  Test:  {test_unique} unique prompts, {test_dups} duplicates removed")

    print("Done!")


if __name__ == "__main__":
    main()
