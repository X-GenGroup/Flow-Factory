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

"""Materialize an enhanced copy of a prompt dataset.

Every rank enhances the rows ``i % world_size == rank`` and appends results to its own
part file, so an interrupted run resumes by skipping the ``(row, variant)`` pairs it
already wrote. Each part file starts with the fingerprint of the settings and source rows
that produced it, and resuming under a different fingerprint fails. The main rank then
writes ``{split}.jsonl`` in source-row order with the variants of each row adjacent; each
output row is an independent training prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from accelerate import PartialState
from PIL import Image

from ..utils.base import isolated_rng
from ..utils.image import require_decoded_rgb_image
from ..utils.logger_utils import setup_logger
from .abc import BasePromptEnhancer
from .args import PromptEnhanceArguments
from .registry import get_prompt_enhancer_class

logger = setup_logger(__name__)

PARTS_DIRNAME = ".parts"
MANIFEST_FILENAME = "prompt_enhance_manifest.json"
MEDIA_DIRNAMES = ("images", "videos", "audios")
SOURCE_PROMPT_KEY = "source_prompt"
VARIANT_KEY = "pe_variant"
FINGERPRINT_KEY = "fingerprint"
DEFAULT_SPLITS = ("train", "test")
SEED_FORMULA = "seed + variant_index * num_source_rows + row_index"


def _split_path(input_dir: str, split: str) -> Optional[Path]:
    """Return ``{split}.jsonl``, else ``{split}.txt``, else ``None``, as ``GeneralDataset`` does."""
    for suffix in (".jsonl", ".txt"):
        path = Path(input_dir) / f"{split}{suffix}"
        if path.is_file():
            return path
    return None


def resolve_splits(input_dir: str, splits: Optional[Sequence[str]]) -> List[str]:
    """Return the splits to enhance, auto-detecting ``train``/``test`` when unset.

    Raises:
        FileNotFoundError: If a requested split, or every default split, is missing.
    """
    candidates = list(DEFAULT_SPLITS) if splits is None else list(splits)
    present = [split for split in candidates if _split_path(input_dir, split) is not None]
    if splits is not None and present != candidates:
        missing = [split for split in candidates if split not in present]
        raise FileNotFoundError(
            f"expected {{split}}.jsonl or {{split}}.txt under {input_dir!r} for splits "
            f"{missing}"
        )
    if not present:
        raise FileNotFoundError(
            f"expected train/test .jsonl or .txt files under {input_dir!r}, found none"
        )
    return present


def load_split_records(input_dir: str, split: str) -> List[Dict[str, Any]]:
    """Load one split as records with a string ``prompt``.

    Raises:
        FileNotFoundError: If the split has no ``.jsonl`` or ``.txt`` file.
        ValueError: If a JSONL row lacks ``prompt``.
        TypeError: If a row is not an object or its ``prompt`` is not a string.
    """
    path = _split_path(input_dir, split)
    if path is None:
        raise FileNotFoundError(f"expected {split}.jsonl or {split}.txt under {input_dir!r}")
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".txt":
            return [{"prompt": line.strip()} for line in f if line.strip()]
        records = [json.loads(line) for line in f if line.strip()]
    for row_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"expected JSON objects in {path}, row {row_index} is "
                f"{type(record).__name__}: {record!r}"
            )
        if "prompt" not in record:
            raise ValueError(f"row {row_index} of {path} has no 'prompt' field")
        if not isinstance(record["prompt"], str):
            raise TypeError(
                f"expected a string prompt in row {row_index} of {path}, "
                f"received {type(record['prompt']).__name__}: {record['prompt']!r}"
            )
    return records


def load_record_images(record: Dict[str, Any], image_dir: str) -> List[Image.Image]:
    """Load a record's reference images the same way ``GeneralDataset`` resolves them."""
    paths = record.get("images", record.get("image"))
    if not paths:
        return []
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        resolved = path if os.path.isabs(path) else os.path.join(image_dir, path)
        with Image.open(resolved) as image:
            decoded = image.convert("RGB")
        images.append(require_decoded_rgb_image(decoded, source=f"prompt-enhance image {path}"))
    return images


def row_seed(base_seed: int, variant_index: int, num_rows: int, row_index: int) -> int:
    """Return the seed of one ``(row, variant)``; it does not depend on batching or sharding."""
    return base_seed + variant_index * num_rows + row_index


def split_fingerprint(
    args: PromptEnhanceArguments, records: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return the settings and source rows that determine one split's enhanced prompts."""
    source = json.dumps(list(records), sort_keys=True, ensure_ascii=False)
    fingerprint = {
        "enhancer": args.enhancer,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
        "dtype": args.dtype,
        "system_prompt": args.system_prompt,
        "generation": args.generation,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    return json.loads(json.dumps(fingerprint, ensure_ascii=False))


def part_file_path(parts_dir: Path, split: str, rank: int) -> Path:
    """Return the resumable result file of one rank and split."""
    return parts_dir / f"{split}.rank{rank:04d}.jsonl"


def read_part_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[Tuple[int, int], str]]:
    """Read the fingerprint header and ``(row, variant) -> prompt`` results of a part file."""
    fingerprint: Optional[Dict[str, Any]] = None
    results: Dict[Tuple[int, int], str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if FINGERPRINT_KEY in item:
                fingerprint = item[FINGERPRINT_KEY]
            else:
                results[(item["row"], item["variant"])] = item["prompt"]
    return fingerprint, results


def read_part_files(paths: Sequence[Path]) -> Dict[Tuple[int, int], str]:
    """Read ``(row, variant) -> prompt`` results from rank part files that exist."""
    results: Dict[Tuple[int, int], str] = {}
    for path in paths:
        if path.exists():
            results.update(read_part_file(path)[1])
    return results


def _open_part_file(part_path: Path, fingerprint: Dict[str, Any]):
    """Return this rank's completed pairs and its part file opened for appending.

    Raises:
        ValueError: If the existing part file was written under a different fingerprint.
    """
    stored, completed = read_part_file(part_path) if part_path.exists() else (None, {})
    if stored is None and completed:
        raise ValueError(f"{part_path} holds results without a fingerprint header")
    if stored is not None and stored != fingerprint:
        changed = sorted(
            key for key in set(stored) | set(fingerprint) if stored.get(key) != fingerprint.get(key)
        )
        raise ValueError(
            f"{part_path} was written with different {changed}; use a new output_dir or "
            f"delete {part_path.parent} to start over"
        )
    part_file = open(part_path, "a", encoding="utf-8")
    if stored is None:
        part_file.write(json.dumps({FINGERPRINT_KEY: fingerprint}, ensure_ascii=False) + "\n")
        part_file.flush()
    return set(completed), part_file


def enhance_split_shard(
    enhancer: BasePromptEnhancer,
    records: Sequence[Dict[str, Any]],
    *,
    split: str,
    image_dir: str,
    parts_dir: Path,
    rank: int,
    world_size: int,
    num_variants: int,
    batch_size: int,
    base_seed: int,
    fingerprint: Dict[str, Any],
) -> Path:
    """Enhance this rank's rows of one split, skipping pairs already in its part file.

    Each call runs under ``isolated_rng`` seeded with the first row's seed and receives the
    seed of every row.

    Returns:
        Path of this rank's part file.

    Raises:
        ValueError: If the part file has a different fingerprint, or the enhancer returns a
            different number of prompts than it was given.
        TypeError: If the enhancer returns a non-string prompt.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_path = part_file_path(parts_dir, split, rank)
    completed, part_file = _open_part_file(part_path, fingerprint)
    rank_rows = [row for row in range(len(records)) if row % world_size == rank]
    with part_file:
        for variant in range(num_variants):
            pending = [row for row in rank_rows if (row, variant) not in completed]
            for start in range(0, len(pending), batch_size):
                batch_rows = pending[start : start + batch_size]
                prompts = [records[row]["prompt"] for row in batch_rows]
                images = [
                    load_record_images(records[row], image_dir) if enhancer.uses_images else []
                    for row in batch_rows
                ]
                seeds = [row_seed(base_seed, variant, len(records), row) for row in batch_rows]
                with isolated_rng(seeds[0]):
                    enhanced = enhancer.enhance(prompts, images, seeds)
                if len(enhanced) != len(batch_rows):
                    raise ValueError(
                        f"{type(enhancer).__name__} returned {len(enhanced)} prompts for a "
                        f"batch of {len(batch_rows)} (split={split!r}, rows={batch_rows})"
                    )
                for row, text in zip(batch_rows, enhanced):
                    if not isinstance(text, str):
                        raise TypeError(
                            f"{type(enhancer).__name__} returned {type(text).__name__} for "
                            f"split={split!r} row={row} variant={variant}, expected str"
                        )
                    part_file.write(
                        json.dumps(
                            {"row": row, "variant": variant, "prompt": text},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                part_file.flush()
    return part_path


def merge_split(
    records: Sequence[Dict[str, Any]],
    *,
    split: str,
    parts_dir: Path,
    world_size: int,
    num_variants: int,
    output_dir: Path,
) -> int:
    """Write ``{split}.jsonl`` with every variant of every row, in row-major order.

    Returns:
        Number of written rows.

    Raises:
        RuntimeError: If a ``(row, variant)`` result is missing from the part files.
    """
    results = read_part_files(
        [part_file_path(parts_dir, split, rank) for rank in range(world_size)]
    )
    missing = [
        (row, variant)
        for row in range(len(records))
        for variant in range(num_variants)
        if (row, variant) not in results
    ]
    if missing:
        raise RuntimeError(
            f"prompt enhancement of split={split!r} is incomplete: {len(missing)} "
            f"(row, variant) results missing, first {missing[:5]}"
        )
    output_path = output_dir / f"{split}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row, record in enumerate(records):
            for variant in range(num_variants):
                enhanced_record = dict(record)
                enhanced_record["prompt"] = results[(row, variant)]
                enhanced_record[SOURCE_PROMPT_KEY] = record["prompt"]
                enhanced_record[VARIANT_KEY] = variant
                f.write(json.dumps(enhanced_record, ensure_ascii=False) + "\n")
    return len(records) * num_variants


def link_media_dirs(input_dir: Path, output_dir: Path) -> List[str]:
    """Symlink the source media directories so relative media paths keep resolving.

    Returns:
        Names of the linked directories.

    Raises:
        FileExistsError: If the output already holds a different entry with that name.
    """
    linked = []
    for name in MEDIA_DIRNAMES:
        source = input_dir / name
        if not source.is_dir():
            continue
        target = output_dir / name
        if target.is_symlink() and target.resolve() == source.resolve():
            linked.append(name)
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"cannot link media directory {source} to {target}: the target already exists"
            )
        target.symlink_to(source.resolve(), target_is_directory=True)
        linked.append(name)
    return linked


def write_manifest(
    args: PromptEnhanceArguments,
    enhancer: BasePromptEnhancer,
    output_dir: Path,
    split_rows: Dict[str, Dict[str, int]],
    linked_media: Sequence[str],
) -> Path:
    """Record the run configuration and output sizes next to the dataset."""
    system_prompt_sha256 = (
        None
        if args.system_prompt is None
        else hashlib.sha256(args.system_prompt.encode("utf-8")).hexdigest()
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "enhancer": args.enhancer,
        "enhancer_class": f"{type(enhancer).__module__}.{type(enhancer).__qualname__}",
        "model_name_or_path": args.model_name_or_path,
        "num_variants": args.num_variants,
        "seed": args.seed,
        "seed_formula": SEED_FORMULA,
        "system_prompt_sha256": system_prompt_sha256,
        "generation": args.generation,
        "enhancer_details": enhancer.describe(),
        "source_dir": os.path.abspath(args.input_dir),
        "linked_media": list(linked_media),
        "splits": split_rows,
        "config": args.to_dict(),
    }
    path = output_dir / MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def run_prompt_enhancement(
    args: PromptEnhanceArguments,
    state: Optional[PartialState] = None,
) -> Path:
    """Enhance every selected split and write the enhanced dataset.

    Args:
        args: Prompt-enhancement configuration.
        state: Distributed state; defaults to the process's Accelerate ``PartialState``.

    Returns:
        Output dataset directory.
    """
    state = PartialState() if state is None else state
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError(f"output_dir must differ from input_dir, both are {input_dir}")
    parts_dir = output_dir / PARTS_DIRNAME
    splits = resolve_splits(args.input_dir, args.splits)
    records_by_split = {split: load_split_records(args.input_dir, split) for split in splits}

    enhancer_class = get_prompt_enhancer_class(args.enhancer)
    enhancer = enhancer_class(args, device=state.device)
    enhancer.load()
    for split, records in records_by_split.items():
        logger.info(
            "Enhancing split=%s rows=%d variants=%d on rank %d/%d",
            split,
            len(records),
            args.num_variants,
            state.process_index,
            state.num_processes,
        )
        enhance_split_shard(
            enhancer,
            records,
            split=split,
            image_dir=str(input_dir / "images"),
            parts_dir=parts_dir,
            rank=state.process_index,
            world_size=state.num_processes,
            num_variants=args.num_variants,
            batch_size=args.batch_size,
            base_seed=args.seed,
            fingerprint=split_fingerprint(args, records),
        )
    state.wait_for_everyone()

    if state.is_main_process:
        split_rows = {}
        for split, records in records_by_split.items():
            output_rows = merge_split(
                records,
                split=split,
                parts_dir=parts_dir,
                world_size=state.num_processes,
                num_variants=args.num_variants,
                output_dir=output_dir,
            )
            split_rows[split] = {"source_rows": len(records), "output_rows": output_rows}
        linked_media = link_media_dirs(input_dir, output_dir)
        manifest_path = write_manifest(args, enhancer, output_dir, split_rows, linked_media)
        logger.info("Wrote enhanced dataset to %s (manifest: %s)", output_dir, manifest_path)
    state.wait_for_everyone()
    return output_dir
