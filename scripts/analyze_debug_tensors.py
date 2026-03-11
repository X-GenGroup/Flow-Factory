#!/usr/bin/env python
"""
Analyze debug tensors dumped by Flow-Factory GRPO train-inference consistency debugging.

This script compares per-step tensors saved during sampling and training
(organized by rank and batch_idx) to find where the first inconsistency appears.

Usage:
    # Analyze all ranks and batches:
    python scripts/analyze_debug_tensors.py /path/to/debug_output

    # Analyze a specific rank and batch:
    python scripts/analyze_debug_tensors.py /path/to/debug_output --rank 0 --batch 0 -v

    # Custom threshold:
    python scripts/analyze_debug_tensors.py /path/to/debug_output --threshold 1e-5

Expected directory structure:
    {debug_dir}/
    ├── sampling/
    │   ├── config.json
    │   ├── rank_000/
    │   │   ├── batch_000/
    │   │   │   ├── metadata.pt
    │   │   │   ├── step_000/{noise_pred,latents,...}.pt
    │   │   │   └── step_001/...
    │   │   └── batch_001/...
    │   └── rank_001/...
    └── training/
        ├── rank_000/
        │   ├── batch_000/
        │   │   ├── metadata.pt
        │   │   ├── step_000/{noise_pred,latents,...}.pt
        │   │   └── step_001/...
        │   └── batch_001/...
        └── rank_001/...
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch


# Tensor names that should match between sampling and training
COMPARISON_TENSORS = [
    "noise_pred",
    "latents",
    "next_latents",
    "next_latents_mean",
    "sigma",
    "sigma_prev",
]

# Training-only tensors for diagnostics
TRAINING_ONLY_TENSORS = [
    "old_log_prob",
    "new_log_prob",
    "ratio",
]


def load_step_tensors(step_dir: str) -> Dict[str, torch.Tensor]:
    """Load all .pt tensors from a step directory."""
    tensors = {}
    if not os.path.isdir(step_dir):
        return tensors
    for fname in sorted(os.listdir(step_dir)):
        if fname.endswith(".pt"):
            name = fname[:-3]
            tensors[name] = torch.load(os.path.join(step_dir, fname), map_location="cpu")
    return tensors


def load_metadata(batch_dir: str) -> Optional[Dict]:
    """Load metadata.pt from a batch directory."""
    path = os.path.join(batch_dir, "metadata.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def discover_ranks(base_dir: str) -> List[int]:
    """Discover all rank_XXX directories."""
    ranks = []
    if not os.path.isdir(base_dir):
        return ranks
    for entry in sorted(os.listdir(base_dir)):
        if entry.startswith("rank_") and os.path.isdir(os.path.join(base_dir, entry)):
            try:
                ranks.append(int(entry.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(ranks)


def discover_batches(rank_dir: str) -> List[int]:
    """Discover all batch_XXX directories under a rank directory."""
    batches = []
    if not os.path.isdir(rank_dir):
        return batches
    for entry in sorted(os.listdir(rank_dir)):
        if entry.startswith("batch_") and os.path.isdir(os.path.join(rank_dir, entry)):
            try:
                batches.append(int(entry.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(batches)


def discover_steps(batch_dir: str) -> List[int]:
    """Discover all step_XXX directories under a batch directory."""
    steps = []
    if not os.path.isdir(batch_dir):
        return steps
    for entry in sorted(os.listdir(batch_dir)):
        if entry.startswith("step_") and os.path.isdir(os.path.join(batch_dir, entry)):
            try:
                steps.append(int(entry.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(steps)


def tensor_stats(t: torch.Tensor) -> str:
    """Return a compact stats string for a tensor."""
    if t.numel() == 0:
        return "empty"
    t_f = t.float()
    return (
        f"shape={list(t.shape)} dtype={t.dtype} "
        f"min={t_f.min().item():.6e} max={t_f.max().item():.6e} "
        f"mean={t_f.mean().item():.6e} std={t_f.std().item():.6e}"
    )


def compare_tensors(
    name: str,
    sampling_t: Optional[torch.Tensor],
    training_t: Optional[torch.Tensor],
) -> Dict[str, float]:
    """Compare two tensors and return difference metrics."""
    result = {}
    if sampling_t is None and training_t is None:
        return result
    if sampling_t is None:
        result[f"{name}_missing_in_sampling"] = 1.0
        return result
    if training_t is None:
        result[f"{name}_missing_in_training"] = 1.0
        return result

    s = sampling_t.float()
    t = training_t.float()

    if s.shape != t.shape:
        result[f"{name}_shape_mismatch"] = 1.0
        result[f"{name}_sampling_shape"] = str(list(s.shape))
        result[f"{name}_training_shape"] = str(list(t.shape))
        return result

    diff = (s - t).abs()
    result[f"{name}_max_abs_diff"] = diff.max().item()
    result[f"{name}_mean_abs_diff"] = diff.mean().item()
    result[f"{name}_max_rel_diff"] = (diff / (t.abs() + 1e-12)).max().item()
    result[f"{name}_exact_match"] = float(torch.equal(s, t))

    return result


def analyze_step(
    sampling_step_dir: str,
    training_step_dir: str,
    verbose: bool = False,
    threshold: float = 1e-4,
) -> Tuple[Dict[str, float], bool]:
    """Analyze a single step for train-inference consistency."""
    sampling_tensors = load_step_tensors(sampling_step_dir)
    training_tensors = load_step_tensors(training_step_dir)

    if not sampling_tensors and not training_tensors:
        return {}, False

    metrics = {}
    has_divergence = False

    # Compare matching tensors
    for name in COMPARISON_TENSORS:
        s_t = sampling_tensors.get(name)
        t_t = training_tensors.get(name)
        step_metrics = compare_tensors(name, s_t, t_t)
        metrics.update(step_metrics)

        max_diff_key = f"{name}_max_abs_diff"
        if max_diff_key in step_metrics and step_metrics[max_diff_key] > threshold:
            has_divergence = True

    # Compare log_prob transport: sampling "log_prob" ↔ training "old_log_prob"
    s_log_prob = sampling_tensors.get("log_prob")
    t_old_log_prob = training_tensors.get("old_log_prob")
    if s_log_prob is not None and t_old_log_prob is not None:
        lp_metrics = compare_tensors("old_log_prob_transport", s_log_prob, t_old_log_prob)
        metrics.update(lp_metrics)
        max_diff_key = "old_log_prob_transport_max_abs_diff"
        if max_diff_key in lp_metrics and lp_metrics[max_diff_key] > 1e-6:
            has_divergence = True

    # Compare sampling log_prob vs training new_log_prob (should match on-policy)
    t_new_log_prob = training_tensors.get("new_log_prob")
    if s_log_prob is not None and t_new_log_prob is not None:
        new_lp_metrics = compare_tensors("log_prob_sampling_vs_training_new", s_log_prob, t_new_log_prob)
        metrics.update(new_lp_metrics)

    # Ratio stats
    t_ratio = training_tensors.get("ratio")
    if t_ratio is not None:
        metrics["ratio_mean"] = t_ratio.mean().item()
        metrics["ratio_std"] = t_ratio.std().item()
        metrics["ratio_max"] = t_ratio.max().item()
        metrics["ratio_min"] = t_ratio.min().item()
        metrics["ratio_max_deviation"] = (t_ratio - 1.0).abs().max().item()

    # Additional tensor comparisons
    for extra_name in ["std_dev_t", "dt", "noise_level"]:
        s_t = sampling_tensors.get(extra_name)
        t_t = training_tensors.get(extra_name)
        if s_t is not None and t_t is not None:
            extra_metrics = compare_tensors(extra_name, s_t, t_t)
            metrics.update(extra_metrics)

    if verbose:
        print(f"    Sampling tensors: {sorted(sampling_tensors.keys())}")
        print(f"    Training tensors: {sorted(training_tensors.keys())}")
        for name in sorted(set(list(sampling_tensors.keys()) + list(training_tensors.keys()))):
            if name in sampling_tensors:
                print(f"      [S] {name}: {tensor_stats(sampling_tensors[name])}")
            if name in training_tensors:
                print(f"      [T] {name}: {tensor_stats(training_tensors[name])}")

    return metrics, has_divergence


def per_sample_analysis(
    sampling_step_dir: str,
    training_step_dir: str,
    top_k: int = 5,
):
    """Detailed per-sample analysis for a single step."""
    s_tensors = load_step_tensors(sampling_step_dir)
    t_tensors = load_step_tensors(training_step_dir)

    s_lp = s_tensors.get("log_prob")
    t_new_lp = t_tensors.get("new_log_prob")
    t_old_lp = t_tensors.get("old_log_prob")
    t_ratio = t_tensors.get("ratio")

    if s_lp is not None and t_new_lp is not None and s_lp.shape == t_new_lp.shape:
        per_sample_diff = (s_lp - t_new_lp).abs()
        print(f"    Per-sample |sampling_log_prob - training_new_log_prob|:")
        for i in range(min(top_k, per_sample_diff.shape[0])):
            worst_idx = per_sample_diff.argmax().item()
            print(
                f"      sample[{worst_idx}]: diff={per_sample_diff[worst_idx].item():.8e} "
                f"sampling_lp={s_lp[worst_idx].item():.6f} "
                f"training_new_lp={t_new_lp[worst_idx].item():.6f}"
            )
            per_sample_diff[worst_idx] = 0

    if t_old_lp is not None and t_new_lp is not None:
        print(f"    Per-sample |old_log_prob - new_log_prob| (should be ~0 for on-policy):")
        on_policy_diff = (t_old_lp - t_new_lp).abs()
        for i in range(min(top_k, on_policy_diff.shape[0])):
            worst_idx = on_policy_diff.argmax().item()
            ratio_val = t_ratio[worst_idx].item() if t_ratio is not None else float("nan")
            print(
                f"      sample[{worst_idx}]: diff={on_policy_diff[worst_idx].item():.8e} "
                f"old_lp={t_old_lp[worst_idx].item():.6f} "
                f"new_lp={t_new_lp[worst_idx].item():.6f} "
                f"ratio={ratio_val:.6f}"
            )
            on_policy_diff[worst_idx] = 0

    # Per-sample noise_pred divergence
    s_noise = s_tensors.get("noise_pred")
    t_noise = t_tensors.get("noise_pred")
    if s_noise is not None and t_noise is not None and s_noise.shape == t_noise.shape:
        per_sample_noise_diff = (s_noise.float() - t_noise.float()).abs()
        spatial_dims = tuple(range(1, per_sample_noise_diff.ndim))
        per_sample_max = per_sample_noise_diff.amax(dim=spatial_dims)
        per_sample_mean = per_sample_noise_diff.mean(dim=spatial_dims)
        print(f"    Per-sample noise_pred max_abs_diff:")
        for i in range(min(top_k, per_sample_max.shape[0])):
            worst_idx = per_sample_max.argmax().item()
            print(
                f"      sample[{worst_idx}]: max={per_sample_max[worst_idx].item():.8e} "
                f"mean={per_sample_mean[worst_idx].item():.8e}"
            )
            per_sample_max[worst_idx] = 0

    # Per-sample next_latents_mean divergence
    s_mean = s_tensors.get("next_latents_mean")
    t_mean = t_tensors.get("next_latents_mean")
    if s_mean is not None and t_mean is not None and s_mean.shape == t_mean.shape:
        per_sample_mean_diff = (s_mean.float() - t_mean.float()).abs()
        spatial_dims = tuple(range(1, per_sample_mean_diff.ndim))
        per_sample_max = per_sample_mean_diff.amax(dim=spatial_dims)
        per_sample_avg = per_sample_mean_diff.mean(dim=spatial_dims)
        print(f"    Per-sample next_latents_mean max_abs_diff:")
        for i in range(min(top_k, per_sample_max.shape[0])):
            worst_idx = per_sample_max.argmax().item()
            print(
                f"      sample[{worst_idx}]: max={per_sample_max[worst_idx].item():.8e} "
                f"mean={per_sample_avg[worst_idx].item():.8e}"
            )
            per_sample_max[worst_idx] = 0


def verify_metadata_correspondence(
    sampling_dir: str,
    training_dir: str,
    rank: int,
    batch_idx: int,
) -> bool:
    """Verify that sampling and training metadata match for a (rank, batch) pair."""
    s_batch_dir = os.path.join(sampling_dir, f"rank_{rank:03d}", f"batch_{batch_idx:03d}")
    t_batch_dir = os.path.join(training_dir, f"rank_{rank:03d}", f"batch_{batch_idx:03d}")

    s_meta = load_metadata(s_batch_dir)
    t_meta = load_metadata(t_batch_dir)

    if s_meta is None or t_meta is None:
        if s_meta is None:
            print(f"    WARNING: No sampling metadata at rank={rank} batch={batch_idx}")
        if t_meta is None:
            print(f"    WARNING: No training metadata at rank={rank} batch={batch_idx}")
        return True  # Can't verify, assume OK

    # Compare prompts
    s_prompts = s_meta.get('prompts', [])
    t_prompts = t_meta.get('prompts', [])
    if s_prompts != t_prompts:
        print(f"    MISMATCH: Prompts differ at rank={rank} batch={batch_idx}!")
        print(f"      Sampling: {s_prompts[:3]}...")
        print(f"      Training: {t_prompts[:3]}...")
        return False

    # Compare batch sizes
    s_bs = s_meta.get('batch_size', -1)
    t_bs = t_meta.get('batch_size', -1)
    if s_bs != t_bs:
        print(f"    MISMATCH: Batch sizes differ at rank={rank} batch={batch_idx}: sampling={s_bs} training={t_bs}")
        return False

    return True


def analyze_rank_batch(
    sampling_dir: str,
    training_dir: str,
    rank: int,
    batch_idx: int,
    verbose: bool = False,
    threshold: float = 1e-4,
    top_k: int = 5,
) -> Tuple[Dict[int, Dict[str, float]], bool]:
    """Analyze all steps for a specific (rank, batch_idx) pair."""
    s_batch_dir = os.path.join(sampling_dir, f"rank_{rank:03d}", f"batch_{batch_idx:03d}")
    t_batch_dir = os.path.join(training_dir, f"rank_{rank:03d}", f"batch_{batch_idx:03d}")

    s_steps = set(discover_steps(s_batch_dir))
    t_steps = set(discover_steps(t_batch_dir))
    common_steps = sorted(s_steps & t_steps)

    if not common_steps:
        return {}, False

    all_metrics = {}
    any_divergence = False

    for step_idx in common_steps:
        s_step_dir = os.path.join(s_batch_dir, f"step_{step_idx:03d}")
        t_step_dir = os.path.join(t_batch_dir, f"step_{step_idx:03d}")

        metrics, has_div = analyze_step(s_step_dir, t_step_dir, verbose=verbose, threshold=threshold)
        all_metrics[step_idx] = metrics
        if has_div:
            any_divergence = True

    return all_metrics, any_divergence


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Flow-Factory GRPO train-inference consistency debug tensors"
    )
    parser.add_argument(
        "debug_dir",
        type=str,
        help="Root debug output directory",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed per-tensor statistics")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Divergence threshold (default: 1e-4)")
    parser.add_argument("--top-k", type=int, default=5, help="Show top-K per-sample divergences (default: 5)")
    parser.add_argument("--rank", type=int, default=None, help="Only analyze this rank (default: all)")
    parser.add_argument("--batch", type=int, default=None, help="Only analyze this batch_idx (default: all)")
    args = parser.parse_args()

    debug_dir = args.debug_dir
    sampling_dir = os.path.join(debug_dir, "sampling")
    training_dir = os.path.join(debug_dir, "training")

    # Load config if available
    config_path = os.path.join(sampling_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print("=== Sampling Config ===")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print()

    # Discover ranks
    s_ranks = set(discover_ranks(sampling_dir))
    t_ranks = set(discover_ranks(training_dir))
    all_ranks = sorted(s_ranks | t_ranks)

    if not all_ranks:
        print(f"ERROR: No rank directories found in {sampling_dir} or {training_dir}")
        print("  Make sure you ran the debug script first.")
        print(f"  Expected: {sampling_dir}/rank_000/batch_000/step_000/...")
        sys.exit(1)

    print(f"=== Rank Discovery ===")
    print(f"  Sampling ranks: {sorted(s_ranks)}")
    print(f"  Training ranks: {sorted(t_ranks)}")
    common_ranks = sorted(s_ranks & t_ranks)
    print(f"  Common ranks:   {common_ranks}")
    print()

    # Filter ranks
    if args.rank is not None:
        if args.rank not in common_ranks:
            print(f"ERROR: Rank {args.rank} not found in common ranks {common_ranks}")
            sys.exit(1)
        common_ranks = [args.rank]

    # Global summary across all (rank, batch) pairs
    global_divergences = []  # [(rank, batch, step, metrics), ...]
    total_pairs_analyzed = 0
    total_diverged_pairs = 0

    for rank in common_ranks:
        s_rank_dir = os.path.join(sampling_dir, f"rank_{rank:03d}")
        t_rank_dir = os.path.join(training_dir, f"rank_{rank:03d}")
        s_batches = set(discover_batches(s_rank_dir))
        t_batches = set(discover_batches(t_rank_dir))
        common_batches = sorted(s_batches & t_batches)

        if args.batch is not None:
            if args.batch not in common_batches:
                print(f"WARNING: Batch {args.batch} not found for rank {rank}, available: {common_batches}")
                continue
            common_batches = [args.batch]

        print(f"=== Rank {rank} (batches: {common_batches}) ===")

        for batch_idx in common_batches:
            total_pairs_analyzed += 1

            # Verify metadata correspondence
            meta_ok = verify_metadata_correspondence(sampling_dir, training_dir, rank, batch_idx)
            if not meta_ok:
                print(f"  [rank={rank} batch={batch_idx}] METADATA MISMATCH — data correspondence broken!")
                total_diverged_pairs += 1
                continue

            # Analyze all steps
            s_batch_dir = os.path.join(s_rank_dir, f"batch_{batch_idx:03d}")
            t_batch_dir = os.path.join(t_rank_dir, f"batch_{batch_idx:03d}")
            s_steps = set(discover_steps(s_batch_dir))
            t_steps = set(discover_steps(t_batch_dir))
            common_steps = sorted(s_steps & t_steps)

            if not common_steps:
                print(f"  [rank={rank} batch={batch_idx}] No common steps")
                continue

            batch_has_divergence = False
            first_div_step = None

            # Print per-step table header
            header = (
                f"  {'step':>6} | {'noise_pred':>14} | {'latents':>14} | {'next_latents':>14} | "
                f"{'next_mean':>14} | {'lp_transport':>14} | {'lp_s_vs_t':>14} | "
                f"{'ratio_dev':>14} | {'status'}"
            )
            print(f"\n  [rank={rank} batch={batch_idx}] Steps: {common_steps}")
            print(header)
            print("  " + "-" * (len(header) - 2))

            for step_idx in common_steps:
                s_step_dir = os.path.join(s_batch_dir, f"step_{step_idx:03d}")
                t_step_dir = os.path.join(t_batch_dir, f"step_{step_idx:03d}")

                metrics, has_div = analyze_step(
                    s_step_dir, t_step_dir,
                    verbose=args.verbose, threshold=args.threshold,
                )

                if has_div:
                    batch_has_divergence = True
                    if first_div_step is None:
                        first_div_step = step_idx
                    global_divergences.append((rank, batch_idx, step_idx, metrics))

                def _fmt(key, precision=6):
                    val = metrics.get(key)
                    if val is None:
                        return "N/A".rjust(14)
                    return f"{val:.{precision}e}".rjust(14)

                status = "DIVERGED" if has_div else "OK"
                print(
                    f"  {step_idx:>6} | "
                    f"{_fmt('noise_pred_max_abs_diff')} | "
                    f"{_fmt('latents_max_abs_diff')} | "
                    f"{_fmt('next_latents_max_abs_diff')} | "
                    f"{_fmt('next_latents_mean_max_abs_diff')} | "
                    f"{_fmt('old_log_prob_transport_max_abs_diff')} | "
                    f"{_fmt('log_prob_sampling_vs_training_new_max_abs_diff')} | "
                    f"{_fmt('ratio_max_deviation')} | "
                    f"{status}"
                )

            if batch_has_divergence:
                total_diverged_pairs += 1
                # Per-sample analysis at first divergence step
                print(f"\n  >>> Per-sample analysis at FIRST divergence step {first_div_step}:")
                per_sample_analysis(
                    os.path.join(s_batch_dir, f"step_{first_div_step:03d}"),
                    os.path.join(t_batch_dir, f"step_{first_div_step:03d}"),
                    top_k=args.top_k,
                )
            print()

    # Global Summary
    print("=" * 80)
    print("=== GLOBAL SUMMARY ===")
    print(f"  Total (rank, batch) pairs analyzed: {total_pairs_analyzed}")
    print(f"  Pairs with divergence:              {total_diverged_pairs}")
    print(f"  Pairs OK:                           {total_pairs_analyzed - total_diverged_pairs}")
    print(f"  Threshold used:                     {args.threshold:.0e}")

    if global_divergences:
        print(f"\n=== Top-{args.top_k} Worst Divergences ===")
        # Sort by worst log_prob diff
        def _sort_key(entry):
            _, _, _, m = entry
            return m.get("log_prob_sampling_vs_training_new_max_abs_diff", 0)

        global_divergences.sort(key=_sort_key, reverse=True)
        for i, (rank, batch_idx, step_idx, metrics) in enumerate(global_divergences[:args.top_k]):
            lp_diff = metrics.get("log_prob_sampling_vs_training_new_max_abs_diff", 0)
            ratio_dev = metrics.get("ratio_max_deviation", 0)
            noise_diff = metrics.get("noise_pred_max_abs_diff", 0)
            print(
                f"  #{i+1} rank={rank} batch={batch_idx} step={step_idx}: "
                f"lp_diff={lp_diff:.8e} ratio_dev={ratio_dev:.8e} noise_diff={noise_diff:.8e}"
            )
    else:
        if total_pairs_analyzed > 0:
            print("\n  ALL (rank, batch, step) tuples are consistent! ✓")
        else:
            print("\n  No data to analyze.")

    # Scheduler diagnostic
    print("\n=== Scheduler Diagnostic ===")
    # Show sigma values from rank 0, batch 0 as reference
    ref_rank = common_ranks[0] if common_ranks else 0
    ref_batch = 0
    s_ref_dir = os.path.join(sampling_dir, f"rank_{ref_rank:03d}", f"batch_{ref_batch:03d}")
    t_ref_dir = os.path.join(training_dir, f"rank_{ref_rank:03d}", f"batch_{ref_batch:03d}")
    ref_steps = sorted(set(discover_steps(s_ref_dir)) & set(discover_steps(t_ref_dir)))
    for step_idx in ref_steps:
        s_t = load_step_tensors(os.path.join(s_ref_dir, f"step_{step_idx:03d}"))
        t_t = load_step_tensors(os.path.join(t_ref_dir, f"step_{step_idx:03d}"))
        parts = [f"  step {step_idx:3d}:"]
        for name, label in [("sigma", "sigma"), ("sigma_prev", "sigma_prev"), ("noise_level", "noise_level")]:
            s_val = s_t.get(name)
            t_val = t_t.get(name)
            if s_val is not None:
                parts.append(f"S_{label}={s_val.item():.6f}")
            if t_val is not None:
                parts.append(f"T_{label}={t_val.item():.6f}")
        print(" ".join(parts))

    print("\nDone.")


if __name__ == "__main__":
    main()
