"""
grid_search.py
--------------
Sweeps ALConfig combinations and aggregates results.

Edit SEARCH_GRID — every combination will be run.
Fixed settings go in FIXED_CONFIG.

Usage:
    python grid_search.py                      # full grid, sequential
    python grid_search.py --dry-run            # print combos only
    python grid_search.py --parallel 4         # 4 parallel workers
    python grid_search.py --only gflownet      # filter by strategy
"""

import argparse
import csv
import itertools
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PHASE1_SRC = Path(__file__).resolve().parent.parent / "Phase1" / "src"
if str(PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE1_SRC))

from reward import reward_peak
from VEM import oracle_fn

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------

SEARCH_GRID = {
    # ── Core experimental variables ──
    "sampling_strategy":      ["random", "lhs", "grid", "gflownet"],
    "acquisition":            ["top_k", "diverse_top_k"],
    "n_candidates_per_iter":  [5, 10, 25],
    "n_init":                 [20, 50, 100],
    "n_iterations":           [10],
    # ── GFN-only (ignored for non-gflownet strategies) ──
    "gfn_n_train_steps":      [100, 500],
    "n_gfn_samples":          [100, 500],
    # ── Reproducibility ──
    "seed":                   [42, 123],
}

FIXED_CONFIG = {
    "output_dir":          "./grid_results_2",
    "proxy_models_path":   "../Phase1/models/al",
    "proxy_model_name":    "XGBoost",
    "ml_model_name":       "XGBoost",
    "ml_retrain_every":    1,
    "diverse_top_k_lambda": 0.3,
    "reward_fn_name":      "reward_peak",
}

# ---------------------------------------------------------------------------
# Prune redundant combos
# ---------------------------------------------------------------------------

def is_valid(combo: dict) -> bool:
    # GFN-specific params ignored for non-gflownet strategies
    if combo["sampling_strategy"] != "gflownet":
        min_steps = min(SEARCH_GRID["gfn_n_train_steps"])
        if combo["gfn_n_train_steps"] != min_steps:
            return False
        if combo["n_gfn_samples"] != min(SEARCH_GRID["n_gfn_samples"]):
            return False
    return True


def generate_combos():
    keys, vals = list(SEARCH_GRID.keys()), list(SEARCH_GRID.values())
    return [dict(zip(keys, v)) for v in itertools.product(*vals) if is_valid(dict(zip(keys, v)))]


def run_name(combo: dict, idx: int) -> str:
    return (f"run{idx:04d}"
            # f"_{combo['sampling_strategy'][:4]}"
            # f"_{combo['acquisition'][:3]}"
            # f"_n{combo['n_candidates_per_iter']}"
            # f"_s{combo['seed']}"
            )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(combo: dict, idx: int, total: int, dry_run: bool = False) -> dict:
    from al_loop import ALConfig, run_al_loop

    name = run_name(combo, idx)
    print(f"\n[{idx+1}/{total}] {name}")
    print(f"  {json.dumps(combo)}")

    if dry_run:
        return {"run": name, "status": "dry_run", **combo}

    cfg = ALConfig(**{**FIXED_CONFIG, **combo})

    t0 = time.time()
    try:
        X_all, y_all = run_al_loop(cfg, reward_fn=reward_peak, oracle_fn=oracle_fn)
        rewards = reward_peak(y_all)
        status = "ok"
        best = float(rewards.max())
        mean = float(rewards.mean())
    except Exception as e:
        traceback.print_exc()
        status = f"error: {e}"
        best = mean = None

    return {
        "run": name, "status": status,
        "elapsed_sec": round(time.time() - t0, 1),
        "best_reward": best, "mean_reward": mean,
        **combo,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def save_summary(results: list, out_dir: str):
    path = Path(out_dir) / "grid_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    results_sorted = sorted(results, key=lambda r: (r.get("best_reward") is None, -(r.get("best_reward") or 0)))
    keys = list(dict.fromkeys(k for r in results_sorted for k in r))
    if path.exists():
        mode = "a"  # append
    else:
        mode = "w"  # write new
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        w.writerows(results_sorted)
    print(f"\n[Summary] {len(results_sorted)} runs → {path}")
    print(f"\n{'run':<35} {'strategy':<10} {'acq':<14} {'best':>8} {'elapsed':>8}")
    print("-" * 80)
    for r in results_sorted[:10]:
        if r.get("best_reward") is not None:
            print(f"{r['run']:<35} {r['sampling_strategy']:<10} {r['acquisition']:<14} "
                  f"{r['best_reward']:>8.4f} {r.get('elapsed_sec','?'):>8}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--only", type=str, default=None, help="filter by sampling_strategy")
    parser.add_argument("--seed", type=int, default=None, help="run only this seed")
    args = parser.parse_args()

    combos = generate_combos()
    if args.only:
        combos = [c for c in combos if c["sampling_strategy"] == args.only]
    if args.seed is not None:
        combos = [c for c in combos if c["seed"] == args.seed]

    total = len(combos)
    print(f"{total} combinations to run")

    if args.dry_run:
        for i, c in enumerate(combos):
            print(f"  [{i+1:3d}] {run_name(c, i)}: {c}")
        return

    results = []
    out_dir = FIXED_CONFIG["output_dir"]

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(run_one, c, i, total): i for i, c in enumerate(combos)}
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"Worker error: {e}")
                save_summary(results, out_dir)
    else:
        for i, c in enumerate(combos):
            # if i != 111:
            #     continue
            # print(f"\n[{i+1}/{total}] {run_name(c, i)}")
            # exit()
            results.append(run_one(c, i, total, dry_run=args.dry_run))
            save_summary(results, out_dir)

    save_summary(results, out_dir)


if __name__ == "__main__":
    main()