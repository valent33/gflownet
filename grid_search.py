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

from reward import reward_peak, reward_latent
from VEM import oracle_fn

REWARD_FNS = {
    "reward_latent": reward_latent,
    "reward_peak": reward_peak,
}

# ---------------------------------------------------------------------------
# Grid search settings
# ---------------------------------------------------------------------------

BUDGET_COMBOS = [
    # (n_init, n_candidates_per_iter, n_iterations)  -> total oracle evals
    (50,  20, 15),   # 350
    (200, 10, 15),   # 350
    (275, 5, 15),   # 350
    # (100, 5,  15),   # 175
    # (25,  10, 15),   # 175
]

SEARCH_GRID = {
    "sampling_strategy": ["random", "lhs", "grid", "gflownet", "gp", "genetic"],
    "acquisition":       ["top_k", "diverse_top_k"],
    "reward_fn_name":    ["reward_peak", "reward_latent"],
    "seed":              [123, 456, 789],
    "init_method":       ["latin_hypercube", "random", "grid"],
    "gfn_loss":          ["detailedbalance", "trajectorybalance", "flowmatching", "forwardlooking", "base"],
}

# gfn_loss -> matching `gflownet` config group (config/gflownet/<name>.yaml)
LOSS_TO_GFN = {
    "detailedbalance": "detailedbalance",
    "trajectorybalance": "trajectorybalance",
    "flowmatching": "flowmatch",
    "forwardlooking": "forwardlooking",
    "base": "base",
    "vargrad": "vargrad",
}

# gfn_loss -> matching `policy` config group (config/policy/<name>.yaml)
LOSS_TO_POLICY = {
    "detailedbalance": "mlp_detailedbalance",
    "trajectorybalance": "mlp_trajectorybalance",
    "flowmatching": "mlp_flowmatch",
    "forwardlooking": "mlp_forwardlooking",
    "base": "multihead_tree",
    "vargrad": "mlp_vargrad",
}

FIXED_CONFIG = {
    "output_dir":          "./grid_results_2",
    "proxy_models_path":   "../Phase1/models/al",
    "proxy_model_name":    "XGBoost",
    "ml_model_name":       "XGBoost",
    "ml_retrain_every":    1,
    "diverse_top_k_lambda": 0.3,
    "gfn_n_train_steps":   1000,
    "n_candidates":       100,
    "gfn_gflownet":        "trajectorybalance",
    "gfn_loss":            "trajectorybalance",
    "gfn_policy":          LOSS_TO_POLICY["trajectorybalance"],
}

# ---------------------------------------------------------------------------
# Prune redundant combos
# ---------------------------------------------------------------------------

def is_valid(combo: dict) -> bool:
    # The gflownet/loss/policy config groups must stay in sync. Combos are
    # generated consistently, but this guards against hand-edited grids.
    return (
        combo["gfn_gflownet"] == LOSS_TO_GFN[combo["gfn_loss"]]
        and combo["gfn_policy"] == LOSS_TO_POLICY[combo["gfn_loss"]]
    )


def generate_combos():
    keys, vals = list(SEARCH_GRID.keys()), list(SEARCH_GRID.values())
    base_combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]

    combos = []
    for base in base_combos:
        for n_init, n_cand, n_iter in BUDGET_COMBOS:
            combo = {
                **base,
                "n_init": n_init,
                "n_candidates_per_iter": n_cand,
                "n_iterations": n_iter,
                "gfn_loss": base["gfn_loss"],
                "gfn_gflownet": LOSS_TO_GFN[base["gfn_loss"]],
                "gfn_policy": LOSS_TO_POLICY[base["gfn_loss"]],
            }
            if is_valid(combo):
                combos.append(combo)
    return combos

def run_name(combo: dict, idx: int) -> str:
    return (f"run_{idx}")

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

    reward_fn = REWARD_FNS[combo["reward_fn_name"]]
    cfg = ALConfig(**{**FIXED_CONFIG, **combo})

    t0 = time.time()
    try:
        X_all, y_all = run_al_loop(cfg, reward_fn=reward_fn, oracle_fn=oracle_fn)
        rewards = reward_fn(y_all)
        status = "ok"
        best = float(rewards.max())
        mean = float(rewards.mean())
    except Exception as e:
        traceback.print_exc()
        status = f"error: {e}"
        best = mean = None

    total_budget = combo["n_init"] + combo["n_candidates_per_iter"] * combo["n_iterations"]

    return {
        "run": name, "status": status,
        "elapsed_sec": round(time.time() - t0, 1),
        "best_reward": best, "mean_reward": mean,
        "total_oracle_budget": total_budget,
        **combo,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def save_summary(result: dict, out_dir: str) -> None:
    """Append one completed run to grid_summary.csv (header written once)."""
    path = Path(out_dir) / "grid_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--only", type=str, default=None, help="filter by sampling_strategy")
    parser.add_argument("--seed", type=int, default=None, help="run only this seed")
    parser.add_argument("--output-dir", type=str, default=FIXED_CONFIG["output_dir"])
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

    FIXED_CONFIG["output_dir"] = args.output_dir
    out_dir = FIXED_CONFIG["output_dir"]

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(run_one, c, i, total): i for i, c in enumerate(combos)}
            completed = 0
            for f in as_completed(futures):
                try:
                    result = f.result()
                except Exception as e:
                    print(f"Worker error: {e}")
                    continue
                save_summary(result, out_dir)
                completed += 1
                print(f"[{completed}/{total}] {result.get('run')} -> {result['status']}")
    else:
        for i, c in enumerate(combos):
            result = run_one(c, i, total)
            save_summary(result, out_dir)


if __name__ == "__main__":
    main()