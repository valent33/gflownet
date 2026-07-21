"""
al_loop.py
----------
Active learning loop for plasma parameter optimisation.

Wires together:
  - machinelearning.py  (run_benchmark / predict_with_model / preprocess)
  - train.py            (first GFlowNet training)
  - resume.py           (subsequent iterations — much faster via checkpoint)
  - eval.py             (sampling from trained model)
  - reward.py           (pluggable reward function)

One iteration:
  1. Train (or resume) GFlowNet with current ML model as proxy
  2. Sample M candidates via eval.py
  3. Evaluate candidates with oracle
  4. Retrain ML model on expanded dataset
  5. Repeat

Parameters
----------
All knobs live in ALConfig below.
"""

import os
import json
import sys
import pickle
import subprocess
import csv
import time
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

PHASE1_SRC = Path(__file__).resolve().parent.parent / "Phase1" / "src"
if str(PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE1_SRC))

from space import Space
from reward import reward_peak, reward_latent
from machinelearning import train_single_model, predict_with_model, preprocess, build_models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ALConfig:
    # Budget
    n_init: int = 70
    n_iterations: int = 5
    n_candidates_per_iter: int = 10

    # Sampling strategy: gflownet | random | lhs | grid
    # The `phase` column in al_metrics.csv records this value per iteration.
    sampling_strategy: str = "gflownet"
    n_gfn_samples: int = 100          # pool drawn before acquisition

    # Acquisition: top_k | diverse_top_k
    acquisition: str = "top_k"
    diverse_top_k_lambda: float = 0.5  # 1.0=exploit, 0.0=diversity

    # ML model — any key from build_models()
    ml_model_name: str = "XGBoost"
    # Optional dict of hyperparameters forwarded to model.set_params().
    # Example: {"n_estimators": 500, "learning_rate": 0.1}
    ml_model_kwargs: Optional[dict] = None
    ml_retrain_every: int = 1

    # Reward function name (for logging only — pass the fn to run_al_loop)
    reward_fn_name: str = "reward_peak"

    # GFlowNet
    gfn_n_train_steps: int = 50
    gfn_resume_steps: int = 10
    proxy_models_path: str = "../Phase1/models/al"
    proxy_model_name: str = "XGBoost"  # stem used in proxy yaml (proxy.n)

    # Output — auto-numbered under output_dir (run_00, run_01, ...)
    output_dir: str = "./al_results"

    # Reproducibility
    seed: int = 33


# ---------------------------------------------------------------------------
# Baseline samplers
# ---------------------------------------------------------------------------

def _next_run_dir(output_dir: str) -> Path:
    """Auto-discover next run folder (run_00, run_01, ...) under output_dir."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(nums) + 1 if nums else 0
    return base / f"run_{next_num:02d}"


def _get_env():
    from space import GFLOWNET_ENV
    return GFLOWNET_ENV


def _to_scalar(value):
    """Convert tuple/array wrappers from samplers into hashable scalar values."""
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return tuple(value.tolist())
    return value


def _sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map(_to_scalar))

# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def acquire(candidates: pd.DataFrame, rewards: np.ndarray, k: int,
            strategy: str, lam: float) -> pd.DataFrame:
    if strategy == "top_k":
        idx = np.argsort(rewards)[::-1][:k]
        return candidates.iloc[idx].reset_index(drop=True)

    elif strategy == "diverse_top_k":
        X_enc = np.column_stack([
            pd.factorize(candidates[col].astype(str))[0]
            for col in candidates.columns
        ])
        X_norm = X_enc / (X_enc.max(axis=0) + 1e-9)
        r_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-9)
        selected, remaining = [], list(range(len(candidates)))
        for _ in range(k):
            if not remaining:
                break
            if not selected:
                best = max(remaining, key=lambda i: r_norm[i])
            else:
                sel_mat = X_norm[selected]
                best = max(remaining, key=lambda i:
                    lam * r_norm[i] - (1-lam) * float(np.max(
                        1 - np.abs(X_norm[i] - sel_mat).mean(axis=1))))
            selected.append(best)
            remaining.remove(best)
        return candidates.iloc[selected].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown acquisition: {strategy}")


# ---------------------------------------------------------------------------
# GFlowNet helpers
# ---------------------------------------------------------------------------

def _save_proxy_models(model, x_pipeline, y_scaler, models_dir: Path, name: str):
    models_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(x_pipeline, open(models_dir / f"x_pipeline_{name}.pkl", "wb"))
    pickle.dump(model,      open(models_dir / f"{name}.pkl", "wb"))
    pickle.dump(y_scaler,   open(models_dir / f"y_scaler_{name}.pkl", "wb"))


def _run(script: str, overrides: list):
    cmd = [sys.executable, script] + overrides
    print(f"    $ {' '.join(str(x) for x in cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed")


def _hydra_run_dir(path: Path) -> str:
    return path.resolve().as_posix()


def gfn_train(config: ALConfig, models_dir: Path, log_dir: Path, total_steps: int) -> Path:
    log_dir = log_dir.resolve()
    _run("train.py", [
        "env=plasma", "proxy=plasma",
        f"proxy.models_path={models_dir}",
        f"proxy.n={config.proxy_model_name}",
        f"gflownet.optimizer.n_train_steps={total_steps}",
        f"n_samples=0",
        f"hydra.run.dir={_hydra_run_dir(log_dir)}",
        f"seed={config.seed}",
    ])
    return log_dir


def gfn_resume(config: ALConfig, rundir: Path, models_dir: Path, log_dir: Path, total_steps: int) -> Path:
    rundir = rundir.resolve()
    log_dir = log_dir.resolve()
    _run("resume.py", [
        f"rundir={rundir}",
        f"n_train_steps={total_steps}",
        f"n_samples=0",
        f"hydra.run.dir={_hydra_run_dir(log_dir)}",
    ])
    # Return the original checkpoint directory so downstream tools (eval.py)
    # can find the full Hydra config.yaml with logger, gflownet, proxy, etc.
    return rundir


def gfn_sample(config: ALConfig, rundir: Path) -> pd.DataFrame:
    rundir = rundir.resolve()
    _run("eval.py", [
        f"rundir={rundir}",
        f"n_samples={config.n_gfn_samples}",
        f"sampling_batch_size=256",
        f"samples_only=True",
        f"output_dir={rundir}",
    ])

    pkl = rundir / "eval" / "samples" / "gfn_samples.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"No samples at {pkl}")

    data = pickle.load(open(pkl, "rb"))
    env = _get_env()
    rows = []
    for state in data["x"]:
        row = {}
        offset = 0
        for param_idx, p in enumerate(env):
            choices = p.get_choices()
            local_idx = state[param_idx] - offset
            choice = choices[local_idx]
            row[p.name] = choice if not isinstance(choice, tuple) else choice[0]
            offset += len(choices)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class ALLogger:
    def __init__(self, path: Path):
        self.path = path
        self.rows = []
        self.meta_path = path.with_name("al_config.json")
        path.parent.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, config: ALConfig):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2, sort_keys=True)

    def log(self, it: int, **kwargs):
        row = {"iteration": it, "time": time.strftime("%H:%M:%S"), **kwargs}
        self.rows.append(row)
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_al_loop(
    config: ALConfig,
    reward_fn: Callable,     # from reward.py, maps y (n, n_out) -> (n,)
    oracle_fn: Callable,     # maps X: DataFrame -> y: np.ndarray (n, n_out)
    initial_X: pd.DataFrame = None,
    initial_y: np.ndarray = None,
):
    np.random.seed(config.seed)
    random.seed(config.seed)

    out = _next_run_dir(config.output_dir)
    run_name = out.name
    out.mkdir(parents=True, exist_ok=True)
    logger = ALLogger(out / "al_metrics.csv")
    logger.save_metadata(config)
    models_dir = Path(config.proxy_models_path).resolve()

    print(f"\n{'='*60}")
    print(f"AL: {run_name}  |  {config.sampling_strategy}  |  {config.n_iterations} iters")
    print(f"{'='*60}\n")

    # Initial data
    if initial_X is None:
        print(f"[Init] LHS {config.n_init} samples...")
        space = Space(_get_env())
        initial_X = space.to_dataframe(space.sample_batch(config.n_init, strategy="latin_hypercube", seed=config.seed))
        initial_X = _sanitize_features(initial_X)
        initial_y = oracle_fn(initial_X)
    else:
        initial_X = _sanitize_features(initial_X)

    X_all, y_all = initial_X.copy(), initial_y.copy()
    model = x_pipeline = y_scaler = None
    gfn_rundir = None
    gfn_target_steps = config.gfn_n_train_steps
    # Resume steps proportional to initial training budget.
    effective_resume_steps = max(1, config.gfn_n_train_steps // 5)
    oracle_evals_total = len(X_all)
    cumulative_best_reward = float(reward_fn(y_all).max())

    logger.log(0, n_samples=len(X_all),
               oracle_evals_total=oracle_evals_total,
               oracle_evals_this_iter=len(X_all),
               best_reward_so_far=cumulative_best_reward,
               mean_reward_so_far=float(reward_fn(y_all).mean()),
               proxy_r2=None,
               proxy_mae=None,
               phase="init")

    for it in range(1, config.n_iterations + 1):
        t0 = time.time()
        print(f"\n--- Iteration {it}/{config.n_iterations} ---")

        # Retrain ML
        if model is None or (it - 1) % config.ml_retrain_every == 0:
            print(f"  [ML] {config.ml_model_name} on {len(X_all)} samples")
            df_y = pd.DataFrame(y_all)
            X_for_ml = _sanitize_features(X_all)
            result, _, x_pipeline, y_scaler = train_single_model(
                X_for_ml,
                df_y,
                model_name=config.ml_model_name,
                model_kwargs=config.ml_model_kwargs,
                normalize=True,
                save_model=False,
            )
            model = result["model"]
            proxy_r2 = float(np.mean(result["r2"]))
            proxy_mae = float(np.mean(result["mae"]))
            _save_proxy_models(model, x_pipeline, y_scaler, models_dir, config.proxy_model_name)
        else:
            proxy_r2 = proxy_mae = None

        # Sample candidates
        if config.sampling_strategy == "gflownet":
            iter_dir = out / f"gfn_iter_{it}"
            if gfn_rundir is None:
                print(f"  [GFN] Training from scratch...")
                gfn_rundir = gfn_train(config, models_dir, iter_dir, gfn_target_steps)
            else:
                print(f"  [GFN] Resuming from checkpoint...")
                gfn_target_steps += effective_resume_steps
                gfn_rundir = gfn_resume(config, gfn_rundir, models_dir, iter_dir, gfn_target_steps)
            candidates = gfn_sample(config, gfn_rundir)
        else:
            space = Space(_get_env())
            if config.sampling_strategy == "random":
                candidates = space.to_dataframe(space.sample_batch(config.n_gfn_samples, strategy="random", seed=config.seed + it))
            elif config.sampling_strategy == "lhs":
                candidates = space.to_dataframe(space.sample_batch(config.n_gfn_samples, strategy="latin_hypercube", seed=config.seed + it))
            elif config.sampling_strategy == "grid":
                candidates = space.to_dataframe(space.sample_batch(config.n_gfn_samples, strategy="grid", seed=config.seed + it))
        candidates = _sanitize_features(candidates)
        
        # Score the sampled pool with the current proxy, then acquire the top subset
        y_pred = predict_with_model(model, candidates, x_pipeline, y_scaler)
        pred_rewards = reward_fn(y_pred)
        selected = acquire(candidates, pred_rewards, config.n_candidates_per_iter,
                           config.acquisition, config.diverse_top_k_lambda)
        selected.to_csv(out / f"candidates_iter_{it}.csv", index=False)

        # Oracle evaluation
        print(f"  [Oracle] Evaluating {len(selected)} candidates...")
        y_new = oracle_fn(selected)
        actual_rewards = reward_fn(y_new)
        oracle_evals_total += len(selected)

        # Update dataset
        X_all = _sanitize_features(pd.concat([X_all, selected], ignore_index=True))
        y_all = np.vstack([y_all, y_new])
        X_all.to_csv(out / f"dataset_iter_{it}.csv", index=False)

        all_rewards = reward_fn(y_all)
        cumulative_best_reward = max(cumulative_best_reward, float(all_rewards.max()))
        logger.log(it,
            n_samples=len(X_all),
            oracle_evals_total=oracle_evals_total,
            oracle_evals_this_iter=len(selected),
            best_reward_so_far=cumulative_best_reward,
            mean_reward_so_far=float(all_rewards.mean()),
            best_reward_this_iter=float(actual_rewards.max()),
            mean_reward_this_iter=float(actual_rewards.mean()),
            best_predicted_reward=float(pred_rewards.max()),
            proxy_r2=proxy_r2,
            proxy_mae=proxy_mae,
            elapsed_sec=round(time.time() - t0, 1),
            phase=config.sampling_strategy,
        )
        print(f"  Best so far: {all_rewards.max():.4f} | This iter: {actual_rewards.max():.4f} | n={len(X_all)}")

    print(f"\nDone → {out}")
    print(f"Saved metrics → {out / 'al_metrics.csv'}")
    return X_all, y_all

if __name__ == "__main__":
    from reward import reward_peak, reward_latent
    from space import Space
    from machinelearning import train_single_model, predict_with_model, preprocess, build_models
    from VEM import oracle_fn

    config = ALConfig()

    run_al_loop(config, reward_fn=reward_peak, oracle_fn=oracle_fn)