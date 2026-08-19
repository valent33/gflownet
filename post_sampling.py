"""
post_sampling.py
----------------
Post-training sampling from a completed AL run, in the same spirit as sampling
k*1000 trajectories from a trained GFlowNet checkpoint.

Given a run directory (e.g. grid_results_9/run_42) and the strategy that produced
it, `sample_post_training` returns `n` candidate configurations (same parameter
columns as the space) drawn from that strategy's *final* state:

  - gflownet  : load the final checkpoint and sample n forward trajectories.
  - random / lhs / grid : stateless, seed-deterministic draws from the space.
  - gp        : reload (or re-fit) the last GP surrogate on all data, then propose
                the top-n pool via UCB over a large LHS candidate set.
  - genetic   : reload the last proxy + full dataset, evolve a population of size n.

`sample_and_evaluate` additionally runs the oracle + reward and (optionally) saves
a CSV (features + targets + reward), mirroring the notebook cell used for GFlowNet
trajectories.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# --- path setup (mirrors al_loop.py / the notebook) ---
_HERE = Path(__file__).resolve()
GFN_DIR = _HERE.parent
PHASE1_SRC = _HERE.parent.parent / "Phase1" / "src"
for _p in (GFN_DIR, PHASE1_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from space import Space, GFLOWNET_ENV, PARAM_NAMES
from reward import reward_peak, reward_latent, reward_peak_new
from al_loop import ALConfig, gp_propose, genetic_propose, VEM_OUTPUT_COLUMNS, _sanitize_features
from gflownet.utils.common import load_gflownet_from_rundir

REWARD_FNS = {
    "reward_peak": reward_peak,
    "reward_latent": reward_latent,
    "reward_peak_new": reward_peak_new,
}


def _load_config(run_dir: Path) -> ALConfig:
    meta = Path(run_dir) / "al_config.json"
    if not meta.exists():
        return ALConfig()
    return ALConfig(**json.loads(meta.read_text()))


def get_reward_fn(run_dir=None, cfg=None):
    """Reward callable used by a run (reads reward_fn_name from its config)."""
    cfg = cfg or _load_config(run_dir)
    return REWARD_FNS.get(cfg.reward_fn_name, reward_peak_new)


def _load_dataset(run_dir: Path):
    df = pd.read_csv(Path(run_dir) / "dataset.csv")
    X_all = _sanitize_features(df[PARAM_NAMES])
    y_all = df[VEM_OUTPUT_COLUMNS].to_numpy()
    return X_all, y_all


def _proxy_dir(run_dir: Path, cfg: ALConfig) -> Path:
    """Per-run proxy dir first (new runs), fall back to the configured path."""
    per_run = Path(run_dir) / "proxy"
    if (per_run / f"{cfg.proxy_model_name}.pkl").exists():
        return per_run
    return Path(cfg.proxy_models_path).resolve()


def _sample_gflownet(run_dir, n, device="cuda", batch_size=4096):
    gfn, _ = load_gflownet_from_rundir(
        rundir=str(Path(run_dir) / "gflownet"),
        device=device, no_wandb=True, load_last_checkpoint=True,
    )
    env = gfn.env
    states = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            bs = min(batch_size, n - i)
            batch, _ = gfn.sample_batch(n_forward=bs, train=False)
            states.extend(batch.get_terminating_states())
    return _sanitize_features(pd.DataFrame([env.state2readable(s) for s in states]))


def _sample_gp(run_dir, n, cfg, space, seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

    pdir = _proxy_dir(run_dir, cfg)
    x_pipeline = pickle.load(open(pdir / f"x_pipeline_{cfg.proxy_model_name}.pkl", "rb"))

    gp_path = pdir / "gp_last.pkl"
    if gp_path.exists():
        gp = pickle.load(open(gp_path, "rb"))
    else:
        # Re-fit the last surrogate on the full dataset (deterministic given seed).
        X_all, y_all = _load_dataset(run_dir)
        X_feat = x_pipeline.transform(_sanitize_features(X_all))
        y_reward = get_reward_fn(cfg=cfg)(y_all)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-2)
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
        gp.fit(X_feat, y_reward)

    pool_size = n * cfg.gp_pool_multiplier
    pool = _sanitize_features(
        space.to_dataframe(space.sample_batch(pool_size, strategy="latin_hypercube", seed=seed)))
    mu, sigma = gp.predict(x_pipeline.transform(pool), return_std=True)
    ucb = mu + cfg.gp_kappa * sigma
    return pool.iloc[np.argsort(ucb)[::-1][:n]].reset_index(drop=True)


def _sample_genetic(run_dir, n, cfg, space, seed):
    pdir = _proxy_dir(run_dir, cfg)
    model = pickle.load(open(pdir / f"{cfg.proxy_model_name}.pkl", "rb"))
    x_pipeline = pickle.load(open(pdir / f"x_pipeline_{cfg.proxy_model_name}.pkl", "rb"))
    y_scaler = pickle.load(open(pdir / f"y_scaler_{cfg.proxy_model_name}.pkl", "rb"))
    X_all, y_all = _load_dataset(run_dir)
    return genetic_propose(
        cfg, space, X_all, y_all, get_reward_fn(cfg=cfg),
        model, x_pipeline, y_scaler, seed=seed, n_population=n)


def sample_post_training(run_dir, n, strategy=None, seed=None, space=None,
                         cfg=None, device="cuda"):
    """Return a DataFrame of `n` configurations drawn post-training from the
    strategy that produced `run_dir` (defaults to the run's own strategy)."""
    run_dir = Path(run_dir)
    cfg = cfg or _load_config(run_dir)
    strategy = strategy or cfg.sampling_strategy
    space = space or Space(GFLOWNET_ENV)
    # Reproduce the final iteration: the loop proposes with seed + iteration.
    seed = (cfg.seed + cfg.n_iterations) if seed is None else seed

    if strategy == "gflownet":
        return _sample_gflownet(run_dir, n, device=device)
    if strategy in ("random", "lhs", "grid"):
        s = {"random": "random", "lhs": "latin_hypercube", "grid": "grid"}[strategy]
        return _sanitize_features(
            space.to_dataframe(space.sample_batch(n, strategy=s, seed=seed)))
    if strategy == "gp":
        return _sample_gp(run_dir, n, cfg, space, seed)
    if strategy == "genetic":
        return _sample_genetic(run_dir, n, cfg, space, seed)
    raise ValueError(f"Unknown strategy: {strategy}")


def sample_and_evaluate(run_dir, n, oracle_fn, strategy=None, space=None,
                        cfg=None, target_cols=None, out_csv=None,
                        device="cuda", seed=None):
    """Sample n configurations, score them with the oracle + reward, and
    optionally save a CSV (features + targets + reward), mirroring the
    GFlowNet-trajectory cell."""
    run_dir = Path(run_dir)
    cfg = cfg or _load_config(run_dir)
    strategy = strategy or cfg.sampling_strategy
    space = space or Space(GFLOWNET_ENV)
    reward_fn = get_reward_fn(cfg=cfg)
    target_cols = target_cols or VEM_OUTPUT_COLUMNS

    dg = sample_post_training(run_dir, n, strategy=strategy, space=space,
                              cfg=cfg, device=device, seed=seed)
    y = oracle_fn(dg, space)
    dg = pd.concat([dg, pd.DataFrame(y, columns=target_cols)], axis=1)
    dg["reward"] = reward_fn(y)
    if out_csv is not None:
        dg.to_csv(out_csv, index=False)
    return dg
