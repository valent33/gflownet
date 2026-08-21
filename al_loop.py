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
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

PHASE1_SRC = Path(__file__).resolve().parent.parent / "Phase1" / "src"
if str(PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE1_SRC))

from space import Space
from reward import reward_peak, reward_latent
from machinelearning import train_single_model, predict_with_model, preprocess, build_models

VEM_OUTPUT_COLUMNS = ["rugosity", "conductivity", "homogeneity", "reflectivity"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ALConfig:
    # Budget
    n_init: int = 100
    init_method: str = "latin_hypercube"  # random | latin_hypercube | grid
    n_iterations: int = 5
    n_candidates_per_iter: int = 10

    # Sampling strategy: gflownet | random | lhs | grid | gp | genetic
    # Fixed strategy string, OR a mid-course schedule: dict / ordered list of
    # single-key dicts mapping strategy name -> iteration count, e.g.
    # {"grid": 10, "gflownet": 15, "genetic": 5}. Total iterations is derived
    # from the schedule and overrides n_iterations when a schedule is given.
    sampling_strategy: Union[str, list] = field(
        default_factory=lambda: [{"lhs": 10}, {"gflownet": 15}, {"genetic": 5}]
    )  
    n_candidates: int = 100          # pool drawn before acquisition

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
    reward_fn_name: str = "reward_peak"  # reward_peak | reward_latent

    # GFlowNet — Hydra config groups under config/gflownet|loss|policy.
    # `gfn_gflownet` selects config/gflownet/<name>.yaml, the three must stay in sync.
    gfn_n_train_steps: int = 500
    gfn_batch_size: int = 10   # forward trajectories per training step (gflownet)
    gfn_gflownet: str = "trajectorybalance"
    gfn_loss: str = "trajectorybalance"
    gfn_policy: str = "mlp_trajectorybalance"
    proxy_models_path: str = "../Phase1/models/al"
    proxy_model_name: str = "XGBoost"  # stem used in proxy yaml (proxy.n)

    # Gaussian Process pool-proposal (sampling_strategy="gp")
    gp_pool_multiplier: int = 5   # pool size = n_candidates * gp_pool_multiplier
    gp_kappa: float = 2.0         # UCB exploration weight (mean + kappa * std)

    # Genetic algorithm pool-proposal (sampling_strategy="genetic")
    ga_generations: int = 5
    ga_elite_frac: float = 0.2
    ga_mutation_rate: float = 0.2

    # Output — auto-numbered under output_dir (run_00, run_01, ...)
    output_dir: str = "./al_results"

    # Reproducibility
    seed: int = 33


# ---------------------------------------------------------------------------
# Baseline samplers
# ---------------------------------------------------------------------------

def _next_run_dir(output_dir: str) -> Path:
    """Auto-discover next run folder (run_0, run_1, ...) under output_dir."""
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
    return base / f"run_{next_num}"


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


def _full_dataset_df(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    iteration_all: np.ndarray,
    reward_fn: Callable,
    run_name: str,
) -> pd.DataFrame:
    """
    Combine features, raw VEM (oracle) outputs, reward, and the selection
    iteration into a single frame.
    """
    df = X_all.reset_index(drop=True).copy()
    if y_all.shape[1] == len(VEM_OUTPUT_COLUMNS):
        y_cols = VEM_OUTPUT_COLUMNS
    else:
        y_cols = [f"y_{i}" for i in range(y_all.shape[1])]
    for i, col in enumerate(y_cols):
        df[col] = y_all[:, i]
    df["reward"] = reward_fn(y_all)
    df["selection_iteration"] = iteration_all
    df["run"] = run_name
    return df

def _expand_schedule(sampling_strategy, n_iterations: int) -> list:
    """Flatten sampling_strategy into a per-iteration list of strategy names.

    Accepts the legacy single string (repeated n_iterations times), a dict
    schedule, or a list of single-key dicts — order is preserved either way:
    {"grid": 10, "gflownet": 15, "genetic": 5}
    [{"grid": 10}, {"gflownet": 15}, {"genetic": 5}]
    """
    if isinstance(sampling_strategy, str):
        return [sampling_strategy] * n_iterations

    if isinstance(sampling_strategy, dict):
        phases = list(sampling_strategy.items())
    elif isinstance(sampling_strategy, list):
        phases = []
        for entry in sampling_strategy:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError("Schedule list entries must be single-key dicts")
            phases.extend(entry.items())
    else:
        raise ValueError(f"Unsupported sampling_strategy type: {type(sampling_strategy)}")

    expanded = []
    for name, count in phases:
        expanded.extend([name] * int(count))
    return expanded

# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def acquire(candidates: pd.DataFrame, rewards: np.ndarray, k: int,
            strategy: str, lam: float,
            X_all: pd.DataFrame = None) -> pd.DataFrame:
    if strategy == "top_k":
        idx = np.argsort(rewards)[::-1][:k]
        return candidates.iloc[idx].reset_index(drop=True)

    elif strategy == "diverse_top_k":
        # Shared categorical encoding so candidates and the already-evaluated
        # points (X_all) live in the same coordinate space. Diversity is then
        # measured against BOTH the picks of this batch and everything that was
        # evaluated in previous iterations, so the selection fills holes in the
        # dataset instead of re-proposing the same region every iteration.
        cats_by_col = {
            col: pd.concat(
                [candidates[col].astype(str)]
                + ([X_all[col].astype(str)] if X_all is not None else [])
            ).unique()
            for col in candidates.columns
        }

        def _encode(df: pd.DataFrame) -> np.ndarray:
            cols = []
            for col in candidates.columns:
                codes = pd.Categorical(
                    df[col].astype(str), categories=cats_by_col[col]
                ).codes
                cols.append(np.clip(codes, 0, None))
            return np.column_stack(cols)

        X_enc = _encode(candidates)
        known_enc = _encode(X_all) if X_all is not None else np.empty((0, X_enc.shape[1]))
        # Normalise with a single shared scale so candidates and X_all compare
        # on the same axis.
        colmax = X_enc.max(axis=0)
        if known_enc.size:
            colmax = np.maximum(colmax, known_enc.max(axis=0))
        colmax = colmax + 1e-9
        X_norm = X_enc / colmax
        known = known_enc / colmax if known_enc.size else np.empty((0, X_enc.shape[1]))

        r_norm = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-9)
        selected, remaining = [], list(range(len(candidates)))
        for _ in range(k):
            if not remaining:
                break
            sel_mat = np.vstack([known, X_norm[selected]]) if selected else known
            if sel_mat.shape[0] == 0:
                best = max(remaining, key=lambda i: r_norm[i])
            else:
                best = max(remaining, key=lambda i:
                    lam * r_norm[i] - (1 - lam) * float(np.max(
                        1 - np.abs(X_norm[i] - sel_mat).mean(axis=1))))
            selected.append(best)
            remaining.remove(best)
        return candidates.iloc[selected].reset_index(drop=True)
    else:
        raise ValueError(f"Unknown acquisition: {strategy}")


# ---------------------------------------------------------------------------
# GP pool proposal
# ---------------------------------------------------------------------------
def gp_propose(config: ALConfig, space: Space, X_all: pd.DataFrame, y_all: np.ndarray,
               reward_fn: Callable, x_pipeline, seed: int = None) -> pd.DataFrame:
    """
    Fit a Gaussian Process surrogate on (featurized X -> reward) using all
    data observed so far, then propose a candidate pool via UCB acquisition
    over a large randomly-sampled set of unevaluated points.

    Reuses x_pipeline (already fit on X_all for the main ML proxy this
    iteration) purely for featurization -- keeps GP inputs consistent with
    whatever encoding/scaling the rest of the pipeline uses.

    Note: the oracle is known to be heteroscedastic / quite noisy, hence the
    explicit WhiteKernel term -- a pure Matern kernel would overfit to noise.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

    X_feat = x_pipeline.transform(_sanitize_features(X_all))
    y_reward = reward_fn(y_all)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-2)
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed
    )
    gp.fit(X_feat, y_reward)
    # Persist the fitted surrogate so post-training sampling can reuse it.
    pickle.dump(gp, open(Path(config.proxy_models_path) / "gp_last.pkl", "wb"))

    pool_size = config.n_candidates * config.gp_pool_multiplier
    pool = space.to_dataframe(
        space.sample_batch(pool_size, strategy="latin_hypercube", seed=seed)
    )
    pool = _sanitize_features(pool)

    pool_feat = x_pipeline.transform(pool)
    mu, sigma = gp.predict(pool_feat, return_std=True)
    ucb = mu + config.gp_kappa * sigma  # exploration-aware acquisition score

    top_idx = np.argsort(ucb)[::-1][:config.n_candidates]
    return pool.iloc[top_idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Genetic algorithm pool proposal
# ---------------------------------------------------------------------------
def genetic_propose(config: ALConfig, space: Space, X_all: pd.DataFrame, y_all: np.ndarray,
                     reward_fn: Callable, model, x_pipeline, y_scaler,
                     seed: int = None, n_population: int = None) -> pd.DataFrame:
    """
    Evolve a population of candidates toward higher predicted reward, using
    the current ML proxy as the fitness function. The initial population is
    seeded with the best-performing samples observed so far, plus fresh
    random individuals for diversity.
    """
    rng = np.random.default_rng(seed)
    # n_population lets post-training sampling scale the evolution beyond the
    # loop's n_candidates pool (defaults to the loop behaviour).
    population_size = config.n_candidates if n_population is None else n_population

    rewards_so_far = reward_fn(y_all)
    n_seed = max(1, population_size // 4)
    top_idx = np.argsort(rewards_so_far)[::-1][:n_seed]
    seed_pop = X_all.iloc[top_idx].reset_index(drop=True)

    n_fresh = max(0, population_size - len(seed_pop))
    fresh_pop = space.to_dataframe(space.sample_batch(n_fresh, strategy="random", seed=seed))
    population = pd.concat([seed_pop, fresh_pop], ignore_index=True)
    population = _sanitize_features(population)

    def _fitness(pop_df):
        y_pred = predict_with_model(model, pop_df, x_pipeline, y_scaler)
        return reward_fn(y_pred)

    def _crossover(p1, p2):
        return {p.name: (p1[p.name] if rng.random() < 0.5 else p2[p.name])
                for p in space.params}

    def _mutate(individual):
        individual = dict(individual)
        for p in space.params:
            if rng.random() < config.ga_mutation_rate:
                individual[p.name] = p.sample()
        return individual

    for _ in range(config.ga_generations):
        fitness = _fitness(population)
        n_elite = max(1, int(config.ga_elite_frac * len(population)))
        elite_idx = np.argsort(fitness)[::-1][:n_elite]
        elites = population.iloc[elite_idx].reset_index(drop=True)
        elite_records = elites.to_dict("records")

        children = []
        while len(children) < population_size - n_elite:
            i1, i2 = rng.integers(0, len(elite_records), size=2)
            child = _crossover(elite_records[i1], elite_records[i2])
            child = _mutate(child)
            children.append(child)

        population = pd.concat([elites, pd.DataFrame(children)], ignore_index=True)
        population = _sanitize_features(population)

    return population.reset_index(drop=True)


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
        f"proxy.reward_fn={config.reward_fn_name}",
        # The gflownet/loss/policy config groups must be set together
        f"gflownet={config.gfn_gflownet}",
        f"loss={config.gfn_loss}",
        f"policy={config.gfn_policy}",
        f"gflownet.optimizer.n_train_steps={total_steps}",
        f"gflownet.optimizer.batch_size.forward={config.gfn_batch_size}",
        f"n_samples=0",
        f"hydra.run.dir={_hydra_run_dir(log_dir)}",
        f"seed={config.seed}",
    ])
    return log_dir

def gfn_resume(rundir: Path, total_steps: int) -> Path:
    # config/resume.yaml has no gflownet/loss/policy config groups, so those
    # cannot be overridden here: a resumed run keeps the config from its rundir.
    rundir = rundir.resolve()

    _run("resume.py", [
        f"rundir={rundir}",
        f"n_train_steps={total_steps}",
        "n_samples=0",
    ])
    return rundir


def gfn_sample(config: ALConfig, rundir: Path) -> pd.DataFrame:
    rundir = rundir.resolve()
    _run("eval.py", [
        f"rundir={rundir}",
        f"n_samples={config.n_candidates}",
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

    strategy_schedule = _expand_schedule(config.sampling_strategy, config.n_iterations)
    n_iterations = len(strategy_schedule)

    out = _next_run_dir(config.output_dir)
    run_name = out.name
    out.mkdir(parents=True, exist_ok=True)
    # Save the proxy under the run dir so each run keeps its own latest model
    # (otherwise sequential/concurrent runs overwrite each other's proxy on
    # disk). Both the metadata file and models_dir pick up the per-run path.
    config.proxy_models_path = str(out / "proxy")
    logger = ALLogger(out / "al_metrics.csv")
    logger.save_metadata(config)
    models_dir = Path(config.proxy_models_path).resolve()

    print(f"\n{'='*60}")
    print(f"AL: {run_name}  |  {strategy_schedule}  |  {n_iterations} iters")
    print(f"{'='*60}\n")

    space = Space(_get_env())

    # Initial data
    if initial_X is None:
        print(f"[Init] LHS {config.n_init} samples...")
        initial_X = space.to_dataframe(space.sample_batch(config.n_init, strategy=config.init_method, seed=config.seed))
        initial_X = _sanitize_features(initial_X)
        initial_y = oracle_fn(initial_X, space)
    else:
        initial_X = _sanitize_features(initial_X)

    X_all, y_all = initial_X.copy(), initial_y.copy()
    iteration_all = np.zeros(len(X_all), dtype=int)   # <-- add this: init pool = iteration 0
    model = x_pipeline = y_scaler = None
    gfn_rundir = None
    gfn_phase_idx = -1
    gfn_target_steps = config.gfn_n_train_steps
    effective_resume_steps = max(1, config.gfn_n_train_steps // 5)
    prev_strategy = None
    oracle_evals_total = len(X_all)
    cumulative_best_reward = float(reward_fn(y_all).max())

    # Persist the initial VEM results + reward right away, so a crash before
    # iteration 1 still leaves a usable dataset.csv behind.
    _full_dataset_df(X_all, y_all, iteration_all, reward_fn, run_name).to_csv(out / "dataset.csv", index=False)

    logger.log(0, n_samples=len(X_all),
               oracle_evals_total=oracle_evals_total,
               oracle_evals_this_iter=len(X_all),
               best_reward_so_far=cumulative_best_reward,
               mean_reward_so_far=float(reward_fn(y_all).mean()),
               proxy_r2=None,
               proxy_mae=None,
               sampling_strategy="init")

    for it in range(1, n_iterations + 1):
        t0 = time.time()
        current_strategy = strategy_schedule[it - 1]
        phase_changed = current_strategy != prev_strategy
        prev_strategy = current_strategy
        print(f"\n--- Iteration {it}/{n_iterations} ({current_strategy}) ---")

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
        if current_strategy == "gflownet":
            if phase_changed:
                gfn_phase_idx += 1
                gfn_rundir = out / f"gflownet_p{gfn_phase_idx}"
                gfn_target_steps = config.gfn_n_train_steps
                print(f"[GFN] Fresh phase {gfn_phase_idx} on {len(X_all)} samples (steps={gfn_target_steps})...")
                gfn_train(config, models_dir, gfn_rundir, gfn_target_steps)
            else:
                print("[GFN] Resuming...")
                gfn_target_steps += effective_resume_steps
                gfn_resume(gfn_rundir, gfn_target_steps)
            candidates = gfn_sample(config, gfn_rundir)
        elif current_strategy == "random":
            candidates = space.to_dataframe(space.sample_batch(config.n_candidates, strategy="random", seed=config.seed + it))
        elif current_strategy == "lhs":
            candidates = space.to_dataframe(space.sample_batch(config.n_candidates, strategy="latin_hypercube", seed=config.seed + it))
        elif current_strategy == "grid":
            candidates = space.to_dataframe(space.sample_batch(config.n_candidates, strategy="grid", seed=config.seed + it))
        elif current_strategy == "gp":
            print(f"  [GP] Fitting surrogate on {len(X_all)} samples, proposing pool...")
            candidates = gp_propose(config, space, X_all, y_all, reward_fn, x_pipeline, seed=config.seed + it)
        elif current_strategy == "genetic":
            print(f"  [GA] Evolving population over {config.ga_generations} generations...")
            candidates = genetic_propose(config, space, X_all, y_all, reward_fn, model, x_pipeline, y_scaler, seed=config.seed + it)
        else:
            raise ValueError(f"Unknown sampling_strategy: {current_strategy}")
        candidates = _sanitize_features(candidates)

        # Score the sampled pool with the current proxy, then acquire the top subset
        y_pred = predict_with_model(model, candidates, x_pipeline, y_scaler)
        pred_rewards = reward_fn(y_pred)
        selected = acquire(candidates, pred_rewards, config.n_candidates_per_iter,
                           config.acquisition, config.diverse_top_k_lambda, X_all)

        # Oracle evaluation
        print(f"  [Oracle] Evaluating {len(selected)} candidates...")
        y_new = oracle_fn(selected, space)
        actual_rewards = reward_fn(y_new)
        oracle_evals_total += len(selected)

        # Update dataset
        X_all = _sanitize_features(pd.concat([X_all, selected], ignore_index=True))
        y_all = np.vstack([y_all, y_new])
        iteration_all = np.append(iteration_all, np.full(len(selected), it, dtype=int))

        # Single, always-current dataset file: features + raw VEM outputs + reward.
        _full_dataset_df(X_all, y_all, iteration_all, reward_fn, run_name).to_csv(out / "dataset.csv", index=False)

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
            sampling_strategy=current_strategy,
        )
        print(f"  Best so far: {all_rewards.max():.4f} | This iter: {actual_rewards.max():.4f} | n={len(X_all)}")

    print(f"\nDone → {out}")
    print(f"Saved metrics → {out / 'al_metrics.csv'}")
    print(f"Saved dataset → {out / 'dataset.csv'}")
    return X_all, y_all

if __name__ == "__main__":
    from reward import reward_peak, reward_latent
    from space import Space
    from machinelearning import train_single_model, predict_with_model, preprocess, build_models
    from VEM import oracle_fn

    config = ALConfig()

    run_al_loop(config, reward_fn=reward_peak, oracle_fn=oracle_fn)