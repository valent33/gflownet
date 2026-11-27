"""
diagnostics.py
--------------
Post-training diagnostics for trained GFlowNet runs (ContinuousPlasma / Plasma).

Computes:
  1. Partition function (logZ) & Trajectory Balance diagnostics:
     - Learned logZ.sum() vs. uniform Monte Carlo estimate of ln(Z)
     - Importance-sampled (IS) logZ, ELBO logZ, Effective Sample Size (ESS)
     - Reverse-KL estimate (mc_true_logZ - elbo_logZ)
     - TB residual mean/MSE and mean trajectory logP_F, logP_B, logR
  2. Sample quality, diversity, and oracle calibration:
     - Uniqueness, categorical normalized entropy, continuous pairwise spread
     - Raw proxy vs. transformed reward statistics
     - True VEM oracle evaluation vs. matched uniform random baseline
     - Proxy-vs-oracle Pearson r, Spearman rho, MAE
     - 5-peak mode coverage (for reward_peak_new)
"""

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve()
GFN_DIR = _HERE.parent
PHASE1_SRC = GFN_DIR.parent / "Phase1" / "src"
for _p in (GFN_DIR, PHASE1_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from VEM import oracle_fn
from gflownet.utils.batch import compute_logprobs_trajectories
from gflownet.utils.common import load_gflownet_from_rundir
from reward import (
    AMPS,
    PEAKS_NEW,
    SIGMA0,
    WIDTHS,
    reward_latent,
    reward_peak,
    reward_peak_new,
)
from space import PARAM_NAMES, PLASMA_ENV, Space

REWARD_FNS = {
    "reward_peak": reward_peak,
    "reward_latent": reward_latent,
    "reward_peak_new": reward_peak_new,
}
PEAK_NAMES = ["GLOBAL", "trap", "local_A", "local_B", "1D_ridge"]


def _uniform_volume(env) -> float:
    """Return the Lebesgue / counting measure volume of the terminating space."""
    if hasattr(env, "categorical_params"):
        return float(math.prod(len(p.get_choices()) for p in env.categorical_params))
    return float(math.prod(len(p.get_choices()) for p in env.parameters))


def _sample_uniform_states(env, n_states: int, seed: int) -> list:
    """Sample uniform terminating states across continuous or discrete plasma envs."""
    if hasattr(env, "get_uniform_terminating_states"):
        try:
            return env.get_uniform_terminating_states(n_states, seed=seed)
        except TypeError:
            return env.get_uniform_terminating_states(n_states)
    sp = Space(PLASMA_ENV)
    df = sp.sample_batch(n_states, strategy="random", seed=seed)
    return [env.readable2state(row) for row in df.values.tolist()]


def compute_logz_diagnostics(
    gfn, n_trajs: int = 256, n_mc: int = 2000, seed: int = 42
) -> dict:
    """Compute logZ, TB balance, IS/ELBO logZ, ESS, and MC ground-truth logZ."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    with torch.no_grad():
        batch, _ = gfn.sample_batch(n_forward=n_trajs, train=True)
        lpf = compute_logprobs_trajectories(
            batch, forward_policy=gfn.forward_policy, backward=False
        )
        lpb = compute_logprobs_trajectories(
            batch, backward_policy=gfn.backward_policy, backward=True
        )
        term_proxy = batch.get_terminating_states(proxy=True, sort_by="trajectory")
        logr = gfn.proxy.rewards(term_proxy, log=True)

        learned_logz = (
            float(gfn.logZ.sum().item()) if getattr(gfn, "logZ", None) is not None else None
        )
        implied_logz = logr + lpb - lpf
        elbo_logz = float(implied_logz.mean().item())
        is_logz = float(
            (torch.logsumexp(implied_logz, 0) - math.log(len(implied_logz))).item()
        )
        weights = torch.softmax(implied_logz, dim=0)
        ess = float((1.0 / (weights**2).sum()).item())
        ess_frac = ess / float(len(implied_logz))

        unif_states = _sample_uniform_states(gfn.env, n_mc, seed=seed)
        unif_logr = gfn.proxy.rewards(gfn.env.states2proxy(unif_states), log=True)
        vol = _uniform_volume(gfn.env)
        mc_true_logz = float(
            math.log(vol)
            + torch.logsumexp(unif_logr, 0).item()
            - math.log(len(unif_logr))
        )

        kl_estimate = mc_true_logz - elbo_logz
        logz_error = (
            (learned_logz - mc_true_logz) if learned_logz is not None else None
        )
        if learned_logz is not None:
            tb_res = gfn.logZ.sum() + lpf - lpb - logr
            tb_res_mean = float(tb_res.mean().item())
            tb_res_mse = float((tb_res**2).mean().item())
        else:
            tb_res_mean = None
            tb_res_mse = None

        traj_lens = [len(t) for t in batch.get_actions_trajectories()]

    return {
        "learned_logZ": learned_logz,
        "mc_true_logZ": mc_true_logz,
        "elbo_logZ": elbo_logz,
        "is_logZ": is_logz,
        "logZ_error": logz_error,
        "kl_estimate": kl_estimate,
        "ess": ess,
        "ess_frac": ess_frac,
        "tb_residual_mean": tb_res_mean,
        "tb_residual_mse": tb_res_mse,
        "mean_lpf": float(lpf.mean().item()),
        "mean_lpb": float(lpb.mean().item()),
        "mean_logR": float(logr.mean().item()),
        "traj_len_min": int(min(traj_lens)),
        "traj_len_median": float(np.median(traj_lens)),
        "traj_len_max": int(max(traj_lens)),
    }


def _categorical_entropy(series: pd.Series, n_categories: int) -> float:
    """Return normalized Shannon entropy in [0, 1] for a categorical column."""
    if n_categories <= 1:
        return 1.0
    probs = series.value_counts(normalize=True).to_numpy(dtype=float)
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(n_categories)


def _peak_coverage(y: np.ndarray, rewards: np.ndarray) -> dict:
    """Assign each point to its dominant Gaussian peak in reward_peak_new."""
    d2 = np.sum(
        ((y[:, None, :] - PEAKS_NEW[None, :, :]) / (SIGMA0 * WIDTHS[None, :, :])) ** 2,
        axis=2,
    )
    vals = AMPS[None, :] * np.exp(-0.5 * d2)
    dom = vals.argmax(axis=1)
    out = {}
    for pi, pname in enumerate(PEAK_NAMES):
        mask = dom == pi
        hi = mask & (rewards > 0.65)
        out[pname] = {"count": int(mask.sum()), "gt_0.65": int(hi.sum())}
    return out


def compute_sample_diagnostics(
    gfn,
    n_samples: int = 1000,
    eval_oracle: bool = True,
    reward_fn_name: str = "reward_peak_new",
    seed: int = 42,
) -> dict:
    """Sample configurations from ``gfn`` and compute diversity/oracle diagnostics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    states = []
    proxy_raw_parts = []
    proxy_rew_parts = []
    with torch.no_grad():
        for start in range(0, n_samples, 256):
            bs = min(256, n_samples - start)
            batch, _ = gfn.sample_batch(n_forward=bs, train=False)
            states.extend(batch.get_terminating_states())
            x_proxy = batch.get_terminating_states(proxy=True)
            proxy_raw_parts.append(gfn.proxy(x_proxy).cpu().numpy())
            proxy_rew_parts.append(gfn.proxy.rewards(x_proxy).cpu().numpy())

    decoded = [gfn.env.state2readable(s) for s in states]
    df_x = pd.DataFrame(decoded, columns=PARAM_NAMES)
    proxy_raw = np.concatenate(proxy_raw_parts)
    proxy_rew = np.concatenate(proxy_rew_parts)

    sp = Space(PLASMA_ENV)
    cat_entropies = {
        p.name: _categorical_entropy(df_x[p.name], len(p.categories))
        for p in sp.params
        if p.type == "categorical"
    }

    encoded = sp.encode_dataframe(df_x)
    cont_indices = [i for i, p in enumerate(sp.params) if p.type == "continuous"]
    sub = encoded[: min(500, len(encoded)), cont_indices]
    if len(sub) > 1:
        diffs = sub[:, None, :] - sub[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        tri_idx = np.triu_indices(len(sub), k=1)
        cont_pairwise_dist = float(dists[tri_idx].mean())
    else:
        cont_pairwise_dist = 0.0

    result = {
        "n_samples": int(len(df_x)),
        "unique_frac": float(len(df_x.drop_duplicates()) / max(1, len(df_x))),
        "cat_entropy": cat_entropies,
        "cont_pairwise_dist": cont_pairwise_dist,
        "proxy_raw_mean": float(proxy_raw.mean()),
        "proxy_raw_max": float(proxy_raw.max()),
        "proxy_rew_mean": float(proxy_rew.mean()),
        "proxy_rew_max": float(proxy_rew.max()),
    }

    if eval_oracle:
        reward_fn = REWARD_FNS.get(reward_fn_name, reward_peak_new)
        np.random.seed(seed)
        y_gfn = oracle_fn(df_x, sp)
        r_gfn = reward_fn(y_gfn)

        df_rand = sp.sample_batch(n_samples, strategy="random", seed=seed)
        np.random.seed(seed)
        y_rand = oracle_fn(df_rand, sp)
        r_rand = reward_fn(y_rand)

        pearson_val = float(pearsonr(proxy_raw, r_gfn).statistic)
        spearman_val = float(spearmanr(proxy_raw, r_gfn).statistic)
        result["oracle_gfn"] = {
            "mean": float(r_gfn.mean()),
            "median": float(np.median(r_gfn)),
            "p90": float(np.percentile(r_gfn, 90)),
            "max": float(r_gfn.max()),
            "gt_0.70": int((r_gfn > 0.70).sum()),
            "gt_0.85": int((r_gfn > 0.85).sum()),
        }
        result["oracle_random"] = {
            "mean": float(r_rand.mean()),
            "median": float(np.median(r_rand)),
            "p90": float(np.percentile(r_rand, 90)),
            "max": float(r_rand.max()),
            "gt_0.70": int((r_rand > 0.70).sum()),
            "gt_0.85": int((r_rand > 0.85).sum()),
        }
        result["proxy_calibration"] = {
            "pearson_r": pearson_val,
            "spearman_rho": spearman_val,
            "mae": float(np.mean(np.abs(proxy_raw - r_gfn))),
        }
        if reward_fn_name == "reward_peak_new":
            result["peaks_gfn"] = _peak_coverage(y_gfn, r_gfn)
            result["peaks_random"] = _peak_coverage(y_rand, r_rand)

    return result


def diagnose_run(
    rundir: Union[str, Path],
    n_trajs: int = 256,
    n_mc: int = 2000,
    n_samples: int = 1000,
    device: str = "cpu",
    eval_oracle: bool = True,
    seed: int = 42,
) -> dict:
    """Load a completed GFlowNet run directory and compute all diagnostics."""
    rundir = Path(rundir)
    gfn, run_config = load_gflownet_from_rundir(
        rundir=str(rundir),
        device=device,
        no_wandb=True,
        load_last_checkpoint=True,
    )
    reward_fn_name = run_config.proxy.get("reward_fn", "reward_peak_new")

    training_info = {}
    metrics_path = rundir / "metrics.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        if len(m):
            training_info = {
                "n_steps": int(m["iteration"].max()),
                "loss_first": float(m["loss"].iloc[0]),
                "loss_last_100": float(m["loss"].tail(100).mean()),
            }

    logz_diag = compute_logz_diagnostics(gfn, n_trajs=n_trajs, n_mc=n_mc, seed=seed)
    sample_diag = compute_sample_diagnostics(
        gfn,
        n_samples=n_samples,
        eval_oracle=eval_oracle,
        reward_fn_name=reward_fn_name,
        seed=seed,
    )
    return {
        "rundir": str(rundir),
        "env": run_config.env.get("id", "unknown"),
        "reward_function": run_config.proxy.get("reward_function", "identity"),
        "reward_function_kwargs": dict(run_config.proxy.get("reward_function_kwargs", {})),
        "reward_fn": reward_fn_name,
        "training": training_info,
        "logz": logz_diag,
        "samples": sample_diag,
    }


def _print_report(reports: list[dict]):
    """Print a concise tabular summary comparing diagnosed runs."""
    print("\n=== Partition Function & TB Balance Diagnostics ===")
    rows = []
    for rep in reports:
        lz = rep["logz"]
        tr = rep["training"]
        rows.append(
            {
                "run": Path(rep["rundir"]).name,
                "reward_fn": rep["reward_function"],
                "steps": tr.get("n_steps"),
                "loss_100": round(tr["loss_last_100"], 3) if "loss_last_100" in tr else None,
                "learned_logZ": round(lz["learned_logZ"], 3) if lz["learned_logZ"] is not None else None,
                "mc_true_logZ": round(lz["mc_true_logZ"], 3),
                "elbo_logZ": round(lz["elbo_logZ"], 3),
                "is_logZ": round(lz["is_logZ"], 3),
                "kl_est": round(lz["kl_estimate"], 3),
                "ess_frac": round(lz["ess_frac"], 3),
                "tb_mse": round(lz["tb_residual_mse"], 3) if lz["tb_residual_mse"] is not None else None,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Sample Quality, Diversity & Oracle Calibration ===")
    srows = []
    for rep in reports:
        sm = rep["samples"]
        row = {
            "run": Path(rep["rundir"]).name,
            "unique": round(sm["unique_frac"], 3),
            "cont_dist": round(sm["cont_pairwise_dist"], 3),
            "proxy_mean": round(sm["proxy_raw_mean"], 4),
            "proxy_max": round(sm["proxy_raw_max"], 4),
        }
        if "oracle_gfn" in sm:
            row.update(
                {
                    "vem_mean": round(sm["oracle_gfn"]["mean"], 4),
                    "vem_p90": round(sm["oracle_gfn"]["p90"], 4),
                    "vem_max": round(sm["oracle_gfn"]["max"], 4),
                    "rand_mean": round(sm["oracle_random"]["mean"], 4),
                    "gt_0.70": sm["oracle_gfn"]["gt_0.70"],
                    "pearson_r": round(sm["proxy_calibration"]["pearson_r"], 3),
                    "spearman": round(sm["proxy_calibration"]["spearman_rho"], 3),
                }
            )
        srows.append(row)
    print(pd.DataFrame(srows).to_string(index=False))

    for rep in reports:
        sm = rep["samples"]
        if "peaks_gfn" in sm:
            peaks = " | ".join(
                f"{k}: {sm['peaks_gfn'][k]['count']} ({sm['peaks_gfn'][k]['gt_0.65']})"
                for k in PEAK_NAMES
            )
            print(f"Peaks [{Path(rep['rundir']).name}]: {peaks}")


def main():
    parser = argparse.ArgumentParser(description="Post-training GFlowNet diagnostics.")
    parser.add_argument(
        "--rundir",
        action="append",
        required=True,
        help="Path to a trained Hydra run directory (can be repeated).",
    )
    parser.add_argument("--n-trajs", type=int, default=256)
    parser.add_argument("--n-mc", type=int, default=2000)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    reports = [
        diagnose_run(
            rundir=rd,
            n_trajs=args.n_trajs,
            n_mc=args.n_mc,
            n_samples=args.n_samples,
            device=args.device,
            eval_oracle=not args.no_oracle,
            seed=args.seed,
        )
        for rd in args.rundir
    ]
    _print_report(reports)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report -> {args.out_json}")


if __name__ == "__main__":
    main()
