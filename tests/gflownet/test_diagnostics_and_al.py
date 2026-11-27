import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GFN_ROOT = Path(__file__).resolve().parents[2]
PHASE1_SRC = GFN_ROOT.parent / "Phase1" / "src"
for _p in (GFN_ROOT, PHASE1_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from VEM import TARGET_NAMES, oracle_fn
from al_loop import ALConfig, acquire, run_al_loop
from diagnostics import compute_logz_diagnostics, compute_sample_diagnostics
from gflownet.envs.cplasma import ContinuousPlasma
from gflownet.proxy.cplasma import ContinuousPlasmaRewardProxy
from hydra import compose, initialize
from gflownet.utils.common import gflownet_from_config
from machinelearning import train_single_model
from post_sampling import sample_and_evaluate
from reward import reward_peak_new
from space import PARAM_NAMES, PLASMA_ENV, Space

MODELS_CONT_DIR = GFN_ROOT.parent / "Phase1" / "models_continuous"


@pytest.mark.skipif(
    not (MODELS_CONT_DIR / "XGBoost_5000.pkl").exists(),
    reason="Continuous surrogate pickles not generated in Phase1/models_continuous",
)
def test__diagnostics_logz_and_samples_on_cplasma():
    with initialize(version_base="1.1", config_path="../../config"):
        cfg = compose(
            config_name="train",
            overrides=[
                "env=cplasma",
                "proxy=cplasma",
                "device=cpu",
                "gflownet.optimizer.n_train_steps=10",
                "logger.do.online=False",
            ],
        )
    gfn = gflownet_from_config(cfg)
    gfn.train()

    logz = compute_logz_diagnostics(gfn, n_trajs=32, n_mc=200, seed=42)
    for key in [
        "learned_logZ",
        "mc_true_logZ",
        "elbo_logZ",
        "is_logZ",
        "kl_estimate",
        "ess",
        "ess_frac",
        "tb_residual_mean",
        "tb_residual_mse",
    ]:
        assert key in logz and np.isfinite(logz[key])
    assert 0.0 < logz["ess_frac"] <= 1.0 + 1e-6
    assert logz["kl_estimate"] >= -1.0

    samples = compute_sample_diagnostics(
        gfn, n_samples=40, eval_oracle=True, reward_fn_name="reward_peak_new", seed=42
    )
    assert samples["n_samples"] == 40
    assert 0.0 < samples["unique_frac"] <= 1.0
    assert "oracle_gfn" in samples and "oracle_random" in samples
    assert "peaks_gfn" in samples
    assert set(samples["peaks_gfn"].keys()) == {
        "GLOBAL",
        "trap",
        "local_A",
        "local_B",
        "1D_ridge",
    }


def test__acquire_all_strategies_continuous():
    sp = Space(PLASMA_ENV)
    X_all = sp.sample_batch(24, strategy="random", seed=10)
    np.random.seed(10)
    y_all = oracle_fn(X_all, sp)
    df_y = pd.DataFrame(y_all, columns=TARGET_NAMES)
    result, _, x_pipeline, _ = train_single_model(
        X_all, df_y, model_name="XGBoost", normalize=True, save_model=False
    )
    assert result["model"] is not None

    candidates = sp.sample_batch(20, strategy="random", seed=99)
    rewards = np.linspace(0.1, 0.9, len(candidates))

    for strat in ["top_k", "diverse_top_k", "ucb", "ei", "diverse_ucb", "gfn_sample"]:
        picked = acquire(
            candidates,
            rewards,
            k=5,
            strategy=strat,
            lam=0.5,
            X_all=X_all,
            space=sp,
            x_pipeline=x_pipeline,
            y_all=y_all,
            reward_fn=reward_peak_new,
            kappa=2.0,
            xi=0.01,
            seed=42,
        )
        assert len(picked) == 5
        assert list(picked.columns) == PARAM_NAMES
        assert len(picked.drop_duplicates()) == 5

    # Verify diverse_top_k avoids a near-duplicate in continuous space
    base_pt = X_all.iloc[0].to_dict()
    near_pt = dict(base_pt)
    near_pt["precursor_flow_rate"] = float(
        np.clip(float(near_pt["precursor_flow_rate"]) + 0.001, 0.0, 500.0)
    )
    far_pt = dict(base_pt)
    far_pt["vector_gas_type"] = (
        "argon" if far_pt["vector_gas_type"] == "nitrogen" else "nitrogen"
    )
    far_pt["precursor_flow_rate"] = 500.0 if float(base_pt["precursor_flow_rate"]) < 250 else 0.0
    far_pt["power"] = 1100.0 if float(base_pt["power"]) < 1000 else 900.0
    low_pt = dict(near_pt)
    low_pt["power"] = 950.0
    cands_div = pd.DataFrame([near_pt, far_pt, low_pt])
    picked_div = acquire(
        cands_div,
        np.array([0.81, 0.80, 0.10]),
        k=1,
        strategy="diverse_top_k",
        lam=0.2,
        X_all=X_all.iloc[[0]],
        space=sp,
    )
    assert np.isclose(
        float(picked_div.iloc[0]["power"]),
        float(far_pt["power"]),
    )


def test__al_loop_and_post_sampling_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(GFN_ROOT)
    cfg = ALConfig(
        n_init=20,
        init_method="latin_hypercube",
        sampling_strategy=[{"lhs": 1}, {"gflownet": 2}, {"gp": 1}, {"genetic": 1}],
        n_candidates=16,
        n_candidates_per_iter=4,
        acquisition="diverse_ucb",
        reward_fn_name="reward_peak_new",
        gfn_env="cplasma",
        gfn_proxy="cplasma",
        gfn_reward_function="exponential",
        gfn_reward_function_kwargs={"alpha": 1.0, "beta": 4.0},
        gfn_n_train_steps=15,
        gfn_batch_size=4,
        device="cpu",
        output_dir=str(tmp_path),
        seed=42,
    )
    X_all, y_all = run_al_loop(cfg, reward_fn=reward_peak_new, oracle_fn=oracle_fn)
    run_dir = tmp_path / "run_0"
    assert (run_dir / "al_metrics.csv").exists()
    assert (run_dir / "dataset.csv").exists()
    assert (run_dir / "gflownet_p0" / "ckpts" / "final.ckpt").exists()
    assert len(X_all) == 20 + 5 * 4
    assert y_all.shape == (40, 4)

    metrics = pd.read_csv(run_dir / "al_metrics.csv")
    assert len(metrics) == 6
    assert metrics.loc[metrics["sampling_strategy"] == "gflownet", "gfn_logZ"].notna().all()

    for strat in ["gflownet", "gp", "genetic", "lhs", "grid", "random"]:
        ev_df = sample_and_evaluate(
            run_dir, n=10, oracle_fn=oracle_fn, strategy=strat, device="cpu", seed=42
        )
        assert len(ev_df) == 10
        assert "reward" in ev_df.columns
        assert np.all(np.isfinite(ev_df["reward"].to_numpy()))
