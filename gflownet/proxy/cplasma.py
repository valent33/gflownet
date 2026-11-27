"""Proxy for the continuous plasma deposition environment."""

import pickle
import sys
from pathlib import Path
from typing import List, Union

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd

PHASE1_SRC = Path(__file__).resolve().parents[3] / "Phase1" / "src"
if str(PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE1_SRC))

from machinelearning import preprocess
from reward import reward_latent, reward_peak, reward_peak_new
from space import PLASMA_ENV

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

REWARD_FNS = {
    "reward_peak": reward_peak,
    "reward_latent": reward_latent,
    "reward_peak_new": reward_peak_new,
}

PARAM_NAMES = [p.name for p in PLASMA_ENV]


def _load_first_existing(path_candidates: List[Path]):
    """Return the first unpickled artifact found among ``path_candidates``."""
    for path in path_candidates:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"None of these files exist: {path_candidates}")


class ContinuousPlasmaRewardProxy(Proxy):
    """Surrogate reward proxy for :class:`gflownet.envs.cplasma.ContinuousPlasma`."""

    def __init__(
        self,
        models_path: str,
        n: int = 5000,
        reward_fn: str = "reward_peak_new",
        **kwargs,
    ):
        if reward_fn not in REWARD_FNS:
            raise ValueError(
                f"Unknown reward_fn '{reward_fn}'. Available: {list(REWARD_FNS)}"
            )
        self.reward_fn = REWARD_FNS[reward_fn]
        self.param_names = PARAM_NAMES

        models_path = Path(models_path)
        if not models_path.is_absolute():
            base_dir = (
                Path(get_original_cwd())
                if hydra.core.hydra_config.HydraConfig.initialized()
                else Path.cwd()
            )
            candidate = base_dir / models_path
            if not candidate.exists():
                repo_candidate = Path(__file__).resolve().parents[2] / models_path
                if repo_candidate.exists():
                    candidate = repo_candidate
            models_path = candidate
        models_path = models_path.resolve()

        self.x_pipeline = _load_first_existing(
            [
                models_path / f"x_pipeline_{n}.pkl",
                models_path / "x_pipeline.pkl",
                models_path / "x_pipeline_XGBoost.pkl",
            ]
        )
        self.model = _load_first_existing(
            [
                models_path / f"XGBoost_{n}.pkl",
                models_path / f"{n}.pkl",
                models_path / "XGBoost.pkl",
            ]
        )
        if hasattr(self.model, "n_jobs"):
            self.model.set_params(n_jobs=1)
        self.y_scaler = _load_first_existing(
            [
                models_path / f"y_scaler_{n}.pkl",
                models_path / "y_scaler.pkl",
                models_path / "y_scaler_XGBoost.pkl",
            ]
        )
        super().__init__(**kwargs)

    def setup(self, env=None):
        """Attach the environment reference if needed by the proxy caller."""
        self.env = env

    def __call__(self, states: Union[List, np.ndarray]) -> torch.Tensor:
        """Predict coating properties for decoded states and score them."""
        if len(states) == 0:
            return tfloat(np.empty(0), device=self.device, float_type=self.float)
        df_x = pd.DataFrame(states, columns=self.param_names)
        X, _ = preprocess(df_x, self.x_pipeline)
        y_pred = self.model.predict(X)
        y_pred = self.y_scaler.inverse_transform(y_pred)
        rewards = self.reward_fn(y_pred)
        return tfloat(np.atleast_1d(rewards), device=self.device, float_type=self.float)
