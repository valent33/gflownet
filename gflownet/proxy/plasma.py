import sys
from pathlib import Path

from hydra.utils import get_original_cwd

PHASE1_SRC = Path(__file__).resolve().parents[3] / "Phase1" / "src"
if str(PHASE1_SRC) not in sys.path:
    sys.path.insert(0, str(PHASE1_SRC))

from space import GFLOWNET_ENV
from reward import reward_peak, reward_latent
from machinelearning import predict_with_model, preprocess

REWARD_FNS = {
    "reward_peak": reward_peak,
    "reward_latent": reward_latent,
}

from typing import List, Union
import pickle
import numpy as np
import pandas as pd
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

PARAM_NAMES = [p.name for p in GFLOWNET_ENV]


def _load_first_existing(path_candidates):
    for path in path_candidates:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"None of these files exist: {path_candidates}")


class PlasmaRewardProxy(Proxy):

    def __init__(self, models_path: str, n: int = 50, reward_fn: str = "reward_peak", **kwargs):
        # Reward applied to proxy predictions during GFN training. Must match
        # the reward the AL loop optimises (ALConfig.reward_fn_name).
        self.reward_fn = REWARD_FNS.get(reward_fn, reward_peak)
        models_path = Path(models_path)
        if not models_path.is_absolute():
            models_path = Path(get_original_cwd()) / models_path
        models_path = models_path.resolve()

        self.x_pipeline = _load_first_existing([
            models_path / f"x_pipeline_{n}.pkl",
            models_path / "x_pipeline.pkl",
            models_path / "x_pipeline_XGBoost.pkl",
        ])
        self.model = _load_first_existing([
            models_path / f"XGBoost_{n}.pkl",
            models_path / f"{n}.pkl",
            models_path / "XGBoost.pkl",
        ])
        # Windows: MultiOutputRegressor with n_jobs>1 spawns joblib/loky
        # processes that cannot unpickle XGBoost boosters (access violation).
        # Force serial predict; XGBoost still uses OpenMP threads internally.
        if hasattr(self.model, "n_jobs"):
            self.model.set_params(n_jobs=1)
        self.y_scaler = _load_first_existing([
            models_path / f"y_scaler_{n}.pkl",
            models_path / "y_scaler.pkl",
            models_path / "y_scaler_XGBoost.pkl",
        ])
        super().__init__(**kwargs)

    def setup(self, env=None):
        self.env = env

    def __call__(self, states) -> torch.Tensor:
        # states is now a list of lists with mixed types (str + float)
        param_names = [p.name for p in GFLOWNET_ENV]
        df_x = pd.DataFrame(states, columns=param_names)

        X, _ = preprocess(df_x, self.x_pipeline)
        y_pred = self.model.predict(X)
        y_pred = self.y_scaler.inverse_transform(y_pred)
        rewards = self.reward_fn(y_pred)
        
         # ── diagnostic block ──────────────────────────────────────────
        if not hasattr(self, '_call_count'):
            self._call_count = 0
        self._call_count += 1

        # if self._call_count % 50 == 1:  # print every 50 proxy calls
        #     print(f"\n=== Proxy call #{self._call_count} ===")
        #     print(f"  Input df_x sample (first row):\n    {df_x.iloc[0].to_dict()}")
        #     print(f"  y_pred (raw, first 5):          {y_pred[:5]}")
        #     # print(f"  y_pred (rescaled, first 5):     {y_pred_rescaled[:5].flatten()}")
        #     print(f"  rewards (first 5):              {rewards[:5]}")
        #     print(f"  rewards — min: {rewards.min():.4f}  max: {rewards.max():.4f}  mean: {rewards.mean():.4f}")
        #     print(f"  n_zeros: {(rewards == 0).sum()} / {len(rewards)}")
        # # ─────────────────────────────────────────────────────────────


        return tfloat(np.atleast_1d(rewards), device=self.device, float_type=self.float)