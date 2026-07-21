"""
metrics_logger.py
-----------------
Drop-in patch for GFlowNet training metrics. Monkey-patches the tqdm progress bar
update so every iteration's metrics are written to a CSV without touching the repo.

Usage — add two lines to train.py after gflownet_from_config():

    from metrics_logger import MetricsLogger
    logger = MetricsLogger(config.logger.logdir.path)
    logger.attach(gflownet)
    gflownet.train()
    logger.save()
"""

import csv
import os
import time
from pathlib import Path


class MetricsLogger:
    """
    Wraps GFlowNet.train() to intercept per-iteration metrics.

    Captured fields (all that appear on the tqdm bar + extras):
        iteration, loss, mean_reward, max_reward, min_reward, jsd,
        n_sampled, elapsed_sec
    """

    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / filename
        self.rows = []
        self._gfn = None
        self._original_train = None
        self._t0 = None

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------

    def attach(self, gflownet):
        """Monkey-patch gflownet so we intercept each training iteration."""
        self._gfn = gflownet
        self._original_step = gflownet._train_step  # save original

        logger = self  # closure reference

        def patched_train_step(it, batch):
            result = logger._original_step(it, batch)
            logger._record(it, batch, result)
            return result

        gflownet._train_step = patched_train_step
        self._t0 = time.time()
        return self

    def _record(self, it, batch, result):
        """Called after each _train_step. Pulls metrics from batch and result."""
        row = {"iteration": it, "elapsed_sec": round(time.time() - self._t0, 2)}

        # result is typically a dict or a scalar loss
        if isinstance(result, dict):
            row.update({k: _to_scalar(v) for k, v in result.items()})
        else:
            row["loss"] = _to_scalar(result)

        # pull reward stats directly from batch if available
        if hasattr(batch, "rewards") and batch.rewards is not None:
            r = batch.rewards
            row["mean_reward"] = round(float(r.mean()), 6)
            row["max_reward"] = round(float(r.max()), 6)
            row["min_reward"] = round(float(r.min()), 6)
            row["n_nonzero_rewards"] = int((r > 0).sum())

        if hasattr(batch, "logrewards") and batch.logrewards is not None:
            row["mean_logreward"] = round(float(batch.logrewards.mean()), 6)

        self.rows.append(row)

        # flush every 100 iterations so you don't lose data on crash
        if len(self.rows) % 100 == 0:
            self.save()

    # ------------------------------------------------------------------
    # Fallback: if _train_step doesn't exist, patch train() directly
    # ------------------------------------------------------------------

    def attach_to_train(self, gflownet):
        """
        Alternative attachment when _train_step is not accessible.
        Wraps the tqdm pbar update inside train() by subclassing.
        Use this if attach() raises AttributeError.
        """
        self._gfn = gflownet
        self._t0 = time.time()

        original_train = gflownet.train
        logger = self

        def patched_train(*args, **kwargs):
            # patch the internal pbar.set_postfix to intercept metrics
            import tqdm as tqdm_module
            original_set_postfix = tqdm_module.tqdm.set_postfix

            def capturing_set_postfix(pbar_self, ordered_dict=None, refresh=True, **kwargs_inner):
                if ordered_dict:
                    it = getattr(pbar_self, "n", len(logger.rows))
                    row = {"iteration": it, "elapsed_sec": round(time.time() - logger._t0, 2)}
                    for k, v in ordered_dict.items():
                        try:
                            row[k.lower().replace(" ", "_")] = float(v)
                        except (ValueError, TypeError):
                            row[k.lower().replace(" ", "_")] = v
                    logger.rows.append(row)
                    if len(logger.rows) % 100 == 0:
                        logger.save()
                return original_set_postfix(pbar_self, ordered_dict, refresh, **kwargs_inner)

            tqdm_module.tqdm.set_postfix = capturing_set_postfix
            try:
                result = original_train(*args, **kwargs)
            finally:
                tqdm_module.tqdm.set_postfix = original_set_postfix
            return result

        gflownet.train = patched_train
        return self

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self):
        if not self.rows:
            return
        fieldnames = list(self.rows[-1].keys())
        # merge all keys seen across rows
        all_keys = []
        seen = set()
        for row in self.rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[MetricsLogger] Saved {len(self.rows)} rows → {self.path}")


def _to_scalar(v):
    try:
        import torch
        if torch.is_tensor(v):
            return round(float(v.item()), 6)
    except ImportError:
        pass
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return v