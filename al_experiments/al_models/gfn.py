from datetime import datetime
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from copy import deepcopy

import hydra
from ioh import ProblemClass, get_problem
import ioh
import pandas as pd
from omegaconf import open_dict
import torch
from tqdm import tqdm

from gflownet.utils.common import gflownet_from_config

def train_and_sample_gfn(config, visited, proxy):
    gflownet = gflownet_from_config(config)
    gflownet.train()

    samples = sample_from_gfn(
        base_dir=Path(config.logger.logdir.path),
        n_samples=config.pbo_al_experiment.n_samples_per_al_iteration,
        batch_size=config.sampling_batch_size,
        gflownet=gflownet
    )

    samples_x_without_duplicates = [item for item in samples["X"] if item not in visited["X"]]
    samples_y_without_duplicates = [samples["y"][i] for i, x in enumerate(samples["X"]) if x not in visited["X"]]

    missing_count = config.pbo_al_experiment.n_samples_per_al_iteration - len(samples_x_without_duplicates)
    strike = 0
    while missing_count > 0 and strike < 50:
        strike += 1
        print(f"Found {missing_count} duplicate samples, resampling to get enough unique samples (strike {strike})...")

        if strike == 50:
            print("Maximum resampling attempts reached, proceeding with available unique samples.")
            break
        
        additional_samples = sample_from_gfn(
            base_dir=Path(config.logger.logdir.path),
            n_samples=missing_count,
            batch_size=config.sampling_batch_size,
            gflownet=gflownet
        )

        samples = {
            "X": samples["X"] + additional_samples["X"],
            "y": samples["y"] + additional_samples["y"]
        }

        samples_x_without_duplicates = [item for item in samples["X"] if item not in visited["X"]]
        samples_y_without_duplicates = [samples["y"][i] for i, x in enumerate(samples["X"]) if x not in visited["X"]]
        missing_count = config.pbo_al_experiment.n_samples_per_al_iteration - len(samples_x_without_duplicates)

    gflownet.logger.end()

    return samples_x_without_duplicates, samples_y_without_duplicates

def sample_from_gfn(base_dir: Path, n_samples: int, batch_size: int, gflownet):
    
    output_dir = base_dir / "output" 
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, bs in enumerate(
            tqdm(get_batch_sizes(n_samples, batch_size))
        ):
            batch, times = gflownet.sample_batch(
                n_forward=bs, env_cond=None, train=False
            )
            x_sampled = batch.get_terminating_states(proxy=True)
            energies = gflownet.proxy(x_sampled)
            x_sampled = batch.get_terminating_states()
            dct = {"X": x_sampled, "y": energies.tolist()}
            pickle.dump(dct, open(tmp_dir / f"gfn_samples_{i}.pkl", "wb"))

    # Concatenate all samples
    dct = {k: [] for k in dct.keys()}
    for f in tqdm(list(tmp_dir.glob("*.pkl"))):
        tmp_dict = pickle.load(open(f, "rb"))
        dct = {k: v + tmp_dict[k] for k, v in dct.items()}
    df = pd.DataFrame(dct)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    pickle.dump(dct, open(output_dir / f"gfn_samples_{timestamp}.pkl", "wb"))
    df.to_csv(output_dir / f"gfn_samples_{timestamp}.csv")

    shutil.rmtree(tmp_dir)

    return dct

def get_batch_sizes(total, b=1):
    """
    Batches an iterable into chunks of size n and returns their expected lengths.

    Example
    -------

    .. code-block:: python

        >>> get_batch_sizes(10, 3)
        [3, 3, 3, 1]

    Parameters
    ----------
    total : int
        total samples to produce
    b : int
        the batch size

    Returns
    -------
    list
        list of batch sizes
    """
    n = total // b
    chunks = [b] * n
    if total % b != 0:
        chunks += [total % b]
    return chunks