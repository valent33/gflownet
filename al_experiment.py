"""
Runnable script with hydra capabilities
"""

import itertools
import os
import pickle
import random
import sys
from copy import deepcopy

import hydra
from hydra.core.hydra_config import HydraConfig
from ioh import ProblemClass, get_problem
import ioh
import pandas as pd
from omegaconf import OmegaConf, open_dict
import numpy as np


import zipfile

from al_experiments.al_models.ucb_sampling_rf import ucb_sampling_rf
from al_experiments.al_models.random_sampling import random_sampling
from al_experiments.al_models.latin_hypercube_sampling import latin_hypercube_sampling, sample
from al_experiments.initial_design_of_experiment import generate_initial_data
from al_experiments.pbo import generate_random_data, integer_list_to_binary_list
from al_experiments.proxy.pbo_model import PBOModelProxy
from al_experiments.train_proxy import train_random_forest, save_model
from al_experiments.al_models.gfn import train_and_sample_gfn
from al_experiments.proxy.pbo_proxy_random_forest import PBOProxyRandomForest

MODELS = {
    "gflownet": {
        "sample": train_and_sample_gfn,
        "model_name": "GFlowNet",
        "model_info": "Generative Flow Network trained as a proxy-based AL model"
    },
    "random": {
        "sample": random_sampling,
        "model_name": "Random Sampling",
        "model_info": "Random sampling of the search space"
    },
    "latin_hypercube": {
        "sample": latin_hypercube_sampling,
        "model_name": "Latin Hypercube Sampling",
        "model_info": "Latin Hypercube Sampling of the search space"
    },
    "ucb_rf": {
        "sample": ucb_sampling_rf,
        "model_name": "UCB Sampling with Random Forest",
        "model_info": "Upper Confidence Bound sampling adapted for Random Forest regression models, with beta=1.0"
    }
}

PROXIES = {
    "pbo_proxy" : PBOProxyRandomForest,
    "pbo_model" : PBOModelProxy
}

@hydra.main(config_path="./config", config_name="al-experiment", version_base="1.1")
def main(config):
    check_config(config)
    set_seeds(config.seed)

    with open_dict(config):
        main_log_dir = HydraConfig.get().runtime.output_dir
        config.logger.logdir.path = (main_log_dir)
    
    print(f"\nWorking directory of this run: {os.getcwd()}")
    print(f"Logging directory of this run: {config.logger.logdir.path}\n")

    al_result_log_paths = []

    # add initial dataset size in config file
    for problem_info in config.pbo_al_experiment.problems:
        problem_name = problem_info.name
        for instance_info in problem_info.instances:
            instance_id = instance_info.id
            for size in instance_info.problem_sizes:
                if isinstance(size, int):
                    size_int = size
                    dim_profile = [2] * size_int  
                else:
                    dim_profile = []
                    size_int = 0
                    for dim in size:
                        dim_profile.append(2**dim)
                        size_int += dim


                problem = get_problem(problem_name, instance_id, size_int, ProblemClass.PBO)

                initial_X = generate_initial_data(dim_profile, n_samples=config.pbo_al_experiment.initial_dataset_size)
                initial_y = [problem(integer_list_to_binary_list(x, dim_profile)) for x in initial_X]#type: ignore

                #initial_X, initial_y = generate_random_data(pbo_problem=problem, dim_profile=dim_profile, n_samples=config.pbo_al_experiment.initial_dataset_size)

                os.makedirs(os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}"), exist_ok=True)

                dct = {"x": initial_X, "y": initial_y}
                pickle.dump(dct, open(os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"initial_data.pkl"), "wb"))
                dct["x"] = [' '.join(str(x)) for x in dct["x"]]
                df = pd.DataFrame(dct)
                df.to_csv(os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"initial_data.csv"))

                proxy = PROXIES[config.proxy.name](
                    main_log_dir=main_log_dir,
                    problem_name=problem_name,
                    instance_id=instance_id,
                    size=size,
                    size_int=size_int,
                    config=config,
                    initial_X=initial_X,
                    initial_y=initial_y,
                    dim_profile=dim_profile
                )

                config.env.dim_profile = dim_profile
            

                for model_name in config.pbo_al_experiment.active_learning_models:
                    if model_name not in MODELS.keys():
                        raise ValueError(f"Model {model_name} not recognized. Available models: {list(MODELS.keys())}")

                    config.logger.logdir.path = os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"{model_name}")


                    print(f"Running AL experiment: problem {problem_name}_{instance_id}, size {size}, model {model_name}")

                    logger = ioh.logger.Analyzer(
                        root=config.logger.logdir.path, # type: ignore
                        folder_name="al_results",
                        algorithm_name=MODELS[model_name]["model_name"],
                        algorithm_info=MODELS[model_name]["model_info"],
                        store_positions=True,
                        triggers=[ioh.logger.trigger.ALWAYS],
                    )

                    al_result_log_paths.append(os.path.join(config.logger.logdir.path, "al_results"))
                    problem.attach_logger(logger)
                    
                    for repeat in range(config.pbo_al_experiment.n_al_repeats):
                        print(f"\n--- AL repeat {repeat+1}/{config.pbo_al_experiment.n_al_repeats} ---")
                        config.logger.logdir.path = os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"{model_name}", f"repeat_{repeat}") 
                        problem.reset()
                        run_al_experiment(model_name, config, problem, proxy, initial_X, initial_y)
                        proxy.reset()
                    problem.reset()

    with zipfile.ZipFile(os.path.join(main_log_dir, "al_experiment_results.zip"), 'w') as zipf:
        for log_path in al_result_log_paths:
            for foldername, subfolders, filenames in os.walk(log_path):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zipf.write(file_path, os.path.relpath(file_path, main_log_dir))
    print(f"\nAL experiment results have been zipped and saved to: {os.path.join(main_log_dir, 'al_experiment_results.zip')}")


                    




def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def check_config(config):
    if config.pbo_al_experiment.n_samples_per_al_iteration <= 0 or config.pbo_al_experiment.n_samples_per_al_iteration > 1e5:
        raise ValueError("n_samples_output must be between 1 and 100,000")



def run_al_experiment(model_name, config, problem, proxy, initial_X, initial_y):
    root = config.logger.logdir.path
    
    visited = {
        "X": deepcopy(initial_X),
        "y": deepcopy(initial_y)
    }

    metrics = {
        "estimated_usefulness": [],
        "true_usefulness": [],
        "diversity": [],
        "novelty": []
    }

    for it in range(config.pbo_al_experiment.active_learning_iterations):
        print(f"\n=== Active Learning iteration {it+1}/{config.pbo_al_experiment.active_learning_iterations} ===")

        config.logger.logdir.path = os.path.join(root, f"iteration_{it+1}")
        os.makedirs(config.logger.logdir.path, exist_ok=True)

        samples_x, samples_y = MODELS[model_name]["sample"](config, visited, proxy)
        true_y = problem([integer_list_to_binary_list(x, config.env.dim_profile) for x in samples_x])

        metrics["estimated_usefulness"].append(sum(samples_y)/len(samples_y))
        metrics["true_usefulness"].append(sum(true_y)/len(true_y))

        all_pairs_within_samples_x = itertools.combinations(samples_x, 2)
        diversity = sum(np.linalg.norm(np.array(x)-np.array(y)) for x, y in all_pairs_within_samples_x if x != y) / (len(samples_x) * (len(samples_x) - 1))
        metrics["diversity"].append(diversity)

        novelty = sum(([min([np.linalg.norm(np.array(sample)-np.array(visited)) for visited in visited["X"]]) for sample in samples_x])) / len(samples_x) 
        metrics["novelty"].append(novelty)

        visited["X"] += samples_x
        visited["y"] += true_y  

        proxy.update(visited_X=visited["X"], visited_y=visited["y"])

    print("Metrics:", metrics)
    pd.DataFrame(metrics).to_csv(os.path.join(root, f"metrics.csv"), index=False)
        

if __name__ == "__main__":
    main()
    sys.exit()
