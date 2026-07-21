import random
import pickle
import numpy as np

def random_sampling(config, visited, proxy):
    samples_x_without_duplicates = []
    
    strike = 0
    while len(samples_x_without_duplicates) < config.pbo_al_experiment.n_samples_per_al_iteration and strike < 50:
        
        sampled = sample_once(config.env.dim_profile)
        if sampled not in visited["X"] and sampled not in samples_x_without_duplicates:
            samples_x_without_duplicates.append(sampled)
        else:
            strike += 1
            print(f"Found duplicate sample, resampling... (strike {strike})")
            if strike == 50:
                print("Maximum resampling attempts reached, proceeding with available unique samples.")
                break

    return samples_x_without_duplicates, proxy.predict(samples_x_without_duplicates)



def sample_once(dim_profile):
    rng = np.random.default_rng()

    return rng.integers(low=0, high=dim_profile, size=(1, len(dim_profile)))[0].tolist()

if __name__ == "__main__":
    print("Testing random sampling...")

    for _ in range(10):
        print(sample_once([5,10,5,5,5]))