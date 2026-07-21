import numpy as np
import pickle

def latin_hypercube_sampling(config, visited, proxy):
    n_samples = config.pbo_al_experiment.n_samples_per_al_iteration
    dimensions = config.env.dim_profile

    samples_x_without_duplicates = sample(n_samples, dimensions, visited)
    

    return samples_x_without_duplicates, proxy.predict(samples_x_without_duplicates)

def sample(n_samples, dimensions, visited):
    

    samples_x_without_duplicates = []
    
    strike = 0
    while len(samples_x_without_duplicates) < n_samples and strike < 50:
        batch_size = min(n_samples, min(dimensions))
        samples = np.zeros((batch_size, len(dimensions)), dtype=int)
        for i in range(len(dimensions)):
            samples[:, i] = np.random.permutation(dimensions[i])[:batch_size]
        for sample in samples:
            if sample.tolist() not in visited["X"] and sample.tolist() not in samples_x_without_duplicates:
                samples_x_without_duplicates.append(sample.tolist())
            else:
                strike += 1
                print(f"Found duplicate sample, resampling... (strike {strike})")
                if strike == 50:
                    print("Maximum resampling attempts reached, proceeding with available unique samples.")
                    break
    
    return samples_x_without_duplicates[:n_samples]

if __name__ == "__main__":
    print("Testing Latin Hypercube Sampling...")
    print(sample(5, [5,10,5,5,5], {"X": []}))
