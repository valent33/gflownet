import numpy as np
import pickle
from ioh import ProblemClass, get_problem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def rf_predict_mean_std(rf, X):
    """
    Compute mean and std of predictions across trees.

    Parameters
    ----------
    rf : trained RandomForestRegressor
    X  : array of shape (n_candidates, n_features)

    Returns
    -------
    mean : (n_candidates,)
    std  : (n_candidates,)
    """
    tree_preds = np.array([
        tree.predict(X) for tree in rf.estimators_
    ])  # shape (n_trees, n_candidates)

    mean = tree_preds.mean(axis=0)
    std = tree_preds.std(axis=0)

    return mean, std

def ucb_select_rf(rf, X_candidates, beta=1.0):
    """
    Select the next point using UCB with a random forest.

    Parameters
    ----------
    rf : trained RandomForestRegressor
    X_candidates : (n_candidates, n_features)
    beta : float, exploration parameter

    Returns
    -------
    idx : index of selected candidate
    ucb_values : UCB score for each candidate
    """
    mean, std = rf_predict_mean_std(rf, X_candidates)
    ucb = mean + beta * std
    idx = np.argmax(ucb)
    return idx, ucb

def sample(n_samples, rf, dim_profile, visited, beta=1.0, n_candidates=100):
    samples_x_without_duplicates = []
    rng = np.random.default_rng()
    
    strike = 0
    while len(samples_x_without_duplicates) < n_samples and strike < 50:
        x_candidates = rng.integers(low=0, high=dim_profile, size=(n_candidates, len(dim_profile)))
        for _ in range(n_samples):
            idx, ucb = ucb_select_rf(rf, x_candidates, beta)

            #print(f"Selected candidate {idx} with UCB {ucb[idx]:.3f}")
            #print(f"Candidate features: {x_candidates[idx]}")

            if x_candidates[idx].tolist() not in visited["X"] and x_candidates[idx].tolist() not in samples_x_without_duplicates:
                samples_x_without_duplicates.append(x_candidates[idx].tolist())
            else:
                print(f"Found duplicate sample, resampling... (strike {strike+1})")
                strike += 1
                if strike == 50:
                    print("Maximum resampling attempts reached, proceeding with available unique samples.")
                    break
            
            x_candidates = np.delete(x_candidates, idx, axis=0)
    
    return samples_x_without_duplicates[:n_samples]

def ucb_sampling_rf(
    config,
    visited,
    proxy
):
    if config.proxy.name != "pbo_proxy":
        raise ValueError("UCB Sampling with Random Forest is only compatible with the PBO Proxy (with Random Forest Model).")

    with open(config.proxy.model_path, 'rb') as f:
        rf = pickle.load(f)

    samples_x_without_duplicates = sample(
        n_samples=config.pbo_al_experiment.n_samples_per_al_iteration,
        rf=rf,
        dim_profile=config.env.dim_profile,
        visited=visited
    )

    return samples_x_without_duplicates, proxy.predict(samples_x_without_duplicates)


if __name__ == "__main__":
    import sys

    # setting path
    sys.path.append('../parentdirectory')
    print("Testing UCB Sampling with Random Forest...")

    problem = get_problem("OneMax", 1, 10, ProblemClass.PBO)

    X = np.random.randint(0, 2, size=(1000, problem.meta_data.n_variables))

    y = [problem(x) for x in X]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("Random Forest Regressor model trained successfully.")
    print(f"\tTraining R-squared: {rf_model.score(X_train, y_train):.4f}")
    print(f"\tTest R-squared: {rf_model.score(X_test, y_test):.4f}")

    samples = sample(
        n_samples=5,
        rf=rf_model,
        dim_profile=[2]*10,
        visited={"X": X.tolist()}
    )

    print(samples)