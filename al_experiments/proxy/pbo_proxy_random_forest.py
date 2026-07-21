from al_experiments.proxy.al_xp_proxy import AlXPProxy
from al_experiments.train_proxy import save_model, train_random_forest
import os
import pickle

class PBOProxyRandomForest(AlXPProxy):
    def __init__(self, **kwargs):
        self.config = kwargs["config"]

        self.problem_name = kwargs["problem_name"]
        self.instance_id = kwargs["instance_id"]
        self.size = kwargs["size"]

        initial_proxy = train_random_forest(kwargs["initial_X"], kwargs["initial_y"])

        self.initial_proxy_path = os.path.join(kwargs["main_log_dir"], f"{kwargs['problem_name']}",f"instance_{kwargs['instance_id']}", f"size_{kwargs['size']}", f"initial_proxy.pkl")
        
        save_model(
            model=initial_proxy,
            filepath=self.initial_proxy_path,
            metadata={
                "problem_name": self.problem_name,
                "problem_instance": self.instance_id,
                "problem_size": self.size,
            }
        )
        
        self.config.proxy.model_path = self.initial_proxy_path
        self.config.proxy.dim_profile = kwargs["dim_profile"]

    def update(self, **kwargs):
        updated_proxy = train_random_forest(kwargs["visited_X"], kwargs["visited_y"])

        tmp_proxy_path = os.path.join(self.config.logger.logdir.path, f"tmp_proxy.pkl")
        
        save_model(
            model=updated_proxy,
            filepath=tmp_proxy_path,
            metadata={
                "problem_name": self.problem_name,
                "problem_instance": self.instance_id,
                "problem_size": self.size,
            }
        )

        self.config.proxy.model_path = tmp_proxy_path

    def reset(self, **kwargs):
        self.config.proxy.model_path = self.initial_proxy_path 

    def predict(self, samples_x):
        with open(self.config.proxy.model_path, 'rb') as f:
            model = pickle.load(f)

        return model.predict(samples_x).tolist()