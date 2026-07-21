import pickle
import torch
from gflownet.proxy.base import Proxy
from torchtyping import TensorType
from ioh import get_problem, ProblemClass
from al_experiments.pbo import integer_list_to_binary_list

class PBOModel(Proxy):
    def __init__(self, problem, instance_id, size_int, dim_profile, **kwargs):
        super().__init__(**kwargs)

        self.problem = get_problem(problem, instance_id, size_int, ProblemClass.PBO)
        self.dim_profile = dim_profile

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        states_np = states.numpy()
        states_np = [[int((y+1)*.5*(self.dim_profile[i]-1)) for i, y in enumerate(x)] for x in states_np]

        predictions = [self.problem(integer_list_to_binary_list(x, self.dim_profile)) for x in states_np]#type: ignore

        return torch.tensor(predictions).float()
    
