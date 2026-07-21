import pickle
import torch
from gflownet.proxy.base import Proxy
from torchtyping import TensorType

class PBOProxy(Proxy):
    def __init__(self, model_path, dim_profile, **kwargs):
        super().__init__(**kwargs)

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        states_np = states.detach().cpu().numpy()

        predictions_np = self.model.predict(states_np)

        return torch.from_numpy(predictions_np).to(device=states.device, dtype=torch.float32)
    