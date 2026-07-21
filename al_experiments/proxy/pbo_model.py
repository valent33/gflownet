from al_experiments.pbo import integer_list_to_binary_list
from al_experiments.proxy.al_xp_proxy import AlXPProxy
from ioh import get_problem, ProblemClass

class PBOModelProxy(AlXPProxy):
    def __init__(self, **kwargs):
        self.config = kwargs["config"]

        self.problem_name = kwargs["problem_name"]
        self.instance_id = kwargs["instance_id"]
        self.size_int = kwargs["size_int"]
        
        self.dim_profile = kwargs["dim_profile"]

        self.config.proxy.dim_profile = self.dim_profile
        self.config.proxy.problem = self.problem_name
        self.config.proxy.instance_id = self.instance_id
        self.config.proxy.size_int = self.size_int

        self.problem = get_problem(self.problem_name, self.instance_id, self.size_int, ProblemClass.PBO)

    def predict(self, samples_x):
        return [self.problem(integer_list_to_binary_list(x, self.dim_profile)) for x in samples_x]#type: ignore
