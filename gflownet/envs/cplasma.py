"""
Continuous Plasma environment: starting from an emtpy sequence, parameters are added one at a time until the maximum length is reached.
"""

import sys

from fastapi import params
sys.path.append("../Phase1/src")
from space import PLASMA_ENV

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tfloat, tlong


class ContinuousPlasma(GFlowNetEnv):
    """
    Continuous Plasma environment: sequences are constructed starting from an empty sequence and
    adding one categorical parameter at a time and then sampling continuous variables at once with increments.

    States are represented by a list of values corresponding to each physical parameter tied to its index.

    Actions are mixed.
    """

    def __init__(
        self,
        parameters: Iterable = None,
        length_traj: int = 1,
        min_incr: float = 0.1,
        n_comp: int = 1,
        beta_params_min: float = 0.1,
        beta_params_max: float = 100.0,
        epsilon: float = 1e-6,
        kappa: float = 1e-3,
        **kwargs,
    ):
        # Main attributes
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = PLASMA_ENV
        self.pad_idx = -1
        self.n_dim = len(self.parameters)
        self.categorical_params = [p for p in self.parameters if p.type == "categorical"]
        self.continuous_params = [p for p in self.parameters if p.type == "continuous"]
        self.n_discrete_actions = sum(len(param.get_choices()) for param in self.categorical_params)
        self.n_continuous_dims = len(self.continuous_params)
        self.min_incr = min_incr
        self.length_traj = length_traj
        self.max_length = self.n_dim

        self.sampling_categorical = True

        # Policy
        self.n_comp = n_comp
        self.beta_params_min = beta_params_min
        self.beta_params_max = beta_params_max

        # Small constant to clamp the inputs to the beta distribution
        self.epsilon = epsilon
        # Small constant to restrict the interval of (test) sets
        self.kappa = kappa

        # Source
        self.source = [self.pad_idx] * self.max_length
        self.eos_idx = -1
        self.eos = (self.eos_idx,)
        
        self._idx_to_value = self._build_index_to_value() 
        self._choice_dims = [len(p.get_choices()) for p in self.parameters]
        self._policy_dim = sum(self._choice_dims)
        self._offsets = [sum(self._choice_dims[:i]) for i in range(self.n_dim)]
        self._param_ranges = [self.idparam2actionrange(i) for i in range(self.n_dim)]
        
        # Base class init
        super().__init__(**kwargs)
        self.continuous = True
        

    def get_action_space(self):
        """Build fixed-width representatives for mixed categorical/cube actions

        Categorical actions begin with (parameter_id, choice_id)
        Continuous actions use (-1, -1) followed by their sampled increments
        """
        categorical_params = [
            (param_id, param)
            for param_id, param in enumerate(self.parameters)
            if param.type == "categorical"
        ]

        action_dim = self.n_continuous_dims + 2 + 1
        zero_increments = (0.0,) * self.n_continuous_dims

        # discrete actions | maybe have only one discrete action template? idk this is confusing to me
        action_space = []
        for param_id, param in categorical_params:
            for choice_id, _ in enumerate(param.get_choices()):
                action_space.append(
                    (param_id, choice_id, *zero_increments, 0)
                )
        # continuous actions
        self.representative_no_source = (
            -1,
            -1,
            *zero_increments,
            0,
        )
        self.representative_source = (
            -1,
            -1,
            *zero_increments,
            1,
        )
        # eos
        self.eos = (np.inf,) * action_dim
        action_space.extend(
            [
                self.representative_no_source,
                self.representative_source,
                self.eos,
            ]
        )
        return action_space

    def action2representative(self, action):
        """
        Converts an action into its representative form.
        """
        if action == self.eos:
            return self.eos
        elif action[0] == -1:
            return self.representative_source if action[-1] == 1 else self.representative_no_source
        return action # hopefully only correct actions are passed in

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.
        """
        return int(np.ceil(1.0 / self.min_incr) + 2) + 1 + 3


    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Returns the policy output dimension for the given parameters.
        """
        # discrete
        categorical_logits = tfloat(
            params.get("categorical_logits", [0.0] * self.n_discrete_actions),
            float_type=self.float,
            device=self.device,
        ).flatten()

        # continuous
        self._len_policy_output_cont = self.n_continuous_dims * self.n_comp * 3
        policy_output_cont = torch.empty(
            self._len_policy_output_cont,
            dtype=self.float,
            device=self.device,
        )
        policy_output_cont[0::3] = params["beta_weights"]
        policy_output_cont[1::3] = self._beta_params_to_policy_outputs("alpha", params)
        policy_output_cont[2::3] = self._beta_params_to_policy_outputs("beta", params)


        # Logit for Bernoulli distribution to model EOS action
        policy_output_eos_logit = torch.logit(
            tfloat(
                [params["bernoulli_eos_prob"]],
                float_type=self.float,
                device=self.device,
            )
        )
        # Logit for Bernoulli distribution to model back-to-source action
        policy_output_bts_logit = torch.logit(
            tfloat(
                [params["bernoulli_bts_prob"]],
                float_type=self.float,
                device=self.device,
            )
        )
        # Concatenate all outputs
        policy_output = torch.cat(
            (
                categorical_logits,
                policy_output_cont,
                policy_output_bts_logit,
                policy_output_eos_logit,
            )
        )
        return policy_output

    def _get_policy_betas_weights(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the weights of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        return policy_output[:, 0 : self._len_policy_output_cont : 3]

    def _get_policy_betas_alpha(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the alphas of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        return policy_output[:, 1 : self._len_policy_output_cont : 3]

    def _get_policy_betas_beta(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the betas of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        return policy_output[:, 2 : self._len_policy_output_cont : 3]

    def _get_policy_eos_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "1"]:
        """
        Reduces a given policy output to the part corresponding to the logit of the
        Bernoulli distribution to model the EOS action.

        See: get_policy_output()
        """
        return policy_output[:, -1]

    
    ############### WIP LINE ###############

    def idparam2actionrange(self, id_param: int) -> Tuple[int, int]:
        """
        Returns the range of actions corresponding to a given parameter index.
        """
        start = sum(len(param.get_choices()) for param in self.parameters[:id_param])
        end = start + len(self.parameters[id_param].get_choices())
        return start, end


    def step(self, action: Tuple[int], skip_mask_check: bool = False):
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False

        self.n_actions += 1

        # EOS is the actual termination signal the batch watches for
        if action == self.eos:
            self.done = True
            return self.state, action, True

        # fill the param
        pos = self._get_seq_length()
        self.state[pos] = action[0]

        return self.state, action, True

    def get_parents(self, state=None, done=None, action=None):
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]  # eos must be a tuple
        if self.equal(state, self.source):
            return [], []
        pos_last_param = self._get_seq_length(state) - 1
        parent = copy(state)
        parent[pos_last_param] = self.pad_idx
        p_action = (state[pos_last_param],)
        return [parent], [p_action]
    
    def _smoke_test(self):
        """Run after __init__ to catch obvious issues early."""
        print(f"action_space size: {len(self.action_space)}")
        print(f"action_space[:5]: {self.action_space[:5]}")
        print(f"action_space_torch shape: {self.action_space_torch.shape}")
        print(f"source: {self.source}")
        print(f"eos: {self.eos}")
        
        # Simulate one full trajectory
        self.reset()
        for step_idx in range(self.n_dim):
            mask = self.get_mask_invalid_actions_forward()
            valid_actions = [a for a, m in zip(self.action_space, mask) if not m]
            print(f"  step {step_idx}: {len(valid_actions)} valid actions, e.g. {valid_actions[0]}")
            state, action, valid = self.step(valid_actions[0])
            print(f"    → state: {state}, done: {self.done}")
        print(f"Final state: {self.state}, done: {self.done}")
    
    def states2proxy(self, states):
        if torch.is_tensor(states[0]):
            states = [s.tolist() for s in states]

        # print(f"states2proxy received {len(states)} states")
        # print(f"first state: {states[0]}")
        # print(f"last state: {states[-1]}")
        # print(f"any incomplete: {any(self.pad_idx in s for s in states)}")
        
        # import traceback

        # for i, state in enumerate(states):
        #     if self.pad_idx in state:
        #         print(f"Incomplete state at index {i}: {state}")
        #         print("Called from:")
        #         traceback.print_stack()
        #         raise AssertionError(f"Incomplete state at index {i}")

        rows = []
        for state in states:
            row = []
            for param_idx, global_action_idx in enumerate(state):
                if global_action_idx == self.pad_idx:
                    row.append(None)  # shouldn't happen for terminating states
                else:
                    start, _ = self._param_ranges[param_idx]
                    local_idx = global_action_idx - start
                    choice = self.parameters[param_idx].get_choices()[local_idx]
                    value = choice[0] if isinstance(choice, tuple) else choice
                    row.append(value)
            rows.append(row)

        return rows

    def states2policy(
        self, states=None
    ) -> TensorType["batch", "policy_input_dim"]:
        if states is None:
            states = [self.state]

        if torch.is_tensor(states[0]):
            states = [s.tolist() for s in states]

        batch_size = len(states)
        out = torch.zeros(batch_size, self._policy_dim, device=self.device)

        for i, state in enumerate(states):
            for param_idx, global_action_idx in enumerate(state):
                if global_action_idx != self.pad_idx:
                    start = self._param_ranges[param_idx][0]
                    local_idx = global_action_idx - start
                    out[i, self._offsets[param_idx] + local_idx] = 1.0

        return out

    def state2readable(self, state: List[int] = None) -> str:
        """
        Converts a state into a human-readable string.
        """
        return [self._idx_to_value[idx] for idx in state if idx != self.pad_idx]

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a state in readable format into the "environment format" (tensor)
        """
        pass
        
    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.
        """
        n_letters = len(self.letters)
        n_per_length = tlong(
            [n_letters**length for length in range(1, self.max_length + 1)],
            device=self.device,
        )
        lengths = Categorical(logits=n_per_length.repeat(n_states, 1)).sample() + 1
        samples = torch.randint(
            low=1, high=n_letters + 1, size=(n_states, self.max_length)
        )
        for idx, length in enumerate(lengths):
            samples[idx, length:] = 0
        return samples.tolist()

    def _pad(self, seq_list: list):
        """
        Pads a sequence represented as a list of indices.
        """
        return seq_list + [self.pad_idx] * (self.max_length - len(seq_list))

    def _unpad(self, seq_list: list):
        """
        Removes the padding from the end off a sequence represented as a list of
        indices.
        """
        if self.pad_idx not in seq_list:
            return seq_list
        return seq_list[: seq_list.index(self.pad_idx)]

    def _get_seq_length(self, state: List[int] = None):
        """
        Returns the effective length of a state, that is ignoring the padding.
        """
        state = self._get_state(state)
        if state[-1] == self.pad_idx:
            return state.index(self.pad_idx)
        else:
            return len(state)


    def _build_index_to_value(self):
        """
        Returns a flat list where index i → actual parameter value.
        Mirrors the order of get_action_space().
        """
        mapping = []
        for param in self.parameters:
            for choice in param.get_choices():
                mapping.append(choice)
        return mapping