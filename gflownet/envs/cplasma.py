"""
Continuous Plasma environment: starting from an emtpy sequence, parameters are added one at a time until the maximum length is reached.
"""

import sys

sys.path.append("../Phase1/src")
from space import PLASMA_ENV

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Beta, Categorical, MixtureSameFamily
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tbool, tfloat


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
        fixed_distr_params: dict = {
            "beta_weights": 1.0,
            "beta_alpha": 10.0,
            "beta_beta": 10.0,
            "bernoulli_bts_prob": 0.1,
            "bernoulli_eos_prob": 0.1,
        },
        random_distr_params: dict = {
            "beta_weights": 1.0,
            "beta_alpha": 10.0,
            "beta_beta": 10.0,
            "bernoulli_bts_prob": 0.1,
            "bernoulli_eos_prob": 0.1,
        },
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
        self.categorical_param_indices = [
            i for i, param in enumerate(self.parameters) if param.type == "categorical"
        ] # 0 to 2
        self.continuous_param_indices = [
            i for i, param in enumerate(self.parameters) if param.type == "continuous"
        ] # 3 to 9
        self.param_idx_to_discrete_idx = {
            param_idx: discrete_idx
            for discrete_idx, param_idx in enumerate(self.categorical_param_indices)
        } # 0 to 2; 0 to 2 yes i know it's useless
        self.param_idx_to_continuous_idx = {
            param_idx: continuous_idx
            for continuous_idx, param_idx in enumerate(self.continuous_param_indices)
        } # 3 to 9; 0 to 6
        self.n_discrete_actions = sum(len(param.get_choices()) for param in self.categorical_params)
        self.n_continuous_dims = len(self.continuous_params)
        self.min_incr = min_incr
        self.length_traj = length_traj
        self.max_length = self.n_dim
        self.mask_dim_base = self.n_discrete_actions + 3  # 3 for continuous, source, eos
        # self.sampling_categorical = True

        # Policy
        self.n_comp = n_comp
        self.beta_params_min = beta_params_min
        self.beta_params_max = beta_params_max

        # Small constant to clamp the inputs to the beta distribution
        self.epsilon = epsilon
        # Small constant to restrict the interval of (test) sets
        self.kappa = kappa

        # Source
        self.source_discrete = [self.pad_idx] * len(self.categorical_params)
        self.source_continuous = [self.pad_idx] * self.n_continuous_dims
        self.source = self._compose_state(self.source_discrete, self.source_continuous)
        self.continuous_action_idx = self.n_discrete_actions
        self.source_flag_idx = self.n_discrete_actions + 1
        self.eos_idx = self.n_discrete_actions + 2
        self.eos = (self.eos_idx,)
        
        self._policy_dim = sum(
            len(param.get_choices()) for param in self.categorical_params
        ) + self.n_continuous_dims
        
        # Base class init
        super().__init__(
            fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,
            **kwargs,
        )
        self.continuous = True
        
    def reset(self, env_id: Union[int, str] = None):
        self.discrete_state = copy(self.source_discrete)
        self.continuous_state = copy(self.source_continuous)
        return super().reset(env_id)

    def set_state(self, state: List, done: Optional[bool] = False):
        state = state.tolist() if torch.is_tensor(state) else copy(state)
        self.discrete_state = [
            state[param_idx] for param_idx in self.categorical_param_indices
        ]
        self.continuous_state = [
            state[param_idx] for param_idx in self.continuous_param_indices
        ]
        return super().set_state(state, done)

    def _compose_state(self, discrete_state, continuous_state):
        state = []
        discrete_idx = 0
        continuous_idx = 0
        for param in self.parameters:
            if param.type == "categorical":
                state.append(discrete_state[discrete_idx])
                discrete_idx += 1
            else:
                state.append(continuous_state[continuous_idx])
                continuous_idx += 1
        return state


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

    def _beta_params_to_policy_outputs(self, param_name: str, params_dict: dict):
        """
        Maps the values of alpha and beta given in the configuration to new values such
        that when passed to _make_increments_distribution, the actual alpha and beta
        passed to the Beta distribution(s) are the ones from the configuration.

        Args
        ----
        param_name : str
            Name of the parameter to transform: alpha or beta

        params_dict : dict
            Dictionary with the complete set of parameters of the distribution.

        See
        ---
        _make_increments_distribution()
        """
        param_value = tfloat(
            params_dict[f"beta_{param_name}"], float_type=self.float, device=self.device
        )
        return torch.logit((param_value - self.beta_params_min) / self.beta_params_max)

    def _get_policy_betas_weights(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the weights of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        start = self.n_discrete_actions
        return policy_output[:, start : start + self._len_policy_output_cont : 3]

    def _get_policy_betas_alpha(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the alphas of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        start = self.n_discrete_actions
        return policy_output[:, start + 1 : start + self._len_policy_output_cont : 3]

    def _get_policy_betas_beta(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the betas of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        start = self.n_discrete_actions
        return policy_output[:, start + 2 : start + self._len_policy_output_cont : 3]

    def _get_policy_eos_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "1"]:
        """
        Reduces a given policy output to the part corresponding to the logit of the
        Bernoulli distribution to model the EOS action.

        See: get_policy_output()
        """
        return policy_output[:, -1]

    def _compute_mask_dim(self):
        """Return one mask entry per categorical action and three continuous flags."""
        return self.mask_dim_base
    
    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        The forward mask has the following structure:
        
        - n_discrete_actions: categorical actions are invalid if their parameter
          has already been filled.
        - next 3 entries: continuous action invalid, source-state flag, EOS invalid.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        if done:
            return [True] * self.mask_dim

        mask = [False] * self.mask_dim_base

        for param_idx, param in enumerate(self.parameters):
            if param.type == "categorical" and state[param_idx] != self.pad_idx:
                start, end = self.idparam2actionrange(param_idx)
                for action_idx in range(start, end):
                    mask[action_idx] = True  # mask choices for this filled parameter

        continuous_dims = [
            state[param_idx]
            for param_idx, param in enumerate(self.parameters)
            if param.type == "continuous"
        ]
        at_continuous_source = all(value == self.pad_idx for value in continuous_dims)
        all_categorical_chosen = all(
            state[param_idx] != self.pad_idx
            for param_idx, param in enumerate(self.parameters)
            if param.type == "categorical"
        )

        if not all_categorical_chosen:
            mask[self.continuous_action_idx] = True # mask continuous sampling until categories are chosen
            mask[self.eos_idx] = True # mask EOS until categories are chosen
            mask[self.source_flag_idx] = True # mask the source action until categories are chosen
        if any(value > 1 - self.min_incr for value in continuous_dims):
            mask[self.continuous_action_idx] = True # mask increments that would overshoot
        if all_categorical_chosen and not at_continuous_source:
            mask[self.source_flag_idx] = True # mark that the continuous state left its source
        if at_continuous_source:
            mask[self.eos_idx] = True # mask EOS until a continuous increment is taken

        return mask


    def get_mask_invalid_actions_backward(self, state=None, done=None):
        """
        The backward mask has the following structure:

        - categorical actions: only allow removing a selected category, after the
          continuous dimensions have returned to their source.
        - next 3 entries: continuous action invalid, back-to-source flag, EOS invalid.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        mask = [True] * self.mask_dim

        if done:
            mask[self.eos_idx] = False # allow undoing EOS
            return mask

        continuous_dims = [
            state[param_idx]
            for param_idx, param in enumerate(self.parameters)
            if param.type == "continuous"
        ]
        at_continuous_source = all(value == self.pad_idx for value in continuous_dims)

        if at_continuous_source:
            for param_idx, param in enumerate(self.parameters):
                if param.type == "categorical" and state[param_idx] != self.pad_idx:
                    start, end = self.idparam2actionrange(param_idx)
                    action_idx = start + int(state[param_idx])
                    if start <= action_idx < end:
                        mask[action_idx] = False # allow removing the selected category
            return mask

        if any(value < self.min_incr for value in continuous_dims):
            mask[self.source_flag_idx] = False # allow returning to the continuous source
        else:
            mask[self.continuous_action_idx] = False # allow a continuous decrement

        return mask

    def get_valid_actions(
        self,
        mask: Optional[List[bool]] = None,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List[Tuple]:
        """
        Returns the list of non-invalid (valid, for short) according to the mask of
        invalid actions.

        As a mixed environment, the returned actions are "representatives", that
        is the actions represented in the action space.

        Parameters
        ----------
        mask : list (optional)
            The mask of a state. If None, it is computed in place.
        state : list (optional)
            A state in GFlowNet format. If None, self.state is used.
        done : bool (optional)
            Whether the trajectory is done. If None, self.done is used.
        backward : bool
            True if the transtion is backwards; False if forward.

        Returns
        -------
        list
            The list of representatives of the valid actions.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if mask is None:
            mask = self.get_mask(state, done, backward)

        actions_valid = [
            action
            for action, is_invalid in zip(
                self.action_space[: self.n_discrete_actions],
                mask[: self.n_discrete_actions],
            )
            if not is_invalid
        ] # Keep each categorical action allowed by its mask entry.
        if backward:
            if (
                mask[self.continuous_action_idx] is False
                or mask[self.source_flag_idx] is False
            ):
                actions_valid.append(self.representative_source) # Add the back-to-source action.
            if mask[self.continuous_action_idx] is False:
                actions_valid.append(self.representative_no_source) # Add a continuous decrement.
        else:
            if mask[self.source_flag_idx] is False:
                actions_valid.append(self.representative_source) # Add the first continuous action.
            elif mask[self.continuous_action_idx] is False:
                actions_valid.append(self.representative_no_source) # Add a later continuous increment.
        if mask[self.eos_idx] is False:
            actions_valid.append(self.eos) # Add EOS when it is valid.
        return actions_valid

    def get_parents(
        self, state: List = None, done: bool = None, action: Tuple[int, float] = None
    ) -> Tuple[List[List], List[Tuple[int, float]]]:
        """
        Defined only because it is required. A ContinuousEnv should be created to avoid
        this issue.
        """
        pass

    def relative_to_absolute_increments(
        self,
        states: TensorType["n_states", "n_dim"],
        increments_rel: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Returns a batch of absolute increments (actions) given a batch of states,
        relative increments and minimum_increments.

        Given a dimension value x, a relative increment r, and a minimum increment m,
        then the absolute increment a is given by:

        Forward:

        a = m + r * (1 - x - m)

        Backward:

        a = m + r * (x - m)
        """
        min_increments = torch.full_like(
            increments_rel, self.min_incr, dtype=self.float, device=self.device
        )
        if is_backward:
            return min_increments + increments_rel * (states - min_increments)
        else:
            return min_increments + increments_rel * (1.0 - states - min_increments)

    def absolute_to_relative_increments(
        self,
        states: TensorType["n_states", "n_dim"],
        increments_abs: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Returns a batch of relative increments (as sampled by the Beta distributions)
        given a batch of states, absolute increments (actions) and minimum_increments.

        Given a dimension value x, an absolute increment a, and a minimum increment m,
        then the relative increment r is given by:

        Forward:

        r = (a - m) / (1 - x - m)

        Backward:

        r = (a - m) / (x - m)
        """
        min_increments = torch.full_like(
            increments_abs, self.min_incr, dtype=self.float, device=self.device
        )
        if is_backward:
            denominator = (states - min_increments).clamp(min=self.epsilon)
            increments_rel = (increments_abs - min_increments) / denominator
            return increments_rel
        else:
            denominator = (1.0 - states - min_increments).clamp(min=self.epsilon)
            return (increments_abs - min_increments) / denominator

    def _step(
        self,
        action: Tuple[float],
        backward: bool,
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Updates self.state given a non-EOS action. This method is called by both step()
        and step_backwards(), with the corresponding value of argument backward.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple of length n_dim, with the
            absolute increment for each dimension.

        backward : bool
            If True, perform backward step. Otherwise (default), perform forward step.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if action[0] != -1: # Categorical actions
            param_idx, choice_idx = int(action[0]), int(action[1])
            discrete_idx = self.param_idx_to_discrete_idx[param_idx] # Convert full parameter ID to the compact category list
            self.discrete_state[discrete_idx] = self.pad_idx if backward else choice_idx # Forward stores the choice, backward resets it
            self.n_actions += 1
            self.state = self._compose_state(self.discrete_state, self.continuous_state) # Rebuild state in parameter order
            return self.state, action, True

        if not backward and action[-1] == 1 and self.continuous_state == self.source_continuous: # First forward action in continuous space
            continuous_state = [0.0] * self.n_continuous_dims # Start numeric values at zero (keep chosen categories seperate)
        else:
            assert action[-1] == 0 # Other continuous actions
            continuous_state = copy(self.continuous_state)
        for dim, increment in enumerate(action[2:-1]):
            if backward:
                continuous_state[dim] -= increment
            else:
                continuous_state[dim] += increment
        if any(value < 0.0 or value > 1.0 for value in continuous_state):
            return self.state, action, False # Sanity check
        self.continuous_state = continuous_state
        self.n_actions += 1
        self.state = self._compose_state(self.discrete_state, self.continuous_state)
        return self.state, action, True

    def step(
        self, action: Tuple[float]
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes step given an action. An action is a choice of a discrete variable or
        the absolute increment of each other continuous dimension

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple of length n_dim, with the
            absolute increment for each dimension.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if self.done:
            return self.state, action, False
        if action == self.eos:
            if (
                any(value == self.pad_idx for value in self.discrete_state)
                or all(value == self.pad_idx for value in self.continuous_state)
            ):
                return self.state, action, False # EOS requires all categories and a populated numeric state
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # Otherwise perform action
        return self._step(action, backward=False)

    def step_backwards(
        self, action: Tuple[int, float]
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes backward step given an action. An action is a choice of a discrete variable or
        the absolute decrement of each other continuous dimension.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple of length n_dim, with the
            absolute decrement for each dimension.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If done is True, set done to False, increment n_actions and return same state
        if self.done:
            assert action == self.eos
            self.done = False # Undo EOS without changing the configuration
            self.n_actions += 1
            return self.state, action, True
        if action == self.eos:
            return self.state, action, False # EOS can only be undone from a terminal state
        if action[0] == -1 and action[-1] == 1: # Back-to-source action
            if (
                any(value == self.pad_idx for value in self.discrete_state)
                or all(value == self.pad_idx for value in self.continuous_state)
            ):
                return self.state, action, False # Back-to-source needs selected categories and numeric values to reset.
            self.n_actions += 1
            self.continuous_state = copy(self.source_continuous) # Reset only numeric values, keep categories for now
            self.state = self._compose_state(self.discrete_state, self.continuous_state)
            return self.state, action, True
        # Otherwise perform action
        return self._step(action, backward=True)

    
    def _get_jacobian_diag(
        self,
        states_from: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Computes the diagonal of the Jacobian of the sampled actions with respect to
        the target states.

        Forward: the sampled variables are the relative increments r_f and the state
        updates (s -> s') are:

        s' = s + m + r_f(1 - s - m)
        r_f = (s' - s - m) / (1 - s - m)

        Therefore, the derivative of r_f wrt to s' is

        dr_f/ds' = 1 / (1 - s - m)

        Backward: the sampled variables are the relative decrements r_b and the state
        updates (s' -> s) are:

        s = s' - m - r_b(s' - m)
        r_b = (s' - s - m) / (s' - m)

        Therefore, the derivative of r_b wrt to s is

        dr_b/ds = -1 / (s' - m)

        We take the absolute value of the derivative (Jacobian).

        The derivatives of the components of r with respect to dimensions of s or s'
        other than itself are zero. Therefore, the Jacobian is diagonal and the
        determinant is the product of the diagonal.
        """
        min_increments = torch.full_like(
            states_from, self.min_incr, dtype=self.float, device=self.device
        )
        if is_backward:
            return 1.0 / ((states_from - min_increments))
        else:
            return 1.0 / ((1.0 - states_from - min_increments))


    def _make_increments_distribution(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
    ) -> MixtureSameFamily:
        mix_logits = self._get_policy_betas_weights(policy_outputs).reshape(
            -1, self.n_continuous_dims, self.n_comp
        )
        mix = Categorical(logits=mix_logits)
        alphas = self._get_policy_betas_alpha(policy_outputs).reshape(
            -1, self.n_continuous_dims, self.n_comp
        )
        alphas = self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
        betas = self._get_policy_betas_beta(policy_outputs).reshape(
            -1, self.n_continuous_dims, self.n_comp
        )
        betas = self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
        beta_distr = Beta(alphas, betas)
        return MixtureSameFamily(mix, beta_distr)


    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
        is_backward: bool,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Parameters
        ----------
        policy_outputs : tensor
            The output of the GFlowNet policy model.
        mask : tensor
            The mask containing information about invalid actions and special cases.
        actions : list or tensor
            The actions (absolute increments) from each state in the batch for which to
            compute the log probability.
        states_from : tensor
            The states originating the actions, in GFlowNet format. They are required
            so as to compute the relative increments and the Jacobian.
        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default). Required, since the computation for forward and backward actions
            is different.
        """
        if is_backward:
            return self._get_logprobs_backward(
                policy_outputs, actions, mask, states_from
            )
        else:
            return self._get_logprobs_forward(
                policy_outputs, actions, mask, states_from
            )

    def _get_logprobs_forward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of forward actions.
        """
        # Initialize variables
        actions = tfloat(actions, float_type=self.float, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )

        # Dicrete variables
        is_categorical = torch.isfinite(actions[:, 0]) & (actions[:, 0] >= 0)
        logprobs_categorical = torch.zeros(
            n_states, dtype=self.float, device=self.device
        )
        if torch.any(is_categorical):
            categorical_logits = policy_outputs[
                is_categorical, : self.n_discrete_actions
            ].clone()
            categorical_mask = mask[is_categorical, : self.n_discrete_actions]
            categorical_logits[categorical_mask] = -torch.inf
            action_indices = torch.tensor(
                [
                    self.idparam2actionrange(int(action[0]))[0] + int(action[1])
                    for action in actions[is_categorical].tolist()
                ],
                dtype=torch.long,
                device=self.device,
            )
            logprobs_categorical[is_categorical] = Categorical(
                logits=categorical_logits
            ).log_prob(action_indices)
        
        is_eos = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        logprobs_eos = torch.zeros(n_states, dtype=self.float, device=self.device)
        logprobs_increments_rel = torch.zeros(
            (n_states, self.n_continuous_dims), dtype=self.float, device=self.device
        )
        log_jacobian_diag = torch.zeros(
            (n_states, self.n_continuous_dims), device=self.device, dtype=self.float
        )
        eos_tensor = tfloat(self.eos, float_type=self.float, device=self.device)
        # Determine source states
        is_source = ~mask[:, self.source_flag_idx]
        # EOS is the only possible action if continuous actions are invalid.
        is_eos_forced = mask[:, self.continuous_action_idx] & ~is_categorical
        is_eos[is_eos_forced] = True
        # Ensure that is_eos_forced does not include any source state
        assert not torch.any(
            torch.logical_and(is_source & ~is_categorical, is_eos_forced)
        )
        # Get sampled EOS actions and get log probs from Bernoulli distribution
        do_eos = torch.logical_and(
            ~is_source & ~is_categorical, ~is_eos_forced
        )
        if torch.any(do_eos):
            is_eos_sampled = torch.zeros_like(do_eos)
            is_eos_sampled[do_eos] = torch.all(actions[do_eos] == eos_tensor, dim=1)
            is_eos[is_eos_sampled] = True
            logits_eos = self._get_policy_eos_logit(policy_outputs)[do_eos]
            distr_eos = Bernoulli(logits=logits_eos)
            logprobs_eos[do_eos] = distr_eos.log_prob(
                is_eos_sampled[do_eos].to(self.float)
            )
        # Get log probs of relative increments if EOS was not the sampled or forced
        # action
        do_increments = ~is_eos & ~is_categorical
        if torch.any(do_increments):
            # Get absolute increments
            increments = actions[do_increments, 2:-1]
            # Make sure increments are finite
            assert torch.any(torch.isfinite(increments))
            # Compute relative increments from absolute increments if state is not
            # source
            is_relative = ~is_source[do_increments]
            if torch.any(is_relative):
                states_from_rel = tfloat(
                    states_from_tensor[do_increments],
                    float_type=self.float,
                    device=self.device,
                )[is_relative][:, self.continuous_param_indices]
                increments[is_relative] = self.absolute_to_relative_increments(
                    states_from_rel,
                    increments[is_relative],
                    is_backward=False,
                )
            # Compute diagonal of the Jacobian (see _get_jacobian_diag()) if state is
            # not source
            is_relative = torch.logical_and(do_increments, ~is_source)
            if torch.any(is_relative):
                log_jacobian_diag[is_relative] = torch.log(
                    self._get_jacobian_diag(
                        states_from_rel,
                        is_backward=False,
                    )
                )
                
            # Get logprobs
            distr_increments = self._make_increments_distribution(
                policy_outputs[do_increments]
            )
            # Clamp because increments of 0.0 or 1.0 would yield nan
            logprobs_increments_rel[do_increments] = distr_increments.log_prob(
                torch.clamp(increments, min=self.epsilon, max=(1 - self.epsilon))
            )
            # Sum log Jacobian across dimensions
        log_det_jacobian = torch.sum(log_jacobian_diag, dim=1)
        # Compute combined probabilities
        sumlogprobs_increments = logprobs_increments_rel.sum(axis=1)
        logprobs = (
            logprobs_categorical
            + logprobs_eos
            + sumlogprobs_increments
            + log_det_jacobian
        )
        return logprobs

    def _get_logprobs_backward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of backward actions.
        """
        # Initialize variables
        actions = tfloat(actions, float_type=self.float, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )
        # Discrete variables
        is_categorical = torch.isfinite(actions[:, 0]) & (actions[:, 0] >= 0)
        logprobs_categorical = torch.zeros(
            n_states, dtype=self.float, device=self.device
        )
        if torch.any(is_categorical):
            categorical_logits = policy_outputs[
                is_categorical, : self.n_discrete_actions
            ].clone()
            categorical_mask = mask[is_categorical, : self.n_discrete_actions]
            categorical_logits[categorical_mask] = -torch.inf
            action_indices = torch.tensor(
                [
                    self.idparam2actionrange(int(action[0]))[0] + int(action[1])
                    for action in actions[is_categorical].tolist()
                ],
                dtype=torch.long,
                device=self.device,
            )
            logprobs_categorical[is_categorical] = Categorical(
                logits=categorical_logits
            ).log_prob(action_indices)
        
        is_bts = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        logprobs_bts = torch.zeros(n_states, dtype=self.float, device=self.device)
        logprobs_increments_rel = torch.zeros(
            (n_states, self.n_continuous_dims), dtype=self.float, device=self.device
        )
        log_jacobian_diag = torch.zeros(
            (n_states, self.n_continuous_dims), device=self.device, dtype=self.float
        )
        # EOS is the only possible action only if done is True.
        is_eos = ~mask[:, self.eos_idx]
        # Back-to-source (BTS) is the only possible action when it is unmasked.
        is_bts_forced = ~mask[:, self.source_flag_idx] & ~is_categorical
        is_bts[is_bts_forced] = True
        # Get sampled BTS actions and get log probs from Bernoulli distribution
        do_bts = torch.logical_and(
            ~is_bts_forced & ~is_categorical, ~is_eos
        )
        if torch.any(do_bts):
            # BTS actions are equal to the originating states
            is_bts_sampled = torch.zeros_like(do_bts)
            is_bts_sampled[do_bts] = actions[do_bts, -1] == 1
            is_bts[is_bts_sampled] = True
            logits_bts = self._get_policy_source_logit(policy_outputs)[do_bts]
            distr_bts = Bernoulli(logits=logits_bts)
            logprobs_bts[do_bts] = distr_bts.log_prob(
                is_bts_sampled[do_bts].to(self.float)
            )
        # Get log probs of relative increments if actions were neither BTS nor EOS
        do_increments = torch.logical_and(
            ~is_bts & ~is_eos, ~is_categorical
        )
        if torch.any(do_increments):
            # Get absolute increments
            increments = actions[do_increments, 2:-1]
            states = states_from_tensor[do_increments][
                :, self.continuous_param_indices
            ]
            # Compute absolute increments from all sampled relative increments
            increments = self.absolute_to_relative_increments(
                states,
                increments,
                is_backward=True,
            )
            # Make sure increments are finite
            assert torch.all(torch.isfinite(increments))
            # Compute diagonal of the Jacobian (see _get_jacobian_diag())
            log_jacobian_diag[do_increments] = torch.log(
                self._get_jacobian_diag(
                    states,
                    is_backward=True,
                )
            )
            
            # Get logprobs
            distr_increments = self._make_increments_distribution(
                policy_outputs[do_increments]
            )
            # Clamp because increments of 0.0 or 1.0 would yield nan
            logprobs_increments_rel[do_increments] = distr_increments.log_prob(
                torch.clamp(increments, min=self.epsilon, max=(1 - self.epsilon))
            )

        # Sum log Jacobian across dimensions
        log_det_jacobian = torch.sum(log_jacobian_diag, dim=1)
        # Compute combined probabilities
        sumlogprobs_increments = logprobs_increments_rel.sum(axis=1)
        logprobs = (
            logprobs_categorical
            + logprobs_bts
            + sumlogprobs_increments
            + log_det_jacobian
        )
        # Ensure that logprobs of forced EOS are 0
        logprobs[is_eos] = 0.0
        return logprobs

    def _get_policy_eos_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "1"]:
        """
        Reduces a given policy output to the part corresponding to the logit of the
        Bernoulli distribution to model the EOS action.

        See: get_policy_output()
        """
        return policy_output[:, -1]

    def _get_policy_source_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "1"]:
        """
        Reduces a given policy output to the part corresponding to the logit of the
        Bernoulli distribution to model the back-to-source action.

        See: get_policy_output()
        """
        return policy_output[:, -2]


    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List = None,
        is_backward: Optional[bool] = False,
        random_action_prob: Optional[float] = 0.0,
        temperature_logits: Optional[float] = 1.0,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.

        This method overwrites the methof of the GFlowNetEnv because it
        is a continious enviroment.

        Parameters
        ----------
        policy_outputs : tensor
            The output of the GFlowNet policy model.
        mask : tensor
            The mask of invalid actions. For continuous or mixed environments, the mask
            may be tensor with an arbitrary length contaning information about special
            states, as defined elsewhere in the environment.
        states_from : tensor
            The states originating the actions, in GFlowNet format. Ignored in discrete
            environments and only required in certain continuous environments.
        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        random_action_prob : float, optional
            The probability of sampling a random action. If larger than one, the model
            outputs will be replaced by a random policy vector with probability
            `random_action_prob`, according to Bernoulli distribution.
        temperature_logits : float, optional
            A scalar by which the model outputs are divided to temper the sampling
            distribution.

        Returns
        -------
        actions : list
            The list of sampled actions.
        """
        if not is_backward:
            return self._sample_actions_batch_forward(
                policy_outputs=policy_outputs,
                mask=mask,
                states_from=states_from,
                random_action_prob=random_action_prob,
                temperature_logits=temperature_logits,
            )
        else:
            return self._sample_actions_batch_backward(
                policy_outputs=policy_outputs,
                mask=mask,
                states_from=states_from,
                random_action_prob=random_action_prob,
                temperature_logits=temperature_logits,
            )

    def _sample_actions_batch_forward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List = None,
        random_action_prob: Optional[float] = 0.0,
        temperature_logits: Optional[float] = 1.0,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a a batch of forward actions from a batch of policy outputs.

        An action indicates, for each dimension, the absolute increment of the
        dimension value. However, in order to ensure that trajectories have finite
        length, increments must have a minumum increment (self.min_incr) except if the
        originating state is the source state (special case, see
        get_mask_invalid_actions_forward()). Furthermore, absolute increments must also
        be smaller than the distance from the dimension value to the edge of the cube
        (1.0). In order to accomodate these constraints, first relative
        increments (in [0, 1]) are sampled from a (mixture of) Beta distribution(s),
        where 0.0 indicates an absolute increment of min_incr and 1.0 indicates an
        absolute increment of 1 - x + min_incr (going to the edge).

        Therefore, given a dimension value x, a relative increment r, a minimum
        increment m and a maximum value 1, the absolute increment a is given by:

        a = m + r * (1 - x - m)

        The continuous distribution to sample the continuous action described above
        must be mixed with the discrete distribution to model the sampling of the EOS
        action. The EOS action can be sampled from any state except from the source
        state or whether the trajectory is done. That the EOS action is invalid is
        indicated by mask[-1] being False.

        Finally, regarding the constraints on the increments, the following special
        cases are taken into account:

        - The originating state is the source state: in this case, the minimum
          increment is 0.0 instead of self.min_incr. This is to ensure that the entire
          state space can be reached. This is indicated by mask[-2] being False.
        - The value at any dimension is at a distance from the cube edge smaller than the
          minimum increment (x > 1 - m). In this case, only EOS is valid.
          This is indicated by mask[0] being True (continuous actions are invalid).
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )
        is_categorical = (~mask[:, : self.n_discrete_actions]).any(dim=1)
        is_continuous = ~is_categorical
        is_eos = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        # Determine source states
        is_source = ~mask[:, self.source_flag_idx]
        # EOS is the only possible action if continuous actions are invalid.
        is_eos_forced = mask[:, self.continuous_action_idx] & is_continuous
        is_eos[is_eos_forced] = True
        # Ensure that is_eos_forced does not include any source state
        assert not torch.any(torch.logical_and(is_source, is_eos_forced))

        # Randomize actions and temper the logits
        logits_sampling = self.randomize_and_temper_sampling_distribution(
            policy_outputs, random_action_prob, temperature_logits
        )

        # Sample categorical choices
        actions_tensor = torch.full(
            (n_states, self.n_continuous_dims + 3),
            torch.inf,
            dtype=self.float,
            device=self.device,
        )
        if torch.any(is_categorical):
            categorical_logits = logits_sampling[
                is_categorical, : self.n_discrete_actions
            ].clone()
            categorical_mask = mask[is_categorical, : self.n_discrete_actions]
            categorical_logits[categorical_mask] = -torch.inf
            categorical_indices = Categorical(logits=categorical_logits).sample()
            categorical_actions = tfloat(
                self.action_space[: self.n_discrete_actions],
                float_type=self.float,
                device=self.device,
            )
            actions_tensor[is_categorical] = categorical_actions[categorical_indices]

        # Sample EOS from Bernoulli distribution
        do_eos = torch.logical_and(
            is_continuous & ~is_source, ~is_eos_forced
        )
        if torch.any(do_eos):
            is_eos_sampled = torch.zeros_like(do_eos)
            logits_eos = self._get_policy_eos_logit(logits_sampling)[do_eos]
            distr_eos = Bernoulli(logits=logits_eos)
            is_eos_sampled[do_eos] = tbool(distr_eos.sample(), device=self.device)
            is_eos[is_eos_sampled] = True
        # Sample (relative) increments if EOS is not the (sampled or forced) action
        do_increments = is_continuous & ~is_eos
        if torch.any(do_increments):
            distr_increments = self._make_increments_distribution(
                logits_sampling[do_increments]
            )
            # Shape of increments: [n_do_increments, n_continuous_dims]
            increments = distr_increments.sample()
            # Compute absolute increments from sampled relative increments if state is
            # not source
            is_relative = ~is_source[do_increments]
            states_from_rel = tfloat(
                states_from_tensor[do_increments],
                float_type=self.float,
                device=self.device,
            )[is_relative][:, self.continuous_param_indices]
            increments[is_relative] = self.relative_to_absolute_increments(
                states_from_rel,
                increments[is_relative],
                is_backward=False,
            )
        # Build actions
        if torch.any(do_increments):
            # Add the continuous-action sentinels and source flag.
            actions_tensor[do_increments, :2] = -1
            actions_tensor[do_increments, 2:-1] = increments
            actions_tensor[do_increments, -1] = 0
        actions_tensor[is_source & do_increments, -1] = 1
        return [tuple(a) for a in actions_tensor.tolist()]

    def _sample_actions_batch_backward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List = None,
        random_action_prob: Optional[float] = 0.0,
        temperature_logits: Optional[float] = 1.0,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a a batch of backward actions from a batch of policy outputs.

        An action indicates, for each dimension, the absolute increment of the
        dimension value. However, in order to ensure that trajectories have finite
        length, increments must have a minumum increment (self.min_incr). Furthermore,
        absolute increments must also be smaller than the distance from the dimension
        value to the edge of the cube. In order to accomodate these constraints, first
        relative increments (in [0, 1]) are sampled from a (mixture of) Beta
        distribution(s), where 0.0 indicates an absolute increment of min_incr and 1.0
        indicates an absolute increment of x (going back to the source).

        Therefore, given a dimension value x, a relative increment r, a minimum
        increment m and a maximum value 1, the absolute increment a is given by:

        a = m + r * (x - m)

        The continuous distribution to sample the continuous action described above
        must be mixed with the discrete distribution to model the sampling of the back
        to source (BTS) action. While the BTS action is also a continuous action, it
        needs to be modelled with a (discrete) Bernoulli distribution in order to
        ensure that this action has positive likelihood.

        Finally, regarding the constraints on the increments, the special case where
        the trajectory is done and the only possible action is EOS, is also taken into
        account.
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        is_categorical = (~mask[:, : self.n_discrete_actions]).any(dim=1)
        is_continuous = ~is_categorical
        is_bts = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        # EOS is the only possible action only if done is True.
        is_eos = ~mask[:, self.eos_idx] & is_continuous
        # Back-to-source (BTS) is the only possible action when it is unmasked.
        is_bts_forced = ~mask[:, self.source_flag_idx] & is_continuous
        is_bts[is_bts_forced] = True

        # Randomize actions and temper the logits
        logits_sampling = self.randomize_and_temper_sampling_distribution(
            policy_outputs, random_action_prob, temperature_logits
        )

        # Sample categorical choices
        actions_tensor = torch.full(
            (n_states, self.n_continuous_dims + 3),
            torch.inf,
            dtype=self.float,
            device=self.device,
        )
        if torch.any(is_categorical):
            categorical_logits = logits_sampling[
                is_categorical, : self.n_discrete_actions
            ].clone()
            categorical_mask = mask[is_categorical, : self.n_discrete_actions]
            categorical_logits[categorical_mask] = -torch.inf
            categorical_indices = Categorical(logits=categorical_logits).sample()
            categorical_actions = tfloat(
                self.action_space[: self.n_discrete_actions],
                float_type=self.float,
                device=self.device,
            )
            actions_tensor[is_categorical] = categorical_actions[categorical_indices]

        # Sample BTS from Bernoulli distribution
        do_bts = is_continuous & ~is_bts_forced & ~is_eos
        if torch.any(do_bts):
            is_bts_sampled = torch.zeros_like(do_bts)
            logits_bts = self._get_policy_source_logit(logits_sampling)[do_bts]
            distr_bts = Bernoulli(logits=logits_bts)
            is_bts_sampled[do_bts] = tbool(distr_bts.sample(), device=self.device)
            is_bts[is_bts_sampled] = True
        # Sample relative increments if actions are neither BTS nor EOS
        do_increments = is_continuous & ~is_bts & ~is_eos
        if torch.any(do_increments):
            distr_increments = self._make_increments_distribution(
                logits_sampling[do_increments]
            )
            # Shape of increments_rel: [n_do_increments, n_continuous_dims]
            increments = distr_increments.sample()
            # Compute absolute increments from all sampled relative increments
            states_from_rel = tfloat(
                states_from, float_type=self.float, device=self.device
            )[do_increments][:, self.continuous_param_indices]
            increments = self.relative_to_absolute_increments(
                states_from_rel,
                increments,
                is_backward=True,
            )
        # Build actions
        actions_tensor[is_eos] = tfloat(
            self.eos, float_type=self.float, device=self.device
        )
        if torch.any(do_increments):
            # Add the continuous-action sentinels and source flag.
            actions_tensor[do_increments, :2] = -1
            actions_tensor[do_increments, 2:-1] = increments
            actions_tensor[do_increments, -1] = 0
        if torch.any(is_bts):
            # BTS uses the representative source action.
            actions_tensor[is_bts] = tfloat(
                self.representative_source,
                float_type=self.float,
                device=self.device,
            )
        return [tuple(a) for a in actions_tensor.tolist()]


    ############### WIP LINE ###############

    def idparam2actionrange(self, id_param: int) -> Tuple[int, int]:
        """
        Returns the categorical-action range for a categorical parameter index.
        """
        param = self.parameters[id_param]
        if param.type != "categorical":
            raise ValueError(f"Parameter '{param.name}' is not categorical")
        start = sum(
            len(previous.get_choices())
            for previous in self.parameters[:id_param]
            if previous.type == "categorical"
        )
        end = start + len(param.get_choices())
        return start, end
    
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
            for param_idx, param in enumerate(self.parameters):
                value = state[param_idx]
                if value == self.pad_idx:
                    row.append(None)  # shouldn't happen for terminating states
                elif param.type == "categorical":
                    row.append(param.decode(int(value)))
                else:
                    row.append(param.decode(float(value)))
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
        n_categorical_features = sum(
            len(param.get_choices()) for param in self.categorical_params
        )

        for i, state in enumerate(states):
            categorical_offset = 0
            continuous_idx = 0
            for param_idx, param in enumerate(self.parameters):
                value = state[param_idx]
                if param.type == "categorical":
                    if value != self.pad_idx:
                        out[i, categorical_offset + int(value)] = 1.0
                    categorical_offset += len(param.get_choices())
                else:
                    out[i, n_categorical_features + continuous_idx] = float(value)
                    continuous_idx += 1

        return out

    def state2readable(self, state: List[int] = None) -> List:
        """
        Converts a state into a human-readable string.
        """
        state = self._get_state(state)
        return self.states2proxy([state])[0]

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
        n_per_length = tfloat(
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

