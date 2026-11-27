from pathlib import Path

import common
import numpy as np
import pytest
import torch
from torch.distributions import Bernoulli, Beta, Categorical

from gflownet.envs.cplasma import ContinuousPlasma
from gflownet.proxy.cplasma import ContinuousPlasmaRewardProxy
from gflownet.utils.common import tbool, tfloat

MODELS_CONT_DIR = Path(__file__).resolve().parents[4] / "Phase1" / "models_continuous"


@pytest.fixture
def cplasma():
    return ContinuousPlasma(n_comp=2, min_incr=0.1)


def test__environment__initializes_and_dimensions_match(cplasma):
    env = cplasma
    assert env.n_dim == 10
    assert env.n_discrete_actions == 9
    assert env.n_continuous_dims == 7
    assert env.action_dim == 10
    assert len(env.action_space) == 12
    assert env.mask_dim == 12
    assert env.policy_input_dim == 16
    assert env.policy_output_dim == 9 + 7 * 2 * 3 + 2


def test__get_policy_output__fixed_and_random(cplasma):
    env = cplasma
    for policy_out, params in [
        (env.fixed_policy_output.unsqueeze(0), env.fixed_distr_params),
        (env.random_policy_output.unsqueeze(0), env.random_distr_params),
    ]:
        assert torch.all(
            env._get_policy_betas_weights(policy_out) == params["beta_weights"]
        )
        assert torch.all(
            env._get_policy_betas_alpha(policy_out)
            == env._beta_params_to_policy_outputs("alpha", params)
        )
        assert torch.all(
            env._get_policy_betas_beta(policy_out)
            == env._beta_params_to_policy_outputs("beta", params)
        )
        assert torch.all(
            env._get_policy_eos_logit(policy_out)
            == torch.logit(torch.tensor(params["bernoulli_eos_prob"]))
        )
        assert torch.all(
            env._get_policy_source_logit(policy_out)
            == torch.logit(torch.tensor(params["bernoulli_bts_prob"]))
        )


def test__masks_forward_and_backward_across_phases(cplasma):
    env = cplasma

    # 1. Source: all 9 categorical actions valid; continuous/source/eos masked
    mask_f = env.get_mask_invalid_actions_forward(env.source, done=False)
    assert mask_f[:9] == [False] * 9
    assert mask_f[9:] == [True, True, True]
    mask_b = env.get_mask_invalid_actions_backward(env.source, done=False)
    assert all(mask_b)

    # 2. Partial categorical: param 0 filled -> choices 0,1 masked; rest categorical valid
    partial_cat = [1, -1, -1] + [-1] * 7
    mask_f = env.get_mask_invalid_actions_forward(partial_cat, done=False)
    assert mask_f[:2] == [True, True]
    assert mask_f[2:9] == [False] * 7
    assert mask_f[9:] == [True, True, True]
    mask_b = env.get_mask_invalid_actions_backward(partial_cat, done=False)
    assert mask_b[1] is False
    assert sum(not m for m in mask_b) == 1

    # 3. Continuous source (all categories chosen, continuous still at -1)
    cont_source = [0, 2, 1] + [-1] * 7
    mask_f = env.get_mask_invalid_actions_forward(cont_source, done=False)
    assert mask_f[:9] == [True] * 9
    assert mask_f[9:] == [False, False, True]
    mask_b = env.get_mask_invalid_actions_backward(cont_source, done=False)
    assert mask_b[0] is False and mask_b[4] is False and mask_b[6] is False
    assert mask_b[9:] == [True, True, True]

    # 4. Interior continuous state (all dims >= min_incr and <= 1 - min_incr)
    interior = [0, 1, 2] + [0.5] * 7
    mask_f = env.get_mask_invalid_actions_forward(interior, done=False)
    assert mask_f[:9] == [True] * 9
    assert mask_f[9:] == [False, True, False]
    mask_b = env.get_mask_invalid_actions_backward(interior, done=False)
    assert mask_b[:9] == [True] * 9
    assert mask_b[9:] == [False, True, True]

    # 5. Upper edge (> 1 - min_incr): continuous invalid, only EOS valid forward
    upper_edge = [0, 1, 2, 0.95] + [0.5] * 6
    mask_f = env.get_mask_invalid_actions_forward(upper_edge, done=False)
    assert mask_f[9:] == [True, True, False]

    # 6. Lower edge (< min_incr): BTS forced backward
    lower_edge = [0, 1, 2, 0.05] + [0.5] * 6
    mask_b = env.get_mask_invalid_actions_backward(lower_edge, done=False)
    assert mask_b[9:] == [True, False, True]

    # 7. Done=True
    assert all(env.get_mask_invalid_actions_forward(interior, done=True))
    mask_b_done = env.get_mask_invalid_actions_backward(interior, done=True)
    assert all(mask_b_done[:-1]) and mask_b_done[-1] is False


def test__states2proxy_preserves_continuous_floats(cplasma):
    env = cplasma
    state = [1, 2, 3, 0.1234, 0.5, 0.25, 0.75, 0.333, 0.666, 0.8765]
    decoded = env.states2proxy([state])[0]
    assert decoded[0] == "nitrogen"
    assert decoded[1] == "type_c"
    assert decoded[2] == "steel"
    assert np.isclose(decoded[3], 0.1234 * 500.0)
    assert np.isclose(decoded[4], 5.0)
    assert np.isclose(decoded[6], 3.0 + 0.75 * 2.0)
    assert np.isclose(decoded[7], 1.0 + 0.333 * 14.0)
    assert np.isclose(decoded[9], 900.0 + 0.8765 * 200.0)


def test__states2policy_and_readable_roundtrip(cplasma):
    env = cplasma
    states = env.get_uniform_terminating_states(4, seed=7)
    pol = env.states2policy(states)
    assert pol.shape == (4, env.policy_input_dim)
    assert pol.dtype == env.float

    for s in states:
        readable = env.state2readable(s)
        assert env.equal(s, env.readable2state(readable))
        assert env.equal(s, env.readable2state(str(readable)))


def test__get_logprobs_analytical_forward_and_backward(cplasma):
    env = cplasma
    # Use n_comp=1 Beta(1, 1) -> Uniform(0, 1) to make analytical values exact
    env_u = ContinuousPlasma(
        n_comp=1,
        min_incr=0.1,
        fixed_distr_params={
            "beta_weights": 1.0,
            "beta_alpha": 1.0,
            "beta_beta": 1.0,
            "bernoulli_bts_prob": 0.25,
            "bernoulli_eos_prob": 0.25,
        },
    )
    policy_out = env_u.fixed_policy_output.unsqueeze(0)

    # 1. Categorical step from source: 9 uniform valid actions -> log(1/9)
    cat_action = (0, 1, *((0.0,) * 7), 0)
    mask_f = tbool(
        [env_u.get_mask_invalid_actions_forward(env_u.source, False)],
        device=env_u.device,
    )
    lp_cat = env_u.get_logprobs(policy_out, [cat_action], mask_f, [env_u.source], False)
    assert torch.isclose(lp_cat, torch.log(torch.tensor([1.0 / 9.0])), atol=1e-5)

    # 2. Continuous jump from continuous source under Beta(1,1) -> logprob == 0.0
    cont_src = [0, 1, 2] + [-1] * 7
    jump_action = (-1, -1, 0.3, 0.4, 0.5, 0.2, 0.6, 0.7, 0.8, 1)
    mask_f = tbool(
        [env_u.get_mask_invalid_actions_forward(cont_src, False)],
        device=env_u.device,
    )
    lp_src = env_u.get_logprobs(policy_out, [jump_action], mask_f, [cont_src], False)
    assert torch.isclose(lp_src, torch.tensor([0.0]), atol=1e-5)

    # 3. Interior continuous forward step + EOS + Jacobian check
    s_int = [0, 1, 2] + [0.3] * 7
    incr = 0.25
    cont_action = (-1, -1, *((incr,) * 7), 0)
    mask_f = tbool(
        [env_u.get_mask_invalid_actions_forward(s_int, False)],
        device=env_u.device,
    )
    lp_int = env_u.get_logprobs(policy_out, [cont_action], mask_f, [s_int], False)
    expected_lp = np.log(0.75) + 7 * np.log(1.0 / (1.0 - 0.3 - 0.1))
    assert torch.isclose(lp_int, torch.tensor([expected_lp], dtype=env_u.float), atol=1e-5)

    # 4. Backward interior step (non-BTS) + Jacobian check
    s_next = [0, 1, 2] + [0.55] * 7
    mask_b = tbool(
        [env_u.get_mask_invalid_actions_backward(s_next, False)],
        device=env_u.device,
    )
    lp_bw = env_u.get_logprobs(policy_out, [cont_action], mask_b, [s_next], True)
    expected_bw = np.log(0.75) + 7 * np.log(1.0 / (0.55 - 0.1))
    assert torch.isclose(lp_bw, torch.tensor([expected_bw], dtype=env_u.float), atol=1e-5)

    # 5. Backward BTS sampled vs forced
    bts_action = (-1, -1, *((0.55,) * 7), 1)
    lp_bts = env_u.get_logprobs(policy_out, [bts_action], mask_b, [s_next], True)
    assert torch.isclose(lp_bts, torch.log(torch.tensor([0.25])), atol=1e-5)

    s_low = [0, 1, 2, 0.05] + [0.5] * 6
    mask_b_low = tbool(
        [env_u.get_mask_invalid_actions_backward(s_low, False)],
        device=env_u.device,
    )
    bts_low = (-1, -1, 0.05, *((0.5,) * 6), 1)
    lp_bts_forced = env_u.get_logprobs(policy_out, [bts_low], mask_b_low, [s_low], True)
    assert torch.isclose(lp_bts_forced, torch.tensor([0.0]), atol=1e-5)


@pytest.mark.skipif(
    not (MODELS_CONT_DIR / "XGBoost_5000.pkl").exists(),
    reason="Continuous surrogate pickles not generated in Phase1/models_continuous",
)
def test__continuous_plasma_reward_proxy_scores_continuously(cplasma):
    proxy = ContinuousPlasmaRewardProxy(
        models_path=str(MODELS_CONT_DIR),
        n=5000,
        reward_fn="reward_peak_new",
    )
    proxy.setup(cplasma)
    s_a = [0, 1, 2, 0.1234, 0.5, 0.25, 0.75, 0.333, 0.666, 0.111]
    s_b = [0, 1, 2, 0.8765, 0.5, 0.25, 0.75, 0.333, 0.666, 0.889]
    rewards = proxy(cplasma.states2proxy([s_a, s_b]))
    assert rewards.shape == (2,)
    assert torch.all(torch.isfinite(rewards))
    assert torch.all((rewards > 0.0) & (rewards <= 1.5))
    assert not torch.isclose(rewards[0], rewards[1])


class TestContinuousPlasmaCommon(common.BaseTestsContinuous):
    @pytest.fixture(autouse=True)
    def setup(self, cplasma):
        self.env = cplasma
        self.repeats = {
            "test__get_logprobs__backward__returns_zero_if_done": 50,
            "test__reset__state_is_source": 10,
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 5,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 5,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 50,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 50,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 4,
        }
