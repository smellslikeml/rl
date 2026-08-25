# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
import torchrl.objectives.afu as afu_objective
import torchrl.objectives.sac as sac_objective
from _objectives_common import (
    _check_td_steady as check_td_steady,
    _has_functorch as has_functorch,
    FUNCTORCH_ERR,
)
from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn

from torchrl.data import Bounded
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import AFULoss, SoftUpdate
from torchrl.objectives.value import TD0Estimator


class ConstantQValue(nn.Module):
    def __init__(self, qvalue):
        super().__init__()
        self.qvalue = nn.Parameter(torch.tensor([qvalue], dtype=torch.float32))

    def forward(self, observation, action):
        return self.qvalue.expand(*observation.shape[:-1], 1)


class ConstantValue(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.tensor([value], dtype=torch.float32))

    def forward(self, observation):
        return self.value.expand(*observation.shape[:-1], 1)


class ConstantAdvantage(nn.Module):
    def __init__(self, advantage):
        super().__init__()
        self.advantage = nn.Parameter(torch.tensor([advantage], dtype=torch.float32))

    def forward(self, observation, action):
        return self.advantage.expand(*observation.shape[:-1], 1)


@pytest.mark.skipif(
    not has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestAFU:
    def make_actor(self, observation_key="observation"):
        action_spec = Bounded(-1, 1, (2,))
        return ProbabilisticActor(
            TensorDictModule(
                nn.Sequential(nn.Linear(3, 4), NormalParamExtractor()),
                in_keys=[observation_key],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
        )

    def make_data(
        self,
        batch_size=(7,),
        observation_key="observation",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
    ):
        def next_key(key):
            return ("next", *key) if isinstance(key, tuple) else ("next", key)

        return TensorDict(
            {
                observation_key: torch.randn(*batch_size, 3),
                "action": torch.randn(*batch_size, 2).tanh(),
                next_key(observation_key): torch.randn(*batch_size, 3),
                next_key(reward_key): torch.randn(*batch_size, 1),
                next_key(done_key): torch.zeros(*batch_size, 1, dtype=torch.bool),
                next_key(terminated_key): torch.zeros(*batch_size, 1, dtype=torch.bool),
            },
            batch_size=batch_size,
        )

    def make_loss(
        self,
        *,
        variant="alpha",
        num_value_nets=2,
        rho=0.5,
        reduction="mean",
        observation_key="observation",
        skip_done_states=False,
    ):
        actor = self.make_actor(observation_key=observation_key)
        critic = ValueOperator(
            MLP(in_features=5, out_features=1, num_cells=[16, 16]),
            in_keys=[observation_key, "action"],
        )
        value = ValueOperator(
            MLP(in_features=3, out_features=1, num_cells=[16, 16]),
            in_keys=[observation_key],
        )
        advantage = TensorDictModule(
            MLP(in_features=5, out_features=1, num_cells=[16, 16]),
            in_keys=[observation_key, "action"],
            out_keys=["advantage"],
        )
        mode_network = None
        if variant == "beta":
            mode_network = TensorDictModule(
                MLP(in_features=3, out_features=2, num_cells=[16, 16]),
                in_keys=[observation_key],
                out_keys=["mode"],
            )
        loss = AFULoss(
            actor,
            critic,
            value,
            advantage,
            mode_network=mode_network,
            variant=variant,
            num_value_nets=num_value_nets,
            rho=rho,
            reduction=reduction,
            skip_done_states=skip_done_states,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss.make_value_estimator(gamma=0.9)
        SoftUpdate(loss, tau=0.05)
        return loss

    @pytest.mark.parametrize("variant", ["alpha", "beta"])
    def test_forward_shapes_and_gradient_isolation(self, variant):
        torch.manual_seed(0)
        batch_size = (2, 3)
        loss = self.make_loss(variant=variant, reduction="none")
        data = self.make_data(batch_size=batch_size)

        with check_td_steady(data):
            output = loss(data)
        expected_keys = {"loss_actor", "loss_qvalue", "loss_value", "loss_alpha"}
        if variant == "beta":
            expected_keys.add("loss_mode")
        assert set(output.keys()) == expected_keys
        assert all(value.shape == torch.Size(batch_size) for value in output.values())
        assert data.get(loss.tensor_keys.priority).shape == torch.Size(batch_size)
        assert output.isfinite().all()

        value_params = list(loss.value_network_params.values(True, True))
        advantage_params = list(loss.advantage_network_params.values(True, True))
        qvalue_params = list(loss.qvalue_network_params.values(True, True))
        actor_params = list(loss.actor_network_params.values(True, True))

        loss.zero_grad(set_to_none=True)
        actor_loss, _ = loss.actor_loss(data)
        actor_loss.mean().backward()
        assert any(parameter.grad is not None for parameter in actor_params)
        assert all(
            parameter.grad is None
            for parameter in qvalue_params + value_params + advantage_params
        )

        loss.zero_grad(set_to_none=True)
        critic_loss, _ = loss.qvalue_loss(data)
        critic_loss.mean().backward()
        assert any(parameter.grad is not None for parameter in qvalue_params)
        assert all(
            parameter.grad is None
            for parameter in actor_params + value_params + advantage_params
        )

        loss.zero_grad(set_to_none=True)
        value_loss, _ = loss.value_loss(data)
        value_loss.mean().backward()
        assert any(parameter.grad is not None for parameter in value_params)
        assert any(parameter.grad is not None for parameter in advantage_params)
        assert all(
            parameter.grad is None for parameter in actor_params + qvalue_params
        )

        if variant == "beta":
            mode_params = list(loss.mode_network_params.values(True, True))
            assert all(parameter.grad is None for parameter in mode_params)
            loss.zero_grad(set_to_none=True)
            mode_loss, _ = loss.mode_loss(data)
            mode_loss.mean().backward()
            assert any(parameter.grad is not None for parameter in mode_params)
            assert all(
                parameter.grad is None
                for parameter in actor_params + qvalue_params + value_params
            )

    @pytest.mark.parametrize("variant", ["alpha", "beta"])
    @pytest.mark.parametrize("all_done", [False, True])
    def test_terminal_nan_is_selected_out(self, variant, all_done):
        loss = self.make_loss(variant=variant, skip_done_states=True)
        data = self.make_data(batch_size=(2, 3))
        if all_done:
            data.get(("next", "done")).fill_(True)
            data.get(("next", "terminated")).fill_(True)
            data.get(("next", "observation")).fill_(float("nan"))
        else:
            data.get(("next", "done"))[0, 0] = True
            data.get(("next", "terminated"))[0, 0] = True
            data.get(("next", "observation"))[0, 0] = float("nan")

        target = loss.compute_target(data)
        assert target.isfinite().all()
        terminal_reward = data.get(("next", "reward"))[0, 0]
        torch.testing.assert_close(target[0, 0], terminal_reward.squeeze(-1))

        output = loss(data)
        total_loss = sum(
            value for key, value in output.items() if key.startswith("loss")
        )
        total_loss.backward()
        gradients = [
            parameter.grad
            for parameter in loss.parameters()
            if parameter.grad is not None
        ]
        assert gradients and all(gradient.isfinite().all() for gradient in gradients)

        if all_done:
            # The masked positions must not contribute to the target: replacing
            # the NaN next observations leaves the target unchanged (the AFU
            # target is actor-free, hence deterministic given the parameters).
            truncated = data.clone()
            truncated.get(("next", "observation")).zero_()
            torch.testing.assert_close(loss.compute_target(truncated), target)

    @pytest.mark.parametrize("deactivate_vmap", [False, True])
    def test_afu_numerical_contract(self, monkeypatch, deactivate_vmap):
        actor = self.make_actor()
        critic = ValueOperator(
            ConstantQValue(2.0), in_keys=["observation", "action"]
        )
        value = [
            ValueOperator(ConstantValue(1.0), in_keys=["observation"]),
            ValueOperator(ConstantValue(3.0), in_keys=["observation"]),
        ]
        advantage = [
            TensorDictModule(
                ConstantAdvantage(-0.5),
                in_keys=["observation", "action"],
                out_keys=["advantage"],
            ),
            TensorDictModule(
                ConstantAdvantage(0.5),
                in_keys=["observation", "action"],
                out_keys=["advantage"],
            ),
        ]
        loss = AFULoss(
            actor,
            critic,
            value,
            advantage,
            rho=0.5,
            loss_function="l2",
            reduction="none",
            scalar_output_mode="exclude",
            deactivate_vmap=deactivate_vmap,
        )
        SoftUpdate(loss, tau=0.05)
        loss.make_value_estimator(gamma=1.0)
        data = self.make_data(batch_size=(1,))
        data.get(("next", "reward")).zero_()
        data.set("steps_to_next_obs", torch.ones(1, 1))

        def sample_with_zero_log_prob(distribution):
            action = distribution.rsample()
            return action, action.new_zeros(action.shape[:-1])

        monkeypatch.setattr(
            sac_objective,
            "compute_rsample_log_prob",
            sample_with_zero_log_prob,
        )

        # The bootstrap target is r + gamma * min_i V_target_i(s') = 0 + 1.
        target = loss.compute_target(data)
        torch.testing.assert_close(target, torch.tensor([1.0]))

        # Eq. (1): (Q - target)^2 = (2 - 1)^2.
        critic_loss, critic_metadata = loss.qvalue_loss(data)
        torch.testing.assert_close(critic_loss, torch.tensor([1.0]))
        torch.testing.assert_close(critic_metadata["td_error"], torch.tensor([1.0]))

        # Eqs. (3)-(5) with rho = 0.5:
        # pair 1: V + A = 0.5 < 1 (indicator on), Upsilon = 1, x = 0,
        #         Z = (x + A)^2 = 0.25;
        # pair 2: V + A = 3.5 > 1 (indicator off), Upsilon = 3, x = 2,
        #         Z = (x + A)^2 = 6.25.
        value_loss, _ = loss.value_loss(data)
        torch.testing.assert_close(value_loss, torch.tensor([6.5]))

        # Eq. (6) with a zero log-probability and alpha = 1: -Q(s, a_s) = -2.
        actor_loss, _ = loss.actor_loss(data)
        torch.testing.assert_close(actor_loss, torch.tensor([-2.0]))

        # Gradient checks: the indicator rescales the gradient of V by
        # (1 - rho) when it fires, and leaves it untouched otherwise.
        loss.zero_grad(set_to_none=True)
        value_loss.sum().backward()
        value_gradient = loss.value_network_params.get(("module", "value")).grad
        torch.testing.assert_close(value_gradient, torch.tensor([[-0.5], [5.0]]))
        advantage_gradient = loss.advantage_network_params.get(
            ("module", "advantage")
        ).grad
        torch.testing.assert_close(advantage_gradient, torch.tensor([[-1.0], [5.0]]))

        loss.zero_grad(set_to_none=True)
        critic_loss.sum().backward()
        critic_gradient = loss.qvalue_network_params.get(("module", "qvalue")).grad
        torch.testing.assert_close(critic_gradient, torch.tensor([[2.0]]))

    def test_projected_action_gradient(self):
        project = afu_objective._ProjectedActionGradient.apply
        # Forward is the identity.
        action = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        direction = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        condition = torch.tensor([True, True])
        out = project(action, direction, condition)
        torch.testing.assert_close(out, action.detach())

        # The gradient reaching this Function in the actor loss is
        # dL/da = -c * grad_Q (c > 0), since the loss is `alpha*log_prob - Q`.
        # AFU-beta projects when the *Q* gradient points away from the mode,
        # grad_Q . d < 0, i.e. when the incoming gradient . d > 0.
        # Sample 0: incoming (1, -3), d (1, 0) -> dot = +1 > 0 -> project, remove
        #   the component along d: (1, -3) - 1*(1, 0) = (0, -3).
        # Sample 1: incoming (-2, -3), d (1, 1) -> dot = -5 < 0 -> unchanged.
        out.backward(torch.tensor([[1.0, -3.0], [-2.0, -3.0]]))
        torch.testing.assert_close(
            action.grad, torch.tensor([[0.0, -3.0], [-2.0, -3.0]])
        )

        # When the condition (Q < min_i V) does not hold, the gradient is left
        # unchanged even when the sign would otherwise trigger projection.
        action2 = torch.tensor([1.0, 2.0], requires_grad=True)
        out2 = project(action2, direction[0], torch.tensor(False))
        out2.backward(torch.tensor([1.0, -3.0]))
        torch.testing.assert_close(action2.grad, torch.tensor([1.0, -3.0]))

        # A zero direction never divides by zero and leaves the gradient
        # unchanged (dot == 0 also never triggers projection).
        action3 = torch.tensor([1.0, 2.0], requires_grad=True)
        out3 = project(action3, torch.zeros(2), torch.tensor(True))
        out3.backward(torch.tensor([1.0, -3.0]))
        torch.testing.assert_close(action3.grad, torch.tensor([1.0, -3.0]))

    def test_beta_forward_backward(self):
        # The beta variant adds the mode-predictor loss and routes the actor
        # gradient through the projection operator; both paths must run and
        # stay finite/differentiable end to end.
        torch.manual_seed(0)
        actor = self.make_actor()
        critic = ValueOperator(
            MLP(in_features=5, out_features=1, num_cells=[16]),
            in_keys=["observation", "action"],
        )
        value = ValueOperator(ConstantValue(0.0), in_keys=["observation"])
        advantage = TensorDictModule(
            ConstantAdvantage(-1.0),
            in_keys=["observation", "action"],
            out_keys=["advantage"],
        )
        mode = TensorDictModule(
            MLP(in_features=3, out_features=2, num_cells=[16]),
            in_keys=["observation"],
            out_keys=["mode"],
        )
        loss_beta = AFULoss(
            actor,
            critic,
            value,
            advantage,
            mode_network=mode,
            variant="beta",
            num_value_nets=1,
        )
        SoftUpdate(loss_beta, tau=0.05)
        loss_beta.make_value_estimator(gamma=0.9)
        data = self.make_data(batch_size=(4,))
        output = loss_beta(data)
        assert set(output.keys()) == {
            "loss_actor",
            "loss_qvalue",
            "loss_value",
            "loss_alpha",
            "loss_mode",
            "alpha",
            "entropy",
        }
        assert output.isfinite().all()
        total_loss = sum(
            value for key, value in output.items() if key.startswith("loss")
        )
        total_loss.backward()
        mode_params = list(loss_beta.mode_network_params.values(True, True))
        assert any(parameter.grad is not None for parameter in mode_params)

    def test_nested_keys_and_value_estimator_contract(self):
        observation_key = ("agent", "observation")
        reward_key = ("metrics", "reward")
        done_key = ("flags", "done")
        terminated_key = ("flags", "terminated")
        priority_key = ("replay", "error")
        loss = self.make_loss(observation_key=observation_key)
        loss.set_keys(
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
            priority=priority_key,
        )
        data = self.make_data(
            batch_size=(5,),
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )
        assert loss(data).isfinite().all()
        assert data.get(priority_key).shape == torch.Size([5])
        assert isinstance(loss.value_estimator, TD0Estimator)
        assert loss.value_estimator.tensor_keys.reward == reward_key

    def test_target_update_only_tracks_value_network(self):
        loss = self.make_loss()
        updater = SoftUpdate(loss, tau=0.5)
        # AFU has target parameters for the value networks only: the critic,
        # the advantage networks and the actor have none (paper, Appendix A).
        assert updater._target_names == ["target_value_network_params"]

        before = {
            key: value.detach().clone()
            for key, value in loss.target_value_network_params.items(True, True)
        }
        for parameter in loss.value_network_params.values(True, True):
            parameter.data.add_(1.0)
        updater.step()
        # tau = 0.5: the target moves halfway towards the perturbed source.
        for key, value in loss.target_value_network_params.items(True, True):
            torch.testing.assert_close(value, before[key] + 0.5)

    def test_rejects_invalid_variant(self):
        with pytest.raises(ValueError, match="variant"):
            self.make_loss(variant="gamma")

    def test_rejects_invalid_rho(self):
        with pytest.raises(ValueError, match="rho"):
            self.make_loss(rho=0.0)
        with pytest.raises(ValueError, match="rho"):
            self.make_loss(rho=1.0)

    def test_beta_requires_mode_network(self):
        actor = self.make_actor()
        critic = ValueOperator(
            MLP(in_features=5, out_features=1, num_cells=[16]),
            in_keys=["observation", "action"],
        )
        value = ValueOperator(
            MLP(in_features=3, out_features=1, num_cells=[16]),
            in_keys=["observation"],
        )
        advantage = TensorDictModule(
            MLP(in_features=5, out_features=1, num_cells=[16]),
            in_keys=["observation", "action"],
            out_keys=["advantage"],
        )
        with pytest.raises(ValueError, match="mode_network"):
            AFULoss(actor, critic, value, advantage, variant="beta")

    def test_alpha_rejects_mode_network(self):
        actor = self.make_actor()
        critic = ValueOperator(
            MLP(in_features=5, out_features=1, num_cells=[16]),
            in_keys=["observation", "action"],
        )
        value = ValueOperator(
            MLP(in_features=3, out_features=1, num_cells=[16]),
            in_keys=["observation"],
        )
        advantage = TensorDictModule(
            MLP(in_features=5, out_features=1, num_cells=[16]),
            in_keys=["observation", "action"],
            out_keys=["advantage"],
        )
        mode = TensorDictModule(
            MLP(in_features=3, out_features=2, num_cells=[16]),
            in_keys=["observation"],
            out_keys=["mode"],
        )
        with pytest.raises(ValueError, match="mode_network"):
            AFULoss(actor, critic, value, advantage, mode_network=mode)

    def test_mode_loss_requires_beta(self):
        loss = self.make_loss(variant="alpha")
        with pytest.raises(RuntimeError, match="beta"):
            loss.mode_loss(self.make_data())


if __name__ == "__main__":
    pytest.main([__file__])
