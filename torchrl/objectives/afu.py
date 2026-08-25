# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import (
    dispatch,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from tensordict.utils import NestedKey
from torch import Tensor

from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.sac import compute_rsample_log_prob, SACLoss
from torchrl.objectives.utils import _cache_values, _vmap_func, distance_loss


class _ProjectedActionGradient(torch.autograd.Function):
    """Identity map with a projected backward pass.

    Implements the modified-gradient operator of AFU-beta (Perrin-Gilbert,
    2024, Section 4): the gradient :math:`v = \\nabla_{a_s} Q_\\psi(s, a_s)`
    that reaches the sampled action through the critic is projected onto the
    orthogonal complement of :math:`d = \\mu_\\zeta(s) - a_s` whenever
    :math:`v \\cdot d < 0` and :math:`Q_\\psi(s, a_s) < \\min_i V_{\\varphi_i}(s)`
    (the ``condition`` input). This removes the gradient components that point
    away from the mode predictor's estimate of the argmax, making actor
    updates less likely to be trapped in deceptive local optima. The forward
    pass is the identity, so the loss value is unchanged.
    """

    @staticmethod
    def forward(ctx, action, direction, condition):
        ctx.save_for_backward(direction, condition)
        return action

    @staticmethod
    def backward(ctx, grad_output):
        direction, condition = ctx.saved_tensors
        # The gradient reaching this Function is dL/da_s = -c * grad_{a_s} Q (c > 0),
        # because the actor loss is ``alpha * log_prob - Q`` (Q enters with a minus
        # sign). AFU-beta projects when the *Q* gradient points away from the mode,
        # i.e. ``grad_Q . direction < 0``; since ``grad_output = -c * grad_Q`` that is
        # equivalent to ``grad_output . direction > 0``. (Testing ``dot < 0`` here
        # would invert the operator and strip helpful components instead.)
        dot = (grad_output * direction).sum(-1, keepdim=True)
        denom = direction.square().sum(-1, keepdim=True)
        denom = denom.clamp_min(torch.finfo(direction.dtype).eps)
        project = condition.unsqueeze(-1) & (dot > 0)
        grad_output = grad_output - torch.where(
            project, dot / denom * direction, torch.zeros_like(grad_output)
        )
        return grad_output, None, None


class AFULoss(SACLoss):
    r"""Actor-Free critic Updates (AFU) loss.

    Presented in `Perrin-Gilbert (2024), Actor-Free critic Updates in
    off-policy RL for continuous control <https://arxiv.org/abs/2404.16159>`_.

    AFU departs from the actor-critic template: the critic updates never
    involve the actor. A single Q-critic :math:`Q_\psi` is trained against a
    bootstrap target built exclusively from target value networks
    (Eq. (1) of the paper):

    .. math::

        L_Q(\psi) = \underset{(s,a,r,s') \in B}{\mathrm{Mean}}
        \left[ \left( Q_\psi(s,a) - r - \gamma \min_{i \in \{1,2\}}
        V_{\varphi_i^{\text{target}}}(s') \right)^2 \right]

    The max-Q problem (estimating :math:`\max_a Q(s,a)` without an actor to
    maximize Q) is solved through a value/advantage decomposition
    :math:`Q(s,a) = V(s) + A(s,a)` trained with *conditional gradient
    scaling*. Writing the mixed gradient/no-grad value term (Eq. (3))

    .. math::

        \Upsilon_i^a(s) = (1 - \varrho I_i^{s,a}) V_{\varphi_i}(s)
        + \varrho I_i^{s,a} V_{\varphi_i^{\text{no\_grad}}}(s),

    with :math:`I_i^{s,a} = 1` when the decomposition underestimates the
    bootstrap target, the value/advantage loss is (Eq. (4)-(5))

    .. math::

        L_{V,A}(\varphi_i, \xi_i) = \underset{(s,a,r,s') \in B}{\mathrm{Mean}}
        \left[ Z\left( \Upsilon_i^a(s) - r - \gamma \min_{i \in \{1,2\}}
        V_{\varphi_i^{\text{target}}}(s'),\; A_{\xi_i}(s,a) \right) \right],

    where :math:`Z(x, y) = (x + y)^2` if :math:`x \geq 0` and
    :math:`x^2 + y^2` otherwise, which softly constrains the advantage to be
    non-positive. Note that, per Eq. (5), the bootstrap target replaces
    :math:`Q_\psi(s,a)` everywhere in the toy loss of Eq. (4), including
    inside the indicator :math:`I_i^{s,a}` within :math:`\Upsilon_i^a`.

    Two variants are provided behind the ``variant`` flag:

    - ``"alpha"`` (AFU-:math:`\alpha`): the actor is the SAC stochastic actor,
      trained with the usual maximum-entropy losses (Eqs. (6)-(7)). Since the
      critic updates are actor-free, the actor is only used for data
      collection and can be swapped freely.
    - ``"beta"`` (AFU-:math:`\beta`): adds a deterministic mode predictor
      :math:`\mu_\zeta` trained by regression on the buffer and actor-sampled
      actions whose Q-value exceeds :math:`\min_i V_{\varphi_i}(s)`
      (Eq. (8)), and projects the chain-rule factor
      :math:`\nabla_{a_s} Q_\psi(s, a_s)` of the actor gradient onto the
      orthogonal complement of :math:`\mu_\zeta(s) - a_s` whenever it points
      away from the predicted mode and
      :math:`Q_\psi(s, a_s) < \min_i V_{\varphi_i}(s)` (the modified-gradient
      operator of Section 4), making actor updates robust to deceptive
      gradients.

    Only the value networks have target networks (updated with
    :class:`~torchrl.objectives.SoftUpdate`, e.g. ``SoftUpdate(loss,
    tau=0.01)`` following Appendix A of the paper); the Q-critic, the
    advantage networks and the mode predictor have none.

    Args:
        actor_network (ProbabilisticTensorDictSequential): stochastic actor
            used for data collection. With ``variant="alpha"`` it is trained
            with the SAC actor loss; with ``variant="beta"`` the actor
            gradient is modified as described above.
        qvalue_network (TensorDictModule or list of TensorDictModule):
            the single Q-critic :math:`Q_\psi`. This module typically outputs
            a ``"state_action_value"`` entry. AFU uses one critic only
            (``num_qvalue_nets`` is fixed to ``1``).
        value_network (TensorDictModule or list of TensorDictModule): state
            value model(s) :math:`V_{\varphi_i}`, typically outputting a
            ``"state_value"`` entry. One module is replicated
            ``num_value_nets`` times; a list supplies each network of the
            ensemble, as in :class:`~torchrl.objectives.SACLoss`.
        advantage_network (TensorDictModule or list of TensorDictModule):
            advantage model(s) :math:`A_{\xi_i}`, typically outputting an
            ``"advantage"`` entry. Replicated or listed like
            ``value_network``; the i-th advantage network is paired with the
            i-th value network.

    Keyword Args:
        mode_network (TensorDictModule, optional): deterministic mode
            predictor :math:`\mu_\zeta`, mapping observations to actions and
            typically outputting a ``"mode"`` entry. Required when
            ``variant="beta"``; must be ``None`` otherwise.
        variant (str, optional): ``"alpha"`` or ``"beta"``. Defaults to
            ``"alpha"``.
        num_value_nets (int, optional): size of the (value, advantage)
            ensemble. The paper uses ``2``. Defaults to ``2``.
        rho (float, optional): conditional gradient scaling coefficient
            :math:`\varrho \in (0, 1)`. The paper reports best results with
            :math:`\varrho \in \{0.2, 0.3\}` and evaluates AFU-beta primarily
            with :math:`\varrho = 0.3`. Defaults to ``0.3``.
        loss_function (str, optional): distance used for the Q-critic loss.
            The paper's Eq. (1) uses the squared error (``"l2"``);
            ``"smooth_l1"`` and ``"l1"`` are also accepted. Defaults to
            ``"l2"``.
        alpha_init (float, optional): initial entropy temperature. Defaults to
            ``1.0``.
        min_alpha (float, optional): temperature floor. Defaults to ``None``.
        max_alpha (float, optional): temperature ceiling. Defaults to ``None``.
        action_spec (TensorSpec, optional): action domain for automatic target
            entropy. Defaults to the actor's spec.
        fixed_alpha (bool, optional): disable temperature learning. Defaults to
            ``False``.
        target_entropy (float or ``"auto"``, optional): entropy target.
            Defaults to ``"auto"``.
        delay_value (bool, optional): maintain delayed target value networks.
            Defaults to ``True``.
        separate_losses (bool, optional): exclude shared actor parameters from
            critic training. Defaults to ``False``.
        reduction (str, optional): ``"none"``, ``"mean"``, or ``"sum"``.
            Defaults to ``"mean"``.
        deactivate_vmap (bool, optional): loop over the ensemble instead of
            using ``vmap``. Defaults to ``False``.
        skip_done_states (bool, optional): skip terminal next-state evaluation
            when bootstrapping. Defaults to ``False``.
        use_prioritized_weights (bool or ``"auto"``, optional): use replay
            weights when present. Defaults to ``"auto"``.
        scalar_output_mode (str, optional): scalar handling when
            ``reduction="none"``. See :class:`~torchrl.objectives.SACLoss`.

    .. note::
        The AFU-beta actor loss relies on a custom autograd function to
        project the action-gradient; it is not compatible with
        ``torch.compile`` fullgraph capture. AFU-alpha has no such
        limitation. Composite (TensorDict-valued) actions are not supported
        by the AFU-beta actor loss.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import NormalParamExtractor, TensorDictModule
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.distributions import TanhNormal
        >>> from torchrl.objectives import AFULoss, SoftUpdate
        >>> n_obs, n_act = 3, 2
        >>> action_spec = Bounded(-1, 1, (n_act,))
        >>> actor = ProbabilisticActor(
        ...     TensorDictModule(
        ...         nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor()),
        ...         in_keys=["observation"],
        ...         out_keys=["loc", "scale"],
        ...     ),
        ...     in_keys=["loc", "scale"],
        ...     spec=action_spec,
        ...     distribution_class=TanhNormal,
        ... )
        >>> critic = ValueOperator(
        ...     MLP(in_features=n_obs + n_act, out_features=1, num_cells=[16]),
        ...     in_keys=["observation", "action"],
        ... )
        >>> value = ValueOperator(
        ...     MLP(in_features=n_obs, out_features=1, num_cells=[16]),
        ...     in_keys=["observation"],
        ... )
        >>> advantage = TensorDictModule(
        ...     MLP(in_features=n_obs + n_act, out_features=1, num_cells=[16]),
        ...     in_keys=["observation", "action"],
        ...     out_keys=["advantage"],
        ... )
        >>> loss = AFULoss(actor, critic, value, advantage)
        >>> loss.make_value_estimator(gamma=0.99)
        >>> target_updater = SoftUpdate(loss, tau=0.01)
        >>> batch = TensorDict(
        ...     {
        ...         "observation": torch.randn(4, n_obs),
        ...         "action": action_spec.rand((4,)),
        ...         "next": {
        ...             "observation": torch.randn(4, n_obs),
        ...             "reward": torch.randn(4, 1),
        ...             "done": torch.zeros(4, 1, dtype=torch.bool),
        ...             "terminated": torch.zeros(4, 1, dtype=torch.bool),
        ...         },
        ...     },
        ...     batch_size=[4],
        ... )
        >>> output = loss(batch)
        >>> output.get("loss_value").shape
        torch.Size([])
    """

    @dataclass
    class _AcceptedKeys(SACLoss._AcceptedKeys):
        """Maintains default values for all configurable tensordict keys.

        In addition to the keys inherited from
        :class:`~torchrl.objectives.SACLoss`:

        Attributes:
            advantage (NestedKey): the tensordict key where the advantage
                networks' output is expected. Defaults to ``"advantage"``.
            mode (NestedKey): the tensordict key where the mode predictor's
                output is expected (AFU-beta only). Defaults to ``"mode"``.
        """

        advantage: NestedKey = "advantage"
        mode: NestedKey = "mode"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys

    advantage_network: TensorDictModule
    mode_network: TensorDictModule | None
    advantage_network_params: TensorDictParams
    mode_network_params: TensorDictParams | None
    target_advantage_network_params: TensorDictParams
    target_mode_network_params: TensorDictParams | None

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential,
        qvalue_network: TensorDictModule | list[TensorDictModule],
        value_network: TensorDictModule | list[TensorDictModule],
        advantage_network: TensorDictModule | list[TensorDictModule],
        *,
        mode_network: TensorDictModule | None = None,
        variant: Literal["alpha", "beta"] = "alpha",
        num_value_nets: int = 2,
        rho: float = 0.3,
        loss_function: str = "l2",
        alpha_init: float = 1.0,
        min_alpha: float | None = None,
        max_alpha: float | None = None,
        action_spec: TensorSpec | None = None,
        fixed_alpha: bool = False,
        target_entropy: Literal["auto"] | float = "auto",
        delay_value: bool = True,
        separate_losses: bool = False,
        reduction: Literal["none", "mean", "sum"] | None = None,
        deactivate_vmap: bool = False,
        skip_done_states: bool = False,
        use_prioritized_weights: Literal["auto"] | bool = "auto",
        scalar_output_mode: Literal["exclude", "non_tensor"] | None = None,
    ) -> None:
        if variant not in ("alpha", "beta"):
            raise ValueError(f"variant must be 'alpha' or 'beta', got {variant!r}.")
        if not 0 < rho < 1:
            raise ValueError(f"rho must be in the open interval (0, 1), got {rho}.")
        if num_value_nets < 1:
            raise ValueError("num_value_nets must be greater than zero.")
        if variant == "beta" and mode_network is None:
            raise ValueError("variant='beta' requires a mode_network.")
        if variant == "alpha" and mode_network is not None:
            raise ValueError("mode_network is only used with variant='beta'.")
        self.variant = variant
        self.num_value_nets = num_value_nets
        self.rho = rho
        super().__init__(
            actor_network=actor_network,
            qvalue_network=qvalue_network,
            value_network=None,
            num_qvalue_nets=1,
            loss_function=loss_function,
            alpha_init=alpha_init,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            action_spec=action_spec,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            delay_actor=False,
            # AFU's critic has no target network: the bootstrap target of
            # Eqs. (1) and (5) is built from the target value networks only.
            delay_qvalue=False,
            delay_value=False,
            separate_losses=separate_losses,
            reduction=reduction,
            skip_done_states=skip_done_states,
            deactivate_vmap=deactivate_vmap,
            use_prioritized_weights=use_prioritized_weights,
            scalar_output_mode=scalar_output_mode,
        )
        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of value/advantage/mode must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        self.delay_value = delay_value
        self.convert_to_functional(
            value_network,
            "value_network",
            num_value_nets,
            create_target_params=delay_value,
            compare_against=policy_params,
        )
        if separate_losses:
            advantage_compare_against = policy_params + list(
                value_network.parameters()
            )
        else:
            advantage_compare_against = None
        self.convert_to_functional(
            advantage_network,
            "advantage_network",
            num_value_nets,
            create_target_params=False,
            compare_against=advantage_compare_against,
        )
        if variant == "beta":
            self.convert_to_functional(
                mode_network,
                "mode_network",
                create_target_params=False,
                compare_against=policy_params,
            )
        self._vmap_value_networkN0 = _vmap_func(
            self.value_network,
            (None, 0),
            randomness=self.vmap_randomness,
            pseudo_vmap=self.deactivate_vmap,
        )
        self._vmap_advantage_networkN0 = _vmap_func(
            self.advantage_network,
            (None, 0),
            randomness=self.vmap_randomness,
            pseudo_vmap=self.deactivate_vmap,
        )
        self._out_keys = [
            "loss_actor",
            "loss_qvalue",
            "loss_value",
            "loss_alpha",
            "alpha",
            "entropy",
        ]
        if self.variant == "beta":
            self._out_keys.append("loss_mode")
        # The super().__init__() call to set_keys() ran before the value,
        # advantage and mode networks were registered: let in_keys be lazily
        # recomputed with the full network set.
        self._in_keys = None

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *self.qvalue_network.in_keys,
        ]
        value_network = getattr(self, "value_network", None)
        if value_network is not None:
            keys.extend(value_network.in_keys)
            keys.extend(("next", key) for key in value_network.in_keys)
        advantage_network = getattr(self, "advantage_network", None)
        if advantage_network is not None:
            keys.extend(advantage_network.in_keys)
        mode_network = getattr(self, "mode_network", None)
        if mode_network is not None:
            keys.extend(mode_network.in_keys)
        self._in_keys = list(set(keys))

    @property
    @_cache_values
    def _cached_detached_value_params(self):
        return self.value_network_params.detach()

    def _compute_target_v2(self, tensordict) -> Tensor:
        # Replaces SAC's actor-based bootstrap target with AFU's actor-free
        # one, so the inherited SAC machinery never evaluates the actor (or
        # the missing target critics) on the next state.
        return self.compute_target(tensordict)

    def compute_target(self, tensordict: TensorDictBase) -> Tensor:
        r"""Actor-free bootstrap target shared by the critic and value/advantage losses.

        Computes :math:`y = r + \gamma \min_i V_{\varphi_i^{\text{target}}}(s')`
        (the target of Eqs. (1) and (5) of the paper) through the configured
        value estimator. No actor is involved at any point.
        """
        steps_key = self.value_estimator.tensor_keys.steps_to_next_obs
        target_tensordict = tensordict.select("next", steps_key, strict=False).clone()
        with torch.no_grad():
            next_tensordict = target_tensordict.get("next")
            selection = None
            selected_tensordict = next_tensordict
            if self.skip_done_states:
                terminated = next_tensordict.get(self.tensor_keys.terminated)
                if terminated.any():
                    selection = ~terminated.squeeze(-1)
                    selected_tensordict = next_tensordict[selection]
            if selected_tensordict.batch_size.numel():
                value_input = selected_tensordict.select(
                    *self.value_network.in_keys, strict=False
                )
                value_output = self._vmap_value_networkN0(
                    value_input, self.target_value_network_params
                )
                next_value = value_output.get(self.tensor_keys.value)
                next_value = next_value.squeeze(-1).min(0)[0]
            else:
                next_value = next_tensordict.get(
                    self.tensor_keys.reward
                ).new_zeros(0)
            if selection is not None:
                full_value = next_tensordict.get(self.tensor_keys.reward).new_zeros(
                    selection.shape
                )
                if next_value.numel():
                    full_value = full_value.masked_scatter(selection, next_value)
                next_value = full_value
            target_tensordict.set(
                ("next", self.value_estimator.tensor_keys.value),
                next_value.unsqueeze(-1),
            )
            return self.value_estimator.value_estimate(target_tensordict).squeeze(-1)

    def qvalue_loss(
        self, tensordict: TensorDictBase, target_value: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Q-critic loss (Eq. (1) of the paper)."""
        weights = self._maybe_get_priority_weight(tensordict)
        if target_value is None:
            target_value = self.compute_target(tensordict)
        tensordict_expand = self._vmap_qnetworkN0(
            tensordict.select(*self.qvalue_network.in_keys, strict=False),
            self.qvalue_network_params,
        )
        pred_val = tensordict_expand.get(self.tensor_keys.state_action_value).squeeze(
            -1
        )
        td_error = abs(pred_val - target_value)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        ).sum(0)
        loss_qval = self._reduce_loss(loss_qval, tensordict=tensordict, weights=weights)
        metadata = {"td_error": td_error.detach().max(0)[0]}
        return loss_qval, metadata

    def value_loss(
        self, tensordict: TensorDictBase, target_value: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Value/advantage loss with conditional gradient scaling (Eqs. (3)-(5))."""
        weights = self._maybe_get_priority_weight(tensordict)
        if target_value is None:
            target_value = self.compute_target(tensordict)
        in_keys = list(
            dict.fromkeys(
                [*self.value_network.in_keys, *self.advantage_network.in_keys]
            )
        )
        tensordict_va = tensordict.select(*in_keys, strict=False)
        value = (
            self._vmap_value_networkN0(tensordict_va, self.value_network_params)
            .get(self.tensor_keys.value)
            .squeeze(-1)
        )
        advantage = (
            self._vmap_advantage_networkN0(tensordict_va, self.advantage_network_params)
            .get(self.tensor_keys.advantage)
            .squeeze(-1)
        )
        # Eq. (3) with the Eq. (5) substitution: the decomposition is compared
        # against the bootstrap target (which replaces Q_psi(s, a) everywhere).
        # Comparisons carry no gradient, so the indicator is constant by
        # construction.
        indicator = (value + advantage < target_value.unsqueeze(0)).to(value.dtype)
        # Mixed grad/no-grad value: when the indicator fires, only a fraction
        # (1 - rho) of the gradient reaches V, putting a downward pressure on
        # it (conditional gradient scaling).
        upsilon = (1 - self.rho * indicator) * value + (
            self.rho * indicator
        ) * value.detach()
        error = upsilon - target_value.unsqueeze(0)
        # Z(x, y) = (x + y)^2 if x >= 0, else x^2 + y^2: regresses V + A onto
        # the target while softly constraining A(s, a) <= 0 (Eq. (4)).
        loss_value = torch.where(
            error >= 0,
            (error + advantage).square(),
            error.square() + advantage.square(),
        )
        # Each (value, advantage) pair is updated with its own Mean-reduced
        # loss in the paper; summing over the ensemble yields the same
        # per-network gradients.
        loss_value = loss_value.sum(0)
        loss_value = self._reduce_loss(
            loss_value, tensordict=tensordict, weights=weights
        )
        return loss_value, {}

    def actor_loss(
        self, tensordict: TensorDictBase
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Actor loss (Eq. (6), with the modified gradient of Section 4 for AFU-beta).

        With ``variant="alpha"`` this is exactly the SAC actor loss.
        """
        if self.variant == "alpha":
            # AFU-alpha uses the SAC actor loss verbatim; with a single
            # critic the ensemble minimum of SACLoss.actor_loss is the
            # identity.
            return super().actor_loss(tensordict)
        weights = self._maybe_get_priority_weight(tensordict)
        with (
            set_exploration_type(ExplorationType.RANDOM),
            self.actor_network_params.to_module(
                self.actor_network, preserve_module_state=False
            ),
        ):
            distribution = self.actor_network.get_dist(tensordict)
            action, log_prob = compute_rsample_log_prob(distribution)
        if not isinstance(action, Tensor):
            raise NotImplementedError(
                "AFULoss with variant='beta' does not support composite "
                "(TensorDict-valued) actions."
            )
        with torch.no_grad():
            mode_input = tensordict.select(*self.mode_network.in_keys, strict=False)
            with self.mode_network_params.to_module(
                self.mode_network, preserve_module_state=False
            ):
                self.mode_network(mode_input)
            mode = mode_input.get(self.tensor_keys.mode)
            min_value = (
                self._vmap_value_networkN0(
                    tensordict.select(*self.value_network.in_keys, strict=False),
                    self._cached_detached_value_params,
                )
                .get(self.tensor_keys.value)
                .squeeze(-1)
                .min(0)[0]
            )
            mask_input = tensordict.select(*self.qvalue_network.in_keys, strict=False)
            mask_input.set(self.tensor_keys.action, action)
            mask_qvalue = (
                self._vmap_qnetworkN0(mask_input, self._cached_detached_qvalue_params)
                .get(self.tensor_keys.state_action_value)
                .min(0)[0]
                .squeeze(-1)
            )
            # Projection is only applied where the sampled action is not
            # already better than the estimated value of the state.
            condition = mask_qvalue < min_value
            direction = mode - action
        # The forward pass is the identity; in the backward pass the
        # chain-rule factor d Q_psi(s, a_s) / d a_s is projected onto the
        # orthogonal complement of (mu(s) - a_s) when it points away from the
        # predicted mode (modified actor gradient of AFU-beta).
        projected_action = _ProjectedActionGradient.apply(action, direction, condition)
        critic_input = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        critic_input.set(self.tensor_keys.action, projected_action)
        qvalue = (
            self._vmap_qnetworkN0(critic_input, self._cached_detached_qvalue_params)
            .get(self.tensor_keys.state_action_value)
            .min(0)[0]
            .squeeze(-1)
        )
        if log_prob.shape != qvalue.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {qvalue.shape}"
            )
        loss_actor = self._alpha * log_prob - qvalue
        loss_actor = self._reduce_loss(
            loss_actor, tensordict=tensordict, weights=weights
        )
        return loss_actor, {"log_prob": log_prob.detach()}

    def mode_loss(
        self, tensordict: TensorDictBase
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Mode predictor loss of AFU-beta (Eq. (8) of the paper)."""
        if self.variant != "beta":
            raise RuntimeError(
                "AFULoss.mode_loss is only available with variant='beta'."
            )
        weights = self._maybe_get_priority_weight(tensordict)
        with torch.no_grad():
            with (
                set_exploration_type(ExplorationType.RANDOM),
                self.actor_network_params.to_module(
                    self.actor_network, preserve_module_state=False
                ),
            ):
                distribution = self.actor_network.get_dist(tensordict)
                sampled_action = distribution.sample()
            min_value = (
                self._vmap_value_networkN0(
                    tensordict.select(*self.value_network.in_keys, strict=False),
                    self._cached_detached_value_params,
                )
                .get(self.tensor_keys.value)
                .squeeze(-1)
                .min(0)[0]
            )
            # The regression set M(B) gathers the buffer actions and the
            # actor-resampled actions whose Q-value exceeds min_i V_i(s).
            buffer_input = tensordict.select(
                *self.qvalue_network.in_keys, strict=False
            )
            buffer_qvalue = (
                self._vmap_qnetworkN0(buffer_input, self._cached_detached_qvalue_params)
                .get(self.tensor_keys.state_action_value)
                .min(0)[0]
                .squeeze(-1)
            )
            sampled_input = tensordict.select(
                *self.qvalue_network.in_keys, strict=False
            )
            sampled_input.set(self.tensor_keys.action, sampled_action)
            sampled_qvalue = (
                self._vmap_qnetworkN0(
                    sampled_input, self._cached_detached_qvalue_params
                )
                .get(self.tensor_keys.state_action_value)
                .min(0)[0]
                .squeeze(-1)
            )
            buffer_action = tensordict.get(self.tensor_keys.action)
            buffer_selected = (buffer_qvalue > min_value).to(min_value.dtype)
            sampled_selected = (sampled_qvalue > min_value).to(min_value.dtype)
        mode_input = tensordict.select(*self.mode_network.in_keys, strict=False)
        with self.mode_network_params.to_module(
            self.mode_network, preserve_module_state=False
        ):
            self.mode_network(mode_input)
        mode = mode_input.get(self.tensor_keys.mode)
        buffer_error = (mode - buffer_action).square().mean(-1)
        sampled_error = (mode - sampled_action).square().mean(-1)
        selected_count = buffer_selected + sampled_selected
        # Eq. (8): a single mean over ALL selected (s, a) pairs in the batch, each
        # weighted 1/|M(B)|. Sum the selected squared errors per sample, then
        # rescale by n_samples / |M(B)| so the per-sample mean reduction below
        # (which divides by n_samples) yields the /|M(B)| of Eq. (8). The scaling
        # keeps the per-sample shape the loss contract expects. The previous
        # /selected_count.clamp_min(1) per-sample average under-weighted pairs from
        # multi-selected samples and diluted the loss with unselected samples.
        selected_error = buffer_error * buffer_selected + sampled_error * sampled_selected
        total_pairs = selected_count.sum().clamp_min(1)
        loss_mode = selected_error * (selected_error.numel() / total_pairs)
        loss_mode = self._reduce_loss(loss_mode, tensordict=tensordict, weights=weights)
        return loss_mode, {"mode_selected": selected_count.detach()}

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        target_value = self.compute_target(tensordict)
        loss_qvalue, value_metadata = self.qvalue_loss(tensordict, target_value)
        loss_value, _ = self.value_loss(tensordict, target_value)
        loss_actor, metadata_actor = self.actor_loss(tensordict)
        loss_alpha = self._alpha_loss(log_prob=metadata_actor["log_prob"])
        weights = self._maybe_get_priority_weight(tensordict)
        loss_alpha = self._reduce_loss(
            loss_alpha, tensordict=tensordict, weights=weights
        )
        if self.variant == "beta":
            loss_mode, _ = self.mode_loss(tensordict)
        else:
            loss_mode = None
        tensordict.set(self.tensor_keys.priority, value_metadata["td_error"])
        if (
            (loss_actor.shape != loss_qvalue.shape)
            or (loss_actor.shape != loss_value.shape)
            or (loss_mode is not None and loss_actor.shape != loss_mode.shape)
        ):
            mode_shape = loss_mode.shape if loss_mode is not None else None
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, {loss_qvalue.shape}, "
                f"{loss_value.shape} and {mode_shape}"
            )
        entropy = -metadata_actor["log_prob"]
        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue,
            "loss_value": loss_value,
            "loss_alpha": loss_alpha,
        }
        if loss_mode is not None:
            out["loss_mode"] = loss_mode

        # Handle batch_size and scalar values (alpha, entropy) based on reduction mode
        if self.reduction == "none":
            batch_size = tensordict.batch_size
            td_out = TensorDict(out, batch_size=batch_size)
            if self.scalar_output_mode == "non_tensor":
                td_out.set_non_tensor("alpha", self._alpha)
                td_out.set_non_tensor("entropy", entropy.detach().mean())
            # else "exclude": scalars are not included (warning was raised in __init__)
        else:
            batch_size = []
            out["alpha"] = self._alpha
            out["entropy"] = entropy.detach().mean()
            td_out = TensorDict(out, batch_size=batch_size)
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
            "qvalue_network_params",
            "value_network_params",
            "advantage_network_params",
            "mode_network_params",
            "target_value_network_params",
        )
        return td_out
