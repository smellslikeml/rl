# Validation — GSPOLoss

## Validated here (CPU, synthetic data)

- `test/llm/test_llm_objectives.py::TestLosses::test_gspo_sequence_level_importance_weight`
  — two multi-token sequences with different valid response lengths, NaN
  sampling log-probs at padded positions and garbage current log-probs there.
  Asserts the loss matches the sequence-level formula from trl's
  `importance_sampling_level="sequence"` branch (`trl/losses/grpo_loss.py`),
  computed inline as a parity oracle, with the GSPO paper Sec. 5.1 clip ranges
  (3e-4 / 4e-4) active; asserts `GSPOLossOutput` type, scalar finite
  `loss_objective` / `clip_fraction` / `kl_approx` / `ESS`, and a clip fraction
  of 1.0 for sequence ratios outside the tight bounds.
- `test_gspo_gradients_flow_through_valid_tokens` — finite, non-zero gradients
  on valid tokens, exactly-zero gradient on masked-out positions.
- `GSPOLoss` docstring example runs as a doctest (`pytest --doctest-modules
  torchrl/objectives/llm/grpo.py -k GSPOLoss`).
- Exports from `torchrl.objectives.llm` (`GSPOLoss`, `GSPOLossOutput`),
  subclass relationships (`GSPOLoss` -> `GRPOLoss`, `GSPOLossOutput` ->
  `LLMLossOutput`), default clip buffers (3e-4 / 4e-4, overridable via
  `clip_epsilon`) and inherited `_AcceptedKeys`.
- Full `test/llm/test_llm_objectives.py` (non-slow) and `test/llm/test_wrapper.py`
  suites: no regressions. (`test_grpo`, `test_failure_missing_entries` and
  `test_sft_assistant_only` cannot run in this environment because
  `transformers` is not installed; they are unmodified and unrelated.)

## Deferred (not this run)

- Full RL-training convergence on a real LLM (e.g. a GRPO-style trainer swap
  to `GSPOLoss` on a small Qwen model), on GPU.
- The paper's MoE-stability headline result (GSPO outperforming GRPO on
  Qwen3-MoE post-training under limited-rollout settings), which requires
  multi-node MoE training runs.
- Interaction with `kl_mask_threshold` and the KL penalties at scale: these
  paths are inherited unchanged from `GRPOLoss` and only unit-exercised here.
