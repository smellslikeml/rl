# Flaky Test Report - 2026-08-24

## Summary

- **Flaky tests**: 9
- **Newly flaky** (last 7 days): 1
- **Resolved**: 0
- **Total tests analyzed**: 31576
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...s/test_tqc.py::TestTQC::test_tqc_numerical_contract[True]` | 8.7% (12/138) | 12 | 0.17 | 2026-08-23 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` | 8.7% (12/138) | 12 | 0.17 | 2026-08-23 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]` | 8.7% (12/138) | 12 | 0.17 | 2026-08-23 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]` | 8.7% (12/138) | 12 | 0.17 | 2026-08-23 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]` | 8.7% (12/138) | 12 | 0.17 | 2026-08-23 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]` | 8.7% (12/138) | 12 | 0.17 | 2026-08-23 |
| `...stDiffusionActor::test_reduced_precision_schedule[dtype0]` 🆕 | 8.7% (11/127) | 11 | 0.17 | 2026-08-23 |
| `...t_rb_core.py::test_replay_buffer_prefetch_dumps_roundtrip` | 7.2% (10/138) | 10 | 0.14 | 2026-08-21 |
| `...py::TestDreamerV3Components::test_block_gru_torch_compile` | 6.5% (9/138) | 9 | 0.13 | 2026-08-20 |


### Newly Flaky Tests

- `test/modules/test_actor.py::TestDiffusionActor::test_reduced_precision_schedule[dtype0]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-24T06:18:43.778609+00:00*