# Flaky Test Report - 2026-08-25

## Summary

- **Flaky tests**: 16
- **Newly flaky** (last 7 days): 9
- **Resolved**: 0
- **Total tests analyzed**: 31588
- **CI runs analyzed**: 45

---

## Flaky Tests

| Test | Failure Rate | Failures | Flaky Score | Last Failed |
|------|--------------|----------|-------------|-------------|
| `...stDiffusionActor::test_reduced_precision_schedule[dtype0]` | 9.3% (13/139) | 13 | 0.19 | 2026-08-24 |
| `...s/test_tqc.py::TestTQC::test_tqc_numerical_contract[True]` | 9.3% (13/139) | 13 | 0.19 | 2026-08-24 |
| `...s_kwargs_have_config_fields[TensorDictReplayBufferConfig]` | 9.3% (13/139) | 13 | 0.19 | 2026-08-24 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype0]` | 9.3% (13/139) | 13 | 0.19 | 2026-08-24 |
| `..._rsample_and_log_prob[device0-True-False--1.0-1.0-dtype1]` | 9.3% (13/139) | 13 | 0.19 | 2026-08-24 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype0]` | 9.3% (13/139) | 13 | 0.19 | 2026-08-24 |
| `..._rsample_and_log_prob[device0-True-False--2.0-3.0-dtype1]` | 9.3% (13/139) | 13 | 0.19 | 2026-08-24 |
| `...t_rssm_rollout_fast_path_matches_tensordict_path[device0]` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-08-24 |
| `...st_rssm_rollout_higher_order_scan_matches_loop[1-device0]` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-08-24 |
| `...st_rssm_rollout_higher_order_scan_matches_loop[3-device0]` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-08-24 |
| `...st_rssm_rollout_higher_order_scan_matches_loop[8-device0]` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-08-24 |
| `...:TestDreamerV3Components::test_rssm_rollout_compile[step]` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-08-24 |
| `...:TestDreamerV3Components::test_rssm_rollout_compile[scan]` 🆕 | 9.1% (5/55) | 5 | 0.18 | 2026-08-24 |
| `...TestIQL::test_iql_deactivate_vmap[None-0.1-0.1-device0-1]` 🆕 | 9.1% (4/44) | 4 | 0.15 | 2026-08-24 |
| `...t_iql.py::TestIQL::test_iql_state_dict[0.1-0.0-device0-1]` 🆕 | 9.1% (4/44) | 4 | 0.15 | 2026-08-24 |
| `...creteIQL::test_discrete_iql_state_dict[0.1-0.0-device0-1]` 🆕 | 9.1% (4/44) | 4 | 0.15 | 2026-08-24 |


### Newly Flaky Tests

- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_rssm_rollout_fast_path_matches_tensordict_path[device0]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_rssm_rollout_higher_order_scan_matches_loop[1-device0]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_rssm_rollout_higher_order_scan_matches_loop[3-device0]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_rssm_rollout_higher_order_scan_matches_loop[8-device0]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_rssm_rollout_compile[step]`
- `test/modules/test_dreamer_components.py::TestDreamerV3Components::test_rssm_rollout_compile[scan]`
- `test/objectives/test_iql.py::TestIQL::test_iql_deactivate_vmap[None-0.1-0.1-device0-1]`
- `test/objectives/test_iql.py::TestIQL::test_iql_state_dict[0.1-0.0-device0-1]`
- `test/objectives/test_iql.py::TestDiscreteIQL::test_discrete_iql_state_dict[0.1-0.0-device0-1]`

---

## Configuration

- Minimum failure rate: 5%
- Maximum failure rate: 95%
- Minimum failures required: 2
- Minimum executions required: 3

---

*Generated at 2026-08-25T06:14:02.069881+00:00*