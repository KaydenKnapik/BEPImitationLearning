# Changes Log

## 2026-04-17

- **What changed:** Reduced `num_envs` from `6144` to `1024` in `legged_gym/legged_gym/envs/g1/g1_29_config.py`.
- **Why:** Training failed with CUDA out-of-memory on a ~8 GB GPU; lowering parallel environments reduces GPU memory usage so rollout storage can be allocated.
- **What changed:** Updated `num_envs` from `1024` to `1020` in `legged_gym/legged_gym/envs/g1/g1_29_config.py`.
- **Why:** The environment code partitions envs into 6 equal regions using integer division; `1024` created region tensors of size `1020`, causing CUDA index out-of-bounds during reset.
- **What changed:** Added `wandb_entity = "i-p-b-bouwmeester-eindhoven-university-of-technology"` to `legged_gym/legged_gym/envs/g1/g1_29_config.py` and updated `rsl_rl/rsl_rl/utils/wandb_utils.py` to use `runner.wandb_entity` as a first-class source (with env var fallback).
- **Why:** Training previously failed with `KeyError: WANDB_USERNAME`; this allows W&B initialization from project config without requiring shell environment setup each run.
