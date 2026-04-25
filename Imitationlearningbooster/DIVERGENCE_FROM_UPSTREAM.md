# Divergence from the paper’s upstream repository

This document tracks how **`Imitationlearningbooster/`** (and this workspace’s stated goals) differ from the original **Humanoid-Goalkeeper** codebase published with the paper.

## Project goal

The paper’s release targets **Unitree G1** goalkeeper-style imitation learning in Isaac Gym / `legged_gym` + `rsl_rl`. **This workspace’s goal** is to **re-target that pipeline to Booster Robotics T1**: same learning ideas where possible, but robot URDF/MJCF, joint counts, observations, actions, and task wiring must match **BoosterT1**, not G1.

Reference implementation (read-only in this workspace): `Humanoid-Goalkeeper/` — treat as upstream mirror, **not** the place for Booster edits.

## Upstream baseline

| Field | Value |
|--------|--------|
| Repository | [https://github.com/InternRobotics/Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper) |
| Commit checked in local `Humanoid-Goalkeeper/` | `976a81ff19b7306bafbe923d2890066b68a85271` (`976a81f` — “update arxiv”) |

Anything not listed below should be assumed to match upstream at that commit unless a newer dated entry says otherwise.

---

## Snapshot: local `Humanoid-Goalkeeper/` vs `origin/main` (reference only)

The folder `Humanoid-Goalkeeper/` is **frozen** for agent edits per Cursor rules, but the checkout on disk may still differ from pure upstream. As of **2026-04-18**, the following **uncommitted** differences exist there relative to `origin/main` (informational; future Booster work must **not** apply fixes by editing this tree):

| Area | Upstream (`origin/main`) | Local tree |
|------|--------------------------|------------|
| `legged_gym/legged_gym/envs/g1/g1_29_config.py` | `num_envs = 6144` | `num_envs = 1020` with comment about divisibility by 6 for region partitioning |
| Same file | No `wandb_entity` in runner config | `wandb_entity = "i-p-b-bouwmeester-eindhoven-university-of-technology"` under `G129CfgPPO` |
| `rsl_rl/rsl_rl/utils/wandb_utils.py` | Entity from `WANDB_USERNAME` env only | Entity from `cfg.get("wandb_entity")` or `WANDB_USERNAME`; updated `KeyError` message |
| New / untracked | — | `changes.md`, `.cursor/rules/`, `Wandbkey.txt`, `isaacgym/` (local install tree) |

Rationale (from `Humanoid-Goalkeeper/changes.md`): lower `num_envs` for GPU memory; `1020` to avoid CUDA index errors from env region split; W&B entity from config to avoid missing `WANDB_USERNAME`.

---

## Changes under `Imitationlearningbooster/` vs upstream

`Imitationlearningbooster/` is the **only** allowed location for new BoosterT1 migration code. Append new **dated** sections below as you implement Booster-specific behavior (configs, envs, assets, scripts).

*(No entries yet — folder initialized 2026-04-18.)*

## 2026-04-18 — Workspace `export/` snapshot for retargeting

- **Files:** `../export/README.md`, `../export/motion_dataset/*`, `../export/unitree_g1/*`, `../export/booster_t1/*`
- **What changed:** Added a sibling **`export/`** directory at the BEP workspace root containing copies of goalkeeper motion tensors + `joint_id.txt`, full Unitree G1 URDF/meshes, and Booster T1 assets from `Boosterversion/booster_t1/`, plus a README describing `.pt` tensor keys and retargeting notes.
- **Why:** Single place for Booster conversion work without editing `Humanoid-Goalkeeper/`.
- **Upstream reference:** `legged_gym/resources/datasets/goalkeeper/`, `legged_gym/resources/robots/g1/`, `legged_gym/legged_gym/envs/g1/g1_utils.py` (`MotionLib`).

### Template for new entries

```markdown
## YYYY-MM-DD — short title

- **Files:** `path/under/Imitationlearningbooster/...`
- **What changed:** …
- **Why:** …
- **Upstream reference:** file(s) in Humanoid-Goalkeeper this parallels or replaces.
```
