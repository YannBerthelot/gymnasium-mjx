# Changelog

## Unreleased

## 0.1.0

First release. Gymnasium's v5 MuJoCo tasks on MJX, moved out of Ajax
(`ajax.environments.mujoco`), where they were added:

- InvertedPendulum, InvertedDoublePendulum, Reacher, Swimmer, HalfCheetah,
  Hopper, Walker2d and Pusher, built by `make("<id>")`, listed in `TASKS`.
- `State`, the task's state between steps, and `GymnasiumTask`, the base
  class; the package needs only `jax`, `mujoco` and `mujoco-mjx`, with
  MuJoCo 3.7 to 3.10, the versions on which MJX and C MuJoCo agree.
- `gymnasium_mjx.playground.PlaygroundTask` (the `playground` extra): a task
  as a mujoco_playground environment, for Playground's wrappers.
- `tools/drift.py` and `tools/same_policy.py`, the checks against Gymnasium
  that the README reports, and `tools/plot_curves.py`, which draws its
  learning-curve figure.

Upgrade notes for Ajax users: Ajax's `"mujoco:<id>"` ids are unchanged. The
reset draws match Ajax's for the same key.
