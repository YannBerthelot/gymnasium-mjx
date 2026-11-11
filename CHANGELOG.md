# Changelog

## Unreleased

## 0.2.0

All eleven of Gymnasium's v5 MuJoCo tasks.

Upgrade notes: the contact change below can move results on the tasks
whose capsules touch the floor or each other (HalfCheetah, Hopper, Walker2d,
Pusher). One step stays within 1e-8 of Gymnasium's, as before, but
trajectories with contacts can drift apart from 0.1.0's, as between two
MuJoCo builds: compare runs across the two versions statistically.

- Ant-v5, Humanoid-v5 and HumanoidStandup-v5, with the contact forces
  (`cfrc_ext`) in their observations and costs. Humanoid's models ask for
  the PGS solver, which MJX lacks; they run with Newton's.
- C MuJoCo's contacts for a capsule on a plane, a sphere on a capsule and two
  capsules, in place of MJX's, while the tasks trace MJX. MJX's differed in
  the friction pyramid's orientation and in the contact normal.
- `set_state` keeps the positions as given, as C MuJoCo does: MJX normalised
  free joints' quaternions, which changed the first observation of a reset.

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
