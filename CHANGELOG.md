# Changelog

## Unreleased

- MuJoCo 3.11 to 3.15 too (was 3.7 to 3.10). From 3.11, C MuJoCo's Newton
  solver stops once it is within `opt.tolerance` of the solution, so
  Gymnasium's own steps sit up to 4e-8 from MJX's (measured on Reacher and
  HumanoidStandup); MJX solves on.
- Ant, Humanoid and HumanoidStandup run on MuJoCo Warp (`physics="warp"`)
  from MuJoCo 3.14, which computes their contact forces after each step.
  Below 3.14 they refuse Warp and say to upgrade.
- On MuJoCo Warp, C MuJoCo's tangent for a capsule on a plane, a deliberate
  workaround while MuJoCo Warp falls back to a fixed axis on capsules
  steeper than 60° (Ant's legs moved by up to 1.4 in velocity). Applied on
  MuJoCo 3.14 and 3.15 only; elsewhere it warns and Ant refuses Warp.
- On MuJoCo Warp, the convex-collision (GJK/EPA) buffer has a slot for each
  of a task's geom pairs in 1,024 worlds, not for the whole contact buffer:
  each physics call allocates scratch for every slot (320 MB a call on
  Pusher, now 47 MB at most), and each CUDA graph Warp keeps reserves
  addresses for it. On an 11 GB GPU, new graphs failed after a few PPO
  evaluations on Pusher with under 1.4 GB in use. Ported from Ajax.
- A `warp` extra, and the `playground` extra now installs the warp-lang
  that the installed MJX release was built against: MJX imports warp-lang
  when present, and another version fails at import.

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
