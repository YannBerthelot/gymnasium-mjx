# Where gymnasium-mjx differs from Gymnasium

Gymnasium is the reference: its model files, and its task code translated
line by line. These are the places where results still differ, and why.

## MuJoCo versions

Python 3.12 or newer, MuJoCo 3.7 to 3.15. The range is deliberate: the aim
is Gymnasium's tasks as its results were produced, so the package takes the
MuJoCo versions on which MJX and C MuJoCo agree. MuJoCo 3.16 rewrote C's
capsule-capsule contact, which gymnasium-mjx matches up to 3.15; newer
versions come in once they are checked. From 3.11, C's solver stops earlier
(see below).

## Random draws

The random draws follow Gymnasium's distributions, not its generator's
stream: a reset is a sample of the same initial-state distribution, not the
same sample for the same seed.

## Where MJX and C MuJoCo differ

Gymnasium runs on MuJoCo's C library; MJX departs from it in two places
these tasks meet.

- **Contacts of capsules.** For a capsule on a plane, MJX orients the
  friction pyramid along a fixed axis once the capsule stands more than 60
  degrees from the plane, where C MuJoCo follows the capsule; for a sphere
  or a capsule on a capsule, it moves the contact normal by up to 7e-3. With
  these, one Ant step differed by up to 0.9 in velocity and one Humanoid
  step by 2e-4. The package replaces those three colliders with C MuJoCo's
  rules while its tasks trace MJX (`gymnasium_mjx/contact.py`), so the
  contacts match C's to 1e-14; other users of MJX in the same process are
  untouched. One case is left: two capsules within a few millionths of a
  radian of parallel get two contacts in C and one here. The same fixes are
  proposed to MJX itself
  ([google-deepmind/mujoco#3863](https://github.com/google-deepmind/mujoco/pull/3863);
  the plane-capsule part also in
  [#3678](https://github.com/google-deepmind/mujoco/pull/3678)); once MJX
  has them, the package drops its replacements.
- **Where C's solver stops (MuJoCo 3.11 and later).** C MuJoCo's Newton
  solver stops once its Newton decrement or duality gap is below
  `opt.tolerance` (1e-8; commits 1e66efd11 and c69ef0308), within its
  tolerance of the solution; MJX solves on. Gymnasium's own steps then sit up
  to 4e-8 from MJX's (measured on Reacher and HumanoidStandup), and agree to
  1e-8 once C solves to convergence (tolerance 0), which the float64 checks
  do.
- **No PGS solver.** Humanoid and HumanoidStandup's models ask for PGS,
  which MJX lacks, so they run with Newton's. Both solve the same problem;
  PGS stops at its 50 iterations before converging, and after one step the
  two differ by up to 3e-4 in velocity.

## On MuJoCo Warp

`make(task_id, physics="warp")` runs a task on MuJoCo Warp, MJX's GPU
physics, which needs a recent NVIDIA GPU and `warp-lang`.

- **Ant, Humanoid and HumanoidStandup need MuJoCo 3.14 or newer.** Their
  rewards read the bodies' contact forces (`cfrc_ext`); MJX computes them
  on its JAX physics only, and MuJoCo Warp from 3.14, after each step. Below
  3.14, `make` raises and says to upgrade.
- **Capsules on the floor, a deliberate workaround.** MuJoCo Warp's
  capsule-on-plane contact turns its tangent to a fixed axis once a capsule
  stands steeper than 60°, where C MuJoCo keeps it along the capsule; Ant's
  legs stand that steep, and its steps moved by up to 1.4 in velocity.
  gymnasium-mjx gives MuJoCo Warp C MuJoCo's tangent (`warp_contact.py`) on
  the MuJoCo versions it was checked on, 3.14 and 3.15. On another version,
  or if MuJoCo Warp already stepped a capsule on a plane in the process, it
  warns and leaves Warp as it is, and Ant then refuses Warp. Reported as
  [mujoco_warp#1949](https://github.com/google-deepmind/mujoco_warp/issues/1949)
  and fixed by
  [mujoco_warp#1950](https://github.com/google-deepmind/mujoco_warp/pull/1950);
  the workaround goes once a MuJoCo release bundles that fix.
- **Room for contacts.** Warp keeps one contact buffer for every
  environment stepped at once. It holds 49,152 contacts, about 24,000
  Hopper environments. A step that needs more raises an error rather than
  dropping contacts.
- The capsule collisions made to match C MuJoCo (above) apply to the JAX
  physics only; Warp uses its own.

## How it differs from Brax's MuJoCo tasks

Brax's tasks (checked in brax 0.14.2 against Gymnasium 1.2.0) are a separate
benchmark, and their results are not comparable with Gymnasium's:

- the model files are rewritten: Euler integration instead of RK4, most
  contacts disabled, Humanoid's tendons removed, Walker2d's right foot
  friction 0.9 instead of 1.9, Pusher's object density 1 instead of 0.01;
- there are no contact forces in observations or costs;
- some tasks differ outright: InvertedDoublePendulum's time step (0.02
  instead of 0.05) and velocity noise (0.01 instead of 0.1),
  HumanoidStandup's height reward, Reacher's observation and goal draw;
- the default physics is Brax's own, not MuJoCo's solver.

gymnasium-mjx keeps Gymnasium's model files byte for byte and MuJoCo's
solver (through MJX), and translates each task's code line by line.
