# gymnasium-mjx

Reproduces Gymnasium's v5 MuJoCo tasks in pure JAX, on
[MJX](https://mujoco.readthedocs.io/en/stable/mjx.html), so that training
runs end to end on the GPU: environment and agent compile into one program,
and thousands of environments step at once.

This is not a Farama Foundation project. The model files are Gymnasium's,
under its MIT licence (`LICENSES/Gymnasium.txt`).

## Why

- **End-to-end JAX.** With a JAX agent, one jitted program steps the
  environments and updates the agent: no Python between steps, no copies
  between host and device, and `vmap` runs thousands of environments, or
  several seeds, at once. A Python environment stops a JAX agent at every
  step.
- **The same tasks as the papers.** JAX training so far meant look-alike
  tasks (Brax's, MuJoCo Playground's) with other models, rewards and
  observations, whose returns cannot be compared with published Gymnasium
  results. Here the model files are Gymnasium's, byte for byte, and the task
  code is Gymnasium's, translated line by line.

## Install

```bash
pip install gymnasium-mjx                 # jax, mujoco and mujoco-mjx only
pip install "gymnasium-mjx[playground]"   # and the mujoco_playground adapter
pip install "gymnasium-mjx[warp]"         # and MuJoCo Warp, on an NVIDIA GPU
```

Python 3.12 or newer, MuJoCo 3.7 to 3.15 (3.14 or newer for Ant and the
Humanoids on MuJoCo Warp).

## Use it as Gymnasium

The same task ids and keyword arguments as `gymnasium.make`. The state is
passed in and returned, as JAX needs:

```python
# doc: skip (two columns side by side)
# Gymnasium                                   # gymnasium-mjx
env = gymnasium.make("Hopper-v5")             task = gymnasium_mjx.make("Hopper-v5")
obs, info = env.reset(seed=0)                 state = task.reset(jax.random.key(0))
obs, reward, terminated, truncated, info = \  state = task.step(state, action)
    env.step(action)                          # state.obs, state.reward, state.done
```

`state.done` is the termination; the time limit, `task.episode_length`, is
the caller's to apply, as Gymnasium's `TimeLimit` wrapper does. Many
environments at once are a `vmap`, and the whole step can be jitted:

```python
import jax
import gymnasium_mjx

task = gymnasium_mjx.make("Hopper-v5")
states = jax.vmap(task.reset)(jax.random.split(jax.random.key(0), 4096))
actions = jax.numpy.zeros((4096, task.action_size))
states = jax.jit(jax.vmap(task.step))(states, actions)
```

## Tasks

| Id | Contacts | Id | Contacts |
|---|---|---|---|
| InvertedPendulum-v5 | no | Hopper-v5 | yes |
| InvertedDoublePendulum-v5 | no | Walker2d-v5 | yes |
| Reacher-v5 | no | Pusher-v5 | yes |
| Swimmer-v5 | no | Ant-v5 | yes |
| HalfCheetah-v5 | yes | Humanoid-v5 | yes |
| | | HumanoidStandup-v5 | yes |

## How close to Gymnasium, and how fast

- One step from the same state gives Gymnasium's observation, reward and
  termination to 1e-8 in float64, on every task.
- The same trained policy, and the same agent trained from scratch, score
  the same on both sides, on the tasks measured so far (learning curves in
  [docs/checks.md](docs/checks.md#learning-curves)).
- Where MJX and C MuJoCo still differ, and by how much:
  [docs/differences.md](docs/differences.md).

<!-- TODO: speed, from tools/speed.py on a GPU host: steps per second at
1, 256 and 4096 environments, Gymnasium against gymnasium-mjx. -->

All checks, the learning curves, the speed method and the tools:
[docs/checks.md](docs/checks.md).

## Development

```bash
uv sync
uv run pre-commit run --all-files
uv run pytest
```

## Licence

MIT (`LICENSE`). The model files in `src/gymnasium_mjx/assets/` are
Gymnasium's, MIT, copyright OpenAI and the Farama Foundation
(`LICENSES/Gymnasium.txt`).
