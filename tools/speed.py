"""Environment steps per second: Gymnasium's C MuJoCo against gymnasium-mjx.

Both sides take uniform random actions within the control range and reset an
environment when it terminates or reaches the time limit, as training does.
Gymnasium runs its vector environments ("sync": one process; "async": one
process per environment). gymnasium-mjx runs a jitted scan of a vmapped step
on the default JAX device; an ended environment takes a reset drawn at the
start, so a step costs the same whether or not an episode ended. Compilation
is timed apart and left out of the rate. Each row is one JSON line.

    python tools/speed.py --task Hopper-v5 Humanoid-v5 --envs 1 64 4096 \
        --side mjx --out speed.jsonl
"""

import argparse
import json
import os
import platform
import time

import jax
import jax.numpy as jnp

from gymnasium_mjx import make

GYMNASIUM_MODES = ("sync", "async")


def device_name() -> str:
    device = jax.devices()[0]
    if device.platform == "cpu":
        return f"cpu ({platform.processor() or platform.machine()}, {os.cpu_count()} cores)"
    return device.device_kind


def mjx_rate(task_id: str, n_envs: int, steps: int) -> dict:
    task = make(task_id)
    low, high = (jnp.asarray(bound) for bound in task.mj_model.actuator_ctrlrange.T)
    keys = jax.random.split(jax.random.key(0), n_envs)
    fresh = jax.vmap(task.reset)(keys)

    def one_step(carry, key):
        states, elapsed = carry
        actions = jax.random.uniform(
            key, (n_envs, task.action_size), minval=low, maxval=high
        )
        states = jax.vmap(task.step)(states, actions)
        elapsed = elapsed + 1
        ended = (states.done > 0) | (elapsed >= task.episode_length)

        def pick(new, old):
            mask = ended.reshape(ended.shape + (1,) * (new.ndim - 1))
            return jnp.where(mask, new, old)

        states = jax.tree.map(pick, fresh, states)
        return (states, jnp.where(ended, 0, elapsed)), None

    @jax.jit
    def run(states, key):
        carry = (states, jnp.zeros(n_envs, jnp.int32))
        return jax.lax.scan(one_step, carry, jax.random.split(key, steps))[0][0]

    start = time.perf_counter()
    compiled = run.lower(fresh, jax.random.key(1)).compile()
    compile_seconds = time.perf_counter() - start
    jax.block_until_ready(compiled(fresh, jax.random.key(1)))  # warm-up
    start = time.perf_counter()
    jax.block_until_ready(compiled(fresh, jax.random.key(2)))
    seconds = time.perf_counter() - start
    return {
        "side": "gymnasium-mjx",
        "device": device_name(),
        "compile_seconds": round(compile_seconds, 1),
        "steps_per_second": n_envs * steps / seconds,
    }


def gymnasium_rate(task_id: str, n_envs: int, steps: int, mode: str) -> dict:
    import gymnasium

    environments = gymnasium.make_vec(task_id, num_envs=n_envs, vectorization_mode=mode)
    environments.reset(seed=0)
    environments.action_space.seed(0)
    actions = [environments.action_space.sample() for _ in range(steps)]
    start = time.perf_counter()
    for action in actions:
        environments.step(action)
    seconds = time.perf_counter() - start
    environments.close()
    return {
        "side": f"gymnasium ({mode})",
        "device": f"cpu ({platform.processor() or platform.machine()}, {os.cpu_count()} cores)",
        "compile_seconds": 0.0,
        "steps_per_second": n_envs * steps / seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", nargs="+", default=["Hopper-v5"])
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 16, 256, 4096])
    parser.add_argument("--steps", type=int, default=1000, help="steps per environment")
    parser.add_argument(
        "--side", choices=("mjx", *GYMNASIUM_MODES), nargs="+", default=["mjx"]
    )
    parser.add_argument("--out", default="speed.jsonl")
    arguments = parser.parse_args()
    with open(arguments.out, "a") as out:
        for task_id in arguments.task:
            for side in arguments.side:
                for n_envs in arguments.envs:
                    if side == "mjx":
                        row = mjx_rate(task_id, n_envs, arguments.steps)
                    else:
                        row = gymnasium_rate(task_id, n_envs, arguments.steps, side)
                    row.update(task=task_id, n_envs=n_envs, steps=arguments.steps)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(
                        f"{task_id:20} {row['side']:18} {n_envs:6} envs "
                        f"{row['steps_per_second']:12,.0f} steps/s "
                        f"(compile {row['compile_seconds']} s)"
                    )


if __name__ == "__main__":
    main()
