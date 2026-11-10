"""A task as a mujoco_playground environment (the ``playground`` extra).

Playground's wrappers (time limit, auto-reset, vectorisation) and the
libraries built on them, such as Brax's training and Ajax, take an MjxEnv.
PlaygroundTask presents a task as one; the physics, rewards and draws stay
the task's own.

    from mujoco_playground import wrapper
    from gymnasium_mjx import make
    from gymnasium_mjx.playground import PlaygroundTask

    task = make("Hopper-v5")
    env = wrapper.wrap_for_brax_training(
        PlaygroundTask(task), episode_length=task.episode_length
    )
"""

from typing import Any

import jax
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from gymnasium_mjx.base import GymnasiumTask, State

STATE_FIELDS = ("data", "obs", "reward", "done", "metrics", "info")


def copy_state(state: Any, kind: type) -> Any:
    """The same six fields in the other library's State."""
    return kind(**{field: getattr(state, field) for field in STATE_FIELDS})


class PlaygroundTask(mjx_env.MjxEnv):
    def __init__(self, task: GymnasiumTask):
        config = config_dict.create(
            ctrl_dt=task.dt,
            sim_dt=task.sim_dt,
            episode_length=task.episode_length,
            impl=task.physics,
        )
        super().__init__(config)
        self.task = task

    def reset(self, rng: jax.Array) -> mjx_env.State:
        return copy_state(self.task.reset(rng), mjx_env.State)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        stepped = self.task.step(copy_state(state, State), action)
        return copy_state(stepped, mjx_env.State)

    @property
    def xml_path(self) -> str:
        return self.task.xml_path

    @property
    def action_size(self) -> int:
        return self.task.action_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.task.mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self.task.mjx_model
