"""PlaygroundTask: a task as a mujoco_playground MjxEnv.

Contract (from the adapter's purpose: Playground's wrappers and the libraries
built on them run the task as it is):

- reset and step give the task's own state, field by field (a field copied
  into the wrong slot, such as the termination into the reward, fails);
- Playground's training wrapper ends an episode at the task's time limit
  (Reacher: Gymnasium's 50 steps), not before;
- an action in [-1, 1] reaches the whole of Gymnasium's action space (each
  motor's ctrlrange: [-0.4, 0.4] on Humanoid), -1 and 1 landing on its
  bounds, and the step is the task's step with that action.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("mujoco_playground")

from mujoco_playground import wrapper

from gymnasium_mjx import make
from gymnasium_mjx.playground import PlaygroundTask


def test_the_adapter_steps_as_the_task_does():
    # Hopper: a reward, a termination and reward terms that all differ.
    task = make("Hopper-v5")
    adapted = PlaygroundTask(task)
    key = jax.random.key(3)
    action = jnp.linspace(-1.0, 1.0, task.action_size)

    expected = task.step(task.reset(key), action)
    observed = adapted.step(adapted.reset(key), action)

    np.testing.assert_array_equal(observed.obs, expected.obs)
    np.testing.assert_array_equal(observed.reward, expected.reward)
    np.testing.assert_array_equal(observed.done, expected.done)
    np.testing.assert_array_equal(observed.data.qpos, expected.data.qpos)
    assert observed.metrics.keys() == expected.metrics.keys()
    assert adapted.observation_size == task.observation_size
    assert adapted.dt == task.dt


def test_playgrounds_wrapper_ends_the_episode_at_the_time_limit():
    task = make("Reacher-v5")
    env = wrapper.wrap_for_brax_training(
        PlaygroundTask(task), episode_length=task.episode_length
    )
    # Playground's wrappers read the batch size off raw (uint32) keys.
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 2))
    step = jax.jit(env.step)
    action = jnp.zeros((2, task.action_size))
    for _ in range(task.episode_length - 1):
        state = step(state, action)
    before_limit = np.asarray(state.done)

    state = step(state, action)

    np.testing.assert_array_equal(before_limit, 0)
    np.testing.assert_array_equal(state.done, 1)
    np.testing.assert_array_equal(state.info["truncation"], 1)


@pytest.mark.parametrize("task_id", ["Humanoid-v5", "Hopper-v5"])
def test_the_adapter_maps_its_actions_onto_gymnasiums_action_space(task_id):
    gymnasium = pytest.importorskip("gymnasium")
    space = gymnasium.make(task_id).action_space
    task = make(task_id)
    adapted = PlaygroundTask(task)
    key = jax.random.key(3)
    ones = jnp.ones(task.action_size)

    for agent, control in [(ones, space.high), (-ones, space.low)]:
        observed = adapted.step(adapted.reset(key), agent)
        expected = task.step(task.reset(key), jnp.asarray(control))

        np.testing.assert_allclose(observed.data.ctrl, control, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(observed.reward, expected.reward)
        np.testing.assert_array_equal(observed.obs, expected.obs)
