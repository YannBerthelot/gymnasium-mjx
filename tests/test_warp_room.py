"""The MuJoCo Warp room check (owner's call, 2026-11-12: stop in the package).

Contract: on MuJoCo Warp, a step that needed more contact slots than the
shared buffer holds, or more constraint rows than a world has, raises when the
program runs, under jit and vmap, naming which ran out; a step within room
(up to and including full) runs on. The realistic bug each case catches is
named in its test. Warp's own counters need a recent GPU, so these feed the
check the counters directly; the check script for real Warp runs is in
ajax-scratch/records/gpu-speed-2026-11-12.
"""

import jax
import jax.numpy as jnp
import mujoco
import pytest

from gymnasium_mjx import make
from gymnasium_mjx.base import stop_if_overflowed

ROOM = 48


def run_check(contacts, rows):
    """One step's counters, as a jitted program sees them."""

    @jax.jit
    def step(contacts, rows):
        stop_if_overflowed(contacts, jnp.asarray(ROOM), rows, jnp.asarray(ROOM))
        return contacts

    return jax.block_until_ready(step(jnp.asarray(contacts), jnp.asarray(rows)))


def test_a_full_buffer_is_not_an_overflow():
    """Catches an off-by-one that stops runs using all the room they have."""
    run_check(ROOM, ROOM)


def test_too_many_contacts_stop_the_run():
    """Catches Warp's silent contact drop going unreported (Hopper at 47
    worlds before the buffer was sized)."""
    with pytest.raises(jax.errors.JaxRuntimeError, match="dropped contacts"):
        run_check(ROOM + 1, 0)


def test_one_world_out_of_constraint_rows_stops_a_batch():
    """Catches a check that looks at one world, or skips the rows, when one
    world of a vmapped batch overflows its constraint rows."""
    rows = jnp.array([0, ROOM + 1, 0])

    @jax.jit
    @jax.vmap
    def batch(rows):
        stop_if_overflowed(jnp.asarray(0), jnp.asarray(ROOM), rows, jnp.asarray(ROOM))
        return rows

    with pytest.raises(jax.errors.JaxRuntimeError, match="needed 49 constraint"):
        jax.block_until_ready(batch(rows))


@pytest.mark.skipif(
    mujoco.mj_version() >= 3_014_000, reason="MuJoCo Warp computes cfrc_ext from 3.14"
)
@pytest.mark.parametrize("task_id", ["Ant-v5", "Humanoid-v5", "HumanoidStandup-v5"])
def test_tasks_reading_contact_forces_refuse_warp_before_3_14(task_id):
    """Catches these tasks failing deep inside their first step on MuJoCo
    Warp before 3.14 (MJX computes cfrc_ext on JAX only): the error comes at
    make, and says to upgrade."""
    with pytest.raises(ValueError, match="upgrade mujoco"):
        make(task_id, physics="warp")
