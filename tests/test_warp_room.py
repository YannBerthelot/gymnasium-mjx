"""The MuJoCo Warp room check (owner's call, 2026-11-12: stop in the package).

Contract: on MuJoCo Warp, a step that needed more contact slots than the
shared buffer holds, or more constraint rows than a world has, raises when the
program runs, under jit and vmap, naming which ran out; a step within room
(up to and including full) runs on; a physics call allocates
convex-collision scratch for the task's geom pairs, not for the whole contact
buffer. The realistic bug each case catches is named in its test. Warp's own counters need a recent GPU, so these feed the
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


def warp_scratch_megabytes(report: str) -> float:
    """The CPU peak of a ``wp.ScopedMemoryTracker`` report, in MB."""
    units = {"B": 1e-6, "KB": 1e-3, "MB": 1.0, "GB": 1e3}
    lines = report.splitlines()
    start = lines.index("  Peak usage:") + 1
    for line in lines[start:]:
        fields = line.split()
        if fields[0] == "cpu":
            return float(fields[1]) * units[fields[2]]
    raise ValueError(f"no CPU peak in the report:\n{report}")


@pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="on a GPU, Warp replays a captured CUDA graph: no allocation to count",
)
def test_a_pusher_physics_call_on_warp_allocates_scratch_for_its_pairs_only():
    """Catches the convex-collision scratch sized for the whole contact
    buffer: MuJoCo Warp sizes it for 49,152 collisions unless told, and
    every Pusher physics call allocated 235 MB (320 MB on a GPU). Each CUDA
    graph Warp keeps of a call reserves its own addresses for that scratch,
    and on an 11 GB GPU new graphs failed after about 76 (21 GiB) with under
    1.4 GB in use. Room for Pusher's 7 geom pairs in 1,024 worlds takes 47 MB
    at most; below 100 MB, the 96 graphs of three Warp functions fit in half
    of that 21 GiB, and the old size fails. MJX runs Warp's kernels on a CPU
    when a task is made directly (no CUDA graphs), where Warp's tracker sees
    every allocation. Moved here from Ajax with the tasks."""
    wp = pytest.importorskip("warp")

    task = make("Pusher-v5", physics="warp")
    keys = jax.random.split(jax.random.key(0), 2)
    states = jax.jit(jax.vmap(task.reset))(keys)
    step = jax.jit(jax.vmap(task.step))
    actions = jnp.ones((2, task.action_size))
    jax.block_until_ready(step(states, actions))  # compiled outside the count

    reports: list[str] = []
    with wp.ScopedMemoryTracker("pusher", report_func=reports.append) as tracker:
        tracker.clear()
        jax.block_until_ready(step(states, actions))

    assert warp_scratch_megabytes(reports[0]) < 100
