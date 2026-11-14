"""What every task shares: its state, Gymnasium's reset and step on MJX, and
the random keys of a reset, derived by name."""

import dataclasses
import zlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from gymnasium_mjx.contact import c_contacts
from gymnasium_mjx.warp_contact import patch_warp_plane_capsule

ASSETS = Path(__file__).parent / "assets"

#: Set in every name's tag, so that no key derived by name is one that a
#: split or an index fold (which count up from 0) hands out.
NAMED_TAG_BIT = 0x80000000

#: MuJoCo Warp's contact buffer (physics="warp"), one for every world that a
#: program steps at once (vmapped environments and seeds share it). Without
#: it, MuJoCo Warp's default for these models is 48 contacts in all: 47
#: Hopper worlds overflowed it, and MuJoCo Warp then drops contacts silently.
#: Room for WARP_WORLDS worlds of 48 contacts each (MuJoCo Warp's default for
#: one world), near the 50,000 Playground gives its locomotion tasks. The JAX
#: physics ignores it.
WARP_WORLDS = 1024
WARP_CONTACTS = 48 * WARP_WORLDS


def derive_key(key: jax.Array, name: str) -> jax.Array:
    """The key named name, off key. A draw keyed by its name moves no other
    draw when a task adds or drops one. The tag is the name's CRC-32 with
    the top bit set: the same in every process (Python's hash is salted per
    process), and the same as Ajax's derive_key, so the draws match it."""
    return jax.random.fold_in(key, zlib.crc32(name.encode()) | NAMED_TAG_BIT)


def stop_if_overflowed(
    contacts: jax.Array, contact_room: jax.Array, rows: jax.Array, row_room: jax.Array
) -> None:
    """Raises, when the jitted program runs, if a step needed more contact
    slots than the buffer every world shares, or a world more constraint rows
    than it has room for. Works under vmap and jit."""
    overflowed = (jnp.max(contacts) > jnp.min(contact_room)) | (
        jnp.max(rows) > jnp.min(row_room)
    )

    def report(pending: jax.Array) -> jax.Array:
        jax.debug.callback(raise_overflow, contacts, contact_room, rows, row_room)
        return jnp.zeros_like(pending)

    # A host callback on every step cost Ajax's SAC 38% of its speed and PPO
    # 63%; this while_loop runs it only on a step that overflowed (batched, it
    # loops while any world's flag holds).
    jax.lax.while_loop(lambda pending: pending, report, overflowed)


def raise_overflow(contacts, contact_room, rows, row_room) -> None:
    # Under vmap, the callback runs once per world: only a world that ran
    # out of room raises, with its own counts.
    if np.max(contacts) > np.min(contact_room):
        raise ValueError(
            f"MuJoCo Warp dropped contacts: a step needed {np.max(contacts)}"
            f" contact slots, and the buffer every world shares holds"
            f" {np.min(contact_room)}. Step fewer environments at once, or use"
            " physics='jax'."
        )
    if np.max(rows) > np.min(row_room):
        raise ValueError(
            f"MuJoCo Warp dropped constraints: a world needed {np.max(rows)}"
            f" constraint rows and has room for {np.min(row_room)}. Use"
            " physics='jax'."
        )


def cfrc_ext(data: mjx.Data) -> jax.Array:
    """The bodies' external forces (Gymnasium's data.cfrc_ext), which MJX
    keeps with each physics' own fields."""
    return data._impl.cfrc_ext


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class State:
    """A task's state between steps. The fields are those of mujoco_playground's
    State, so the playground adapter only copies them across."""

    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    info: dict[str, Any]

    def replace(self, **changes: Any) -> "State":
        return dataclasses.replace(self, **changes)


class GymnasiumTask:
    """A Gymnasium v5 MuJoCo task: its model file, and its reset and step.

    A step sets the action as the control and runs frame_skip physics steps,
    as Gymnasium's do_simulation does; the action enters the costs as given,
    and MuJoCo clamps the control to the model's range. ``done`` is the
    termination only: the time limit (``episode_length``, from Gymnasium's
    registry) is the caller's to apply, as Gymnasium's TimeLimit wrapper does.
    """

    xml_file: str
    frame_skip: int
    episode_length: int = 1000
    #: Whether the task reads cfrc_ext, which Gymnasium computes after every
    #: step (mj_rnePostConstraint) and MJX only on request.
    contact_forces: bool = False
    #: Whether the task's capsules stand steep enough on the floor that
    #: MuJoCo Warp's own capsule-on-plane tangent moves its steps away from
    #: C MuJoCo's (see warp_contact): on Warp it needs the fix.
    steep_capsules_on_plane: bool = False

    def __init__(self, physics: str = "jax"):
        warp_contact_forces = physics == "warp" and self.contact_forces
        if warp_contact_forces and mujoco.mj_version() < 3_014_000:
            # MJX's rne_postconstraint raises on MuJoCo Warp; from 3.14, Warp
            # runs its own after each step (opt.run_rne_postconstraint).
            raise ValueError(
                f"{type(self).__name__} reads the bodies' contact forces"
                " (cfrc_ext), which MuJoCo Warp computes from MuJoCo 3.14."
                f" This is {mujoco.__version__}: upgrade mujoco and mujoco-mjx,"
                " or use physics='jax'."
            )
        if physics == "warp":
            fixed = patch_warp_plane_capsule()
            if self.steep_capsules_on_plane and not fixed:
                raise ValueError(
                    f"{type(self).__name__} on MuJoCo Warp needs C MuJoCo's"
                    " capsule-on-plane contacts, which gymnasium-mjx could not"
                    " give MuJoCo Warp here (see the warning above). Use"
                    " physics='jax'."
                )
        self.physics = physics
        self.xml_path = str(ASSETS / self.xml_file)
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        if self.mj_model.opt.solver == mujoco.mjtSolver.mjSOL_PGS:
            # MJX has no PGS (Humanoid's models ask for it). Newton solves the
            # same convex problem; PGS stopping at its iteration limit is what
            # separates them (up to 3e-4 in velocity after one step).
            self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mjx_model = mjx.put_model(self.mj_model, impl=physics)
        # Reset's mj_forward, which in C MuJoCo leaves cfrc_ext as reset
        # left it (zero): on Warp, the model without the contact forces.
        self.forward_model = self.mjx_model
        if warp_contact_forces:
            # Gymnasium calls mj_rnePostConstraint after mj_step. Warp runs it
            # at the end of the step's forward pass instead; cfrc_ext reads
            # only the contact forces and the bodies' poses, which the
            # integration after it leaves alone, so the two agree.
            option = self.mjx_model.opt
            warp_option = option._impl.replace(run_rne_postconstraint=True)
            self.mjx_model = self.mjx_model.replace(
                opt=option.replace(_impl=warp_option)
            )
        self.sim_dt = float(self.mj_model.opt.timestep)
        self.dt = self.sim_dt * self.frame_skip
        self.action_size = self.mj_model.nu
        # Gymnasium's init_qpos and init_qvel: the state mj_resetData restores.
        self.init_qpos = jnp.asarray(self.mj_model.qpos0)
        self.init_qvel = jnp.zeros(self.mj_model.nv)

    @property
    def observation_size(self) -> int:
        return jax.eval_shape(self.reset, jax.random.key(0)).obs.shape[-1]

    def uniform(self, key: jax.Array, scale: float, size: int) -> jax.Array:
        return jax.random.uniform(key, (size,), minval=-scale, maxval=scale)

    def set_state(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Gymnasium's set_state: the state, then mj_forward."""
        data = mjx.make_data(
            self.mj_model,
            impl=self.physics,
            naconmax=WARP_CONTACTS,
            naccdmax=self.warp_convex_room(),
        )
        with c_contacts():
            data = mjx.forward(self.forward_model, data.replace(qpos=qpos, qvel=qvel))
        self.check_warp_room(data)
        # MJX's forward writes free joints' quaternions back normalised; C
        # MuJoCo's leaves qpos as set, and Gymnasium's reset observation shows
        # the noisy, unnormalised quaternion.
        return data.replace(qpos=qpos)

    def warp_convex_room(self) -> int | None:
        """Slots in MuJoCo Warp's buffer for convex (GJK/EPA) collisions,
        which every world of a batch shares: one for each geom pair that can
        collide, in each of WARP_WORLDS worlds, and no more than the contact
        buffer, as MuJoCo Warp requires. None on the JAX physics.

        MuJoCo Warp's default is the whole contact buffer, and each physics
        call allocates about 6.5 kB of scratch per slot: 320 MB a call on
        Pusher, which has 7 pairs. Warp keeps a CUDA graph of a call for each
        new set of XLA buffer addresses, up to 32 per function. The graphs
        share their physical memory, but each reserves its own addresses for
        its scratch, and on an 11 GB GPU the driver refused new ones after
        about 76 graphs (21 GiB), with under 1.4 GB in use: Ajax's PPO on
        Pusher stopped after a few evaluations."""
        if self.physics != "warp":
            return None
        pairs = sum(self.mjx_model._impl.geom_pair_type_count)
        return min(int(pairs) * WARP_WORLDS, WARP_CONTACTS)

    def initial_state(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The task's reset_model draw: (qpos, qvel)."""
        raise NotImplementedError

    def observe(self, data: mjx.Data) -> jax.Array:
        raise NotImplementedError

    def outcome(
        self, before: mjx.Data, after: mjx.Data, action: jax.Array
    ) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
        """The step's reward, termination and reward terms."""
        raise NotImplementedError

    def reset(self, key: jax.Array) -> State:
        qpos, qvel = self.initial_state(key)
        data = self.set_state(qpos, qvel)
        # The metrics carry the reward terms, so a reset starts them at zero.
        terms = jax.eval_shape(self.outcome, data, data, jnp.zeros(self.action_size))[2]
        metrics = {name: jnp.zeros(()) for name in terms}
        return State(
            data, self.observe(data), jnp.zeros(()), jnp.zeros(()), metrics, {}
        )

    def physics_step(self, data: mjx.Data, action: jax.Array) -> mjx.Data:
        def substep(carry: tuple, _: None) -> tuple[tuple, None]:
            data, used = carry
            data = mjx.step(self.mjx_model, data.replace(ctrl=action))
            # Warp's counters hold only the latest substep: keep the most any
            # substep used, and check once per step (a check per substep cost
            # 11% on Hopper, against 2-4% once per step).
            return (data, self.warp_room_used(data, used)), None

        with c_contacts():
            carry = (data, self.warp_room_used(data))
            data, used = jax.lax.scan(substep, carry, None, self.frame_skip)[0]
        self.check_warp_room(data, used)
        if self.contact_forces and self.physics != "warp":  # Warp's step did it
            data = mjx.rne_postconstraint(self.mjx_model, data)
        return data

    def warp_room_used(self, data: mjx.Data, most: tuple = ()) -> tuple:
        """On MuJoCo Warp, the contact slots and constraint rows data's step
        used, or the most used since most; nothing on the JAX physics."""
        if self.physics != "warp":
            return ()
        warp = data._impl  # MJX's MuJoCo Warp fields: counters and capacities
        used = (jnp.maximum(warp.nacon, warp.ncollision), warp.nefc)
        if not most:
            return used
        return (jnp.maximum(used[0], most[0]), jnp.maximum(used[1], most[1]))

    def check_warp_room(self, data: mjx.Data, used: tuple = ()) -> None:
        """On MuJoCo Warp, stop with an error when a physics step ran out of
        room: Warp drops the contacts or constraint rows that do not fit and
        goes on with wrong physics. The JAX physics has no such limit."""
        if self.physics != "warp":
            return
        contacts, rows = used or self.warp_room_used(data)
        warp = data._impl
        stop_if_overflowed(contacts, warp.naconmax, rows, warp.njmax)

    def step(self, state: State, action: jax.Array) -> State:
        data = self.physics_step(state.data, action)
        reward, terminated, terms = self.outcome(state.data, data, action)
        return state.replace(
            data=data,
            obs=self.observe(data),
            reward=reward,
            done=terminated.astype(reward.dtype),
            metrics=terms,
        )
