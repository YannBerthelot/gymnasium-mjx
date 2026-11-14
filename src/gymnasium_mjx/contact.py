"""C MuJoCo's contacts for the capsule pairs, for MJX.

MJX (3.7 to 3.10) departs from C MuJoCo in three of the contact pairs
Gymnasium's models use; measured on random poses, every other pair they use
agrees to 1e-15:

- a capsule on a plane: with a pyramidal friction cone (Gymnasium's tasks all
  use one) the tangent frame's orientation changes the contact forces. Both
  take the first tangent along the capsule's axis projected on the plane, but
  MJX falls back to a fixed axis once that projection is shorter than 0.5
  (the capsule more than 60 degrees from the plane), while C MuJoCo keeps it
  unless it vanishes (shorter than mjMINVAL, 1e-15: then the x axis). Ant's
  legs often stand that steep, and one step moved velocities by up to 0.9.
- a sphere on a capsule, and two capsules: MJX's closest points on segments
  add 1e-6 to a denominator, which moves the contact's normal by up to 1e-4
  (sphere) and 7e-3 (capsules) on Humanoid-sized capsules. Humanoid's limbs
  touch each other, and one step then moved velocities by 2e-4.

This is a deliberate workaround on MJX internals: it swaps these entries of
MJX's collider table while a task traces MJX's forward or step, so other
users of MJX in the process are untouched. Remove it once MJX follows C.
"""

import contextlib
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import mujoco
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src.collision_primitive import _plane_sphere, _sphere_sphere, collider
from mujoco.mjx._src.collision_types import Collision, GeomInfo
from mujoco.mjx._src.types import GeomType


@collider(ncon=2)
def plane_capsule(plane: GeomInfo, capsule: GeomInfo) -> Collision:
    """MJX's plane_capsule, with C MuJoCo's tangent: two contacts, one at each
    end of the capsule's segment."""
    normal, axis = plane.mat[:, 2], capsule.mat[:, 2]
    tangent, length = mjx_math.normalize_with_norm(
        axis - normal * jnp.dot(normal, axis)
    )
    tangent = jnp.where(length < mujoco.mjMINVAL, jnp.array([1.0, 0.0, 0.0]), tangent)
    frame = jnp.array([[normal, tangent, jnp.cross(normal, tangent)]])
    segment = axis * capsule.size[1]
    contacts = [
        _plane_sphere(normal, plane.pos, capsule.pos + end, capsule.size[0])
        for end in (segment, -segment)
    ]
    distance = jnp.stack([contact[0] for contact in contacts])
    position = jnp.stack([contact[1] for contact in contacts])
    return distance, position, jnp.concatenate([frame, frame])


def closest_point_on_segment(
    centre: jax.Array, half_axis: jax.Array, point: jax.Array
) -> jax.Array:
    """The point of the segment centre +- half_axis nearest to point."""
    position = jnp.dot(point - centre, half_axis) / jnp.dot(half_axis, half_axis)
    return centre + jnp.clip(position, -1.0, 1.0) * half_axis


def closest_points_between_segments(
    centre1: jax.Array, half_axis1: jax.Array, centre2: jax.Array, half_axis2: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """The nearest pair of points of two segments, centre +- half_axis each:
    the lines' nearest pair, clamped to the first segment, projected on the
    second and clamped, then projected back on the first (Ericson, Real-Time
    Collision Detection, 5.1.9). Parallel segments take the first one's
    centre."""
    square1 = jnp.dot(half_axis1, half_axis1)
    square2 = jnp.dot(half_axis2, half_axis2)
    cross = jnp.dot(half_axis1, half_axis2)
    offset = centre1 - centre2
    along1, along2 = jnp.dot(half_axis1, offset), jnp.dot(half_axis2, offset)
    determinant = square1 * square2 - cross * cross
    parallel = determinant < mujoco.mjMINVAL
    position1 = jnp.where(
        parallel,
        0.0,
        (cross * along2 - square2 * along1) / jnp.where(parallel, 1.0, determinant),
    )
    position1 = jnp.clip(position1, -1.0, 1.0)
    position2 = jnp.clip((cross * position1 + along2) / square2, -1.0, 1.0)
    position1 = jnp.clip((cross * position2 - along1) / square1, -1.0, 1.0)
    return centre1 + position1 * half_axis1, centre2 + position2 * half_axis2


@collider(ncon=1)
def sphere_capsule(sphere: GeomInfo, capsule: GeomInfo) -> Collision:
    """MJX's sphere_capsule, with the exact nearest point of the segment."""
    half_axis = capsule.mat[:, 2] * capsule.size[1]
    nearest = closest_point_on_segment(capsule.pos, half_axis, sphere.pos)
    distance, position, normal = _sphere_sphere(
        sphere.pos, sphere.size[0], nearest, capsule.size[0]
    )
    return distance, position, mjx_math.make_frame(normal)


@collider(ncon=1)
def capsule_capsule(capsule1: GeomInfo, capsule2: GeomInfo) -> Collision:
    """MJX's capsule_capsule, with the exact nearest points of the segments."""
    nearest1, nearest2 = closest_points_between_segments(
        capsule1.pos,
        capsule1.mat[:, 2] * capsule1.size[1],
        capsule2.pos,
        capsule2.mat[:, 2] * capsule2.size[1],
    )
    distance, position, normal = _sphere_sphere(
        nearest1, capsule1.size[0], nearest2, capsule2.size[0]
    )
    return distance, position, mjx_math.make_frame(normal)


C_COLLIDERS = {
    (GeomType.PLANE, GeomType.CAPSULE): plane_capsule,
    (GeomType.SPHERE, GeomType.CAPSULE): sphere_capsule,
    (GeomType.CAPSULE, GeomType.CAPSULE): capsule_capsule,
}


@contextlib.contextmanager
def c_contacts() -> Iterator[None]:
    """MJX's colliders with C MuJoCo's capsule contacts, while tracing."""
    table = collision_driver._COLLISION_FUNC
    mjx_colliders = {pair: table[pair] for pair in C_COLLIDERS}
    table.update(C_COLLIDERS)
    try:
        yield
    finally:
        table.update(mjx_colliders)
