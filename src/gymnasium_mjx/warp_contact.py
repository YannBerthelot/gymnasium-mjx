"""C MuJoCo's capsule-on-plane tangent for MuJoCo Warp.

A deliberate workaround, as contact.py is for MJX's JAX physics. MuJoCo
Warp's plane_capsule (mujoco_warp/_src/collision_primitive_core.py) takes the
first contact tangent along the capsule's axis projected on the plane, as C
MuJoCo does, but falls back to a fixed axis once that projection is shorter
than 0.5 (the capsule more than 60 degrees from the plane); C MuJoCo keeps it
unless it vanishes (shorter than mjMINVAL: then the x axis). With a pyramidal
friction cone the tangent turns the contact forces: Ant's legs stand that
steep, and one step on MuJoCo Warp 3.15 moved Ant's velocities by up to 1.4.
The same bug in MJX's JAX physics is MuJoCo PR #3863.

The narrowphase kernel calls plane_capsule by its name in Warp's
collision_primitive module, resolved when Warp builds the kernel, so this
rebinds that name before the first build. Warp keys its compiled kernels by
the source of the functions they call, so neither the patched nor the
original kernel is reused for the other. Applied only on the MuJoCo versions
checked against C MuJoCo (CHECKED_RELEASES); remove it once MuJoCo Warp
follows C.
"""

import warnings

import mujoco

#: MuJoCo releases (major, minor) on which MuJoCo Warp's plane_capsule was
#: read and the patched one checked against C MuJoCo (Ant's one-step tests
#: on an L40S). mj_version() is 1,000,000 x major + 1,000 x minor + patch.
CHECKED_RELEASES = ((3, 14), (3, 15))

_patched = False


def patch_warp_plane_capsule() -> bool:
    """Gives MuJoCo Warp's narrowphase C MuJoCo's capsule-on-plane tangent.
    Returns whether it holds; False, with a warning, on an unchecked MuJoCo
    version or once Warp has already built its capsule-on-plane kernel."""
    global _patched
    if _patched:
        return True
    version = mujoco.mj_version()
    if (version // 1_000_000, version // 1_000 % 1_000) not in CHECKED_RELEASES:
        warnings.warn(
            f"gymnasium-mjx's capsule-on-plane fix for MuJoCo Warp is off: it"
            f" was checked on MuJoCo 3.14 and 3.15, not {mujoco.__version__}.",
            stacklevel=3,
        )
        return False

    import warp as wp
    from mujoco.mjx.third_party.mujoco_warp._src import (
        collision_primitive,
        collision_primitive_core,
    )
    from mujoco.mjx.third_party.mujoco_warp._src.math import normalize_with_norm
    from mujoco.mjx.third_party.mujoco_warp._src.types import MJ_MINVAL, GeomType

    pair = (GeomType.PLANE, GeomType.CAPSULE)
    if pair in collision_primitive._PRIMITIVE_COLLISION_TYPES:
        # Its kernel is built and cached in this process with the original.
        warnings.warn(
            "gymnasium-mjx's capsule-on-plane fix for MuJoCo Warp is off: Warp"
            " already stepped a capsule on a plane in this process. Make the"
            " task before any other MuJoCo Warp model steps.",
            stacklevel=3,
        )
        return False
    plane_sphere = collision_primitive_core.plane_sphere
    mat23f = collision_primitive_core.mat23f

    @wp.func
    def plane_capsule(
        plane_normal: wp.vec3,
        plane_pos: wp.vec3,
        capsule_pos: wp.vec3,
        capsule_axis: wp.vec3,
        capsule_radius: float,
        capsule_half_length: float,
    ):
        n = plane_normal
        axis = capsule_axis
        b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))
        if b_norm < MJ_MINVAL:  # C MuJoCo: mju_normalize3's x axis
            b = wp.vec3(1.0, 0.0, 0.0)
        c = wp.cross(n, b)
        frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])  # type: ignore[index]  # Warp kernel code
        segment = axis * capsule_half_length
        dist1, pos1 = plane_sphere(n, plane_pos, capsule_pos + segment, capsule_radius)
        dist2, pos2 = plane_sphere(n, plane_pos, capsule_pos - segment, capsule_radius)
        dist = wp.vec2(dist1, dist2)
        pos = mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2])
        return dist, pos, frame

    collision_primitive.plane_capsule = plane_capsule
    _patched = True
    return True
