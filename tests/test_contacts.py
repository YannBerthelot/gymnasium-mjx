"""The contacts a task's MJX step sees are C MuJoCo's.

Contract (gymnasium_mjx.contact): for each pair of geometries Gymnasium's
models put in contact, at the same poses, MJX under the package's colliders
finds C MuJoCo's contacts: the same distance, point and frame. The oracle is
C MuJoCo itself. Capsules within mjMINVAL of parallel are left out: there C
makes two contacts and the package one (README).
"""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from conftest import x64
from gymnasium_mjx.contact import c_contacts

SIZES = {"plane": "5 5 .1", "sphere": ".07", "capsule": ".05 .1"}
# Same arithmetic, different order: 1e-14 measured on these sizes.
TOLERANCE = 1e-12


def pair_model(first, second):
    body = '<body><freejoint/><geom type="{}" size="{}"/></body>'
    fixed = '<geom type="plane" size="{}"/>'
    first_xml = (fixed if first == "plane" else body).format(
        *([first, SIZES[first]] if first != "plane" else [SIZES[first]])
    )
    second_xml = body.format(second, SIZES[second])
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{first_xml}{second_xml}</worldbody></mujoco>"
    )


def random_poses(model, count):
    """Poses where the two geometries often touch: random orientations,
    centres within a few radii."""
    random = np.random.default_rng(0)
    poses = []
    for _ in range(count):
        qpos = np.zeros(model.nq)
        for start in range(0, model.nq, 7):
            rotation = random.normal(size=4)
            qpos[start : start + 7] = [*random.normal(size=3) * 0.08, *rotation]
            qpos[start + 3 : start + 7] /= np.linalg.norm(rotation)
        if model.nq == 14:
            qpos[:3] = 0
        poses.append(qpos)
    return poses


@pytest.mark.parametrize(
    "first, second",
    [("plane", "capsule"), ("sphere", "capsule"), ("capsule", "capsule")],
)
def test_contacts_are_c_mujocos(first, second):
    model = pair_model(first, second)
    compared = 0
    with x64(True):
        mjx_model = mjx.put_model(model)

        @jax.jit
        def forward(qpos):
            with c_contacts():
                data = mjx.make_data(model).replace(qpos=qpos)
                return mjx.forward(mjx_model, data)._impl.contact

        for qpos in random_poses(model, 200):
            reference = mujoco.MjData(model)
            reference.qpos[:] = qpos
            mujoco.mj_forward(model, reference)
            expected = [c for c in reference.contact[: reference.ncon] if c.dist < 0]
            contact = forward(jnp.asarray(qpos))
            touching = np.nonzero(np.asarray(contact.dist) < 0)[0]

            assert len(touching) == len(expected)
            for index, c in zip(touching, expected, strict=True):
                np.testing.assert_allclose(contact.dist[index], c.dist, atol=TOLERANCE)
                np.testing.assert_allclose(contact.pos[index], c.pos, atol=TOLERANCE)
                np.testing.assert_allclose(
                    contact.frame[index].ravel(), c.frame, atol=TOLERANCE
                )
            compared += len(expected)
    # The poses must produce contacts for the comparison to mean anything.
    assert compared > 50
