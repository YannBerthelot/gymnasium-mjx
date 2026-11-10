"""Gymnasium's v5 MuJoCo tasks, unchanged, stepped in JAX by MJX: make("<id>")
builds one, such as make("Hopper-v5"); TASKS lists them."""

from importlib.metadata import version

from gymnasium_mjx.base import GymnasiumTask, State
from gymnasium_mjx.tasks import (
    TASKS,
    HalfCheetah,
    Hopper,
    InvertedDoublePendulum,
    InvertedPendulum,
    Pusher,
    Reacher,
    Swimmer,
    Walker2d,
    make,
)

__version__ = version("gymnasium-mjx")

__all__ = [
    "TASKS",
    "GymnasiumTask",
    "HalfCheetah",
    "Hopper",
    "InvertedDoublePendulum",
    "InvertedPendulum",
    "Pusher",
    "Reacher",
    "State",
    "Swimmer",
    "Walker2d",
    "make",
]
