import os

# The tests are small: CPU only, and the same numbers on every machine.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax


def x64(enabled):
    """jax.enable_x64 as a context. Older jax, down to the floor (0.5), has it
    only as jax.experimental.enable_x64."""
    context = getattr(jax, "enable_x64", None)
    if context is None:
        from jax.experimental import enable_x64 as context
    return context(enabled)
