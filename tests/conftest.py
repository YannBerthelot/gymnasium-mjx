import os

# The tests are small: CPU only, and the same numbers on every machine.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
