import os

try:
    import pynvml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pynvml = None
import pytest


def set_jax_cpu_backend_if_no_gpu() -> None:
    if pynvml is None:
        os.environ["JAX_PLATFORMS"] = "cpu"
        return
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
