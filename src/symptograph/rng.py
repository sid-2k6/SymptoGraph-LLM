"""Capture and restore RNG state across a checkpoint, safely on GPU.

``torch.load(..., map_location="cuda")`` moves *every* tensor in the checkpoint
to the GPU, including the RNG state. ``torch.set_rng_state`` requires a CPU
``uint8`` tensor, so restoring a GPU-saved checkpoint raised::

    TypeError: RNG state must be a torch.ByteTensor

This only ever showed up on a GPU run; with ``map_location="cpu"`` the tensor
stays on the CPU and the bug is invisible. The helpers below always hand back a
CPU ``uint8`` tensor, and also carry the CUDA generator states so a resumed run
continues from the same stream of random numbers rather than a fresh one.
"""
from __future__ import annotations

import random

import numpy as np


def capture_rng_state() -> dict:
    import torch
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(value):
    """Coerce whatever came out of the checkpoint into a CPU uint8 tensor."""
    import torch
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().to(device="cpu", dtype=torch.uint8)


def restore_rng_state(state: dict | None) -> None:
    """Restore Python, NumPy and torch RNG state. Tolerates older checkpoints."""
    import torch
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])

    cpu_state = _as_cpu_byte_tensor(state.get("torch"))
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)

    cuda_states = state.get("cuda")
    if cuda_states and torch.cuda.is_available():
        coerced = [_as_cpu_byte_tensor(s) for s in cuda_states]
        if len(coerced) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(coerced)
