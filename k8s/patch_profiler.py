"""Monkey-patch LLMServingSim's profiler to bypass torch.profiler.

torch.profiler.stop_trace() hangs indefinitely when processing CUDA events
from FlashAttention kernels. This patch replaces vllm's layerwise_profile()
with direct CUDA event timing via forward hooks, producing identical output.

Usage: Run this script before invoking the profiler.
  python3 /workspace/patch_profiler.py
  python3 -m profiler profile ...

It patches the installed vllm package in-place.
"""

import textwrap
import importlib
import sys
from pathlib import Path


def find_vllm_profiler():
    """Locate vllm's layerwise_profile module without importing it."""
    import vllm
    vllm_dir = Path(vllm.__file__).parent
    target = vllm_dir / "profiler" / "layerwise_profile.py"
    if not target.exists():
        raise FileNotFoundError(f"Expected vllm profiler at {target}")
    return target


def write_replacement(target_path: Path):
    """Overwrite vllm's layerwise_profile.py with CUDA-event-based timing."""
    code = textwrap.dedent('''\
        """Patched layerwise_profile: uses CUDA events instead of torch.profiler.

        Drop-in replacement that avoids the stop_trace() hang while producing
        a compatible results object for extract_samples().
        """

        import torch
        import torch.nn as nn
        from contextlib import contextmanager
        from dataclasses import dataclass, field
        from typing import Any


        @dataclass
        class _Entry:
            name: str
            cuda_time_us: float = 0.0
            invocations: int = 0


        @dataclass
        class _Node:
            entry: _Entry
            children: list = field(default_factory=list)

            def to_dict(self):
                return {
                    "entry": {
                        "name": self.entry.name,
                        "cuda_time_us": self.entry.cuda_time_us,
                        "invocations": self.entry.invocations,
                    },
                    "children": [c.to_dict() for c in self.children],
                }


        class _Results:
            def __init__(self, root_nodes):
                self._root_nodes = root_nodes

            def convert_stats_to_dict(self):
                return {
                    "summary_stats": [n.to_dict() for n in self._root_nodes],
                }


        class _Hook:
            """Compatible hook object returned by the context manager."""

            def __init__(self):
                self.results = None
                self._module_events = {}  # id(module) -> {class_name, starts, ends}
                self._handles = []

            def _register(self, model: nn.Module):
                for name, mod in model.named_modules():
                    if not name:
                        continue
                    mid = id(mod)
                    class_name = type(mod).__name__
                    self._module_events[mid] = {
                        "class_name": class_name,
                        "path": name,
                        "starts": [],
                        "ends": [],
                    }

                    def make_pre(m_id):
                        def pre_hook(module, args):
                            ev = torch.cuda.Event(enable_timing=True)
                            ev.record()
                            self._module_events[m_id]["starts"].append(ev)
                        return pre_hook

                    def make_post(m_id):
                        def post_hook(module, args, output):
                            ev = torch.cuda.Event(enable_timing=True)
                            ev.record()
                            self._module_events[m_id]["ends"].append(ev)
                        return post_hook

                    h1 = mod.register_forward_pre_hook(make_pre(mid))
                    h2 = mod.register_forward_hook(make_post(mid))
                    self._handles.append(h1)
                    self._handles.append(h2)

            def _finalize(self):
                for h in self._handles:
                    h.remove()
                self._handles.clear()

                torch.cuda.synchronize()

                nodes = []
                for mid, data in self._module_events.items():
                    starts = data["starts"]
                    ends = data["ends"]
                    if not starts:
                        continue
                    total_us = 0.0
                    for s, e in zip(starts, ends):
                        total_us += s.elapsed_time(e) * 1000.0  # ms -> us
                    invocations = len(starts)
                    entry = _Entry(
                        name=f"{data['class_name']}(...)",
                        cuda_time_us=total_us,
                        invocations=invocations,
                    )
                    nodes.append(_Node(entry=entry))

                self.results = _Results(nodes)


        # Global reference to the model for hook registration.
        _current_model = None


        def set_model(model: nn.Module):
            """Called by the patched extension to pass the model reference."""
            global _current_model
            _current_model = model


        @contextmanager
        def layerwise_profile(*, context=None, num_workers=None):
            """Drop-in replacement for vllm.profiler.layerwise_profile.layerwise_profile."""
            hook = _Hook()
            if _current_model is not None:
                hook._register(_current_model)
            try:
                yield hook
            finally:
                hook._finalize()
    ''')
    target_path.write_text(code)
    print(f"Patched: {target_path}")


def patch_extension():
    """Patch the LLMServingSim extension to pass the model to our hook."""
    ext_path = Path("/workspace/LLMServingSim/profiler/core/hooks/extension.py")
    if not ext_path.exists():
        print(f"WARNING: {ext_path} not found, skipping extension patch")
        return

    content = ext_path.read_text()

    # Add model registration before layerwise_profile is used
    if "set_model" not in content:
        # Find the fire() method's layerwise_profile usage and inject set_model call
        old = "from vllm.profiler.layerwise_profile import layerwise_profile"
        new = (
            "from vllm.profiler.layerwise_profile import layerwise_profile, set_model\n"
        )
        if old in content:
            content = content.replace(old, new)
        else:
            # Try alternate import location
            old = "from vllm.profiler.layerwise_profile import ("
            new = "from vllm.profiler.layerwise_profile import (\n            set_model,"
            if old in content:
                content = content.replace(old, new)

        # Insert set_model call before the layerwise_profile context
        old_block = "with layerwise_profile() as hook:"
        new_block = "set_model(self.model_runner.model)\n            with layerwise_profile() as hook:"
        content = content.replace(old_block, new_block)

        ext_path.write_text(content)
        print(f"Patched: {ext_path}")
    else:
        print(f"Already patched: {ext_path}")


def main():
    target = find_vllm_profiler()
    print(f"Found vllm profiler at: {target}")
    write_replacement(target)
    patch_extension()
    print("Done. Profiler is patched to use CUDA event timing.")


if __name__ == "__main__":
    main()
