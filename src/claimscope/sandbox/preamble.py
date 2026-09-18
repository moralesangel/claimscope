"""Code injected ahead of a generated experiment to restrict what it can do.

This runs inside the child process, before the experiment's own code. It is a
best-effort barrier, not a security boundary: anything running in the same
process can undo it. It exists to stop *accidental* network access -- a
generated script calling ``download=True`` -- not to contain hostile code.

For that reason the subprocess backend is opt-in and says so loudly. Docker
remains the only backend that actually contains what it runs.
"""

from __future__ import annotations

# Written to a file and passed to python via -X importtime style preloading.
# Kept as a string rather than imported so nothing in the parent process is
# affected by it.
NETWORK_BLOCK_PREAMBLE = '''\
"""Injected by ClaimScope. Blocks outbound network access in this process."""

import socket as _socket


class _NetworkBlocked(OSError):
    """Raised instead of opening a connection."""


def _blocked(*_args, **_kwargs):
    raise _NetworkBlocked(
        "Network access is disabled in the ClaimScope sandbox. "
        "Generate data locally or use a dataset bundled in an installed package."
    )


# Cover the paths a training script realistically takes: direct sockets, the
# convenience helper, and name resolution, so a hostname fails fast and clearly.
_socket.socket.connect = _blocked
_socket.socket.connect_ex = _blocked
_socket.create_connection = _blocked
_socket.getaddrinfo = _blocked
_socket.gethostbyname = _blocked

try:
    import urllib.request as _urllib_request

    _urllib_request.urlopen = _blocked
    _urllib_request.urlretrieve = _blocked
except ImportError:
    pass
'''


RESOURCE_LIMIT_PREAMBLE = '''\
"""Injected by ClaimScope. Caps memory, CPU time and process count."""

try:
    import resource as _resource

    _MEMORY_BYTES = {memory_bytes}
    _CPU_SECONDS = {cpu_seconds}

    # Address space, so a runaway allocation fails rather than swapping the host.
    _resource.setrlimit(_resource.RLIMIT_AS, (_MEMORY_BYTES, _MEMORY_BYTES))
    # CPU time, as a backstop to the wall-clock timeout the parent enforces.
    _resource.setrlimit(_resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
    # No forking a process tree out of the sandbox.
    _resource.setrlimit(_resource.RLIMIT_NPROC, (64, 64))
except (ImportError, ValueError, OSError):
    # Not available on Windows, and some limits are refused in containers.
    # The parent's timeout still applies.
    pass
'''


def build_preamble(memory_mb: int, cpu_seconds: int) -> str:
    """The full preamble for one sandboxed run."""
    return "\n".join(
        [
            NETWORK_BLOCK_PREAMBLE,
            RESOURCE_LIMIT_PREAMBLE.format(
                memory_bytes=memory_mb * 1024 * 1024,
                cpu_seconds=cpu_seconds,
            ),
        ]
    )
