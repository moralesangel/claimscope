"""Regression tests for the pure-Python xxhash shim (see vendor/xxhash_pure/).

The shim only exists because Smart App Control blocks the native wheel on this
machine, but LangGraph derives task ids from it, so a wrong digest would silently
corrupt checkpoint identity. These vectors are sampled from the official xxHash
sanity suite and cover every length-dependent code path in XXH3-128.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from xxhash import xxh3_128, xxh3_128_hexdigest

VECTORS = json.loads(
    (pathlib.Path(__file__).parent / "data" / "xxh3_128_vectors.json").read_text(encoding="utf-8")
)

PRIME32 = 2654435761
PRIME64 = 11400714785074694797
MASK64 = 0xFFFFFFFFFFFFFFFF


def _sanity_buffer(size: int) -> bytes:
    """Reproduce XSUM_fillTestBuffer from the xxHash reference suite."""
    buf = bytearray(size)
    byte_gen = PRIME32
    for i in range(size):
        buf[i] = (byte_gen >> 56) & 0xFF
        byte_gen = (byte_gen * PRIME64) & MASK64
    return bytes(buf)


BUFFER = _sanity_buffer(max(v["len"] for v in VECTORS))


@pytest.mark.parametrize(
    "vector",
    VECTORS,
    ids=lambda v: f"len{v['len']}_seed{v['seed']:x}",
)
def test_matches_reference_vector(vector: dict[str, int]) -> None:
    expected = f"{vector['hi']:016x}{vector['lo']:016x}"
    assert xxh3_128_hexdigest(BUFFER[: vector["len"]], vector["seed"]) == expected


def test_incremental_matches_oneshot() -> None:
    data = BUFFER[:500]
    hasher = xxh3_128()
    for start in range(0, len(data), 37):
        hasher.update(data[start : start + 37])
    assert hasher.hexdigest() == xxh3_128_hexdigest(data)


def test_digest_is_16_bytes() -> None:
    assert len(xxh3_128(b"abc").digest()) == 16


def test_rejects_str_like_the_native_package() -> None:
    with pytest.raises(TypeError):
        xxh3_128_hexdigest("abc")  # type: ignore[arg-type]
