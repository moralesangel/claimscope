"""Pure-Python XXH3-128 shim.

Why this exists
---------------
Smart App Control on this machine blocks unsigned native DLLs, which includes the
compiled ``_xxhash`` extension shipped in every PyPI ``xxhash`` wheel. LangGraph
imports ``xxh3_128_hexdigest`` at module scope (``langgraph/types.py``,
``langgraph/pregel/_algo.py``), so the whole graph engine fails to import without it.

The dependency tree uses exactly two names -- ``xxh3_128`` and ``xxh3_128_hexdigest``
-- and only to derive deterministic task ids. This module implements the real XXH3-128
algorithm in pure Python so those digests match the reference implementation: a
checkpoint written here stays readable by a normal ``xxhash`` install elsewhere.

It is a development-environment workaround, not a dependency of ClaimScope itself.
Speed is irrelevant here: hashes are short task-id strings, not bulk data.
"""

from __future__ import annotations

import struct

__all__ = ["VERSION", "xxh3_128", "xxh3_128_digest", "xxh3_128_hexdigest", "xxh3_128_intdigest"]

VERSION = "0.0.0-pure"

_MASK64 = 0xFFFFFFFFFFFFFFFF
_PRIME32_1 = 0x9E3779B1
_PRIME32_2 = 0x85EBCA77
_PRIME32_3 = 0xC2B2AE3D
_PRIME64_1 = 0x9E3779B185EBCA87
_PRIME64_2 = 0xC2B2AE3D27D4EB4F
_PRIME64_3 = 0x165667B19E3779F9
_PRIME64_4 = 0x85EBCA77C2B2AE63
_PRIME64_5 = 0x27D4EB2F165667C5

_SECRET = bytes.fromhex(
    "b8fe6c3923a44bbe7c01812cf721ad1cded46de9839097db7240a4a4b7b3671f"
    "cb79e64eccc0e578825ad07dccff7221b8084674f743248ee03590e6813a264c"
    "3c2852bb91c300cb88d0658b1b532ea371644897a20df94e3819ef46a9deacd8"
    "a8fa763fe39c343ff9dcbbc7c70b4f1d8a51e04bcdb45931c89f7ec9d9787364"
    "eac5ac8334d3ebc3c581a0fffa1363eb170ddd51b7f0da49d316552629d4689e"
    "2b16be587d47a1fc8ff8b8d17ad031ce45cb3a8f95160428afd7fbcabb4b407e"
)

_STRIPE_LEN = 64
_SECRET_CONSUME_RATE = 8
_ACC_NB = 8
_INIT_ACC = (
    _PRIME32_3,
    _PRIME64_1,
    _PRIME64_2,
    _PRIME64_3,
    _PRIME64_4,
    _PRIME32_2,
    _PRIME64_5,
    _PRIME32_1,
)


def _u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _swap64(x: int) -> int:
    return int.from_bytes((x & _MASK64).to_bytes(8, "little"), "big")


def _swap32(x: int) -> int:
    return int.from_bytes((x & 0xFFFFFFFF).to_bytes(4, "little"), "big")


def _mul128_fold64(a: int, b: int) -> int:
    product = (a & _MASK64) * (b & _MASK64)
    return (product & _MASK64) ^ ((product >> 64) & _MASK64)


def _xorshift64(value: int, shift: int) -> int:
    value &= _MASK64
    return value ^ (value >> shift)


def _avalanche(h: int) -> int:
    """XXH3_avalanche."""
    h = _xorshift64(h, 37)
    h = (h * 0x165667919E3779F9) & _MASK64
    return _xorshift64(h, 32)


def _xxh64_avalanche(h: int) -> int:
    """XXH64_avalanche -- a different finalizer, used by the short-input paths."""
    h &= _MASK64
    h ^= h >> 33
    h = (h * _PRIME64_2) & _MASK64
    h ^= h >> 29
    h = (h * _PRIME64_3) & _MASK64
    return h ^ (h >> 32)


def _rrmxmx(h: int, length: int) -> int:
    h &= _MASK64
    h ^= ((h << 49) | (h >> 15)) ^ ((h << 24) | (h >> 40))
    h &= _MASK64
    h = (h * 0x9FB21C651E98DF25) & _MASK64
    h ^= (h >> 35) + length
    h &= _MASK64
    h = (h * 0x9FB21C651E98DF25) & _MASK64
    return _xorshift64(h, 28)


def _len_1to3_128b(data: bytes, seed: int) -> tuple[int, int]:
    length = len(data)
    c1, c2, c3 = data[0], data[length >> 1], data[-1]
    combinedl = (c1 << 16) | (c2 << 24) | c3 | (length << 8)
    # XXH_rotl32(XXH_swap32(combinedl), 13)
    swapped = _swap32(combinedl)
    combinedh = ((swapped << 13) | (swapped >> 19)) & 0xFFFFFFFF
    bitflipl = ((_u32(_SECRET, 0) ^ _u32(_SECRET, 4)) + seed) & _MASK64
    bitfliph = ((_u32(_SECRET, 8) ^ _u32(_SECRET, 12)) - seed) & _MASK64
    return _xxh64_avalanche(combinedl ^ bitflipl), _xxh64_avalanche(combinedh ^ bitfliph)


def _len_4to8_128b(data: bytes, seed: int) -> tuple[int, int]:
    length = len(data)
    seed ^= (_swap32(seed & 0xFFFFFFFF) << 32) & _MASK64
    input_lo = _u32(data, 0)
    input_hi = _u32(data, length - 4)
    input_64 = (input_lo + (input_hi << 32)) & _MASK64
    bitflip = ((_u64(_SECRET, 16) ^ _u64(_SECRET, 24)) + seed) & _MASK64
    keyed = input_64 ^ bitflip

    product = keyed * (_PRIME64_1 + (length << 2))
    m128_lo = product & _MASK64
    m128_hi = (product >> 64) & _MASK64

    m128_hi = (m128_hi + (m128_lo << 1)) & _MASK64
    m128_lo ^= m128_hi >> 3

    m128_lo = _xorshift64(m128_lo, 35)
    m128_lo = (m128_lo * 0x9FB21C651E98DF25) & _MASK64
    m128_lo = _xorshift64(m128_lo, 28)
    return m128_lo, _avalanche(m128_hi)


def _len_9to16_128b(data: bytes, seed: int) -> tuple[int, int]:
    length = len(data)
    bitflipl = ((_u64(_SECRET, 32) ^ _u64(_SECRET, 40)) - seed) & _MASK64
    bitfliph = ((_u64(_SECRET, 48) ^ _u64(_SECRET, 56)) + seed) & _MASK64
    input_lo = _u64(data, 0)
    input_hi = _u64(data, length - 8)

    product = (input_lo ^ input_hi ^ bitflipl) * _PRIME64_1
    m128_lo = product & _MASK64
    m128_hi = (product >> 64) & _MASK64

    m128_lo = (m128_lo + ((length - 1) << 54)) & _MASK64
    input_hi ^= bitfliph

    m128_hi = (m128_hi + input_hi + (input_hi & 0xFFFFFFFF) * (_PRIME32_2 - 1)) & _MASK64

    m128_lo ^= _swap64(m128_hi)

    product2 = m128_lo * _PRIME64_2
    h128_lo = product2 & _MASK64
    h128_hi = (product2 >> 64) & _MASK64
    h128_hi = (h128_hi + (m128_hi * _PRIME64_2)) & _MASK64

    return _avalanche(h128_lo), _avalanche(h128_hi)


def _len_0to16_128b(data: bytes, seed: int) -> tuple[int, int]:
    length = len(data)
    if length > 8:
        return _len_9to16_128b(data, seed)
    if length >= 4:
        return _len_4to8_128b(data, seed)
    if length:
        return _len_1to3_128b(data, seed)
    bitflipl = _u64(_SECRET, 64) ^ _u64(_SECRET, 72)
    bitfliph = _u64(_SECRET, 80) ^ _u64(_SECRET, 88)
    return _xxh64_avalanche(seed ^ bitflipl), _xxh64_avalanche(seed ^ bitfliph)


def _mix16b(data: bytes, offset: int, secret: bytes, secret_offset: int, seed: int) -> int:
    input_lo = _u64(data, offset)
    input_hi = _u64(data, offset + 8)
    return _mul128_fold64(
        input_lo ^ ((_u64(secret, secret_offset) + seed) & _MASK64),
        input_hi ^ ((_u64(secret, secret_offset + 8) - seed) & _MASK64),
    )


def _mix32b(
    acc: tuple[int, int],
    data: bytes,
    off1: int,
    off2: int,
    secret: bytes,
    secret_offset: int,
    seed: int,
) -> tuple[int, int]:
    acc_lo, acc_hi = acc
    acc_lo = (acc_lo + _mix16b(data, off1, secret, secret_offset, seed)) & _MASK64
    acc_lo ^= (_u64(data, off2) + _u64(data, off2 + 8)) & _MASK64
    acc_hi = (acc_hi + _mix16b(data, off2, secret, secret_offset + 16, seed)) & _MASK64
    acc_hi ^= (_u64(data, off1) + _u64(data, off1 + 8)) & _MASK64
    return acc_lo, acc_hi


def _len_17to128_128b(data: bytes, seed: int) -> tuple[int, int]:
    length = len(data)
    acc = ((length * _PRIME64_1) & _MASK64, 0)
    nb_rounds = (length - 1) // 32

    for i in range(nb_rounds, -1, -1):
        acc = _mix32b(acc, data, 16 * i, length - 16 * (i + 1), _SECRET, 32 * i, seed)

    acc_lo, acc_hi = acc
    h128_lo = (acc_lo + acc_hi) & _MASK64
    h128_hi = (
        (acc_lo * _PRIME64_1) + (acc_hi * _PRIME64_4) + ((length - seed) * _PRIME64_2)
    ) & _MASK64
    return _avalanche(h128_lo), (-_avalanche(h128_hi)) & _MASK64


def _len_129to240_128b(data: bytes, seed: int) -> tuple[int, int]:
    length = len(data)
    acc_lo = (length * _PRIME64_1) & _MASK64
    acc_hi = 0

    for i in range(4):
        acc_lo, acc_hi = _mix32b((acc_lo, acc_hi), data, 32 * i, 32 * i + 16, _SECRET, 32 * i, seed)
    acc_lo = _avalanche(acc_lo)
    acc_hi = _avalanche(acc_hi)

    nb_rounds = length // 32
    for i in range(4, nb_rounds):
        acc_lo, acc_hi = _mix32b(
            (acc_lo, acc_hi), data, 32 * i, 32 * i + 16, _SECRET, 3 + 32 * (i - 4), seed
        )

    acc_lo, acc_hi = _mix32b(
        (acc_lo, acc_hi),
        data,
        length - 16,
        length - 32,
        _SECRET,
        136 - 17 - 16,
        (-seed) & _MASK64,
    )

    h128_lo = (acc_lo + acc_hi) & _MASK64
    h128_hi = (
        (acc_lo * _PRIME64_1) + (acc_hi * _PRIME64_4) + ((length - seed) * _PRIME64_2)
    ) & _MASK64
    return _avalanche(h128_lo), (-_avalanche(h128_hi)) & _MASK64


def _accumulate_512(acc: list[int], data: bytes, offset: int, secret: bytes, sec_off: int) -> None:
    for i in range(_ACC_NB):
        data_val = _u64(data, offset + 8 * i)
        data_key = data_val ^ _u64(secret, sec_off + 8 * i)
        acc[i ^ 1] = (acc[i ^ 1] + data_val) & _MASK64
        acc[i] = (acc[i] + (data_key & 0xFFFFFFFF) * (data_key >> 32)) & _MASK64


def _scramble_acc(acc: list[int], secret: bytes, sec_off: int) -> None:
    for i in range(_ACC_NB):
        acc[i] = _xorshift64(acc[i], 47)
        acc[i] ^= _u64(secret, sec_off + 8 * i)
        acc[i] = (acc[i] * _PRIME32_1) & _MASK64


def _init_custom_secret(seed: int) -> bytes:
    """XXH3_initCustomSecret: a non-zero seed regenerates the secret for long inputs."""
    if seed == 0:
        return _SECRET
    out = bytearray(len(_SECRET))
    for i in range(len(_SECRET) // 16):
        lo = (_u64(_SECRET, i * 16) + seed) & _MASK64
        hi = (_u64(_SECRET, i * 16 + 8) - seed) & _MASK64
        struct.pack_into("<Q", out, i * 16, lo)
        struct.pack_into("<Q", out, i * 16 + 8, hi)
    return bytes(out)


def _hash_long_internal(data: bytes, secret: bytes) -> list[int]:
    length = len(data)
    acc = list(_INIT_ACC)
    secret_len = len(secret)
    nb_stripes_per_block = (secret_len - _STRIPE_LEN) // _SECRET_CONSUME_RATE
    block_len = _STRIPE_LEN * nb_stripes_per_block
    nb_blocks = (length - 1) // block_len

    for n in range(nb_blocks):
        for i in range(nb_stripes_per_block):
            _accumulate_512(
                acc, data, n * block_len + i * _STRIPE_LEN, secret, i * _SECRET_CONSUME_RATE
            )
        _scramble_acc(acc, secret, secret_len - _STRIPE_LEN)

    nb_stripes = ((length - 1) - block_len * nb_blocks) // _STRIPE_LEN
    for i in range(nb_stripes):
        _accumulate_512(
            acc, data, nb_blocks * block_len + i * _STRIPE_LEN, secret, i * _SECRET_CONSUME_RATE
        )

    _accumulate_512(acc, data, length - _STRIPE_LEN, secret, secret_len - _STRIPE_LEN - 7)
    return acc


def _merge_accs(acc: list[int], secret: bytes, sec_off: int, start: int) -> int:
    result = start & _MASK64
    for i in range(4):
        result = (
            result
            + _mul128_fold64(
                acc[2 * i] ^ _u64(secret, sec_off + 16 * i),
                acc[2 * i + 1] ^ _u64(secret, sec_off + 16 * i + 8),
            )
        ) & _MASK64
    return _avalanche(result)


def _hash_long_128b(data: bytes, seed: int) -> tuple[int, int]:
    length = len(data)
    secret = _init_custom_secret(seed)
    acc = _hash_long_internal(data, secret)
    secret_len = len(secret)
    h_lo = _merge_accs(acc, secret, 11, (length * _PRIME64_1) & _MASK64)
    h_hi = _merge_accs(
        acc,
        secret,
        secret_len - _STRIPE_LEN - 11,
        (~((length * _PRIME64_2) & _MASK64)) & _MASK64,
    )
    return h_lo, h_hi


def _xxh3_128(data: bytes, seed: int = 0) -> int:
    length = len(data)
    if length <= 16:
        lo, hi = _len_0to16_128b(data, seed)
    elif length <= 128:
        lo, hi = _len_17to128_128b(data, seed)
    elif length <= 240:
        lo, hi = _len_129to240_128b(data, seed)
    else:
        lo, hi = _hash_long_128b(data, seed)
    return (hi << 64) | lo


def _coerce(data: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(data, str):
        raise TypeError("Unicode-objects must be encoded before hashing")
    return bytes(data)


class xxh3_128:  # lowercase name mirrors the upstream xxhash package
    """Incremental XXH3-128 hasher, matching the ``xxhash`` package's interface."""

    digest_size = 16
    block_size = 64
    name = "xxh3_128"

    def __init__(self, data: bytes | bytearray | memoryview | str = b"", seed: int = 0) -> None:
        self._buffer = bytearray()
        self._seed = seed
        if data:
            self.update(data)

    def update(self, data: bytes | bytearray | memoryview | str) -> None:
        self._buffer.extend(_coerce(data))

    def digest(self) -> bytes:
        return _xxh3_128(bytes(self._buffer), self._seed).to_bytes(16, "big")

    def hexdigest(self) -> str:
        return self.digest().hex()

    def intdigest(self) -> int:
        return _xxh3_128(bytes(self._buffer), self._seed)

    def copy(self) -> xxh3_128:
        clone = xxh3_128(seed=self._seed)
        clone._buffer = bytearray(self._buffer)
        return clone

    def reset(self) -> None:
        self._buffer = bytearray()

    @property
    def seed(self) -> int:
        return self._seed


def xxh3_128_digest(data: bytes | bytearray | memoryview | str, seed: int = 0) -> bytes:
    return _xxh3_128(_coerce(data), seed).to_bytes(16, "big")


def xxh3_128_hexdigest(data: bytes | bytearray | memoryview | str, seed: int = 0) -> str:
    return xxh3_128_digest(data, seed).hex()


def xxh3_128_intdigest(data: bytes | bytearray | memoryview | str, seed: int = 0) -> int:
    return _xxh3_128(_coerce(data), seed)
