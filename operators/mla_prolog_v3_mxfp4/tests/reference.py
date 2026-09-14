"""Independent storage/math reference. No claim to emulate hardware FP4 rounding.

Sources: ops-nn@2a77283 QuantMatmulWeightNz FP4 example (low nibble first),
quant_matmul_v4_common.h A4W4 C0=64, Prolog RopeVFImpl and per-tile cache.
"""
import numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def unpack_codes(packed):
    p = np.asarray(packed, dtype=np.uint8)
    return np.stack((p & 15, p >> 4), axis=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)


def pack_codes(codes):
    c = np.asarray(codes)
    if c.shape[-1] % 2 or np.any(c > 15) or np.any(c < 0):
        raise ValueError('FP4 codes require an even last dimension and values 0..15')
    return (c[..., ::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def decode_e2m1(packed):
    c = unpack_codes(packed)
    return np.copysign(E2M1[c & 7], np.where(c & 8, -1.0, 1.0))


def decode_e8m0(scale):
    s = np.asarray(scale, dtype=np.uint8)
    result = np.exp2(s.astype(np.float64) - 127)
    result[s == 255] = np.nan
    return result


def dequant_rows(packed, scale):
    values = decode_e2m1(packed)
    s = np.asarray(scale, dtype=np.uint8)
    if values.shape[-1] != s.shape[-1] * 32 or values.shape[:-1] != s.shape[:-1]:
        raise ValueError('E8M0 scales must be [rows,K/32]')
    return values * np.repeat(decode_e8m0(s), 32, axis=-1)


def pack_nz_from_codes(kn):
    """Logical [K,N] nibble codes -> [N/64,K/16,16,32] bytes."""
    k, n = kn.shape
    if k % 64 or n % 64:
        raise ValueError('initial A4W4 NZ contract requires K,N multiples of 64')
    return pack_codes(np.asarray(kn).reshape(k // 16, 16, n // 64, 64).transpose(2, 0, 1, 3).copy())


def unpack_nz_codes(nz):
    if nz.ndim != 4 or nz.shape[-2:] != (16, 32):
        raise ValueError('expected [N/64,K/16,16,32] byte storage')
    c = unpack_codes(nz)
    return c.transpose(1, 2, 0, 3).reshape(nz.shape[1] * 16, nz.shape[0] * 64)


def prolog_to_matmul_scale(s):
    """[N,K/32] E8M0 bytes -> [K/64,N,2], preserving K32 ordering."""
    if s.ndim != 2 or s.shape[1] % 2:
        raise ValueError('scale requires paired K32 groups')
    return s.reshape(s.shape[0], -1, 2).transpose(1, 0, 2).copy()


def matmul_to_prolog_scale(s):
    if s.ndim != 3 or s.shape[2] != 2:
        raise ValueError('expected [K/64,N,2]')
    return s.transpose(1, 0, 2).reshape(s.shape[1], -1).copy()


def bf16_round(x):
    a = np.asarray(x, dtype=np.float32)
    bits = a.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def rmsnorm_bf16(x, gamma, eps=1e-5):
    a = np.asarray(x, dtype=np.float64)
    return bf16_round(a / np.sqrt(np.mean(a*a, axis=-1, keepdims=True) + eps) * gamma)


def prolog_rope(x, cos, signed_sin):
    """RopeVFImpl: interleaved input -> half-split output; sin=[-sin,+sin]."""
    even, odd = x[..., ::2], x[..., 1::2]
    h = x.shape[-1] // 2
    return np.concatenate((even*cos[..., :h] + odd*signed_sin[..., :h],
                           odd*cos[..., h:] + even*signed_sin[..., h:]), axis=-1)
