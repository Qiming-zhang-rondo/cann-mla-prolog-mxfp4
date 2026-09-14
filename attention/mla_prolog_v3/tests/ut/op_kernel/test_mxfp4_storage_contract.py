"""CPU storage-model checks; these do not compile or execute the Ascend C kernel.

Run: python3 attention/mla_prolog_v3/tests/ut/op_kernel/test_mxfp4_storage_contract.py
"""

import random
import unittest


def encode_weight_nz(nd, k, n):
    """Independent reshape/transpose of packed ND into [N/64,K/16,16,32B]."""
    return b"".join(
        nd[row * (n // 2) + nb * 32:row * (n // 2) + (nb + 1) * 32]
        for nb in range(n // 64)
        for kb in range(k // 16)
        for row in range(kb * 16, (kb + 1) * 16)
    )


def dma_weight_tile(nz, k, n_offset, k_offset, n_size, k_size):
    """Model the actual byte DMA blockCount/blockLen/srcStride fields."""
    start = (k * n_offset + k_offset * 64) // 2
    block_bytes = k_size * 32
    return b"".join(
        nz[start + block * k * 32:start + block * k * 32 + block_bytes]
        for block in range(n_size // 64)
    )


class Mxfp4StorageContractTest(unittest.TestCase):
    def test_nz_cross_k_and_n_slices(self):
        rng = random.Random(3510)
        k, n = 512, 384
        nd = rng.randbytes(k * n // 2)
        nz = encode_weight_nz(nd, k, n)
        for n_size in (64, 128):
            for n_off in range(0, n - n_size + 1, 64):
                for k_size in (128, 256):
                    for k_off in range(0, k, k_size):
                        # Build the expected smaller logical matrix from ND rows,
                        # then format-convert it independently from the DMA model.
                        expected_nd = b"".join(
                            nd[row * n // 2 + n_off // 2:row * n // 2 + (n_off + n_size) // 2]
                            for row in range(k_off, k_off + k_size)
                        )
                        self.assertEqual(
                            dma_weight_tile(nz, k, n_off, k_off, n_size, k_size),
                            encode_weight_nz(expected_nd, k_size, n_size),
                        )

    def test_scale_pairs_are_independent_of_data_nibbles(self):
        # Distinguish every row and K32 group, including second N and K tiles.
        rows, full_groups, groups = 128, 192, 8
        for row_start in (0, 64):
            for group_start in (0, 8, 184):
                nd = [[(r, g) for g in range(full_groups)] for r in range(rows)]
                tile = [row[group_start:group_start + groups] for row in nd[row_start:row_start + 64]]
                packed_pairs = [
                    tile[r][g]
                    for rb in range(4)
                    for gp in range(groups // 2)
                    for r in range(rb * 16, (rb + 1) * 16)
                    for g in (gp * 2, gp * 2 + 1)
                ]
                for r in range(64):
                    for g in range(groups):
                        index = ((r // 16) * (groups // 2) + g // 2) * 32 + (r % 16) * 2 + g % 2
                        self.assertEqual(packed_pairs[index], (row_start + r, group_start + g))

    def test_glm_tensor_offsets_are_logical_before_byte_boundary(self):
        # The API/typed GlobalTensor side counts FP4 values. The helper converts
        # once to bytes; multiplying the half-byte factor twice would fail here.
        he, hcq = 6144, 2048
        for token in (0, 1, 15, 63, 127):
            self.assertEqual(token * he // 2, token * 3072)
            self.assertEqual(token * hcq // 2, token * 1024)
        for heads in (1, 2, 4, 8, 16, 32, 64):
            n = heads * (192 + 64)
            self.assertEqual((n // 64) * (hcq // 16) * 16 * 32, hcq * n // 2)

    def test_l1_l0_and_workspace_bounds(self):
        for m in (1, 15, 16, 17, 64, 128):
            aligned_m = (m + 15) // 16 * 16
            for n in (64, 128):
                for k in (128, 256):
                    self.assertLessEqual(aligned_m * k // 2, 32 * 1024)  # L0A
                    self.assertLessEqual(n * k // 2, 32 * 1024)  # L0B
                    self.assertLessEqual(aligned_m * n * 4, 64 * 1024)  # L0C
                    self.assertLessEqual(n * k // 2, 64 * 1024)  # L1B data half
                    self.assertLessEqual(64 * 1024 + (aligned_m + n) * (k // 32), 128 * 1024)
            for hcq in (1536, 2048):
                for heads in (1, 2, 4, 8, 16, 32, 64):
                    pieces = [
                        m * ((hcq // 32 + 31) // 32 * 32),
                        m * (512 + 64) * 2,
                        m * hcq * 2,
                        m * hcq // 2,
                        m * heads * (192 + 64) * 2,
                        m * heads * 192 * 2,
                    ]
                    offset = 0
                    for size in pieces:
                        self.assertEqual(offset % 32, 0)
                        self.assertGreater(size, 0)
                        offset += size
                    # QueryNorm is external and has its own BF16 extent.
                    self.assertEqual(m * hcq * 2, pieces[3] * 4)
                    self.assertLess(offset, 32 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
