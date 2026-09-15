"""Check the real test producers against QuantMatmul's stride contract on CPU."""

import ast
import unittest
from pathlib import Path

import torch
from reference import pack_nz_from_codes, unpack_codes


class NativeMatmulLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("run_a5.py")
        names = {"native_weight_views", "fused_native_weight", "weight", "mm"}
        nodes = [
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]

        def quant_rows(x):
            # Distinct bytes per row/group; this fixture does not emulate the
            # NPU quantizer. It exercises packing, views and concatenation.
            n, k = x.shape
            payload = (torch.arange(n * (k // 2)).reshape(n, k // 2) % 256).byte()
            scales = (110 + torch.arange(n * (k // 32)).reshape(n, k // 32) % 32).byte()
            return payload, scales.view(torch.float8_e8m0fnu)

        namespace = {
            "torch": torch,
            "quant_rows": quant_rows,
            "pack_nz_from_codes": pack_nz_from_codes,
            "unpack_codes": unpack_codes,
        }
        exec(  # noqa: S102
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            namespace,
        )
        cls.functions = namespace

    @staticmethod
    def transposed(tensor, dimension):
        # Ascend/op-plugin QuantMatmulKernelNpuOpApi.cpp uses this predicate
        # for x2's last two dimensions and scale's third/second-last dimensions.
        return tensor.stride(dimension + 1) == tensor.stride(dimension) * tensor.size(
            dimension
        )

    def check_layout(self, native, packed, scales):
        k = packed.shape[1] * 2
        n = packed.shape[0]
        self.assertEqual(native["native"].shape, (k // 2, n))
        self.assertEqual(native["native_scale"].shape, (k // 64, n, 2))
        self.assertEqual(native["native"].stride(), (1, k // 2))
        self.assertEqual(native["native_scale"].stride(), (2, k // 32, 1))
        self.assertTrue(self.transposed(native["native"], 0))
        self.assertTrue(self.transposed(native["native_scale"], 0))
        torch.testing.assert_close(native["native"].T, packed, rtol=0, atol=0)
        actual = native["native_scale"].view(torch.uint8)
        expected = scales.view(torch.uint8)
        for row in (0, n // 2, n - 1):
            for group in range(k // 32):
                self.assertEqual(
                    actual[group // 2, row, group % 2].item(),
                    expected[row, group].item(),
                )

    def test_real_weight_producer_retains_transposed_weight_and_scale(self):
        for k, n in ((128, 64), (2048, 64), (6144, 64)):
            with self.subTest(k=k, n=n):
                w = self.functions["weight"](
                    k, n, torch.Generator().manual_seed(3), "cpu"
                )
                self.check_layout(w, w["row_packed"], w["scale"])
                # Same values and shape can still violate the device contract.
                self.assertFalse(self.transposed(w["native_scale"].contiguous(), 0))

    def test_fused_down_concatenates_channels_before_transposing(self):
        weights = [
            self.functions["weight"](128, n, torch.Generator().manual_seed(n), "cpu")
            for n in (64, 128)
        ]
        native = self.functions["fused_native_weight"](weights)
        packed = torch.cat([w["row_packed"] for w in weights], dim=0)
        scales = torch.cat([w["scale"].view(torch.uint8) for w in weights], dim=0)
        self.check_layout(native, packed, scales)


if __name__ == "__main__":
    unittest.main()
