"""Check the real runner's GM factors and QR/KR against public RoPE semantics."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class RopeInputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("run_a5.py")
        module = ast.parse(path.read_text())
        rope = next(
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "rope"
        )
        main = next(
            n
            for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        # Execute the actual input producer, without importing torch_npu or
        # changing the runner's NPU execution path.
        assignments = {
            n.targets[0].id: n
            for n in ast.walk(main)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in {"angle", "cos", "sin"}
        }
        cls.producer = compile(
            ast.Module(
                body=[assignments[name] for name in ("angle", "cos", "sin")],
                type_ignores=[],
            ),
            str(path),
            "exec",
        )
        namespace = {"torch": torch}
        exec(  # noqa: S102
            compile(ast.Module(body=[rope], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        cls.rope = staticmethod(namespace["rope"])

    def factors(self, tokens):
        namespace = {
            "torch": torch,
            "tokens": tokens,
            "args": SimpleNamespace(device="cpu"),
        }
        exec(self.producer, namespace)  # noqa: S102
        return namespace["cos"], namespace["sin"], namespace["angle"]

    def test_gm_factors_have_no_kernel_internal_sign_preprocessing(self):
        for tokens in (1, 17, 128):
            with self.subTest(tokens=tokens):
                cos, sin, angle = self.factors(tokens)
                for half in (slice(0, 32), slice(32, 64)):
                    torch.testing.assert_close(
                        cos[:, half], angle.cos().bfloat16(), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        sin[:, half], angle.sin().bfloat16(), rtol=0, atol=0
                    )

    def test_native_qr_and_kr_use_one_rotation_sign(self):
        generator = torch.Generator().manual_seed(19)
        for tokens, heads in ((1, None), (1, 4), (17, None), (17, 8)):
            with self.subTest(tokens=tokens, heads=heads):
                cos, sin, _ = self.factors(tokens)
                shape = (tokens, 64) if heads is None else (tokens, heads, 64)
                x = (torch.randn(shape, generator=generator) * 32).bfloat16()
                if heads is not None:
                    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
                # Public reference: interleave -> halves, then rotate_half.
                # GatherSinCos supplies this minus sign inside the kernel.
                even, odd = x.float()[..., ::2], x.float()[..., 1::2]
                half = torch.cat((even, odd), dim=-1)
                rotated = torch.cat((-odd, even), dim=-1)
                expected = (half * cos.float() + rotated * sin.float()).bfloat16()
                actual = self.rope(x, cos, sin)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
