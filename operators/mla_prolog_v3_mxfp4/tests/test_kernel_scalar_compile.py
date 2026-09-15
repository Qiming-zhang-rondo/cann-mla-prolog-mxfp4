"""Compile the real scalar stride expression without Ascend C SDK dependencies.

This regression checks C++ types and stride arithmetic, not the device kernel.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class KernelScalarCompileTests(unittest.TestCase):
    def test_mxfp4_scale_stride_compiles_with_real_parameter_types(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required for this regression")
        root = Path(__file__).resolve().parents[3]
        kernel = root / "attention/mla_prolog/op_kernel/arch35"
        common = (kernel / "mla_prolog_comm_arch35.h").read_text()
        service = (kernel / "service_matmul_mxfp4_arch35.h").read_text()

        def extract(pattern, source):
            match = re.search(pattern, source, re.DOTALL)
            self.assertIsNotNone(match, pattern)
            return match.group(0)

        align = extract(
            r"template <typename T>\s+__aicore__ inline T Align\(.*?\n\}", common
        )
        params = extract(r"struct MMParams \{.*?\n\};", common)
        block = extract(r"constexpr uint64_t BYTE_BLOCK = [^;]+;", common)
        stride = extract(r"const uint32_t scaleStrideA = [^;]+;", service)
        source = f"""#include <cstdint>
#include <initializer_list>
#define __aicore__
{align}
{params}
{block}
uint32_t stride(const MMParams &p, bool paddedAScale) {{
    {stride}
    return scaleStrideA;
}}
int main() {{
    for (uint32_t groups : {{4u, 31u, 32u, 33u, 48u, 64u, 192u}}) {{
        MMParams p{{}}; p.kScale = groups;
        if (stride(p, false) != groups) return 1;
        if (stride(p, true) != ((groups + 31u) / 32u) * 32u) return 2;
    }}
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "scale_stride.cpp"
            executable = Path(directory) / "scale_stride"
            cpp.write_text(source)
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(cpp),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(executable)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
