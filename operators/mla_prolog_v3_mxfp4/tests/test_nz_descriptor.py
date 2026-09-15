"""Compile the real wrapper helper and inspect its aclCreateTensor arguments.

The direct WeightNz example uses physical NZ dimensions for both ACL shapes.
This checks the frontend contract without simulating CANN tiling or a kernel.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class NzDescriptorTests(unittest.TestCase):
    def test_packed_weights_use_physical_nz_view_and_storage(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("A host C++ compiler is required for this regression")
        root = Path(__file__).resolve().parents[3]
        wrapper = root / "attention/mla_prolog_v3/torch_extension/csrc/mla_prolog.cpp"
        match = re.search(
            r"aclTensor \*MakePackedMxfp4NzDescriptor\([^\n]+\)\n\{.*?\n\}",
            wrapper.read_text(),
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        source = r"""#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

enum aclDataType { ACL_FLOAT4_E2M1 };
enum aclFormat { ACL_FORMAT_FRACTAL_NZ };
struct aclTensor {};
namespace at {
struct Tensor {
    std::array<int64_t, 4> shape;
    void *payload;
    int64_t size(int64_t dim) const { return shape.at(dim); }
    int64_t storage_offset() const { return 0; }
    struct Storage { void *ptr; const void *data() const { return ptr; } };
    Storage storage() const { return {payload}; }
};
}
std::vector<int64_t> observedView, observedStorage;
void *observedData = nullptr;
bool observedNullStrides = false;
aclTensor descriptor;
aclTensor *aclCreateTensor(const int64_t *view, uint64_t viewRank, aclDataType dtype,
                          const int64_t *strides, int64_t offset, aclFormat format,
                          const int64_t *storage, uint64_t storageRank, void *data)
{
    assert(dtype == ACL_FLOAT4_E2M1);
    assert(format == ACL_FORMAT_FRACTAL_NZ);
    assert(offset == 0);
    observedView.assign(view, view + viewRank);
    observedStorage.assign(storage, storage + storageRank);
    observedNullStrides = strides == nullptr;
    observedData = data;
    return &descriptor;
}
#define GET_OP_API_FUNC(name) (&name)
#define TORCH_CHECK(condition, ...) assert(condition)
""" + match.group(0)
        source += r"""
int main()
{
    // DQ, UQ at 4/8 heads, DKV, and the Hcq=1536 boundary.
    const int64_t projections[][2] = {
        {6144, 2048}, {2048, 1024}, {2048, 2048}, {6144, 576}, {1024, 1536}
    };
    for (const auto &projection : projections) {
        const int64_t k = projection[0], n = projection[1];
        // No payload is read: this test inspects the real descriptor arguments.
        uint8_t marker = 0;
        const at::Tensor weight{{n / 64, k / 16, 16, 32}, &marker};
        assert(MakePackedMxfp4NzDescriptor(weight) == &descriptor);
        const std::vector<int64_t> expected{n / 64, k / 16, 16, 64};
        assert(observedView == expected);
        assert(observedStorage == expected);
        assert(observedNullStrides);
        assert(observedData == &marker);
        int64_t elements = 1, bytes = 1;
        for (int64_t dim : observedStorage) elements *= dim;
        for (int64_t dim : weight.shape) bytes *= dim;
        assert(elements == k * n && elements == 2 * bytes);
    }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "nz_descriptor.cpp"
            executable = Path(directory) / "nz_descriptor"
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
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
