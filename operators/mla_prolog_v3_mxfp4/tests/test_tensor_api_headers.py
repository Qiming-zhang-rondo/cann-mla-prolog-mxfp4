"""Run real CMake header selection, staging and installation without CANN."""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HEADER_TREES = (
    "impl/tensor_api",
    "include/tensor_api",
    "impl/c_api",
    "include/c_api",
)


class TensorApiHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmake = os.environ.get("CMAKE") or shutil.which("cmake")
        if not cls.cmake:
            raise unittest.SkipTest("An installed CMake is required")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tensor-api-headers-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "fixture"
        self.source.mkdir()
        self.build = self.root / "build"
        self.install = self.root / "installed"
        self.optensor = self.root / "ops-tensor"
        self.cann = self.root / "cann"
        self.other_cann = self.root / "other-cann"

    @staticmethod
    def headers(root, trees=HEADER_TREES, *, label="selected"):
        for tree in trees:
            directory = root / tree
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "fixture.h").write_text(f"// {label}: {tree}\n")
        return root

    def configure(self, *, explicit=None, extra=""):
        settings = {
            "OPTENSOR_SOURCE_PATH": self.optensor,
            "ASCEND_DIR": self.cann,
            "ASCEND_CANN_PACKAGE_PATH": self.other_cann,
            "SYSTEM_PREFIX": "aarch64-linux",
        }
        if explicit is not None:
            settings["TENSOR_API"] = explicit
        assignments = "\n".join(
            f"set({name} [=[{value}]=])" for name, value in settings.items()
        )
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "project(tensor_api_fixture NONE)\n"
            + assignments
            + f'\ninclude("{REPO}/cmake/third_party/tensor_api.cmake")\n'
            + "resolve_tensor_api_headers()\n"
            + 'file(WRITE "${CMAKE_BINARY_DIR}/resolved.txt" "${TENSOR_API}")\n'
            + extra
        )
        return subprocess.run(
            [self.cmake, "-S", str(self.source), "-B", str(self.build)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def assert_selected(self, result, expected):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            Path((self.build / "resolved.txt").read_text()).resolve(),
            expected.resolve(),
        )

    def test_complete_source_preferred_to_cann(self):
        source = self.headers(self.optensor / "include/tensor_api", label="source")
        self.headers(self.cann / "aarch64-linux/asc", label="cann")
        self.assert_selected(self.configure(), source)

    def test_complete_explicit_root_preferred_to_source(self):
        explicit = self.headers(self.root / "explicit-asc", label="explicit")
        self.headers(self.optensor / "include/tensor_api", label="source")
        self.assert_selected(self.configure(explicit=explicit), explicit)

    def test_empty_or_missing_source_uses_installed_cann(self):
        installed = self.headers(self.cann / "aarch64-linux/asc")
        self.assert_selected(self.configure(), installed)
        for tree in HEADER_TREES:
            (self.optensor / "include/tensor_api" / tree).mkdir(parents=True)
        self.assert_selected(self.configure(), installed)

    def test_incomplete_explicit_and_source_use_complete_cann_root(self):
        explicit = self.headers(self.root / "explicit-asc", HEADER_TREES[:1])
        self.headers(self.optensor / "include/tensor_api", HEADER_TREES[:3])
        installed = self.headers(self.other_cann / "aarch64-linux/asc")
        self.assert_selected(self.configure(explicit=explicit), installed)

    def test_complete_archless_cann_layout(self):
        installed = self.headers(self.other_cann / "asc")
        self.assert_selected(self.configure(), installed)

    def test_partial_cann_roots_are_not_combined(self):
        first = self.headers(self.cann / "aarch64-linux/asc", HEADER_TREES[:2])
        second = self.headers(self.other_cann / "asc", HEADER_TREES[2:])
        result = self.configure()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        diagnostic = result.stdout + result.stderr
        self.assertIn(str(first), diagnostic)
        self.assertIn(str(second), diagnostic)
        self.assertFalse((self.build / "resolved.txt").exists())

    def test_missing_headers_fail_during_configure_before_build(self):
        result = self.configure(
            extra="add_custom_target(should_not_build ALL\n"
            '    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_BINARY_DIR}/built")\n'
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        diagnostic = result.stdout + result.stderr
        self.assertIn(str(self.optensor / "include/tensor_api"), diagnostic)
        self.assertIn(str(self.cann / "aarch64-linux/asc"), diagnostic)
        self.assertFalse((self.build / "built").exists())
        self.assertFalse((self.build / "resolved.txt").exists())

    def test_installed_headers_stage_and_install_all_four_trees(self):
        installed = self.headers(self.cann / "aarch64-linux/asc", label="cann")
        for name in ("utils", "cgmct", "gmm", "catlass/include", "blaze", "op_kernel"):
            directory = self.source / name
            directory.mkdir(parents=True)
            (directory / "dummy.h").write_text(f"// {name}\n")

        # Use the production install rules, isolated from unrelated SDK targets.
        production = (REPO / "cmake/custom_build.cmake").read_text()
        install_rules = []
        for tree in HEADER_TREES:
            pattern = (
                r"install\(DIRECTORY \$\{TENSOR_API\}/"
                + re.escape(tree)
                + r"\s+DESTINATION [^)]+\)"
            )
            match = re.search(pattern, production)
            self.assertIsNotNone(match, tree)
            install_rules.append(match.group(0))
        extra = f'''
include("{REPO}/cmake/func.cmake")
set(OPS_ADV_UTILS_KERNEL_INC "${{CMAKE_SOURCE_DIR}}/utils")
set(OPS_CGMCT "${{CMAKE_SOURCE_DIR}}/cgmct")
set(OPS_GMM_COMMON_KERNEL_INC "${{CMAKE_SOURCE_DIR}}/gmm")
set(CATLASS "${{CMAKE_SOURCE_DIR}}/catlass")
set(BLAZE "${{CMAKE_SOURCE_DIR}}/blaze")
add_ops_src_copy(TARGET_NAME fixture_stage
    SRC "${{CMAKE_SOURCE_DIR}}/op_kernel"
    DST "${{CMAKE_BINARY_DIR}}/stage/operator"
    COMPUTE_UNIT ascend950)
set(IMPL_INSTALL_DIR "op_impl")
'''
        result = self.configure(extra=extra + "\n".join(install_rules))
        self.assert_selected(result, installed)
        for command in (
            [self.cmake, "--build", str(self.build), "--target", "fixture_stage"],
            [self.cmake, "--install", str(self.build), "--prefix", str(self.install)],
        ):
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        staged = self.build / "stage/ascendc/common"
        self.assertTrue((staged / ".utils_copy.stamp").is_file())
        for tree in HEADER_TREES:
            expected = (installed / tree / "fixture.h").read_text()
            for copied_root in (
                staged / "tensor_api",
                self.install / "op_impl/ascendc/common/tensor_api",
            ):
                self.assertEqual(
                    (copied_root / tree / "fixture.h").read_text(), expected
                )


if __name__ == "__main__":
    unittest.main()
