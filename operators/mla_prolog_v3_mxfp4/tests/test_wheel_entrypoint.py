"""Build and import the vendor wheel on CPU; never compile or execute NPU code."""

import configparser
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PACKAGE = "cann_ops_transformer_mla_mxfp4"


class WheelEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[3]
        cls.temporary = tempfile.TemporaryDirectory(prefix="mla-wheel-entrypoint-")
        cls.addClassCleanup(cls.temporary.cleanup)
        temporary = Path(cls.temporary.name)
        source = temporary / "source"
        shutil.copytree(
            repo / "torch_extension",
            source / "torch_extension",
            ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
        )
        shutil.copytree(
            repo / "attention/mla_prolog_v3/torch_extension",
            source / "attention/mla_prolog_v3/torch_extension",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        # Exercise the production shell builder, including fallback when
        # `import build` resolves a package with no runnable build frontend.
        blockers = temporary / "no-build-frontend"
        (blockers / "build").mkdir(parents=True)
        (blockers / "build/__init__.py").write_text("")
        binaries = temporary / "bin"
        binaries.mkdir()
        python_launcher = binaries / "python3"
        python_launcher.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n'
        )
        python_launcher.chmod(0o755)
        environment = dict(
            os.environ,
            TORCH_EXTENSION_OPS="mla_prolog_v3",
            TORCH_EXTENSION_VENDOR="mla_mxfp4",
            TORCH_DEVICE_BACKEND_AUTOLOAD="0",
            FLA_NPU_DISABLE_PTH="1",
            PATH=str(binaries) + os.pathsep + os.environ.get("PATH", ""),
            PYTHONPATH=str(blockers) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )
        build_source = (repo / "build.sh").read_text()
        function = build_source.split("function build_torch_extension_whl() {", 1)[1]
        function = "function build_torch_extension_whl() {" + function.split("\nbuild_lib()", 1)[0]
        cls.builder = temporary / "build-wheel.sh"
        cls.builder.write_text(
            '#!/usr/bin/env bash\nset -eu\nCURRENT_DIR="$1"\n'
            'ascend_op_name=mla_prolog_v3\nvendor_name=mla_mxfp4\n'
            'log() { printf "%s\\n" "$*"; }\n'
            + function
            + '\nbuild_torch_extension_whl\n'
        )
        built = subprocess.run(
            ["bash", str(cls.builder), str(source)],
            cwd=source,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if built.returncode:
            raise AssertionError(
                "Vendor wheel build failed:\n" + (built.stdout + built.stderr)[-8000:]
            )
        if "using installed setuptools/wheel" not in built.stdout:
            raise AssertionError("Missing-build fallback was not exercised")
        wheels = list((source / "torch_extension/dist").glob("*.whl"))
        if len(wheels) != 1:
            raise AssertionError(f"Expected one vendor wheel, got {wheels}")
        cls.unpacked = temporary / "unpacked"
        with zipfile.ZipFile(wheels[0]) as wheel:
            wheel.extractall(cls.unpacked)
        cls.environment = environment

    def test_failed_packaging_returns_error(self):
        with tempfile.TemporaryDirectory(prefix="mla-failed-wheel-") as directory:
            source = Path(directory)
            extension = source / "torch_extension"
            extension.mkdir()
            (extension / "setup.py").write_text('raise RuntimeError("backend failure")\n')
            failed = subprocess.run(
                ["bash", str(self.builder), str(source)],
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("backend failure", failed.stdout)
            self.assertNotIn("whl package built successfully", failed.stdout)

    def test_wheel_metadata_and_vendor_import_rewrite(self):
        metadata = list(self.unpacked.glob("*.dist-info/entry_points.txt"))
        self.assertEqual(len(metadata), 1)
        parser = configparser.ConfigParser()
        parser.read(metadata[0])
        self.assertEqual(
            dict(parser["cann_ops_transformer.ops"]),
            {"mla_prolog_v3": f"{PACKAGE}.ops.attention.mla_prolog_v3:mla_prolog_v3"},
        )
        wrapper = (
            self.unpacked / PACKAGE / "ops/attention/mla_prolog_v3/mla_prolog.py"
        ).read_text()
        self.assertIn(
            f"from {PACKAGE}.op_builder import OpBuilder, get_as_library", wrapper
        )
        self.assertNotIn("from cann_ops_transformer.op_builder", wrapper)
        self.assertTrue(
            (self.unpacked / PACKAGE / "csrc/attention/mla_prolog.cpp").is_file()
        )

    def test_installed_entrypoint_aliases_call_this_vendor_builder(self):
        script = r"""
import importlib
import importlib.metadata as metadata
from pathlib import Path
import sys
import types
from unittest.mock import patch

root, package_name = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root))
distributions = list(metadata.distributions(path=[str(root)]))
entrypoints = [ep for dist in distributions for ep in dist.entry_points
               if ep.group == 'cann_ops_transformer.ops']
assert len(entrypoints) == 1, entrypoints
entrypoint = entrypoints[0]
# A later, same-name entrypoint from another installed vendor must not win.
foreign = metadata.EntryPoint(name='mla_prolog_v3',
    value='foreign_vendor.ops.attention.mla_prolog_v3:mla_prolog_v3',
    group='cann_ops_transformer.ops')
all_entries = metadata.EntryPoints([entrypoint, foreign])
import torch
fake_npu = types.ModuleType('torch_npu')
fake_npu.__file__ = str(root / 'fake_torch_npu/__init__.py')
with patch.dict(sys.modules, {'torch_npu': fake_npu}), \
     patch.object(metadata, 'entry_points', return_value=all_entries):
    loaded = entrypoint.load()
    package = importlib.import_module(package_name)
    implementation = importlib.import_module(package_name + '.ops.attention.mla_prolog_v3.mla_prolog')
    builder = importlib.import_module(package_name + '.op_builder.builder')
    assert callable(loaded)
    assert package.ops.mla_prolog_v3 is loaded
    assert package.ops.mla_prolog is loaded
    assert implementation.mla_prolog is loaded
    assert isinstance(implementation.mla_prolog_op_builder, builder.OpBuilder)
    assert 'cann_ops_transformer.op_builder' not in sys.modules
    assert 'foreign_vendor' not in sys.modules
    compiled, invoked = [], []
    marker = object()
    def cpp(*args):
        invoked.append(args)
        return marker
    def jit(**kwargs):
        compiled.append(kwargs)
        return types.SimpleNamespace(mla_prolog=cpp)
    inputs = [torch.empty(0) for _ in range(9)]
    with patch.object(builder, 'load', side_effect=jit):
        assert package.ops.mla_prolog_v3(*inputs, weight_quant_mode=6) is marker
        assert package.ops.mla_prolog(*inputs, weight_quant_mode=6) is marker
    assert len(compiled) == 1 and len(invoked) == 2
    assert Path(compiled[0]['sources'][0]) == root / package_name / 'csrc/attention/mla_prolog.cpp'
    assert invoked[0][25] == 6, invoked[0]
print('vendor wheel entrypoint, legacy alias and builder dispatch passed')
"""
        executed = subprocess.run(
            [sys.executable, "-c", script, str(self.unpacked), PACKAGE],
            cwd=self.unpacked,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
        self.assertIn("builder dispatch passed", executed.stdout)


if __name__ == "__main__":
    unittest.main()
