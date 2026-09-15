"""Exercise startup isolation without building or importing an NPU backend."""

import ast
import builtins
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


class LauncherIsolationTests(unittest.TestCase):
    def test_flags_are_set_before_first_python_process(self):
        with tempfile.TemporaryDirectory() as directory:
            bins = Path(directory)
            for name, body in {
                "uname": "printf 'Linux\\n'",
                "python3": 'printf "%s %s\\n" "$FLA_NPU_DISABLE_PTH" "$TORCH_DEVICE_BACKEND_AUTOLOAD"; exit 73',
            }.items():
                executable = bins / name
                executable.write_text("#!/bin/sh\n" + body + "\n")
                executable.chmod(0o755)
            environment = dict(
                os.environ,
                PATH=f"{bins}:{os.environ['PATH']}",
                ASCEND_HOME_PATH="/unused-cann",
                FLA_NPU_DISABLE_PTH="0",
                TORCH_DEVICE_BACKEND_AUTOLOAD="1",
            )
            result = subprocess.run(
                ["bash", str(ROOT / "operators/mla_prolog_v3_mxfp4/scripts/run_a5.sh")],
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 73, result.stderr)
            self.assertEqual(result.stdout.strip(), "1 0")

    def test_runner_restores_private_vendor_before_extension_import(self):
        path = ROOT / "operators/mla_prolog_v3_mxfp4/tests/run_a5.py"
        prefix = []
        for node in ast.parse(path.read_text()).body:
            prefix.append(node)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "cann_ops_transformer"
                for target in node.targets
            ):
                break
        extension_imports = []

        def import_extension(name):
            extension_imports.append((name, os.environ.get("ASCEND_CUSTOM_OPP_PATH")))
            return SimpleNamespace()

        def fake_import(name, *args, **kwargs):
            if name == "importlib":
                return SimpleNamespace(import_module=import_extension)
            if name == "torch_npu":
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = "/fla/vendor:/private/vendor"
            if name in ("torch", "torch_npu", "numpy"):
                return SimpleNamespace()
            return builtins.__import__(name, *args, **kwargs)

        with patch.dict(
            os.environ,
            {
                "ASCEND_CUSTOM_OPP_PATH": "/private/vendor",
                "MLA_MXFP4_TORCH_PACKAGE": "cann_ops_transformer_mla_mxfp4",
            },
        ):
            namespace = {"__builtins__": dict(vars(builtins), __import__=fake_import)}
            # Execute repository startup code under fake backend imports only.
            exec(  # noqa: S102
                compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec"),
                namespace,
            )
        self.assertEqual(
            extension_imports, [("cann_ops_transformer_mla_mxfp4", "/private/vendor")]
        )


if __name__ == "__main__":
    unittest.main()
