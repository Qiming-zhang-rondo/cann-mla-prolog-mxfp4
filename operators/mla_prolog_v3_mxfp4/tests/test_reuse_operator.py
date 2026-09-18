"""Execute the real reuse launcher with temporary build/NPU/pip substitutes.

Installation and wheel discovery run in real Python. No operator, wheel, or
dependency is installed, and no NPU is accessed by these orchestration tests.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_a5.sh"


class ReuseOperatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mla-reuse-test-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repo with spaces"
        self.repo.mkdir()
        self.script = self.repo / "operators/mla_prolog_v3_mxfp4/scripts/run_a5.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)
        self.calls = self.repo / "python-calls.jsonl"
        self.build_calls = self.repo / "build-calls.txt"
        self.install_calls = self.repo / "install-calls.txt"
        binaries = self.repo / "fake-bin"
        binaries.mkdir()
        self.write_executable(binaries / "uname", "#!/bin/sh\nprintf 'Linux\\n'\n")
        self.write_executable(
            binaries / "git",
            "#!/bin/sh\nprintf '0123456789012345678901234567890123456789\\n'\n",
        )
        self.write_executable(
            binaries / "python3",
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "args = sys.argv[1:]\n"
            "kind = None\n"
            "if len(args) > 1 and args[0] == '-c' and args[1].startswith('import torch, torch_npu;'):\n"
            "    kind = 'npu_probe'\n"
            "elif args[:2] == ['-m', 'pip']:\n"
            "    kind = 'pip'\n"
            "elif len(args) > 1 and args[0] == '-u' and args[1].endswith('/tests/run_a5.py'):\n"
            "    kind = 'runner'\n"
            "record = dict(kind=kind or 'real_python', args=args,\n"
            "    opp=os.environ.get('ASCEND_CUSTOM_OPP_PATH'),\n"
            "    ld=os.environ.get('LD_LIBRARY_PATH'),\n"
            "    package=os.environ.get('MLA_MXFP4_TORCH_PACKAGE'),\n"
            "    fla_disabled=os.environ.get('FLA_NPU_DISABLE_PTH'),\n"
            "    autoload=os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD'))\n"
            "with open(os.environ['MLA_REUSE_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps(record) + '\\n')\n"
            "if kind is None:\n"
            "    os.execv(sys.executable, [sys.executable, *args])\n",
        )
        self.write_executable(
            self.repo / "build.sh",
            "#!/usr/bin/env bash\nset -eu\n"
            'printf \'%s\\n\' "$*" >> "$MLA_REUSE_BUILD_CALLS"\n'
            "[[ ${1:-} == --torch_extension ]] || { echo 'Unexpected CANN build' >&2; exit 93; }\n"
            "mkdir -p build_out\n"
            ": > build_out/cann_ops_transformer_mla_mxfp4-0.0.0-py3-none-any.whl\n",
        )
        # The launcher only checks find_spec for these build dependencies.
        # The fake build never imports them or builds a real wheel.
        module_stubs = self.repo / "module-stubs"
        module_stubs.mkdir()
        for name in ("setuptools", "wheel"):
            (module_stubs / f"{name}.py").write_text(
                "# Discovery-only test substitute.\n"
            )
        bash_env = self.repo / "bash-test-env.sh"
        bash_env.write_text(
            "# macOS ships Bash 3.2; only emulate the mapfile -t form used here.\n"
            "if (( BASH_VERSINFO[0] < 4 )); then\n"
            "  mapfile() {\n"
            "    [[ $1 == -t ]] || return 94\n"
            "    local name=$2 line\n"
            "    case $name in task_wheels|task_packages) ;; *) return 95 ;; esac\n"
            '    eval "$name=()"\n'
            '    while IFS= read -r line; do eval "$name+=(\\"\\$line\\")"; done\n'
            "  }\n"
            "fi\n"
        )
        self.env = dict(os.environ)
        self.env.pop("MLA_MXFP4_INSTALL_DIR", None)
        self.env.update(
            PATH=str(binaries) + os.pathsep + os.environ.get("PATH", ""),
            PYTHONPATH=str(module_stubs),
            BASH_ENV=str(bash_env),
            ASCEND_HOME_PATH=str(self.repo / "installed-cann"),
            ASCEND_CUSTOM_OPP_PATH="unrelated-existing-vendor",
            LD_LIBRARY_PATH="unrelated-existing-library",
            MLA_REUSE_CALLS=str(self.calls),
            MLA_REUSE_BUILD_CALLS=str(self.build_calls),
            MLA_REUSE_INSTALL_CALLS=str(self.install_calls),
        )

    @staticmethod
    def write_executable(path, content):
        path.write_text(content)
        path.chmod(0o755)

    def install(self, stamp, *, vendor="mla_mxfp4_transformer", complete=True):
        directory = self.repo / ".a5-install" / stamp
        opp = directory / "nested/opp/vendors" / vendor
        (opp / "op_api/lib").mkdir(parents=True)
        if complete:
            (opp / "op_api/lib/libcust_opapi.so").write_bytes(
                b"test-library-placeholder"
            )
        return directory, opp.resolve()

    def enable_package_build(self, *, exit_code=0):
        self.write_executable(
            self.repo / "package-fixture.run",
            "#!/usr/bin/env bash\nset -eu\n"
            'printf \'%s\\n\' "$*" >> "$MLA_REUSE_INSTALL_CALLS"\n'
            'for arg in "$@"; do\n'
            '  case "$arg" in --install-path=*) prefix=${arg#--install-path=} ;; esac\n'
            "done\n"
            'library="$prefix/opp/vendors/mla_mxfp4_transformer/op_api/lib/libcust_opapi.so"\n'
            'mkdir -p "$(dirname "$library")"\n'
            ': > "$library"\n',
        )
        self.write_executable(
            self.repo / "build.sh",
            "#!/usr/bin/env bash\nset -eu\n"
            'printf \'%s\\n\' "$*" >> "$MLA_REUSE_BUILD_CALLS"\n'
            'case "$1" in\n'
            "  --pkg)\n"
            f"    if (( {exit_code} )); then exit {exit_code}; fi\n"
            "    mkdir -p build_out\n"
            "    cp package-fixture.run build_out/cann-ops-transformer-mla_mxfp4-test.run\n"
            "    ;;\n"
            "  --torch_extension)\n"
            "    mkdir -p build_out\n"
            "    : > build_out/cann_ops_transformer_mla_mxfp4-0.0.0-py3-none-any.whl\n"
            "    ;;\n"
            "  *) exit 93 ;;\n"
            "esac\n",
        )
        self.env["MLA_MXFP4_INSTALL_DIR"] = str(self.repo / "target installation")

    def launch(self, *args):
        result = subprocess.run(
            ["bash", str(self.script), *args],
            cwd=self.repo,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        calls = (
            [json.loads(line) for line in self.calls.read_text().splitlines()]
            if self.calls.exists()
            else []
        )
        builds = (
            self.build_calls.read_text().splitlines()
            if self.build_calls.exists()
            else []
        )
        return result, calls, builds

    def test_reuse_selects_latest_complete_private_install_and_forwards_args(self):
        self.install("20260914-100000")
        _, expected_opp = self.install("20260915-100000")
        self.install("20260915-110000", complete=False)
        self.install("20260915-120000", vendor="unrelated_vendor")
        forwarded = [
            "--tokens",
            "1",
            "17",
            "--heads",
            "4",
            "--iterations",
            "2",
            "--output",
            "results with spaces.json",
        ]
        result, calls, builds = self.launch("--reuse-op", *forwarded)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            builds,
            [
                "--torch_extension --ops=mla_prolog_v3 --vendor_name=mla_mxfp4 --incremental"
            ],
        )
        pip = [call for call in calls if call["kind"] == "pip"]
        self.assertEqual(len(pip), 1)
        self.assertEqual(
            pip[0]["args"],
            [
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--disable-pip-version-check",
                "--force-reinstall",
                "build_out/cann_ops_transformer_mla_mxfp4-0.0.0-py3-none-any.whl",
            ],
        )
        runners = [call for call in calls if call["kind"] == "runner"]
        self.assertEqual(len(runners), 1)
        self.assertEqual(runners[0]["args"][2:], forwarded)
        self.assertNotIn("--reuse-op", runners[0]["args"])
        self.assertEqual(runners[0]["opp"], str(expected_opp))
        self.assertEqual(
            runners[0]["ld"],
            str(expected_opp / "op_api/lib") + ":unrelated-existing-library",
        )
        self.assertEqual(runners[0]["package"], "cann_ops_transformer_mla_mxfp4")
        self.assertTrue(
            all(
                call["fla_disabled"] == "1" and call["autoload"] == "0"
                for call in calls
            )
        )
        real = [call for call in calls if call["kind"] == "real_python"]
        self.assertTrue(
            any(call["args"][0] == "-" for call in real)
        )  # installation discovery
        self.assertTrue(
            any("glob" in " ".join(call["args"]) for call in real)
        )  # wheel discovery

    def test_explicit_install_overrides_newer_automatic_install(self):
        explicit, expected_opp = self.install("20260914-100000")
        self.install("20260915-100000")
        self.env["MLA_MXFP4_INSTALL_DIR"] = str(explicit)
        result, calls, builds = self.launch("--reuse-op", "--tokens", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(builds), 1)
        runner = next(call for call in calls if call["kind"] == "runner")
        self.assertEqual(runner["opp"], str(expected_opp))

    def test_missing_private_library_fails_before_build_or_install(self):
        self.install("20260915-110000", complete=False)
        self.install("20260915-120000", vendor="unrelated_vendor")
        result, calls, builds = self.launch("--reuse-op", "--tokens", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No installed MLA MXFP4 operator found", result.stderr)
        self.assertEqual(builds, [])
        self.assertFalse(any(call["kind"] in ("pip", "runner") for call in calls))

    def test_incremental_resumes_package_build_then_installs_and_forwards_test_args(
        self,
    ):
        self.enable_package_build()
        forwarded = [
            "--tokens",
            "17",
            "--iterations",
            "2",
            "--output",
            "resumed results.json",
        ]
        result, calls, builds = self.launch("--incremental", *forwarded)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(builds), 2)
        package_args = builds[0].split()
        self.assertEqual(package_args[0], "--pkg")
        self.assertEqual(package_args.count("--incremental"), 1)
        self.assertIn("--ops=mla_prolog_v3", package_args)
        self.assertTrue(builds[1].startswith("--torch_extension "))
        self.assertEqual(len(self.install_calls.read_text().splitlines()), 1)
        runner = next(call for call in calls if call["kind"] == "runner")
        self.assertEqual(runner["args"][2:], forwarded)
        expected_opp = (
            Path(self.env["MLA_MXFP4_INSTALL_DIR"])
            / "opp/vendors/mla_mxfp4_transformer"
        )
        self.assertEqual(runner["opp"], str(expected_opp))
        self.assertTrue((expected_opp / "op_api/lib/libcust_opapi.so").is_file())

    def test_default_package_build_remains_clean(self):
        self.enable_package_build()
        result, calls, builds = self.launch("--tokens", "1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(builds), 2)
        self.assertTrue(builds[0].startswith("--pkg "))
        self.assertNotIn("--incremental", builds[0].split())
        self.assertTrue(any(call["kind"] == "runner" for call in calls))

    def test_failed_incremental_build_does_not_install_stale_package_or_run_tests(self):
        self.enable_package_build(exit_code=24)
        stale_package = self.repo / "build_out/cann-ops-transformer-mla_mxfp4-stale.run"
        stale_package.parent.mkdir()
        shutil.copyfile(self.repo / "package-fixture.run", stale_package)
        result, calls, builds = self.launch("--incremental", "--tokens", "1")
        self.assertEqual(result.returncode, 24, result.stdout + result.stderr)
        self.assertEqual(len(builds), 1)
        self.assertIn("--incremental", builds[0].split())
        self.assertFalse(self.install_calls.exists())
        self.assertFalse(
            any(Path(self.env["MLA_MXFP4_INSTALL_DIR"]).rglob("libcust_opapi.so"))
        )
        self.assertFalse(any(call["kind"] in ("pip", "runner") for call in calls))

    def test_reuse_and_incremental_are_rejected_before_npu_probe(self):
        result, calls, builds = self.launch("--reuse-op", "--incremental")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--reuse-op and --incremental cannot be combined", result.stderr)
        self.assertEqual(calls, [])
        self.assertEqual(builds, [])


if __name__ == "__main__":
    unittest.main()
