#!/usr/bin/env bash
# Build and test this checkout, leaving system CANN installation intact.
set -euo pipefail
# Disable Python startup injection before the first Python process is launched.
export FLA_NPU_DISABLE_PTH=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
task_reuse_op=0
task_test_args=()
for task_arg in "$@"; do
  if [[ $task_arg == --reuse-op ]]; then
    task_reuse_op=1
  else
    task_test_args+=("$task_arg")
  fi
done
task_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_repo=$(cd -- "$task_script_dir/../../.." && pwd)
cd "$task_repo"
if [[ $(uname -s) != Linux ]]; then
  echo 'This command needs the A5 Linux container; it cannot run on macOS.' >&2
  exit 1
fi
if [[ -z ${ASCEND_HOME_PATH:-} ]]; then
  task_env=/usr/local/Ascend/ascend-toolkit/set_env.sh
  [[ -f $task_env ]] || { echo 'CANN set_env.sh unavailable; source your installed CANN environment.' >&2; exit 1; }
  set +u
  source "$task_env"
  set -u
fi
python3 -c 'import torch, torch_npu; assert torch.npu.is_available(), "NPU unavailable"; print("Device:", torch.npu.get_device_name())'
python3 -c 'import importlib.util; missing=[m for m in ("setuptools","wheel") if importlib.util.find_spec(m) is None]; assert not missing, "Missing wheel-build dependencies: "+str(missing)+"; use a container with setuptools and wheel installed"'
task_ref=$(git rev-parse HEAD)
echo "Testing commit $task_ref, custom V3 weight_quant_mode=6"
if ((task_reuse_op)); then
  task_opp=$(python3 - "$task_repo/.a5-install" "${MLA_MXFP4_INSTALL_DIR:-}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
explicit = sys.argv[2]
installs = [pathlib.Path(explicit)] if explicit else sorted(
    (p for p in root.glob('*') if p.is_dir()), key=lambda p: p.name, reverse=True)
for install in installs:
    libraries = list(install.glob('**/vendors/mla_mxfp4_transformer/op_api/lib/libcust_opapi.so'))
    if len(libraries) > 1:
        raise SystemExit(f'Ambiguous private operator installation: {install}')
    if libraries and libraries[0].is_file():
        print(libraries[0].resolve().parents[2])
        break
else:
    raise SystemExit('No installed MLA MXFP4 operator found; run once without --reuse-op')
PY
  )
  echo "Reuse installed private operator; skip CANN build and installation: $task_opp"
else
task_stamp=$(date +%Y%m%d-%H%M%S)
task_install=${MLA_MXFP4_INSTALL_DIR:-$task_repo/.a5-install/$task_stamp}
mkdir -p "$task_install"
bash build.sh --pkg --soc="${MLA_MXFP4_SOC:-ascend950}" --ops=mla_prolog_v3 --vendor_name=mla_mxfp4 -j"${MAX_JOBS:-8}"
mapfile -t task_packages < <(python3 -c 'import pathlib; print("\n".join(str(p) for p in pathlib.Path("build_out").glob("cann-ops-transformer-mla_mxfp4*.run")))')
[[ ${#task_packages[@]} == 1 ]] || { echo 'Expected exactly one fresh .run package in build_out.' >&2; exit 1; }
bash "${task_packages[0]}" --quiet --install-path="$task_install"
task_opp=$(python3 -c 'import pathlib,sys; roots={str(p.parent.parent.parent) for p in pathlib.Path(sys.argv[1]).rglob("libcust_opapi.so")}; assert len(roots)==1, f"Expected one custom OPP root, got {roots}"; print(roots.pop())' "$task_install")
fi
export ASCEND_CUSTOM_OPP_PATH="$task_opp"
export LD_LIBRARY_PATH="$task_opp/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "Using custom OPP: $task_opp"
bash build.sh --torch_extension --ops=mla_prolog_v3 --vendor_name=mla_mxfp4 --incremental
mapfile -t task_wheels < <(python3 -c 'import pathlib; print("\n".join(str(p) for p in pathlib.Path("build_out").glob("cann_ops_transformer_mla_mxfp4-*.whl")))')
[[ ${#task_wheels[@]} == 1 ]] || { echo 'Expected exactly one extension wheel.' >&2; exit 1; }
python3 -m pip install --no-index --no-deps --disable-pip-version-check --force-reinstall "${task_wheels[0]}"
export MLA_MXFP4_TORCH_PACKAGE=cann_ops_transformer_mla_mxfp4
python3 -u operators/mla_prolog_v3_mxfp4/tests/run_a5.py "${task_test_args[@]}"
