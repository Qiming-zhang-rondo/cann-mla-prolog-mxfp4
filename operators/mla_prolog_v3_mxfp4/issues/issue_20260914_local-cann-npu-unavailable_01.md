# 本地缺少 CANN/NPU，设备验收待内网 A5

- 日期：2026-09-14
- 状态：开放；影响后续 CANN 编译、NPU 精度及性能验收，不阻塞源码分析。
- 目标平台：用户指定 Ascend A5；本机没有证据可确定真实 full-soc-version、NpuArch、设备数量或对应 CANN 包版本，不能用目标名称代替设备检测结果。

## 环境证据

| 检查 | 实际结果 |
|---|---|
| `uname -srm` | `Darwin 23.6.0 arm64` |
| `command -v python3` | `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` |
| `command -v npu-smi` / `command -v ccec` | 均未找到 |
| `/usr/local/Ascend` / `/opt/Ascend` | 标准路径未找到 |
| `ASCEND_HOME_PATH` / `ASCEND_OPP_PATH` | 未设置 |
| skill `scripts/check_env.sh` | 退出码 2；报告“无法定位 CANN Toolkit 目录”，后续第 140 行 `declare -g: invalid option`，因此不是完整环境检查通过 |
| skill `scripts/npu_info.sh` | 退出码 1；报告“npu-smi 和 asys 均不可用” |
| skill `scripts/get_npu_arch.py --json` | 退出码 1；full_soc、npu_arch、short_soc、ccec_aiv_version、variant_dir、ini_path、npu_count 均为 null；asys 与 DSMI 不可用或无设备 |

## 处理方式

遵循用户已明确的“无，只能你改好代码我下拉到内网进行修改”：本地继续可执行的源码分析和静态检查，保留真实硬件验证未完成状态。未安装系统依赖、未反复探测不存在的设备，未修改业务代码。skill 环境脚本与 macOS 默认 Bash 的兼容问题仅记录，不为此修改上游 skill。

后续内网 A5 验证应记录 CANN/torch_npu 版本、CANN 路径、由 asys/DSMI 与精确 ini 匹配取得的真实 SoC/NpuArch，以及设备信息；运行 CANN 编译及新增算子的精度和性能用例。只有这些实际运行成功后才能关闭对应硬件验收缺口。
