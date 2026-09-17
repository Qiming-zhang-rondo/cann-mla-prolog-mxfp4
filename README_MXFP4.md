# A5 MLA Prolog V3 MXFP4 实验实现

为 GLM-5.2 / 同结构的 GLM-5.3 扩展 CANN `MlaPrologV3`，新增本分支专用 `weight_quant_mode=6`：FP4 E2M1 激活和三个量化投影权重、K32 E8M0 scale、真实 FP4 Cube 计算与内部 RMSNorm 后 FP4 再量化。`weightUk` 和外部 `query_norm` 保持 BF16，KV cache 支持现有 BF16 / FP8 两种模式。

**当前状态：用户 A5 日志已确认 CANN 算子执行完成，首例 native DQ 和融合 query 检查误差为0；完整 Prolog 精度/性能、GLM 端到端尚未通过验证。** 最新修复纠正测试输入的 RoPE sin 重复取反，待 A5 重测。此仓提供单算子开发和验收入口，vLLM-Ascend 的自动路由还没有接入此扩展。

## A5 一条命令

在有 Ascend C 编译工具链、torch/torch_npu 和 A5 设备的 Linux 容器中运行：

```bash
git clone --depth 1 https://github.com/Qiming-zhang-rondo/cann-mla-prolog-mxfp4.git && cd cann-mla-prolog-mxfp4 && bash test_mla_mxfp4.sh
```

脚本编译自定义 CANN 算子、安装到仓库内 `.a5-install/`、构建并安装独立 torch 扩展，然后比较融合算子与 native MXFP4 小算子链。无需模型权重或逐文件替换 VA。编译过程仍使用上游构建系统及其依赖；容器必须具备匹配的 CANN/编译环境和依赖访问能力。当前 CANN 版本兼容性尚待实测，不能把旧容器中已有的 MXFP4 QuantMatmul 能力视为本扩展已编译可用。

安装的 Python 包为 `cann_ops_transformer_mla_mxfp4`，OPP 为 `mla_mxfp4_transformer`；自定义库优先级只在测试进程内生效。默认 SoC 为 `ascend950`，可用 `MLA_MXFP4_SOC` 指定实际编译目标；必须使用 A5 / arch35，A3 不在实现范围内。`MAX_JOBS` 可控制编译并行度。

Python wheel 打包复用容器已有的 setuptools/wheel；没有 `build` 模块时自动使用 `setup.py bdist_wheel`，不下载 Python 打包依赖。wheel 安装使用 `--no-index --no-deps`，保留容器内 torch/torch_npu。

入口会在第一个 Python 进程启动前关闭 FLA `.pth` 注入和 torch 自动 backend 加载，显式导入 torch_npu 后恢复本次私有 OPP。日志分别显示 V3 workspace 与 compute API 所属库，并拒绝二者来自不同库；这不能单独证明设备 kernel 的来源或正确性。仅构建 wheel 时保留刚完成的算子构建目录。

已有目录更新后重测：

```bash
git pull --ff-only && bash test_mla_mxfp4.sh
```

已经成功编译并安装过本分支的 CANN 算子、此次仅更新 Python 包、torch C++ 绑定或测试时，复用最近一次完整私有安装：

```bash
git pull --ff-only && bash test_mla_mxfp4.sh --reuse-op
```

`--reuse-op` 跳过 CANN 算子编译和安装，仅重打包/安装本地 Python wheel 并运行测试。它从 `.a5-install/` 选择最近一个含 `mla_mxfp4_transformer` 库的安装；若指定 `MLA_MXFP4_INSTALL_DIR` 则只使用该路径。找不到安装会报错，不自动开始编译。修改 kernel/host/API 后须去掉该参数重建。首次成功调用时仍可能 JIT 编译 torch C++ 绑定，这与重新编译 CANN kernel 不同。日志输出所复用的路径。

结果保存在仓库根目录：带时间戳的 `mla_mxfp4_a5_*.log`，成功时 `mla_mxfp4_a5_results.json`，测试失败时 `mla_mxfp4_a5_failure.json`。日志记录实际加载的 V3 API 共享库；精度失败返回非零退出码。默认测试 T=1/17、TP 后头数=4/8、KV0/3、queryNorm 两种 flag；可运行 `bash test_mla_mxfp4.sh --tokens 1 17 128` 增加边界用例。

精度结果包含 query、RoPE、BF16 queryNorm、cache 和 scale；性能包含预量化融合耗时、含输入量化的融合耗时、native 计算以及含量化/cache copy 的完整对照，报告 P50/P90。容差为初步单算子验收门槛，性能是连续 slot 的单算子实验结果，均不代表 GLM 模型精度或吞吐已经验收。

## 依据与修改范围

- 基于 [cann/ops-transformer](https://gitcode.com/cann/ops-transformer) `632dddba712a4e6cace3f8b44f198aef8a82ce3e`。
- 参考 [cann/ops-nn](https://gitcode.com/cann/ops-nn) `2a77283db46e6648ff47bc8277442cf9c721e3c2` 的 MXFP4 quant / Cube / NZ contract。
- 对照 [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) `0979baf25c0ee501ab0e4ef9b88ea7d9a6d29839` 的 native W4A4 和 SFA 前处理。
- 按用户指定的 [cannbot-skills](https://gitcode.com/cann/cannbot-skills) `3074681ba927916f99f5a9ac808b4e6797934ce5` 做契约分析、模块实现、交叉检查和专用测试；通用 spec 执行器不适配项保留真实 FAIL/SKIP 记录。

具体文件、函数与数据布局见 [实现说明](operators/mla_prolog_v3_mxfp4/docs/IMPLEMENTATION.md)，测试步骤与限制见 [测试说明](operators/mla_prolog_v3_mxfp4/docs/TEST.md)。代码遵循仓库 [LICENSE](LICENSE)，现有上游文档仍适用于其原有功能；上游对 Prolog V3 的支持声明不包含此实验 mode6。

## 本地检查

```bash
python3 -m unittest discover -s operators/mla_prolog_v3_mxfp4/tests -p 'test_*.py' -v
python3 attention/mla_prolog_v3/tests/ut/op_kernel/test_mxfp4_storage_contract.py -v
```

前一组需要 numpy、torch 和 setuptools 的 wheel 打包能力；两组共 25 项测试已在开发机通过，包括真实 C++ 标量对齐表达式和 NZ 描述符函数的编译执行回归（需要 clang++ 或 g++）、Python 启动隔离、native 权重/scale stride、QR/KR 的公共 RoPE 输入合同、真实 wheel 导出导入，以及复用算子的 shell 流程。另有 C++ Host UT，尚未在 CANN 环境中编译运行。这些本地测试不会执行 Ascend C kernel，也不能证明设备寻址、同步或 FP4 指令行为。
