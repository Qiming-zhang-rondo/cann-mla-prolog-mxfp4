# MlaPrologV3 MXFP4 当前实现

2026-09-15 更新。**本分支已加入 mode6 的 API、host、kernel、torch 测试入口和验证脚本；用户 A5 日志确认 CANN 算子和 torch C++ 绑定编译成功，自定义 Prolog 精度/性能仍待验收。** mode6 是本扩展新增值，上游基线不接受该值。本文记录已写入代码的契约；[REQUIREMENTS.md](REQUIREMENTS.md) 和 [接口草案](aclnnMlaPrologV3WeightNz.md) 保留的是最初需求提案。

目标是 MLA 三组投影的 W4A4 MXFP4 融合。三组投影直接使用 FP4 Cube，内部 Q RMSNorm 后再量化为 FP4；`weightUk`、query、queryRope 和对外 queryNorm 保持 BF16。这里没有新增 FP4 KV cache，也没有修改 Indexer QLI。尚未把此 CANN 扩展接入 VA 的 GLM 路由，因此不能据此宣称完整模型已走 fused MXFP4 Prolog。

## 支持范围和数据契约

新路径仅 `MlaPrologV3`、A5/DAV_3510（Ascend950）、`weight_quant_mode=6`、二维 token、`1 <= T <= 128`、非空 `PA_BSND` cache；`query_quant_mode=0`、`do_rope=true`、`qc_qr_scale=kc_scale=1`。KV mode 只允许0或3；其他新组合在 host 拒绝。未启用旧 full-A、split-M 优化。

合法维度为 He∈{1024,2048,3072,4096,5120,6144,7168,7680,8192}，Hcq∈{1536,2048}，Hckv=512，D∈{128,192}，Dr=64，N∈[1,128]、Nkv=1。K 和每个量化投影的输出通道需64对齐。**GLM-5.2/5.3 对应 He6144/Hcq2048/D192/Dr64/Hckv512，N 按 TP 切分**；允许的维度范围不等于每种组合都已完成设备验证。

| 张量 | 当前 contract |
|---|---|
| tokenX | 逻辑 `[T,He]`，ACL_FLOAT4_E2M1、ND；两值/byte，偶数下标低 nibble、奇数下标高 nibble；torch 入口用连续 uint8 `[T,He/2]` 加显式 FP4 dtype |
| weightDq / weightUqQr / weightDkvKr | 分别为逻辑 `[He,Hcq]`、`[Hcq,N*(D+Dr)]`、`[He,Hckv+Dr]`；非转置 A4W4 FRACTAL_NZ，逻辑 storage `[Nout/64,K/16,16,64]`；torch 入口为已完成 NZ 排列的 uint8 `[Nout/64,K/16,16,32]`，普通 ND reshape 不能生成 NZ |
| dequantScaleX | 连续 ND E8M0 `[T,He/32]` |
| dequantScaleWDq / WUqQr / WDkvKr | 连续 ND E8M0 `[Hcq,He/32]`、`[N*(D+Dr),Hcq/32]`、`[Hckv+Dr,He/32]`；每输出通道的 K32 scales 连续，两个相邻 K32 构成一对 |
| weightUk / gamma / sin、cos | BF16；`weightUk[N,D,512]`，gamma `[Hcq]` / `[512]`，sin、cos `[T,64]`；RMSNorm epsilon 沿用现有正值校验 |
| query / queryRope | BF16 `[T,N,512]` / `[T,N,64]` |
| queryNorm | flag=true 时 BF16 `[T,Hcq]`，取 RMSNorm 后、内部 FP4 量化前的值；flag=false 时空 BF16 `[0]`；不从 FP4 反量化恢复该输出 |
| dequantScaleQNope / dequantScaleQNorm | 均为空；ACLNN 调用传 nullptr，内部 TensorHolder/IR 为 FLOAT `[0]`；本仓 torch 测试 wrapper 返回空 FLOAT tensor，非 None |
| cacheIndex | INT64 `[T]`；合法 slot 或 -1 哨兵。合法值范围 `[0,Pages*Block)` 是调用者前置条件，host 不读取 device slot 值，不能承诺越界 slot 返回 host 错误 |

FP4 ACL 的 shape、stride、offset 以逻辑元素计，内存分配和 byte DMA 以两个元素/byte 计。直连 WeightNz ACLNN 的权重 wrapper 将 view 和 storage 都设为四维 `[Nout/64,K/16,16,64]`，strides 为 nullptr，对齐官方 `attention/mla_prolog_v3/examples/arch35/test_aclnn_mla_prolog_v3_fqkvq.cpp::CreateAclTensorNZ`。底层 uint8 容器末轴仍为32，不再转换或重新打包。先前二维 view/四维 storage 在 A5 的 tiling 中被视为二维，触发 mode6 的四维校验；此修复只改直连 wrapper，不改变图模式的二维原始 shape 规则。QuantMatmul 权重 scale 的 `[K/64,Nout,2]` 与本算子的 `[Nout,K/32]` 排列不同，转换必须保留数据对应关系；测试 producer/consumer 见 `tests/reference.py` 和 `run_a5.py::native_weight_views`。

| 缓存模式 | 原地写入 contract |
|---|---|
| KV0 | KV BF16 `[Pages,Block,1,512]`，KR BF16 `[Pages,Block,1,64]`；两个 repo mode 为0 |
| KV3 | KV 的 FP8 E4M3 容器 `[Pages,Block,1,656]`：512 byte FP8 latent + 128 byte BF16 RoPE + 16 byte（4个FP32）scale；KR 为空 BF16 `[0]`；两个 repo mode 为1，tile_size=128 |

Pages>0；Block∈[16,1024] 且为16倍数。`quantScaleCkv`、`quantScaleCkr`、`smoothScalesCq`、`actualSeqLen`、`kNopeClipAlpha` 在新模式必须缺省或为空；非空明确拒绝。KV3 的 norm 先舍入为 BF16，再进入原 FP8 per-tile 量化，保持本期 native 基线的数值边界。

三组 FP4 GEMM 使用 FP32 L0C 累加，完整 K 归约后写 BF16。内部 Q4 的 block size=32，量化使用 E2M1/E8M0 OCP scale 与 ROUND 路径；字节、ties、异常值与 native DynamicMxQuant 的一致性仍待 A5 验证。RoPE 沿用原 Prolog 实现；GLM interleave 与 Prolog half-split 的 sin/cos、权重列和输出排列由测试明确转换。

## Workspace 和 tiling key

令 `S=Align(Hcq/32,32)`（byte）。内部 workspace 按顺序为：`T*S` E8M0 Q scale、`T*(Hckv+Dr)*2` BF16 down-KV、`T*Hcq*2` BF16 down-Q、`T*Hcq/2` packed Q4、`T*N*(D+Dr)*2` BF16 QcQr、`T*N*D*2` extracted Qc。Host 总申请额外保留已有 `libapiSize_`。外部 queryNorm 不占这些缓冲，关闭/开启该输出均必须保留内部 Q4。

Host 语义组合 enum 追加16/17，但不将其塞入旧4-bit字段。`SCENARIO=3` 下 `QUANT_MODE=0/1` 分别表示 KV0/KV3；PA_BSND、dequant opt、CV1:2 下 key 为 **3933233 / 3933297**。旧 QUANT_MODE 位宽、后续字段位位置与所有原 selectors 保持不变；新实例仅开放 V3。

## 改动位置

以下路径相对 ops-transformer 根目录。

| 文件 | 关键函数或变更 |
|---|---|
| `attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.cpp` | `CheckWeightQuantModeValidity`、`CheckKvCacheQuantModeValidity`、`aclnnMlaPrologV3WeightNzGetWorkspaceSize`；mode6 准入和 BF16/空 scale TensorHolder；函数签名不变 |
| `attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp` | `MlaPrologV3` 的 A5 注册新增两个 dtype 列；原 A2/A3 注册不变 |
| `attention/mla_prolog_v3/op_host/mla_prolog_v3_infershape.{h,cpp}` | 新 mode 常量、`SetQueryNormShape`、`InferDataTypeMlaPrologV3` |
| `attention/mla_prolog/op_host/mla_prolog_tiling.h` | mode/组合枚举及正反查表、`GetMlaWeightNzC0` |
| `attention/mla_prolog/op_host/mla_prolog_tiling_check.{h,cpp}` | `CheckMxfp4Scope`、单参校验、`FillCommonParamInfo`、`FillMxfp4FullQuantParamInfo`/KV3、queryNorm shape/dtype |
| `attention/mla_prolog/op_host/mla_prolog_tiling.cpp` | `SetShapeInfo`、`FillMatmul{1,2,3}Tiling`、`ProcessBaseInputs`、`CalcWorkSpace`、`GenTilingKey` |
| `attention/mla_prolog/op_kernel/mla_prolog_template_tiling_key.h`；`attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp` | Scenario3 两种 FP4 实例及 dispatch |
| `attention/mla_prolog/op_kernel/arch35/mla_prolog_comm_arch35.h`、`service_rms_norm_arch35.h` | FP4 类型/场景、输出类型和新量化 helper 引入 |
| 同目录 `service_matmul_mxfp4_arch35.h`；`vf/vf_dynamic_quant_mxfp4.h` | 新 `MatmulMxFp4SplitK`、scale 搬运、FP4 LoadData/Mmad；`DynamicQuantPerBlockMxfp4Vf` |
| 同目录 `kernel_mla_prolog_split_n_arch35.h` | 输出/scale/workspace 初始化、`MatmulSplitN`/`MatmulQcQrSplitN`、`RmsNormCqProcess`、`CastQcQrSplitN`、KV3 norm 舍入 |
| `attention/mla_prolog_v3/torch_extension/csrc/mla_prolog.cpp`；`mla_prolog.py` | `MakePackedMxfp4NzDescriptor`、`CheckMxfp4PackedWeight`、`mla_prolog` 的独立 V3 mode6 调用；`_meta_outputs` 返回 BF16 queryNorm；旧 mode 继续调用 V4 |
| `attention/mla_prolog_v3/tests/ut/op_host/*mxfp4*`；`tests/ut/op_kernel/test_mxfp4_storage_contract.py` | 新模式 Host UT、CPU 存储检查 |
| 本目录上级 `tests/reference.py`、`tests/run_a5.py`、`tests/test_*.py`、`scripts/run_a5.sh` | 独立布局参考、CPU 测试、A5 native/fused 对比和构建入口 |

## 固定基准与验证状态

| Repo / ref | 用途与关键来源 |
|---|---|
| `cann/ops-transformer@632dddba712a4e6cace3f8b44f198aef8a82ce3e` | 开发基线；V3 ACLNN、shared arch35 Prolog；`torch_extension/cann_ops_transformer/common/aclnn_common.h::CollectB4ShapeInfo` 等提供 FP4 descriptor 依据 |
| `cann/ops-nn@2a77283db46e6648ff47bc8277442cf9c721e3c2` | `matmul/quant_batch_matmul_v3/op_api/quant_matmul_v4_common.h::SelectNzK0Value`、`aclnn_quant_matmul_weight_nz.cpp::GetWeightNzShape`、`docs/aclnnQuantMatmulWeightNz.md` 确认 A4W4 NZ/scale；DynamicMxQuant/RMSNorm MX quant 提供 FP4 量化与 Cube 加载参考 |
| `vllm-project/vllm-ascend@0979baf25c0ee501ab0e4ef9b88ea7d9a6d29839` | 调研时固定 main；`vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py` 和 `attention/sfa_v1.py` 的 native/prolog/Indexer 消费契约；本次未修改 VA |
| `zai-org/GLM-5.2@f6142f127a14b58dc602592e996cd7d8ff139351` | `config.json` 提供 GLM6144/2048/192/512/64 和 interleave RoPE 形状基线；5.3 同结构依据用户确认 |

本地已执行：8项 Python contract/metadata 测试、4项 kernel CPU storage 测试、Python 语法和 shell 语法检查、`git diff --check`；另静态确认旧 key selectors/位宽、A2/A3 IR、原 A5 dtype 列、V4 API 和 tiling data 未改变。新增6组 C++ Host UT 已写入但未编译运行。详见 [CPU记录](../tests/reports/CPU_VALIDATION.md)、[Host记录](../tests/reports/HOST_IMPLEMENTATION.md)、[Kernel记录](../tests/reports/KERNEL_IMPLEMENTATION.md)。

后续本地回归及 A5 反馈见 [LOG.md](LOG.md)。A5 已完成 CANN 算子和 C++ wrapper 编译，native DQ 解码参考检查通过；自定义 Prolog 调用仍在排障，尚无融合算子精度或性能通过结果。完整旧模式 Host UT、graph capture/replay、GLM/Indexer 消费与模型端到端也未验收。通用 spec validator 的 FAIL/SKIP 如实保留，见 [SPEC_VALIDATION.md](../tests/reports/SPEC_VALIDATION.md)。当前 BF16 数值阈值是 runner 的初步门槛，不是已验证的模型精度承诺。

A5 在已拉取本分支的容器执行 `bash operators/mla_prolog_v3_mxfp4/scripts/run_a5.sh`，构建并加载本地自定义算子、独立 torch 扩展包后运行对比；具体环境要求、用例及计时范围见 [TEST.md](TEST.md)。首次设备验证重点是 SDK FP4 descriptor/LoadData 编译、NZ/scale 地址、ROUND 规则、AIC/AIV 同步、queryNorm 与 KV3 的 BF16 边界。
