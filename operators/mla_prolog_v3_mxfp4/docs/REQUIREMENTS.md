# MlaPrologV3 W4A4 MXFP4 扩展需求

> **历史需求提案（v0.1）**：下文保留立项时的待定项和源码基线描述，不代表当前开发状态。现分支已写入 mode6、NZ/scale、host/kernel 和测试入口；准确实现范围、编号及已测/未测状态以 [IMPLEMENTATION.md](IMPLEMENTATION.md) 为准。尚未通过 CANN 编译或 A5 验收。

| 版本 | 日期 | 作者 | 内容 |
|---|---|---|---|
| 0.1 proposal | 2026-09-14 | Codex | 源码需求分析；未进入 spec、设计或开发 |

**状态：供评审的扩展提案，不是上游已支持声明。** 本文中的“要求”描述拟实现行为；现有行为以证据表为准。技术开发记录见 [LOG.md](LOG.md)。

## 1. 背景与边界

用户要求在 A5 上将 GLM-5.3 的 MLA Prolog V3 扩展为支持 MXFP4 输入；用户确认 5.3 与 5.2 同结构。本期指 MLA 投影 W4A4 MXFP4，既不修改 Indexer compute op，也不新增 FP4 KV cache。模型具体 checkpoint 的量化描述、TP 切分后实际 shape 仍需内网确认，不能凭模型名称认定所有投影都为 W4A4。

数值和性能基线是 VA 的 native 链：BF16 hidden states 经 `npu_dynamic_mx_quant` 与 `npu_quant_matmul` 完成量化投影，RMSNorm 后向 Q 投影与 Indexer 分别提供激活。融合要求减少算子调用/中间访存，同时保留两条消费者的数据语义。不能把输入从 FP8 改为 FP4 后就宣告完整支持。

## 2. 固定源码证据

以下 CANN 路径均相对 `cann/ops-transformer`，ref **`632dddba712a4e6cace3f8b44f198aef8a82ce3e`**，本地 `work/ops-transformer-mla-mxfp4`。VA 路径相对 `vllm-project/vllm-ascend`，ref **`0979baf25c0ee501ab0e4ef9b88ea7d9a6d29839`** (`origin/main`)，不以本地 Indexer 实验分支代替官方基线。

| 编号 | repo / path / function | 现有行为与需求影响 |
|---|---|---|
| C1 | CANN `attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.h` / `aclnnMlaPrologV3WeightNzGetWorkspaceSize` | 已有两段式接口、四组输入反量化 scale、可选 queryNorm 输出；可评估在现有 ABI 扩展 mode |
| C2 | 同目录 `.cpp` / `CheckWeightQuantModeValidity`, `CheckKvCacheQuantModeValidity`, `aclnnMlaPrologV3WeightNzGetWorkspaceSize` | DAV_3510 当前只接受 weight mode 0–5；创建 queryNorm 的 TensorHolder dtype 由 mode 决定；新增 dtype 必须贯穿 API 分配和检查 |
| C3 | CANN `attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp` / `MlaPrologV3`; `mla_prolog_v3_infershape.cpp` / `SetQueryNormShape`, `InferDataTypeMlaPrologV3` | 注册、推导目前有 MXFP8，没有 MXFP4；MXFP8 queryNorm 逻辑 shape `[T,Hcq]`，scale `[T,Hcq/32]`；不能只改 kernel |
| C4 | CANN `attention/mla_prolog/op_host/mla_prolog_tiling.h` / `WEIGHT_QUANT_MODE`, `QUANT_MODE` | 公共枚举包含 MXFP8/FP8/HiF8，没有 MXFP4；weight enum 和组合 tiling enum 是两套编号，不得混用 |
| C5 | CANN `attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp` / `FillWeightAndNormShapes`, `FillMxfp8FullQuantParamInfo`，NZ storage-shape 校验段 | 三个权重的逻辑矩阵 shape 与 scale 语义明确；4D NZ 校验使用 `32 / ge::GetSizeByDataType(dtype)`，FP4 的位宽不能直接假定该整数公式适用 |
| C6 | CANN `attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp` / `mla_prolog_v3` | MXFP8 实例以 FP8E4M3 数据、E8M0 scale 实例化 shared arch35 kernel，当前无 FP4 实例 |
| C7 | CANN `attention/mla_prolog/op_kernel/arch35/service_rms_norm_arch35.h` / `RmsNormDynamicQuant` | RMSNorm FP32 计算后转 BF16，再调用 `DynamicQuantPerBlockMxfp8Vf<...,fp8_e4m3fn_t>`；必须增加内部 MXFP4 再量化，不能只接受外部 FP4 token |
| C8 | CANN `attention/mla_prolog/op_kernel/arch35/service_matmul_arch35.h` / `LoadL1AAndScale`, `LoadL1BAndScale`, `LoadDataL1ToL0Mxfp8`, `MatmulL0` | 数据/scale 搬运、L1/L0 布局和 Cube 调用都属于 MXFP4 能力边界；dtype 注册不证明这些环节可复用 |
| V1 | VA `vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py` / `AscendW4A4MXFP4DynamicLinearMethod.apply`, `process_weights_after_loading` | native 使用 `float4_e2m1fn_x2`、E8M0 scale、`round_mode="round"`、group size；权重 scale 经配对与转置处理，不等于 Prolog 的 `[N,K/32]` 格式 |
| V2 | VA `vllm_ascend/attention/sfa_v1.py` / `AscendSFAImpl._resolve_preprocess_type`, `_sfa_preprocess_prolog_v3` | 当前入口仅准入既有类型；MXFP8 mode=3；`query_quant_mode=0`、`cache_mode="PA_BSND"`、KV mode=0 或3；W4A4 尚无 Prolog 分支 |
| V3 | 同文件 / `_sfa_preprocess_prolog_v3`, `forward` 及 Indexer 的 `wq_b` 消费分支 | native 提供 RMSNorm 后 BF16 `q_c`；Prolog 返回的 `q_c` 会 `.view(-1,q_lora_rank)`，tuple 分支只专门识别 FP8 block scale；packed FP4 不能直接套用 reshape/消费规则 |

## 3. 环境与调用方式

目标为用户 A5、候选 Ascend950/DAV_3510（arch35）。依据 cannbot `ops/npu-arch/references/npu-arch-guide.md`，该架构具有 MXFP4 Cube 能力，但不能据此推导所有 Ascend C API、NZ 搬运接口或 CANN 版本均可用。准确 SoC、CANN/torch_npu 版本、编译宏和接口支持需由用户内网环境确认。

本机 Darwin arm64，没有 CANN、NPU/asys/DSMI；见 [环境问题](../issues/issue_20260914_local-cann-npu-unavailable_01.md)。本地静态检查不能替代 CANN 编译及 A5 运行。

| 调用方式 | 本期目标 |
|---|---|
| ACLNN 两段式 | 必需，真实 A5 单算子精度与性能验收入口 |
| GE IR 注册、shape/dtype 推导 | 必需，新增 mode 与 ACLNN contract 一致 |
| torch_npu / VA | 必需的下游接入验收；需确认 torch_npu binding 能传新 mode 与 FP4 ACL 描述，否则增加适配，不能宣称仅 CANN 修改已完成 VA 支持 |
| torch.compile/GE 动态图 | 不因注册自动宣称支持；本期先以现有 VA NPU graph capture/replay 验证部署相关入口，额外图后端另行验收 |

## 4. 数学与数据契约（proposal）

记 `T=B*S`，`He` 为输入隐藏维，`Hcq` 为 Q LoRA rank，`Hckv` 为 KV LoRA rank，`N` 为本 TP rank 的头数，`D` 为 Q nope head dim，`Dr` 为 RoPE dim。首期以 VA 合并 token 的二维 `[T,He]` 入口为目标。

定义 `DQ4(Q,S)` 为按归约 K 维每 32 个 E2M1 值共享一个 E8M0 scale 的反量化；`Q4(A)` 返回 packed E2M1 与 E8M0 scale。round/saturation、subnormal、零 scale、非有限值规则必须在后续 spec 中依据实际 DynamicMxQuant 官方实现锁定，本文不编造位级规则。

1. 外部 producer 对 BF16 `X` 做 `Q4(X)`，传入 tokenX 与 dequantScaleX。三组量化权重分别对应 `W_DQ[He,Hcq]`、`W_UQ_QR[Hcq,N*(D+Dr)]`、`W_DKV_KR[He,Hckv+Dr]`。
2. `Zq = DQ4(tokenX,Sx) @ DQ4(W_DQ,Sdq)`；`Cq = BF16(RMSNorm(Zq,gamma_q,eps_q))`。以 native 投影 BF16 输出和 norm 的舍入边界为对齐基线，实际融合是否保留该边界须在 spec/精度实验中明确。
3. **保留未经过 FP4 再量化的 Cq，作为 query_norm 的 BF16 输出。** 内部 Q 上投影另取 `Q4(Cq)`，得到 `U = DQ4(Q4(Cq)) @ DQ4(W_UQ_QR,Suq)`，拆出 Q nope 与 Q RoPE 分量。
4. Q nope 再与 **BF16** `weightUk[N,D,Hckv]` 计算；RoPE 对 Q RoPE 分量执行现有位置编码。沿用现有 qcQrScale 语义；VA 本期调用固定为1，扩展不擅改其一般语义。
5. `Zkv = DQ4(tokenX,Sx) @ DQ4(W_DKV_KR,Sdkv)`，拆分后 KV latent 执行 RMSNorm 与 kcScale，RoPE 部分执行既有 RoPE，按 cacheIndex 写入既有 BF16 或 FP8 per-tile cache。KV 不经过 MXFP4 压缩。

### 4.1 输入/输出规格

| 张量 | 数学逻辑 shape | dtype / layout 目标 |
|---|---|---|
| tokenX | `[T,He]` | E2M1 FP4；ND packed 存储；外部携带真实 FP4 ACL dtype/逻辑维信息 |
| weightDq / weightUqQr / weightDkvKr | 上述三组 `[K,Nout]` | E2M1 FP4，FRACTAL_NZ；必须由受支持的转换生成、保持逻辑 shape 与物理 storage shape 分离 |
| weightUk | `[N,D,Hckv]` | BF16，保持现有受支持格式和 strides；本期不量化 |
| dequantScaleX | `[T,He/32]` | E8M0、每 K32 一项，逻辑 scale；物理 scale pack 契约待绑定核实 |
| dequantScaleWDq / WUqQr / WDkvKr | `[Hcq,He/32]` / `[N*(D+Dr),Hcq/32]` / `[Hckv+Dr,He/32]` | E8M0，与 C5 的 MXFP8 逻辑 block 语义对齐；不得直接传 VA 配对转置后的存储形状 |
| RMSNorm gamma、sin/cos | gamma `[Hcq]` / `[Hckv]`；sin/cos 与现有二维 token/RoPE 规则一致 | 首期 BF16、保持现有约束 |
| query / queryRope | `[T,N,Hckv]` / `[T,N,Dr]` | BF16，`query_quant_mode=0` |
| queryNorm（flag=true） | `[T,Hcq]` | **BF16 ND，非 FP4 packed，非 FP4 反量化恢复值** |
| dequantScaleQNorm | 空/不提供 | queryNorm 为 BF16 时不需要；IR 空 tensor 与 torch 返回 None 的桥接须验证 |
| KV/kr cache | 现有 PA_BSND shape / packed C8 容器 | mode0 保持 BF16；mode3 沿用当前 FP8 per-tile 布局、scale 和 alias 语义；不新增 FP4 cache |

### 4.2 Packed、NZ、scale 的强制约束

- 二个逻辑 FP4 值占一个 byte；连续 ND `[T,He]` payload 为 `T*He/2` 字节（He 偶数）。byte 容器最后一维为 He/2 不能据此把算子的隐藏维判成 He/2。
- FP4 nibble 次序、NZ 块排列/补齐维、scale pair 排列是独立 contract，后续必须给 producer→ACL desc→kernel→consumer 的往返样例并在 A5 对齐。当前 8-bit NZ storage shape 公式不得机械复制为 FP4 结论。
- 首期 group=32，He/Hcq 必须满足 group 对齐及实际 Cube/NZ 额外对齐要求。具体 NZ 对齐数尚未锁定，不能自创常量；unsupported shape 必须在 host 明确拒绝。非对齐 K 的补零与 scale padding 只有在完整实现并测试后才能开放。
- VA loader 当前量化权重存储、转置和 scale `(Kgroups/2,Nout,2)` 表达需显式转换，不能把 `view` 当成 NZ 转换或 scale 重排。
- 外部 quantX 与内部 norm 后 Q4 必须使用同一 E2M1/E8M0 block 规则。矩阵归约结果仍需 BF16/FP32 正常数值累加；FP4 只作为量化数据，不把累加器设为 FP4。

### 4.3 本期 mode 覆盖（proposal）

新增独立 `weightQuantMode=MXFP4_FULL_QUANT`（**符号提案，数值待接口评审，不指定 6 已合法**）。组合 tiling keys 同样待定义，保留既有0–5语义和旧组合编号。

| 项 | 首期拟覆盖 |
|---|---|
| 激活/三组量化权重 | W4A4 E2M1，K32 E8M0；保持 weightUk BF16 |
| `kv_cache_quant_mode` | 0（BF16）及3（现有 FP8 per-tile），两者均须验收；其余新 mode 组合明确拒绝 |
| `query_quant_mode` | 0；不增加 FP4 queryOut |
| `cache_mode` | `PA_BSND`，覆盖 VA decode/spec-decode 使用场景 |
| `query_norm_flag` | true/false 均测试；true 返回前述 BF16 数值副本 |
| `qc_qr_scale` / `kc_scale` | 初期 VA 场景均为1；其他值是否开放须明确复用现有契约并新增验证 |
| 非本期 | A2/A3、MXFP4 KV cache、Indexer QLI 路径、E1M2、任意 block size、未验收的 prefill/大 M split-M 快速路径 |

### 4.4 精度标准与验收分层

不可把相对于 FP32 的 FP4 固有量化误差与融合实现误差混在同一阈值下，也不可声称已有 GLM 精度通过。

1. **存储与接口**：确定 nibble/block/NZ 重排后，pack/unpack、scale 重排、queryNorm false 空输出、cache alias、无效 slot 不改动等必须符合确定的位级契约。
2. **算子实现误差**：对相同 FP4 权重和相同外部/内部 quant 规则，以 A5 native 小算子拼接为主基线，独立高精度反量化数学实现作辅助；分别比较 query、RoPE、queryNorm、cache 和 cache scale。BF16 输出暂建议使用 skill `ops-precision-standard/references/float_compute.md` 的 `rtol=atol=2^-6`、matched ratio>=0.99、max error<=max(1,32ULP) 作为候选门槛；**这是待评审 proposal，不是已取得模型认可的融合容差**。FP4 字节不能直接套用 BF16 容差。
3. **消费者与模型**：queryNorm 与 native BF16 norm 对齐后继续运行相同 Indexer 投影/选择，检查 Top-k 变化；GLM-5.2/5.3 实际样本需验证无额外可接受范围外损失。数据集、Top-k/输出 token 差异及模型指标阈值尚未由用户提供，必须在内网验收前锁定，不能挪用此前 QLI recall 阈值。
4. **性能**：同硬件、shape、权重、warmup、同步、stream/graph 模式比较 native W4A4 与 fused MXFP4；报告设备计时 P50/P90、host 发射耗时、workspace、峰值内存和端到端前处理耗时。必须包含输入 quant/必要 layout 转换的完整融合入口成本，不能仅比较内核排除新增转换。提速比例待实测，不预设“FP4必然2倍”。

## 5. API 与 IR

拟保持已有 `aclnnMlaPrologV3WeightNz{GetWorkspaceSize,}` 函数签名，通过新增 weight mode 扩展 dtype 与行为；详见 [接口提案](aclnnMlaPrologV3WeightNz.md)。新增 mode 的 queryNorm BF16 / 空 scale 必须同步到 API TensorHolder、IR dtype/shape、host optional-output 检查和 kernel store，且旧 mode 不变。

IR 名称保持 MlaPrologV3。本期不额外发明 graph 属性；如果 torch binding 无法表达 FP4 ACL tensor 描述或空 scale 输出，则先完成兼容性设计再决定 ABI/binding 变更。新增 mode 不能在旧 CANN 上静默回落为其他 quant mode。

## 6. 可行性、风险与进入下一阶段条件

架构 MXFP4 能力、VA 已有 QuantMatmul 入口、现有 Prolog MXFP8 全量化链使本目标具备源码扩展基础；**尚不能认定复用现有模板即可编译或正确执行**。主要风险是 packed bit-addressing、NZ 与 scale 双布局、内部 norm 再量化、queryNorm 额外量化损失、shared MLA 实现回归、内网 CANN/torch_npu 版本不匹配。

共享 `attention/mla_prolog/op_kernel/mla_prolog_template_tiling_key.h:75` 的 QUANT_MODE 当前使用 `ASCENDC_TPL_4_BW` 且0–15已列满；新增组合超过现有位宽，扩大位宽可能移动后续标志编码。必须评估 host/device tiling-key 一致性和 V1/V2/V3/V4 共用实现回归，不能承诺“扩大 bit 完全兼容”。

后续需要修改的代码范围至少涉及 C1–C8 对应 API/IR/shape/tiling/arch35 quant/matmul/kernel dispatch、测试与文档；VA 另需 type gate、MXFP4 输入/权重/scale producer 及 BF16 queryNorm 消费分支。此列表是 gap inventory，不是已确认实现设计。

另据 VA 同 ref `sfa_v1.py::_get_fused_type_unsupported_reasons`，`qk_rope_head_dim==0` 会强制 native；因此 CANN 新模式完成后也不能无条件宣称 GLM 自动进入 Prolog。NoPE/RoPE 和模型 config 必须以用户内网实际模型为准并验证该 gate。

需求评审后方可进入 spec/设计。设计前需核实 FP4 CANN tensor/storage contract、可用 MXFP4 Ascend C 指令/API 和 NZ 搬运能力；没有这些证据不得用未定义 API 拼实现。A5 编译、真实算子数值/性能与 VA 模型检查均由用户内网补齐。当前没有任何 NPU 通过结果。
