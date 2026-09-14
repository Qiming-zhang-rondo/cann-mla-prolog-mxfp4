# MlaPrologV3 MXFP4 底层设计准备

状态：2026-09-14 源码证据与实施约束，尚无 CANN 编译或 A5 运行结果。本文件只补充设计材料，不改变业务源码。

## 固定依据

- CANN ops-transformer `632dddba712a4e6cace3f8b44f198aef8a82ce3e`，本地 `work/ops-transformer-mla-mxfp4`。
- CANN ops-nn `2a77283db46e6648ff47bc8277442cf9c721e3c2`，本地 `work/ops-nn-reference`。`matmul/common` 未展开到 sparse 工作区的文件用 `git show HEAD:<path>` 读取。
- cannbot-skills `3074681ba927916f99f5a9ac808b4e6797934ce5`，已读 `ops/ascendc-api-best-practices/SKILL.md` 和 `references/api-loaddata.md`。API skill 对 MXFP4 给出 K tile 的 64 元素对齐要求，实际参数解释以以下正式算子源码交叉验证，不能把文档里的 `sizeof(dtype)` 字面式用于 C++ 的半字节类型。

## 已确认的 FP4 Cube 与搬运路径

| 依据 | 可用结论 |
|---|---|
| ops-transformer `attention/quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_cube_arch35.h`：`QLIV2Matmul` 类型声明、`LoadQueryToL0a`、`LoadKeyToL0b`、`ComputeL0c` | FP4 GM/L1 按 uint8 packed 字节存放；L1 以 `fp4x2_e2m1_t` 语义视图传给 MX `LoadData`；L0 同为 FP4，再直接 `Mmad`，累加 float。没有先反量化至 FP8/BF16。数据 `kStep=K/64`；`MmadParams.k=K` 仍为逻辑值；scale 每 K32 一项、两个 E8M0 合作一个 L1 scale 单位。 |
| ops-nn `matmul/common/cmct/block/block_mmad_mx.h`：`CopyInA1`、`CopyInB1` | 即使 tensor 类型是 `fp4x2_e2m1_t`，ND2NZ 的 `dValue` 和 `srcDValue` 仍显式除2；行数保持逻辑行数。不能仅改模板 dtype。 |
| 同文件 `CopyInB1WeightNz` | 非转置数学权重 `[K,N]` 的 NZ 物理读按 `N/C0` 个块，每块 `align(K,16)*C0/2` 字节；FP4 C0=64。权重沿 N 连续 nibble，数据 scale 沿 K32 分组，两者布局不同。 |
| 同文件 `CopyInL0B`（约566行） | `[K,N]` 权重的 L1→L0B 使用 `ifTranspose=true` 的 MX `LoadData`；FP4 确实有正式实现。数据 `kStep=ceil(N/64)`、scale `yStep=ceil(K/64)`，scale `xStep=ceil(N/16)`。N/16 非4倍时需要分段 `mStep=4` 搬运，后续段的 MX scale 参数全0避免重复写scale。 |
| 同文件 `CopyBInL1`；ops-nn `quant_batch_matmul_v3_base.h::GetSizeWithDataType` | FP4 typed tensor 的下标沿用逻辑元素偏移，而字节数 helper 为 `shapeSize/2`。不要把 typed 下标与 uint8 下标混用：采用前者时 buffer 字节偏移转元素为 `bytes*2`，采用后者时逻辑偏移转 byte 为 `/2`。 |

首期建议限定三个 FP4 GEMM 的 K、N 及分核/tile 起点为64倍数，baseK 64倍数，tail N 也64倍数。GLM 常见 64维 RoPE 尾可表达，但不能仅检查总 N 而漏掉 host 的 `mm1SingleCoreN/mm2SingleCoreN` 切片。这样可以采用正式 `CopyInL0B` 的完整步长分支，暂不引入非64 N 的复杂分段搬运。具体可接收 GLM shape 必须和 host 设计统一。

### 实施建议：显式区分逻辑元素与字节

增加 FP4 专用的物理换算/搬运实现比机械修改所有旧 FP8 条件稳妥。可沿用现有调度和同步，但需统一选择存储语义：

1. 若内部全用 `GlobalTensor/LocalTensor<fp4x2_e2m1_t>`，GM tensor 下标继续使用逻辑偏移；所有分配字节数、L1/L0 ping-pong 偏移通过 FP4-aware helper 换算，ND2NZ 与 raw byte DMA 字段显式换算。
2. 若采用 QLI 的 uint8 存储策略，全部 GM/L1 tensor 下标都为 packed byte，只有 Cube `LoadData/Mmad` 边界转为 FP4 tensor。不能拿 uint8 的数值类型触发 `MLAPType` 的 int8 量化分支。

不能以 `sizeof(fp4x2_e2m1_t)==1` 推断一个逻辑值占1字节。原 `GetC0Num`、`L1_A_SIZE/sizeof(T)`、`aOffsetUnit=mSize*baseK`、`bOffsetUnit=GetC0Num<T>()*baseK`、`tensorBGm[para.k*nL1Offset]` 属于不同单位，必须分别审核。

## 内部 RMSNorm→Q4 与 BF16 query_norm

真实可提取的最小 quant 参考为 ops-nn `norm/rms_norm_dynamic_mx_quant/op_kernel/arch35/rms_norm_dynamic_mx_quant_common.h`：

- `ComputeMaxExpOCP`（约196行）：BF16 指数按32元素组归约。
- `ComputeScaleOCP`（約257行）：E2M1 的 exponent 常量 `0x0100`，即 emax=2；最大指数先钳到该值，sharedExp 差右移7后保存为 E8M0。Inf/NaN 的 scale 为 `0xff`；sharedExp==0 的内部 reciprocal 置0。必须按正式实现保留小数/异常数规则，不复用现有 Prolog FP8 的不同 zero-mask 时序。
- `ComputeDataMxfp4General`（约535行）：BF16 乘 BF16 reciprocal，`Interleave` 后 `Reg::Cast<fp4x2_e2m1_t,bfloat16_t,castTraitRM<RoundMode::CAST_ROUND>>`，以 `DIST_PACK4_B32` 存到 int8 字节缓冲。`full_load.h::ComputeMxQuantAndOutput`（约301行）展示正式调用关系。
- `norm/norm_common/op_kernel/mx_quant_cast_traits.h` 定义 CastTrait。只提取需要的 BF16→E2M1 + OCP 子集并保留来源/版权，避免把整个外部 operator 类或所有 dtype/roundMode 复制到 Prolog。
- `quant/dynamic_mx_quant/docs/aclnnDynamicMxQuant.md` 明确 FP4 `scaleAlg=0`，支持 `rint/floor/round`；VA native W4A4 使用 `round`，因此内部不能使用原 FP8 的 rint。tie、正负零的位级处理由真实 `CAST_ROUND` 实现决定，需 A5 与 native DynamicMxQuant 的字节比较确认，CPU 黄金不能擅自假设 ties-to-even。

推荐新增 `DynamicQuantPerBlockMxfp4Vf`。输入是已经完成 RMSNorm 舍入的 BF16，输出 payload `row*col/2` bytes、scale `row*col/32` bytes。原 `DynamicQuantPerBlockMxfp8Vf` 与旧模式保持行为。

`kernel_mla_prolog_split_n_arch35.h` 的现有 `rmsNormCqResGm_` 同时承担内部 Q 上投影激活及可选 `queryNormOut`。`OutputInit` 在 flag=true 时把它指向外部输出，`WorkspaceInit` 也因此不分配内部激活空间；这与新契约冲突。必须做以下拆分：

1. FP4 mode 始终为内部 Q4 激活和 E8M0 scale 分配 step workspace，与 flag 无关；内部 workspace 偏移按每step复用，不能再因为 flag=true 随全局 token 增长。
2. 单独新增 BF16 queryNorm GM tensor，形状 `[T,Hcq]`。RMSNorm FP32结果先 cast到 BF16 UB，直接保留/写出该副本，再由同一 BF16 UB 产生内部 Q4，绝不能从 FP4 反量化恢复 queryNorm。
3. RMSNorm helper 显式接收/返回 BF16 UB 副本，避免 caller 偷用 helper 私有 scratch 字节偏移。queryNorm写出完成之前不能重用这段 scratch；MTE3/V 同步纳入已有事件协议。
4. 新mode `CopyQueryNormScale` 不写外部 scale，API/IR 返回空scale；`CopyDequantScaleCq` 仍写内部 scale，保持下个 GEMM 的scale消费者。

### 与 native 的舍入边界

新mode三次 FP4 GEMM 必须 float L0C 累加，但建议通过既有 Fixpipe F322BF16 写 BF16 GM，匹配 native QuantMatmul 的 BF16 输出；然后做 RMSNorm/RoPE/下一段 BF16 matmul。现 `MLAPType::GetMatmulOutType_t` 对FP8返回float，不能为了FP4复用而直接把它加入该条件。若采用float GM→RMSNorm，将引入区别于native的精度边界，即使 queryNorm 最终dtype为BF16也不等价。此建议需要纳入spec黄金，不是已测精度结论。

## 推荐改动文件与审核点

| ops-transformer 路径 | 改动/审核内容 |
|---|---|
| `attention/mla_prolog/op_kernel/arch35/mla_prolog_comm_arch35.h` | FP4语义别名；MX判断helper；新增组合enum；明确矩阵中间输出BF16而L0C float；保持 FP8 原类型关系。 |
| `.../service_matmul_arch35.h`（可辅以独立FP4 helper头） | FP4 C0=64、ND/NZ DMA参数、L1/L0 buffer单位、FP4 LoadData和Mmad、K循环offset与E8M0 scale同步。优先限制可处理tile再拓展tail。 |
| `.../service_rms_norm_arch35.h`、`.../vf/vf_dynamic_quant.h`或新VF头 | 真BF16→FP4 OCP量化；独立BF16 queryNorm副本；scratch大小和生命周期。 |
| `.../kernel_mla_prolog_split_n_arch35.h` | 新模式覆盖AIC/AIV选择，独立queryNorm输出及packed workspace；token/weight/scale偏移；跳过已由Cube完成的反量化；不把FP4错误导入INT8 per-token路径；cache复用既有BF16/FP8 per-tile。 |
| `attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp` | 仅V3 arch3510新模式实例化；不开放未适配的split-M/其他版本。 |
| host及API相关文件 | 由host方案负责；必须与上面workspace和64倍数tile约束一致。 |

host提出的新 `Scenario=3`、`QuantMode=0/1` 分别表达 FP4+KV0/KV3 可行；原Scenario为0/1/2且字段2bit，旧QuantMode字段无需从4bit增宽。host内部组合enum可16/17，但编码成key前必须转换，device根据新scenario dispatch，不能直接把16塞入4bit。这样可保留已有key位位置和既有0–15模式。

## 必须覆盖的验证

- 源码/CPU：每个物理偏移单位、buffer上界、NZ pack往返、跨K/N tile scale选择、新key唯一性、旧key值不变。CPU测试证明这些数学/存储模型一致，不证明设备指令正确。
- A5基础：FP4输入/weight NZ构建；Mx LoadData+Mmad 最小GEMM与native QuantMatmul；K跨多个tile、N跨两个分片、尾64、M=1/非16倍数。
- A5量化：内部Q4 payload/scale与native DynamicMxQuant逐字节比，包含zero/subnormal/临界舍入/溢出/Inf/NaN；BF16 queryNorm与量化前值直接对比。
- A5完整：queryNorm flag true/false、KV0/KV3、step workspace复用、无效cache slot不写、捕获重放。query/cache/RoPE分别与同精度边界native黄金比较，再验证GLM Indexer消费者。
- A5性能：融合入口包含外部Q4与必要layout成本，对比native同shape；不承诺在未测的硬件环境已提速。

结论：已有真实FP4 Cube和内部vector quant源码足够支持可实施设计，不存在已知必须退回FP8/BF16 Cube的障碍；工作量集中在物理布局、scratch/输出拆分及旧模板路径选择。CANN头文件版本、真实编译、硬件异常数/packed位序一致性和端到端精度/性能仍必须内网验收，不能报告“已经支持并通过”。
