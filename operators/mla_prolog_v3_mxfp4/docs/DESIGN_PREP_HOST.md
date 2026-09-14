# Host/API/tiling 证据准备

2026-09-14；本文件为证据准备，不宣称设计、CANN 编译或 NPU 测试通过。来源：`cann/ops-transformer@632dddba712a4e6cace3f8b44f198aef8a82ce3e`（以下 T）、`cann/ops-nn@2a77283db4`（以下 N）。按用户“进行下一步实现”推进，最终数学契约以本轮 spec 评审为准。

## 1. 已核实的 descriptor 与 NZ

- T `torch_extension/cann_ops_transformer/common/aclnn_common.h::CollectB4ShapeInfo`, `ConvertToAclStorageOffset`, `ConvertType(at::Tensor)`：FP4 用 byte tensor 承载时，连续内轴 shape ×2，跨内轴 strides ×2，storage offset ×2。ACL shape/stride/offset 以逻辑 nibble 元素计；payload 两项/byte。N `matmul/quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz.md::CreateAclTensorFp4` 直接使用完整逻辑 shape、byte allocation `(elements+1)/2`，打包 `low=element[2i]`, `high=element[2i+1]`, `(high<<4)|low`。这不是用 uint8 ACL dtype 调算子。
- N `matmul/quant_batch_matmul_v3/op_api/quant_matmul_v4_common.h::SelectNzK0Value`：A4W4 FP4 的块内连续逻辑维为 **64**；A8W4 FP4 返回32，是另一种布局，禁止混用。
- N `.../op_api/aclnn_quant_matmul_weight_nz.cpp::GetWeightNzShape`：非转置 `[K,N]` 的 A4W4 FP4 NZ storage shape 为 `[ceil(N/64),ceil(K/16),16,64]`。本期要求无 padding，严格 N%64=0、K%64=0，因此直接使用 `[N/64,K/16,16,64]`。
- T `attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp::FillCommonParamInfo` 正在按 `[N/C0,K/16,16,C0]` 检查已有权重。需要单独返回 FP4 C0=64；不能 `32 / ge::GetSizeByDataType(FP4)`，也不要将一个整数 dtype-byte map 的 FP4 大小填为1/0。
- T `.../mla_prolog_tiling.cpp::SetShapeInfo` 当前从4D weightDq storage 第0维与 dtype 推 Hcq，也必须改为相同 FP4 C0=64。
- A4W4 的实际 `aclnnNpuFormatCast` producer 与 ACL format subtype 需要 A5 roundtrip 验证。N 文档中指定 C0=32 的 A8W4 示例**不证明**本期 A4W4 producer 已正确；不能以成功的 format cast 调用代替布局断言。

## 2. Scale/输出契约建议

- T `.../mla_prolog_tiling_check.cpp::FillMxfp8FullQuantParamInfo`：Prolog scale 为连续 ND `[T,He/32]`、`[Hcq,He/32]`、`[N*(D+Dr),Hcq/32]`、`[Hckv+Dr,He/32]`，dtype E8M0。MXFP4 可使用相同的 K32 数学布局；非转置 QuantMatmul 的 `[K/64,N,2]` scale 不能直接 `.view(N,K/32)`，必须 reorder 为按每个输出通道连续 K-group 的格式。
- N `.../docs/aclnnQuantMatmulWeightNz.md` MX 表（约619行）给 x scale `[M,ceil(K/64),2]`、非转置 weight scale `[ceil(K/64),N,2]`，group `[1,1,32]`。因此 native baseline/VA loader 与 Prolog 的 scale conversion 必须可逆并独立测。
- 新 mode 返回 BF16 queryNorm `[T,Hcq]`，不是 FP4 的反量化恢复值。`dequantScaleQNorm` 为空，IR dtype FLOAT shape `[0]`，ACLNN API 调用参数 `nullptr`。API TensorHolder 内部创建该空 FLOAT descriptor；不能因为 weight mode 非0而强制输出 scale。
- `query_norm_flag=false` 的 queryNorm 同样为空 BF16 `[0]`。`query/queryRope` BF16；weightUk BF16；KV0/KV3 与既有 cache contract 相同。

## 3. mode 与旧 tiling keys

建议接口 mode=6，命名 `MXFP4_FULL_QUANT`，明确这是本分支新增值、未获上游分配。原0–5不改。host 组合 enum 追加16/17表示 KV0/KV3。

**推荐避免扩大 QUANT_MODE 的4bit字段**：`attention/mla_prolog/op_kernel/mla_prolog_template_tiling_key.h` 当前 SCENARIO 占2bit、合法列表0/1/2，新增 SCENARIO=3 专用于 MXFP4；其 QUANT_MODE field 用0/1分别表示 KV0/KV3。host `GenTilingKey` 将组合 enum16/17映射到这两个编码，device 对 Scenario==3 作 FP4 template dispatch。原0–15、原 scenario 与所有后续 flag 位保持原样；CV_MODE 必须仍位于模板参数末尾。新增选择仅 `MLA_PROLOG_VERSION == 3`，dtype 必须 FP4/E8M0，SplitM=0、CACHE_MODE=PA_BSND、CV1:2。

不能仅把16/17 cast进原4bit字段，会截断/碰撞。也不建议复制整份旧 key header；会产生长期同步问题。新增路径应有明确 host semantic enum ↔ template field 映射注释和对照测试。

## 4. 限制与防回归

新 mode 仅 DAV_3510、V3、2D token、PA_BSND、query mode0、KV mode0或3、do_rope=true、qcQrScale=kcScale=1；三量化权重 FP4 NZ，外部四个 E8M0 scale 必需。He/Hcq及各 projection N 严格64对齐；weightUk仍服从既有 `[N,D,Hckv]` 约束。本期 SplitM关闭；T范围以 kernel 资源检查结论定，不能任意开放新 prefill范围。

当前 `CheckBaseShape` / `CheckHcqSize` / `CheckDSize` 允许 He∈{1024,2048,3072,4096,5120,6144,7168,7680,8192}、Hcq∈{1536,2048}、Hckv=512、D∈{128,192}、Dr=64、N∈[1,128]、Nkv=1。**GLM-5.2/5.3 必须覆盖 He6144/Hcq2048/D192/Dr64/Hckv512，N按TP切分**，不得仅保留DeepSeek的Hcq1536/D128。父agent已核对 GLM-5.2 config ref `f6142f127a14b58dc602592e996cd7d8ff139351`。

KV0为BF16 `[blockNum,blockSize,1,512]`，KR为BF16 `[blockNum,blockSize,1,64]`；KV3为FP8E4M3容器 `[blockNum,blockSize,1,656]`，KR为空BF16 `[0]`，两个repo modes都为1、tile_size128。656包含512个FP8、64个BF16 RoPE（128byte）和4个FP32 scale（16byte）；不是656个统一FP8数值。BlockSize∈[16,1024]且为16倍数。

`actual_seq_len`、smooth scale、k_nope_clip_alpha、额外 per-token/per-tensor query scale 都不得被新模式静默忽略。新模式 KV3 的 repo modes 由现有 CheckRepoMode/CheckDtileSize 检查，并依据 kernel 已有 per-tile逻辑保留。

V4 ACLNN 位于V3目录，但其 mode validity独立；保持不开放mode6，不能误改到V4。shared host 按 opType==V3+arch3510+mode6隔离；旧 V1/V2 与 V3/V4 0–5的dtype列表、形状和tiling key不变。

## 5. 文件/function 清单

| 文件（T） | 必要修改 |
|---|---|
| `attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.cpp` | `CheckWeightQuantModeValidity`、`CheckKvCacheQuantModeValidity` 新mode；入口限制query mode；queryNorm TensorHolder BF16+空scale；不改函数签名 |
| `attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp` | ascend950配置成对新增KV0/KV3两个类型列；四FP4输入、四E8M0scale；BF16 queryNorm+FLOAT空scale；A2/A3配置不变 |
| `attention/mla_prolog_v3/op_host/mla_prolog_v3_infershape.{cpp,h}` | 新mode常量、`SetQueryNormShape` 空scale、`InferDataTypeMlaPrologV3` BF16 queryNorm；新模式不走“输出dtype等于量化weight”旧默认分支 |
| `attention/mla_prolog/op_host/mla_prolog_tiling.h` | weight enum追加6、组合enum追加16/17、正反查表；FP4 NZ C0 helper若放此处则只有FP4分支改变 |
| `attention/mla_prolog/op_host/mla_prolog_tiling_check.{cpp,h}` | 单参FP4接受仅V3 A5；`CheckAttrsRange`和新组合限制；`CheckScenarios`；`FillCommonParamInfo` C0；新MXFP4 filling与queryNorm；强制shape/group/optional contract |
| `attention/mla_prolog/op_host/mla_prolog_tiling.cpp` | `SetShapeInfo` FP4 C0、`SetScenarioInfo`/splitM gating、dequant optimization dispatch、workspace检查、`GenTilingKey`新scenario映射 |
| `attention/mla_prolog/op_kernel/mla_prolog_template_tiling_key.h` | 新SCENARIO值和V3 FP4严格组合；旧编码不移动 |
| `attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp`（kernel agent） | Scenario3两类FP4 dispatch；shared kernel实质数据搬运/量化由kernel agent负责 |
| `attention/mla_prolog_v3/tests/ut/op_host/test_mla_prolog_v3_infershape.cpp`、`test_mla_prolog_v3_tiling.cpp` 或新测试文件 | 两种cache、两种queryNorm flag；拒绝A3/shape/scale/new mode非法组合；旧key/旧dtype回归 |

## 6. 实施前待闭合

1. kernel负责人确认C0=64的 N分块与scale load、内部queryNorm BF16副本的workspace需求。
2. spec锁定首期shape及新的mode数值/tiling映射；上述建议与最终spec保持一致。
3. A5验证 CANN头文件存在FP4 dtype且注册编译成功；本地macOS不能完成此项。Host C++ tests亦需要仓库CANN/gtest构建环境，不宣称已运行。
