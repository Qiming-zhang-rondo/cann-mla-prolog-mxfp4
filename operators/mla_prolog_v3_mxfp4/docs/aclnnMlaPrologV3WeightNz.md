# aclnnMlaPrologV3WeightNz：MXFP4 扩展提案

> **历史接口草案（v0.1）**：本文保留最初“尚未分配 mode/未实现”等提案文字，不应作为当前调用契约。现分支新增 `weight_quant_mode=6`，实际 FP4/NZ/scale、输出、cache 和支持范围见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。已有代码与测试入口，尚未通过 CANN 编译或 A5 验收；不代表上游已经支持。

| 版本 | 日期 | 修订 |
|---|---|---|
| 0.1 proposal | 2026-09-14 | 需求阶段接口草案；未实现、未 A5 验证 |

本文不覆盖上游现有正式接口文档，不代表上游已支持 MXFP4。原始头文件来源：`cann/ops-transformer@632dddba712a4e6cace3f8b44f198aef8a82ce3e:attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v3_weight_nz.h`。完整数值、shape、layout、模式范围与依据见 [REQUIREMENTS.md](REQUIREMENTS.md)。

## 产品支持情况

本扩展目标：A5、候选 Ascend950/DAV_3510。实际型号、CANN 与 torch_npu 版本待内网确认。A2/A3 新增模式不支持；现有 mode 的上游支持矩阵不改变。本机无 CANN/NPU，尚未编译或执行。

## 功能说明

扩展 MLA 前处理三组投影为 E2M1 MXFP4 权重/激活输入，E8M0 K32 block scale。Q 下投影与 RMSNorm 后，保留 BF16 norm 输出供 VA Indexer 使用，同时在内部重新量化为 MXFP4 供 Q 上投影使用；`weightUk` 继续 BF16。Query/RoPE 输出 BF16，cache 保持现有 BF16 或 FP8 per-tile，不产生 C4 cache。

量化矩阵的数学语义为 `sum_k(DQ4(A,Sa)*DQ4(B,Sb))`；RMSNorm 为 `gamma*z/sqrt(mean(z*z)+eps)`。内部 Q4 与外部 token Q4 的 block/round/saturation 规则一致，以 VA native 使用的 DynamicMxQuant 及已确认官方契约为基线。不得把 FP4 数据 byte container 直接当 INT8 数值矩阵。

## 函数原型

拟保留原签名，尚未增加/分配 MXFP4 mode 的正式数值：

```cpp
aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq,
    const aclTensor *weightUqQr, const aclTensor *weightUk,
    const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq,
    const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin,
    const aclTensor *ropeCos, aclTensor *kvCacheRef, aclTensor *krCacheRef,
    const aclTensor *cacheIndexOptional, const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional,
    const aclTensor *kNopeClipAlphaOptional,
    double rmsnormEpsilonCq, double rmsnormEpsilonCkv,
    char *cacheModeOptional, int64_t weightQuantMode,
    int64_t kvCacheQuantMode, int64_t queryQuantMode,
    int64_t ckvkrRepoMode, int64_t quantScaleRepoMode,
    int64_t tileSize, double qcQrScale, double kcScale,
    const aclTensor *queryOut, const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional,
    const aclTensor *queryNormOutOptional,
    const aclTensor *dequantScaleQNormOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMlaPrologV3WeightNz(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, const aclrtStream stream);
```

注意：公开 ACLNN 签名没有单独 `queryNormFlag` 参数，源实现从可选输出是否存在推导该语义；IR/torch 的 `query_norm_flag` 必须保持一致，不能向上述 ABI 凭空增加 bool 参数。

## 参数说明与约束（新 mode proposal）

| 参数 | 新 mode 契约 |
|---|---|
| tokenX | 逻辑 `[T,He]` E2M1 ND；连续 packed payload `T*He/2` byte；FP4 ACL descriptor 的实际表示待实现前确认 |
| weightDq / weightUqQr / weightDkvKr | 分别 `[He,Hcq]` / `[Hcq,N*(D+Dr)]` / `[He,Hckv+Dr]` 逻辑 FP4；权重为 FRACTAL_NZ，不能用普通 byte reshape 冒充 NZ |
| weightUk | `[N,D,Hckv]` BF16，保持既有格式 |
| dequantScaleXOptional | 新 mode 必需；E8M0 `[T,He/32]` 逻辑 scale |
| 三组 weight scale Optional | 新 mode 必需；逻辑 `[Nout,K/32]` E8M0；物理 pack/strides 独立验证 |
| rmsnormGammaCq / Ckv | `[Hcq]` / `[Hckv]` BF16；epsilon 沿用现有有效域 |
| ropeSin / ropeCos | 沿用 PA_BSND 的 token/RoPE shape 与 BF16 语义 |
| kvCacheRef / krCacheRef | 原地引用更新，mode0 BF16；mode3 沿用 FP8 per-tile packed cache及既有 kr dummy/repo 语义 |
| cacheIndexOptional | PA_BSND 必需；按既有 INT64 slot mapping 语义，非法 slot 行为不得变化 |
| weightQuantMode | 新增 `MXFP4_FULL_QUANT` 为符号提案；正式数值尚未分配，旧 mode 3 仍是 MXFP8 |
| kvCacheQuantMode | 新 mode 首期只覆盖0和3 |
| queryQuantMode | 新 mode 首期只覆盖0 |
| cacheModeOptional | 首期 `PA_BSND` |
| ckvkrRepoMode / quantScaleRepoMode / tileSize / kNopeClipAlphaOptional | mode3 保持当前 VA C8 布局参数与数学语义，不改为 FP4 scale |
| quantScaleCkvOptional / quantScaleCkrOptional | 仅在既有缓存模式需要时提供；新 mode 不引入新的 FP4 cache scale |
| smoothScalesCqOptional | 首期不开放额外 smooth 语义，按已确认 MXFP 类契约处理或 host 拒绝；不得静默忽略非空数值 |
| actualSeqLenOptional | 沿用已支持 cache/token 模式约束，未验收的新组合明确拒绝 |
| qcQrScale / kcScale | VA 首期固定1；一般值支持需额外验证现有缩放语义 |
| queryOut / queryRopeOut | BF16 `[T,N,Hckv]` / `[T,N,Dr]` |
| queryNormOutOptional | 若提供，BF16 `[T,Hcq]`，RMSNorm 后、内部 FP4 quant 前的值；不能从 FP4 解量化恢复 |
| dequantScaleQNormOutOptional | BF16 queryNorm 不需要 scale，null/空输出映射需与 IR/torch 一致；不使用假 scale |
| dequantScaleQNopeOutOptional | query_quant_mode=0 时按现有推导/可选输出契约处理，无新增 FP4 scale |
| workspaceSize / executor / workspace / stream | 沿用两段式执行契约；workspace 大小需根据真实 packed bytes、scale 和保留 BF16 norm 副本计算 |

He/Hcq 首期须 K32 对齐并满足实际 NZ/Cube 条件；额外对齐、最大 T/shape 范围须在后续 spec 锁定。超出范围、scale 缺失/shape 不匹配、错 dtype、错误 storage format、非 A5 芯片必须在 host 返回明确错误，不能静默走 FP8。空 T=0 等边界必须沿用或显式定义并测试，当前不宣称新增支持。

## 返回值

两段式函数返回 `aclnnStatus`。成功、无效参数、不支持平台/组合、workspace/运行错误的返回遵循仓库现有 ACLNN 规范；本提案不发明错误码数值。后续实现必须测试非法 mode、packed storage shape 和 scale mismatch 的失败路径。

## 调用示例与验收

调用示例留待实现及真实 FP4 tensor descriptor/NZ 转换 API 确认后提供；当前不给出无法编译的假 API 示例。新增 mode 未实现前不可作为可运行命令使用。

验收需包含同 FP4 输入下与 native 小算子拼接的 query/RoPE/queryNorm/cache 对齐、旧模式回归、scale/NZ 往返、VA Indexer 消费与 A5 设备计时。精度阈值候选、模型门槛未定项及运行环境缺口见需求文档。CANN 编译通过也不等于 A5 运行通过。
