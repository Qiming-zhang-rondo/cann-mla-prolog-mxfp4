# MXFP4 kernel实现与本地验证

2026-09-14。实现基线 `632dddba712a4e6cace3f8b44f198aef8a82ce3e`；本报告记录源码和 CPU 检查结果。

## 已实现

- 新 `arch35/service_matmul_mxfp4_arch35.h::MatmulMxFp4SplitK`：FP4 typed GM入口按逻辑offset取址，再转uint8 byte DMA。A为ND、B为C0=64的NZ；每个baseK独立读E8M0 scale并通过MX LoadData装入L0，以FP4直接Mmad，float L0C完整累加后一次Fixpipe到BF16。没有float/FP8展开代替FP4 Cube，也没有BF16 atomic partial sums。
- 新 `arch35/vf/vf_dynamic_quant_mxfp4.h`：从ops-nn正式RMSNorm MX quant提取BF16/E2M1子集，OCP scale emax2，round量化和nibble打包；保留独立旧FP8实现。
- shared splitN新增FP4分支，内部Q4始终在workspace，外部queryNorm为量化前BF16副本；Q上投影结果BF16，经原splitN事件协议提取compact Q-nope送BF16 Wuk matmul，RoPE仍用既有实现。
- KV0保持BF16；KV3的RMSNorm先保留native BF16舍入边界后进入原FP8 per-tile cache量化/布局。FP4不进入cache。
- V3 kernel按Scenario3+QuantMode0/1实例化；旧key字段不移动。新FP4不启用旧FP8 A full-load优化或splitM模板。

## 已执行的本地检查

`python3 attention/mla_prolog_v3/tests/ut/op_kernel/test_mxfp4_storage_contract.py`：4个test methods通过。

覆盖NZ跨K/N分片与独立format转换的byte对比、scale pair轴顺序、GLM6144/2048/192/512/64逻辑/byte偏移、L1/L0/workspace上界及BF16 queryNorm独立大小。`python3 -m py_compile`通过。限定kernel变更路径的`git diff --check`通过。

这些是CPU存储模型验证，不运行CANN Ascend C源码，不证明NPU数值、API编译或设备同步正确。未执行CANN编译、NPU kernel测试、精度测试或性能测试。内网必须执行TEST_KERNEL.md中的设备验收；当前交付只能标记实验实现待A5验证。

## 仍需审查/设备验证

FP4 typed tensor GetPhyAddr与Ascend C版本的兼容、真实A4W4 NZ producer、LoadData转置及scale配对、FP4 CAST_ROUND的tie/非有限值与native逐byte一致、跨AIC/AIV事件时序、KV0/KV3精度与性能。主流程另负责API/IR/host contract和测试runner；不能单凭此kernel变更宣称VA已接入。
