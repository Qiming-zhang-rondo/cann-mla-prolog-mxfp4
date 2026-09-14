# spec 校验器不适配 Prolog MXFP4 复合接口

- 状态：开放；源码 contract 可供独立 review，通用门禁没有通过。
- Skill ref：cannbot-skills `3074681ba927916f99f5a9ac808b4e6797934ce5`。
- 实际结果：[SPEC_VALIDATION.md](../tests/reports/SPEC_VALIDATION.md)。

`ops/ops-spec-gen/scripts/validate_spec.py` Quantization 检查固定名称 scale/dequant_scale/x1Scale+x2Scale，不接受真实 Prolog 的 dequant_scale_x/dequant_scale_w_dq 等。没有为过验证新增不存在的参数或重命名真实接口。

`registries/chip_registry.yaml` 的950 dtype列表遗漏 E8M0，但 `ops-transformer@632dddba:attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp` 已明确用 E8M0 注册950 MXFP8 scale；不因工具遗漏删除真实 dtype。

composition白名单只含基础 cast/gemm/reduce/elementwise等；当前草案用明确 MX quant、RMSNorm、scatter 语义名称，尚未展开为符合工具注册表的最小原语图。因此 stage1/2失败包含待适配图表达；并非所有失败都能等同已验证语义。没有把多输入 scatter 或 block quant 假称 elementwise来获取PASS。

stage5 对全部异构输入施加同rank/no-broadcast约束，不能表达 Prolog不同投影之间的独立归约与RMSNorm gamma局部广播。stage8/10/11需执行真正的FP4 block量化、BF16中间舍入和cache写入参考；未用简单matmul公式替代。

后续专用测试必须独立执行 pack/NZ/scale contract、zero输入退化、queryNorm不随W_UQ_QR变化、无效slot不改cache、native小算子拼接精度与A5设备性能，保留所有跳过/未执行事实。
