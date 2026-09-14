# MXFP4 Prolog spec 通用校验结果

**状态：❌通用 11-stage 校验未通过。** 这不是 NPU 算子执行结果；本机无 CANN/NPU。

| Stage | 名称 | 结果 |
|---|---|---|
| 1 | schema_static | FAIL：复合语义原语 MX quant/dequant、RMSNorm、scatter 不在当前 composition 白名单 |
| 2 | category_paradigm_consistency | FAIL：同上；真实 Prolog scale 参数名不在量化白名单；950 芯片表遗漏 E8M0 |
| 3 | shape_closure | PASS |
| 4 | dtype_closure | PASS |
| 5 | broadcast_legality | FAIL：通用检查把异构输入 rank 当作必须整体广播，不适配此融合接口 |
| 6 | boundary_min_set | FAIL：ScatterUpdate要求越界执行用例，但当前host不读device索引，非法正值是caller违约，不能编造host报错；专门测试只执行合法slot与-1哨兵 |
| 7 | tolerance_coverage | PASS，仅覆盖性/启发式检查，不是设备精度通过 |
| 8 | formula_smoke_eval | SKIP：textual_only；需专门参考执行器 |
| 9 | oracle_reachable | SKIP：没有已核实的单 callable 新模式 oracle，不能用 torch.matmul 冒充 |
| 10 | formula_oracle_equiv | SKIP：需要专门 FP4 + cache native reference |
| 11 | invariant_exec | SKIP：需专门参考执行器执行值级不变量 |

命令：`work/test-venv/bin/python work/cannbot-skills/ops/ops-spec-gen/scripts/validate_spec.py operators/mla_prolog_v3_mxfp4/docs/spec.yaml --json`。生成骨架使用同目录 `generate_spec.py`；仅在既有 `work/test-venv` 安装 PyYAML/jsonschema，无系统软件安装。

## 已锁定的源码 contract

- 扩展公开 weight mode=6；旧0–5语义不变。两个新语义组合16/17拟通过原 SCENARIO=3 + QUANT_MODE低字段0/1编码，不能移动旧key字段。
- A4W4 NZ C0=64，非转置 `[K,N]` 权重 storage `[N/64,K/16,16,64]` 以逻辑 FP4 元素计，payload 每元素半字节；A8W4 C0=32不可混用。
- 四组 Prolog E8M0 scale 均为 ND 连续 `[Nout,K/32]`，相邻 K32 pair 供 L1 的 BF16 reinterpret；不使用 QuantMatmul `[K/64,Nout,2]` 的物理排列。
- 内部 Q4 使用 OCP scaleAlg0、K32、emax2、硬件 CAST_ROUND；规范引用 ops-nn `ComputeScaleOCP` / `ComputeDataMxfp4General`，不杜撰 NumPy tie 行为。
- 三次 FP4 GEMM 均 FP32 累加并通过 F322BF16 输出，保持 native BF16 投影→RMSNorm/拆分的舍入边界；query_norm 是内部 FP4 quant 前 BF16 副本，无 scale。
- 首期 PA_BSND：KV0 BF16 `[Pages,Block,1,512]`，kr BF16 `[Pages,Block,1,64]`；KV3 FP8E4M3 容器 `[Pages,Block,1,656]`，kr BF16 `[0]`，tile128、两repoMode=1。
- `actual_seq_len` IR dtype为INT32，首期要求不传或空；非空明确拒绝。cache_index只支持-1或有效slot；未承诺device非法正索引会被host报错。
- 实际 GLM-5.2 验收 shape：He6144、Hcq2048、Hckv512、全局N64、D192、Dr64、eps1e-5、rope_interleave=true；来源 `zai-org/GLM-5.2@f6142f127a14b58dc602592e996cd7d8ff139351/config.json`，由主 Agent 核实。5.3同结构为用户给定假设。

## 后续处理

用独立接口/源码 review 和专用 CPU decoded-reference、A5 native composition 测试弥补通用校验不适配；不能将此草案记录为“11-stage 全 PASS”。模型指标、设备容差与性能仍须 A5 实测。详见 [适配 issue](../../issues/issue_20260914_spec-validator-composite_02.md)。
