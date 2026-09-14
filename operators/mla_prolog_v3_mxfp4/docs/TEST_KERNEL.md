# Kernel测试目标（实现前）

1. 本地可执行：逐个验证FP4 ND/NZ字节offset与分块重建；E8M0 L1 pair布局、K/N分片独立scale选择；各buffer峰值和workspace范围，queryNorm flag无布局变化。检查GLM6144/2048/192/512/64及N=1/2/4/8/16/32/64，M=1/15/16/17/64/128；旧模板分支源码不被误触发。
2. A5必需：专用FP4 GEMM与native QuantMatmul输出比较；内部Q4 payload/scale与DynamicMxQuant(block32,scaleAlg0,round)逐字节比；BF16 queryNorm来自量化前副本；zero/subnormal/Inf/NaN/舍入边界按设备官方量化基线判定。
3. A5完整：KV0与KV3，queryNorm true/false，cache无效slot及alias、非16倍M、跨K/N tile、图捕获重放；分别比较query/queryRope/queryNorm/cache/scale，并检查GLM Indexer消费者。
4. A5性能：同shape、同权重、同计时方式native与fused，含输入量化和必要layout成本，报告P50/P90；不提前承诺提速。

本机无CANN/NPU；本地测试结果只能标记存储/契约验证，A5项目一律待内网运行。
