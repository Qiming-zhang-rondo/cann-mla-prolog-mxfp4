# Kernel contract review

2026-09-14。已读取需求、spec.yaml、DESIGN_PREP_HOST.md、DESIGN_PREP_KERNEL.md，并交叉核实 ops-transformer@632dddba 与 ops-nn@2a77283 的正式源码。

无阻止离线实施的实质API缺口：E2M1 MX LoadData/Mmad、非转置NZ权重、BF16内部OCP量化均有正式源码。编译与NPU正确性尚未验证，不能视作已通过。

实施锁定并已传给host：mode6，Scenario3+QuantMode0/1；三个FP4 GEMM float L0C累加、Fixpipe BF16输出；queryNorm为独立量化前BF16；K/N及N分片64倍数；GLM6144/2048/192/512/64覆盖；T初期1..128；splitM0/cv2/enableDequantOpt=true。现spec中的T大范围和“未锁定”描述需主流程同步，不作为开放任意shape的依据。

专用FP4 matmul helper以typed FP4 GM入口接收逻辑偏移，入口GetPhyAddr转uint8，内部DMA偏移全byte。每个Ktile≤256、Ntile≤128、M≤128，A/B数据及E8M0 scale独立区域，不复用旧FP8的整K scale缓存与sizeof公式。L0 data再转FP4 semantic view，Mmad.k为逻辑K，严禁转FP8/BF16计算冒充FP4。

Workspace顺序（S=stepBatchSize）：S*Align(Hcq/32,32) + S*(Hckv+Dr)*2 + S*Hcq*2 + S*Hcq/2 + S*N*(D+Dr)*2 + S*N*D*2 bytes。queryNorm在外部单独分配；flag不会改变internalQ4 workspace布局。

风险已显式保留：FP4 typed GetPhyAddr下标、硬件Cast tie/异常数、设备头文件版本、实际NZ producer；必须A5验证。该review仅表示源码contract已可实施，不是编译通过、性能通过或部署可用声明。
