# Host contract 独立核对

2026-09-14。已读取开发/UT skill（custom, auto, ascend950, ophost），当前无 todowrite 工具，以文件记录目标；没有 CANN/NPU，C++/NPU编译及执行暂不具备。

对 spec.yaml、DESIGN_PREP_HOST.md 和源码基线逐项核对：mode6仅V3 A5；FP4 NZ C0=64；四E8M0 scale按输出通道连续；queryNorm BF16+空FLOAT scale；KV0/KV3和656byte混合容器；GLM Hcq2048/D192；新Scenario3保留旧key位布局，均一致。

两项需spec修订已反馈作者：actual_seq_len现有dtype应为INT32，本期非空明确拒绝；host不读取device cache_index，不能宣称越界值会返回主机错误，需定义合法index/-1哨兵为调用者前置条件。He/Hcq要求64对齐。原generic validator的融合rank/E8M0白名单限制仍原样保留失败记录，不伪造PASS。

结论：上述修订不阻塞独立host接入；T范围、分核N64及workspace大小需与kernel负责人共同锁定，最终合并前必须核对。当前只有接口设计的源码依据，无硬件通过结论。
