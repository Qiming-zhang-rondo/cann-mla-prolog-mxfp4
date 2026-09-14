# Host 测试目标

1. A5 GLM形状 He6144/Hcq2048/Hckv512/D192/Dr64、N8：KV0/KV3 × queryNorm true/false，期待正确新key和workspace。
2. 同时保留Hcq1536/D128的合法形状；4D FP4 NZ需要C0=64，拒绝伪FP8 C0=32与packed byte逻辑shape。
3. 缺失/错误dtype/错误shape的任一E8M0 scale拒绝；queryNorm必须BF16、scale为空。
4. mode6在A2/A3、非2D token、非PA_BSND、querymode1、KV1/2、do_rope=false、非1qc/kc scale、非空无关optional均拒绝；新增模式不得静默走旧路径。
5. InferShape/InferDataType验证BF16 queryNorm及空FLOAT scale，两个flag和两cache均覆盖。
6. 全量旧host UT作为旧key和dtype回归；追加静态检查证明原template声明位宽、旧selector保持原样。

实现中的单元测试使用仓库现有Faker/Executor。C++ host UT必须在CANN测试构建环境执行；本机只做代码与源码contract静态检查，不以此替代UT运行。
