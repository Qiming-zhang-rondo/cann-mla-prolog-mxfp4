# 本地测试记录

2026-09-14，macOS arm64，`work/test-venv/bin/python`。

- `python -m unittest discover -s operators/mla_prolog_v3_mxfp4/tests -p 'test_*.py' -v`：8/8通过。
- `py_compile`：reference、A5 runner、torch wrapper Python语法通过。
- `bash -n operators/mla_prolog_v3_mxfp4/scripts/run_a5.sh`：通过。
- `git diff --check`：通过。

8项分别验证FP4全部编码与负零、拒绝错误layout、每K32 scale应用、非均匀scale重排、RMSNorm零值与符号、跨K/N tile NZ布局、GLM interleave→Prolog半分RoPE、真实Python metadata函数的mode6 queryNorm逻辑shape/BF16/空scale。

本机没有CANN/NPU，**C++ wrapper/host/kernel没有编译通过记录，A5精度和性能没有执行记录**。CPU不模拟硬件CAST_ROUND ties；设备runner使用真实DynamicMxQuant输入。单算子通过也不等于GLM端到端或性能已验收。
