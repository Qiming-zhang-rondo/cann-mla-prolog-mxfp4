# MXFP4 Prolog 验证

状态：实验分支验证工程；本机只能运行CPU。A5构建、执行和性能均未完成。

## 入口

在已拉取本分支的A5 Linux容器运行：

```bash
bash operators/mla_prolog_v3_mxfp4/scripts/run_a5.sh 2>&1 | tee mla_mxfp4_a5.log
```

自动build自定义CANN包→安装到checkout内独立路径→优先加载该包→build/install本仓torch extension→运行mode6。旧mode仍调用V4；mode6显式调用V3。日志显示commit、PyTorch/torch_npu版本、真实API所在共享库。旧CANN不接受mode6时失败，不回退。

自定义wheel及runner统一使用独立包`cann_ops_transformer_mla_mxfp4`，不会导入系统旧`cann_ops_transformer`。启动前检查setuptools/wheel依赖；没有可选的Python build模块时复用容器setuptools构建wheel。

同一容器、同一checkout构建失败后，可以保留已有构建目录重试：

```bash
git pull --ff-only
MAX_JOBS=8 bash test_mla_mxfp4.sh --incremental
```

`--incremental`重新配置CMake并继续CANN构建；与跳过CANN构建的`--reuse-op`不能同时使用。默认不加参数仍是完整构建。失败的构建不会继续安装旧包或运行测试。

若`ops-tensor/include/tensor_api`不完整，CMake会从当前CANN安装的`<arch>-linux/asc`或`asc`目录复用头文件，并打印`Tensor API headers: ...`。四组`impl/include`下的`tensor_api/c_api`头文件必须来自同一个完整根目录；复制和打包共用该路径。如果都不完整，在配置阶段报告缺失项，不创建空目录跳过。这个头文件修复不下载依赖；上游其他源码依赖的获取流程保持原样，整个构建不保证离线。

## CPU检查

```bash
python3 -m unittest discover -s operators/mla_prolog_v3_mxfp4/tests -p 'test_*.py' -v
```

覆盖16种FP4编码/负零、跨K/N tile的NZ地址、非均匀E8M0 pair重排、K32反量化、RMSNorm退化、GLM interleave RoPE与Prolog half-split输出的等价排列。不在CPU模拟CAST_ROUND tie；A5 native DynamicMxQuant负责量化，CPU只解码其真实输出。

## A5用例与判定

默认GLM真实shape：He6144/Hcq2048/Hckv512/D192/Dr64，N=4和8（TP后），T=1和17；可通过`--tokens 1 17 128`增加边界。模式KV0/KV3与queryNorm两flag全部执行，不静默skip。未覆盖完整模型、prefill、大M或其他shape。

三组native量化权重与Prolog共享同一FP4值/scale；通过独立decoded contraction检查native producer，再比较query、RoPE、未再量化BF16 queryNorm、KV、KR和KV3四个FP32 scale。cache初始化非零sentinel，检查未写行及-1 slot保持原字节。输入sin为[-sin,+sin]，cos为[cos,cos]，权重RoPE列为原始interleave；输出为half-split，与GLM interleave后的维度置换对应，不能只用零角度测试。

新mode的KV3同样先保留BF16 RMSNorm结果再进行FP32 per-tile cache量化，以对齐VA native；这有别于旧Prolog MXFP8直接FP32 norm→C8的边界，测试按新mode实际代码执行。

候选数值门槛：BF16 matched>=99%，atol=rtol=2^-6，maxabs<=1；FP8采用对应FP8容差，cache scale采用小atol。阈值是初步实现门槛，不是模型精度承诺。

KV3验收比较反量化后的latent值与FP32 scale，另报告原始FP8 code mismatch比例，避免把raw FP8高幅值的一ULP误差错误套入BF16绝对阈值。所有输出另外检查dtype；query mode0的两个scale输出必须为空FLOAT。

性能使用NPU Events、5次warmup、20次同步采样，报告P50/P90/mean。分别报告预量化输入的fused、含输入quant的fused、native计算分项、native含quant及连续slot cache copy的全流程。native以一次fused QKV投影、BF16 bmm和npu_rms_norm执行；全流程比值仅用于本用例，不代表任意分页布局或模型吞吐。离线weight NZ转换不计入每次调用。输出JSON包含所有精度指标；失败另存failure JSON及完整异常日志。

## 尚待执行/补齐

- 无本地CANN：C++ wrapper、host/kernel编译及A5运行均未验收。
- 专门internal-Q4字节dump、异常值/舍入ties、graph capture/replay、GLM端到端还需后续A5验证。
- 通用spec门禁FAIL/SKIP仍保留在SPEC_VALIDATION报告，CPU测试通过不覆盖该执行器全部缺口。

## 构建脚本源码核对

- `build.sh::build_torch_extension_whl`（849行起）接受`--ops`与`--vendor_name`，设置`TORCH_EXTENSION_VENDOR=mla_mxfp4`；`torch_extension/setup.py`43–48行生成`cann_ops_transformer_mla_mxfp4`。`build.sh`2383–2386行把`torch_extension/dist/*.whl`复制到`build_out`。
- `CMakeLists.txt`56–60行支持`--soc=ascend950`；`build.sh`1948行解析vendor；`cmake/custom_build.cmake`1317行生成`cann-ops-transformer-mla_mxfp4_linux-<arch>.run`，`build.sh`799–804行复制到`build_out`。脚本按vendor过滤产物，避免误装目录中的其他包。
- 自定义包安装脚本是`cmake/scripts/custom/install.sh`，接受`--quiet --install-path=绝对路径`，无需内置包的`--full`。`cmake/custom_build.cmake`1279行替换vendor为`mla_mxfp4_transformer`；99–103行将`libcust_opapi.so`安装到`vendors/mla_mxfp4_transformer/op_api/lib`。脚本从库位置上溯三层找到vendor OPP根，再前置`ASCEND_CUSTOM_OPP_PATH`与`LD_LIBRARY_PATH`。
- 已注入一次模拟精度异常执行runner实际末尾failure-handler：退出码1，failure JSON包含`status=FAIL`、异常及已收集accuracy；这仅验证错误处理，不是A5执行。
