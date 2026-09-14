# Host implementation and validation

2026-09-14, based on ops-transformer `632dddba712a4e6cace3f8b44f198aef8a82ce3e`.

Implemented V3 A5-only weight mode 6, semantic quant combinations 16/17, and scenario 3 template selection without changing existing key field positions. The new keys are 3933233 (KV0) and 3933297 (KV3). The original QUANT_MODE field remains four bits; all previous selectors retain their original content.

The API, A5 IR dtype columns, inference, parameter checks and tiling agree on FP4 token and three FP4 A4W4 NZ weights, four E8M0 scale tensors, BF16 weight_uk and query_norm, and an empty FLOAT query_norm scale. The implementation explicitly rejects the unsupported layouts, attributes, devices and nonempty unused optional inputs listed in spec.yaml. The GLM shape He6144/Hcq2048/D192 is included.

The workspace calculation matches six kernel buffers: padded internal E8M0 Q scale, BF16 down-KV, BF16 down-Q, packed internal Q4, BF16 QcQr, and extracted BF16 Qc. An external query_norm output does not remove the internal Q4 buffer. T is restricted to [1,128].

Added host tests in `attention/mla_prolog_v3/tests/ut/op_host/`: `test_mla_prolog_v3_mxfp4_tiling.cpp` (four test groups) and `test_mla_prolog_v3_mxfp4_infershape.cpp` (two groups), with a common fixture. These filenames match the repository's CMake host test globs. Tests cover KV0/KV3, query_norm flag, T boundaries, GLM and Hcq1536/D128 shapes, descriptor rejection, scope rejection, workspace and output inference.

Executed source checks passed: old key selector contents and field width, original A5 dtype columns before the two appended columns, A2/A3 IR configuration, V4 API and tiling data structure unchanged; `git diff --check` passed. The existing CRLF `mla_prolog_tiling_check.h` was normalized to LF to avoid trailing-whitespace diagnostics on additions.

Not executed: C++ host UT compilation/run, ACLNN build, Ascend C compilation, NPU numerical tests or performance. This host does not have CANN or an A5. Source checks are not substitutes for those gates.
