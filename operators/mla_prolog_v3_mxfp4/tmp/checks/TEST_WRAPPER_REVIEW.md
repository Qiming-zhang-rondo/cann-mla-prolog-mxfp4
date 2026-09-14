# 独立 wrapper / A5 runner review

Reviewed 2026-09-14 against the working tree based on ops-transformer
`632dddba712a4e6cace3f8b44f198aef8a82ce3e`. This is source review, not a CANN
compile or an A5 accuracy/performance result.

## Verified contracts

- `attention/mla_prolog_v3/torch_extension/csrc/mla_prolog.cpp::mla_prolog`
  dispatches `weight_quant_mode=6` to **aclnnMlaPrologV3WeightNz**, without the
  V4-only `do_rope` argument. Other modes retain the existing V4 call. The V3
  workspace API symbol is required and its shared-library path is logged;
  there is no native/FP8 fallback. An installed old V3 API rejecting mode 6 is
  still an A5 environment failure, not proof that the custom implementation ran.
- `MakePackedMxfp4NzDescriptor` converts `[N/64,K/16,16,32]` physical **bytes**
  into ACL FLOAT4_E2M1 logical `[K,N]`, FRACTAL_NZ storage
  `[N/64,K/16,16,64]`, without copying or expanding nibbles twice. It uses the
  existing ACLNN_CMD descriptor release path. The ND token wrapper uses the
  generic B4 conversion once to expose `[T,He]` from `[T,He/2]` bytes.
- `CheckMxfp4PackedWeight` rejects noncontiguous/nonzero-offset/non-base-format
  weight containers. The host's `IsTokenXValid`, `IsWeightDqValid`,
  `FillMxfp4FullQuantParamInfo` and `CheckMxfp4Contract` retain the complete
  dtype, shape and mode validation beyond the wrapper's early checks.
- `tests/run_a5.py::weight` uses one actual A5 DynamicMxQuant producer for
  each weight. The native transpose and Prolog NZ repack use the same FP4 codes
  and E8M0 scales. The pair permutation is `[N,K/32] -> [K/64,N,2]`, not a
  plain transpose/view. `native_dq_vs_decoded` independently checks the first
  eight DQ output channels against CPU decoding of the actual quantized data.
- `native` uses BF16 down-projection outputs, BF16 RMSNorm output before the
  internal query FP4 quantization, BF16 UQ output and BF16 bmm. This matches
  the new kernel's FP32 Cube accumulation followed by BF16 intermediate
  stores. Public query_norm is the retained BF16 norm, not internal FP4.
- KV3 now uses BF16 RMSNorm -> FP32 tile amax -> FP8 cache quantization.
  The same boundary is in
  `kernel_mla_prolog_split_n_arch35.h::RmsNormAndQuantizeCkv` (FP4-only
  BF16 round trip). It deliberately differs from legacy MXFP8 Prolog's
  direct FP32 norm -> C8 quantization and matches the VA native norm boundary.
- Nonzero-angle RoPE tests use original interleaved projection columns,
  signed half-split sin, and half-split outputs. KV3 bytes are split into
  512 FP8 K bytes, 128 BF16 RoPE bytes and 16 FP32 scale bytes.
- The performance ratio compares fused including input quant against native
  including input quant and cache copy. Native compute without scatter is
  reported separately. Native down-projection is a single concatenated
  QKV projection. Offline weight quantization/repacking is excluded for both.
  The cache copy baseline is explicitly limited to contiguous test slots;
  this is not a general vLLM throughput benchmark.
- `scripts/run_a5.sh` now builds and imports the same isolated
  `cann_ops_transformer_mla_mxfp4` package. `torch_extension/setup.py`
  rewrites source imports for the vendor package. The script checks wheel
  build dependencies, uses a checkout-local custom OPP install, and prepends
  that custom OPP and op_api library path before loading the extension.

## Findings sent to test author

1. Device output assertions did not cover all five output contracts:
   query/query_rope BF16, empty FLOAT dequant_scale_q_nope and empty FLOAT
   dequant_scale_q_norm must be asserted. Numerical `.float()` comparison
   alone hides an output dtype regression. The current BF16 query_norm
   assertion is correct. Requested fix from `prolog_prepare`.
2. `metric` applies a universal `max_abs <= 1` while `kv_fp8` requests
   `rtol=0.25`. At large E4M3 magnitudes, a one-code FP8 difference is
   16 or 32, so these provisional criteria conflict. Requested a separately
   justified FP8 criterion or dequantized KV comparison, while retaining raw
   code/value diagnostics. Thresholds must remain explicit and provisional;
   changing a threshold is not evidence of correctness.

## Coverage limits (not hidden skips)

- Default inputs cover GLM He6144/Hcq2048/D192/Hckv512/Dr64, T1/17,
  N4/8, KV0/3 and both queryNorm flags, not the full host-advertised shape
  domain. T128 and other TP head counts require explicit additional runs.
- Cache sentinel checks cover untouched rows and an invalid -1 final slot;
  random page order, nonzero destination slots, duplicate-slot semantics
  and graph replay are not covered by this runner.
- Internal query FP4 bytes/scales are not exposed; the runner tests their
  effect through UQ outputs. Dedicated tie/zero/NaN/internal-byte tests and
  model-level checks remain pending, as recorded in TEST.md.
- C++ ABI/API availability, custom package build/load, Cube instructions,
  synchronization and actual A5 numerical/performance behavior all require
  device execution. CPU/storage checks do not close those gaps.

## Related kernel hardening

As suggested by the root review, `MatmulSplitN` now uses
`if constexpr (needCheckAFullLoad && !isMxFp4)` around the legacy full-A
implementation, ensuring it cannot be instantiated for FP4 even if a future
caller enables the full-A template parameter. The host reviewer was notified;
this changes no FP4 mathematics or physical offsets.
