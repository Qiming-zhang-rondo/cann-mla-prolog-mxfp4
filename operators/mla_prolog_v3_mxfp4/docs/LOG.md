# MLA Prolog V3 MXFP4 technical development record

2026-09-14. Scope: A5 MLA W4A4 E2M1/K32 E8M0 with BF16 queryNorm and existing KV0/KV3 caches.

- Upstream CANN ops-transformer base: `632dddba712a4e6cace3f8b44f198aef8a82ce3e`.
- Workflow reference: cannbot-skills `3074681ba927916f99f5a9ac808b4e6797934ce5`.
- Requirements, host/kernel designs, independent source reviews, implementation and dedicated tests are included in this directory.
- The generic spec validator has real FAIL/SKIP records; these have not been relabeled as successful gates.
- 8 CPU reference/metadata tests and 4 CPU storage tests passed. Python/shell syntax and diff whitespace checks passed.
- CANN/Ascend C/C++ compilation and A5 accuracy/performance have NOT been executed; local development had no CANN or NPU.
- GLM routing in vLLM-Ascend is not changed by this operator repository.

See [current implementation](IMPLEMENTATION.md), [test guide](TEST.md), and [source snapshot provenance](../../../SOURCE_SNAPSHOT.md).

2026-09-15. First A5 compile feedback and targeted review:

- The supplied log fails to compile `MatmulMxFp4SplitK`: `MMParams::kScale` is `uint32_t`, but `BYTE_BLOCK` is `uint64_t`; `Align(T, T)` cannot deduce one type. Cast the constant to `uint32_t` at this call. The original expression reproduces the compiler error locally; the corrected expression compiles and preserves padded/unpadded stride arithmetic, including 48-to-64 scale padding.
- Reviewed the added alignment calls, FP4 NZ byte offsets, K32 scale pairs, L0/L1 sizes, BF16 queryNorm workspace and official MX quant/LoadData references. No additional confirmed kernel blocker was found; hardware compilation, numerical accuracy and synchronization remain unverified.
- Disable FLA `.pth` injection before every launcher Python child and retain the selected private vendor across explicit backend imports. Log both V3 workspace/compute symbol owners and reject mixed libraries. API ownership does not establish device-kernel provenance.
- The wheel-only build now uses `--incremental` to retain the preceding operator build. The initial operator build still performs the upstream clean rebuild; installed Python dependencies are reused. CMake may fetch missing source dependencies, so this is not a strict offline-build guarantee.
- 11 reference/metadata/scalar-compile/launcher tests and 4 storage tests passed locally. Full Ascend C compilation and A5 accuracy/performance must be rerun with this fix.

2026-09-15, after A5 build and wheel installation completed:

- The test stopped in native `npu_quant_matmul` before calling the custom Prolog. `weight()` materialized scale as contiguous `[K/64,N,2]` while packed weight was a transposed `[K/2,N]` view. QuantMatmul requires matching transpose states, determined by strides. Keep scales as `[N,K/64,2].transpose(0,1)` with stride `(2,K/32,1)`, matching VA `w4a4_mxfp4.py::process_weights_after_loading`; concatenate original N rows before forming the fused-down views. Values and quantization are unchanged.
- Fix the wheel directory/entrypoint name `mla_prolog_v3` versus exported callable `mla_prolog`. Export both aliases, preserve them during entrypoint discovery, and keep a vendor package's entrypoints confined to that vendor. The runner explicitly uses `ops.mla_prolog_v3` and checks the export before preparing inputs. Real wheel build/unpack/import tests cover this path using fake NPU/JIT backends.
- Add `--reuse-op` for this Python-only correction: reuse the latest complete private installation, rebuild the Python wheel, and install it with `--no-index --no-deps`. No CANN kernel/host/API or C++ wrapper source changed in this correction. The first successful custom call can still JIT compile its torch binding.
- 18 reference/metadata/compiler/launcher/native-stride/wheel-import tests and 4 storage tests pass locally. This does not establish A5 Prolog accuracy or performance; those checks have not yet run successfully.
