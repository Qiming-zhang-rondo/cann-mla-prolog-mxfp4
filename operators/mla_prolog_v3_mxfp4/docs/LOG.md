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

2026-09-18, Prolog build stops at common-header staging:

- The user log fails copying `third_party/ops-tensor/include/tensor_api/impl/tensor_api`; the later `-Wfloat-equal` messages are warnings. No new Prolog numerical result is produced by this build.
- Validate the Tensor API root at configuration time and fall back to the selected CANN toolkit's existing `asc` headers when the source checkout lacks them. Require all four header subtrees from one root. Both `add_ops_src_copy` and the package install retain their required copies; no empty directories or optional-copy bypass is introduced. The source resolver already used the CANN location for a sibling ops-tensor checkout, but not for the third_party checkout in the failure log.
- Add launcher `--incremental` to retain this failed build's objects and let CMake regenerate and continue. Keep default clean builds and `--reuse-op` semantics. A failed build still stops before installing any stale package.
- Validate resolution and actual copy/install using local CMake fixtures and launcher orchestration tests. Full Ascend C compilation on this container remains to be rerun; kernel source and numerical thresholds are unchanged.

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

2026-09-15, A5 feedback at `47c211cc`:

- `native_dq_vs_decoded` passed with matched ratio 1.0, max absolute error 0 and RMSE 0. The torch C++ binding compiled and both V3 ACLNN symbols resolved to the selected private installation. The first fused call then failed during workspace/tiling: weightDq had rank 2, expected rank 4. No Prolog kernel accuracy or performance result was produced.
- `MakePackedMxfp4NzDescriptor` previously supplied view `[K,N]` and storage `[N/64,K/16,16,64]`. The mode6 host checks `GetStorageShape().GetDimNum()` in `IsSingleParamValid` and requires rank 4 for all three packed weights; the actual tiling invocation saw rank 2. The generated Inner API is absent from the source tree, so the exact internal shape mapping has not been inspected.
- Match the direct ACLNN construction in upstream snapshot `632dddba` / local import `a41c4e6c`, `attention/mla_prolog_v3/examples/arch35/test_aclnn_mla_prolog_v3_fqkvq.cpp::CreateAclTensorNZ`: both view and storage use the four-dimensional NZ element shape and strides are null. Apply this to DQ, UQ and DKV through their common wrapper helper. The FP4 dtype, NZ byte buffer, scales and mode6 host validation remain unchanged. Graph infershape uses a separate two-dimensional original-shape contract; this patch does not alter it.
- A compiled regression invokes the actual descriptor helper with an instrumented `aclCreateTensor`, covering all three GLM projections, UQ head counts and Hcq=1536. It checks both shapes, dtype, format, strides, offset, pointer identity and FP4 storage byte extent. The original code fails its view-shape assertion; the fix passes. This test does not execute CANN tiling.
- Only the torch C++ wrapper, its regression and documentation change. `git pull --ff-only && bash test_mla_mxfp4.sh --reuse-op` reuses the installed CANN operator and rebuilds the wheel/JIT binding. 19 local reference/wrapper/launcher tests and 4 CPU storage tests pass; the corrected direct call still requires A5 verification.

2026-09-15, A5 feedback at `b07f6a11`:

- The first fused call completed; `query` matched native exactly (max absolute error and RMSE 0). `query_rope_halfsplit` failed with matched ratio 0.53515625, max absolute error 149.75 and RMSE 22.54436. This is a numerical comparison failure, not tiling rejection or kernel timeout. Later cache checks and timing were not reached.
- The test producer incorrectly passed GM sine as `[-sin,+sin]`. Existing upstream `service_gather_sin_cos_arch35.h::GatherSinCos` already negates the lower half before `vf_rope.h::RopeVFImpl`. The two sign changes made fused lower-half rotation disagree with the native reference; the upper-half formula was unchanged. The original CPU reference modeled only the inner VF, omitting the public API's preprocessing.
- Correct the shared test input to `[sin,sin]` and make both native/NumPy references implement the lower-half subtraction explicitly. This fixes the QR and KR input contract together; weights, scales, layouts, kernel and numerical tolerances are unchanged. The public rotation also matches upstream `prologv2_no_quant_pa_bsnd.py::rotate_half`, which returns `[-x2,x1]` and consumes ordinary sine input.
- Added regressions executing the actual runner's factor producer and native `rope` function on CPU, for T=1/17/128 factor inputs and QR/KR shapes. Both tests fail on the previous script and pass after the correction. 21 local reference/wrapper/launcher tests and 4 CPU storage tests pass. Full A5 precision/performance remains pending.
- Only Python test/reference files and documentation change. Update with `git pull --ff-only && bash test_mla_mxfp4.sh --reuse-op`; the installed CANN operator and unchanged C++ binding can be reused.
