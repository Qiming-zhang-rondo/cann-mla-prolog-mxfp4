# Independent kernel / host contract review

2026-09-14. Read-only review by the host implementation agent; no device execution.

Reviewed `arch35/kernel_mla_prolog_split_n_arch35.h` against the new host contract, especially OutputInit, ScaleInit, WorkspaceInit, AicProcess, RmsNormCqProcess, CopyDequantScaleCq, MatmulQcQr, CastQcQrSplitN, DequantAndRopeSplitNSyncMMQcQr and the KV3 norm/rope/scatter functions.

- Workspace offsets agree with CalcWorkSpace: internal scale row Align(Hcq/32,32), down-KV BF16, down-Q BF16, packed Q4, QcQr BF16, extracted Qc BF16. QueryNorm is a separate BF16 output, independent of internal Q4.
- FP4 GM offsets at callers remain logical-element offsets. The helper converts the already-offset tensor address to uint8 GM only for byte DMA. The Q4 store copies Hcq/2 bytes; no new FP4 path multiplies logical offsets by sizeof(FP4) for that store. SDK typed FP4 offset semantics still require the actual CANN compile/device tests; this source review does not validate an SDK ABI.
- Hcq1536 uses a padded 64-byte scale row containing 48 live E8M0 values; Hcq2048 has 64 live values. The MM3 scale stride follows the padded row, while K tiles consume only live groups.
- New tiling aligns down-projection N partitions to 64 and MM3 partitions to whole (D+64) heads. MM3 baseN128 with GLM D192 waits for both pieces before consuming Qc and Qr. Existing split-N flags cover the extracted BF16 Qc buffer, including the BF16-to-BF16 copy branch.
- KV3 first rounds norm results to BF16, then converts to FLOAT for existing FP8 tile quantization; the rope portion remains BF16 in the 656-byte mixed cache container. The absent clip alpha is not read in the E8M0 path.

No additional blocking producer/consumer mismatch was identified in these reviewed functions. Root review covers the new matrix-load and FP4 quantization helpers separately. The key remaining gates are CANN compilation, A5 synchronization/addressing validation, and numerical comparison to the documented native BF16 boundaries.
