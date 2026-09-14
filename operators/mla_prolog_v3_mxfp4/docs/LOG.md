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
