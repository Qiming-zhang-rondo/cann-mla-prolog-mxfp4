# Source provenance

This repository contains a source snapshot of CANN ops-transformer, followed by an experimental MLA Prolog V3 MXFP4 change. It does not reproduce the upstream Git history.

- Upstream repository: https://gitcode.com/cann/ops-transformer
- Exact upstream source commit: `632dddba712a4e6cace3f8b44f198aef8a82ce3e`
- Local implementation commit copied into this snapshot: `115cc6182124dd2c98da33b8b09012bf46ff13e0`
- Tag `upstream-632dddba` identifies the unmodified imported upstream source tree in this repository. It is a snapshot commit, not the original upstream commit object.
- `attention/mla_prolog` and `attention/mla_prolog_v3` modified files are byte-identical to the implementation commit. The public development log contains technical records only.
- Upstream licensing and notices are retained. This repository is an experimental derivative for Ascend A5 validation, not an upstream supported release.

Review the operator changes with:

```bash
git diff upstream-632dddba -- attention/mla_prolog attention/mla_prolog_v3
```

See [the A5 test entry](README_MXFP4.md) for validation status and setup. CANN compilation and A5 execution remain unverified.
