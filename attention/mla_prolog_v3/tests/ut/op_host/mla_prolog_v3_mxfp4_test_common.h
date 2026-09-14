/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MLA_PROLOG_V3_MXFP4_TEST_COMMON_H
#define MLA_PROLOG_V3_MXFP4_TEST_COMMON_H

#include "test_mla_prolog_v3_tiling_common.h"

namespace {
gert::TilingContextPara Mxfp4Context(optiling::MlaPrologCompileInfo &compileInfo, int64_t kvMode = 0,
                                     bool queryNorm = true, int64_t tokens = 8, int64_t hcq = 2048,
                                     int64_t headDim = 192)
{
    constexpr int64_t he = 6144;
    constexpr int64_t heads = 8;
    const int64_t nout = heads * (headDim + 64);
    const bool perTile = kvMode == 3;
    const auto cacheDtype = perTile ? ge::DT_FLOAT8_E4M3FN : ge::DT_BF16;
    const int64_t dtile = perTile ? 656 : 512;
    const gert::StorageShape krShape = perTile ? gert::StorageShape{{0}, {0}} :
                                                  gert::StorageShape{{16, 128, 1, 64}, {16, 128, 1, 64}};
    const gert::StorageShape normShape = queryNorm ? gert::StorageShape{{tokens, hcq}, {tokens, hcq}} :
                                                      gert::StorageShape{{0}, {0}};
    return gert::TilingContextPara(
        "MlaPrologV3",
        {
            {{{tokens, he}, {tokens, he}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{he, hcq}, {hcq / 64, he / 16, 16, 64}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ},
            {{{hcq, nout}, {nout / 64, hcq / 16, 16, 64}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ},
            {{{heads, headDim, 512}, {heads, headDim, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{he, 576}, {9, he / 16, 16, 64}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ},
            {{{hcq}, {hcq}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512}, {512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{tokens, 64}, {tokens, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{tokens, 64}, {tokens, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 128, 1, dtile}, {16, 128, 1, dtile}}, cacheDtype, ge::FORMAT_ND},
            {krShape, ge::DT_BF16, ge::FORMAT_ND},
            {{{tokens}, {tokens}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{tokens, he / 32}, {tokens, he / 32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{hcq, he / 32}, {hcq, he / 32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{nout, hcq / 32}, {nout, hcq / 32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{576, he / 32}, {576, he / 32}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{tokens, heads, 512}, {tokens, heads, 512}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{tokens, heads, 64}, {tokens, heads, 64}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{16, 128, 1, dtile}, {16, 128, 1, dtile}}, cacheDtype, ge::FORMAT_ND},
            {krShape, ge::DT_BF16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {normShape, ge::DT_BF16, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rmsnorm_epsilon_cq", Ops::Transformer::AnyValue::CreateFrom<float>(1e-5f)},
            {"rmsnorm_epsilon_ckv", Ops::Transformer::AnyValue::CreateFrom<float>(1e-5f)},
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("PA_BSND")},
            {"query_norm_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(queryNorm)},
            {"weight_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
            {"kv_cache_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvMode)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"ckvkr_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(perTile ? 1 : 0)},
            {"quant_scale_repo_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(perTile ? 1 : 0)},
            {"tile_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"qc_qr_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"kc_scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
            {"do_rope", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileInfo, "Ascend950", MlaPrologV3_tiling_950SocInfo, 4096);
}
} // namespace


#endif
