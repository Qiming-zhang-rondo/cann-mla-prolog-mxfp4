/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mla_prolog_v3_mxfp4_test_common.h"

TEST_F(MlaPrologV3, Mxfp4GlmCacheAndNormContracts)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    for (int64_t kvMode : {0, 3}) {
        for (bool queryNorm : {false, true}) {
            for (int64_t tokens : {1, 8, 128}) {
                auto para = Mxfp4Context(compileInfo, kvMode, queryNorm, tokens);
                // Legacy fields keep their positions: cache=1, scenario=3, quant=0/1, dequant=true.
                ExecuteTestCase(para, ge::GRAPH_SUCCESS, kvMode == 0 ? 3933233U : 3933297U);
            }
        }
    }
    auto dsShape = Mxfp4Context(compileInfo, 0, true, 8, 1536, 128);
    ExecuteTestCase(dsShape, ge::GRAPH_SUCCESS, 3933233U);
}

TEST_F(MlaPrologV3, Mxfp4RejectsUnimplementedAttributeCombinations)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    for (int64_t kvMode : {1, 2}) {
        ExecuteTestCase(Mxfp4Context(compileInfo, kvMode), ge::GRAPH_FAILED);
    }
    for (int64_t tokens : {0, 129}) {
        ExecuteTestCase(Mxfp4Context(compileInfo, 0, true, tokens), ge::GRAPH_FAILED);
    }
    for (const auto &cacheMode : {"PA_NZ", "TND", "PA_BLK_BSND"}) {
        auto para = Mxfp4Context(compileInfo);
        para.attrs_[2].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>(cacheMode);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
    for (size_t attrIndex : {6U, 7U, 8U}) {
        auto para = Mxfp4Context(compileInfo);
        para.attrs_[attrIndex].attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(1);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
    for (size_t attrIndex : {10U, 11U}) {
        auto para = Mxfp4Context(compileInfo);
        para.attrs_[attrIndex].attr_ = Ops::Transformer::AnyValue::CreateFrom<float>(0.5f);
        ExecuteTestCase(para, ge::GRAPH_FAILED);
    }
    auto noRope = Mxfp4Context(compileInfo);
    noRope.attrs_[12].attr_ = Ops::Transformer::AnyValue::CreateFrom<bool>(false);
    ExecuteTestCase(noRope, ge::GRAPH_FAILED);
    for (const auto &soc : {"Ascend910_B3", "Ascend910_93"}) {
        auto olderSoc = Mxfp4Context(compileInfo);
        olderSoc.socVersion_ = soc;
        olderSoc.socInfoString_ = MlaPrologV3_tiling_A2SocInfo;
        const auto offset = olderSoc.socInfoString_.find("Ascend910_B3");
        olderSoc.socInfoString_.replace(offset, std::string("Ascend910_B3").size(), soc);
        ExecuteTestCase(olderSoc, ge::GRAPH_FAILED);
    }
}

TEST_F(MlaPrologV3, Mxfp4RejectsBrokenProducerAndConsumerDescriptors)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    for (size_t index : {12U, 13U, 14U, 15U}) {
        auto wrongDtype = Mxfp4Context(compileInfo);
        wrongDtype.inputTensorDesc_[index].dtype_ = ge::DT_FLOAT;
        ExecuteTestCase(wrongDtype, ge::GRAPH_FAILED);
        auto absent = Mxfp4Context(compileInfo);
        absent.inputTensorDesc_[index].shape_ = gert::StorageShape{{}, {}};
        ExecuteTestCase(absent, ge::GRAPH_FAILED);
        auto wrongShape = Mxfp4Context(compileInfo);
        wrongShape.inputTensorDesc_[index].shape_ = gert::StorageShape{{1, 1}, {1, 1}};
        ExecuteTestCase(wrongShape, ge::GRAPH_FAILED);
    }
    auto fp8Nz = Mxfp4Context(compileInfo);
    fp8Nz.inputTensorDesc_[1].shape_ = gert::StorageShape{{6144, 2048}, {64, 384, 16, 32}};
    ExecuteTestCase(fp8Nz, ge::GRAPH_FAILED);
    auto byteShape = Mxfp4Context(compileInfo);
    byteShape.inputTensorDesc_[0].shape_ = gert::StorageShape{{8, 3072}, {8, 3072}};
    ExecuteTestCase(byteShape, ge::GRAPH_FAILED);
    auto quantNorm = Mxfp4Context(compileInfo);
    quantNorm.outputTensorDesc_[5].dtype_ = ge::DT_FLOAT4_E2M1;
    ExecuteTestCase(quantNorm, ge::GRAPH_FAILED);
    auto normScale = Mxfp4Context(compileInfo);
    normScale.outputTensorDesc_[6] = {{{8, 64}, {8, 64}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND};
    ExecuteTestCase(normScale, ge::GRAPH_FAILED);
    for (size_t index : {16U, 17U, 18U, 19U, 20U}) {
        auto unsupported = Mxfp4Context(compileInfo);
        unsupported.inputTensorDesc_[index].shape_ = gert::StorageShape{{1}, {1}};
        ExecuteTestCase(unsupported, ge::GRAPH_FAILED);
    }
}

TEST_F(MlaPrologV3, Mxfp4WorkspaceIncludesPackedActivationAndPaddedScale)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    TilingInfo one, two;
    ASSERT_TRUE(ExecuteTiling(Mxfp4Context(compileInfo, 0, false, 1), one));
    ASSERT_TRUE(ExecuteTiling(Mxfp4Context(compileInfo, 0, true, 2), two));
    ASSERT_EQ(one.workspaceSizes.size(), 1U);
    ASSERT_EQ(two.workspaceSizes.size(), 1U);
    constexpr int64_t bytesPerToken = 64 + 576 * 2 + 2048 * 2 + 2048 / 2 + 8 * 256 * 2 + 8 * 192 * 2;
    EXPECT_GE(one.workspaceSizes[0], bytesPerToken);
    EXPECT_EQ(two.workspaceSizes[0] - one.workspaceSizes[0], bytesPerToken);
}
