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
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

TEST_F(MlaPrologV3, Mxfp4InferShapeKeepsBf16NormAndEmptyScale)
{
    optiling::MlaPrologCompileInfo compileInfo = {48};
    for (int64_t kvMode : {0, 3}) {
        for (bool queryNorm : {false, true}) {
            auto tilingPara = Mxfp4Context(compileInfo, kvMode, queryNorm);
            std::vector<gert::InfershapeContextPara::TensorDescription> inputs, outputs;
            std::vector<gert::InfershapeContextPara::OpAttr> attrs;
            for (const auto &tensor : tilingPara.inputTensorDesc_) {
                inputs.emplace_back(tensor.shape_, tensor.dtype_, tensor.format_);
            }
            for (const auto &tensor : tilingPara.outputTensorDesc_) {
                outputs.emplace_back(tensor.shape_, tensor.dtype_, tensor.format_);
            }
            for (const auto &attr : tilingPara.attrs_) {
                attrs.emplace_back(attr.attrName_, attr.attr_);
            }
            const std::vector<int64_t> kr = kvMode == 3 ? std::vector<int64_t>{0} :
                                                                          std::vector<int64_t>{16, 128, 1, 64};
            const std::vector<int64_t> norm = queryNorm ? std::vector<int64_t>{8, 2048} : std::vector<int64_t>{0};
            ExecuteTestCase(gert::InfershapeContextPara("MlaPrologV3", inputs, outputs, attrs), ge::GRAPH_SUCCESS,
                            {{8, 8, 512}, {8, 8, 64}, {16, 128, 1, kvMode == 3 ? 656 : 512}, kr, {0}, norm, {0}});
        }
    }
}

TEST_F(MlaPrologV3, Mxfp4InferDtypeDoesNotReturnPackedNorm)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("MlaPrologV3");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    optiling::MlaPrologCompileInfo compileInfo = {48};
    for (int64_t kvMode : {0, 3}) {
        auto para = Mxfp4Context(compileInfo, kvMode);
        std::vector<void *> inputDtypes;
        for (auto &tensor : para.inputTensorDesc_) {
            inputDtypes.push_back(&tensor.dtype_);
        }
        std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> attrs;
        for (const auto &attr : para.attrs_) {
            attrs.emplace_back(attr.attrName_, attr.attr_);
        }
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(21, 7)
                          .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(4, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(5, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(6, ge::FORMAT_ND, ge::FORMAT_ND)
                          .InputDataTypes(inputDtypes)
                          .NodeAttrs(attrs)
                          .Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
        EXPECT_EQ(context->GetOutputDataType(5), ge::DT_BF16);
        EXPECT_EQ(context->GetOutputDataType(6), ge::DT_FLOAT);
    }
}
