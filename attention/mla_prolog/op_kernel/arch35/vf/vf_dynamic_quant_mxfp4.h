/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BF16/E2M1 subset adapted from ops-nn rms_norm_dynamic_mx_quant_common.h
// (2a77283db46e6648ff47bc8277442cf9c721e3c2). Keep OCP special-value behavior.
#ifndef MLA_PROLOG_VF_DYNAMIC_QUANT_MXFP4_H
#define MLA_PROLOG_VF_DYNAMIC_QUANT_MXFP4_H
#include "vf_dynamic_quant.h"
namespace MlaProlog {
namespace MxFp4Quant {
constexpr uint16_t VL_B16 = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
constexpr uint16_t NUM_TWO = 2;
constexpr AscendC::Reg::CastTrait FP4_ROUND = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
__aicore__ inline void ComputeScaleMxfp4Ocp(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr,
                                       __ubuf__ uint16_t* halfScaleLocalAddr, uint32_t totalScaleInUB,
                                       uint16_t loopNumScale)
{
    constexpr uint16_t dtypeEmax = 0x0100; // E2M1 emax = 2

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint16_t> vdMaxExp;
        AscendC::Reg::MaskReg cmpResult;
        AscendC::Reg::RegTensor<uint16_t> expMask;
        AscendC::Reg::MaskReg zeroMask;
        AscendC::Reg::MaskReg preMaskScale;
        AscendC::Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        AscendC::Reg::RegTensor<uint16_t> maxExpValue;
        AscendC::Reg::Duplicate(maxExpValue, dtypeEmax);
        AscendC::Reg::RegTensor<uint16_t> sharedExp;
        AscendC::Reg::RegTensor<uint16_t> scaleValue;
        AscendC::Reg::RegTensor<uint16_t> scaleBias;
        AscendC::Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        AscendC::Reg::RegTensor<uint16_t> halfScale;
        AscendC::Reg::RegTensor<uint16_t> fp8NanRegTensor;
        AscendC::Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        AscendC::Reg::RegTensor<uint16_t> zeroRegTensor;
        AscendC::Reg::Duplicate(zeroRegTensor, 0);
        AscendC::Reg::RegTensor<uint16_t> nanRegTensor;
        AscendC::Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        AscendC::Reg::MaskReg invalidDataMask;
        AscendC::Reg::MaskReg specialDataMask;
        AscendC::Reg::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::Reg::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::Reg::LoadAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(vdMaxExp, maxExpAddr,
                                                                                           VL_B16);
            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            AscendC::Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            AscendC::Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            AscendC::Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

            AscendC::Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);

            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                     AscendC::Reg::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                             VL_B16 / NUM_TWO, preMaskScale);

            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroRegTensor, preMaskScale);
            AscendC::Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            AscendC::Reg::StoreAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, VL_B16, preMaskScale);
        }
    }
    return;
}

__aicore__ inline void ComputeData(__ubuf__ bfloat16_t *src, __ubuf__ uint16_t *reciprocal,
                                    __ubuf__ int8_t *dst, uint32_t count, uint16_t loops)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> x0, x1;
        Reg::RegTensor<uint16_t> scale;
        Reg::RegTensor<fp4x2_e2m1_t> y0, y1;
        for (uint16_t i = 0; i < loops; ++i) {
            auto mask = Reg::UpdateMask<bfloat16_t>(count);
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                           Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, src, VL_B16 * 2);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                           Reg::LoadDist::DIST_E2B_B16>(scale, reciprocal, 8);
            Reg::Mul(x0, x0, (Reg::RegTensor<bfloat16_t> &)scale, mask);
            Reg::Mul(x1, x1, (Reg::RegTensor<bfloat16_t> &)scale, mask);
            Reg::Interleave(x0, x1, x0, x1);
            Reg::Cast<fp4x2_e2m1_t, bfloat16_t, FP4_ROUND>(y0, x0, mask);
            Reg::Cast<fp4x2_e2m1_t, bfloat16_t, FP4_ROUND>(y1, x1, mask);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                dst, (Reg::RegTensor<int8_t> &)y0, VL_B16 / 2, mask);
            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                dst, (Reg::RegTensor<int8_t> &)y1, VL_B16 / 2, mask);
        }
    }
}
} // namespace MxFp4Quant

// col is a multiple of 256 in the supported Hcq={1536,2048} contract.
// tmp: at least 2 * col * sizeof(uint16_t) bytes, output: col/2 bytes.
__aicore__ inline void DynamicQuantPerBlockMxfp4Vf(
    const LocalTensor<FP4E2M1> &output, const LocalTensor<FP8E8M0> &scales,
    const LocalTensor<bfloat16_t> &input, const LocalTensor<uint8_t> &tmp, uint32_t col)
{
    auto maxExp = tmp.ReinterpretCast<uint16_t>();
    auto reciprocal = maxExp[col];
    auto src = (__ubuf__ bfloat16_t *)input.GetPhyAddr();
    auto maxAddr = (__ubuf__ uint16_t *)maxExp.GetPhyAddr();
    auto invAddr = (__ubuf__ uint16_t *)reciprocal.GetPhyAddr();
    auto scaleAddr = (__ubuf__ uint16_t *)scales.GetPhyAddr();
    auto outAddr = (__ubuf__ int8_t *)output.GetPhyAddr();
    const uint16_t loops = col / (2 * MxFp4Quant::VL_B16);
    const uint32_t groups = col / 32;
    const uint16_t scaleLoops = CeilDivT(groups, static_cast<uint32_t>(MxFp4Quant::VL_B16));
    ComputeMaxExpVF<bfloat16_t, FP4E2M1>(src, maxAddr, col, loops, MxFp4Quant::VL_B16);
    MxFp4Quant::ComputeScaleMxfp4Ocp(maxAddr, scaleAddr, invAddr, groups, scaleLoops);
    MxFp4Quant::ComputeData(src, invAddr, outAddr, col, loops);
}
} // namespace MlaProlog
#endif
