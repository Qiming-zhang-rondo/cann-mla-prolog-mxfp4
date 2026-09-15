/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SERVICE_MATMUL_MXFP4_ARCH35_H
#define SERVICE_MATMUL_MXFP4_ARCH35_H

#include "service_matmul_arch35.h"

namespace MlaProlog {

// All offsets below are bytes except the FP4-typed arguments and Mmad dimensions.
// The MX LoadData contract follows ops-nn BlockMmadMx::CopyInL0A/CopyInL0B.
constexpr uint32_t MXFP4_C0 = 64;
constexpr uint32_t MXFP4_GROUP = 32;

__aicore__ inline void CopyMxFp4ScaleToL1(const LocalTensor<uint8_t> &dst,
                                         const GlobalTensor<FP8E8M0> &src,
                                         uint32_t rows, uint32_t groups, uint32_t rowStride)
{
    GlobalTensor<bfloat16_t> srcPairs;
    srcPairs.SetGlobalBuffer((__gm__ bfloat16_t *)src.GetPhyAddr());
    auto dstPairs = dst.ReinterpretCast<bfloat16_t>();
    Dn2NzParams copy{};
    copy.dnNum = 1;
    copy.nValue = groups / 2;
    copy.dValue = rows;
    copy.srcDValue = rowStride / 2;
    copy.dstNzC0Stride = groups / 2;
    copy.dstNzNStride = 1;
    copy.dstNzMatrixStride = groups / 2;
    DataCopy(dstPairs, srcPairs, copy);
}

// Conservative first implementation: one baseK tile per transfer, no full-A
// preload or atomic partial sums. The entire K reduction stays in float L0C.
__aicore__ inline void MatmulMxFp4SplitK(
    const GlobalTensor<bfloat16_t> &cGm, const GlobalTensor<FP4E2M1> &aGm,
    const GlobalTensor<FP4E2M1> &bGm, const MMParams &p, MMBufParams &buf,
    uint32_t nOffset, uint32_t nSize, const GlobalTensor<FP8E8M0> &aScaleGm,
    const GlobalTensor<FP8E8M0> &bScaleGm, bool paddedAScale = false)
{
    GlobalTensor<uint8_t> aBytes, bBytes;
    aBytes.SetGlobalBuffer((__gm__ uint8_t *)aGm.GetPhyAddr());
    bBytes.SetGlobalBuffer((__gm__ uint8_t *)bGm.GetPhyAddr());
    mmLocalTensors<uint8_t, float> local;
    local.Init(buf);
    const uint32_t mAlign = Align(p.m, BLOCK_CUBE_SIZE);
    const uint32_t scaleStrideA = paddedAScale ? Align(p.kScale, static_cast<uint32_t>(BYTE_BLOCK)) : p.kScale;
    const uint32_t cIdx = buf.cL0BufIter & 1u;
    auto cL0 = local.cL0Tensor[cIdx * L0C_PP_SIZE / sizeof(float)];
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cIdx);

    for (uint32_t kOffset = 0; kOffset < p.k; kOffset += p.baseK) {
        const uint32_t kSize = p.k - kOffset < p.baseK ? p.k - kOffset : p.baseK;
        const uint32_t groups = kSize / MXFP4_GROUP;
        const uint32_t aIdx = buf.aL1BufIter & 1u;
        const uint32_t bIdx = buf.bL1BufIter & 1u;
        auto aL1 = local.aL1Tensor[aIdx * L1_A_SIZE];
        auto bL1 = local.bL1Tensor[bIdx * L1_B_SIZE];
        auto aScaleL1 = bL1[L1_B_SIZE / 2];
        auto bScaleL1 = aScaleL1[mAlign * groups];
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + aIdx);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + bIdx);
        CopyNDGmToL1(aL1, aBytes[kOffset / 2], p.m, kSize / 2, p.k / 2);
        // NZ [N/64,K/16,16,64], with adjacent N nibbles in one byte.
        DataCopyParams bCopy{};
        bCopy.blockCount = nSize / MXFP4_C0;
        bCopy.blockLen = kSize;
        bCopy.srcStride = p.k - kSize;
        DataCopy(bL1, bBytes[(static_cast<uint64_t>(p.k) * nOffset + kOffset * MXFP4_C0) / 2], bCopy);
        CopyMxFp4ScaleToL1(aScaleL1, aScaleGm[kOffset / MXFP4_GROUP], p.m, groups, scaleStrideA);
        CopyMxFp4ScaleToL1(bScaleL1, bScaleGm[nOffset * p.kScale + kOffset / MXFP4_GROUP],
                          nSize, groups, p.kScale);
        SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + aIdx);
        SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + bIdx);
        WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + aIdx);
        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + bIdx);

        const uint32_t laIdx = buf.aL0BufIter & 1u;
        const uint32_t lbIdx = buf.bL0BufIter & 1u;
        auto aL0 = local.aL0Tensor[laIdx * L0A_PP_SIZE].ReinterpretCast<FP4E2M1>();
        auto bL0 = local.bL0Tensor[lbIdx * L0B_PP_SIZE].ReinterpretCast<FP4E2M1>();
        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + laIdx);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + lbIdx);
        LoadData2DParamsV2 aLoad{};
        aLoad.mStep = mAlign / BLOCK_CUBE_SIZE;
        aLoad.kStep = kSize / MXFP4_C0;
        aLoad.srcStride = aLoad.mStep;
        aLoad.dstStride = aLoad.mStep;
        LoadData2DMxParams asLoad{};
        asLoad.xStep = aLoad.mStep;
        asLoad.yStep = kSize / MXFP4_C0;
        asLoad.srcStride = asLoad.yStep;
        asLoad.dstStride = asLoad.yStep;
        LoadData(aL0, aL1.ReinterpretCast<FP4E2M1>(), aScaleL1.ReinterpretCast<FP8E8M0>(), aLoad, asLoad);
        LoadData2DParamsV2 bLoad{};
        bLoad.mStep = kSize / BLOCK_CUBE_SIZE;
        bLoad.kStep = nSize / MXFP4_C0;
        bLoad.srcStride = bLoad.mStep;
        bLoad.dstStride = nSize / BLOCK_CUBE_SIZE;
        bLoad.ifTranspose = true;
        LoadData2DMxParams bsLoad{};
        bsLoad.xStep = nSize / BLOCK_CUBE_SIZE;
        bsLoad.yStep = kSize / MXFP4_C0;
        bsLoad.srcStride = bsLoad.yStep;
        bsLoad.dstStride = bsLoad.yStep;
        LoadData(bL0, bL1.ReinterpretCast<FP4E2M1>(), bScaleL1.ReinterpretCast<FP8E8M0>(), bLoad, bsLoad);
        SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + laIdx);
        SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + lbIdx);
        WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + laIdx);
        WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + lbIdx);
        MmadParams mm(mAlign, nSize, kSize, UNIT_FLAG_DISABLE, false, kOffset == 0);
        Mmad(cL0, aL0, bL0, mm);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + laIdx);
        SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + lbIdx);
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + aIdx);
        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + bIdx);
        buf.aL0BufIter++;
        buf.bL0BufIter++;
        buf.aL1BufIter++;
        buf.bL1BufIter++;
    }
    GetTensorC<uint8_t, bfloat16_t, float>(cGm[nOffset], cL0, p.m, nSize, mAlign, p.orgKc, buf);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cIdx);
    buf.cL0BufIter++;
}
} // namespace MlaProlog
#endif
