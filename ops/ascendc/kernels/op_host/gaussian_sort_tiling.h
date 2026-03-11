/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file gaussian_sort_tiling.h
 * \brief gaussian sort op host tiling
 */

#ifndef GAUSSIAN_SORT_TILING_H
#define GAUSSIAN_SORT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GaussianSortTilingData)
TILING_DATA_FIELD_DEF(uint32_t, nGauss);
TILING_DATA_FIELD_DEF(uint32_t, ubSize);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);       // mask阶段整核个数
TILING_DATA_FIELD_DEF(uint32_t, formerTileNum);   // mask阶段整核处理Tile个数
TILING_DATA_FIELD_DEF(uint32_t, tailTileNum);     // mask阶段尾核处理Tile个数
TILING_DATA_FIELD_DEF(uint32_t, maskLoopNum);     // mask阶段循环次数
TILING_DATA_FIELD_DEF(uint32_t, maskNumPerLoop);  // mask整块处理高斯球个数
TILING_DATA_FIELD_DEF(uint32_t, maskTailNum);     // mask尾块处理高斯球个数
TILING_DATA_FIELD_DEF(uint32_t, maskAlignedNum);  // mask尾块对齐时补齐高斯球数
TILING_DATA_FIELD_DEF(uint32_t, maxSortNum);      // UB单次最大支持排序高斯球数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GaussianSort, GaussianSortTilingData)
}  // namespace optiling

#endif