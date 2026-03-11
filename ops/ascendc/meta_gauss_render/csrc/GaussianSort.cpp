/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file GaussianSort.cpp
 * \brief gaussian sort pybind adapter
 */

#include <string>

#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

std::tuple<at::Tensor, at::Tensor> gaussian_sort(const at::Tensor& all_in_mask, const at::Tensor& depths)
{
    TORCH_CHECK(all_in_mask.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(depths.device() == all_in_mask.device(), "Inconsistent device.");
    TORCH_CHECK(depths.sizes() == all_in_mask.sizes()[1], "Invalid shape.");
    TORCH_CHECK(all_in_mask.scalar_type() == at::kFloat,
                "all_in_mask: float32 tensor expected but got a tensor with dtype: ", all_in_mask.scalar_type());
    TORCH_CHECK(depths.scalar_type() == at::kFloat,
                "depths: float32 tensor expected but got a tensor with dtype: ", depths.scalar_type());

    auto device = all_in_mask.device();
    uint32_t tileNum = all_in_mask.sizes()[0];    // rows
    uint32_t nGaussNum = all_in_mask.sizes()[1];  // cols
    auto intOptions = at::TensorOptions().dtype(at::kInt).layout(at::kStrided).device(device);
    auto tile_sums = at::sum(all_in_mask, 1, false, at::kInt);
    TORCH_CHECK(at::all(tile_sums >= 0).item<bool>(), "tile_sums has negative! Check tile_sums.");
    auto tile_offsets = at::cumsum(tile_sums, 0, at::kInt);
    int32_t maskGauss = tile_offsets.index({-1}).item<int32_t>();
    at::Tensor sorted_gs_ids = at::empty({maskGauss}, intOptions);
    EXEC_NPU_CMD(aclnnGaussianSort, all_in_mask, tile_sums, tile_offsets, depths, sorted_gs_ids);

    return std::tie(sorted_gs_ids, tile_offsets);
}
