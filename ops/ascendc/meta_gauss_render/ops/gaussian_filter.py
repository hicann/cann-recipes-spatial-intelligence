# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import meta_gauss_render._C


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def gaussian_filter(means,
            colors,
            det,
            opacities,
            means2d,
            depths,
            radius,
            conics,
            covars2d,
            compensations,
            width,
            height,
            near_plane,
            far_plane):
    means_culling, colors_culling, means2d_culling, depths_culling, radius_culling,\
        covars2d_culling, conics_culling, opacities_culling, proj_filter, cnt = meta_gauss_render._C.gaussian_filter(
        means,
        colors,
        det,
        opacities,
        means2d,
        depths,
        radius,
        conics,
        covars2d,
        compensations,
        width,
        height,
        near_plane,
        far_plane)
    return means_culling, colors_culling, means2d_culling, depths_culling, radius_culling,\
           covars2d_culling, conics_culling, opacities_culling, proj_filter, cnt
