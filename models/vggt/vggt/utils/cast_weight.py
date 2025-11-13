# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# This file is a part of the CANN Open Software
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import torch
import torch_npu


def cast_model_weight(model):
    def __format_cast(module, class_name):
        if issubclass(class_name, torch.nn.Conv2d):
            if module.groups > 1:
                return _
            if hasattr(module, "weight") and module.weight is not None and \
                "weight" in dict(module.named_parameters()):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 4)
        return module

    def cast_weight(module):
        current_class = module.__class__
        module = __format_cast(module, current_class)
        if not module.children:
            return _
        for sub_module in module.children():
            if isinstance(sub_module, torch.nn.Module):
                sub_module = cast_weight(sub_module)
        return module
    for _, module in model.named_modules():
        module = cast_weight(module)
    return model
