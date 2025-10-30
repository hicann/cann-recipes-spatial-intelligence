# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# This file is a part of the CANN Open Software
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import argparse
import time
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.cast_weight import cast_model_weight
from eval.general_utils import fix_random_seed

def get_all_files_paths(dir_path):
    file_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def sync_and_get_time(start_time=None, use_syn=True):
    if use_syn:
        torch.npu.synchronize()
    timestamp = time.time()
    if start_time is not None:
        timestamp -= start_time
        print(f"VGGT inference time cost is: {timestamp*1000:.2f} ms" )
        return timestamp
    return timestamp

def quick_start(pt_path, image_paths):
    fix_random_seed(42)
    # Device check
    device = "npu:6" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    dtype = torch.bfloat16
    model = VGGT()
    checkpoint_path = pt_path  # Path to the model checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to(dtype)
    
    model.to(device).eval()
    model = cast_model_weight(model)
    image_names = get_all_files_paths(image_paths)
    image_names = sorted(image_names)
    images = load_and_preprocess_images(image_names).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images) #warm up
            exec_time_list = []
            for steps in range(6):
                start_time = sync_and_get_time()
                predictions = model(images)
                exec_time = sync_and_get_time(start_time)
                exec_time_list.append(exec_time)
            print(f"The execution time of inferences:{exec_time_list} and the average time is {sum(exec_time_list)/len(exec_time_list)}" )

def parse_args():
    parser = argparse.ArgumentParser("VGGT quick start.", add_help=False)
    parser.add_argument("--ckpt", help="checkpoint location")
    parser.add_argument("--images_path", default="examples/kitchen/images", help="dataset location")
    return parser.parse_args()

def main():
    args = parse_args()
    quick_start(args.ckpt, args.images_path)

if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=False)
    main()

