# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import time
import logging

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.cast_weight import cast_model_weight
from eval.general_utils import fix_random_seed
from quant.vggt_utils import replace_linear_in_vggt, set_ignore_quantize
from quant.vggt_linear import LinearW8A8

logging.basicConfig(level=logging.INFO)

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
        logging.info(f"VGGT inference time cost is: {timestamp*1000:.2f} ms")
        return timestamp
    return timestamp


def quick_start(args):
    fix_random_seed(42)
    # Device check
    device = "npu:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    dtype = torch.bfloat16

    checkpoint_path = args.ckpt
    if args.enableW8A8:
        model = torch.load(checkpoint_path, map_location=device)
        model.to(device).eval()
    else:
        model = VGGT()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model = model.to(dtype)
        model.to(device).eval()
        model = cast_model_weight(model)
        if args.buildW8A8:
            # build model_W8A8
            set_ignore_quantize(model, ignore_quantize=True)
            replace_linear_in_vggt(model, device=device)
            save_path = os.path.join(os.getcwd(), "VGGT_model_W8A8.pt")
            torch.save(model, save_path)
            return

    image_paths = args.images_path
    image_names = get_all_files_paths(image_paths)
    image_names = sorted(image_names)
    images = load_and_preprocess_images(image_names).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
            exec_time_list = []
            for steps in range(6):
                start_time = sync_and_get_time()
                predictions = model(images)
                exec_time = sync_and_get_time(start_time)
                exec_time_list.append(exec_time)
            logging.info(f"The execution time (ms) of inferences: {exec_time_list} and \
                    the average time is {sum(exec_time_list) / len(exec_time_list)} s.")


def parse_args():
    parser = argparse.ArgumentParser("VGGT quick start.", add_help=False)
    parser.add_argument("--ckpt", help="checkpoint location")
    parser.add_argument("--buildW8A8", action="store_true", help="build W8A8 model")
    parser.add_argument("--enableW8A8", action="store_true", help="apply W8A8 model")
    parser.add_argument("--images_path", default="examples/kitchen/images", help="dataset location")
    return parser.parse_args()


def main():
    args = parse_args()
    quick_start(args)


if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=False)
    main()

