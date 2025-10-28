# coding=utf-8
# Adapted from
# https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/minimal_demo.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This code is based on Tencent-Hunyuan's Hunyuan3D-2 library and the Hunyuan3D-2
# implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to Hunyuan3D-2 used by Tencent-Hunyuan team that trained the model.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import argparse
import os

import torch
from PIL import Image
from torch_npu.contrib import transfer_to_npu

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False

def main():
    parser = argparse.ArgumentParser(
        description='Hunyuan3D'
    )
    
    parser.add_argument('--model_path', type=str, default='tencent/Hunyuan3D-2', help="模型路径")
    parser.add_argument('--mutiview', action='store_true', default=False, help="多视角输入")
    parser.add_argument('--face_reduce', action='store_false', default=True, help="减少mesh面片数量")

    args = parser.parse_args()
    if args.mutiview:
        images = {
            "front": "assets/example_mv_images/1/front.png",
            "left": "assets/example_mv_images/1/left.png",
            "back": "assets/example_mv_images/1/back.png"
        }

        for key in images:
            image = Image.open(images[key]).convert("RGBA")
            if image.mode == 'RGB':
                rembg = BackgroundRemover()
                image = rembg(image)
            images[key] = image

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model_path+"mv",
            subfolder='hunyuan3d-dit-v2-mv',
            variant='fp16'
        )
        mesh = pipeline(
            image=images,
            num_inference_steps=50,
            octree_resolution=380,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]

    else:
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)
        image_path = 'assets/demo.png'
        image = Image.open(image_path).convert("RGBA")
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)

        mesh = pipeline_shapegen(image=image)[0]
    
    if args.face_reduce:
        mesh = mesh.simplify_quadric_decimation(face_count=20000) #减少面片数量
    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(args.model_path)
    mesh = pipeline_texgen(mesh, image=image)
    mesh.export('demo.glb')

if __name__ == '__main__':
    main()
