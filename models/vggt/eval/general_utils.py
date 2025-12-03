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
import argparse
import random
import numpy as np
import torch


def fix_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_depth_estimation_opts():
    parser = argparse.ArgumentParser(description="Test VGGT depth estimation on DTU datasets.", add_help=False)


    parser.add_argument('--n_views', type=int, default=5, help='num of view')
    parser.add_argument('--levels', type=int, default=4, help='num of stages')
    parser.add_argument('--depth_conf_thres', type=int, default=3, 
        help='the range of depth conf is [1, +inf]. Ususally > 3 or > 5 should be good enough')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for testing')
    parser.add_argument('--testpath', help='data path for testing')
    parser.add_argument('--testlist', help='data list for testing')
    parser.add_argument("--ckpt", help="checkpoint location")
    parser.add_argument('--outdir', default='./outputs', help='output dir')

    return parser.parse_args()


def get_point_map_estimation_opts():
    parser = argparse.ArgumentParser("Test VGGT Point Map Estimation on ETH3d datasets.", add_help=False)
    parser.add_argument("--ckpt", help="checkpoint location")
    parser.add_argument("--dataset_dir", help="dataset location")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--use_proj", action="store_true")
    return parser.parse_args()


def get_pose_evaluation_opts():
    """Set up command-line arguments for the CO3D evaluation script."""
    parser = argparse.ArgumentParser(description='Test VGGT Pose Evaluation on CO3D dataset.', add_help=False)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (only test on apple category)')
    parser.add_argument('--min_num_images', type=int, default=50, help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    parser.add_argument('--co3d_dir', type=str, required=True, help='Path to CO3D dataset')
    parser.add_argument('--co3d_anno_dir', type=str, required=True, help='Path to CO3D annotations')
    parser.add_argument("--ckpt", help="checkpoint location")
    return parser.parse_args()
