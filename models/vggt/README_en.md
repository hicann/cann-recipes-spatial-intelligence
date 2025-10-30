# VGGT inference on Ascend Atlas A2
## CANN Environment Preparaton
1. The inference of VGGT depends on the CANN development kit package (`cann-toolkit`) and the CANN binaray operator package(`cann-kernels`). The supported CANN software version is CANN 8.0.RC3.beta1.
   
    Download the `Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run` packages from the [CANN Software Package Download Page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1) and install them by referring to the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit).

2. The required versions of torch and torch_npu are 2.1.0 and 2.1.0.post13.

    Download the binary package from [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html) and install torch and torch_npu.
    ```shell
    conda create -n vggt python==3.11.13
    conda activate vggt
    pip3 install torch==2.1.0
    pip3 install torch-npu==2.1.0.post13
    ```

## VGGT Model Preparation
1. Download the open-source [VGGT network code](https://github.com/facebookresearch/vggt/tree/main) from the github repo.
   ```shell
    git clone https://github.com/facebookresearch/vggt.git
   ```
2. Download the code of this repository:
   ```shell
   git clone https://gitcode.com/chenhongyang/cann-recipes-spatial-intelligence.git
   ```
3. Copy the code from the VGGT repository to this project directory in non-overwrite mode:
    ```shell
    cp -rn vggt/vggt cann-recipes-spatial-intelligence/models/vggt/vggt 
    ```
4. Install Python dependencies:
    ```shell
    pip3 install -r requirements.txt
    ```
5. Download [VGGT model weights](https://huggingface.co/spaces/facebook/vggt) and copy it to the local path `ckpt`.
    ```
    VGGT
        +--- datasets
        +--- demo_infer.py
        +--- eval
        +--- ckpt
            +--- model.pt
        +--- vggt
            +--- dependency
            +--- heads
            +--- layers
            +--- models
            +--- utils
    ```
## Quick Start
This repo provides script to test the functionality and the performance of VGGT model on NPU.
1. Before executing the test scripts, refer to the Ascend Community CANN installation tutorial to set environment variables:
    ```shell 
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    ```
2. Run the inference script:
   ```shell
    python demo_infer.py --ckpt "ckpt/model.pt"
   ```
## Accurancy Benchmark
This repo provides accurancy benchmark to evaluate the VGGT model on NPU. The full benchmark include three programs to test the accurancy of VGGT on Pose Evaluation, Point Map Evaluation and Depth Evaluation. 

Since the full dataste of benchmark is large, we can initially test the accurancy of VGGT model in Pose Evaluation with the subset of the full Co3DV2 dataset.

### Dataset Preparation:
1. Download data `CO3D_apple.zip` and data `CO3D_backpack.zip` from [CO3D website](https://ai.meta.com/datasets/co3d-downloads/) and unzip them to `datasets/co3d/co3d_data/`.
    ```
    vggt
        +--- datasets
            +--- co3d
                +--- co3d_data
                        +--- apple
                        +--- backpack
                    ...
    ```
2. Prepare metadata of the dataset:
    ```shell
   export VGGT_DIR=$(pwd)
   cd eval/pose_evaluation/dataset_prepare
   python preprocess_co3d.py --category all --co3d_v2_dir $VGGT_DIR/datasets/co3d/co3d_data/ --output_dir $VGGT_DIR/datasets/co3d/co3d_anno/ 
    ```
### Accurancy Measurement
- Execute the benchmark program:
    ```shell
    export VGGT_DIR=$(pwd)
    cd eval/pose_evaluation
    python eval_co3d.py --co3d_dir $VGGT_DIR/datasets/co3d/co3d_data/
        --co3d_anno_dir $VGGT_DIR/datasets/co3d/co3d_anno/  --ckpt $VGGT_DIR/ckpt/model.pt
    ```
- **Currently, the measurement accurancy is about 0.85.**