# 在昇腾Atlas A2环境上适配VGGT模型的推理
本样例基于[VGGT开源模型](https://github.com/facebookresearch/vggt)完成其在NPU上的推理适配，并提供其在相机位姿估计、点云重建、深度估计三个任务上的精度评测脚本。详细内容可至[精度评测章节](https://gitcode.com/cann/cann-recipes-spatial-intelligence/blob/master/docs/models/vggt/vggt_accurancy_evaluation.md)查看。

此外，本样例基于VGGT模型在NPU进行了性能优化，目前VGGT模型在25张图片输入下，推理时间下降至1.12秒。详细内容可至[性能优化章节](https://gitcode.com/cann/cann-recipes-spatial-intelligence/blob/master/docs/models/vggt/vggt_optimization.md)查看。

---
## 执行样例
### CANN环境准备
1. 本样例的执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），目前使用CANN软件版本为`CANN.8.0.RC3.beta1`。
请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.0.RC3.beta1)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)进行安装。

2. 本样例依赖的torch以及torch_npu版本为2.1.0。
请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html)下载torch与torch_npu安装包，本样例依赖的torch与torch_npu版本分别为2.1.0和2.1.0.post12。
    ```shell
    conda create -n vggt python==3.11.13
    conda activate vggt
    pip3 install torch==2.1.0
    pip3 install torch_npu==2.1.0.post12
    ```
### 网络模型代码准备
- 本仓库依赖[VGGT](https://github.com/facebookresearch/vggt/tree/main)的开源仓库代码。
- 进入VGGT的官方仓库，下载VGGT模型网络结构代码：
  ```shell
  git clone https://github.com/facebookresearch/vggt.git
  ```
- 下载本仓库代码：
  ```shell
  git clone https://gitcode.com/cann/cann-recipes-spatial-intelligence.git
  ```
- 将VGGT仓库的网络模型文件以**非覆盖模式**复制到本项目目录下。
   ```shell
  cp vggt/visual_utils.py cann-recipes-spatial-intelligence/models/vggt/
  cp -r vggt/examples cann-recipes-spatial-intelligence/models/vggt/
  cp -rn vggt/vggt/dependency cann-recipes-spatial-intelligence/models/vggt/vggt/dependency
  cp -rn vggt/vggt/heads cann-recipes-spatial-intelligence/models/vggt/vggt/
  cp -rn vggt/vggt/layers cann-recipes-spatial-intelligence/models/vggt/vggt/
  cp -rn vggt/vggt/utils cann-recipes-spatial-intelligence/models/vggt/vggt/ 
  ```
- 安装Python依赖：
  ```shell
  cd cann-recipes-spatial-intelligence/models/vggt/
  pip3 install -r requirements.txt
  ```
- VGGT 模型权重下载：[VGGT model checkpoint](https://huggingface.co/spaces/facebook/vggt)，并将权重文件`model.pt`复制到ckpt目录下。
- 模型权重与模型结构在文件目录中罗列如下：
  ```
  VGGT
    +--- examples
    +--- demo_infer.py
    +--- eval
    +--- ckpt
          +--- model.pt
    +--- quant
    +--- vggt
          +--- dependency
          +--- heads
          +--- layers
          +--- models
          +--- utils
  ```

### 快速启动
本样例准备了单卡环境下的推理样例脚本。
执行脚本前，请参考[Ascend社区](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)中的CANN安装软件教程，配置环境变量：
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```
推理bf16模型脚本运行：
```python
python demo_infer.py --ckpt "ckpt/model.pt"
```

推理int8模型，需要先生成int8模型(当前实现中，只将VGGT模型中K=4096的Linear层进行了8bit量化)：
```python
python demo_infer.py --ckpt "ckpt/model.pt" --buildW8A8
```
in8模型会生成在当前路径，再使用该int8模型进行推理：
```python
python demo_infer.py --ckpt VGGT_model_W8A8.pt --enableW8A8
```
---
## Citation
```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
