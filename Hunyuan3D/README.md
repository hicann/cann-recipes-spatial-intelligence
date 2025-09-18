# 在昇腾训练平台上适配Hunyuan3D 2.0 模型的推理

Hunyuan3D模型是腾讯混元系列在2025年推出的一款3D资产创作模型，用于生成带有高分辨率纹理贴图的高保真度3D模型，本项目旨在提供Hunyuan3D的NPU适配版本，方便用户能够在昇腾生态上直接使用Hunyuan3D。

## 执行样例

### 环境准备

1. 本样例采用CANN 8.2.RC1（安装低于该版本可能会导致conv2d算子性能问题）。请参考[Ascend社区](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)中的CANN安装软件教程，配置环境变量，部署在昇腾Atlas A2硬件平台

2. 本样例的torch以及torch_npu版本为2.5.1

3. 本仓库基于Hunyuan3D源代码进行修改，使其适配NPU的运行环境。用户可根据以下文件结构进行修改：
```
Hunyuan3-2
    +--- hy3dgen
        +--- texgen
            +--- custom_rasterizer ==> custom_rasterizer in current depository #替换当前文件
            +--- differentiable_renderer
                +--- mesh_render.py ==> differentiable_renderer/mesh_render.py #替换文件
                +--- rasterizer.py #添加新文件
    +--- minimal_demo_npu.py #添加新文件

```
对```hy3dgen/shapegen/__init__.py```文件下代码做如下修改
```python
# 去除不支持的第三方库
15 from .pipelines import Hunyuan3DDiTPipeline, Hunyuan3DDiTFlowMatchingPipeline
16 # from .postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier
17 # from .preprocessors import ImageProcessorV2, IMAGE_PROCESSORS, DEFAULT_IMAGEPROCESSOR
```

对```hy3dgen/texgen/utils/dehighlight_utils.py```文件下代码做如下修改，确保数据类型符合NPU要求
```python
# Hunyuan3D源码
52 # src_mean, src_stddev = torch.mean(src_flat[:, i]), torch.std(src_flat[:, i])
53 # target_mean, target_stddev = torch.mean(target_flat[:, i]), torch.std(target_flat[:, i])

# 修改为
52 src_mean, src_stddev = torch.mean(src_flat[:, i].to(torch.float32)), torch.std(src_flat[:, i].to(torch.float32))
53 target_mean, target_stddev = torch.mean(target_flat[:, i].to(torch.float32)), torch.std(target_flat[:, i].to(torch.float32))
```
### 依赖安装

```bash
# 创建虚拟环境
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
conda create -n Hunyuan3D python=3.10
conda activate Hunyuan3D 

cd Hunyuan3D-2
pip install -r requirements.txt
pip install -e .

# for texture
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```


### 权重模型
NPU环境支持Hunyuan3D模型列表，用户可根据需要下载蒸馏版本：


**Hunyuan3D-2mv 系列**

| Model                     | Description                    | Date       | Size | Huggingface                                                                                  |
|---------------------------|--------------------------------|------------|------|----------------------------------------------------------------------------------------------| 
| Hunyuan3D-DiT-v2-mv       | Multiview Image to Shape Model | 2025-03-18 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/Hunyuan3D-dit-v2-mv)       |

**Hunyuan3D-2 系列**

| Model                      | Description                 | Date       | Size | Huggingface                                                                               |
|----------------------------|-----------------------------|------------|------|-------------------------------------------------------------------------------------------| 
| Hunyuan3D-DiT-v2-0         | Image to Shape Model        | 2025-01-21 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/Hunyuan3D-dit-v2-0)         |
| Hunyuan3D-Paint-v2-0       | Texture Generation Model    | 2025-01-21 | 1.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/Hunyuan3D-paint-v2-0)       |
| Hunyuan3D-Delight-v2-0     | Image Delight Model         | 2025-01-21 | 1.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/Hunyuan3D-delight-v2-0)     | 


## 执行推理
### 执行推理

```bash
python minimal_demo_npu.py
```
``` minimal_demo_npu.py```采用默认设置执行单图像推理，运行不同配置可参考以下脚本执行：

```bash
python minimal_demo_npu.py --model_path tencent/Hunyuan3D-2 --mutiview True --face_reduce True
```
```model_path``` 选择模型路径，```mutiview``` 设置是否采用多视角推理，```reduce_face``` 设置是否减少三角面片。

### 推理步骤介绍

Hunyuan3D源码参考```diffusers```接口API，用于物体生成，包括**Hunyuan3D-DiT**和**Hunyuan3D-Paint**两个部分。

运行**Hunyuan3D-DiT**方式如下：
```python
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from torch_npu.contrib import transfer_to_npu

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]
```

运行**Hunyuan3D-Paint**方式如下：
```python
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from torch_npu.contrib import transfer_to_npu

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/demo.png')
```
**Hunyuan3D-DiT**执行过程中，我们在NPU环境下提供多视角模型推理，具体方式如下：
```python
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from torch_npu.contrib import transfer_to_npu

images = {
    "front": "assets/example_mv_images/1/front.png",
    "left": "assets/example_mv_images/1/left.png",
    "back": "assets/example_mv_images/1/back.png"
}

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
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

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/demo.png')
```

## Citation
```
@misc{lai2025hunyuan3d25highfidelity3d,
      title={Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details}, 
      author={Tencent Hunyuan3D Team},
      year={2025},
      eprint={2506.16504},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.16504}, 
}

@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yang2024hunyuan3d,
    title={Hunyuan3D 1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Tencent Hunyuan3D Team},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```