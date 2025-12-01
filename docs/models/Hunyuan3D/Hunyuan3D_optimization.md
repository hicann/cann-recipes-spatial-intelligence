# NPU Hunyuan3D 模型推理优化实践
本文主要介绍Hunyuan3D模型基于NPU的推理优化策略，其中包括以下优化点：

shapegen部分：
- 使能融合算子与高性能计算算子
- PFA算子适配
- torchair图模式适配
  - DIT 图模式适配
  - VAE-decoder 图模式适配
- CPU-NPU搬运优化

texgen部分：
- 多线程并行光栅化
- aicpu算子迁移
- Inpaint计算优化

## shapegen性能优化介绍
### 融合算子优化
**优化原因：** RmsNorm算子常见于LLaMA、LLaMA2、Baichuan等LLM模型中，由于torch侧没有提供RmsNorm算子的接口，因此在模型中通常是以自定义类的形式出现，在forward函数下定义计算逻辑，需要的运行算子比较多，计算公式为
```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

**优化方式：** 将`hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`文件中torch.rsqrt函数替换如下
```python
'''替换部分
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

'''
#替换后
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        return torch_npu.npu_rms_norm(x, self.scale, epsilon=self.eps)[0]

```

### 替代高性能算子
#### 替代高性能算子
- **优化原因：** 在神经网络中，GELU是一个重要的激活函数，其灵感来源于relu和dropout，在激活中引入了随机正则的思想。为了提升GELU算子在NPU上的运行性能，业界提出了FastGelu等版本。本接口FasterGelu是针对FastGelu的化简版本，可以大幅度提升计算性能。原理是使用泰勒展开逼近计算结果。
- **优化方式：** 修改`hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py`文件中GELU函数，使用`torch_npu.npu_fast_gelu`替换`nn.functional.gelu`的算子实现。
```python
'''替换部分
class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(x, approximate=self.approximate)
'''
#替换后
class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return torch_npu.npu_fast_gelu(x)
```

#### FIA算子适配和优化
- **优化原因：** 官方使用的`scaled_dot_product_attention`对应npu算子是`npu_fusion_attention`，`npu_fusion_attention`不支持图模式，因此如果将`npu_fusion_attention`做静态图编译时，会导致`npu_fusion_attention`编译为三个attention核心算子进行运算，带来性能劣化。
    - `BatchMatMul`
    - `SoftmaxV2`
    - `BatchMatMul`
- **优化方式：** 使用`torch_npu.fused_infer_attention_score`代替`torch_npu.npu_fusion_attention`，FIA同时适配增量&全量推理场景的FlashAttention算子，能够进行图编译，并且同时解决了`Softmax`运算溢出问题以及内存连续问题。进行如下优化：
```python
'''替换部分
scaled_dot_product_attention = nn.functional.scaled_dot_product_attention

def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    x = scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x
'''
# 替换后
def npu_fia(q, k, v, scale):
    attn_mask = None
    batch_size, num_head, seq_len, head_dim = q.shape
    out = torch_npu.torch_npu.fused_infer_attention_score(
        q, k, v, num_heads=num_head, input_layout="BNSD", scale=scale, atten_mask=attn_mask
    )[0]
    return out

def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    x = npu_fia(q, k, v, scale=(1 / math.sqrt(head_dim)))
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

```
### torchair图模式适配
- **优化原因：**  在 PyTorch 的单算子执行模式下，算子不仅可能面临主机端下发的host bound，还可能错过未被识别的算子融合机会而未能充分发挥性能。相较于单算子模式，图模式可以通过计算图优化、多流并行、内存复用和模型下沉等技术手段，加速模型执行效率，减少模型内存占用。因此可通过 torch.compile 启用图模式以实现性能优化。[TorchAir（即 Torch Ascend Intermediate Representation）](https://www.hiascend.com/document/detail/zh/Pytorch/720/modthirdparty/torchairuseguide/torchair_00003.html)作为 Ascend Extension for PyTorch（torch_npu）的图模式功能拓展工具库，提供了昇腾设备亲和的torch.compile图模式后端。它能够助力 PyTorch 网络在昇腾 NPU 上开启图模式推理，进而实现运算加速与性能层面的优化提升。
-  **优化方式：** 在模型推理场景下，使能DIT model模块以及VAE decoder模块图编译可以避免标量入图导致的图编译失败，从而提高推理性能。同时启用tiling图下沉优化。
```python
#增加部分
if args.full_graph:
    import torchair
    import torch._dynamo
    config = torchair.CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
#编译入图部分     
if args.full_graph:
    pipeline.model = torch.compile(pipeline.model, dynamic=False, backend=npu_backend, fullgraph=True)
    pipeline.vae.geo_decoder = torch.compile(pipeline.vae.geo_decoder, dynamic=False, backend=npu_backend, fullgraph=True)
#图模式指令      
python minimal_demo_npu.py --full_graph 
```      
### CPU-NPU搬运优化
- **优化原因：** 从CPU搬运到NPU存在搬运耗时，频繁的搬运会导致性能劣化，在生成时间步的步骤尤其明显，
- **优化方式：** 直接在NPU上生成时间步
```python
'''替换部分
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )
'''
#替换后
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32,device=t.device) / half)
```

## texgen性能优化介绍
### 多线程并行光栅化
**优化原因：** CPU侧执行光栅化会带来巨大时延，Hunyuan3D texgen在6个不同视角分别进行法向信息、位姿信息、纹理信息投影，共进行18次光栅化。由于光栅化相互之间与delighting来自不同数据通路，没有数据依赖关系，可以采用并行计算的方式，隐藏光栅化时延。

**优化方法：** 将法向计算和位姿计算，以及delighting过程中相互间没有依赖关系的计算过程施加并行，提升计算效率。对hy3dgen/texgen/pipelines.py中的代码做如下替换：

```python
'''替换部分
images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]
...
normal_maps = self.render_normal_multiview(
selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
position_maps = self.render_position_multiview(
selected_camera_elevs, selected_camera_azims)
...
texture, mask = self.bake_from_multiview(multiviews,
                                         selected_camera_elevs,selected_camera_azims,selected_view_weights,
                                         method=self.config.merge_method)
'''

#替换后

def _delighting_render_multiview(self,images_prompt,selected_camera_elevs,selected_camera_azims
    ,use_abs_coor=True):
    total_tasks = len(images_prompt) + 2 * len(selected_camera_elevs)
    max_workers = min(20, total_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        normal_futures = {}
        position_futures = {}

        for idx,(elev,azim) in enumerate(zip(selected_camera_elevs, selected_camera_azims)):
            normal_future = executor.submit(
                self.render.render_normal, elev, azim,
                use_abs_coor = use_abs_coor,return_type = 'pl'
            )
            normal_futures[normal_future] = idx

            position_future = executor.submit(
                self.render.render_position, elev, azim,
                use_abs_coor=use_abs_coor, return_type='pl'
            )
            position_futures[position_future] = idx

    rast_out_maps = [None] * len(selected_camera_elevs)
    normal_maps = [None] * len(selected_camera_elevs)
    position_maps = [None] * len(selected_camera_elevs)
    images_prompt = [self.moels['delight_model'](image_prompt) for image_prompt in images_prompt]

    for future in as_completed(list(normal_futures.keys()) + list(position_futures.keys())):
        try:
            result = future.result()
            if future in normal_futures:
                idx = normal_futures[future]
                normal_maps[idx] = normal
            elif future in position_futures:
                idx = position_futures[future]
                position_maps[idx] = result
        except Exception as e:
            print(f"Task failed with error: {e}")
    
    return images_prompt, normal_maps, position_maps

def _bake_from_multiview(self, view, camera_elevs,
                         camera_azim, view_weights,method='graphcut'):
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for idx,(view,camera_elev, camera_azim, weight) in enumerate(zip(
            views, camera_elevs, camera_azims, view_weights)):
            future = executor.submit(self.render.back_project,
                                     view, camera_elev, camera_azim)
            futures.append(future)
        
        project_textures = []
        project_weighted_cos_maps = []
        project_boundary_maps = []

        for future in as_completed(futures):
            project_texture, project_weighted_cos_map, project_boundary_map = future.result()
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_weighted_cos_map)
            project_boundary_maps.append(project_boundary_map)

def __call__(self,mesh,image):
    ...
    # 对应光栅化和delighting部分
    images_prompt, normal_maps, position_maps = self._delighting_render_multiview(images_prompt, selected_camera_elevs,
                                  selected_camera_azims, use_abs_coor=True)
    ...
    # 对应multiview部分
    texture, mask = self._bake_from_multiview(multiviews, selected_camera_elevs
                                              selected_camera_azims, selected_view_weights,
                                              method = self.config.merge_method)
```

### aicpu算子迁移
**优化原因：** FP32数据格式会导致torch.mean、torch.std算子运行在aicpu上，性能下降。

**优化方法：** 将FP32类型的数据手动进行转换，使其运行在aicore上。

```python
# hy3dgen/texgen/pipelines.py
'''替换部分
texture = torch.tensor(texture_np / 255).float().to(texture.device)
'''

#替换为
texture = torch.tensor(texture_np / 255).to(torch.float16).to(texture.device)
```

```python
# hy3dgen/texgen/utils/dehighlight_utils.py
'''替换部分
src_mean, src_stddev = torch.mean(src_flat[:, i].to(torch.float32)), torch.std(src_flat[:, i].to(torch.float32))
target_mean, target_stddev = torch.mean(target_flat[:, i].to(torch.float32)), torch.std(target_flat[:, i].to(torch.float32))
'''

#替换为
src_mean, src_stddev = torch.mean(src_flat[:, i].to(torch.float16)), torch.std(src_flat[:, i].to(torch.float16))
target_mean, target_stddev = torch.mean(target_flat[:, i].to(torch.float16)), torch.std(target_flat[:, i].to(torch.float16))

'''替换部分
image_tensor = torch.tensor(np.array(image)/255.0).to(self.device)
'''

#替换为
image_tensor = torch.tensor(np.array(image) / 255.0).to(torch.float16).to(self.device)
```

### Inpaint计算优化
**优化原因：** 生成的3Dmesh纹理仍存在部分未上色点，需要通过对其平滑化，补全未上色点。源码实现了python代码和c++代码，默认使用python代码进行着色点平滑，但是c++具有更好的性能，需要手动进行替换。

**优化方法：** 手动替换为c++代码。
```python
# hy3dgen/texgen/differentiable_renderer/mesh_render.py
'''替换部分
from .mesh_processor import meshVerticeInpaint
'''

#替换为
import mesh_processor

'''替换部分
texture_np, mask = meshVerticeInpaint(
                   texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
'''

#替换为
texture_np, mask = mesh_processor.meshVerticeInpaint(
                   texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
```

### 光栅化结果复用优化
**优化原因：** 源码的18次光栅化计算中，只有6种不同输入数据，其余12次计算为冗余计算，造成性能损失。

**优化方法：** 将第一次计算的光栅化过程进行保存，每当输入相同的相机位姿，算法就从缓存中直接读取计算结果。
```python
'''替换部分
rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)
'''
#替换为
if (self.save_render and 
            (elev, azim, camera_distance) in self.render_result.keys()):
            rast_out = self.render_result[(elev, azim, camera_distance)]

...

self.render_result[(elev, azim, camera_distance)] = rast_out
```
### 光栅化过程npu迁移
**优化原因：** Hunyuan3D源码光栅化没有可在NPU执行的版本，会将光栅化过程迁移至CPU侧执行，不必要的内存搬运和串行计算会带来巨大时延。为了适配NPU架构，提升光栅化性能，本仓库引入render_npu，将光栅化的计算过程迁移至NPU执行。

**优化方法：** 修改光栅化算法，将原先采用遍历mesh顶点的方式，改为对二维平面分块，每个区域中的三角网格合并成一个矩阵计算深度、重心坐标等参数，最终判断每个像素点对应的三角面片。由NPU侧执行小算子计算不能同时进行光栅化并行。
```python
'''替换部分
rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)
'''
#替换为
if self.use_render_npu:
    rast_out, _ = render_npu_rasterize(
                    resolution, self.pos_idx, pos_clip
    )
else:
    rast_out, _ = self.raster_rasterize(
                    pos_clip, self.pos_idx, resolution=resolution
    )
```
## 性能优化指标
### shapegen性能优化指标
本方案使用1卡Atlas 800I A2推理产品，输入hunyuan3D 2.0 提供的样例数据(`assets/example_images/004.png`)，在扩散步数为num_inference_steps=100（minimal_demo_npu.py第89行与111行）情况下,性能指标如下
|使能方法|DIT 扩散耗时（s）|VAE 生成耗时（s）|
|:---:|:---:|:---:|
|baseline|21.63|17.98|
|算子优化＋搬运优化|20.07|18.71|
|算子优化+图模式编译|19.00|16.68|

### texgen性能优化指标
本方案使用1卡Atlas 800I A2推理产品，输入hunyuan3D 2.0 提供的样例数据(`assets/example_images/004.png`)，输入mesh网格面数量为20000，开启减少面数量（--face_reduce）性能指标如下
|使能方法|texgen运行时长（s）|
|:---:|:---:|
|baseline|59.83|
|光栅化并行|35.43|
|光栅化并行+算子优化|30.46|
|NPU光栅化+算子优化+Inpaint优化+光栅化结果复用|32.86|
|光栅化并行+算子优化+Inpaint优化+光栅化结果复用|26.52|