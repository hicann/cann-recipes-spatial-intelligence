# cann-recipes-spatial-intelligence

## 🚀Latest News
- [2025/11] Hunyuan3D模型在昇腾Atlas A2系列上已支持推理，代码已开源。
- [2025/11] VGGT模型在昇腾Atlas A2系列上已支持推理，代码已开源。

## 🎉概述
cann-recipes-spatial-intelligence仓库旨在针对空间智能业务中的典型模型、加速算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地基于CANN平台使用空间智能模型。


## ✨样例列表
|实践|简介|
|-----|-----|
|[VGGT](models/vggt/README.md)|基于VGGT开源模型，完成其在Atlas A2上的推理适配，并提供其在相机位姿估计、点云重建、深度估计三个任务上的精度评测脚本。
|[Hunyuan3D](models/Hunyuan3D/README.md)|基于Hunyuan3D开源模型，完成其在Atlas A2上的推理适配，并通过使能融合算子、图模式、多线程并行光栅化等优化手段，实现了较低的推理时延。


## 📖目录结构说明
```
├── docs                                        # 文档目录
|  └── models                                   # 模型文档目录
|     ├── Hunyuan3D                             # Hunyuan3D相关文档
|     ├── vggt                                  # VGGT相关文档
|     └── ...
├── models                                      # 模型脚本目录
|  ├── Hunyuan3D                                # Hunyuan3D的模型脚本及执行配置
|  ├── vggt                                     # vggt的模型脚本及执行配置
│  └── ...
└── contrib                                     # 社区用户贡献的模型与文档目录
|  ├── README.md
│  └── ...
└── CONTRIBUTION.md
└── DISCLAIMER.md
└── LICENSE
└── README.md
└── ...
```

## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- [许可证](./LICENSE)
    
    cann-recipes-spatial-intelligence仓涉及的模型，如模型目录下存在License的以该License为准。如模型目录下不存在License的，遵循Apache 2.0许可证，对应协议文本可查阅[LICENSE](./LICENSE)
- [免责声明](./DISCLAIMER.md)