# Research-on-Infrared-In-Situ-Small-Target-Detection-Method-Based-on-Deep-Learning
Research-on-Infrared-In-Situ-Small-Target-Detection-Method-Based-on-Deep-Learning
### 环境配置

执行下列指令创建并激活虚拟环境

```bash

conda create -n yolo python==3.8.5

conda activate yolo

```

执行下列执行安装pytorch

```bash

conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本

conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令



在\*\*项目目录下\*\*执行下列指令进行其他库的安装



```bash

pip install -v -e .

```

环境创建完成之后请使用pycharm打开你的项目，并在pycharm的右下角选择你项目对应的虚拟环境。

1. 在`block.py`或者`conv.py`中添加你要修改的模块，比如我在这里添加了se的类，包含了输入和输出的通道数。

2. 在`init.py`文件中引用。

3. 在`task.py`文件中引用。



 CBAM: Convolutional Block Attention Module

     论文地址：[[1807.06521\] CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

     CBAM（Convolutional Block Attention Module）是一种轻量级、可扩展的注意力机制模块，首次提出于论文《CBAM: Convolutional Block Attention Module》（ECCV 2018）。CBAM 在通道注意力（Channel Attention）和空间注意力（Spatial Attention）之间引入了模块化的设计，允许模型更好地关注重要的特征通道和位置。

     CBAM 由两个模块组成：

     **通道注意力模块 (Channel Attention Module)**: 学习每个通道的重要性权重，通过加权增强重要通道的特征。

     **空间注意力模块 (Spatial Attention Module)**: 学习空间位置的重要性权重，通过加权关注关键位置的特征。

     该模块的代码实现如下：

     ```python
     import torch
     import torch.nn as nn
 
     class ChannelAttention(nn.Module):
         def __init__(self, in_channels, reduction=16):
             """
             通道注意力模块
             Args:
                 in_channels (int): 输入通道数
                 reduction (int): 缩减比例因子
             """
             super(ChannelAttention, self).__init__()
             self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
             self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化
 
             self.fc = nn.Sequential(
                 nn.Linear(in_channels, in_channels // reduction, bias=False),
                 nn.ReLU(inplace=True),
                 nn.Linear(in_channels // reduction, in_channels, bias=False)
             )
             self.sigmoid = nn.Sigmoid()
 
         def forward(self, x):
             batch, channels, _, _ = x.size()
 
             # 全局平均池化
             avg_out = self.fc(self.avg_pool(x).view(batch, channels))
             # 全局最大池化
             max_out = self.fc(self.max_pool(x).view(batch, channels))
 
             # 加和后通过 Sigmoid
             out = avg_out + max_out
             out = self.sigmoid(out).view(batch, channels, 1, 1)
 
             # 通道加权
             return x * out
 
 
     class SpatialAttention(nn.Module):
         def __init__(self, kernel_size=7):
             """
             空间注意力模块
             Args:
                 kernel_size (int): 卷积核大小
             """
             super(SpatialAttention, self).__init__()
             self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
             self.sigmoid = nn.Sigmoid()
 
         def forward(self, x):
             # 通道维度求平均和最大值
             avg_out = torch.mean(x, dim=1, keepdim=True)
             max_out, _ = torch.max(x, dim=1, keepdim=True)
             combined = torch.cat([avg_out, max_out], dim=1)  # 拼接
 
             # 卷积处理
             out = self.conv(combined)
             out = self.sigmoid(out)
 
             # 空间加权
             return x * out
 
 
     class CBAM(nn.Module):
         def __init__(self, in_channels, reduction=16, kernel_size=7):
             """
             CBAM 模块
             Args:
                 in_channels (int): 输入通道数
                 reduction (int): 缩减比例因子
                 kernel_size (int): 空间注意力卷积核大小
             """
             super(CBAM, self).__init__()
             self.channel_attention = ChannelAttention(in_channels, reduction)
             self.spatial_attention = SpatialAttention(kernel_size)
 
         def forward(self, x):
             # 通道注意力模块
             x = self.channel_attention(x)
             # 空间注意力模块
             x = self.spatial_attention(x)
             return x
     ```
 
  GhostConv

     **Ghost Convolution** 是一种轻量化卷积操作，首次提出于论文《GhostNet: More Features from Cheap Operations》（CVPR 2020）。GhostConv 的核心思想是利用便宜的操作生成额外的特征图，以减少计算复杂度和参数量。、

     GhostConv的核心思想如是，卷积操作会生成冗余的特征图。许多特征图之间存在高相关性。GhostConv 的目标是通过减少冗余特征图的计算来加速网络的推理。GhostConv 的结构如下：

     ![image-20250109220155390](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20250109220155390.png)

     **主特征图**: 使用标准卷积生成一部分特征图。

     **副特征图**: 从主特征图中通过简单的线性操作（如深度卷积）生成。

     代码实现如下：

     ```python
     import torch
     import torch.nn as nn
 
     class GhostConv(nn.Module):
         def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=2, dw_kernel_size=3):
             """
             Ghost Convolution 实现
             Args:
                 in_channels (int): 输入通道数
                 out_channels (int): 输出通道数
                 kernel_size (int): 卷积核大小
                 stride (int): 卷积步幅
                 padding (int): 卷积填充
                 ratio (int): 副特征与主特征的比例
                 dw_kernel_size (int): 深度卷积的卷积核大小
             """
             super(GhostConv, self).__init__()
             self.out_channels = out_channels
             self.primary_channels = out_channels // ratio  # 主特征图通道数
             self.ghost_channels = out_channels - self.primary_channels  # 副特征图通道数
 
             # 主特征图的标准卷积
             self.primary_conv = nn.Conv2d(
                 in_channels, self.primary_channels, kernel_size, stride, padding, bias=False
             )
             self.bn1 = nn.BatchNorm2d(self.primary_channels)
 
             # 副特征图的深度卷积
             self.ghost_conv = nn.Conv2d(
                 self.primary_channels, self.ghost_channels, dw_kernel_size, stride=1,
                 padding=dw_kernel_size // 2, groups=self.primary_channels, bias=False
             )
             self.bn2 = nn.BatchNorm2d(self.ghost_channels)
 
             self.relu = nn.ReLU(inplace=True)
 
         def forward(self, x):
             # 主特征图
             primary_features = self.primary_conv(x)
             primary_features = self.bn1(primary_features)
 
             # 副特征图
             ghost_features = self.ghost_conv(primary_features)
             ghost_features = self.bn2(ghost_features)
 
             # 合并主特征图和副特征图
             output = torch.cat([primary_features, ghost_features], dim=1)
             output = self.relu(output)
 
             return output
     ```

### 本地模型训练

模型训练使用的脚本为`step1\_start\_train.py`，进行模型训练之前，请先按照配置好你本地的数据集。数据集在` ultralytics\\cfg\\datasets\\A\_my\_data.yaml`目录下，你需要将数据集的根目录更换为你自己本地的目录。更换之后修改训练脚本配置文件的路径，直接右键即可开始训练。

### 模型测试
模型的测试主要是对map、p、r等指标进行计算，使用的脚本为` step2\_start\_val.py`。
