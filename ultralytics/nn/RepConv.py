import torch
import torch.nn as nn
from torch.nn import functional as F


class RepConv(nn.Module):
    """
    实现重参数化卷积（RepConv），通过对传统卷积核的动态调整来提升表示能力。

    参数:
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int): 卷积核大小。
    - stride (int): 卷积步长。
    - padding (int, optional): 卷积填充大小，默认根据kernel_size自动计算。
    - groups (int): 分组卷积中的组数，默认为1，表示不使用分组卷积。
    - map_k (int): 用于动态调整卷积核的小卷积核大小，默认为3。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1, map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size  # 确保动态调整卷积核的大小不大于原卷积核大小

        # 记录原始卷积核形状
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        # 注册一个buffer用于存储零初始化的原始卷积核权重
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))

        # 计算G的值，用于分组卷积时的组数计算
        G = in_channels * out_channels // (groups ** 2)
        # 计算2D卷积核数量
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size

        # 定义用于卷积核动态调整的小卷积层
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        # 初始化偏置为None，为了与传统卷积保持一致，这里不使用偏置
        self.bias = None

        self.stride = stride
        self.groups = groups

        # 如果未指定padding，自动计算
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        """
        前向传播函数。

        参数:
        - inputs (Tensor): 输入特征图。

        返回:
        - 经过重参数化卷积操作后的输出特征图。
        """
        # 重塑原始卷积核权重以适应动态调整的小卷积层
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        # 计算动态调整后的卷积核
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        # 使用动态调整后的卷积核执行卷积操作
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                        bias=self.bias)


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = RepConv(64, 64, kernel_size=3, stride=1).cuda()
    input = torch.rand(3, 64, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())
