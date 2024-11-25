import torch
import torch.nn as nn

import torch_scatter

class LayerNorm1D(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1D, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def forward(self, x):
        return self.norm(x.transpose(1,2)).transpose(1,2)

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()) #激活函数，将输入的值映射到0-1之间

        self.res = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)

class SoftAgg(nn.Module):#继承自 nn.Module（用于实现一种基于索引的特征聚合机制）
    def __init__(self, dim=512, expand=True):#输出参数为特征维度dim=512，expand=True（决定是否在最后返回时扩展结果）
        super(SoftAgg, self).__init__()
        self.dim = dim
        self.expand = expand
        # 设置三个全连接层
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim, self.dim)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix):#输入的为特征图x和特征图的索引ix
        # unique是去重函数，返回去重后的值和索引，jx为去重后的索引，进而可以知道特征图中，哪些特征是同一个patch的
        _, jx = torch.unique(ix, return_inverse=True)#获取哪些特征属于同一个patch
        #根据去重的索引jx， 用torch_scatter操作，计算相同索引的加权特征
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)#将特征图x在第1维度上按照索引jx进行softmax操作
        # 计算聚合特征
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]
            
        return self.h(y)

class SoftAggBasic(nn.Module):
    def __init__(self, dim=512, expand=True):
        super(SoftAggBasic, self).__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(self.dim, self.dim)
        self.g = nn.Linear(self.dim,        1)
        self.h = nn.Linear(self.dim, self.dim)

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        w = torch_scatter.scatter_softmax(self.g(x), jx, dim=1)
        y = torch_scatter.scatter_sum(self.f(x) * w, jx, dim=1)

        if self.expand:
            return self.h(y)[:,jx]
            
        return self.h(y)


### Gradient Clipping（梯度剪裁） and Zeroing Operations ###

GRAD_CLIP = 0.1

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        return grad_x.clamp(min=-0.01, max=0.01)

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)

class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x), torch.zeros_like(grad_x), grad_x)
        grad_x = torch.where(torch.abs(grad_x) > GRAD_CLIP, torch.zeros_like(grad_x), grad_x)
        return grad_x

class GradientZero(nn.Module):
    def __init__(self):
        super(GradientZero, self).__init__()

    def forward(self, x):
        return GradZero.apply(x)


class GradMag(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        print(grad_x.abs().mean())
        return grad_x
