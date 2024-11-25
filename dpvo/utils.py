import torch
import torch.nn.functional as F


all_times = []

class Timer:
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled

        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()
        
    def __exit__(self, type, value, traceback):
        global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed = self.start.elapsed_time(self.end)
            all_times.append(elapsed)
            print(self.name, elapsed)


def coords_grid(b, n, h, w, **kwargs):
    """ coordinate grid """
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)
    coords = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    return coords[[1,0]].view(1, 1, 2, h, w).repeat(b, n, 1, 1, 1)

# 用于生成包含帧索引的坐标网格
def coords_grid_with_index(d, **kwargs):
    """ coordinate grid with frame index"""
    # d为输入的视差图，前面初始化为1，size跟输入的image一样。
    b, n, h, w = d.shape
    i = torch.ones_like(d)
    # 使用 torch.arange 生成从 0 到 w-1 的一维张量 x 和从 0 到 h-1 的一维张量 y
    x = torch.arange(0, w, dtype=torch.float, **kwargs)
    y = torch.arange(0, h, dtype=torch.float, **kwargs)

    # 使用 torch.meshgrid 生成二维网格坐标，并使用 torch.stack 将其堆叠成 [2, h, w] 的张量。
    y, x = torch.stack(torch.meshgrid(y, x, indexing="ij"))
    # 将 y 和 x 调整形状为 [1, 1, h, w]，然后沿批量维度和帧数维度重复，使其形状变为 [b, n, h, w]。
    # 获取二维坐标网格
    y = y.view(1, 1, h, w).repeat(b, n, 1, 1)
    x = x.view(1, 1, h, w).repeat(b, n, 1, 1)

    # 组合坐标与帧索引：
    coords = torch.stack([x, y, d], dim=2) #将 x、y 和 d 堆叠在一起，形成形状为 [b, n, 3, h, w] 的张量 coords，其中 3 维表示 (x, y, d) 坐标。
    # 生成帧索引：
    index = torch.arange(0, n, dtype=torch.float, **kwargs)#使用 torch.arange 生成从 0 到 n-1 的一维张量 index。
    # 将 index 调整形状为 [1, n, 1, 1, 1]，然后沿批量维度和空间维度重复，使其形状变为 [b, n, 1, h, w]。
    index = index.view(1, n, 1, 1, 1).repeat(b, 1, 1, h, w)

    return coords, index

def patchify(x, patch_size=3):
    """ extract patches from video """
    b, n, c, h, w = x.shape
    x = x.view(b*n, c, h, w)
    y = F.unfold(x, patch_size)
    y = y.transpose(1,2)
    return y.reshape(b, -1, c, patch_size, patch_size)


def pyramidify(fmap, lvls=[1]):#输入的lvls为[1,4]，fmap是H/4*W/4的matching feature
    """ turn fmap into a pyramid """
    b, n, c, h, w = fmap.shape #解fmap 的形状。

    pyramid = [] #初始化一个空列表，用于存储金字塔的每一层。
    for lvl in lvls:
        # 将 fmap 重新形状为 (b*n, c, h, w)，然后使用平均池化操作进行下采样。池化核大小和步长均为 lvl。
        gmap =  F.avg_pool2d(fmap.view(b*n, c, h, w), lvl, stride=lvl) 
        # 将池化后的特征图重新形状为 (b, n, c, h//lvl, w//lvl)，并添加到 pyramid 列表中。
        pyramid += [ gmap.view(b, n, c, h//lvl, w//lvl) ]
    
    #对于 lvl = 1：池化后尺寸不变。即将 fmap 重新形状为 (b, n, c, h/4, w/4)，并添加到 pyramid 列表中。
    # 对于 lvl = 4：池化后尺寸为原来的 1/4。即 (b, n, c, H/16, W/16)。
        
    return pyramid

def all_pairs_exclusive(n, **kwargs):
    ii, jj = torch.meshgrid(torch.arange(n, **kwargs), torch.arange(n, **kwargs))
    k = ii != jj
    return ii[k].reshape(-1), jj[k].reshape(-1)

def set_depth(patches, depth):
    patches[...,2,:,:] = depth[...,None,None]
    return patches

def flatmeshgrid(*args, **kwargs):
    # 生成网格坐标，参数对应网格的坐标
    grid = torch.meshgrid(*args, **kwargs)
    return (x.reshape(-1) for x in grid) #将网格中的每个张量展平为一维张量。

