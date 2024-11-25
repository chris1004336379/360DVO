import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fastba
from . import altcorr
from .lietorch import SE3

from .extractor import BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.amp.autocast
# import matplotlib.pyplot as plt

DIM = 384

# update operator，是一个Recurrent网络
class Update(nn.Module):
    # 初始化函数（输入patch size）
    def __init__(self, p):
        # 调用了父类 nn.Module 的构造函数，确保父类中的初始化操作也得以执行。
        super(Update, self).__init__()

        # 全链接层，输入维度为DIM，输出维度为DIM
        # 每个序列包含两个线性层和一个 ReLU 激活函数。
        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),#，inplace=True 表示直接在输入上进行操作以节省内存。
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),#，inplace=True 表示直接在输入上进行操作以节省内存。
            nn.Linear(DIM, DIM))
        
        # 层归一化层，标准化输入以改善训练的稳定性。
        self.norm = nn.LayerNorm(DIM, eps=1e-3) #eps=1e-3：一个小值，用于避免除零错误。

        # softMax aggregation(文献：Superglue，做learning的特征点匹配中出现类似结构)
        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        # 包含两个层归一化层和两个GatedResidual模块。（用于做transition block处理）
        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        # 全连接层序列（用于处理correlation matching feature，将其转换为384）
        # 输入维度为 2*49*p*p，输出维度为 DIM，包含3个全连接层，2 ReLU 激活函数和1层归一化。
        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        # 全连接层序列，包含自定义模块GradientClip。
        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        # 全连接层序列，包含自定义模块GradientClip。相当于权重？
        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid()) #最后包含一个 Sigmoid 激活函数，将输出压缩到 [0, 1] 范围。


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """
        # corr应该是correlation matching feature？
        # imap应该是patch的context feature
        # 那么net应该就是hidden state？
        # 传入的为(net, imap[:,kk], corr, None, ii, jj, kk)
        # ii是关键帧k patch对应的时间。j为重投映的时间。kk是关键帧列表？

        net = net + inp + self.corr(corr)#self.corr处理correlation matching feature，将其转换为384
        net = self.norm(net) #归一化层

        ix, jx = fastba.neighbors(kk, jj)#获取邻居索引
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        # 进行1D卷积操作
        # net的size是b, len(kk), DIM,
        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        # 基于索引的特征聚合
        net = net + self.agg_kk(net, kk) #这代表的应该是同一个patch在不同时间戳的聚合
        net = net + self.agg_ij(net, ii*12345 + jj) #这代表的应该是两个时间戳，不同的patch的聚合
        # 乘以 12345 是为了确保 ii 和 jj 的组合是唯一的，不会产生冲突？？？？

        # 做transition block处理
        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)

# 负责将输入图像分割成小块（patch）并进行处理。
class Patchifier(nn.Module): #继承自 nn.Module 的类，表示一个神经网络模型。
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__() #继承父类的初始化函数
        self.patch_size = patch_size #patch_size为3（传入及默认的都为3）
        # 卷积网络的编码器。主要作用是对输入图像进行特征提取，经过多个卷积层和归一化层的处理，最后输出一个指定维度的特征图。
        # fornt-net（输入均为image）
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance') #输出维度为128，norm_fn为归一化形式（进行patch特征的提取）
        # inner-net（输入均为image）
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')#输出维度为384（DIM）
    
    # 计算图像的梯度
    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    # 用于从输入图像中提取特征块。
    def forward(self, images, patches_per_image=80, disps=None, gradient_bias=False, return_color=False):
        """ extract patches from input images """

        # 进行特征提取（将提取的特征图缩放到四分之一大小。）
        fmap = self.fnet(images) / 4.0 #通过fnet对输入图像进行特征提取，然后除以4.0，获取特征图 fmap（matching feature）
        imap = self.inet(images) / 4.0 #通过inet对输入图像进行特征提取，然后除以4.0，获取内部特征图 imap （context feature）

        # 获取特征图的形状，分别是批次大小 b、图像数量 n、通道数 c、高度 h 和宽度 w。
        b, n, c, h, w = fmap.shape

        # bias patch selection towards regions with high gradient
        if gradient_bias:#参数传入为false
            # 计算图像的梯度
            g = self.__image_gradient(images)

            # 随机生成多个候选坐标 x 和 y。
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")
            # torch.randint 用于生成范围在 [1, w-1) 和 [1, h-1) 之间的随机整数。
            # n 是图像的数量，patches_per_image 是每张图像的补丁数量。

            # 将 x 和 y 坐标堆叠在一起形成一个新的张量 coords，其形状为 [n, 3*patches_per_image, 2]。
            # dim=-1 表示在最后一个维度上堆叠，使得每个坐标对 (x, y) 成为二维坐标。
            coords = torch.stack([x, y], dim=-1).float()
            # 根据坐标 coords 从梯度图 g 中提取patch。然后展平成一维张量。
            g = altcorr.patchify(g, coords, 0).view(-1)
            
            ix = torch.argsort(g)#进行梯度值的排序
            # 提取梯度值最大的 patches_per_image 个坐标。
            x = x[:, ix[-patches_per_image:]]
            y = y[:, ix[-patches_per_image:]]

        else:#若为false，则直接随机生成坐标
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")
        
        coords = torch.stack([x, y], dim=-1).float() #patch的坐标（dim=-1 表示在最后一个维度上堆叠，使得每个坐标对 (x, y) 成为二维坐标。）
        # 获取对应patch坐标的特征图
        # gmap为 patch的matching feature
        # imap为 patch的context feature
        # fmap为 image/frame的的matching feature
        # view的作用是将张量重塑为指定形状
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)#注意形状为384*1*1（相当于只关注中心点的context feature）
        gmap = altcorr.patchify(fmap[0], coords, 1).view(b, -1, 128, 3, 3)#注意形状为128*3*3（p=3）

        #如果需要返回颜色信息，则从图像中提取对应的颜色特征块 clr（直接从图像提取，而前面的是从特征图中提取）。
        if return_color: 
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        # 如果未提供视差图，则创建一个全为1的视差图。
        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        # 坐标网格生成（用来表示每个像素在图像序列中的空间和时间位置。）
        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        # 从网格中提取特征块 patches。
        patches = altcorr.patchify(grid[0], coords, 1).view(b, -1, 3, 3, 3)

        # 生成特征块的索引 index。（这个索引可以不用管，因为返回了也不用~~~）
        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        # 获取信息：image对应的fmap（matching feature）、patch特征图gmap（matching feature）、patch对应的imap （context feature）和图像块（patches），同时获取颜色信息（clr）
        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index

# 计算图像块（patch特征图gmap）与特征图fmap之间的相关性
class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels #[1,4]

        self.gmap = gmap #patch特征图
        self.pyramid = pyramidify(fmap, lvls=levels) #将特征图转换为金字塔形式

    def __call__(self, ii, jj, coords):
        corrs = []
        # 遍历每个金字塔层次 levels（默认应该是1~4）。每层再调用 altcorr.corr 函数计算图像块之间的相关性。
        for i in range(len(self.levels)):
            # 将每层次的相关性结果存入 corrs 列表。
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1) #将所有层次的相关性结果堆叠并调整形状返回。

# 定义的DPVO网络
class VONet(nn.Module):#一个继承自nn.Module的类，表示一个神经网络模型。
    # 初始化（构造）函数
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__() #继承父类的初始化函数
        self.P = 3  #patch size为3
        self.patchify = Patchifier(self.P)#创建一个Patchify block，进行特征提取
        # Patchifier返回的包括：特征图fmap，patch特征图gmap，patch内部特征图imap，图像块patches，索引index（以及颜色信息clr)
        self.update = Update(self.P)#创建一个Update block，进行更新操作

        self.DIM = DIM  #输出的特征维度为384
        self.RES = 4


    @autocast('cuda', enabled=False) #用于控制自动混合精度（使用 GPU 进行训练时，AMP 可以帮助加速训练并减少显存使用量）
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5 #将图像数据归一化到 [-0.5, 1.5] 范围内。
        intrinsics = intrinsics / 4.0 #将内参数据除以4.0，缩放到四分之一大小。
        disps = disps[:, :, 1::4, 1::4].float() #将视差图的高度和宽度缩小四倍。，视差图记录深度值

        # Patchifier返回的包括：特征图fmap，patch特征图gmap，patch内部特征图imap，图像块patches，patch的索引index
        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)
        # image对应的fmap（matching feature）、patch特征图gmap（matching feature）、patch对应的imap （context feature）

        # 通过 CorrBlock 类计算图像块之间的相关性。具体地是计算 patch 特征图 gmap 和特征图 fmap 之间的相关性。
        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape #获取特征图的形状，分别是批次大小 b、图像数量 n、通道数 c、高度 h 和宽度 w。
        p = self.P #patch size

        patches_gt = patches.clone() #克隆patches，patches_gt为patches的克隆
        Ps = poses #相机位姿，GT pose（应该是xyz，qw，qx，qy，qz）

        d = patches[..., 2, p//2, p//2]#获取patches的第二个通道的中心像素值
        patches = set_depth(patches, torch.rand_like(d))#设置patches的深度值，随机初始化

        # 生成一维的网格索引
        # torch.where(ix < 8)[0]：这个操作会返回满足条件 ix < 8 的索引。torch.arange(0, 8, device="cuda")：生成从 0 到 7 的张量，并放置在 GPU 上
        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
        ii = ix[kk]#kk应该是指当前patch的索引

        imap = imap.view(b, -1, DIM) #将patch的context feature imap 的形状重塑为 (b, -1, DIM
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        # net的size为b, len(kk), DIM。kk是patch的索引（所有的关键帧），DIM是特征维度。
        
        Gs = SE3.IdentityLike(poses)#将其转换为 SE3 对象，然后调用 IdentityLike 函数，生成一个与 poses 相同形状的单位矩阵。

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            # 分离梯度，避免梯度累积
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            
            # update的时候会返回delta可用于后续的计算(进行RNN的update操作)
            # corr应该是correlation matching feature？
            # imap应该是patch的context feature
            # 那么net应该就是hidden state？
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4 #针对update operator的阻尼系数
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                # 通过BA优化来计算pose，Gs（同时更新patch graph）
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            # 用估算的pose进行变换
            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            # 用真值的pose进行变换
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))#GS为estimate的结果，PS才是真值

        return traj

