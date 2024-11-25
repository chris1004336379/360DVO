import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3
from .visualize import Viser
from .net import VONet
from .utils import *
from . import projective_ops as pops

autocast = torch.amp.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:
    def __init__(self, cfg, network, ht=480, wd=640, viz=False):
        self.cfg = cfg
        self.load_weights(network)#从网络中加载权重
        self.is_initialized = False
        self.enable_timing = False
        
        self.n = 0      # number of frames（第几帧）
        self.m = 0      # number of patches
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE # buffer size

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = [] #一个列表，用于存储时间戳
        self.counter = 0

        # dummy image for visualization
        self.image_ = torch.zeros(self.ht, self.wd, 3, dtype=torch.uint8, device="cpu")

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda") #内参

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)#imap 为patch的context feature（384*1*1,只关注中心点的context feature）
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)#gmap为patch的matching feature（128*p*p）

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs) #这个应该是隐藏层的特征
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

        self.viser = None
        if viz:#如果可视化为True，则启动查看器。
            self.viser = Viser()

    # 加载权重（self:是该方法所属类的实例。）
    def load_weights(self, network):
        # load network from checkpoint file（如果 network 是一个字符串（即路径），则从文件中加载网络权重。）
        if isinstance(network, str):#  检查是否为字符串类型
            from collections import OrderedDict
            state_dict = torch.load(network) #采用torch.load函数加载权重文件
            # 创建一个新的有序字典 （保证按输入的顺序）new_state_dict。
            new_state_dict = OrderedDict()
            # 遍历读取的state_dict键值对，如果键中不包含“update.lmbda”，则将其添加到new_state_dict中。
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            # 创建新的VO网络
            self.network = VONet()
            # 加载新的网络权重
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes（复制网络属性）
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        # 配置网络
        self.network.cuda() #将网络移动到 GPU 上。
        self.network.eval() #将网络设置为评估模式（即禁用 dropout 和 batch normalization 的训练行为）。

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()

    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast('cuda', enabled=self.cfg.MIXED_PRECISION):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        # 使用了一个计时器，记录了代码块中的执行时间，用于性能分析。
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject() #重投影

            with autocast('cuda', enabled=True):#自动混合精度计算
                corr = self.corr(coords) #计算相关性，获取当前帧与上一帧之间的特征匹配信息。
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")#给定值，不像droid那样需要计算
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        # 进行BA优化
        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            try:
                fastba.BA(self.poses, self.patches, self.intrinsics, 
                    target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
            except:
                print("Warning BA failed...")
            
            # 更新点云
            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME #patch的生命周期
        t0 = self.M * max((self.n - r), 0) #计算前向边的起始时间点，等于当前帧索引减去生命周期的帧数，然后乘以一个步长 M。
        t1 = self.M * max((self.n - 1), 0) #计算前向边的结束时间点，等于当前帧索引减去1的帧数，然后乘以步长 M。
        # flatmeshgrid生成一个网格，用于表示帧之间的边。
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"), #生成从 t0 到 t1 之间的整数序列，表示前向边的起点。
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij') #生成从 self.n - 1 到 self.n 之间的整数序列，表示前向边的终点。

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')
    
    #开始迭代处理数据 （输入时间戳、图像、内参）
    def __call__(self, tstamp, image, intrinsics):
        """ track new frame """

        # 如果可视化为True，则更新查看器的图像
        if self.viser is not None:
            self.viser.update(self.n, image, self.poses, self.points_, self.colors_)

        # 归一化图像，将像素值从 [0, 255] 映射到 [-0.5, 1.5] 之间。
        image = 2 * (image[None,None] / 255.0) - 0.5
        
        # 使用自动混合精度（autocast）加速计算。
        # 用patchify进行特征提取，调用Patchifier的forward函数
        # 获取信息：从图像中提取特征图（fmap）、全局特征图（gmap）、内部特征图（imap）和图像块（patches），同时获取颜色信息（clr）
        with autocast('cuda', enabled=self.cfg.MIXED_PRECISION):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME, ##每一帧的patch数量
                    gradient_bias=self.cfg.GRADIENT_BIAS, #是否考虑梯度的bias
                    return_color=True)#进行特征的提取

        ### update state attributes （状态属性更新） ###
        self.tlist.append(tstamp) #更新时间戳列表 
        self.tstamps_[self.n] = self.counter
        self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization（可视化颜色信息）
        # 调整颜色信息并存储在 self.colors_ 中。
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        # 索引更新
        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M #地图索引

        # 运动模型和姿态估计
        if self.n > 1:
            # 如果运动模型为 'DAMPED_LINEAR'，则使用阻尼线性模型计算新姿态。
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                # 转换姿态到 SE3 对象
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])
                
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data #通过指数映射回到李群（刚体变换）
                self.poses_[self.n] = tvec_qvec
            else:#否则的话，将姿态设置为上一帧的姿态。
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # TODO better depth initialization（深度的初始化）
        patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None]) #初始化深度信息。
        if self.is_initialized:#如果已初始化，则使用过去几帧的深度中值进行深度初始化。
            s = torch.median(self.patches_[self.n-3:self.n,:,2])
            patches[:,:,2] = s

        self.patches_[self.n] = patches #更新patches

        ### update network attributes 更新网络属性 ###
        self.imap_[self.n % self.mem] = imap.squeeze()#去掉一维数据
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        # 计数器和初始化检查
        self.counter += 1 #计数器加1       
        # 检查是否初始化，如果未初始化且运动探测值小于 2.0，则更新 self.delta 并返回。
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        # 帧计数和关键帧处理
        self.n += 1
        self.m += self.M #总的patch的数量

        # relative pose（添加前向和后向因子）
        # 这两个方法的主要功能是为当前帧计算前向和后向的边。
        # 这些边表示帧之间的关系，用于视觉里程计中的图优化或者帧之间的关联性计算。
        # 从而更好地估计和优化相机的运动轨迹。
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        # 如果帧数达到 8 并且未初始化，则进行初始化。
        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            for itr in range(12):
                self.update()
        # 如果已初始化，则更新并处理关键帧。
        elif self.is_initialized:
            self.update()
            self.keyframe()

            





