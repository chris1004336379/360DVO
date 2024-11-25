import torch.nn as nn

# 创建一个ResidualBlock类，继承自nn.Module
class ResidualBlock(nn.Module):
    #self.in_planes应该是32，dim（planes）也是32
    # 输入参数：in_planes（输入通道数）、planes（输出通道数）、norm_fn（归一化层的类型，默认为'group'）、stride（步幅，默认为1）
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        # conv1 和 conv2：两个3x3的卷积层+ReLU激活函数
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8 #用于 GroupNorm 的 num_groups 参数

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:#默认为1，不进行下采样
            self.downsample = None
        # 如果步幅不为1，定义 downsample 为一个 nn.Sequential，包含一个1x1的卷积层和前面选择的归一化层 norm3。这一步骤用于调整输入的尺寸和通道数，以便与主路径上的输出相加。
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y))) #第一层：卷积+归一化+激活
        y = self.relu(self.norm2(self.conv2(y))) #第二层：卷积+归一化+激活

        if self.downsample is not None:
            x = self.downsample(x) #下采样

        return self.relu(x+y) #残差连接


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

DIM=32

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM,  stride=1)
        self.layer2 = self._make_layer(2*DIM, stride=2)
        self.layer3 = self._make_layer(4*DIM, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(4*DIM, output_dim, kernel_size=1)

        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # 用于创建由两个残差块（Residual Block）组成的卷积层组
    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)

# 卷积网络的编码器。主要作用是对输入图像进行特征提取，经过多个卷积层和归一化层的处理，最后输出一个指定维度的特征图。
class BasicEncoder4(nn.Module): #继承自 nn.Module
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder4, self).__init__()#继承父类的初始化函数
        self.norm_fn = norm_fn #归一化层的类型
        self.multidim = multidim #是否使用多维度，定义为false

        # 根据 norm_fn 选择不同的归一化层。
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential() #使用一个空的 Sequential 容器，相当于不使用归一化。

        # 第一个卷积层及激活函数（输入通道数为 3，输出通道数为 DIM，卷积核大小为 7，步幅为 2，填充为 3。）
        # 此处DIM为32
        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True) #第一个 ReLU 激活函数，使用 inplace=True 以节省内存

        # 中间层
        self.in_planes = DIM#输入通道数
        # 通过 _make_layer 方法创建两个网络的block(实际上为residual network)。
        self.layer1 = self._make_layer(DIM,  stride=1) #layer1 的输出通道数为 DIM，步幅为 1；
        self.layer2 = self._make_layer(2*DIM, stride=2) #layer2 的输出通道数为 2*DIM，步幅为 2。

        # output convolution（输出的卷积层）
        self.conv2 = nn.Conv2d(2*DIM, output_dim, kernel_size=1)

        # 根据 dropout （默认为0.0就是不用）的值决定是否添加 Dropout 层。
        # 如果 dropout 大于 0，则添加一个 Dropout2d 层，否则 self.dropout 设置为 None。
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        # 对网络的权重进行初始化（遍历模型的所有子模块并进行权重初始化）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):#检查当前模块 m 是否是 nn.Conv2d（二维卷积层）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #如果是，就使用 Kaiming 正态初始化。
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                # 对归一化层（BatchNorm2d、InstanceNorm2d、GroupNorm），如果有权重和偏置，则将权重初始化为 1，偏置初始化为 0。
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride) #self.in_planes应该是32，dim也是32
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2) # layer1 和 layer2 组合成一个元组 layers。
        
        self.in_planes = dim
        return nn.Sequential(*layers) #使用 nn.Sequential 将 layers 中的层组合成一个顺序容器，并返回。

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape #输入形状
        x = x.view(b*n, c1, h1, w1) #调整输入形状

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2) #恢复输出形状
