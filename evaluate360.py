import os
import torch
from multiprocessing import Process, Queue #导入多进程模块，进行多线程处理
from pathlib import Path
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

# 导入自定义模块（下面是dpvo自己定义的模块）
from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg #应该就是配置文件对应的对象
from dpvo.stream import image_stream, video_stream


@torch.no_grad() #表示该函数不会计算梯度（由于是导入网络权重的）
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):

    slam = None #初始化slam为空
    queue = Queue(maxsize=8) #创建一个队列，队列的最大长度为8，应该是用于多线程操作？

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start() #启动进程，启动了一个进程并从队列中获取数据进行处理

    while 1:
        # 从队列中获取数据
        (t, image, intrinsics) = queue.get()
        if t < 0: break #如果t小于0，跳出循环（最后一个数据输入了t=-1，代表着结束）

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        # 将内参数据从 NumPy 数组转换为 PyTorch 张量。
        intrinsics = torch.from_numpy(intrinsics).cuda()

        # 初始化SLAM对象（DPVO），并传入配置参数cfg、网络权重模型network、图像的高度和宽度、是否可视化。等参数
        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics) #调用DPVO对象的__call__方法，传入时间戳、图像数据和内参数据

    #更新12次？？？然后等待子进程结束 
    for _ in range(12):
        slam.update()

    reader.join() #等待子进程结束
    print('finished!!!')

    points = slam.points_.cpu().numpy()[:slam.m]
    colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))

# 主函数
if __name__ == '__main__':
    import argparse #导入argparse模块，该模块用于解析命令行参数。
    # 创建一个ArgumentParser对象，parser。这个对象包含将要添加的所有命令行参数及其处理方式。
    parser = argparse.ArgumentParser()

    # 添加命令行参数以及其默认值
    parser.add_argument('--network', type=str, default='dpvo.pth') # 预训练的网络模型
    parser.add_argument('--calib', type=str, default="calib/360.txt") #相机内参，校准参数
    parser.add_argument('--stride', type=int, default=1) #取数据的步长
    parser.add_argument('--skip', type=int, default=0) #取数据的时候跳过的帧数（跳过前面几帧）
    parser.add_argument('--config', default="config/default.yaml") #VO配置文件
    parser.add_argument('--timeit', action='store_true') #如果命令行中包含这个参数，为True，否则为False。
    parser.add_argument('--viz', action="store_true") #如果命令行中包含这个参数，为True，否则为False。
    parser.add_argument('--trials', type=int, default=5)
    
    # 解析命令行参数并将结果赋值给args。
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    # name_list=['seq0','seq1','seq2','seq3','seq4','seq5','seq6','seq7','seq8','seq9']
    name_list=['seq2','seq3','seq4','seq5','seq6','seq7','seq8','seq9']
    for seqname in name_list:
        print("Runnning " + seqname)
        imagedir = 'datasets/Dataset360/' + seqname + '/images/'
        for i in range (args.trials):
            args.name = seqname + '_' + str(i)
            print('Trial ' + str(i))
            (poses, tstamps), (points, colors, calib) = run(cfg, args.network, imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit)
            trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

            Path("saved_trajectories").mkdir(exist_ok=True)
            file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)

        

