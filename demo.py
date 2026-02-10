import os
import torch
import math
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.utils import Timer
from dpvo.dpvo import DPVO
from dpvo.config import cfg 
from dpvo.stream import image_stream, video_stream
from dpvo.plot_utils import plot_trajectory, save_ply
import cv2

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)


def compute_intrinsics(H, W):
    fx = W / (2 * math.pi)
    fy = - H / math.pi
    cx = W / 2
    cy = H / 2
    intrinsics = np.array([fx, fy, cx, cy])
    intrinsics = torch.from_numpy(intrinsics).cuda()
    return intrinsics

@torch.no_grad() 
def run(cfg, network, imagedir, stride=1, skip=0, viz=False, timeit=False):

    slam = None 
    queue = Queue(maxsize=8) 

    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, stride, skip))

    reader.start() 

    while 1:
        (t, image) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        C, H, W = image.shape
        intrinsics = compute_intrinsics(H, W)

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)


    for _ in range(12):
        slam.update()

    reader.join()
    print('finished!!!')

    points = slam.points_.cpu().numpy()[:slam.m]
    colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), points, colors

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--network', type=str, default='360dvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/360.yaml")
    parser.add_argument('--timeit', action='store_true') 
    parser.add_argument('--viz', action="store_true") 
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--save_ply', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--name', type=str, help='name your run', default='test')
    
    args = parser.parse_args()

    cfg.merge_from_file(args.config)

    print("Running with config...")
    print(cfg)

    (poses, tstamps), points, colors = run(cfg, args.network, args.imagedir, args.stride, args.skip, args.viz, args.timeit)
    trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

    print("Saving the result...")

    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)
    
    if args.save_ply:
        save_ply(args.name, points, colors)
    
    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory, title=f"Trajectory Prediction for {args.name}", filename=f"trajectory_plots/{args.name}.pdf")

        

