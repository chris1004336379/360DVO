
import numpy as np
import glob
import cv2
import os.path as osp

from .base import RGBDDataset

# cur_path = osp.dirname(osp.abspath(__file__))
# test_split = osp.join(cur_path, 'tartan_test.txt')
# test_split = open(test_split).read().split()


test_split = [
]

class TartanAir(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(TartanAir, self).__init__(name='TartanAir', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building TartanAir dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/*/*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'image_left/*.png')))
            depths = sorted(glob.glob(osp.join(scene, 'depth_left/*.npy')))

            if len(images) != len(depths):
                continue
            
            poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read():
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth


