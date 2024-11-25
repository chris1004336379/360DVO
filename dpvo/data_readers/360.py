
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

class Dataset360(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(Dataset360, self).__init__(name='Dataset360', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building Dataset360 dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*/*/*/*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'images/*.png')))
            
            poses = np.loadtxt(osp.join(scene, 'gt.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= Dataset360.DEPTH_SCALE
            intrinsics = [Dataset360.calib_read(images[0].shape)] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'poses': poses,
                            'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read(shape):
        H = shape[0]
        W = shape[1]
        return np.array([W/(2*np.pi), -H/np.pi, W/2, H/2])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)


