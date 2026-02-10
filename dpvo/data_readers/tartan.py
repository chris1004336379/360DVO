import numpy as np
import glob
import cv2
import os.path as osp

# from ..lietorch import SE3
from .base import RGBDDataset

test_split = ['House/Data_easy/P000',
            'House/Data_hard/P000',
            'House/Data_hard/P001',
            'VictorianStreet/Data_easy/P000',
            'VictorianStreet/Data_hard/P000',
            'VictorianStreet/Data_hard/P001',
            'OldTownNight/Data_easy/P000',
            'OldTownNight/Data_hard/P000',
            'OldTownNight/Data_hard/P001'

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
        scenes = glob.glob(osp.join(self.root, '*/*/*'))
        for scene in tqdm(sorted(scenes)):
            images = sorted(glob.glob(osp.join(scene, 'image_lcam_equirect/*.png')))
            depths = sorted(glob.glob(osp.join(scene, 'depth_lcam_equirect/*.png')))

            if len(images) != len(depths):
                continue
            
            poses = np.loadtxt(osp.join(scene, 'pose_lcam_left.txt'), delimiter=' ')
            poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
            poses[:,:3] /= TartanAir.DEPTH_SCALE
            intrinsics = [TartanAir.calib_read()] * len(images)

            # graph of co-visible frames based on flow
            # graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            # scene_info[scene] = {'images': images, 'depths': depths, 
            #     'poses': poses, 'intrinsics': intrinsics, 'graph': graph}
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics}
        return scene_info

    @staticmethod
    def calib_read():
        return np.array([325.95, -325.95, 1024.0, 512.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)
    
    @staticmethod
    def depth_read(depthpath):
        depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
        depth = depth_rgba.view("<f4")
        depth = np.squeeze(depth, axis=-1) / TartanAir.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth

