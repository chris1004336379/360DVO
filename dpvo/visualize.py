import viser
import numpy as np
from .lietorch import SE3
import torch
import cv2

class Viser:
    def __init__(self):
        self.server = viser.ViserServer()
        self.pre_position = np.zeros(3)
        self.point_size = 0.03
        self.number = 0
    
    def update(self, image, poses, points_, colors_):

        img = image.cpu().numpy()
 
        img = np.transpose(img, (1,2,0))[:,:,[2,1,0]]
        
        img = cv2.resize(img, (1920, 960))
        h,w = img.shape[:2]
        new_h, new_w = h*2, w*2
        
        canvas = np.tile(np.array((255,255,255), dtype=np.uint8), (new_h, new_w, 1))
        y_end= new_h-1
        x_start = 0
        # canvas[y_start:y_start+h, x_start:x_start+w] = img
        canvas[y_end-h:y_end, x_start:x_start+w] = img
        
        self.server.scene.set_background_image(canvas, 'png')
        poses_cut = poses[self.number:].cpu().numpy()
        poses = poses_cut[np.any(np.isclose(poses_cut[:, :3], 0), axis=1) == False]
        length = poses.shape[0] 
        
        for i in range(length):
            self.number += 1
            pose = poses[i]
            pose_t = torch.from_numpy(pose)
            pose_new = SE3(pose_t).inv().data.cpu().numpy()

            position = pose_new[:3] * 10
            # xyzw = pose_new[3:]
            # wxyz = np.array([xyzw[-1]] + xyzw[:-1].tolist())

            # self.server.scene.add_frame(
            #     name=f"/frames/{self.number}",
            #     show_axes=False,
            #     wxyz=wxyz,
            #     position=position,
            # )
            # self.server.scene.add_icosphere(
            #         name=f'sphere/{self.number}',
            #         radius=0.12,
            #         color=(255,0,0),
            #         subdivisions=1,
            #         wxyz=wxyz,
            #         position=position,
            #     )
            

            line = np.array([self.pre_position, position])
            self.pre_position = position
            line = np.expand_dims(line, axis=0)
    
            self.server.scene.add_line_segments(
                f'/lines/{self.number}',
                line,
                (0,255,0),
                line_width=3,
            )
            
        points = points_.cpu().numpy() * 10
        colors = colors_.cpu().numpy().reshape(-1,3)
        
        self.server.scene.add_point_cloud(
            '/point_cloud',
            points,
            colors,
            point_size=self.point_size,
            point_shape='square',
        )
        del points
        del colors