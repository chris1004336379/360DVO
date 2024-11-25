import viser
import numpy as np

class Viser:
    def __init__(self):
        self.server = viser.ViserServer()
        self.pre_position = np.zeros(3)
        self.lines = np.zeros((1,2,3))

    def update(self, n, image, poses, points_, colors_):
        # img = image.cpu().numpy()
        # img = np.transpose(img, (1,2,0))[:,:,[2,1,0]]

        # self.server.scene.add_image(
        #     "/image",
        #     img,
        #     6.4,
        #     4.8,
        #     format="png",
        #     wxyz=(1.0, 0.0, 0.0, 0.0),
        #     position=(5.0, 5.0, 5.0),
        # )

        pose = poses[0,n].cpu().numpy()
        xyzw = pose[-4:]
        wxyz = np.array([xyzw[-1]] + xyzw[:-1].tolist())
        position = pose[:3]
        self.server.scene.add_frame(
            name=f"/frames/{n}",
            show_axes=False,
            wxyz=wxyz,
            position=position,
        )

        if (abs(position.sum()) > 1e-5):
            line = np.array([[self.pre_position, position]])
            self.pre_position = position
            self.lines = np.concatenate((self.lines, line), axis=0)

        self.server.scene.add_line_segments(
            '/lines',
            self.lines,
            (255,0,0),
            5,
        )
        points = points_.cpu().numpy()
        colors = colors_.cpu().numpy().reshape(-1,3)
        self.server.scene.add_point_cloud(
            '/point_cloud',
            points,
            colors,
            0.0001,
            'rounded',
        )
