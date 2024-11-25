import os
import cv2
import numpy as np

# 读取图片，对图像数据进行处理，全部放到queue中
def image_stream(queue, imagedir, calib, stride, skip=0):
    """ image generator """

    # 从txt文件中读入相机内参
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4] #相机内参，前面四个参数

    # 创建相机内参矩阵
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    # 返回值是一个包含目录中所有文件和子目录名称的列表。（sorted() 函数对传入的列表进行排序。排序依据是文件名的字母顺序或数字顺序）
    image_list = sorted(os.listdir(imagedir))[skip::stride]
    # skip：从第 skip 个元素开始（索引从 0 开始）。
    # stride：步长，即每隔多少个元素提取一个。

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))#读取图片
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:]) #对图像进行去畸变处理，用标定文件4及以后的几个参数

        if 0:#那就肯定不执行
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]#用于调整图像的尺寸，以确保图像的高度和宽度都能被16整除

        queue.put((t, image, intrinsics)) #t为当前索引？

    queue.put((-1, image, intrinsics))

# 处理视频数据
def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

