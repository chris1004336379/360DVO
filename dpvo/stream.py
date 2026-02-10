import os
import cv2

def image_stream(queue, imagedir, stride, skip=0):
    """ image generator """
    image_list = sorted(os.listdir(imagedir))[skip::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        queue.put((t, image))
    queue.put((-1, image))

def video_stream(queue, imagedir, stride, skip=0):
    """ video generator """
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

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        queue.put((t, image))

        t += 1

    queue.put((-1, image))
    cap.release()

