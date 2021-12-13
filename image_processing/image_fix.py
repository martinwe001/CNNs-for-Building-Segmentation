import cv2
import os

# read the image file

#img = cv2.imread('building-segmentation/masks/train_labels/22678915_15.png', 2)

def make_binary_jpg(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        ret, bw_img = cv2.threshold(img, 254, 1, cv2.THRESH_BINARY)

        cv2.imwrite('building-segmentation/masks/train/' + f"{filename.split('.')[0]}.jpg", bw_img)
    return

make_binary_jpg('building-segmentation/masks/train_labels/')


