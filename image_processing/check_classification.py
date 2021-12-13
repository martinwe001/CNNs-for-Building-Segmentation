import cv2
import os

# read the image file


def check_for_any_white(folder, filename):
    counter = 0
    img = cv2.imread(os.path.join(folder, filename))
    dim = img.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            if img[i,j][0] != 0:
                counter += 1

    return counter
