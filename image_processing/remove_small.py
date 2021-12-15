import cv2
import os



def check_size(folder):

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        dim = img.shape
        if dim[0] != 64 or dim[1] != 64:
            os.remove(f'building-segmentation/test/test_64/{filename}')

    return
check_size('building-segmentation/test/test_64/')