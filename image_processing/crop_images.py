import cv2
import os
from check_classification import check_for_any_white

folder_path = 'building-segmentation/masks/cropped_masks_train_128/'

interval = 64
stride = 64

# os.remove("demofile.txt")

def crop_images(folder_path):
    for filename in os.listdir(folder_path):
        count = 0
        img = cv2.imread(os.path.join(folder_path, filename))
        for i in range(0, img.shape[0], interval):
            for j in range(0, img.shape[1], interval):
                cropped_img = img[j:j + stride, i:i + stride]
                count += 1
                name = filename.split('.')[0]
                cv2.imwrite(f'building-segmentation/images/cropped_images_test_32/{name}_crop_' + str(count) + '.jpg', cropped_img)
    cv2.waitKey()

crop_images('building-segmentation/images/images_64/')

def number_of_pixel_value_check(folder_p):

    for filename in os.listdir(folder_p):
        pixel_value = check_for_any_white(folder_p, filename)
        if pixel_value < 200:
            os.remove(f'building-segmentation/masks/cropped_masks_test_64/{filename}')
            os.remove(f'building-segmentation/images/cropped_images_test_64/{filename}')
    return

#number_of_pixel_value_check('building-segmentation/masks/cropped_masks_train_128/')