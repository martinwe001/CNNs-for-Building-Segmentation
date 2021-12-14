import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow_addons as tfa


if __name__ == "__main__":
    """ Load the test images """
    test_images = glob("building-segmentation/test/test_64/*")

    """ Load the model """

    model = 'unet'
    epochs = 34
    res = 64

    model = tf.keras.models.load_model(f"{model}_models/{model}_{epochs}_epochs_{res}.h5", custom_objects={'MaxUnpooling2D': tfa.layers.MaxUnpooling2D})

    for path in tqdm(test_images, total=len(test_images)):
        x = cv2.imread(path, cv2.IMREAD_COLOR)
        original_image = x
        h, w, _ = x.shape

        x = cv2.resize(x, (256, 256))
        #x = x/255.0
        x = x.astype(np.float32)

        x = np.expand_dims(x, axis=0)
        pred_mask = model.predict(x)[0]

        pred_mask = np.concatenate(
            [
                pred_mask,
                pred_mask,
                pred_mask
            ], axis=2)
        pred_mask = (pred_mask > 0.5) * 255
        pred_mask = pred_mask.astype(np.float32)
        pred_mask = cv2.resize(pred_mask, (w, h))

        original_image = original_image.astype(np.float32)

        alpha_image = 0.8
        alpha_mask = 1
        cv2.addWeighted(pred_mask, alpha_mask, original_image, alpha_image, 0, original_image)

        name = path.split("/")[-1]
        cv2.imwrite(f"save_images/{name}", original_image)
