from keras.layers import Input, MaxPooling2D
import tensorflow as tf
from keras.layers.convolutional import Convolution2D, UpSampling2D, Conv2D
from keras.layers.core import Activation, Reshape
from keras.models import Model
from tensorflow.keras.layers import BatchNormalization
import tensorflow_addons as tfa


def build_segnet(input_shape, n_labels=1, kernel=3, pool_size=(2, 2), output_mode="softmax"):
    # encoder
    inputs = Input(input_shape)

    conv_1 = Conv2D(64, kernel, padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Conv2D(64, kernel, padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = tf.nn.max_pool_with_argmax(conv_2, 3, 2, padding="SAME")

    conv_3 = Conv2D(128, kernel, padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Conv2D(128, kernel, padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = tf.nn.max_pool_with_argmax(conv_4, 3, 2, padding="SAME")

    conv_5 = Conv2D(256, kernel, padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Conv2D(256, kernel, padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Conv2D(256, kernel, padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = tf.nn.max_pool_with_argmax(conv_7, 3, 2, padding="SAME")

    conv_8 = Conv2D(512, kernel, padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Conv2D(512, kernel, padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Conv2D(512, kernel, padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = tf.nn.max_pool_with_argmax(conv_10, 3, 2, padding="SAME")

    conv_11 = Conv2D(512, kernel, padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Conv2D(512, kernel, padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Conv2D(512, kernel, padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = tf.nn.max_pool_with_argmax(conv_13, 3, 2, padding="SAME")
    print("Build encoder done..")

    # decoder
    unpool_1 = tfa.layers.MaxUnpooling2D(pool_size)(pool_5, mask_5)

    conv_14 = Conv2D(512, kernel, padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Conv2D(512, kernel, padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Conv2D(512, kernel, padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = tfa.layers.MaxUnpooling2D(pool_size)(conv_16, mask_4)

    conv_17 = Conv2D(512, kernel, padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Conv2D(512, kernel, padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Conv2D(256, kernel, padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = tfa.layers.MaxUnpooling2D(pool_size)(conv_19, mask_3)

    conv_20 = Conv2D(256, kernel, padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Conv2D(256, kernel, padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Conv2D(128, kernel, padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = tfa.layers.MaxUnpooling2D(pool_size)(conv_22, mask_2)

    conv_23 = Conv2D(128, kernel, padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Conv2D(64, kernel, padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = tfa.layers.MaxUnpooling2D(pool_size)(conv_24, mask_1)

    conv_25 = Conv2D(64, kernel, padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Conv2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape(
        (input_shape[0], input_shape[1], n_labels)
    )(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    return model



if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_segnet(input_shape)
    model.summary()
