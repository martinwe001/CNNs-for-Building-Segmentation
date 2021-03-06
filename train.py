import os
import numpy as np
import tensorflow as tf
from unet_model import build_unet
from segnet_model import build_segnet
from new_model import build_simpler_model, build_even_simpler_model
from data import load_dataset, tf_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

model_types = ['segnet-master', 'unet-master', 'simpler', 'even-simpler']

if __name__ == "__main__":
    """ Hyperparamaters """
    dataset_path = "building-segmentation"
    input_shape = (64, 64, 3)
    batch_size = 20
    model = 3
    epochs = 5
    res = 64
    lr = 1e-3
    model_path = f"{model_types[model]}_models/{model_types[model]}_{epochs}_epochs_{res}_batch20.h5"
    csv_path = f"csv/data_{model_types[model]}_{epochs}_{res}_batch20.csv"

    """ Load the dataset """
    (train_images, train_masks), (val_images, val_masks) = load_dataset(dataset_path)
    print(f"Train: {len(train_images)} - {len(train_masks)}")
    print(f"Validation: {len(val_images)} - {len(val_masks)}")
    print(f'Now training {model_types[model]} model with batch size: {batch_size} and for {epochs} epochs')

    train_dataset = tf_dataset(train_images, train_masks, batch=batch_size)
    val_dataset = tf_dataset(val_images, val_masks, batch=batch_size)

    """ Model """
    if model == 0:
        model = build_segnet(input_shape)
    if model == 1:
        model = build_unet(input_shape)
    if model == 2:
        model = build_simpler_model(input_shape)
    if model == 3:
        model = build_even_simpler_model(input_shape)

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[
            tf.keras.metrics.MeanIoU(num_classes=2),
            tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0]),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision()
        ]
    )


    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=10, factor=0.1, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor="val_loss", patience=10)
    ]

    train_steps = len(train_images)//batch_size
    if len(train_images) % batch_size != 0:
        train_steps += 1

    test_steps = len(val_images)//batch_size
    if len(val_images) % batch_size != 0:
        test_steps += 1

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=test_steps,
        callbacks=callbacks
    )
