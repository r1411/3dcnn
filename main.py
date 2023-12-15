import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from scipy import ndimage


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))
    
    x = keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", dilation_rate=(3,3,3))(inputs)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu", dilation_rate=(2,2,2))(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu", dilation_rate=(2,2,2))(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(units=512, activation="relu")(x)
    x = keras.layers.Dense(units=512, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model
	
	


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    with open('saved_scans.pickle', 'rb') as file:
        loaded_state = pickle.load(file)

    abnormal_scans = loaded_state['abnormal_scans']
    normal_scans = loaded_state['normal_scans']

    abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
    normal_labels = np.array([0 for _ in range(len(normal_scans))])

    print(f'Abnorm length: {len(abnormal_scans)}')
    print(f'Norm length: {len(normal_scans)}')

    # Build model.
    model = get_model(width=128, height=128, depth=64)

    # Split data in the ratio 70-30 for training and validation.
    x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
    y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
    x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
    y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
    print(
        "Number of samples in train and validation are %d and %d."
        % (x_train.shape[0], x_val.shape[0])
    )

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    print('Start augmentation')
    batch_size = 2
    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    print('Validation preprocessing')
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
    )
    # Обучение -------------------------------------------------------------------------------------------------
    print('Start train')
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    # Define callbacks.
    # В этот файл будет записана ЛУЧШАЯ модель всх эпох и народов
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_image_classification.h5", save_best_only=True, monitor="val_acc"
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=100) # Количество эпох без изменения, чтобы остановить обучение раньше

    # Train the model, doing validation at the end of each epoch
    epochs = 100
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    model.save('base_model', save_format='tf')


if __name__ == '__main__':
    main()