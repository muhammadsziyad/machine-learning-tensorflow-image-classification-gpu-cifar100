# data/cifar100_data.py

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def load_data():
    # Load CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # Normalize the data to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    return (x_train, y_train), (x_test, y_test)