#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: 20191217 1. fc train
@time: 2019/12/17 13:58
"""

import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, datasets


def mnist_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


def train(epoch):
    pass


if __name__ == '__main__':
    model = keras.Sequential([
        layers.Reshape((28 * 28,), input_shape=(28, 28)),
        layers.Dense(100, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(10)
    ])
    optimizer = optimizers.Adam()
    mnist_dataset()
