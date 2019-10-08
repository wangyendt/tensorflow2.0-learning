#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: 1. TensorFlow2.0教程-Keras 快速入门
@time: 2019/10/8 13:47
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

print(tf.__version__)
print(tf.keras.__version__)

model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.glorot_normal))
model.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(10, activation='softmax'))



model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

train_x, train_y = np.random.random((1000, 72)), np.random.random((1000, 10))
val_x, val_y = np.random.random((200, 72)), np.random.random((200, 10))
model.fit(train_x, train_y, epochs=10, batch_size=64, validation_data=(val_x, val_y))
#
print(model.summary())
tf.keras.utils.plot_model(model,'model.png')
tf.keras.utils.plot_model(model,'model_with_shape.png',show_shapes=True)

