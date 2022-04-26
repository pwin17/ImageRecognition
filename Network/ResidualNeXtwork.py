"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""
from tensorflow.keras import layers, models
import tensorflow.nn as nn
import tensorflow as tf
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

class GroupConv2D(layers.Layer):
    def __init__(self, input_channels, output_channels, kernel_size, strides=(1, 1), padding='same',
                 groups=1, use_bias=False, kernel_initializer='he_uniform'):
        super(GroupConv2D, self).__init__()

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for _ in range(self.groups):
            self.conv_list.append(layers.Conv2D(filters=self.group_out_num, kernel_size=kernel_size,
                                                         strides=strides, padding=padding, use_bias=use_bias))

    def call(self, inputs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out

class ResNeXt_BottleNeck(layers.Layer):
    def __init__(self, filters, strides, groups):
        super(ResNeXt_BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.group_conv = GroupConv2D(input_channels=filters, output_channels=filters, kernel_size=(3, 3), strides=strides, groups=groups)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=2 * filters, kernel_size=(1, 1), strides=1, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.shortcut_conv = layers.Conv2D(filters=2 * filters, kernel_size=(1, 1), strides=strides, padding="same")
        self.shortcut_bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = nn.relu(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = nn.relu(layers.add([x, shortcut]))
        return output

def residualBlock(filters, strides, groups, repeat_num):
    block = tf.keras.Sequential()
    block.add(ResNeXt_BottleNeck(filters=filters,
                                 strides=strides,
                                 groups=groups))
    for _ in range(1, repeat_num):
        block.add(ResNeXt_BottleNeck(filters=filters,
                                     strides=1,
                                     groups=groups))

    return block

class ResNeXt(models.Model):
    def __init__(self, repeat_num_list, cardinality):
        if len(repeat_num_list) != 4:
            raise ValueError("The length of repeat_num_list must be four.")
        super(ResNeXt, self).__init__()
        self.conv1 = layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")
        self.block1 = residualBlock(filters=64,
                                          strides=1,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[0])
        self.block2 = residualBlock(filters=128,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[1])
        self.block3 = residualBlock(filters=256,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[2])
        self.pool2 = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(units=10,
                                        activation=nn.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)

        return x
