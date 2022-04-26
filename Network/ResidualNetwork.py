"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""
## ResNet50 ##
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True
## Aman's code to enable the GPU https://stackoverflow.com/questions/67352841/tensorflow-is-not-using-my-m1-macbook-gpu-during-training
# from tensorflow.python.compiler.mlcompute import mlcompute
# tf.compat.v1.disable_eager_execution()
# mlcompute.set_mlc_device(device_name='gpu')
# print("is_apple_mlc_enabled %s" % mlcompute.is_apple_mlc_enabled())
# print("is_tf_compiled_with_apple_mlc %s" % mlcompute.is_tf_compiled_with_apple_mlc())
# print(f"eagerly? {tf.executing_eagerly()}")
# print(tf.config.list_logical_devices())


class ResidualBlock(layers.Layer):
    def __init__(self, filter_val, stride, reg, bnEps, bnMom, reduce=False):
        super().__init__()
        self.reduce = reduce
        self.batchnorm1 = layers.BatchNormalization(epsilon=bnEps, momentum=bnMom, axis = -1)
        self.activation1 = layers.Activation(tf.nn.relu)
        self.conv1 = layers.Conv2D(int(filter_val/4), kernel_size=(1,1), use_bias=False, kernel_regularizer= regularizers.L2(l2 = reg))
        self.batchnorm2 = layers.BatchNormalization(epsilon=bnEps, momentum=bnMom, axis = -1)
        self.activation2 = layers.Activation(tf.nn.relu)
        self.conv2 = layers.Conv2D(int(filter_val/4), kernel_size=(3,3), strides=stride, padding='same', use_bias=False, kernel_regularizer= regularizers.L2(l2 = reg))
        self.batchnorm3 = layers.BatchNormalization(epsilon=bnEps, momentum=bnMom, axis = -1)
        self.activation3 = layers.Activation(tf.nn.relu)
        self.conv3 = layers.Conv2D(filter_val, kernel_size=(1,1), use_bias=False, kernel_regularizer= regularizers.L2(l2 = reg))
        self.conv_reduce = layers.Conv2D(filter_val, kernel_size=(1,1), strides=stride, use_bias=False, kernel_regularizer= regularizers.L2(l2 = reg))
        self.adding_layer = layers.Add()

    def call(self, x):
        y = x
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = self.activation3(x)
        x = self.conv3(x)
        if self.reduce:
            y =self.conv_reduce(y)
        x = self.adding_layer([x,y])
        return x

class ConvBlock(layers.Layer):
    def __init__(self, filter_val, reg, bnEps, bnMom):
        super().__init__()
        self.conv1 = layers.Conv2D(filter_val, kernel_size=(3,3), strides=2, padding='same', kernel_regularizer= regularizers.L2(l2 = reg))
        self.conv2 = layers.Conv2D(filter_val, kernel_size=(3,3), padding='same', kernel_regularizer= regularizers.L2(l2 = reg))
        self.conv3 = layers.Conv2D(filter_val, kernel_size=(1,1), strides=2, padding='same', kernel_regularizer= regularizers.L2(l2 = reg))
        self.batchnorm = layers.BatchNormalization(epsilon=bnEps, momentum=bnMom, axis = -1)
        self.activation = layers.Activation(tf.nn.relu)
        self.adding_layer = layers.Add()
    def call(self, x):
        y = x
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        y = self.conv3(y)
        x = self.adding_layer([x,y])
        x = self.activation(x)
        return x


class ResNet34(models.Model):
    def __init__(self, filter_sizes):
        super().__init__()
        self.reg = 0.0001
        self.bnEps = 2e-5
        self.bnMom = 0.9
        self.batchnorm = layers.BatchNormalization(epsilon = self.bnEps, momentum = self.bnMom, axis = -1)
        self.conv1 = layers.Conv2D(filter_sizes[0], kernel_size=5, use_bias=False, padding='same', 
                kernel_regularizer= tf.keras.regularizers.L2(l2 = self.reg))
        self.batchnorm2 = layers.BatchNormalization(epsilon = self.bnEps, momentum = self.bnMom, axis = -1)
        self.activation = layers.Activation(tf.nn.relu)
        self.padding_layer = layers.ZeroPadding2D(padding=(1,1))
        self.max_pooling = layers.MaxPooling2D((3,3),strides=(2,2),padding='same')

        # inputs: filter_val, stride, reg, bnEps, bnMom, reduce=False
        self.block_layers1_1 = ResidualBlock(filter_sizes[1],(1,1), self.reg, self.bnEps, self.bnMom, reduce=True)
        self.block_layers1_2 = ResidualBlock(filter_sizes[1],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)
        self.block_layers1_3 = ResidualBlock(filter_sizes[1],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)

        self.block_layers2_1 = ResidualBlock(filter_sizes[2],(2,2), self.reg, self.bnEps, self.bnMom, reduce=True)
        self.block_layers2_2 = ResidualBlock(filter_sizes[2],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)
        self.block_layers2_3 = ResidualBlock(filter_sizes[2],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)
        self.block_layers2_4 = ResidualBlock(filter_sizes[2],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)

        self.block_layers3_1 = ResidualBlock(filter_sizes[3],(2,2), self.reg, self.bnEps, self.bnMom, reduce=True)
        self.block_layers3_2 = ResidualBlock(filter_sizes[3],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)
        self.block_layers3_3 = ResidualBlock(filter_sizes[3],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)
        self.block_layers3_4 = ResidualBlock(filter_sizes[3],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)
        self.block_layers3_5 = ResidualBlock(filter_sizes[3],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)
        self.block_layers3_6 = ResidualBlock(filter_sizes[3],(1,1), self.reg, self.bnEps, self.bnMom, reduce=False)

        self.batchnorm3 = layers.BatchNormalization(epsilon = self.bnEps, momentum = self.bnMom, axis = -1)
        self.activation2 = layers.Activation(tf.nn.relu)
        self.avg_pooling = layers.AveragePooling2D((3,3))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(10, kernel_regularizer= tf.keras.regularizers.L2(l2 = self.reg))
        self.activation3 = layers.Activation(tf.nn.softmax)

    def call(self, x):
        x = self.batchnorm(x)
        x = self.conv1(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.padding_layer(x)
        x = self.max_pooling(x)

        x = self.block_layers1_1(x)
        x = self.block_layers1_2(x)
        x = self.block_layers1_3(x)

        x = self.block_layers2_1(x)
        x = self.block_layers2_2(x)
        x = self.block_layers2_3(x)
        x = self.block_layers2_4(x)

        x = self.block_layers3_1(x)
        x = self.block_layers3_2(x)
        x = self.block_layers3_3(x)
        x = self.block_layers3_4(x)
        x = self.block_layers3_5(x)
        x = self.block_layers3_6(x)

        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.avg_pooling(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation3(x)
        return x
