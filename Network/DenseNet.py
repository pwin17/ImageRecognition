from tensorflow.keras import layers, models
import tensorflow as tf
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

class DenseBlock(layers.Layer):
    def __init__(self, num_blocks, growth_rate):
        super().__init__()
        self.layers_concat = []
        for _ in range(num_blocks):
            self.layers_concat.append(SubDenseBlock(growth_rate))
    def call(self, x):
        for layer in self.layers_concat.layers:
            x = layer(x)
        return x

class SubDenseBlock(layers.Layer):
    def __init__(self, growth_rate):
        super().__init__()
        self.sublayers_concat = [
            ## Bottle Neck Block
            layers.Conv2D(4*growth_rate, kernel_size=1, strides=1, padding='same', kernel_regularizer= tf.keras.regularizers.L2(l2 = 0.001)),
            layers.Activation(tf.nn.relu),
            layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            ## Conv Block
            layers.Conv2D(growth_rate, kernel_size=3, strides=1, padding='same', kernel_regularizer= tf.keras.regularizers.L2(l2 = 0.001)),
            layers.Activation(tf.nn.relu),
            layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        ]
    def call(self, x):
        y = x
        for layer in self.sublayers_concat.layers:
            y = layer(y)
        return layers.concatenate([x,y])


class TransitionBlock(layers.Layer):
    def __init__(self, channels, compression_factor, dropout_rate=0.2):
        super().__init__()
        self.batch_norm = layers.BatchNormalization(momentum=0.99, epsilon=0.001)
        self.conv = layers.Conv2D(int(channels * compression_factor), kernel_size=1, strides=1, padding="same", kernel_regularizer= tf.keras.regularizers.L2(l2 = 0.001))
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.Activation(tf.nn.relu)
        self.pooling = layers.AveragePooling2D((2,2), strides=2)
    def call(self, x):
        y = self.batch_norm(self.activation(self.dropout(self.conv(x))))
        y = self.pooling(y)
        return y

class ClassificationLayer(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.avgpool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.dense2 = layers.Dense(256, activation=tf.nn.relu)
        self.dense3 = layers.Dense(10, activation=tf.nn.softmax)
    def call(self, x):
        y = self.avgpool(x)
        y = self.flatten(y)
        y= self.dense2(y)
        y = self.dense3(y)
        return y

class DenseNet(models.Model):
    def __init__(self, growth_rate, block_sizes, compression_factor):
        super().__init__()
        self.block_sizes = block_sizes
        self.conv_layer1 = layers.Conv2D(2*growth_rate, kernel_size=7, strides=(2,2), padding='same',activation='relu')
        self.pooling1 = layers.MaxPooling2D((2,2),strides=(2,2))
        
        self.channels = 2*growth_rate
        
        self.denseblock1_1 = SubDenseBlock(growth_rate)
        self.denseblock1_2 = SubDenseBlock(growth_rate)
        self.denseblock1_3 = SubDenseBlock(growth_rate)
        self.channels += block_sizes[0] * growth_rate
        self.transition1 = TransitionBlock(self.channels, compression_factor)
        
        self.denseblock2_1 = SubDenseBlock(growth_rate)
        self.denseblock2_2 = SubDenseBlock(growth_rate)
        # self.denseblock2_3 = SubDenseBlock(growth_rate)
        self.channels += block_sizes[1] * growth_rate
        self.transition2 = TransitionBlock(self.channels, compression_factor)

        self.denseblock3_1 = SubDenseBlock(growth_rate)
        self.denseblock3_2 = SubDenseBlock(growth_rate)
        self.denseblock3_3 = SubDenseBlock(growth_rate)
        self.channels += block_sizes[2] * growth_rate
        self.transition3 = TransitionBlock(self.channels, compression_factor)

        self.denseblock4_1 = SubDenseBlock(growth_rate)
        self.denseblock4_2 = SubDenseBlock(growth_rate)
        # self.denseblock4_3 = SubDenseBlock(growth_rate)
        self.classifcation = ClassificationLayer()

    def call(self, x):
        y = x
        y = self.conv_layer1(y)
        y = self.pooling1(y)
        y = self.denseblock1_1(y)
        y = self.denseblock1_2(y)
        y = self.denseblock1_3(y)
        y = self.transition1(y)
        y = self.denseblock2_1(y)
        y = self.denseblock2_2(y)
        # y = self.denseblock2_3(y)
        y = self.transition2(y)
        y = self.denseblock3_1(y)
        y = self.denseblock3_2(y)
        y = self.denseblock3_3(y)
        y = self.transition3(y)
        y = self.denseblock4_1(y)
        y = self.denseblock4_2(y)
        # y = self.denseblock4_3(y)
        y = self.classifcation(y)
        return y 
