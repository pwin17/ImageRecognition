"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""
'''
different models/data to test:
2 layers + data augmentation
2 layers + data augmentation + batch normalization
2 layers + batch normalization
3 layers + data augmentation
3 layers + data augmentation + batch normalization
3 layers + batch normalization
'''
from tensorflow.keras import layers, models
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

def optimizedNetwork(img_shape):
    """
    Inputs: 
    ImageSize - Size of the Image
    Outputs:
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    ## Basic model
    inputs = layers.Input(shape=img_shape)
    # l1
    optimizedModel = layers.Conv2D(padding='same', filters=32, kernel_size=3, input_shape=img_shape, activation='relu')(inputs)
    optimizedModel - layers.BatchNormalization()(optimizedModel)
    optimizedModel = layers.MaxPooling2D(pool_size=(2,2), padding='same')(optimizedModel)
    #l2
    optimizedModel = layers.Conv2D(padding='same', filters=64, kernel_size=3, activation='relu')(optimizedModel)
    optimizedModel - layers.BatchNormalization()(optimizedModel)
    optimizedModel = layers.MaxPooling2D(pool_size=(2,2), padding='same')(optimizedModel)
    #l3
    optimizedModel = layers.Conv2D(padding='same', filters=64, kernel_size=3, activation='relu')(optimizedModel)
    optimizedModel - layers.BatchNormalization()(optimizedModel)
    optimizedModel = layers.MaxPooling2D(pool_size=(2,2), padding='same')(optimizedModel)
    #l4
    optimizedModel = layers.Flatten()(optimizedModel)
    optimizedModel = layers.Dense(64, activation='relu')(optimizedModel)
    prSoftmax = layers.Dense(10, activation='softmax')(optimizedModel)
    optimizedModel = models.Model(inputs=inputs, outputs = prSoftmax, name="optimizedModel")
    return optimizedModel
