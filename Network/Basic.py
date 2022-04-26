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
import sys
# Don't generate pyc codes
sys.dont_write_bytecode = True

def basicNetwork(img_shape):
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
    basicModel = layers.Conv2D(padding='same', filters=8, kernel_size=3, input_shape=img_shape, activation='relu')(inputs)
    #l2
    basicModel = layers.MaxPooling2D(pool_size=(2,2), padding='same')(basicModel)
    #l3
    basicModel = layers.Conv2D(padding='same', filters=16, kernel_size=3, activation='relu')(basicModel)
    #l4
    basicModel = layers.MaxPooling2D(pool_size=(2,2), padding='same')(basicModel)
    #l5
    basicModel = layers.Flatten()(basicModel)
    #l6
    prSoftmax = layers.Dense(units=10, activation='softmax')(basicModel)
    basicModel = models.Model(inputs=inputs, outputs = prSoftmax, name="BasicModel")
    return basicModel
