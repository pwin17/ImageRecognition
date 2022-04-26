#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code
"""

import datetime
from tensorflow.keras import datasets, utils, optimizers, losses, callbacks
import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from Network.Basic import basicNetwork
from Network.BetterNetwork import optimizedNetwork
from Network.ResidualNetwork import ResNet34
from Network.DenseNet import DenseNet
from Network.ResidualNeXtwork import ResNeXt
import argparse

# Don't generate pyc codes
sys.dont_write_bytecode = True

def preprocessing():
    '''
    Downloads CIFAR10 dataset
    '''
    train_set, test_set = datasets.cifar10.load_data()
    train_images, test_images = train_set[0].astype(np.float32)/255.0 , test_set[0].astype(np.float32)/255.0
    train_labels = utils.to_categorical(train_set[1], num_classes=10)
    test_labels = utils.to_categorical(test_set[1], num_classes=10)
    return train_images, train_labels, test_images, test_labels

def data_augmentation(train_images, train_labels, test_images, test_labels):
    '''
    randomly pick 30% of data to perform data augmentation and add to the dataset
    Perform horizontal flip, vertical flip, random brightness to random 5% each of the train_set and test set
    '''
    train_set_random, test_set_random = np.random.randint(0, len(train_images)-1, size=int(len(train_images)*0.15)), np.random.randint(0, len(test_images)-1, size=int(len(test_images)*0.15))
    temp_train_img = np.empty([int(len(train_images)*0.15),32,32,3])
    temp_train_lbl = np.empty([int(len(train_images)*0.15),10])
    temp_test_img = np.empty([int(len(test_images)*0.15),32,32,3])
    temp_test_lbl = np.empty([int(len(test_images)*0.15),10])
    for i in range(0,len(train_set_random), 3):
        img1 = tf.image.flip_left_right(train_images[train_set_random[i]])
        label1 = train_labels[train_set_random[i]]
        img2 = tf.image.flip_up_down(train_images[train_set_random[i+1]])
        label2 = train_labels[train_set_random[i+1]]
        img3 = tf.image.random_brightness(train_images[train_set_random[i+2]], max_delta=0.2)
        label3 = train_labels[train_set_random[i+2]]
        temp_train_img[i] = img1
        temp_train_lbl[i] = label1
        temp_train_img[i+1] = img2
        temp_train_lbl[i+1] = label2
        temp_train_img[i+2] = img3
        temp_train_lbl[i+2] = label3
    for i in range(0,len(test_set_random), 3):
        img1 = tf.image.flip_left_right(test_images[test_set_random[i]])
        label1 = test_labels[test_set_random[i]]
        img2 = tf.image.flip_up_down(test_images[test_set_random[i+1]])
        label2 = test_labels[test_set_random[i+1]]
        img3 = tf.image.random_brightness(test_images[test_set_random[i+2]], max_delta=0.2)
        label3 = test_labels[test_set_random[i+2]]
        temp_test_img[i] = img1
        temp_test_lbl[i] = label1
        temp_test_img[i+1] = img2
        temp_test_lbl[i+1] = label2
        temp_test_img[i+2] = img3
        temp_test_lbl[i+2] = label3
    train_images, test_images = np.concatenate((train_images, temp_train_img), axis=0), np.concatenate((test_images, temp_test_img), axis=0)
    train_labels, test_labels = np.concatenate((train_labels, temp_train_lbl), axis=0), np.concatenate((test_labels, temp_test_lbl), axis=0)
    return train_images, train_labels, test_images, test_labels

def plot_results(history_dict, path, model):
    if path[-1] == '/':
        path = path[:-1]
    acc_img = f"{path}/{model}_accuracy_3l_.jpg"
    loss_img = f"{path}/{model}_loss_3l_.jpg"
    
    y1 = np.array(history_dict.history['categorical_accuracy'], dtype=float)
    y2 = np.array(history_dict.history['val_categorical_accuracy'], dtype=float)

    y3 = history_dict.history['loss']
    y4 = history_dict.history['val_loss']
    x = [i for i in range(1,len(y1)+1)]
    plt.plot(x,y1, label="train_set accuracy")
    plt.plot(x,y2, label="test_set accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.xticks(x)
    plt.legend()
    plt.savefig(acc_img)
    plt.close()

    plt.plot(x,y3, label="train_set loss")
    plt.plot(x,y4, label="test_set loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xticks(x)
    plt.legend()
    plt.savefig(loss_img)
    plt.close()


def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--model_name', '-m', default='basic', help='Name of the model to train and test')
    Parser.add_argument('--batch_size', '-b', default=128, type=int, help='batch size to train')
    Parser.add_argument('--epochs', '-e', default=10, type=int, help='batch size to train')
    Parser.add_argument('--ckpt_dir', '-c', default='/Users/pyone/Desktop/spring22/computer_vision_class/submission/Phase2/checkpoints', help='Path to save checkpoints')
    Parser.add_argument('--log_dir', '-l', default='/Users/pyone/Desktop/spring22/computer_vision_class/submission/Phase2/logs/fit', help='Path to save traning log')
    Parser.add_argument('--plot_dir', '-p', default='/Users/pyone/Desktop/spring22/computer_vision_class/submission/Phase2/plots', help='Path to save result over epoch plots')
    Args = Parser.parse_args()

    model_name = Args.model_name

    ckpt_dir = Args.ckpt_dir
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"{Args.log_dir}/{model_name}_{dt}"
    plot_dir = Args.plot_dir
    ckpt_name = f'{ckpt_dir}/{model_name}_wts.hdf5'

    # data #
    train_images, train_labels, test_images, test_labels = preprocessing()
    # train_images, train_labels, test_images, test_labels = data_augmentation(train_images, train_labels, test_images, test_labels)
    input_shape = np.shape(train_images[0])

    # hyper params  #
    batch_size = Args.batch_size
    epochs = Args.epochs
    opt = optimizers.Adam(learning_rate = 0.001)
    loss = losses.CategoricalCrossentropy()

    model_checkpoint_callback = callbacks.ModelCheckpoint(
                        filepath=ckpt_name,
                        monitor='val_categorical_accuracy',
                        mode='max',
                        save_best_only=True,
                        save_freq = 'epoch')
    tensor_board_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    mycallbacks = [
        model_checkpoint_callback,
        tensor_board_callback
    ]

    if model_name.lower() == 'basic':
        model = basicNetwork(input_shape)
    elif model_name.lower() == 'better':
        model = optimizedNetwork(input_shape)
    elif model_name.lower() == 'resnet':
        filter_sizes = [64,128,256,512]
        initial_learning_rate = 0.1
        lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000,
                                                        decay_rate=0.5, staircase=True)
        opt = optimizers.SGD(learning_rate = lr_schedule)
        model = ResNet34(filter_sizes)
        model.build((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    elif model_name.lower() == 'densenet':
        growth_rate=12
        dense_block_sizes = [6,12,24,16] ##DenseNet121
        compression_factor = 0.5
        batch_size=32
        model = DenseNet(growth_rate, dense_block_sizes, compression_factor)
        model.build((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    elif model_name.lower() == 'resnext':
        model = ResNeXt(repeat_num_list=[3, 3, 3, 3], cardinality=6)
        model.build((batch_size, input_shape[0], input_shape[1], input_shape[2]))
    else:
        print("Input model is not available. Exiting......")
        exit()
    model.summary()
    
    model.compile(optimizer=opt, loss = loss, metrics=['categorical_accuracy'])
    hist1 = model.fit(x=train_images, y=train_labels, validation_data=(test_images, test_labels),
                    callbacks=mycallbacks, batch_size=batch_size, epochs=epochs, verbose=2, use_multiprocessing=True)
    plot_results(hist1, plot_dir, model_name)

    ##model predictions
    train_predictions = model.predict(train_images)
    max_train_predictions = []
    for p in train_predictions:
        max_train_predictions.append(np.argmax(p))

    test_predictions = model.predict(test_images)
    max_test_predictions = []
    for p in test_predictions:
        max_test_predictions.append(np.argmax(p))

    # actual label
    train_label_class = []
    for i in range(len(train_labels)):
        for j in range(len(train_labels[i])):
            if train_labels[i,j] == 1:
                train_label_class.append(j)

    test_label_class = []
    for i in range(len(test_labels)):
        for j in range(len(test_labels[i])):
            if test_labels[i,j] == 1:
                test_label_class.append(j)

    train_cm = tf.math.confusion_matrix(labels=train_label_class, predictions=max_train_predictions, num_classes=10)
    test_cm = tf.math.confusion_matrix(labels=test_label_class, predictions=max_test_predictions, num_classes=10)
    print(train_cm)
    print(test_cm)
if __name__ == '__main__':
    main()
 
