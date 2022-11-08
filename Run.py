# Import libraries
from functools import partial
import os
import sys
import numpy as np
import glob
import SimpleITK as sitk
from sklearn.model_selection import KFold

from Model import unet3d
from Model import config

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping


def resampling(img, size=config["image_shape"], is_mask=0):
    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputDirection(img.GetDirection())
    spacing = np.array(img.GetSize())*np.array(img.GetSpacing())/np.array(size)
    resampler.SetOutputSpacing( spacing )
    resampler.SetOutputOrigin( img.GetOrigin() )
    resampler.SetSize( size )
    if is_mask:
        resampler.SetInterpolator(1)
    out_img = resampler.Execute(img)
    return out_img

def get_training_datasets(data_folder, label= config["labels"], is_truth_mask = config["is_truth_mask"]):
    imgs = list()
    masks = list()
    subjects = list()
    n=0
    for subject in glob.glob(data_folder + "/*"):
        subjects.append(os.path.basename(subject))
        img = sitk.ReadImage(subject+ config["filename_image"])
        img = resampling(img)
        mask = sitk.ReadImage(subject + config["filename_truth"])
        mask = resampling(mask,is_mask=is_truth_mask)
        np_img = np.expand_dims(sitk.GetArrayFromImage(img), axis=0).astype(np.float32)
        np_mask = np.expand_dims(sitk.GetArrayFromImage(mask), axis=0).astype(np.float32)
        for i in range(len(label)):
            np_mask[np_mask == label[i]] = i + 1
        n+=1
        if n%20 == 0:
            print("reading training datasets: subject "+str(n))
        imgs.append(np_img)
        masks.append(np_mask)
    return np.array(imgs), np.array(masks), subjects

data_folder = config["data_folder"]
imgs, segs, subjects = get_training_datasets(data_folder)

# Check gpus available

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#     # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

# Validation split
ids = np.random.permutation(len(segs))
n_train = int(len(segs)*config["validation_split_ratio"])
training_imgs = imgs[ids[:n_train]]
training_segs = segs[ids[:n_train]]

np.save(config["training_files"][0], training_imgs)  
np.save(config["training_files"][1], training_segs)  
with open(config["training_files"][2], 'w') as f:
    for item in ids[:n_train]:
        f.write("%s\n" % subjects[item])

testing_imgs = imgs[ids[n_train:]]
testing_segs = segs[ids[n_train:]]
np.save(config["validation_files"][0], testing_imgs)  
np.save(config["validation_files"][1], testing_segs)  
with open(config["validation_files"][2], 'w') as f:
    for item in ids[n_train:]:
        f.write("%s\n" % subjects[item])

# Model

model = unet3d(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"],
                                   option_loss = config["option_loss"])

# Generator
import Augmentor 

gen = Augmentor.image_augmentation(training_imgs,training_segs).aug_train_iterator()


# Training

def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks

model_file = config["model_file"] 

model.fit_generator(generator=gen, epochs=config["n_epochs"], verbose=1, steps_per_epoch=config["steps_per_epoch"],
                        validation_data=(testing_imgs,testing_segs),
                        callbacks=get_callbacks(model_file,
                                                initial_learning_rate=config["initial_learning_rate"] ,
                                                learning_rate_drop=config["learning_rate_drop"],
                                                learning_rate_epochs=None,
                                                learning_rate_patience=config["patience"],
                                                early_stopping_patience=config["early_stop"]))



