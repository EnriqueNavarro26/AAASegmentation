import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, SpatialDropout3D, LeakyReLU, Add, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.layers.merge import concatenate
import numpy as np
import pydot
from keras.utils.vis_utils import plot_model

from functools import partial

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


#Generate dict to add info and hiperparameters

config = dict()

# Inputs
config["image_shape"] = [96,96,64] # cubic for augmentation. shape = tuple([1] + list(config["image_shape"]))
config["patch_shape"] = None
config["nb_channels"] = 1
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(reversed(config["patch_shape"])))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(reversed(config["image_shape"])))
    
config["labels"] = (1,)
config["n_labels"] = len(config["labels"])


# Input filenames
config["filename_image"] = "/img.nii.gz"
config["filename_mask"] = "/mask.nii.gz"
config["filename_distancemap"] = "/dist.nii.gz"

# Data load images
config["data_folder"] = "./cropped"

# Config loss functions
config["option_loss"] = 1 # 0 for contour, 1 for dice
config["drain"] = 1.8
config["is_truth_mask"] = config["option_loss"] 
if config["option_loss"]==1:
    config["filename_truth"] = config["filename_mask"] 
elif config["option_loss"]==0:
    config["filename_truth"] = config["filename_distancemap"] 
else:
    print("wrong choice of loss fuctions!")


# Validation
config["validation_split_ratio"] = 0.8

# Model options
config["n_base_filters"] = 16
config["initial_learning_rate"] = 5e-4

# Fitting generator options
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  
config["batch_size"] = 1
config["n_epochs"] = 250
config["steps_per_epoch"] = 200

# Augmentation options
config["rotation_range"]=10
config["width_shift_range"]=0.08
config["height_shift_range"]=0.08
config["horizontal_flip"] = False
config["vertical_flip"] = False
config["zoom_range"] = 0.2

# Output filenames
config["model_file"] = "Net.hdf5"
config["training_files"] = ["./training_imgs.npy","./training_segs.npy", "./training_subjects.txt"]
config["validation_files"] = ["./validation_imgs.npy","./validation_segs.npy", "./validation_subjects.txt"]
config["folder_prediction"] = "./prediction_crop/"
config["output_filenames"] = ["/img.mha", "/gt.mha", "/pred.mha"]

# Blocks for U-Net model

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False, data_format="channels_first"):
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, data_format=data_format)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def create_localization_module(input_layer, n_filters, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer, n_filters, data_format=data_format)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1), data_format=data_format)
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2), data_format="channels_first"):
    up_sample = UpSampling3D(size=size, data_format=data_format)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters, data_format=data_format)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters, data_format=data_format)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters, data_format=data_format)
    return convolution2

# Metric for loss function

# Dice coeffcient loss
def dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

# Countour loss
def contour_loss(y_true, y_pred):  
    sobelFilters = K.variable([ 
                         [ [ [-1, -2, -1], [-2, -4, -2], [-1, -2, -1] ],
                                [ [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
                                 [ [1, 2, 1], [2, 4, 2], [1, 2, 1] ] ],
                         [ [ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ],
                                [ [2, 4, 2], [0, 0, 0],  [-2, -4, -2] ],
                                 [ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ] ],
                         [ [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ],
                                [ [-2, 0, 2], [-4, 0, 4], [-2, 0, 2] ],
                                 [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ] ]
                            ])
    sobelFilters = K.expand_dims(sobelFilters, axis=-1)
    sobelFilters = K.expand_dims(sobelFilters, axis=-1)
    contour = K.sum( K.concatenate(
            [K.abs(K.conv3d(y_pred, sobelFilters[0], padding='same', data_format='channels_first')),
                K.abs(K.conv3d(y_pred, sobelFilters[1], padding='same', data_format='channels_first')),
                    K.abs(K.conv3d(y_pred, sobelFilters[2], padding='same', data_format='channels_first'))]
                , axis=0), axis=0)
    contour_f = K.batch_flatten(contour)
    y_true_f = K.batch_flatten( K.abs(y_true) - config["drain"])

    finalChamferDistanceSum = K.sum(contour_f * y_true_f, axis=1, keepdims=True)

    return K.mean(finalChamferDistanceSum)


# Jaccard Index
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)




# Model

def unet3d(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=1, optimizer=Adam, initial_learning_rate=5e-4,
                      activation_name="sigmoid", data_format="channels_first", option_loss = 1):
    inputs = Input(input_shape)
    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2), data_format=data_format)(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)

    losses = [contour_loss, dice_coefficient_loss]
    model.compile(optimizer=optimizer(learning_rate=initial_learning_rate), loss=losses[option_loss], metrics = jacard_coef_loss)
    return model


# check how the model looks like
model = unet3d(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"])
                                  
# plot_model(model, to_file='./model.png') 
