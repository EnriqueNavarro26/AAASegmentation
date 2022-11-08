from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.simplefilter('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Model import config



# by Zihao Wang.

class image_augmentation:
    def __init__(self, X, Y):
        self.X = np.squeeze(X)
        self.Y = np.squeeze(Y)
        print('Data augment')

    def aug_train_iterator(self, is_binary=True):

        seed = 7  # make sure that two iterators give same tomato each time...

        ig = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=False,
            rotation_range=config["rotation_range"],
            width_shift_range=config["width_shift_range"],
            height_shift_range=config["height_shift_range"],
            horizontal_flip=config["horizontal_flip"],
            vertical_flip=config["vertical_flip"],
            zoom_range=config["zoom_range"],
            data_format = "channels_first")

        # ig.fit(self.X) ## pay attention to the pixel value generated of y (segmentation ground truth) and convert into 0, 1

        shape = tuple([1] + list(reversed(config["image_shape"])))
        for batch in zip(ig.flow(self.X, seed=seed), ig.flow(self.Y, seed=seed)):
            for i in range(len(batch[0])):
                x1 = batch[0][i].reshape(shape)
                x2 = batch[1][i].reshape(shape)
                yield (np.expand_dims(x1,1), np.expand_dims(x2,1))
