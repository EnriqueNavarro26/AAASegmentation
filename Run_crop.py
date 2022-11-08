from functools import partial
import os
import sys
from tabnanny import verbose
import numpy as np
import glob
import SimpleITK as sitk
from Model import unet3d
from Model import config
import tensorflow as tf
from sklearn.model_selection import KFold
tf.config.run_functions_eagerly(True)

from sklearn.model_selection import cross_val_score

from Model import  dice_coefficient, dice_coefficient_loss, contour_loss,  jacard_coef, jacard_coef_loss
# import seg_metrics.seg_metrics as sg


def resampling(img, size=config["image_shape"], is_mask=0):
    ''' Resize volumes '''
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
    ''' Get data from folders with format extension
        return all imgs, masks in array numpy and subjects and spacing from vol array    
    '''
    imgs = list()
    masks = list()
    subjects = list()
    spacing = list()

    n=0
    for subject in glob.glob(data_folder + "/*"):
        subjects.append(os.path.basename(subject))
        img = sitk.ReadImage(subject+ config["filename_image"]) #img.nii.gz
        spa = img.GetSpacing()
        img = resampling(img) #resize
        
        mask = sitk.ReadImage(subject + config["filename_truth"]) #mask.nii.gz
        mask = resampling(mask,is_mask=is_truth_mask)#resize

        #imgs and masks in Numpy array --> float32
        np_img = np.expand_dims(sitk.GetArrayFromImage(img), axis=0).astype(np.float32)
        np_mask = np.expand_dims(sitk.GetArrayFromImage(mask), axis=0).astype(np.float32)

        for i in range(len(label)):
            np_mask[np_mask == label[i]] = i + 1
        n+=1
        if n%20 == 0:
            print("reading training datasets: subject "+str(n))
        imgs.append(np_img)
        masks.append(np_mask)
        spacing.append(spa)
    return np.array(imgs), np.array(masks), subjects, np.array(spacing)

data_folder = config["data_folder"]
imgs, segs, subjects, spacing = get_training_datasets(data_folder)

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

# Save imgs in segs as npy
np.save('imagenes.npy', imgs)  
np.save('segmentaciones.npy', segs)  


# Generator
import Augmentor 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping


dice_array = [] # 
jacard_array=[]


kfold = KFold(n_splits=5, shuffle=True, random_state = 42)

fold_no = 1
for train, test in kfold.split(imgs, segs):

    training_imgs = imgs[train]
    testing_imgs = imgs[test]
    training_segs = segs[train]
    testing_segs = segs[test]

    if not os.path.exists('./arrays_fold/'):
            os.makedirs('./arrays_fold/')

    np.save(f'./arrays_fold/training_imgs_FOLD{fold_no}.npy', training_imgs)
    np.save(f'./arrays_fold/training_segs_FOLD{fold_no}.npy', training_segs)
    np.save(f'./arrays_fold/testing_imgs_FOLD{fold_no}.npy', testing_imgs)
    np.save(f'./arrays_fold/testing_segs_FOLD{fold_no}.npy', testing_segs)


    with open(f'Subjects_TRAINING_FOLD{fold_no}.txt', 'w') as f:
        for item in train:
             f.write("%s\n" % subjects[item])

    with open(f'Subjects_TEST_FOLD{fold_no}.txt', 'w') as f:
        for item in test:
             f.write("%s\n" % subjects[item])

    gen = Augmentor.image_augmentation(training_imgs,training_segs).aug_train_iterator()

    model = unet3d(input_shape=config["input_shape"], n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"],
                                   option_loss = config["option_loss"])


    def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                    learning_rate_patience=50, logging_file=f"training_FOLD{fold_no}.log", verbosity=1,
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

    model_file = f'Net_FOLD{fold_no}.hdf5'

    print('------------------------------------------------')
    print(f'Training for fold {fold_no}...')

    model.fit_generator(generator=gen, epochs=config["n_epochs"], verbose=1, steps_per_epoch=config["steps_per_epoch"],
                        validation_data=(testing_imgs,testing_segs),
                        callbacks=get_callbacks(model_file,
                                                initial_learning_rate=config["initial_learning_rate"] ,
                                                learning_rate_drop=config["learning_rate_drop"],
                                                learning_rate_epochs=None,
                                                learning_rate_patience=config["patience"],
                                                early_stopping_patience=config["early_stop"]))

    predictions = model.predict(testing_imgs, batch_size=1, verbose=1)
    
    _j = []
    j = []
    _d = []
    d = []

    for i in range(len(predictions)):
        
        result_j = jacard_coef_loss(testing_segs[i], predictions[i])
        print('JACARD', result_j)
        _j.append(result_j.numpy())
        j.append(_j)

        result_d = dice_coefficient_loss(testing_segs[i], predictions[i])
        print('DICE', result_d)
        _d.append(result_d.numpy())
        d.append(_d)
    
    if not os.path.exists('./results_fold/'):
            os.makedirs('./results_fold/')

    print(f'KFOLD{fold_no} --> {np.mean(j)} +/- {np.std(j)}')
    # jacard_array.append(j)
    with open(f'./results_fold/resultados_{fold_no}.txt', 'w') as f:
        f.write(f'JACARD, {np.mean(j)}, {np.std(j)}')
        f.write(f'\nDICE, {np.mean(d)}, {np.std(d)}')

    scores = model.evaluate(testing_imgs, testing_segs, verbose=0)
    # print(f'Score for fold {fold_no}: {scores}')
    # print(f'Score for fold {fold_no}')
    # print('SCORES', scores)
    # print('DICE -->', scores[0])
    # print('JACARD -->', scores[1])

    dice_array.append(scores[0])
    jacard_array.append(scores[1])

    fold_no = fold_no + 1

# jacard_array = np.array(jacard_array)
# for i in range(len(jacard_array)):
#     for index, value in enumerate(jacard_array[i]):
#         print(f'JACARD KFOLD {index} --> {np.mean(jacard_array[i][index])} +/- {np.std(jacard_array[i][index])}')
# print(f'jacard {np.mean(jacard_array)} +/- {np.std(jacard_array)}')



print(f'DICE {np.mean(dice_array)} +/- {np.std(dice_array)}')
print(f'JACARD {np.mean(jacard_array)} +/- {np.std(jacard_array)}')

np.save('dice_array.npy', dice_array)
np.save('jacard_array.npy', jacard_array )