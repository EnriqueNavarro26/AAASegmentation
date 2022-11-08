
import numpy as np
import os
import SimpleITK as sitk
from tensorflow.keras.models import load_model

from Model import config, dice_coefficient, dice_coefficient_loss, jacard_coef, jacard_coef_loss



def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                       'jacard_coef_loss': jacard_coef_loss, 'jacard_coef': jacard_coef }
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error


def get_mask(data):
    return ( np.array(data) >= 0.5 ).astype(int)


def get_prediction_from_array_with_reference(array, ref_img):
    img = sitk.GetImageFromArray(np.squeeze(array))

    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputDirection(img.GetDirection())
    spacing = np.array(img.GetSize())*np.array(img.GetSpacing())/np.array(ref_img.GetSize())
    resampler.SetOutputSpacing( spacing )
    resampler.SetOutputOrigin( img.GetOrigin() )
    resampler.SetSize( ref_img.GetSize() )    
    out_img = resampler.Execute(img)

    out_array = sitk.GetArrayFromImage(out_img)
    out_array = get_mask(out_array)
    out_img = sitk.GetImageFromArray(out_array)

    out_img.CopyInformation(ref_img)  
    return out_img


def filter_mask_to_center(mask):
    connected = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(connected, 1)
    np_mask = sitk.GetArrayFromImage(relabeled)
    coordinates = np.argwhere(np_mask==1)
    z_range = (np.amin(coordinates[:,0]), np.amax(coordinates[:,0]))
    y_range = (np.amin(coordinates[:,1]), np.amax(coordinates[:,1]))
    x_range = (np.amin(coordinates[:,2]), np.amax(coordinates[:,2]))
    coordinates = np.array((z_range, y_range, x_range))    
    return coordinates


def crop_image(in_img, coordinates, p=20, mask=False):

    image = sitk.GetArrayFromImage(in_img)
    out_img = image[coordinates[0][0]-p:coordinates[0][1]+p,coordinates[1][0]-p:coordinates[1][1]+p,coordinates[2][0]-p:coordinates[2][1]+p] #crop
    out_img = sitk.GetImageFromArray(out_img)

    volume = sitk.Cast(out_img, sitk.sitkFloat32) # cast volume to float32
    resample_vol = sitk.ResampleImageFilter()

    if mask: 
        resample_vol.SetInterpolator = sitk.sitkNearestNeighbor  # set interpolator for mask (it does not create new masks)
    else:
        resample_vol.SetInterpolator = sitk.sitkLinear

    resample_vol.SetOutputDirection = volume.GetDirection()  # set output volume direction equal to input volume direction
    resample_vol.SetOutputOrigin = volume.GetOrigin()
    new_spacing = np.array(in_img.GetSpacing())
    resample_vol.SetOutputSpacing(new_spacing)
    orig_size = np.array(volume.GetSize(), dtype=np.int)
    orig_spacing = volume.GetSpacing()
    new_size = orig_size * (np.divide(orig_spacing, new_spacing))
    new_size = np.ceil(new_size).astype(np.int)  # volume dimensions are in integers
    new_size = [int(s) for s in new_size]  # convert from np.array to list
    resample_vol.SetSize(new_size)
    resampled_volume = resample_vol.Execute(volume)

    if mask:
        resampled_volume = sitk.Cast(resampled_volume, sitk.sitkInt16)

    return resampled_volume

def crop_results(folder, data_folder, imgs, subjects, model, overwrite=False):
    if not os.path.exists(folder) or overwrite:
        if not os.path.exists(folder):
            os.makedirs(folder)
    predictions = model.predict(imgs, batch_size=1, verbose=1)

    for i in range(len(imgs)):
        pred = predictions[i]
        sub = subjects[i]
        print(sub)
        original_image = sitk.ReadImage(data_folder + "/" + sub + "/" + config["filename_image"])
        original_gt = sitk.ReadImage(data_folder + "/" + sub + "/" + config["filename_mask"])

        im_pred = get_prediction_from_array_with_reference(predictions[i], original_image)
        coordinates = filter_mask_to_center(im_pred)

        im_img = crop_image(original_image, coordinates)
        im_gt = crop_image(original_gt, coordinates, mask=True)

        subject_folder = os.path.join(folder, sub)
        if not os.path.exists(subject_folder) or overwrite:
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)
            sitk.WriteImage(im_img, subject_folder + config["filename_image"], True)
            sitk.WriteImage(im_gt, subject_folder + config["filename_mask"], True)


#load subjects
testing_imgs = np.load(config["validation_files"][0])
testing_subjects = list()
with open(config["validation_files"][2], 'r') as f:  
    for line in f:
        sub = line[:-1]
        testing_subjects.append(sub)
training_imgs = np.load(config["training_files"][0])
training_subjects = list()
with open(config["training_files"][2], 'r') as f:  
    for line in f:
        sub = line[:-1]
        training_subjects.append(sub)


model = load_old_model(config["model_file"])

crop_results("./cropped/", config["data_folder"], testing_imgs, testing_subjects, model)

crop_results("./cropped/", config["data_folder"], training_imgs, training_subjects, model)

