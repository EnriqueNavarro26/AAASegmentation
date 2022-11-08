import numpy as np
import os
import SimpleITK as sitk
from tensorflow.keras.models import load_model

from Model import config, dice_coefficient, dice_coefficient_loss, contour_loss

testing_imgs = np.load(config["validation_files"][0])
testing_segs = np.load(config["validation_files"][1])
testing_subjects = list()
with open(config["validation_files"][2], 'r') as f:  
    for line in f:
        sub = line[:-1]
        testing_subjects.append(sub)

def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                        'contour_loss' : contour_loss}
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

model = load_old_model(config["model_file"])

predictions = model.predict(testing_imgs, batch_size=1, verbose=1)

def get_mask(data):
    return ( np.array(data) >= 0.5 ).astype(np.uint8)

def get_image_from_array_with_reference(array, ref_img, is_mask=0):
    img = sitk.GetImageFromArray(np.squeeze(array))

    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputDirection(img.GetDirection())
    spacing = np.array(img.GetSize())*np.array(img.GetSpacing())/np.array(ref_img.GetSize())
    resampler.SetOutputSpacing( spacing )
    resampler.SetOutputOrigin( img.GetOrigin() )
    resampler.SetSize( ref_img.GetSize() )    
    if is_mask == 1:
        resampler.SetInterpolator(1)

    out_img = resampler.Execute(img)

    if is_mask == 2:
        out_array = sitk.GetArrayFromImage(out_img)
        out_array = get_mask(out_array)
        out_img = sitk.GetImageFromArray(out_array)

    out_img.CopyInformation(ref_img)  
    return out_img

def write_results(folder, data_folder, overwrite=False):
    if not os.path.exists(folder) or overwrite:
        if not os.path.exists(folder):
            os.makedirs(folder)
    dice = 0 
    for i in range(len(testing_segs)):
        gt = testing_segs[i]
        if not config["is_truth_mask"]:
            gt = ( np.array(gt) < 0 ).astype(np.uint8)
        img = testing_imgs[i]
        pred = get_mask(predictions[i])
        dice = dice + (2*np.sum(gt * pred))/(np.sum(gt)+np.sum(pred))

        sub = testing_subjects[i]
        original_image = sitk.ReadImage(data_folder + "/" + sub + "/" + config["filename_image"])

        im_img = get_image_from_array_with_reference(img, original_image, 0)
        im_pred = get_image_from_array_with_reference(predictions[i], original_image, 2)
        im_gt = get_image_from_array_with_reference(gt, original_image, 1)
        subject_folder = os.path.join(folder, sub)
        if not os.path.exists(subject_folder) or overwrite:
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)
            sitk.WriteImage(im_img, subject_folder + config["output_filenames"][0], True)
            sitk.WriteImage(im_gt, subject_folder + config["output_filenames"][1], True)
            sitk.WriteImage(im_pred, subject_folder + config["output_filenames"][2], True)
    print("dice = " + str(dice/len(testing_segs)))

write_results(config["folder_prediction"], config["data_folder"])

'''
import matplotlib.pyplot as plt
def plot(img, gt, pred):
    plt.imshow(img[0,:,:])
    plt.show()
    plt.imshow(gt[0,:,:])
    plt.show()
    plt.imshow(pred[0,:,:])
    plt.show()

'''

