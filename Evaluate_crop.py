import numpy as np
import os
import SimpleITK as sitk
from tensorflow.keras.models import load_model

from Model import config, dice_coefficient, dice_coefficient_loss, contour_loss, jacard_coef_loss, jacard_coef

from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr

from skimage.measure import label, regionprops, regionprops_table

# testing_imgs = np.load(config["validation_files"][0])
# testing_segs = np.load(config["validation_files"][1])
# testing_subjects = list()

k = 5

print('[INFO] -- EVALUANDO BOLSA ', k)

testing_imgs = np.load(f'./arrays_fold/testing_imgs_FOLD{k}.npy')
testing_segs = np.load(f'./arrays_fold/testing_segs_FOLD{k}.npy')
testing_subjects = list()

validation_subjects = 'Subjects_TEST_FOLD'+str(k)+'.txt'

with open(validation_subjects, 'r') as f:  
    for line in f:
        sub = line[:-1]
        testing_subjects.append(sub)

def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                        'contour_loss' : contour_loss, 'jacard_coef_loss': jacard_coef_loss, 'jacard_coef': jacard_coef}
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


def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    https://loli.github.io/medpy/_modules/medpy/metric/binary.html
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def asd(result, reference, voxelspacing=None, connectivity=1):

    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd
 
def ravd(result, reference):
    
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
        
    vol1 = np.count_nonzero(result)
    vol2 = np.count_nonzero(reference)
    
    if 0 == vol2:
        raise RuntimeError('The second supplied array does not contain any binary object.')
    
    return (vol1 - vol2) / float(vol2)
    



def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds


def errorDiameter(d_mask, d_pred):
    return abs(d_mask - d_pred)

def getMaxDiameters(mask_vol, pred_vol, spacing):

    d_mask=[]
    d_pred=[]

    for i in range(mask_vol.shape[2]):
        mask = mask_vol[:,:,i]
        pred = pred_vol[:,:,i]
        label_pred = label(pred)
        regions_pred = regionprops(label_pred)

        label_mask = label(mask)
        regions_mask = regionprops(label_mask)


        for props in regions_mask:
            # print(i, spacing*props.equivalent_diameter_area)
            
            d_mask.append(spacing*props.equivalent_diameter_area)

        for props_2 in regions_pred:
            # print(props_2.equivalent_diameter_area)
            d_pred.append(spacing*props_2.equivalent_diameter_area)

    d_mask = np.array(d_mask)
    d_pred = np.array(d_pred)
    
    return d_mask.max(), d_pred.max()

model = load_old_model(f'NET_FOLD{k}.hdf5')

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

def getSpacing(original_image):
    return original_image.GetSpacing()

def write_results(folder, data_folder, overwrite=True):
    if not os.path.exists(folder) or overwrite:
        if not os.path.exists(folder):
            os.makedirs(folder)


    dice_array = []
    jaccard_array =[]
    hd95_array = []
    asd_array=[]
    ravd_array=[]


    for i in range(len(testing_segs)):
        gt = testing_segs[i]
        if not config["is_truth_mask"]:
            gt = ( np.array(gt) < 0 ).astype(np.uint8)
        img = testing_imgs[i]
        pred = get_mask(predictions[i])
        # dice = dice + (2*np.sum(gt * pred))/(np.sum(gt)+np.sum(pred))
        dice = (2*np.sum(gt * pred))/(np.sum(gt)+np.sum(pred))
        jaccard = (np.sum(gt * pred)) / ((np.sum(gt)+np.sum(pred)) - (np.sum(gt * pred)))

        sub = testing_subjects[i]
        print(sub)
        print('DICE:', dice)
        print('JACCARD:', jaccard)
        
        original_image = sitk.ReadImage(data_folder + "/" + sub + "/" + config["filename_image"])

        spacing = getSpacing(original_image)
        
        

        hd = hd95(gt, pred, voxelspacing=(1, spacing[0], spacing[1], spacing[2]), connectivity=1)
        print('HD95:', hd)

        a = asd(gt, pred, voxelspacing=(1, spacing[0], spacing[1], spacing[2]), connectivity=1)
        print('ASD:', a)

        print('VOLUME', ravd(gt, pred))

        
        

        dice_array.append(dice)
        jaccard_array.append(jaccard)
        hd95_array.append(hd)
        asd_array.append(a)
        ravd_array.append(ravd(gt, pred))

        print()

        
        im_img = get_image_from_array_with_reference(img, original_image, 0)
        im_pred = get_image_from_array_with_reference(predictions[i], original_image, 2)
        im_gt = get_image_from_array_with_reference(gt, original_image, 1)




        subject_folder = os.path.join(folder, sub)
        if not os.path.exists(subject_folder) or overwrite:
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)
            # sitk.WriteImage(im_img, subject_folder + config["output_filenames"][0], True)
            # sitk.WriteImage(im_gt, subject_folder + config["output_filenames"][1], True)
            # sitk.WriteImage(im_pred, subject_folder + config["output_filenames"][2], True)
            sitk.WriteImage(im_img, subject_folder + "/img.nii.gz", True)
            sitk.WriteImage(im_gt, subject_folder + "/gt.nii.gz", True)
            sitk.WriteImage(im_pred, subject_folder + "/pred.nii.gz", True)



    # print("dice = " + str(dice/len(testing_segs)))
    print()
    print('-----RESULTADOS-----')
    print('DICE:', np.mean(dice_array), np.std(dice_array))
    print('JACCARD:', np.mean(jaccard_array), np.std(jaccard_array))
    print('HD95:', np.mean(hd95_array), np.std(hd95_array))
    print('ASD:', np.mean(asd_array), np.std(asd_array))
    print('RAVD:', np.mean(ravd_array), np.std(ravd_array))



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

