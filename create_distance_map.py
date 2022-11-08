import glob
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from Model import config

def create_distance_map(subject_folder):
    mask = sitk.ReadImage(subject_folder + config["filename_mask"])
    np_mask = sitk.GetArrayFromImage(mask)
    np_mask_invert = np_mask == 0
    np_dist = distance_transform_edt(np_mask, sampling = list(reversed(mask.GetSpacing())))
    np_dist_invert = distance_transform_edt(np_mask_invert, sampling = list(reversed(mask.GetSpacing())) )
    np_dist = [y - x for x, y in zip(np_dist, np_dist_invert)]
    dist = sitk.GetImageFromArray(np_dist)
    dist.CopyInformation(mask)
    sitk.WriteImage(dist, subject_folder + config["filename_distancemap"], True)

def run_create_distance_map(data_folder):
    for subject_folder in glob.glob(os.path.join(data_folder, "*")):
        create_distance_map(subject_folder)

data_folder = config["data_folder"]
run_create_distance_map(data_folder)

data_folder

