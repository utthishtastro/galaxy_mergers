"""
    Util functions
"""

import glob
import json

from tqdm import tqdm
import cv2
import numpy as np


def load_json(json_path):
    with open(json_path, "r") as json_file:
        return json.load(json_file)


def _get_label(file_name, class_mappings):
    class_name = file_name.split("/")[-3]
    return class_mappings[class_name]


def _display_image(image):
    cv2.imshow("image window", image)
    cv2.waitKey(10)


def load_data(
        data_path, class_mappings, resize=None, show_image=False):
    files_list = glob.glob(data_path)
    image_data = []
    labels = []
    for each_file in tqdm(files_list):
        image = cv2.imread(each_file, -1)
        if resize:
            image = cv2.resize(image, resize)
        labels.append(_get_label(each_file, class_mappings))
        image_data.append(np.expand_dims(image, axis=0))
        if show_image:
            _display_image(image)
    image_data = np.vstack(image_data)
    image_data = image_data.reshape(image_data.shape[0], -1)
    return np.vstack(image_data), np.array(labels)
