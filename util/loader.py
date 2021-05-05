# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import json
import os
import cv2
import numpy as np
import pickle
import random
import colorsys
from glob import glob

def readConfig(config_file_name = "./config.json"):
    """ Read the configure file (json). """
    with open(config_file_name) as json_file:
        config = json.load(json_file)
    return config

def readAnchorBoxes(anchor_boxes_file = "./anchors.txt"):
    """ Read the anchor boxes found by k means """
    anchor_boxes = []
    with open(anchor_boxes_file) as txt_file:
        lines = txt_file.readlines()
        for line in lines:
            box = []
            characters = line.split(" ")
            for charact in characters:
                if charact != '\n':
                    box.append(int(charact))
            anchor_boxes.append(box)
    if len(anchor_boxes) == 0:
        anchor_boxes = None
    else:
        anchor_boxes = np.array(anchor_boxes)
        if anchor_boxes_file is not "./anchors.txt":
            anchor_boxes = sortAnchorBoxes(anchor_boxes, "2D")
        else:
            anchor_boxes = sortAnchorBoxes(anchor_boxes)
    return anchor_boxes

def sortAnchorBoxes(anchor_boxes, mode="3D"):
    """ Sort anchor boxes according to its area """
    if mode == "3D":
        anchor_box_areas = anchor_boxes[:, 0] * anchor_boxes[:, 1] * anchor_boxes[:, 2]
    else:
        anchor_box_areas = anchor_boxes[:, 0] * anchor_boxes[:, 1]
    anchor_boxes_order = np.argsort(anchor_box_areas)
    # print(anchor_box_areas[anchor_boxes_order]) # print out the boxes' areas
    return anchor_boxes[anchor_boxes_order]

def getSequenceNumbers(radar_dir, data_format):
    """ Get all the numbers from input file names """
    assert isinstance(data_format, list)
    assert all(isinstance(x, str) for x in data_format)
    assert (len(data_format) == 2 or len(data_format) == 3)
    sequence_numbers = []
    for this_file in glob(os.path.join(radar_dir, "*"+data_format[-1])):
        digits = int("".join((s for s in this_file.replace(radar_dir, "") \
                    if s.isdigit())))
        sequence_numbers.append(digits)
    if len(sequence_numbers) == 0:
        sequence_numbers = None
    else:
        sequence_numbers = np.sort(sequence_numbers)
    return sequence_numbers

def randomColors(classes, bright=True): 
    """ Define colors for all categories. """
    assert isinstance(classes, list)
    assert all(isinstance(x, str) for x in classes)
    N = len(classes)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(8888)
    random.shuffle(colors)
    return colors
 
def gtfileFromRADfile(RAD_file, prefix):
    """ Transfer RAD filename to gt filename """
    RAD_file_spec = RAD_file.split("RAD")[-1]
    gt_file = os.path.join(prefix, "gt") + RAD_file_spec.replace("npy", "pickle")
    return gt_file
 
def imgfileFromRADfile(RAD_file, prefix):
    """ Transfer RAD filename to gt filename """
    RAD_file_spec = RAD_file.split("RAD")[-1]
    gt_file = os.path.join(prefix, "stereo_image") + RAD_file_spec.replace("npy", "jpg")
    return gt_file

def readRAD(filename):
    """ read input RAD matrices """
    if os.path.exists(filename):
        return np.load(filename)
    else:
        return None

def readRadarInstances(pickle_file):
    """ read output radar instances. """
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            radar_instances = pickle.load(f)
        if len(radar_instances['classes']) == 0:
            radar_instances = None
    else:
        radar_instances = None
    return radar_instances
   
def readStereoLeft(img_filename):
    """ read stereo left image for verification. """
    if os.path.exists(img_filename):
        stereo_image = cv2.imread(img_filename)
        left_image = stereo_image[:, :stereo_image.shape[1]//2, ...][..., ::-1]
        return left_image
    else:
        return None
    
def readSingleImage(img_filename):
    """ read stereo left image for verification. """
    if os.path.exists(img_filename):
        image = cv2.imread(img_filename)
        image = image[..., ::-1]
        return image
    else:
        return None
   
