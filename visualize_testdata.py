# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

import util.loader as loader
import util.helper as helper
import util.drawer as drawer

import multiprocessing as mp
from functools import partial

def cutImage(image_name):
    image = cv2.imread(image_name)
    part_1 = image[:, 1650:2550, :]
    part_2 = image[:, 3950:4250, :]
    part_3 = image[:, 5750:6500, :]
    part_4 = image[:, 7450:8850, :]
    new_img = np.concatenate([part_1, part_2, part_3, part_4], axis=1)
    cv2.imwrite(image_name, new_img)

def process(RAD_filename, frame_id, config_data, config_radar, colors, \
            fig, axes, interpolation=15, canvas_draw=False):
    RAD = loader.readRAD(RAD_filename)
    if "train" in RAD_filename: 
        prefix = config_data["train_set_dir"]
    else:
        prefix = config_data["test_set_dir"]
    gt_file = loader.gtfileFromRADfile(RAD_filename, prefix)
    gt_instances = loader.readRadarInstances(gt_file)
    img_file = loader.imgfileFromRADfile(RAD_filename, prefix)
    stereo_left_image = loader.readStereoLeft(img_file)
    if RAD is not None and gt_instances is not None and \
                            stereo_left_image is not None:
        RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD, power_order=2), \
                                            target_axis=-1), scalar=10, log_10=True)
        RD = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD, power_order=2), \
                                            target_axis=1), scalar=10, log_10=True)

        ### NOTE: change the interval number if high resolution is needed for Cartesian ###
        RA_cart = helper.toCartesianMask(RA, config_radar, \
                                gapfill_interval_num=interpolation)

        RA_img = helper.norm2Image(RA)[..., :3]
        RD_img = helper.norm2Image(RD)[..., :3]
        RA_cart_img = helper.norm2Image(RA_cart)[..., :3]

        drawer.clearAxes(axes)
        drawer.drawRadarBoxes(stereo_left_image, RD_img, RA_img, RA_cart_img, \
                            gt_instances, config_data["all_classes"], colors, axes)
        if not canvas_draw:
            drawer.saveFigure("./images/samples/", "%.6d.png"%(frame_id))
            cutImage("./images/samples/" + "%.6d.png"%(frame_id))
        else:
            drawer.keepDrawing(fig, 0.1)

def main(canvas_draw=False):
    config = loader.readConfig()
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    colors = loader.randomColors(config_data["all_classes"])
    if not canvas_draw:
        fig, axes = drawer.prepareFigure(4, figsize=(100, 8))
        interpolation = 15
    else:
        fig, axes = drawer.prepareFigure(4, figsize=(20, 8))
        interpolation = 1

    all_RAD_files = glob(os.path.join(config_data["test_set_dir"], \
                                        "RAD/*/*.npy"))

    for i in tqdm(range(len(all_RAD_files))):
        RAD_filename = all_RAD_files[i]
        process(
                RAD_filename=RAD_filename, \
                frame_id = i, \
                config_data=config_data, \
                config_radar=config_radar, \
                colors=colors, \
                fig=fig, \
                axes=axes, \
                interpolation=interpolation, \
                canvas_draw=canvas_draw)

if __name__ == "__main__":
    main(canvas_draw=True)

