# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt

import shutil
from glob import glob
from tqdm import tqdm
import time

import model.model as M
import model.model_cart as MCart
from dataset.batch_data_generator import DataGenerator
import metrics.mAP as mAP

import util.loader as loader
import util.helper as helper
import util.drawer as drawer


def cutImage(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1540:1750, :]
    part_2 = image[:, 2970:3550, :]
    part_3 = image[:, 4370:5400, :]
    part_4 = image[:, 6200:6850, :]
    new_img = np.concatenate([part_1, part_2, part_3, part_4], axis=1)
    cv2.imwrite(image_name, new_img)


def cutImage3Axes(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1780:2000, :]
    part_2 = image[:, 3800:4350, :]
    part_3 = image[:, 5950:6620, :]
    new_img = np.concatenate([part_1, part_2, part_3], axis=1)
    cv2.imwrite(image_name, new_img)


def loadDataForPlot(all_RAD_files, config_data, config_inference, \
                    config_radar, interpolation=15.):
    """ Load data one by one for generating evaluation images """
    sequence_num = -1
    for RAD_file in all_RAD_files:
        sequence_num += 1
        ### load RAD input ###
        RAD_complex = loader.readRAD(RAD_file)

        ### NOTE: real time visualization ###
        RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                        power_order=2), target_axis=-1), scalar=10, log_10=True)
        RD = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                        power_order=2), target_axis=1), scalar=10, log_10=True)
        RA_cart = helper.toCartesianMask(RA, config_radar, \
                                gapfill_interval_num=int(interpolation))
        RA_img = helper.norm2Image(RA)[..., :3]
        RD_img = helper.norm2Image(RD)[..., :3]
        RA_cart_img = helper.norm2Image(RA_cart)[..., :3]

        img_file = loader.imgfileFromRADfile(RAD_file, config_data["test_set_dir"])
        stereo_left_image = loader.readStereoLeft(img_file)

        RAD_data = helper.complexTo2Channels(RAD_complex)
        RAD_data = (RAD_data - config_data["global_mean_log"]) / \
                            config_data["global_variance_log"]
        data = tf.expand_dims(tf.constant(RAD_data, dtype=tf.float32), axis=0)
        yield sequence_num, data, stereo_left_image, RD_img, RA_img, RA_cart_img


def main():
    ### NOTE: GPU manipulation, you may can print this out if necessary ###
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    config = loader.readConfig()
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_evaluate = config["EVALUATE"]
    config_inference = config["INFERENCE"]

    anchor_boxes = loader.readAnchorBoxes() # load anchor boxes with order
    anchor_cart = loader.readAnchorBoxes(anchor_boxes_file="./anchors_cartboxes.txt")
    num_classes = len(config_data["all_classes"])

    ### NOTE: using the yolo head shape out from model for data generator ###
    model = M.RADDet(config_model, config_data, config_train, anchor_boxes)
    model.build([None] + config_model["input_shape"])
    model.backbone_stage.summary()
    model.summary()

    ### NOTE: building another model for Cartesian Boxes ###
    model_cart = MCart.RADDetCart(config_model, config_data, config_train, \
                                anchor_cart, list(model.backbone_fmp_shape))
    model_cart.build([None] + model.backbone_fmp_shape)
    model_cart.summary()

    ### NOTE: RAD Boxes ckpt ###
    logdir = os.path.join(config_inference["log_dir"], \
                        "b_" + str(config_train["batch_size"]) + \
                        "lr_" + str(config_train["learningrate_init"]))
    if not os.path.exists(logdir):
        raise ValueError("RAD Boxes model not loaded, please check the ckpt path.")
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    optimizer = K.optimizers.Adam(learning_rate=config_train["learningrate_init"])
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model, step=global_steps)
    manager = tf.train.CheckpointManager(ckpt, \
                os.path.join(logdir, "ckpt"), max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored RAD Boxes Model from {}".format(manager.latest_checkpoint))
    else:
        raise ValueError("RAD Boxes model not loaded, please check the ckpt path.")

    ### NOTE: Cartesian Boxes ckpt ###
    if_evaluate_cart = True
    logdir_cart = os.path.join(config_inference["log_dir"], "cartesian_" + \
                        "b_" + str(config_train["batch_size"]) + \
                        "lr_" + str(config_train["learningrate_init"]))
                        # "lr_" + str(config_train["learningrate_init"]) + \
                        # "_" + str(config_train["log_cart_add"]))
    if not os.path.exists(logdir_cart):
        if_evaluate_cart = False
        print("*************************************************************")
        print("Cartesian ckpt not found, skipping evaluating Cartesian Boxes")
        print("*************************************************************")
    if if_evaluate_cart:
        global_steps_cart = tf.Variable(1, trainable=False, dtype=tf.int64)
        optimizer_cart = K.optimizers.Adam(learning_rate=config_train["learningrate_init"])
        ckpt_cart = tf.train.Checkpoint(optimizer=optimizer_cart, model=model_cart, \
                                        step=global_steps_cart)
        manager_cart = tf.train.CheckpointManager(ckpt_cart, \
                    os.path.join(logdir_cart, "ckpt"), max_to_keep=3)
        ckpt_cart.restore(manager_cart.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored Cartesian Boxes Model from {}".format\
                                            (manager_cart.latest_checkpoint))

    def inferencePlotting(all_RAD_files):
        """ Plot the predictions of all data in dataset """
        if if_evaluate_cart:
            fig, axes = drawer.prepareFigure(4, figsize=(80, 6))
        else:
            fig, axes = drawer.prepareFigure(3, figsize=(80, 6))
        colors = loader.randomColors(config_data["all_classes"])

        image_save_dir = "./images/inference_plots/"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        else:
            shutil.rmtree(image_save_dir)
            os.makedirs(image_save_dir)
        print("Start plotting, it might take a while...")
        pbar = tqdm(total=len(all_RAD_files))
        model_RAD_st = []
        model_cart_st = []
        for sequence_num, data, stereo_left_image, RD_img, RA_img, RA_cart_img in \
                loadDataForPlot(all_RAD_files, config_data, config_inference, \
                                config_radar):
            if data is None or stereo_left_image is None:
                pbar.update(1)
                continue
            model_RAD_time_start = time.time()
            feature = model(data)
            pred_raw, pred = model.decodeYolo(feature)
            pred_frame = pred[0]
            predicitons = helper.yoloheadToPredictions(pred_frame, \
                                    conf_threshold=config_evaluate["confidence_threshold"])
            nms_pred = helper.nms(predicitons, \
                                    config_inference["nms_iou3d_threshold"], \
                                    config_model["input_shape"], \
                                    sigma=0.3, method="nms")
            model_RAD_st.append(time.time() - model_RAD_time_start)
            if if_evaluate_cart:
                model_cart_time_start = time.time()
                backbone_fmp = model.backbone_stage(data)
                pred_raw_cart = model_cart(backbone_fmp)
                pred_cart = model_cart.decodeYolo(pred_raw_cart)
                pred_frame_cart = pred_cart[0]
                predicitons_cart = helper.yoloheadToPredictions2D(pred_frame_cart, \
                                                        conf_threshold=0.5)
                nms_pred_cart = helper.nms2D(predicitons_cart, \
                                        config_inference["nms_iou3d_threshold"], \
                                        config_model["input_shape"], \
                                        sigma=0.3, method="nms")
                model_cart_st.append(time.time() - model_cart_time_start)
            else:
                nms_pred_cart = None
            drawer.clearAxes(axes)
            drawer.drawInference(stereo_left_image, RD_img, \
                    RA_img, RA_cart_img, nms_pred, \
                    config_data["all_classes"], colors, axes, \
                    radar_cart_nms=nms_pred_cart)
            drawer.saveFigure(image_save_dir, "%.6d.png"%(sequence_num))
            if if_evaluate_cart:
                cutImage(image_save_dir, "%.6d.png"%(sequence_num))
            else:
                cutImage3Axes(image_save_dir, "%.6d.png"%(sequence_num))
            pbar.update(1)
        print("------", " The average inference time for RAD Boxes: ", \
                                                np.mean(model_RAD_st))
        print("======", " The average inference time for Cartesian Boxes: ", \
                                                np.mean(model_cart_st))


    ### NOTE: inference starting from here ###
    all_RAD_files = glob(os.path.join(config_data["test_set_dir"], "RAD/*/*.npy"))
    inferencePlotting(all_RAD_files)


if __name__ == "__main__":
    main()
