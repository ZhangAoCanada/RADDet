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
from time import sleep
from tabulate import tabulate

import model.model as M
import model.model_cart as MCart
from dataset.batch_data_generator import DataGenerator
import metrics.mAP as mAP

import util.loader as loader
import util.helper as helper
import util.drawer as drawer

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

    ### NOTE: preparing data ###
    data_generator = DataGenerator(config_data, config_train, config_model, \
                    model.features_shape, anchor_boxes, \
                    anchors_cart=anchor_cart, cart_shape=model_cart.features_shape)
    train_generator = data_generator.trainGenerator()
    test_generator = data_generator.testGenerator()
    train_cart_generator = data_generator.trainCartGenerator()
    test_cart_generator = data_generator.testCartGenerator()

    ### NOTE: RAD Boxes ckpt ###
    logdir = os.path.join(config_evaluate["log_dir"], \
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
    logdir_cart = os.path.join(config_evaluate["log_dir"], "cartesian_" + \
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



    ### NOTE: define testing step for RAD Boxes Model ###
    # @tf.function
    def test_step(map_iou_threshold_list):
        mean_ap_test_all = [] 
        ap_all_class_test_all = []
        ap_all_class_all = []
        for i in range(len(map_iou_threshold_list)):
            mean_ap_test_all.append(0.0)
            ap_all_class_test_all.append([])
            ap_all_class = []
            for class_id in range(num_classes):
                ap_all_class.append([])
            ap_all_class_all.append(ap_all_class)
        print("Start evaluating RAD Boxes on the entire dataset, it might take a while...")
        pbar = tqdm(total=int(data_generator.total_test_batches))
        for data, label, raw_boxes in test_generator.repeat().\
            batch(data_generator.batch_size).take(data_generator.total_test_batches):
            feature = model(data)
            pred_raw, pred = model.decodeYolo(feature)
            for batch_id in range(raw_boxes.shape[0]):
                raw_boxes_frame = raw_boxes[batch_id]
                pred_frame = pred[batch_id]
                predicitons = helper.yoloheadToPredictions(pred_frame, \
                                    conf_threshold=config_evaluate["confidence_threshold"])
                nms_pred = helper.nms(predicitons, \
                                    config_evaluate["nms_iou3d_threshold"], \
                                    config_model["input_shape"], sigma=0.3, method="nms")
                for j in range(len(map_iou_threshold_list)):
                    map_iou_threshold = map_iou_threshold_list[j]
                    mean_ap, ap_all_class_all[j] = mAP.mAP(nms_pred, \
                                                    raw_boxes_frame.numpy(), \
                                                    config_model["input_shape"], \
                                                    ap_all_class_all[j], \
                                                    tp_iou_threshold=map_iou_threshold)
                    mean_ap_test_all[j] += mean_ap
            pbar.update(1)
        for iou_threshold_i in range(len(map_iou_threshold_list)):
            ap_all_class = ap_all_class_all[iou_threshold_i]
            for ap_class_i in ap_all_class:
                if len(ap_class_i) == 0:
                    class_ap = 0.
                else:
                    class_ap = np.mean(ap_class_i)
                ap_all_class_test_all[iou_threshold_i].append(class_ap)
            mean_ap_test_all[iou_threshold_i] /= \
                        data_generator.batch_size*data_generator.total_test_batches
        return mean_ap_test_all, ap_all_class_test_all


    ### NOTE: define testing step for Cartesian Boxes Model ###
    # @tf.function
    def test_step_cart(map_iou_threshold_list):
        mean_ap_test_all = [] 
        ap_all_class_test_all = []
        ap_all_class_all = []
        for i in range(len(map_iou_threshold_list)):
            mean_ap_test_all.append(0.0)
            ap_all_class_test_all.append([])
            ap_all_class = []
            for class_id in range(num_classes):
                ap_all_class.append([])
            ap_all_class_all.append(ap_all_class)
        print("Start evaluating Cartesian Boxes on the entire dataset, \
                                                it might take a while...")
        pbar = tqdm(total=int(data_generator.total_test_batches))
        for data, label, raw_boxes in test_cart_generator.repeat().\
            batch(data_generator.batch_size).take(data_generator.total_test_batches):
            backbone_fmp = model.backbone_stage(data)
            pred_raw = model_cart(backbone_fmp)
            pred = model_cart.decodeYolo(pred_raw)
            for batch_id in range(raw_boxes.shape[0]):
                raw_boxes_frame = raw_boxes[batch_id]
                pred_frame = pred[batch_id]
                predicitons = helper.yoloheadToPredictions2D(pred_frame, \
                                                            conf_threshold=0.5)
                nms_pred = helper.nms2D(predicitons, \
                                    config_evaluate["nms_iou3d_threshold"], \
                                    config_model["input_shape"], sigma=0.3, method="nms")
                for j in range(len(map_iou_threshold_list)):
                    map_iou_threshold = map_iou_threshold_list[j]
                    mean_ap, ap_all_class_all[j] = mAP.mAP2D(nms_pred, \
                                                    raw_boxes_frame.numpy(), \
                                                    config_model["input_shape"], \
                                                    ap_all_class_all[j], \
                                                    tp_iou_threshold=map_iou_threshold)
                    mean_ap_test_all[j] += mean_ap
            pbar.update(1)
        for iou_threshold_i in range(len(map_iou_threshold_list)):
            ap_all_class = ap_all_class_all[iou_threshold_i]
            for ap_class_i in ap_all_class:
                if len(ap_class_i) == 0:
                    class_ap = 0.
                else:
                    class_ap = np.mean(ap_class_i)
                ap_all_class_test_all[iou_threshold_i].append(class_ap)
            mean_ap_test_all[iou_threshold_i] /= \
                        data_generator.batch_size*data_generator.total_test_batches
        return mean_ap_test_all, ap_all_class_test_all


    def loadDataForPlot(sequences):
        """ Load data one by one for generating evaluation images """
        for sequence_num in sequences:
            ### load RAD input ###
            RAD_complex = loader.readRAD(config_data["input_dir"], sequence_num, \
                                        config_data["input_name_format"])

            ### NOTE: real time visualization ###
            interpolation = 1
            RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                            power_order=2), target_axis=-1), scalar=10, log_10=True)
            RD = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                            power_order=2), target_axis=1), scalar=10, log_10=True)
            RA_cart = helper.toCartesianMask(RA, config_radar, \
                                    gapfill_interval_num=interpolation)
            RA_img = helper.norm2Image(RA)[..., :3]
            RD_img = helper.norm2Image(RD)[..., :3]
            RA_cart_img = helper.norm2Image(RA_cart)[..., :3]
            stereo_left_image = loader.readStereoLeft(config_data["image_dir"], \
                                                    sequence_num, \
                                                    config_data["image_name_format"])

            RAD_data = helper.complexTo2Channels(RAD_complex)
            gt_instances = loader.readRadarInstances(config_data["gt_dir"], sequence_num, \
                                                    config_data["gt_name_format"])
            gt_labels, has_label, raw_boxes = data_generator.encodeToLabels(gt_instances)
            gt_labels_cart, has_label_cart, raw_boxes_cart = \
                                    data_generator.encodeToCartBoxesLabels(gt_instances)

            data = tf.expand_dims(tf.constant(RAD_data, dtype=tf.float32), axis=0)
            label = tf.expand_dims(tf.constant(gt_labels[0], dtype=tf.float32), axis=0)
            label = tf.where(label==0, 1e-10, label)
            label_cart = tf.expand_dims(tf.constant(gt_labels_cart, dtype=tf.float32), \
                                                                            axis=0)
            label_cart = tf.where(label_cart==0, 1e-10, label_cart)
            raw_boxes = tf.expand_dims(tf.constant(raw_boxes, dtype=tf.float32), axis=0)
            raw_boxes_cart = tf.expand_dims(tf.constant(raw_boxes_cart, \
                                                dtype=tf.float32), axis=0)
            yield sequence_num, data, label, label_cart, raw_boxes, raw_boxes_cart, \
                            stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances


    def predictionPlots():
        """ Plot the predictions of all data in dataset """
        if if_evaluate_cart:
            fig, axes = drawer.prepareFigure(4, figsize=(25, 8))
        else:
            fig, axes = drawer.prepareFigure(3, figsize=(20, 8))
        colors = loader.randomColors(config_data["all_classes"])

        image_save_dir = "./images/evaluate_images/"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        else:
            shutil.rmtree(image_save_dir)
            os.makedirs(image_save_dir)
        print("Start plotting, it might take a while...")
        pbar = tqdm(total=len(data_generator.sequences_test))
        for sequence_num, data, label, label_cart, raw_boxes, raw_boxes_cart, \
                stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances in \
                loadDataForPlot(data_generator.sequences_test):
            feature = model(data)
            pred_raw, pred = model.decodeYolo(feature)
            pred_frame = pred[0]
            predicitons = helper.yoloheadToPredictions(pred_frame, conf_threshold=0.5)
            nms_pred = helper.nmsOverClass(predicitons, \
                                    config_evaluate["nms_iou3d_threshold"], \
                                    config_model["input_shape"], \
                                    sigma=0.3, method="nms")
            
            if if_evaluate_cart:
                backbone_fmp = model.backbone_stage(data)
                pred_raw_cart = model_cart(backbone_fmp)
                pred_cart = model_cart.decodeYolo(pred_raw_cart)
                pred_frame_cart = pred_cart[0]
                predicitons_cart = helper.yoloheadToPredictions2D(pred_frame_cart, \
                                                        conf_threshold=0.5)
                nms_pred_cart = helper.nms2DOverClass(predicitons_cart, \
                                        config_model["nms_iou3d_threshold"], \
                                        config_model["input_shape"], \
                                        sigma=0.3, method="nms")
            else:
                nms_pred_cart = None

            drawer.clearAxes(axes)
            drawer.drawRadarPredWithGt(stereo_left_image, RD_img, \
                    RA_img, RA_cart_img, gt_instances, nms_pred, \
                    config_data["all_classes"], colors, axes, \
                    radar_cart_nms=nms_pred_cart)
            drawer.saveFigure(image_save_dir, "%.6d.png"%(sequence_num))
            pbar.update(1)



    ### NOTE: evaluate RAD Boxes under different mAP_iou ###
    all_mean_aps, all_ap_classes = test_step(config_evaluate["mAP_iou3d_threshold"])
    all_mean_aps = np.array(all_mean_aps)
    all_ap_classes = np.array(all_ap_classes)

    table = []
    row = []
    for i in range(len(all_mean_aps)):
        if i == 0:
            row.append("mAP")
        row.append(all_mean_aps[i])
    table.append(row)
    row = []
    for j in range(all_ap_classes.shape[1]):
        ap_current_class = all_ap_classes[:, j]
        for k in range(len(ap_current_class)):
            if k == 0:
                row.append("AP_" + config_data["all_classes"][j])
            row.append(ap_current_class[k])
        table.append(row)
        row = []
    headers = []
    for ap_iou_i in config_evaluate["mAP_iou3d_threshold"]:
        if ap_iou_i == 0:
            headers.append("AP name")
        headers.append("AP_%.2f"%(ap_iou_i))
    print("==================== RAD Boxes AP ========================")
    print(tabulate(table, headers=headers))
    print("==========================================================")


    ### NOTE: evaluate Cart Boxes under different mAP_iou ###
    if if_evaluate_cart:
        all_mean_aps, all_ap_classes = test_step_cart(\
                                config_evaluate["mAP_iou3d_threshold"])
        all_mean_aps = np.array(all_mean_aps)
        all_ap_classes = np.array(all_ap_classes)

        table = []
        row = []
        for i in range(len(all_mean_aps)):
            if i == 0:
                row.append("mAP")
            row.append(all_mean_aps[i])
        table.append(row)
        row = []
        for j in range(all_ap_classes.shape[1]):
            ap_current_class = all_ap_classes[:, j]
            for k in range(len(ap_current_class)):
                if k == 0:
                    row.append("AP_" + config_data["all_classes"][j])
                row.append(ap_current_class[k])
            table.append(row)
            row = []
        headers = []
        for ap_iou_i in config_evaluate["mAP_iou3d_threshold"]:
            if ap_iou_i == 0:
                headers.append("AP name")
            headers.append("AP_%.2f"%(ap_iou_i))
        print("================= Cartesian Boxes AP =====================")
        print(tabulate(table, headers=headers))
        print("==========================================================")


    ### NOTE: plot the predictions on the entire dataset ###
    # predictionPlots()


if __name__ == "__main__":
    main()

