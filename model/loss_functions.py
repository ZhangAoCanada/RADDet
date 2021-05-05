# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model.layers as L
import util.helper as helper


def extractYoloInfo(yolo_output_format_data):
    """ Extract box, objectness, class from yolo output format data """
    box = yolo_output_format_data[..., :6]
    conf = yolo_output_format_data[..., 6:7]
    category = yolo_output_format_data[..., 7:]
    return box, conf, category
        
def yolo1Loss(pred_box, gt_box, gt_conf, input_size, if_box_loss_scale=True):
    """ loss function for box regression \cite{YOLOV1} """
    assert pred_box.shape == gt_box.shape
    if if_box_loss_scale:
        scale = 2.0 - 1.0 * gt_box[..., 3:4] * gt_box[..., 4:5] * gt_box[..., 5:6] /\
                                    (input_size[0] * input_size[1] * input_size[2])
    else:
        scale = 1.0
    ### NOTE: YOLOv1 original loss function ###
    giou_loss = gt_conf * scale * (tf.square(pred_box[..., :3] - gt_box[..., :3]) + \
                    tf.square(tf.sqrt(pred_box[..., 3:]) - tf.sqrt(gt_box[..., 3:])))
    return giou_loss

def focalLoss(raw_conf, pred_conf, gt_conf, pred_box, raw_boxes, input_size, \
            iou_loss_threshold=0.5):
    """ Calculate focal loss for objectness """
    iou = helper.tf_iou3d(pred_box[:, :, :, :, :, tf.newaxis, :], \
                    raw_boxes[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :],\
                    input_size) 
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    gt_conf_negative = (1.0 - gt_conf) * tf.cast(max_iou < iou_loss_threshold, tf.float32)
    conf_focal = tf.pow(gt_conf - pred_conf, 2)
    alpha = 0.01

    ###### TODO: think, whether we have to seperate logits with decoded outputs #######
    focal_loss = conf_focal * (\
            gt_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf, \
                                                            logits=raw_conf) \
            + \
            alpha * gt_conf_negative * \
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf, \
                                                            logits=raw_conf) \
            )
    return focal_loss

def categoryLoss(raw_category, pred_category, gt_category, gt_conf):
    """ Category Cross Entropy loss """
    category_loss = gt_conf * \
        tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_category, logits=raw_category)
    return category_loss

def lossYolo(pred_raw, pred, label, raw_boxes, input_size, focal_loss_iou_threshold):
    """ Calculate loss function of YOLO HEAD 
    Args:
        feature_stages      ->      3 different feature stages after YOLO HEAD
                                    with shape [None, r, a, d, num_anchors, 7+num_class]
        gt_stages           ->      3 different ground truth stages 
                                    with shape [None, r, a, d, num_anchors, 7+num_class]"""
    assert len(raw_boxes.shape) == 3
    input_size = tf.cast(input_size, tf.float32)
    assert pred_raw.shape == label.shape
    assert pred_raw.shape[0] == len(raw_boxes)
    assert pred.shape == label.shape
    assert pred.shape[0] == len(raw_boxes)
    raw_box, raw_conf, raw_category = extractYoloInfo(pred_raw)
    pred_box, pred_conf, pred_category = extractYoloInfo(pred)
    gt_box, gt_conf, gt_category = extractYoloInfo(label)
    giou_loss = yolo1Loss(pred_box, gt_box, gt_conf, input_size, \
                            if_box_loss_scale=False)
    focal_loss = focalLoss(raw_conf, pred_conf, gt_conf, pred_box, raw_boxes, \
                            input_size, focal_loss_iou_threshold)
    category_loss = categoryLoss(raw_category, pred_category, gt_category, gt_conf)
    giou_total_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_total_loss = tf.reduce_mean(tf.reduce_sum(focal_loss, axis=[1,2,3,4]))
    category_total_loss = tf.reduce_mean(tf.reduce_sum(category_loss, axis=[1,2,3,4]))
    return giou_total_loss, conf_total_loss, category_total_loss

