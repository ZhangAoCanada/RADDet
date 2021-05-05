# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model.layers as L
import util.helper as helper


def singleLayerHead(feature_map, num_anchors_layer, num_class, last_channel):
    """ YOLO HEAD for one specific feature stage (after FPN) """
    assert isinstance(num_anchors_layer, int)
    assert isinstance(num_class, int)
    ### NOTE: 7 means [objectness, x, y, z, w, h ,d]
    final_output_channels = int(last_channel * num_anchors_layer * (num_class + 7))
    final_output_reshape = [-1] + list(feature_map.shape[1:-1]) + \
                        [int(last_channel), int(num_anchors_layer) * (num_class + 7)]
    ### NOTE: either use attention or not ###
    conv = feature_map
    ### NOTE: size up the channels is the way how YOLOv4 did it,
    ### other options may also be worth trying ###
    conv = L.convolution2D(conv, feature_map.shape[-1]*2, \
            3, (1,1), "same", "relu", use_bias=True, bn=True, \
            if_regularization=False)

    conv = L.convolution2D(conv, final_output_channels, \
            1, (1,1), "same", None, use_activation=False, use_bias=True, bn=False, \
            if_regularization=False)
    conv = tf.reshape(conv, final_output_reshape)
    return conv


def boxDecoder(yolohead_output, input_size, anchors_layer, num_class, scale=1.):
    """ Decoder output from yolo head to boxes """
    grid_size = yolohead_output.shape[1:4]
    num_anchors_layer = len(anchors_layer)
    grid_strides = np.array(input_size) / np.array(list(grid_size))
    reshape_size = [tf.shape(yolohead_output)[0]] + list(grid_size) + \
                    [num_anchors_layer, 7+num_class]
    reshape_size = tuple(reshape_size)
    pred_raw = tf.reshape(yolohead_output, reshape_size)
    raw_xyz, raw_whd, raw_conf, raw_prob = tf.split(pred_raw, \
                                        (3,3,1,num_class), axis=-1)

    xyz_grid = tf.meshgrid(tf.range(grid_size[0]), \
                            tf.range(grid_size[1]), \
                            tf.range(grid_size[2]))
    xyz_grid = tf.expand_dims(tf.stack(xyz_grid, axis=-1), axis=3)
    ### NOTE: swap axes seems necessary, don't know why ###
    xyz_grid = tf.transpose(xyz_grid, perm=[1,0,2,3,4])
    xyz_grid = tf.tile(tf.expand_dims(xyz_grid, axis=0), \
                    [tf.shape(yolohead_output)[0], 1, 1, 1,  len(anchors_layer), 1])
    xyz_grid = tf.cast(xyz_grid, tf.float32)

    ### NOTE: not sure about this SCALE, but it appears in YOLOv4 tf version ###
    pred_xyz = ((tf.sigmoid(raw_xyz) * scale) - 0.5 * (scale - 1) + xyz_grid) * \
                grid_strides

    ###---------------- clipping values --------------------###
    raw_whd = tf.clip_by_value(raw_whd, 1e-12, 1e12)
    ###-----------------------------------------------------###
    pred_whd = tf.exp(raw_whd) * anchors_layer
    pred_xyzwhd = tf.concat([pred_xyz, pred_whd], axis=-1)

    pred_conf = tf.sigmoid(raw_conf)
    pred_prob = tf.sigmoid(raw_prob)
    return pred_raw, tf.concat([pred_xyzwhd, pred_conf, pred_prob], axis=-1)


def yoloHead(feature, anchors, num_class):
    """ YOLO HEAD main 
    Args:
        feature_stages      ->      feature stages after FPN, [big, mid, small]
        anchor_stages       ->      how many anchors for each stage, 
                                    e.g. [[0,1], [2,3], [4,5]]
        num_class           ->      number of all the classes
    """
    anchor_num = len(anchors)
    yolohead_raw = singleLayerHead(feature, anchor_num, num_class, \
                                    int(feature.shape[1]/4))
    return yolohead_raw
