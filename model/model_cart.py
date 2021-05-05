# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model.layers as L
import util.helper as helper


class RADDetCart(K.Model):
    def __init__(self, config_model, config_data, config_train, \
                        anchor_boxes, input_shape):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables 
        could be initialized """
        super(RADDetCart, self).__init__()
        assert (isinstance(input_shape, tuple) or isinstance(input_shape, list))
        self.config_model = config_model
        self.config_data = config_data
        self.config_train = config_train
        self.input_size = input_shape
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = anchor_boxes
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]
        self.model = self.buildModel()

    def buildModel(self,):
        """ attention: building the model at last few lines of this 
        function is important """
        input_tensor = K.layers.Input(self.input_size)
        ### for convenience ###
        conv = input_tensor
        ### NOTE: channel-wise MLP ###
        conv = tf.transpose(conv, perm=[0, 3, 1, 2])
        conv = tf.reshape(conv, [-1, int(conv.shape[1]), \
                                int(conv.shape[2]*conv.shape[3])])

        conv = L.denseLayer(conv, int(conv.shape[-1]*2), if_regularization=False)
        conv = L.denseLayer(conv, int(conv.shape[-1]), if_regularization=False)
        conv = tf.reshape(conv, [-1, int(conv.shape[1]), \
                    int(self.input_size[0]), int(self.input_size[1]*2)])
        conv = tf.transpose(conv, perm=[0, 2, 3, 1])

        ### NOTE: residual block ###
        conv_shortcut = conv
        conv = L.convolution2D(conv, conv.shape[-1], 3, \
                                (1,1), "same", "relu", use_bias=True, bn=True)
        conv = L.convolution2D(conv, conv.shape[-1], 3, \
                                (1,1), "same", "relu", use_bias=True, bn=True)
        conv = L.convolution2D(conv, conv.shape[-1], 1, \
                                (1,1), "same", "relu", use_bias=True, bn=True)
        conv = conv + conv_shortcut

        ### NOTE: yolo head ###
        conv = L.convolution2D(conv, int(conv.shape[-1] * 2), \
                3, (1,1), "same", "relu", use_bias=True, bn=True, \
                if_regularization=False)
        conv = L.convolution2D(conv, len(self.anchor_boxes) * (self.num_class + 5), \
                1, (1,1), "same", None, use_activation=False, use_bias=True, bn=False, \
                if_regularization=False)
        conv = tf.reshape(conv, [-1] + list(conv.shape[1:-1]) + \
                            [len(self.anchor_boxes), self.num_class + 5])
        yolo_raw = conv
        self.features_shape = conv.shape
        print("===== Cartesian YOLO Head =====", yolo_raw.shape)

        ### necessary step: building the model ###
        model = K.Model(input_tensor, yolo_raw)
        return model

    def decodeYolo(self, yolo_raw):
        output_size = [int(self.config_model["input_shape"][0]), \
                        int(2*self.config_model["input_shape"][0])]
        strides = np.array(output_size) / np.array(list(yolo_raw.shape[1:3]))
        raw_xy, raw_wh, raw_conf, raw_prob = tf.split(yolo_raw, \
                                            (2,2,1,self.num_class), axis=-1)

        xy_grid = tf.meshgrid(tf.range(yolo_raw.shape[1]), tf.range(yolo_raw.shape[2]))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
        xy_grid = tf.transpose(xy_grid, perm=[1,0,2,3])
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), \
                        [tf.shape(yolo_raw)[0], 1, 1, len(self.anchor_boxes), 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        scale = self.config_model["yolohead_xyz_scales"][0]
        ### TODO: not sure about this SCALE, but it appears in YOLOv4 tf version ###
        pred_xy = ((tf.sigmoid(raw_xy) * scale) - 0.5 * (scale - 1) + xy_grid) * strides

        ###---------------- clipping values --------------------###
        raw_wh = tf.clip_by_value(raw_wh, 1e-12, 1e12)
        ###-----------------------------------------------------###
        pred_wh = tf.exp(raw_wh) * self.anchor_boxes
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(raw_conf)
        pred_prob = tf.sigmoid(raw_prob)
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def extractYoloInfo(self, yoloformat_data):
        box = yoloformat_data[..., :4]
        conf = yoloformat_data[..., 4:5]
        category = yoloformat_data[..., 5:]
        return box, conf, category

    def loss(self, pred_raw, pred, gt, raw_boxes):
        raw_box, raw_conf, raw_category = self.extractYoloInfo(pred_raw)
        pred_box, pred_conf, pred_category = self.extractYoloInfo(pred)
        gt_box, gt_conf, gt_category = self.extractYoloInfo(gt)

        ### NOTE: box regression (YOLOv1 Loss Function) ###
        box_loss = gt_conf * (tf.square(pred_box[..., :2] - gt_box[..., :2]) + \
                tf.square(tf.sqrt(pred_box[..., 2:]) - tf.sqrt(gt_box[..., 2:])))

        ### NOTE: focal loss function ###
        iou = helper.tf_iou2d(pred_box[:, :, :, :, tf.newaxis, :],\
                    raw_boxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        gt_conf_negative = (1.0 - gt_conf) * tf.cast(max_iou < \
                self.config_train["focal_loss_iou_threshold"], tf.float32)
        conf_focal = tf.pow(gt_conf - pred_conf, 2)
        alpha = 0.01
        conf_loss = conf_focal * (\
                gt_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf, \
                                                                logits=raw_conf) \
                + \
                alpha * gt_conf_negative * \
                        tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_conf, \
                                                                logits=raw_conf))

        ### NOTE: category loss function ###
        category_loss = gt_conf * \
                tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_category, \
                                                        logits=raw_category)
        
        ### NOTE: combine together ###
        box_loss_all = tf.reduce_mean(tf.reduce_sum(box_loss, axis=[1,2,3,4]))
        box_loss_all *= 1e-1
        conf_loss_all = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        category_loss_all = tf.reduce_mean(tf.reduce_sum(category_loss, axis=[1,2,3,4]))
        total_loss = box_loss_all + conf_loss_all + category_loss_all
        return total_loss, box_loss_all, conf_loss_all, category_loss_all

    def call(self, x):
        x = self.model(x)
        return x
