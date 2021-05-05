# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model.layers as L

import model.backbone_VGG as vgg
import model.backbone_radarResNet as radarResNet
import model.head_YOLO as yolohead
import model.loss_functions as loss_func

class RADDet(K.Model):
    def __init__(self, config_model, config_data, config_train, anchor_boxes):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables 
        could be initialized """
        super(RADDet, self).__init__()
        assert (isinstance(config_model["input_shape"], tuple) or \
                isinstance(config_model["input_shape"], list))
        self.input_size = list(config_model["input_shape"])
        self.input_channels = self.input_size[-1]
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

        ### NOTE: backbones ###
        # features = vgg.radarVGG3D(conv, )
        features = radarResNet.radarResNet3D(conv, )

        ### NOTE: define backbone model for 2D Boxes ###
        self.backbone_stage = K.Model(input_tensor, features)
        self.backbone_fmp_shape = features.shape[1:]

        yolo_raw = yolohead.yoloHead(features, self.anchor_boxes, self.num_class)

        self.features_shape = np.array([s for s in yolo_raw.shape])

        ### necessary step: building the model ###
        model = K.Model(input_tensor, yolo_raw)
        return model

    def decodeYolo(self, yolo_raw):
        pred_raw, pred = yolohead.boxDecoder(yolo_raw, self.input_size, \
                self.anchor_boxes, self.num_class, self.yolohead_xyz_scales[0])
        return pred_raw, pred

    def loss(self, pred_raw, pred, gt, raw_boxes):
        box_loss, conf_loss, category_loss = loss_func.lossYolo(pred_raw, pred, gt, \
                            raw_boxes, self.input_size, self.focal_loss_iou_threshold)
        box_loss *= 1e-1
        total_loss = box_loss + conf_loss + category_loss
        return total_loss, box_loss, conf_loss, category_loss

    def call(self, x):
        x = self.model(x)
        return x
