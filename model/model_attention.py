import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import model.layers as L
import util.helper as helper
import model.head_YOLO as yolohead
import model.loss_functions as loss_func

class ROLOSA(K.Model):
    def __init__(self, config_model, config_data, config_train, \
                        anchor_boxes, input_shape):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables 
        could be initialized """
        super(ROLOSA, self).__init__()
        assert (isinstance(input_shape, tuple) or isinstance(input_shape, list))
        ### see if all the anchor numbers equals to the anchors length ###
        assert len(np.unique(config_model["cart_anchor_stages"])) == len(anchor_boxes)

        self.config_model = config_model
        self.config_data = config_data
        self.config_train = config_train
        self.input_size = input_shape
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = anchor_boxes
        self.anchor_num_stages = config_model["anchor_stages"]
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]
        self.features_shape = []
        self.model = self.buildModel()

    def buildModel(self,):
        """ attention: building the model at last few lines of this 
        function is important """
        ### TODO: choose one: ambiguously define the shape or ###
        input_tensor = K.layers.Input(self.input_size)

        ### TODO: self-attention layers ###
        conv = input_tensor
        conv = L.attentionLayer(conv, hidden_channels = int(conv.shape[-1]/2))

        ### TODO: starting buiding 3D box only ###
        neck_stages = [conv]
        yolo_raw = yolohead.yoloHead(neck_stages, self.anchor_boxes, \
                                self.anchor_num_stages, self.num_class)

        for rawyolo_i in yolo_raw:
            self.features_shape.append([s for s in rawyolo_i.shape])
        self.features_shape = np.array(self.features_shape)

        ### necessary step: building the model ###
        model = K.Model(input_tensor, yolo_raw[-1])

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

