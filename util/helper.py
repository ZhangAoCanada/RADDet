# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import mixture

################ coordinates transformation ################
def cartesianToPolar(x, y):
    """ Cartesian to Polar """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polarToCartesian(rho, phi):
    """ Polar to Cartesian """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

################ functions of RAD processing ################
def complexTo2Channels(target_array):
    """ transfer complex a + bi to [a, b]"""
    assert target_array.dtype == np.complex64
    ### NOTE: transfer complex to (magnitude) ###
    output_array = getMagnitude(target_array)
    output_array = getLog(output_array)
    return output_array

def getMagnitude(target_array, power_order=2):
    """ get magnitude out of complex number """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """ get Log values """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array

def getSumDim(target_array, target_axis):
    """ sum up one dimension """
    output = np.sum(target_array, axis=target_axis)
    return output 

def switchCols(target_array, cols):
    """ switch columns """
    assert isinstance(cols, tuple) or isinstance(cols, list)
    assert len(cols) == 2
    assert np.max(cols) <= target_array.shape[-1] - 1
    cols = np.sort(cols)
    output_axes = []
    for i in range(target_array.shape[-1]):
        if i == cols[0]:
            idx = cols[1]
        elif i == cols[1]:
            idx = cols[0]
        else:
            idx = i
        output_axes.append(idx)
    return target_array[..., output_axes]

def switchAxes(target_array, axes):
    """ switch axes """
    assert isinstance(axes, tuple) or isinstance(axes, list)
    assert len(axes) == 2
    assert np.max(axes) <= len(target_array.shape) - 1
    return np.swapaxes(target_array, axes[0], axes[1])

def norm2Image(array):
    """ normalize to image format (uint8) """
    norm_sig = plt.Normalize()
    img = plt.cm.viridis(norm_sig(array))
    img *= 255.
    img = img.astype(np.uint8)
    return img

def toCartesianMask(RA_mask, radar_config, gapfill_interval_num=1):
    """ transfer RA mask to Cartesian mask for plotting """
    output_mask = np.ones([RA_mask.shape[0], RA_mask.shape[0]*2]) * np.amin(RA_mask)
    point_angle_previous = None
    for i in range(RA_mask.shape[0]):
        for j in range(1, RA_mask.shape[1]):
            if RA_mask[i, j] > 0:
                point_range = ((radar_config["range_size"]-1) - i) * \
                                radar_config["range_resolution"]
                point_angle = (j * (2*np.pi/radar_config["azimuth_size"]) - np.pi) / \
                                (2*np.pi*0.5*radar_config["config_frequency"]/ \
                                radar_config["designed_frequency"])
                point_angle_current = np.arcsin(point_angle)
                if point_angle_previous is None:
                    point_angle_previous = point_angle_current
                for point_angle in np.linspace(point_angle_previous, point_angle_current, \
                                                gapfill_interval_num):
                    point_zx = polarToCartesian(point_range, point_angle)
                    new_i = int(output_mask.shape[0] - \
                            np.round(point_zx[0]/radar_config["range_resolution"])-1)
                    new_j = int(np.round((point_zx[1]+50)/radar_config["range_resolution"])-1)
                    output_mask[new_i,new_j] = RA_mask[i, j] 
                point_angle_previous = point_angle_current
    return output_mask

def GaussianModel(pcl):
    """ Get the center and covariance from gaussian model. """
    model = mixture.GaussianMixture(n_components=1, covariance_type='full')
    model.fit(pcl)
    return model.means_[0], model.covariances_[0]

################ ground truth manipulation ################
def boxLocationsToWHD(boxes):
    """ Transfer boxes from [x_min, x_max, y_min, y_max, z_min, z_max] to
    [x_center, y_center, z_center, width, height, depth] """
    new_boxes = np.zeros(boxes.shape)
    if len(boxes.shape) == 2:
        assert boxes.shape[-1] == 6
        new_boxes[:, 0] = np.round((boxes[:, 0] + boxes[:, 1]) / 2)
        new_boxes[:, 1] = np.round((boxes[:, 2] + boxes[:, 3]) / 2)
        new_boxes[:, 2] = np.round((boxes[:, 4] + boxes[:, 5]) / 2)
        new_boxes[:, 3] = np.round(boxes[:, 1] - boxes[:, 0])
        new_boxes[:, 4] = np.round(boxes[:, 3] - boxes[:, 2])
        new_boxes[:, 5] = np.round(boxes[:, 5] - boxes[:, 4])
        return new_boxes
    elif len(boxes.shape) == 1:
        assert boxes.shape[0] == 6
        new_boxes[0] = np.round((boxes[0] + boxes[1]) / 2)
        new_boxes[1] = np.round((boxes[2] + boxes[3]) / 2)
        new_boxes[2] = np.round((boxes[4] + boxes[5]) / 2)
        new_boxes[3] = np.round(boxes[1] - boxes[0])
        new_boxes[4] = np.round(boxes[3] - boxes[2])
        new_boxes[5] = np.round(boxes[5] - boxes[4])
        return new_boxes
    else:
        raise ValueError("Wrong input boxes, please check the input")
  
def iou2d(box_xywh_1, box_xywh_2):
    """ Numpy version of 3D bounding box IOU calculation 
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou

def iou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Numpy version of 3D bounding box IOU calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = np.array([0, 0, input_size[2]/2])
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou
 
def giou3d(box_xyzwhd_1, box_xyzwhd_2):
    """ Numpy version of 3D bounding box GIOU (Generalized IOU) calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert len(box_xyzwhd_1.shape) == 2
    assert len(box_xyzwhd_2.shape) == 2
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    ### areas of both boxes
    box1_area = box_xyzwhd_1[:, 3] * box_xyzwhd_1[:, 4] * box_xyzwhd_1[:, 5]
    box2_area = box_xyzwhd_2[:, 3] * box_xyzwhd_2[:, 4] * box_xyzwhd_2[:, 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[:, :3] - box_xyzwhd_1[:, 3:] * 0.5
    box1_max = box_xyzwhd_1[:, :3] + box_xyzwhd_1[:, 3:] * 0.5
    box2_min = box_xyzwhd_2[:, :3] - box_xyzwhd_2[:, 3:] * 0.5
    box2_max = box_xyzwhd_2[:, :3] + box_xyzwhd_2[:, 3:] * 0.5
    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get normal area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[:, 0] * intersection[:, 1] * intersection[:, 2]
    union_area = box1_area + box2_area - intersection_area
    iou = np.nan_to_num(intersection_area / union_area)
    ### get enclose area
    enclose_left_top = np.minimum(box1_min, box2_min)
    enclose_bottom_right = np.maximum(box1_max, box2_max)
    enclose_section = enclose_bottom_right - enclose_left_top
    enclose_area = enclose_section[:, 0] * enclose_section[:, 1] * enclose_section[:, 2]
    ### get giou
    giou = iou - np.nan_to_num((enclose_area - union_area) / (enclose_area + 1e-10))
    return giou
  
def tf_iou2d(box_xywh_1, box_xywh_2):
    """ Tensorflow version of 3D bounding box IOU calculation 
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    left_top = tf.maximum(box1_min, box2_min)
    bottom_right = tf.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = tf.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = tf.math.divide_no_nan(intersection_area, union_area + 1e-10)
    return iou

def tf_iou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Tensorflow version of 3D bounding box IOU calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = [0, 0, input_size[2]/2]
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    left_top = tf.maximum(box1_min, box2_min)
    bottom_right = tf.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = tf.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = tf.math.divide_no_nan(intersection_area, union_area + 1e-10)
    return iou
 
def tf_giou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Tensorflow version of 3D bounding box GIOU (Generalized IOU) calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = np.array([0, 0, input_size[2]/2])
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    left_top = tf.maximum(box1_min, box2_min)
    bottom_right = tf.minimum(box1_max, box2_max)
    ### get normal area
    intersection = tf.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    union_area = box1_area + box2_area - intersection_area
    iou = tf.math.divide_no_nan(intersection_area, union_area)
    ### get enclose area
    enclose_left_top = tf.minimum(box1_min, box2_min)
    enclose_bottom_right = tf.maximum(box1_max, box2_max)
    enclose_section = enclose_bottom_right - enclose_left_top
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1] * enclose_section[..., 2]
    ### get giou
    giou = iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area + 1e-10)
    return giou

def smoothOnehot(class_num, hm_classes, smooth_coef=0.01):
    """ Transfer class index to one hot class (smoothed) """
    assert isinstance(class_num, int)
    assert isinstance(hm_classes, int)
    assert class_num < hm_classes
    ### building onehot 
    onehot = np.zeros(hm_classes, dtype=np.float32) 
    onehot[class_num] = 1.
    ### smoothing onehot
    uniform_distribution = np.full(hm_classes, 1.0/hm_classes)
    smooth_onehot = (1-smooth_coef) * onehot + smooth_coef * uniform_distribution
    return smooth_onehot

def yoloheadToPredictions(yolohead_output, conf_threshold=0.5):
    """ Transfer YOLO HEAD output to [:, 8], where 8 means
    [x, y, z, w, h, d, score, class_index]"""
    prediction = yolohead_output.numpy().reshape(-1, yolohead_output.shape[-1])
    prediction_class = np.argmax(prediction[:, 7:], axis=-1)
    predictions = np.concatenate([prediction[:, :7], \
                    np.expand_dims(prediction_class, axis=-1)], axis=-1)
    conf_mask = (predictions[:, 6] >= conf_threshold)
    predictions = predictions[conf_mask]
    return predictions

def yoloheadToPredictions2D(yolohead_output, conf_threshold=0.5):
    """ Transfer YOLO HEAD output to [:, 6], where 6 means
    [x, y, w, h, score, class_index]"""
    prediction = yolohead_output.numpy().reshape(-1, yolohead_output.shape[-1])
    prediction_class = np.argmax(prediction[:, 5:], axis=-1)
    predictions = np.concatenate([prediction[:, :5], \
                    np.expand_dims(prediction_class, axis=-1)], axis=-1)
    conf_mask = (predictions[:, 4] >= conf_threshold)
    predictions = predictions[conf_mask]
    return predictions

def nms(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        all_pred_classes = list(set(bboxes[:, 7]))
        unique_classes = list(np.unique(all_pred_classes))
        best_bboxes = []
        for cls in unique_classes:
            cls_mask = (bboxes[:, 7] == cls)
            cls_bboxes = bboxes[cls_mask]
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 6])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = iou3d(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], \
                            input_size)
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes

def nmsOverClass(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        best_bboxes = []
        ### NOTE: start looping over boxes to find the best one ###
        while len(bboxes) > 0:
            max_ind = np.argmax(bboxes[:, 6])
            best_bbox = bboxes[max_ind]
            best_bboxes.append(best_bbox)
            bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
            iou = iou3d(best_bbox[np.newaxis, :6], bboxes[:, :6], \
                        input_size)
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            bboxes[:, 6] = bboxes[:, 6] * weight
            score_mask = bboxes[:, 6] > 0.
            bboxes = bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes

def nms2D(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, w, h, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 6])
    else:
        all_pred_classes = list(set(bboxes[:, 5]))
        unique_classes = list(np.unique(all_pred_classes))
        best_bboxes = []
        for cls in unique_classes:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], \
                                            cls_bboxes[max_ind + 1:]])
                iou = iou2d(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 6])
    return best_bboxes

def nms2DOverClass(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, w, h, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 6])
    else:
        best_bboxes = []
        ### NOTE: start looping over boxes to find the best one ###
        while len(bboxes) > 0:
            max_ind = np.argmax(bboxes[:, 4])
            best_bbox = bboxes[max_ind]
            best_bboxes.append(best_bbox)
            bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
            iou = iou2d(best_bbox[np.newaxis, :4], bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            bboxes[:, 4] = bboxes[:, 4] * weight
            score_mask = bboxes[:, 4] > 0.
            bboxes = bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 6])
    return best_bboxes

