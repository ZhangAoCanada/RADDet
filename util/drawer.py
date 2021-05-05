# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os
import cv2
import numpy as np
import pickle
from glob import glob
import random
import colorsys

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle

from skimage.measure import find_contours
import util.helper as helper

################ opencv realted drawing  ################
def RandomColors(N, bright=True): 
    """ Define colors for all categories. """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(1111)
    random.shuffle(colors)
    return colors

def applyMask(image, mask, color, alpha=0.5):
    """ Apply the given mask to the image. """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                 image[:, :, c] *
                                (1 - alpha) + alpha * color[c] * 255,
                                 image[:, :, c])
    return image

################ Matplotlib.pyplot plotting ################
def prepareFigure(num_axes, figsize=None):
    """ define figures and subplots """
    assert num_axes <= 6
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    if num_axes == 1:
        ax1 = fig.add_subplot(111)
        return fig, [ax1]
    if num_axes == 2: 
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        return fig, [ax1, ax2]
    if num_axes == 3:
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        return fig, [ax1, ax2, ax3]
    if num_axes == 4:
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)
        return fig, [ax1, ax2, ax3, ax4]
    if num_axes == 5:
        raise ValueError("Adding 5 subplots is not easy to orgnize.")
    if num_axes == 6:
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)
        return fig, [ax1, ax2, ax3, ax4, ax5, ax6]

def clearAxes(ax_list):
    """ clear axes for continuous drawing """
    assert len(ax_list) >=1 
    plt.cla()
    for ax_i in ax_list:
        ax_i.clear()

def drawContour(mask, axe, color):
    """ Draw mask contour onto the image. """
    mask_padded = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
    mask_padded[1:-1, 1:-1] = mask
    contours = find_contours(mask_padded, 0.1, fully_connected='low')
    for verts in contours:
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        axe.add_patch(p)

def linePlot(x_list, y_list, ax, color_list=None, label_list=None, \
            xlimits=None, ylimits=None, title=None, \
            xlabel=None, ylabel=None, linestyle="solid", linewidth=2):
    """ Plot lines using x y values """
    if color_list == None:
        color_list = ["blue" for _ in range(len(y_list))]
    if label_list is not None:
        assert len(x_list) == len(y_list)
        assert len(x_list) == len(label_list)
        for i in range(len(x_list)):
            x, y, label, color = x_list[i], y_list[i], label_list[i], color_list[i]
            assert len(x) == len(y)
            ax.plot(x, y, color=color, label=label, \
                    linestyle=linestyle, linewidth=linewidth)
        if len(y_list) > 1:
            ax.legend()
    else:
        x, y, label, color = x_list[0], y_list[0], None, color_list[0]
        assert len(x) == len(y)
        ax.plot(x, y, color=color, label=label, \
                linestyle=linestyle, linewidth=linewidth)
    if xlimits is not None:
        ax.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

def barChart(x, y, ax, title=None, xlabel=None, ylabel=None, color=None, \
            align="center", width=0.8, xtricks=None, ytricks=None, \
            vertical=True, show_numbers=False):
    """ Plot lines using x y values """
    if color == None: color="blue"
    if vertical:
        ax.bar(x, y, color=color, width=width, align=align)
    else:
        ax.barh(y, x, color=color, height=width, align=align)
    if xlabel is not None:
        if xtricks is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(xtricks)
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        if ytricks is not None:
            ax.set_yticks(y)
            ax.set_yticklabels(ytricks)
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if show_numbers:
        if vertical:
            for i, v in enumerate(y):
                if v == np.amax(x):
                    ax.text(i+0, 0.9*v, str(v), color='black', fontweight='bold')
                else:
                    ax.text(i+0, 1.05*v, str(v), color='black', fontweight='bold')
        else:
            for i, v in enumerate(x):
                if v == np.amax(x):
                    ax.text(0.9*v, i+0, str(v), color='black', fontweight='bold')
                else:
                    ax.text(1.05*v, i+0, str(v), color='black', fontweight='bold')

def histChart(x, bins, range, ax, title=None, xlabel=None, ylabel=None, \
                color=None, xticks=None, xticklabels=None):
    """ Plot histogram using matplotlib """
    ax.hist(x, bins=bins, range=range, color=color)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)

def pclScatter(pcl_list, color_list, label_list, ax, xlimits, ylimits, title):
    """ scatter point cloud """
    assert len(pcl_list) == len(color_list)
    if label_list is not None:
        assert len(pcl_list) == len(label_list)
    for i in range(len(pcl_list)):
        pcl = pcl_list[i]
        color = color_list[i]
        if label_list == None:
            label = None
        else:
            label = label_list[i]
        ax.scatter(pcl[:, 0], pcl[:, 1], s=1, c=color, label=label)
    if xlimits is not None:
        ax.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)
    if title is not None:
        ax.set_title(title)

def imgPlot(img, ax, cmap, alpha, title=None):
    """ image plotting (customized when plotting RAD) """
    ax.imshow(img, cmap=cmap, alpha=alpha)
    if title == "RD":
        title = "Range-Doppler"
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_xticklabels([-13, -6.5, 0, 6.5, 13])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("velocity (m/s)")
        ax.set_ylabel("range (m)")
    elif title == "RA":
        title = "Range-Azimuth"
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels([-85.87, -42.93, 0, 42.93, 85.87])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("angle (degrees)")
        ax.set_ylabel("range (m)")
    elif title == "Cartesian":
        ax.set_xticks([0, 128, 256, 384, 512])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    else:
        ax.axis('off')
    if title is not None:
        ax.set_title(title)

def keepDrawing(fig, time_duration):
    """ keep drawing frames """
    fig.canvas.draw()
    plt.pause(time_duration)

def saveFigure(save_dir, name):
    """ save the figure """
    plt.savefig(os.path.join(save_dir, name))

def mask2BoxOrEllipse(mask, mode="box"):
    """ Find bounding box from mask. """
    idxes = []
    output = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                idxes.append([i, j])
    current_cluster = np.array(idxes)
    if mode == "box":
        x_min = np.amin(current_cluster[:, 0])
        x_max = np.amax(current_cluster[:, 0])
        y_min = np.amin(current_cluster[:, 1])
        y_max = np.amax(current_cluster[:, 1])
        output.append([(x_min+x_max)/2, (y_min+y_max)/2, \
                        x_max-x_min+1, y_max-y_min+1])
        output = np.array(output)
        return output
    elif mode == "ellipse":
        cluster_mean, cluster_cov = helper.GaussianModel(current_cluster)
        output.append([cluster_mean, cluster_cov])
        return output
    else:
        raise ValueError("Wrong input parameter ------ mode")

def getEllipse(color, means, covariances, scale_factor=1):
    """ Draw 2D Gaussian Ellipse. """
    sign = np.sign(means[0] / means[1])
    eigen, eigen_vec = np.linalg.eig(covariances)

    eigen_root_x = np.sqrt(eigen[0]) * scale_factor
    eigen_root_y = np.sqrt(eigen[1]) * scale_factor
    theta = np.degrees(np.arctan2(*eigen_vec[:,0][::-1]))

    ell = Ellipse(xy = (means[0], means[1]), width = eigen_root_x,
                height = eigen_root_y, angle = theta, \
                facecolor = 'none', edgecolor = color)
    return ell

def drawBoxOrEllipse(inputs, class_name, axe, color, x_shape=0, mode="box", \
                    if_facecolor=False):
    """ Draw bounding box onto the image. """
    if if_facecolor: 
        facecolor = color
    else:
        facecolor = "none"
    if mode == "box":
        for box in inputs:
            # y1, y2, x1, x2 = box
            y_c, x_c, h, w = box
            y1, y2, x1, x2 = int(y_c-h/2), int(y_c+h/2), int(x_c-w/2), int(x_c+w/2)
            if x1 < 0:
                x1 += x_shape
                r = Rectangle((x1, y1), x_shape - x1, y2 - y1, linewidth=1.5,
                            alpha=0.5, linestyle="dashed", edgecolor=color,
                            facecolor=facecolor)
                axe.add_patch(r)
                r = Rectangle((0, y1), x2 - 0, y2 - y1, linewidth=1.5,
                            alpha=0.5, linestyle="dashed", edgecolor=color,
                            facecolor=facecolor)
                axe.add_patch(r)
            elif x2 >= x_shape:
                x2 -= x_shape
                r = Rectangle((x1, y1), x_shape - x1, y2 - y1, linewidth=1.5,
                            alpha=0.5, linestyle="dashed", edgecolor=color,
                            facecolor=facecolor)
                axe.add_patch(r)
                r = Rectangle((0, y1), x2 - 0, y2 - y1, linewidth=1.5,
                            alpha=0.5, linestyle="dashed", edgecolor=color,
                            facecolor=facecolor)
                axe.add_patch(r)
            else:
                r = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                            alpha=0.5, linestyle="dashed", edgecolor=color,
                            facecolor=facecolor)
                axe.add_patch(r)

        axe.text(x1+1, y1-3, class_name, size=10, verticalalignment='baseline',
                color='w', backgroundcolor="none",
                bbox={'facecolor': color, 'alpha': 0.5,
                    'pad': 2, 'edgecolor': 'none'})
    elif mode == "ellipse":
        for e in inputs:
            mean, cov = e[0], e[1]
            mean = np.flip(mean)
            cov = np.flip(cov)
            x1, y1 = mean
            ell = getEllipse(color, mean, cov, scale_factor=5)
            axe.add_patch(ell)
        axe.text(x1, y1, class_name, size=5, verticalalignment='center',
                color='w', backgroundcolor="none",
                bbox={'facecolor': color, 'alpha': 0.5,
                        'pad': 2, 'edgecolor': 'none'})
    else:
        raise ValueError("Wrong input parameter ------ mode")


def drawRadarInstances(RD_img, RA_img, RA_cart_img, radar_instances, radar_config, \
                        all_classes, colors, axes):
    """ draw all the masks, boxes on the input images """
    assert len(radar_instances["masks"]) == len(radar_instances["classes"])
    assert len(axes) == 3
    for i in range(len(radar_instances["classes"])):
        mask = radar_instances["masks"][i]
        bbox3d = radar_instances["boxes"][i]
        if len(mask[mask>0]) > 0:
            cls = radar_instances["classes"][i]
            color = colors[all_classes.index(cls)]
            mask_RD = np.where(helper.getSumDim(mask, 1) >= 1, 1, 0)
            mask_RA = np.where(helper.getSumDim(mask, -1) >= 1, 1, 0)
            mask_cart = np.where(helper.toCartesianMask(mask_RA, radar_config)>0., 1., 0.)
            applyMask(RD_img, mask_RD, color)
            applyMask(RA_img, mask_RA, color)
            applyMask(RA_cart_img, mask_cart, color)
            ### draw mask contour
            drawContour(mask_RD, axes[0], color)
            drawContour(mask_RA, axes[1], color)
            drawContour(mask_cart, axes[2], color)
            ### draw box
            mode = "box" # either "box" or "ellipse"
            ### get the boxes from masks
            ### if in the future, boxes can be read directly from gt, 
            ### comment these lines
            # RD_box = mask2BoxOrEllipse(mask_RD, mode)
            # RA_box = mask2BoxOrEllipse(mask_RA, mode)
            ### boxes information added in the ground truth dictionary
            RD_box = np.array([[bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]]])
            RA_box = np.array([[bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]]])
            RA_cart_box = mask2BoxOrEllipse(mask_cart, mode)
            ### draw boxes
            drawBoxOrEllipse(RD_box, cls, axes[0], color, \
                            x_shape=RD_img.shape[1], mode=mode)
            drawBoxOrEllipse(RA_box, cls, axes[1], color, \
                            x_shape=RA_img.shape[1], mode=mode)
            drawBoxOrEllipse(cart_box, cls, axes[2], color, \
                            x_shape=RA_cart_img.shape[1], mode=mode)

    imgPlot(RD_img, axes[0], None, 1, "RD") 
    imgPlot(RA_img, axes[1], None, 1, "RA") 
    imgPlot(RA_cart_img, axes[2], None, 1, "RA mask in cartesian") 


def drawRadarBoxes(stereo_left_image, RD_img, RA_img, RA_cart_img, \
                    radar_instances, all_classes, colors, axes):
    """ draw only boxes on the input images """
    assert len(radar_instances["boxes"]) == len(radar_instances["classes"])
    assert len(axes) == 4
    for i in range(len(radar_instances["classes"])):
        bbox3d = radar_instances["boxes"][i]
        cls = radar_instances["classes"][i]
        cart_box = radar_instances["cart_boxes"][i]
        color = colors[all_classes.index(cls)]
        ### draw box
        mode = "box" # either "box" or "ellipse"
        ### boxes information added in the ground truth dictionary
        RD_box = np.array([[bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]]])
        RA_box = np.array([[bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]]])
        cart_box = np.array([cart_box])
        ### draw boxes
        drawBoxOrEllipse(RD_box, cls, axes[1], color, \
                        x_shape=RD_img.shape[1], mode=mode)
        drawBoxOrEllipse(RA_box, cls, axes[2], color, \
                        x_shape=RA_img.shape[1], mode=mode)
        drawBoxOrEllipse(cart_box, cls, axes[3], color, \
                        x_shape=RA_cart_img.shape[1], mode=mode)

    imgPlot(stereo_left_image, axes[0], None, None, "camera")
    imgPlot(RD_img, axes[1], None, 1, "RD") 
    imgPlot(RA_img, axes[2], None, 1, "RA") 
    imgPlot(RA_cart_img, axes[3], None, 1, "Cartesian") 


def drawRadarPredWithGt(stereo_left_image, RD_img, RA_img, RA_cart_img, \
                        radar_instances, radar_nms_pred, all_classes, colors, axes, \
                        radar_cart_nms=None):
    """ draw only boxes on the input images """
    assert len(radar_instances["boxes"]) == len(radar_instances["classes"])
    color_black = [0., 0., 0.]
    for i in range(len(radar_instances["classes"])):
        bbox3d = radar_instances["boxes"][i]
        cls = radar_instances["classes"][i]
        cart_box = radar_instances["cart_boxes"][i]
        color = colors[all_classes.index(cls)]
        ### draw box
        mode = "box" # either "box" or "ellipse"
        ### boxes information added in the ground truth dictionary
        RD_box = np.array([[bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]]])
        RA_box = np.array([[bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]]])
        cart_box = np.array([cart_box])
        ### draw boxes
        drawBoxOrEllipse(RD_box, cls, axes[0], color, \
                        x_shape=RD_img.shape[1], mode=mode, if_facecolor=True)
        drawBoxOrEllipse(RA_box, cls, axes[1], color, \
                        x_shape=RA_img.shape[1], mode=mode, if_facecolor=True)
        if len(axes) > 3:
            drawBoxOrEllipse(cart_box, cls, axes[2], color, \
                        x_shape=RA_cart_img.shape[1], mode=mode, if_facecolor=True)

    for i in range(len(radar_nms_pred)):
        bbox3d = radar_nms_pred[i, :6]
        cls = int(radar_nms_pred[i, 7])
        color = colors[int(cls)]
        ### draw box
        mode = "box" # either "box" or "ellipse"
        ### boxes information added in the ground truth dictionary
        RD_box = np.array([[bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]]])
        RA_box = np.array([[bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]]])
        ### draw boxes
        drawBoxOrEllipse(RD_box, all_classes[cls], axes[0], color, \
                        x_shape=RD_img.shape[1], mode=mode)
        drawBoxOrEllipse(RA_box, all_classes[cls], axes[1], color, \
                        x_shape=RA_img.shape[1], mode=mode)

    if len(axes) > 3 and radar_cart_nms is not None:
        for i in range(len(radar_cart_nms)):
            cart_box = np.expand_dims(radar_cart_nms[i, :4], axis=0)
            cls = int(radar_cart_nms[i, 5])
            class_name = all_classes[cls]
            color = colors[int(cls)]
            mode = "box" # either "box" or "ellipse"
            drawBoxOrEllipse(cart_box, class_name, axes[2], color, \
                                x_shape=RA_cart_img.shape[1], mode=mode)

    imgPlot(stereo_left_image, axes[-1], None, None, "camera")
    imgPlot(RD_img, axes[0], None, 1, "RD") 
    imgPlot(RA_img, axes[1], None, 1, "RA") 
    if len(axes) > 3:
        imgPlot(RA_cart_img, axes[2], None, 1, "Cartesian") 


def drawInference(stereo_left_image, RD_img, RA_img, RA_cart_img, radar_nms_pred, \
                all_classes, colors, axes, radar_cart_nms=None):
    """ draw only boxes on the input images """
    color_black = [0., 0., 0.]
    for i in range(len(radar_nms_pred)):
        bbox3d = radar_nms_pred[i, :6]
        cls = int(radar_nms_pred[i, 7])
        color = colors[int(cls)]
        ### draw box
        mode = "box" # either "box" or "ellipse"
        ### boxes information added in the ground truth dictionary
        RD_box = np.array([[bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5]]])
        RA_box = np.array([[bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4]]])
        ### draw boxes
        drawBoxOrEllipse(RD_box, all_classes[cls], axes[0], color, \
                        x_shape=RD_img.shape[1], mode=mode)
        drawBoxOrEllipse(RA_box, all_classes[cls], axes[1], color, \
                        x_shape=RA_img.shape[1], mode=mode)

    if len(axes) > 3 and radar_cart_nms is not None:
        for i in range(len(radar_cart_nms)):
            cart_box = np.expand_dims(radar_cart_nms[i, :4], axis=0)
            cls = int(radar_cart_nms[i, 5])
            class_name = all_classes[cls]
            color = colors[int(cls)]
            mode = "box" # either "box" or "ellipse"
            drawBoxOrEllipse(cart_box, class_name, axes[2], color, \
                                x_shape=RA_cart_img.shape[1], mode=mode)

    imgPlot(stereo_left_image, axes[-1], None, None, "camera")
    imgPlot(RD_img, axes[0], None, 1, "RD") 
    imgPlot(RA_img, axes[1], None, 1, "RA") 
    if len(axes) > 3:
        imgPlot(RA_cart_img, axes[2], None, 1, "Cartesian") 


