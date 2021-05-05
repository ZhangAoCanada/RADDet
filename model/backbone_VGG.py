# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import tensorflow as tf
import tensorflow.keras as K
import model.layers as L

def twoConvLayersBlock(x, output_channels, strides=(1,1), \
                    use_bias=True, bn=False, if_pool=True):
    """ VGG typical block: 2 convolutional layers + 1 max pool layer """
    assert isinstance(strides, tuple)
    conv = x
    conv = L.convolution2D(conv, output_channels, 3, strides, \
                            "same", "relu", use_bias=use_bias, bn=bn, \
                            if_regularization=True)
    conv = L.convolution2D(conv, output_channels, 3, strides, \
                            "same", "relu", use_bias=use_bias, bn=bn, \
                            if_regularization=True)
    if if_pool:
        conv = L.maxPooling2D(conv)
    return conv

def threeConvLayersBlock(x, output_channels, strides=(1,1), \
                    use_bias=True, bn=False, if_pool=True):
    """ VGG typical block: 3 convolutional layers + 1 max pool layer """
    assert isinstance(strides, tuple)
    conv = x
    conv = L.convolution2D(conv, output_channels, 3, strides, \
                            "same", "relu", use_bias=use_bias, bn=bn, \
                            if_regularization=True)
    conv = L.convolution2D(conv, output_channels, 3, strides, \
                            "same", "relu", use_bias=use_bias, bn=bn, \
                            if_regularization=True)
    conv = L.convolution2D(conv, output_channels, 3, strides, \
                            "same", "relu", use_bias=use_bias, bn=bn, \
                            if_regularization=True)
    if if_pool:
        conv = L.maxPooling2D(conv)
    return conv

def radarVGG3D(x, ):
    """ VGG main (customized) """
    ##### Parameters setup #####
    filters_expansions = [1, 1, 2, 2, 2]

    conv = x
    feature_stages = []
    ##### VGG16 3D #####
    ### Block 1
    conv = twoConvLayersBlock(conv, int(conv.shape[-1]*filters_expansions[0]), \
                                                                if_pool=False)
    ### Block 2
    conv = twoConvLayersBlock(conv, int(conv.shape[-1]*filters_expansions[1]), \
                                                                if_pool=True)
    ### Block 3
    conv = threeConvLayersBlock(conv, int(conv.shape[-1]*filters_expansions[2]), \
                                                                if_pool=True)
    ### Block 4
    conv = threeConvLayersBlock(conv, int(conv.shape[-1]*filters_expansions[3]), \
                                                                if_pool=True)
    ### Block 5
    conv = threeConvLayersBlock(conv, int(conv.shape[-1]*filters_expansions[4]), \
                                                                if_pool=True)

    return conv
