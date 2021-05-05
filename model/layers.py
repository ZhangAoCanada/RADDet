# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import os


def denseLayer(input_tensor, layer_units, activation="relu", use_bias=True, \
                if_regularization=True):
    """ Dense layer for Cartesian box transformation 
    Args:
        input_tensor        ->          input tensor, [None, ..., inputshape]
        layer_units         ->          filters size, int"""
    assert isinstance(layer_units, int)

    ### NOTE: add regularizer to all layers for reducing overfitting ###
    if if_regularization:
        kernel_regularizer = K.regularizers.L2(1e-2)
        if use_bias:
            bias_regularizer = K.regularizers.L2(1e-2)
        else:
            bias_regularizer = None
    else:
        kernel_regularizer = None
        bias_regularizer = None

    output_tensor = K.layers.Dense(layer_units, activation="relu", use_bias=use_bias, \
            kernel_initializer='glorot_uniform', bias_initializer='zeros', \
            kernel_regularizer=kernel_regularizer, \
            bias_regularizer=bias_regularizer)(input_tensor)
    return output_tensor


def convolution1D(input_tensor, filters, kernel_size, strides, padding, \
                    activation, use_activation=True, use_bias=True, bn=True,\
                    if_regularization=True):
    """ 1D convolutional layer in this research 
    Args:
        input_tensor        ->          input 1D tensor, [None, w, h, d, channels]
        filters             ->          output channels, int
        kernel_size         ->          kernel size, int
        strides             ->          strides, tuple, (strides, strides)
        padding             ->          "same" or "valid", no captical letters
        activation          ->          "relu" or "mish", TODO: leaky_relu
    """
    assert isinstance(kernel_size, int) 
    assert isinstance(filters, int)
    assert isinstance(strides, int)
    assert (padding == "same" or padding == "valid")

    ### NOTE: add regularizer to all layers for reducing overfitting ###
    if if_regularization:
        kernel_regularizer = K.regularizers.L2(1e-2)
        if use_bias:
            bias_regularizer = K.regularizers.L2(1e-2)
        else:
            bias_regularizer = None
    else:
        kernel_regularizer = None
        bias_regularizer = None

    conv_output = K.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
        activation=None, input_shape=input_tensor.shape[-2:], use_bias=use_bias,
        kernel_initializer='glorot_uniform', bias_initializer='zeros',
        kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer,)(input_tensor)

    if bn: conv_output = BatchNormalization()(conv_output)
 
    if use_activation:
        if activation == "relu":
            conv_output = tf.nn.relu(conv_output)
        elif activation  == "mish":
            conv_output = mishActivation(conv_output)
        else:
            raise ValueError("Wrong input activation mode")
    return conv_output


def convolution2D(input_tensor, filters, kernel_size, strides, padding, \
                    activation, use_activation=True, use_bias=True, bn=True,\
                    if_regularization=False):
    """ 2D convolutional layer in this research 
    Args:
        input_tensor        ->          input 2D tensor, [None, w, h, channels]
        filters             ->          output channels, int
        kernel_size         ->          kernel size, int
        strides             ->          strides, tuple, (strides, strides)
        padding             ->          "same" or "valid", no captical letters
        activation          ->          "relu" or "mish", TODO: leaky_relu
    """
    assert isinstance(kernel_size, int) 
    assert isinstance(filters, int)
    assert isinstance(strides, tuple)
    assert len(strides) == 2
    assert (padding == "same" or padding == "valid")

    ### NOTE: add regularizer to all layers for reducing overfitting ###
    if if_regularization:
        kernel_regularizer = K.regularizers.L2(1e-2)
        if use_bias:
            bias_regularizer = K.regularizers.L2(1e-2)
        else:
            bias_regularizer = None
    else:
        kernel_regularizer = None
        bias_regularizer = None

    conv_output = K.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
        activation=None, use_bias=use_bias, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer,)(input_tensor)

    if bn: conv_output = BatchNormalization()(conv_output)
    
    if use_activation:
        if activation == "relu":
            conv_output = tf.nn.relu(conv_output)
        elif activation  == "mish":
            conv_output = mishActivation(conv_output)
        else:
            raise ValueError("Wrong input activation mode")
    return conv_output


def convolution3D(input_tensor, filters, kernel_size, strides, padding, \
                    activation, use_activation=True, use_bias=True, bn=True):
    """ 3D convolutional layer in this research 
    Args:
        input_tensor        ->          input 3D tensor, [None, w, h, d, channels]
        filters             ->          output channels, int
        kernel_size         ->          kernel size, int
        strides             ->          strides, tuple, (strides, strides, strides)
        padding             ->          "same" or "valid", no captical letters
        activation          ->          "relu" or "mish", TODO: leaky_relu
    """
    assert isinstance(kernel_size, int) 
    assert isinstance(filters, int)
    assert isinstance(strides, tuple)
    assert len(strides) == 3
    assert (padding == "same" or padding == "valid")
    ##### NOTE: batch_normalization and use_bias cannot co-exist #####
    # if bn: use_bias = False

    conv_output = K.layers.Conv3D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
        activation=None, use_bias=use_bias, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
        )(input_tensor)

    if bn: conv_output = BatchNormalization()(conv_output)
    
    if use_activation:
        if activation == "relu":
            conv_output = tf.nn.relu(conv_output)
        elif activation  == "mish":
            conv_output = mishActivation(conv_output)
        else:
            raise ValueError("Wrong input activation mode")
    return conv_output


def maxPooling2D(input_tensor, strides=(2,2), padding="same", pool_size=(2,2)):
    """ Max pooling layer in this research (for 2D tensors) """
    assert isinstance(strides, tuple)
    assert len(strides) == 2
    assert all(isinstance(stride, int) for stride in strides)
    assert isinstance(pool_size, tuple)
    assert len(pool_size) == 2
    assert all(isinstance(pool, int) for pool in pool_size)
    assert (padding == "same" or padding == "valid")

    pool_output = tf.keras.layers.MaxPool2D(
                    pool_size=pool_size, strides=strides, padding=padding
                    )(input_tensor)
    return pool_output


def convPooling2D(input_tensor, use_bias=True, bn=True):
    """ Try Pooling on the Azimuth dimension and conv2D on range dimension """
    output_tensor = tf.keras.layers.MaxPool2D(
                    pool_size=(1,2), strides=(1,2), padding="same"
                    )(input_tensor)
    output_tensor = convolution2D(output_tensor, output_tensor.shape[-1], 3, \
                    (2,1), "same", "relu", use_bias=use_bias, bn=bn, \
                    if_regularization=False)
    return output_tensor


def maxPooling3D(input_tensor, strides=(2,2,2), padding="same", pool_size=(2,2,2)):
    """ Max pooling layer in this research (for 3D tensors) """
    assert isinstance(strides, tuple)
    assert len(strides) == 3
    assert all(isinstance(stride, int) for stride in strides)
    assert isinstance(pool_size, tuple)
    assert len(pool_size) == 3
    assert all(isinstance(pool, int) for pool in pool_size)
    assert (padding == "same" or padding == "valid")

    pool_output = tf.keras.layers.MaxPool3D(
                    pool_size=pool_size, strides=strides, padding=padding
                    )(input_tensor)
    return pool_output


def attentionLayer(conv, hidden_channels=None, use_bias=True):
    """ Self-Attention layers, reference: SA-GAN """
    print("[INFO]: using self-attention layer.""")
    if hidden_channels is None: hidden_channels = conv.shape[-1]
    input_shape = conv.shape

    query = convolution2D(conv, hidden_channels, 1, \
                                (1,1), "same", "relu", use_bias=use_bias, \
                                bn=False, if_regularization=False)

    key = convolution2D(conv, hidden_channels, 1, \
                                (1,1), "same", "relu", use_bias=use_bias, \
                                bn=False, if_regularization=False)

    value = convolution2D(conv, conv.shape[-1], 1, \
                                (1,1), "same", "relu", use_bias=use_bias, \
                                bn=False, if_regularization=False)
    value = conv

    query = tf.reshape(query, [-1, input_shape[1]*input_shape[2], hidden_channels])
    key = tf.reshape(key, [-1, input_shape[1]*input_shape[2], hidden_channels])
    value = tf.reshape(value, [-1, input_shape[1]*input_shape[2], input_shape[3]])

    scores = tf.matmul(query, key, transpose_b=True)
    attention = tf.nn.softmax(scores)

    output_tensor = tf.matmul(attention, value, transpose_a=True)
    output_tensor = tf.reshape(output_tensor, \
                    [-1, input_shape[1], input_shape[2], input_shape[3]])

    output_tensor = convolution2D(output_tensor, output_tensor.shape[-1], 1, \
                                (1,1), "same", "relu", use_bias=use_bias, \
                                bn=False, if_regularization=False)

    return output_tensor


def attentionLayerUnet(conv, hidden_channels=None):
    """ Reference: Attention U-Net """
    print("[INFO]: using self-attention (Attention U-Net) layer.""")
    if hidden_channels is None: hidden_channels = conv.shape[-1]
    input_shape = conv.shape
    ### NOTE: define theta ###
    theta_conv = convolution2D(conv, hidden_channels, 3, \
                                (1,1), "same", "relu", use_activation=False, \
                                use_bias=False, \
                                bn=False, if_regularization=False)
    theta_conv_shape = theta_conv.shape
    ### NOTE: define gating ###
    gating = convolution2D(conv, int(conv.shape[-1]), 3, \
                                (1,1), "same", "relu", use_activation=False, \
                                use_bias=True, \
                                bn=True, if_regularization=False)
    # gating = maxPooling2D(conv)
    ### NOTE: define phi ###
    phi_gating = convolution2D(gating, hidden_channels, 1, \
                                (1,1), "same", "relu", use_activation=False, \
                                use_bias=True, \
                                bn=False, if_regularization=False)
    # phi_gating = K.layers.UpSampling2D(size=(2,2))(phi_gating)
    ### NOTE: combine theta, phi ###
    f = tf.nn.relu(theta_conv + phi_gating)
    psi_f = convolution2D(f, 1, 1, \
                        (1,1), "same", "relu", use_activation=False, \
                        use_bias=True, \
                        bn=False, if_regularization=False)
    sigmoid_psi_f = tf.math.sigmoid(psi_f)
    ### NOTE: multiply score with input ###
    y = sigmoid_psi_f * conv
    y = convolution2D(y, int(conv.shape[-1]), 1, \
                                (1,1), "same", "relu", use_activation=False, \
                                use_bias=True, \
                                bn=True, if_regularization=False)
    return y


##### NOTE: tf.nn.convolution has no use_bias term #####
class customizeConv(K.layers.Layer):
    """ Building this class for adding Variables into trainable_variables """
    def __init__(self, kernel_size, strides, padding):
        super(customizeConv, self).__init__()
        self.initializer = tf.initializers.GlorotNormal()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.getVariables()

    def getVariables(self, ):
        self.kernel = tf.Variable(
                    initial_value=self.initializer(self.kernel_size), 
                    trainable=True)

    def call(self, x):
        conv_output = tf.nn.convolution(x, self.kernel, \
                        strides = self.strides, padding = self.padding)
        return conv_output


def convolution2Don3D(input_tensor, filters, kernel_size, strides, padding, \
                    activation, axis_remain, bn=True):
    """ customized convolutional layer """
    """ convolution on 2 dimensions of a 3D tensor """
    assert isinstance(kernel_size, int) 
    assert isinstance(filters, int)
    assert isinstance(strides, int)
    assert isinstance(axis_remain, int)
    assert axis_remain < 3
    assert (padding == "same" or padding == "valid")
    if padding == "same":
        padding = "SAME"
    else:
        padding = "VALID"

    kernel_shape = []
    strides_full = []
    for i in range(3):
        if i == axis_remain:
            kernel_shape.append(1)
            strides_full.append(1)
        else:
            kernel_shape.append(kernel_size)
            strides_full.append(strides)
    kernel_shape += [input_tensor.shape[-1], filters]
    conv_layer = customizeConv(kernel_shape, strides_full, padding)
    conv_output = conv_layer(input_tensor)

    if bn: conv_output = BatchNormalization()(conv_output)
    
    if activation == "relu":
        conv_output = tf.nn.relu(conv_output)
    elif activation  == "mish":
        conv_output = mishActivation(conv_output)
    else:
        raise ValueError("Wrong input activation mode")
    return conv_output


def convolution1Don3D(input_tensor, filters, kernel_size, strides, padding, \
                    activation, axis_work, bn=True):
    """ customized convolutional layer """
    """ convolution on 1 dimension of a 3D tensor """
    assert isinstance(kernel_size, int) 
    assert isinstance(filters, int)
    assert isinstance(strides, int)
    assert isinstance(axis_work, int)
    assert axis_work < 3
    assert (padding == "same" or padding == "valid")
    if padding == "same":
        padding = "SAME"
    else:
        padding = "VALID"

    kernel_shape = []
    strides_full = []
    for i in range(3):
        if i != axis_work:
            kernel_shape.append(1)
            strides_full.append(1)
        else:
            kernel_shape.append(kernel_size)
            strides_full.append(strides)
    kernel_shape += [input_tensor.shape[-1], filters]
    conv_layer = customizeConv(kernel_shape, strides_full, padding)
    conv_output = conv_layer(input_tensor)

    if bn: conv_output = BatchNormalization()(conv_output)
    
    if activation == "relu":
        conv_output = tf.nn.relu(conv_output)
    elif activation  == "mish":
        conv_output = mishActivation(conv_output)
    else:
        raise ValueError("Wrong input activation mode")
    return conv_output


def nearestUpsample3D(input_tensor):
    """ Double the size of the input tensor """
    ##### NOTE: seems implementation of bilinear on 3D is impractical #####
    return K.layers.UpSampling3D(size=(2, 2, 2))(input_tensor)


def bilinearUpsample2Don3D(input_tensor):
    """ Using tf.image.resize for bilinear interpolation """
    """ ATTENTION: bilinear seems only works on 2D image """
    ##### NOTE: this bilinaer upsampling works on axies [1,2] in [0,1,2,3,4] #####
    input_shape = input_tensor.shape
    switch_axes = [0,3,1,2,4]
    reverse_axes = [0,2,3,1,4]
    switched_shape = [input_shape[i] for i in switch_axes]
    switched_shape = [switched_shape[0]*switched_shape[1], switched_shape[2], \
                        switched_shape[3], switched_shape[4]]
    reverse_shape = [input_shape[i] for i in switch_axes]
    reverse_shape[2] *= 2
    reverse_shape[3] *= 2
    transposed = tf.transpose(input_tensor, switch_axes)
    reshaped = tf.reshape(transposed, switched_shape)
    resized = tf.image.resize(reshaped, \
                            (reshaped.shape[1]*2, reshaped.shape[2]*2), \
                            method="bilinear")
    undo_reshape = tf.reshape(resized, reverse_shape)
    undo_transpose = tf.transpose(undo_reshape, reverse_axes)
    return undo_transpose


def mishActivation(x):
    """ Mish Activation Function """
    return x * tf.math.tanh(tf.math.softplus(x))


class BatchNormalization(K.layers.BatchNormalization):
    """
    REFERENCE: comes from YOLOv4.
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
