#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
# import tensorflow_addons as tfa
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
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

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
            # conv = softplus(conv)
            # conv = conv * tf.math.tanh(tf.math.softplus(conv))
            # conv = conv * tf.tanh(softplus(conv))
            # conv = tf.nn.leaky_relu(conv, alpha=0.1)
            # conv = tfa.activations.mish(conv)
            # conv = conv * tf.nn.tanh(tf.keras.activations.relu(tf.nn.softplus(conv), max_value=20))
            # conv = tf.nn.softplus(conv)
            # conv = tf.keras.activations.relu(tf.nn.softplus(conv), max_value=20)

    return conv
def softplus(x, threshold = 20.):
    def f1():
        return x
    def f2():
        return tf.exp(x)
    def f3():
        return tf.math.log(1 + tf.exp(x))
    # mask = tf.greater(x, threshold)
    # x = tf.exp(x[mask])
    # return tf.exp(x)
    return tf.case([(tf.greater(x, tf.constant(threshold)), lambda:f1()), (tf.less(x, tf.constant(-threshold)), lambda:f2())], default=lambda:f3())
    # return tf.case([(tf.greater(x, threshold), lambda:f1())])
def mish(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)
    # return tf.keras.layers.Lambda(lambda x: softplus(x))(x)
    # return tf.keras.layers.Lambda(lambda x: x * tf.tanh(softplus(x)))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

