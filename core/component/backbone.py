import tensorflow as tf
import core.component.common as common


def csp_darknet53(input_data):
    """
    CSPDarknet53 implementation.

    https://jonathan-hui.medium.com/yolov4-c9901eaa8e61
    """

    input_data = common.convolutional(input_data, (3, 3,  3,  32), activate_type="mish")

    # First downsampling
    input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    for i in range(1):
        input_data = common.residual_block(input_data,  64,  32, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")


    # Second downsampling
    input_data = common.convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
    for i in range(2):
        input_data = common.residual_block(input_data, 64,  64, 64, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")


    # Third downsampling
    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 128, 128, 128, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    route_1 = input_data # output 1


    # Fourth downsampling
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 256, 256, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    route_2 = input_data # output 2


    # Fifth downsampling
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
    for i in range(4):
        input_data = common.residual_block(input_data, 512, 512, 512, activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    # Spatial Pyramid Pooling (SPP Block)
    input_data = tf.concat([
        tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1),
        tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1),
        tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1),
        input_data], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512)) # output 3

    return route_1, route_2, input_data # output 1, output 2, output 3


def csp_darknet53_tiny(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 32), downsample=True)
    input_data = common.convolutional(input_data, (3, 3, 32, 64), downsample=True)
    input_data = common.convolutional(input_data, (3, 3, 64, 64))

    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 64, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = common.convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 256))
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = common.convolutional(input_data, (3, 3, 512, 512))

    return route_1, input_data
