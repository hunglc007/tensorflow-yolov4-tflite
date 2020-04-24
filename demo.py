from core.yolov4 import YOLOv4, decode
import tensorflow as tf
import core.utils as utils

input_layer = tf.keras.layers.Input([608, 608, 3])
feature_maps = YOLOv4(input_layer)
model = tf.keras.Model(input_layer, feature_maps)
utils.load_weights(model, "./data/yolov4.weights")
model.summary()