import numpy as np
import tensorflow as tf
import time
import cv2
from core.yolov4 import YOLOv4, decode
from absl import app, flags, logging
from absl.flags import FLAGS
from core import utils

flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_integer('size', 416, 'resize images to')

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    input_layer = tf.keras.layers.Input([FLAGS.size, FLAGS.size, 3])
    feature_maps = YOLOv4(input_layer)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, FLAGS.weights)
    logging.info('weights loaded')

    # Test the TensorFlow Lite model on random input data.
    for i in range(1000):
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

        original_image = cv2.imread(FLAGS.image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = utils.image_preporcess(np.copy(original_image), [FLAGS.size, FLAGS.size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        prev_time = time.time()
        pred_bbox = model.predict(image_data)
        # pred_bbox = pred_bbox.numpy()
        curr_time = time.time()
        exec_time = curr_time - prev_time

        info = "time:" + str(round(1000 * exec_time, 2)) + " ms, FPS: " + str(round((1000 / (1000 * exec_time)), 1))
        print(info)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
