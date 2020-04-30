import numpy as np
import tensorflow as tf
import time
import cv2
from core.yolov4 import YOLOv4, YOLOv3_tiny, YOLOv3, decode
from absl import app, flags, logging
from absl.flags import FLAGS
from core import utils
from core.config import cfg

flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_integer('size', 416, 'resize images to')


def main(_argv):
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    input_size = FLAGS.size
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        if FLAGS.tiny:
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_tiny(model, FLAGS.weights)
        else:
            if FLAGS.model == 'yolov3':
                feature_maps = YOLOv3(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights_v3(model, FLAGS.weights)
            elif FLAGS.model == 'yolov4':
                feature_maps = YOLOv4(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights(model, FLAGS.weights)
    logging.info('weights loaded')

    @tf.function
    def run_model(x):
        return model(x)

    # Test the TensorFlow Lite model on random input data.
    sum = 0
    original_image = cv2.imread(FLAGS.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [FLAGS.size, FLAGS.size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    img_raw = tf.image.decode_image(
        open(FLAGS.image, 'rb').read(), channels=3)
    img_raw = tf.expand_dims(img_raw, 0)
    img_raw = tf.image.resize(img_raw, (FLAGS.size, FLAGS.size))
    for i in range(1000):
        prev_time = time.time()
        # pred_bbox = model.predict(image_data)
        pred_bbox = run_model(image_data)
        # pred_bbox = pred_bbox.numpy()
        curr_time = time.time()
        exec_time = curr_time - prev_time
        if i == 0: continue
        sum += (1 / exec_time)
        info = str(i) + " time:" + str(round(exec_time, 3)) + "average FPS:" + str(round(sum / i, 2)) + ", FPS: " + str(
            round((1 / exec_time), 1))
        print(info)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
