import numpy as np
import tensorflow as tf
import time
import cv2
from core.yolov4 import YOLOv4, YOLOv3_tiny, YOLOv3, decode
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from core import utils
from core.config import cfg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_integer('size', 416, 'resize images to')


def main(_argv):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
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
    elif FLAGS.framework == 'trt':
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        print(signature_keys)
        infer = saved_model_loaded.signatures['serving_default']

    logging.info('weights loaded')

    @tf.function
    def run_model(x):
        return model(x)

    # Test the TensorFlow Lite model on random input data.
    sum = 0
    original_image = cv2.imread(FLAGS.image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.size, FLAGS.size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    img_raw = tf.image.decode_image(
        open(FLAGS.image, 'rb').read(), channels=3)
    img_raw = tf.expand_dims(img_raw, 0)
    img_raw = tf.image.resize(img_raw, (FLAGS.size, FLAGS.size))
    batched_input = tf.constant(image_data)
    for i in range(1000):
        prev_time = time.time()
        # pred_bbox = model.predict(image_data)
        if FLAGS.framework == 'tf':
            pred_bbox = []
            result = run_model(image_data)
            for value in result:
                value = value.numpy()
                pred_bbox.append(value)
            if FLAGS.model == 'yolov4':
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            else:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
            bboxes = utils.nms(bboxes, 0.213, method='nms')
        elif FLAGS.framework == 'trt':
            pred_bbox = []
            result = infer(batched_input)
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
            if FLAGS.model == 'yolov4':
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            else:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
            bboxes = utils.nms(bboxes, 0.213, method='nms')
        # pred_bbox = pred_bbox.numpy()
        curr_time = time.time()
        exec_time = curr_time - prev_time
        if i == 0: continue
        sum += (1 / exec_time)
        info = str(i) + " time:" + str(round(exec_time, 3)) + " average FPS:" + str(round(sum / i, 2)) + ", FPS: " + str(
            round((1 / exec_time), 1))
        print(info)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
