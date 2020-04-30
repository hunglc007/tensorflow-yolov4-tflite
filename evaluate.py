from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode

flags.DEFINE_string('weights', './data/yolov3.weights',
                    'path to weights file')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite)'
                    'path to weights file')
flags.DEFINE_string('model', 'yolov3', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 512, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/dataset/val2017.txt", 'annotation path')
flags.DEFINE_string('write_image_path', "./data/detection/", 'write image path')

def main(_argv):
    INPUT_SIZE = FLAGS.size
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
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

    else:
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)

    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            if FLAGS.framework == "tf":
                pred_bbox = model.predict(image_data)
            else:
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3':
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
            elif FLAGS.model == 'yolov4':
                XYSCALE = cfg.YOLO.XYSCALE
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=XYSCALE)

            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
            bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

            if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                image = utils.draw_bbox(image, bboxes)
                cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image)

            with open(predict_result_path, 'w') as f:
                for bbox in bboxes:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


