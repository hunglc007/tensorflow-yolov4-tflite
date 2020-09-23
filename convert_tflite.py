import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
import core.utils as utils
import os
from core.config import cfg

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416-fp32.tflite', 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float32', 'quantize mode (int8, float16, float32, mixedint)')
flags.DEFINE_string('dataset', "/Volumes/Elements/data/coco_dataset/coco/5k.txt", 'path to dataset')

def representative_data_gen():
  lines = open(FLAGS.dataset).read().split("\n")
  line = 0
  found = 0
  samples = 10
  
  for input_value in range(samples):
    line += 1
    file = lines[input_value].split(" ")[0]

    if os.path.exists(file):
      original_image = cv2.imread(file)
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
      img_in = image_data[np.newaxis, ...].astype(np.float32)
      print("Reading calibration image {}".format(file))
      found += 1
      yield [img_in]
    else:
      print("File does not exist %s in %s at line %d" % (file, FLAGS.dataset, line))
      continue

  if found < samples:
    raise ValueError("Failed to read %d calibration sample images from %s" % (samples, FLAGS.dataset))


def save_tflite():
  model = tf.keras.models.load_model(FLAGS.weights)
  model.compile()
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  if FLAGS.quantize_mode == 'float32':
    pass
  elif FLAGS.quantize_mode == 'float16':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_types = [tf.float16]
  elif FLAGS.quantize_mode == 'int8':
    # https://www.tensorflow.org/lite/performance/post_training_quantization#integer_only
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tf.uint8
    #converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_data_gen
  elif FLAGS.quantize_mode == 'mixedint':
    # https://www.tensorflow.org/lite/performance/post_training_quantization#integer_only_16-bit_activations_with_8-bit_weights_experimental
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    #converter.inference_input_type = tf.int16
    #converter.inference_output_type = tf.int16
    converter.representative_dataset = representative_data_gen
  else:
    raise ValueError("Unkown quantize_mode: " + str(FLAGS.quantize_mode))

  tflite_model = converter.convert()
  open(FLAGS.output, 'wb').write(tflite_model)

  logging.info("model saved to: {}".format(FLAGS.output))

def demo():
  interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
  interpreter.allocate_tensors()
  logging.info('tflite model loaded')

  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  input_shape = input_details[0]['shape']

  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

  print(output_data)

def main(_argv):
  save_tflite()
  demo()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


