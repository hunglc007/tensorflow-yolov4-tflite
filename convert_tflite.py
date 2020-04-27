import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
from core.yolov4 import YOLOv4, decode
import core.utils as utils
import os

flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './data/yolov4.tflite', 'path to output')
flags.DEFINE_boolean('tiny', False, 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', "int8", 'quantize mode (int8, float16, full_int8)')
flags.DEFINE_string('dataset', "/media/user/Source/Data/coco_dataset/coco/5k.txt", 'path to dataset')

def representative_data_gen():
  fimage = open(FLAGS.dataset).read().split()
  for input_value in range(5):
    if os.path.exists(fimage[input_value]):
      original_image=cv2.imread(fimage[input_value])
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      image_data = utils.image_preporcess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
      img_in = image_data[np.newaxis, ...].astype(np.float32)
      print(input_value)
      yield [img_in]
    else:
      continue

# def apply_quantization_to_dense(layer):
#   # print(layer.name)
#   if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization,
#                         tf.keras.layers.ZeroPadding2D, tf.keras.layers.ReLU)):
#     print(layer.name)
#     return tfmot.quantization.keras.quantize_annotate_layer(layer)
#   return layer

def save_tflite():
  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  if FLAGS.tiny:
    feature_maps = YOLOv4(input_layer)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
      bbox_tensor = decode(fm, i)
      bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()
    utils.load_weights_tiny(model, FLAGS.weights)
  else:
    feature_maps = YOLOv4(input_layer)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
      bbox_tensor = decode(fm, i)
      bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()
    utils.load_weights(model, FLAGS.weights)



  # annotated_model = tf.keras.models.clone_model(
  #   model,
  #   clone_function=apply_quantization_to_dense,
  # )
  # quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
  # quant_aware_model.summary()

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  if FLAGS.quantize_mode == 'int8':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.default_ranges_stats = (0, 6)
    # converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
    # converter.output_format = tf.compat.v1.lite.constants.TFLITE
    # converter.allow_custom_ops = True
    # converter.quantized_input_stats = {"input0": (0., 1.)}

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.inference_input_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
    # converter.inference_output_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8

    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
  elif FLAGS.quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
  elif FLAGS.quantize_mode == 'full_int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_data_gen

  tflite_model = converter.convert()
  open(FLAGS.output, 'wb').write(tflite_model)

  # tflite_model = converter.convert()
  # tf.GFile(FLAGS.output, "wb").write(tflite_model)

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


