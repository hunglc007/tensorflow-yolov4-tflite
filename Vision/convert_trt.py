from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import cv2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import core.utils as utils
from tensorflow.python.saved_model import signature_constants
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-trt-fp16-416', 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float16', 'quantize mode (int8, float16)')
flags.DEFINE_string('dataset', "/media/user/Source/Data/coco_dataset/coco/5k.txt", 'path to dataset')
flags.DEFINE_integer('loop', 8, 'loop')

def representative_data_gen():
  fimage = open(FLAGS.dataset).read().split()
  batched_input = np.zeros((FLAGS.loop, FLAGS.input_size, FLAGS.input_size, 3), dtype=np.float32)
  for input_value in range(FLAGS.loop):
    if os.path.exists(fimage[input_value]):
      original_image=cv2.imread(fimage[input_value])
      original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
      image_data = utils.image_preporcess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
      img_in = image_data[np.newaxis, ...].astype(np.float32)
      batched_input[input_value, :] = img_in
      # batched_input = tf.constant(img_in)
      print(input_value)
      # yield (batched_input, )
      # yield tf.random.normal((1, 416, 416, 3)),
    else:
      continue
  batched_input = tf.constant(batched_input)
  yield (batched_input,)

def save_trt():

  if FLAGS.quantize_mode == 'int8':
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
      precision_mode=trt.TrtPrecisionMode.INT8,
      max_workspace_size_bytes=4000000000,
      use_calibration=True,
      max_batch_size=8)
    converter = trt.TrtGraphConverterV2(
      input_saved_model_dir=FLAGS.weights,
      conversion_params=conversion_params)
    converter.convert(calibration_input_fn=representative_data_gen)
  elif FLAGS.quantize_mode == 'float16':
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
      precision_mode=trt.TrtPrecisionMode.FP16,
      max_workspace_size_bytes=4000000000,
      max_batch_size=8)
    converter = trt.TrtGraphConverterV2(
      input_saved_model_dir=FLAGS.weights, conversion_params=conversion_params)
    converter.convert()
  else :
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
      precision_mode=trt.TrtPrecisionMode.FP32,
      max_workspace_size_bytes=4000000000,
      max_batch_size=8)
    converter = trt.TrtGraphConverterV2(
      input_saved_model_dir=FLAGS.weights, conversion_params=conversion_params)
    converter.convert()

  # converter.build(input_fn=representative_data_gen)
  converter.save(output_saved_model_dir=FLAGS.output)
  print('Done Converting to TF-TRT')

  saved_model_loaded = tf.saved_model.load(FLAGS.output)
  graph_func = saved_model_loaded.signatures[
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
  trt_graph = graph_func.graph.as_graph_def()
  for n in trt_graph.node:
    print(n.op)
    if n.op == "TRTEngineOp":
      print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
    else:
      print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))
  logging.info("model saved to: {}".format(FLAGS.output))

  trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
  print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
  all_nodes = len([1 for n in trt_graph.node])
  print("numb. of all_nodes in TensorRT graph:", all_nodes)

def main(_argv):
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
  save_trt()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


