import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
execution_root = os.path.basename(os.getcwd())
if execution_root == "tools":
    os.chdir("..")
sys.path.insert(1, os.getcwd())

import tensorflow as tf
import numpy as np

available_gpus = tf.config.experimental.list_physical_devices('GPU')
if available_gpus:
    tf.config.experimental.set_memory_growth(available_gpus[0], True)

from yolov3.configs import *
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def generate_calibration_set():
    for _ in range(100):
        random_sample = np.random.rand(1, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, 3).astype(np.float32)
        yield (tf.constant(random_sample),)

trt_config = trt.DEFAULT_TRT_CONVERSION_PARAMS
trt_config = trt_config._replace(max_workspace_size_bytes=4_000_000_000)
trt_config = trt_config._replace(precision_mode=YOLO_TRT_QUANTIZE_MODE)
trt_config = trt_config._replace(max_batch_size=1)

if YOLO_TRT_QUANTIZE_MODE == 'INT8':
    trt_config = trt_config._replace(use_calibration=True)

source_model_path = f'./checkpoints/yolov3-{YOLO_INPUT_SIZE}'
destination_model_path = f'./checkpoints/yolov3-trt-{YOLO_TRT_QUANTIZE_MODE}-{YOLO_INPUT_SIZE}'

engine_converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=source_model_path,
    conversion_params=trt_config
)

if YOLO_TRT_QUANTIZE_MODE == 'INT8':
    engine_converter.convert(calibration_input_fn=generate_calibration_set)
else:
    engine_converter.convert()

engine_converter.save(output_saved_model_dir=destination_model_path)

print(f'Done Converting to TensorRT, model saved to: {destination_model_path}')
#