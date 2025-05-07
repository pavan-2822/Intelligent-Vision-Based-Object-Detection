import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

cosmos_root = os.path.basename(os.getcwd())
if cosmos_root == "tools":
    os.chdir("..")
sys.path.insert(1, os.getcwd())

import tensorflow as tf
from yolov3.yolov4 import Launch_StellarNet
from yolov3.utils import inject_darkmatter
from yolov3.configs import *

nebula_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

if not YOLO_CUSTOM_WEIGHTS:
    model = Launch_StellarNet(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
    inject_darkmatter(model, nebula_weights)
else:
    model = Launch_StellarNet(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    model.load_weights(YOLO_CUSTOM_WEIGHTS)

model.summary()
model.save(f'./checkpoints/yolov3-{YOLO_INPUT_SIZE}')

print(f"model saves to /checkpoints/yolov3-{YOLO_INPUT_SIZE}")
#