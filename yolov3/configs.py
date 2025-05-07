YOLO_TYPE                   = "yolov3"
YOLO_FRAMEWORK              = "tf"
YOLO_V3_WEIGHTS             = "model_data/yolov3.weights"
YOLO_V3_TINY_WEIGHTS        = "model_data/yolov3-tiny.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8"
YOLO_CUSTOM_WEIGHTS         = False
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416

YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156,198], [373,326]]]

TRAIN_YOLO_TINY             = False
TRAIN_SAVE_BEST_ONLY        = True
TRAIN_SAVE_CHECKPOINT       = False
TRAIN_CLASSES               = "mnist/mnist.names"
TRAIN_ANNOT_PATH            = "mnist/mnist_train.txt"
TRAIN_LOGDIR                = "log"
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}_custom"
TRAIN_LOAD_IMAGES_TO_RAM    = True
TRAIN_BATCH_SIZE            = 4
TRAIN_INPUT_SIZE            = 416
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = False
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 100

TEST_ANNOT_PATH             = "mnist/mnist_test.txt"
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45

if TRAIN_YOLO_TINY:
    YOLO_STRIDES = [16, 32]
    YOLO_ANCHORS = [[[10, 14], [23, 27], [37, 58]],
                    [[81, 82], [135, 169], [344, 319]]]
