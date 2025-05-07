import os
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolov3.configs import *
from yolov3.yolov3 import Create_Yolov3
from tensorflow.python.saved_model import tag_constants

def inject_weights(model, weight_path):
    tf.keras.backend.clear_session()
    scope = 75 if not TRAIN_YOLO_TINY else 13
    skip_bn = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]

    with open(weight_path, 'rb') as wf:
        np.fromfile(wf, dtype=np.int32, count=5)
        j = 0
        for i in range(scope):
            conv_layer = model.get_layer(f'conv2d_{i}' if i > 0 else 'conv2d')
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in skip_bn:
                bn_layer = model.get_layer(f'batch_normalization_{j}' if j > 0 else 'batch_normalization')
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters).reshape((4, filters))[[1, 0, 2, 3]]
                j += 1
            else:
                bias = np.fromfile(wf, dtype=np.float32, count=filters)

            shape = (filters, in_dim, k_size, k_size)
            weights = np.fromfile(wf, dtype=np.float32, count=np.product(shape)).reshape(shape).transpose([2, 3, 1, 0])

            if i not in skip_bn:
                conv_layer.set_weights([weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([weights, bias])
        assert len(wf.read()) == 0

def prepare_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    if YOLO_FRAMEWORK == "tf":
        path = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
        if not YOLO_CUSTOM_WEIGHTS:
            print("Using default weights from:", path)
            model = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
            inject_weights(model, path)
        else:
            custom_path = f"./checkpoints/{TRAIN_MODEL_NAME}{'_Tiny' if TRAIN_YOLO_TINY else ''}"
            print("Using custom weights from:", custom_path)
            model = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            model.load_weights(custom_path)
    else:
        loaded = tf.saved_model.load(YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING])
        model = loaded.signatures['serving_default']
    return model

def scale_image(image, target_size, boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    resized = cv2.resize(image, (nw, nh))
    padded = np.full((ih, iw, 3), 128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    padded[dh:nh + dh, dw:nw + dw] = resized
    result = padded / 255.
    if boxes is not None:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dw
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dh
        return result, boxes
    return result

def decode_classes(path):
    return {i: name.strip() for i, name in enumerate(open(path))}

def display_boxes(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_names=True, show_scores=True, color=(255, 255, 0), outline='', use_id=False):
    label_map = decode_classes(CLASSES)
    height, width, _ = image.shape
    palette = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in map(colorsys.hsv_to_rgb, [(1.0 * i / len(label_map), 1., 1.) for i in range(len(label_map))])]
    random.seed(0)
    random.shuffle(palette)
    for item in bboxes:
        pos = np.array(item[:4], dtype=np.int32)
        score = item[4]
        idx = int(item[5])
        box_color = outline if outline else palette[idx]
        thickness = max(1, int(0.6 * (height + width) / 1000))
        font = 0.75 * thickness
        label = f"{label_map[idx]} {score:.2f}" if show_scores else label_map[idx]
        cv2.rectangle(image, tuple(pos[:2]), tuple(pos[2:]), box_color, thickness * 2)
        if show_names:
            (w, h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, font, thickness)
            cv2.rectangle(image, (pos[0], pos[1]), (pos[0] + w, pos[1] - h - base), box_color, thickness=cv2.FILLED)
            cv2.putText(image, label, (pos[0], pos[1] - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, font, color, thickness, lineType=cv2.LINE_AA)
    return image

def get_iou(set1, set2):
    set1, set2 = np.array(set1), np.array(set2)
    area1 = (set1[..., 2] - set1[..., 0]) * (set1[..., 3] - set1[..., 1])
    area2 = (set2[..., 2] - set2[..., 0]) * (set2[..., 3] - set2[..., 1])
    intersect = np.maximum(set1[..., :2], set2[..., :2]), np.minimum(set1[..., 2:], set2[..., 2:])
    diff = np.maximum(intersect[1] - intersect[0], 0.0)
    inter_area = diff[..., 0] * diff[..., 1]
    union_area = area1 + area2 - inter_area
    return np.maximum(inter_area / union_area, np.finfo(np.float32).eps)

def suppress_boxes(bboxes, threshold, mode='nms', sigma=0.3):
    keep = []
    for c in set(bboxes[:, 5]):
        cls_boxes = bboxes[bboxes[:, 5] == c]
        while len(cls_boxes):
            top = np.argmax(cls_boxes[:, 4])
            best = cls_boxes[top]
            keep.append(best)
            cls_boxes = np.concatenate([cls_boxes[:top], cls_boxes[top+1:]])
            scores = get_iou(best[np.newaxis, :4], cls_boxes[:, :4])
            if mode == 'nms':
                cls_boxes = cls_boxes[scores <= threshold]
            else:
                cls_boxes[:, 4] *= np.exp(-(scores**2 / sigma))
                cls_boxes = cls_boxes[cls_boxes[:, 4] > 0.]
    return keep

def refine_predictions(preds, original, size, thresh):
    preds = np.array(preds)
    pos = np.concatenate([preds[:, :2] - preds[:, 2:] * 0.5, preds[:, :2] + preds[:, 2:] * 0.5], axis=-1)
    h, w = original.shape[:2]
    r = min(size / w, size / h)
    dw, dh = (size - r * w) / 2, (size - r * h) / 2
    pos[:, 0::2] = (pos[:, 0::2] - dw) / r
    pos[:, 1::2] = (pos[:, 1::2] - dh) / r
    pos = np.concatenate([np.maximum(pos[:, :2], 0), np.minimum(pos[:, 2:], [w - 1, h - 1])], axis=-1)
    scale = np.sqrt(np.prod(pos[:, 2:4] - pos[:, 0:2], axis=-1))
    mask = (scale > 0) & (preds[:, 4] * np.max(preds[:, 5:], axis=-1) > thresh)
    classes = np.argmax(preds[:, 5:], axis=-1)
    scores = preds[:, 4] * preds[np.arange(len(preds)), classes + 5]
    return np.concatenate([pos[mask], scores[mask, None], classes[mask, None]], axis=-1)

def detect_frame(model, image_path, output_path, size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 threshold=0.3, overlap=0.45, color=''):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    data = scale_image(np.copy(image), [size, size])[np.newaxis, ...].astype(np.float32)
    pred = model.predict(data) if YOLO_FRAMEWORK == "tf" else [v.numpy() for _, v in model(tf.constant(data)).items()]
    pred = tf.concat([tf.reshape(p, (-1, tf.shape(p)[-1])) for p in pred], axis=0)
    refined = refine_predictions(pred, image, size, threshold)
    boxed = suppress_boxes(refined, overlap)
    result = display_boxes(image, boxed, CLASSES=CLASSES, rectangle_colors=color)
    if output_path: cv2.imwrite(output_path, result)
    if show:
        cv2.imshow("Output", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return result
