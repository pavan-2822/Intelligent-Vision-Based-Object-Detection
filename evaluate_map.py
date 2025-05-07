import os
import cv2
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Paths
ANNOT_PATH = "annotations/instances_val2017.json"
IMG_DIR = "val2017"
COCO_LABELS = open("model_data/coco.names").read().strip().split("\n")
YOLO_CONFIG = "model_data/yolov.cfg"
YOLO_WEIGHTS = "model_data/yolov3.weights"

# Number of images to evaluate
NUM_IMAGES = 100

# Load ground truth
coco_gt = COCO(ANNOT_PATH)
img_ids = coco_gt.getImgIds()[:NUM_IMAGES]
imgs = coco_gt.loadImgs(img_ids)

# Load EfficientDet
eff_model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

# Load YOLO model
yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)

# Create prediction lists
eff_preds, yolo_preds = [], []

def run_yolo(img, image_id):
    if img is None:
        return []  # Safe return

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    ln = yolo_net.getLayerNames()
    ln = [ln[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    outputs = yolo_net.forward(ln)

    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                cx, cy, bw, bh = det[0:4] * np.array([w, h, w, h])
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    result = []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if indices is not None and len(indices) > 0:
        indices = np.array(indices).flatten()
        for i in indices:
            x, y, bw, bh = boxes[i]
            result.append({
                "image_id": int(image_id),
                "category_id": int(class_ids[i]),
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "score": float(confidences[i])
            })
    return result  # Always return list (even if empty)

def run_efficientdet(image_rgb, image_id):
    if image_rgb is None:
        return []

    if image_rgb.dtype == np.float32 or image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    detections = eff_model(input_tensor)

    boxes = detections['detection_boxes'].numpy()[0]
    scores = detections['detection_scores'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(np.int32)

    results = []
    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.5:
            continue
        ymin, xmin, ymax, xmax = box
        x1, y1 = int(xmin * image_rgb.shape[1]), int(ymin * image_rgb.shape[0])
        x2, y2 = int(xmax * image_rgb.shape[1]), int(ymax * image_rgb.shape[0])
        results.append({
            "image_id": int(image_id),
            "category_id": int(cls),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score)
        })

    return results

for img_meta in tqdm(imgs, desc="Evaluating"):
    image_id = img_meta["id"]
    img_path = os.path.join(IMG_DIR, img_meta["file_name"])
    image = cv2.imread(img_path)

    if image is None:
        print(f"Warning: Could not read image {img_path}")
        continue

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error converting image {img_path}: {e}")
        continue

    eff_result = run_efficientdet(image_rgb, image_id)
    yolo_result = run_yolo(image, image_id)

    if eff_result:
        eff_preds.extend(eff_result)

    if yolo_result:
        yolo_preds.extend(yolo_result)

# Save prediction files
with open("efficientdet_predictions.json", "w") as f:
    json.dump(eff_preds, f)

with open("yolov_predictions.json", "w") as f:
    json.dump(yolo_preds, f)

# Load predictions from files
with open("efficientdet_predictions.json") as f:
    eff_preds_data = json.load(f)

with open("yolov_predictions.json") as f:
    yolo_preds_data = json.load(f)

# Group predictions by image_id
def group_by_image_id(pred_list):
    grouped = {}
    for pred in pred_list:
        img_id = pred["image_id"]
        if img_id not in grouped:
            grouped[img_id] = []
        grouped[img_id].append(pred)
    return grouped

eff_preds_by_img = group_by_image_id(eff_preds_data)
yolo_preds_by_img = group_by_image_id(yolo_preds_data)

# Prepare ground truth for all evaluated images
ground_truth = {}
for img_meta in imgs:
    image_id = img_meta["id"]
    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
    anns = coco_gt.loadAnns(ann_ids)
    ground_truth[image_id] = [
        {"category_id": ann["category_id"], "bbox": ann["bbox"]}
        for ann in anns
    ]

# Your custom evaluation function
def custom_eval(predictions_by_img, ground_truth, model_name="Custom Model"):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for img_id in ground_truth:
        gt_categories = set(gt["category_id"] for gt in ground_truth[img_id])
        pred_categories = set(pred["category_id"] for pred in predictions_by_img.get(img_id, []))

        true_positives += len(gt_categories & pred_categories)
        false_positives += len(pred_categories - gt_categories)
        false_negatives += len(gt_categories - pred_categories)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    print(f"\n===== {model_name} Evaluation =====")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return precision, recall, f1

# Evaluate
custom_eval(eff_preds_by_img, ground_truth, model_name="EfficientDet")

custom_eval(yolo_preds_by_img, ground_truth, model_name="YOLOv3")
