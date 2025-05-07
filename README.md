Intelligent Vision Based Object Detection

This project implements object detection using the YOLOv3 model. It enables detection of objects images using  weights and evaluates model performance using the COCO validation dataset.


Requirements
------------
Install dependencies using:

    pip install -r requirements.txt

Dataset Setup
-------------
To evaluate YOLOv3 on the COCO dataset:

1. Download COCO 2017 Validation Images
   - URL: https://cocodataset.org/#download
   - File: val2017.zip
   - Unzip into your project directory as:
     project-root/val2017/

2. Download COCO 2017 Annotations
   - File: annotations_trainval2017.zip
   - Extract only the file:
     annotations/instances_val2017.json

Running Inference
-----------------
To detect objects in images using YOLOv3:

    python detection_model_yolov3.py

You’ll be prompted to select an image, and the script will display detection results.

Evaluation
----------
To evaluate the model using COCO metrics (Precision, Recall, F1-score):

    python evaluate_map.py

Ensure that val2017/ and annotations/instances_val2017.json are present as described.

Notes
-----
- Model weights and configuration files (yolov3.weights, yolov.cfg, coco.names) must be inside the model_data/ directory.
- This implementation uses OpenCV for inference and TensorFlow for evaluation.
