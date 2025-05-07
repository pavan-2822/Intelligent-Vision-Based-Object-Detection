
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization as KerasBN, MaxPool2D
from tensorflow.keras.regularizers import l2
from yolov3.utils import read_class_names
from yolov3.configs import *

# Calculate anchor boxes adjusted by strides
STRIDES = np.array(YOLO_STRIDES)
ANCHORS = (np.array(YOLO_ANCHORS).T / STRIDES).T

# Custom BatchNormalization layer to handle frozen layers during inference
class BatchNormalization(KerasBN):
    def call(self, inputs, training=False):
        """
        Override the call method to support 'frozen' layers during training.

        When self.trainable is False, use stored moving mean and variance,
        and avoid updating gamma/beta during training.
        """
        training = tf.logical_and(training, self.trainable)
        return super().call(inputs, training=training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    """
    Constructs a convolutional block with optional downsampling, batch normalization, and LeakyReLU activation.

    Args:
        input_layer: Input tensor.
        filters_shape: Tuple or list in the form (kernel_size, _, num_filters).
        downsample: Whether to apply downsampling with stride 2.
        activate: Whether to apply LeakyReLU activation.
        bn: Whether to apply BatchNormalization.

    Returns:
        Output tensor after applying Conv2D (+ optional BN and activation).
    """
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        strides = 2
        padding = 'valid'
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(
        filters=filters_shape[-1],
        kernel_size=filters_shape[0],
        strides=strides,
        padding=padding,
        use_bias=not bn,
        kernel_regularizer=l2(0.0005),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.constant_initializer(0.0)
    )(input_layer)

    if bn:
        conv = BatchNormalization()(conv)

    if activate:
        conv = LeakyReLU(alpha=0.1)(conv)

    return conv


def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    """
    Constructs a residual block with two convolutional layers.

    Args:
        input_layer: Input tensor.
        input_channel: Number of channels in the input tensor.
        filter_num1: Number of filters in the first 1x1 convolution.
        filter_num2: Number of filters in the second 3x3 convolution.

    Returns:
        Output tensor after applying the residual connection.
    """
    shortcut = input_layer

    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))

    return shortcut + conv


def upsample(input_layer):
    """
    Upsamples the input tensor by a factor of 2 using nearest neighbor interpolation.

    Args:
        input_layer: Input tensor of shape (batch_size, height, width, channels).

    Returns:
        Upsampled tensor with doubled height and width.
    """
    new_height = input_layer.shape[1] * 2
    new_width = input_layer.shape[2] * 2
    return tf.image.resize(input_layer, (new_height, new_width), method='nearest')


def darknet53(input_data):
    """
    Builds the Darknet-53 backbone network for feature extraction.

    Args:
        input_data: Input image tensor.

    Returns:
        route_1: Feature map from an earlier layer (for small object detection).
        route_2: Intermediate feature map (for medium object detection).
        input_data: Final feature map (for large object detection).
    """
    # Initial layers
    input_data = convolutional(input_data, (3, 3, 3, 32))
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True)

    for _ in range(1):
        input_data = residual_block(input_data, 64, 32, 64)

    input_data = convolutional(input_data, (3, 3, 64, 128), downsample=True)

    for _ in range(2):
        input_data = residual_block(input_data, 128, 64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for _ in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data  # for small object detection

    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for _ in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data  # for medium object detection

    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for _ in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


def darknet19_tiny(input_data):
    """
    Builds the Tiny version of the Darknet-19 backbone.

    Args:
        input_data: Input tensor (image).

    Returns:
        route_1: Intermediate feature map for use in detection layers.
        input_data: Final feature map for object detection.
    """
    # Convolution + MaxPool blocks
    input_data = convolutional(input_data, (3, 3, 3, 16))
    input_data = MaxPool2D(pool_size=2, strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, (3, 3, 16, 32))
    input_data = MaxPool2D(pool_size=2, strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, (3, 3, 32, 64))
    input_data = MaxPool2D(pool_size=2, strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, (3, 3, 64, 128))
    input_data = MaxPool2D(pool_size=2, strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, (3, 3, 128, 256))
    route_1 = input_data  # for detection use
    input_data = MaxPool2D(pool_size=2, strides=2, padding='same')(input_data)
    input_data = convolutional(input_data, (3, 3, 256, 512))
    input_data = MaxPool2D(pool_size=2, strides=1, padding='same')(input_data)
    input_data = convolutional(input_data, (3, 3, 512, 1024))
    return route_1, input_data

def YOLOv3(input_layer, NUM_CLASS):
    """
    Builds the full YOLOv3 detection head on top of the Darknet-53 backbone.

    Args:
        input_layer: Input image tensor.
        NUM_CLASS: Number of object classes.

    Returns:
        List of prediction tensors [conv_sbbox, conv_mbbox, conv_lbbox] for small, medium, and large objects.
    """
    # Backbone feature extraction
    route_1, route_2, conv = darknet53(input_layer)

    # Large object detection branch
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # Medium object detection branch
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = upsample(conv)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # Small object detection branch
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def YOLOv3_tiny(input_layer, NUM_CLASS):
    """
    Builds the YOLOv3-Tiny detection head on top of the Darknet-19-Tiny backbone.

    Args:
        input_layer: Input image tensor.
        NUM_CLASS: Number of object classes.

    Returns:
        List of prediction tensors [conv_mbbox, conv_lbbox] for medium and large objects.
    """
    route_1, conv = darknet19_tiny(input_layer)

    # Large object detection branch
    conv = convolutional(conv, (1, 1, 1024, 256))
    conv_lobj_branch = convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    # Medium object detection branch
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)
    conv = tf.concat([conv, route_1], axis=-1)

    conv_mobj_branch = convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def Create_Yolov3(input_size=416, channels=3, training=False, CLASSES=YOLO_COCO_CLASSES):
    """
    Constructs a complete YOLOv3 or YOLOv3-Tiny Keras model.

    Args:
        input_size: Input image size (assumed square).
        channels: Number of image channels (default 3 for RGB).
        training: Whether the model is being built for training or inference.
        CLASSES: Path to class names file.

    Returns:
        A tf.keras.Model object representing the full YOLOv3 (or Tiny) model.
    """
    NUM_CLASS = len(read_class_names(CLASSES))
    input_layer = Input([input_size, input_size, channels])

    # Choose YOLOv3 or YOLOv3-Tiny based on config
    conv_tensors = YOLOv3_tiny(input_layer, NUM_CLASS) if TRAIN_YOLO_TINY else YOLOv3(input_layer, NUM_CLASS)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, i)
        if training:
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(inputs=input_layer, outputs=output_tensors)
    return model



def decode(conv_output, NUM_CLASS, i=0):
    """
    Decodes YOLO convolutional output into bounding boxes, confidence scores, and class probabilities.

    Args:
        conv_output: Raw output tensor from YOLO head, shape [batch, grid, grid, 3*(5+NUM_CLASS)].
        NUM_CLASS: Number of object classes.
        i: Index of the scale (0=large, 1=medium, 2=small).

    Returns:
        Tensor of shape [batch, grid, grid, 3, 5+NUM_CLASS] containing:
            [x, y, w, h, objectness, class probabilities...]
    """
    conv_shape = tf.shape(conv_output)
    batch_size, output_size = conv_shape[0], conv_shape[1]

    # Reshape to [batch, grid, grid, 3, 5 + NUM_CLASS]
    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # Split raw predictions
    raw_xy   = conv_output[..., 0:2]  # center offset
    raw_wh   = conv_output[..., 2:4]  # width/height offset
    raw_conf = conv_output[..., 4:5]  # objectness score
    raw_prob = conv_output[..., 5:]   # class probabilities

    # Generate grid (for offset decoding)
    grid_y = tf.range(output_size, dtype=tf.int32)
    grid_x = tf.range(output_size, dtype=tf.int32)

    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    xy_grid = tf.stack([grid_x, grid_y], axis=-1)  # shape: [grid, grid, 2]
    xy_grid = tf.expand_dims(xy_grid, axis=2)      # shape: [grid, grid, 1, 2]
    xy_grid = tf.cast(tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1]), tf.float32)

    # Decode coordinates
    pred_xy = (tf.sigmoid(raw_xy) + xy_grid) * STRIDES[i]
    pred_wh = tf.exp(raw_wh) * ANCHORS[i] * STRIDES[i]

    # Concatenate final prediction
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(raw_conf)
    pred_prob = tf.sigmoid(raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def bbox_iou(boxes1, boxes2):
    """
    Calculates the standard Intersection over Union (IoU) between bounding boxes.

    Args:
        boxes1, boxes2: Tensors of shape [..., 4] with format [cx, cy, w, h].

    Returns:
        Tensor with IoU values.
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1_corners = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_corners = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1_corners[..., :2], boxes2_corners[..., :2])
    right_down = tf.minimum(boxes1_corners[..., 2:], boxes2_corners[..., 2:])

    inter_wh = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / tf.maximum(union_area, 1e-10)


def bbox_giou(boxes1, boxes2):
    """
    Calculates Generalized IoU (GIoU) between boxes.

    Args:
        boxes1, boxes2: Tensors of shape [..., 4] with format [cx, cy, w, h].

    Returns:
        Tensor with GIoU values.
    """
    # Convert to corner format
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    inter_right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = tf.maximum(inter_right_down - inter_left_up, 0.0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / tf.maximum(union_area, 1e-10)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

    giou = iou - (enclose_area - union_area) / tf.maximum(enclose_area, 1e-10)
    return giou


def bbox_ciou(boxes1, boxes2):
    """
    Calculates Complete IoU (CIoU) between boxes.

    Args:
        boxes1, boxes2: Tensors of shape [..., 4] with format [cx, cy, w, h].

    Returns:
        Tensor with CIoU values.
    """
    boxes1_corners = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_corners = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    # Enclosing box diagonal distance squared
    enclose_left_up = tf.minimum(boxes1_corners[..., :2], boxes2_corners[..., :2])
    enclose_right_down = tf.maximum(boxes1_corners[..., 2:], boxes2_corners[..., 2:])
    c2 = tf.reduce_sum(tf.square(enclose_right_down - enclose_left_up), axis=-1)

    # Center distance squared
    center_dist = tf.reduce_sum(tf.square(boxes1[..., :2] - boxes2[..., :2]), axis=-1)

    # IoU
    iou = bbox_iou(boxes1, boxes2)

    # Aspect ratio term
    w1, h1 = boxes1[..., 2], boxes1[..., 3]
    w2, h2 = boxes2[..., 2], boxes2[..., 3]
    v = (4 / (np.pi ** 2)) * tf.square(tf.atan(w2 / tf.maximum(h2, 1e-10)) - tf.atan(w1 / tf.maximum(h1, 1e-10)))

    alpha = v / tf.maximum((1.0 - iou + v), 1e-10)

    ciou = iou - (center_dist / tf.maximum(c2, 1e-10) + alpha * v)
    return ciou



def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=YOLO_COCO_CLASSES):
    """
    Computes YOLOv3 loss components: GIoU loss, confidence loss, and classification loss.

    Args:
        pred: Decoded predictions [batch, grid, grid, 3, 5 + num_classes].
        conv: Raw convolutional output from YOLO head (pre-activation).
        label: Ground truth label tensor.
        bboxes: Ground truth bounding boxes.
        i: Index for scale (0 = large, 1 = medium, 2 = small).
        CLASSES: Path to class names.

    Returns:
        giou_loss: Localization loss using GIoU.
        conf_loss: Confidence loss for objectness prediction.
        prob_loss: Classification loss for object categories.
    """
    NUM_CLASS = len(read_class_names(CLASSES))
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size

    # Reshape conv output to [batch, grid, grid, 3, 5 + num_classes]
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    conv_raw_conf = conv[..., 4:5]
    conv_raw_prob = conv[..., 5:]

    pred_xywh = pred[..., 0:4]
    pred_conf = pred[..., 4:5]

    label_xywh = label[..., 0:4]
    respond_bbox = label[..., 4:5]  # 1 if object exists in that anchor
    label_prob = label[..., 5:]     # one-hot class label

    # --- GIoU Loss ---
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - (label_xywh[..., 2:3] * label_xywh[..., 3:4]) / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1.0 - giou)

    # --- Confidence Loss ---
    iou = bbox_iou(pred_xywh[..., tf.newaxis, :], bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < YOLO_IOU_LOSS_THRESH, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2.0)
    conf_loss = conf_focal * (
        respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) +
        respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    # --- Classification Loss ---
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # --- Reduce losses ---
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss
