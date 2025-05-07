import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import shutil
import numpy as np
import tensorflow as tf
from yolov3.dataset import Dataset
from yolov3.yolov3 import Create_Yolov3, compute_loss
from yolov3.utils import load_yolo_weights
from yolov3.configs import *
from Precision_Monitor_mAP import get_mAP

if TRAIN_YOLO_TINY:
    TRAIN_MODEL_NAME += "_Tiny"

def execute_training():
    available_gpus = tf.config.experimental.list_physical_devices('GPU')
    if available_gpus:
        try:
            tf.config.experimental.set_memory_growth(available_gpus[0], True)
        except RuntimeError:
            pass

    if os.path.exists(TRAIN_LOGDIR):
        shutil.rmtree(TRAIN_LOGDIR)
    log_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    train_data = Dataset('train')
    validation_data = Dataset('test')

    total_batches = len(train_data)
    counter = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup = TRAIN_WARMUP_EPOCHS * total_batches
    total_iterations = TRAIN_EPOCHS * total_batches

    detector = Create_Yolov3(input_size=YOLO_INPUT_SIZE, training=True, CLASSES=TRAIN_CLASSES)
    if TRAIN_FROM_CHECKPOINT:
        try:
            detector.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
        except ValueError:
            TRAIN_FROM_CHECKPOINT = False

    optimizer = tf.keras.optimizers.Adam()

    def run_train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = detector(inputs, training=True)
            giou, conf, prob = 0, 0, 0
            num_layers = 2 if TRAIN_YOLO_TINY else 3
            for idx in range(num_layers):
                c, p = predictions[idx * 2], predictions[idx * 2 + 1]
                losses = compute_loss(p, c, *targets[idx], idx, CLASSES=TRAIN_CLASSES)
                giou += losses[0]
                conf += losses[1]
                prob += losses[2]
            total = giou + conf + prob

        grads = tape.gradient(total, detector.trainable_variables)
        optimizer.apply_gradients(zip(grads, detector.trainable_variables))

        counter.assign_add(1)
        if counter < warmup:
            lr_value = counter / warmup * TRAIN_LR_INIT
        else:
            cycle = (counter - warmup) / (total_iterations - warmup)
            lr_value = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END) * (1 + tf.cos(cycle * np.pi))
        optimizer.lr.assign(lr_value.numpy())

        with log_writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=counter)
            tf.summary.scalar("loss/total", total, step=counter)
            tf.summary.scalar("loss/giou", giou, step=counter)
            tf.summary.scalar("loss/conf", conf, step=counter)
            tf.summary.scalar("loss/prob", prob, step=counter)
        log_writer.flush()

        return counter.numpy(), optimizer.lr.numpy(), giou.numpy(), conf.numpy(), prob.numpy(), total.numpy()

    eval_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    def run_validation_step(inputs, targets):
        predictions = detector(inputs, training=False)
        giou, conf, prob = 0, 0, 0
        count = 2 if TRAIN_YOLO_TINY else 3
        for idx in range(count):
            c, p = predictions[idx * 2], predictions[idx * 2 + 1]
            losses = compute_loss(p, c, *targets[idx], idx, CLASSES=TRAIN_CLASSES)
            giou += losses[0]
            conf += losses[1]
            prob += losses[2]
        return giou.numpy(), conf.numpy(), prob.numpy(), (giou + conf + prob).numpy()

    eval_model = Create_Yolov3(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    lowest_val_loss = float('inf')

    for ep in range(TRAIN_EPOCHS):
        for img, tgt in train_data:
            stats = run_train_step(img, tgt)
            prog = stats[0] % total_batches
            print(f"epoch:{ep:2} step:{prog:5}/{total_batches}, lr:{stats[1]:.6f}, giou:{stats[2]:7.2f}, conf:{stats[3]:7.2f}, prob:{stats[4]:7.2f}, total:{stats[5]:7.2f}")

        if not validation_data:
            detector.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            continue

        v_count = v_giou = v_conf = v_prob = v_total = 0
        for img, tgt in validation_data:
            g, c, p, t = run_validation_step(img, tgt)
            v_count += 1
            v_giou += g
            v_conf += c
            v_prob += p
            v_total += t

        with eval_writer.as_default():
            tf.summary.scalar("val/total", v_total / v_count, step=ep)
            tf.summary.scalar("val/giou", v_giou / v_count, step=ep)
            tf.summary.scalar("val/conf", v_conf / v_count, step=ep)
            tf.summary.scalar("val/prob", v_prob / v_count, step=ep)
        eval_writer.flush()

        print(f"\nValidation — giou:{v_giou / v_count:.2f}, conf:{v_conf / v_count:.2f}, prob:{v_prob / v_count:.2f}, total:{v_total / v_count:.2f}\n")

        final_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            final_path += f"_val_{v_total / v_count:.2f}"
            detector.save_weights(final_path)
        elif TRAIN_SAVE_BEST_ONLY and lowest_val_loss > v_total / v_count:
            detector.save_weights(final_path)
            lowest_val_loss = v_total / v_count
        elif not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            detector.save_weights(final_path)

    try:
        eval_model.load_weights(final_path)
        get_mAP(eval_model, validation_data, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
    except UnboundLocalError:
        print("Weights not found for evaluation.")

if __name__ == '__main__':
    execute_training()
#