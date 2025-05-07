import os
import cv2
import random
import numpy as np
import tensorflow as tf
from yolov3.utils import read_class_names, image_preprocess
from yolov3.yolov3 import bbox_iou
from yolov3.configs import *


class Forge:
    def __init__(self, tag, resize=TEST_INPUT_SIZE):
        self.src = TRAIN_ANNOT_PATH if tag == 'train' else TEST_ANNOT_PATH
        self.size = TRAIN_INPUT_SIZE if tag == 'train' else resize
        self.load = TRAIN_LOAD_IMAGES_TO_RAM
        self.augment = TRAIN_DATA_AUG if tag == 'train' else TEST_DATA_AUG
        self.mini = TRAIN_YOLO_TINY
        self.batch = TRAIN_BATCH_SIZE if tag == 'train' else TEST_BATCH_SIZE
        self.strides = np.array(YOLO_STRIDES)
        self.classes = read_class_names(TRAIN_CLASSES)
        self.nclass = len(self.classes)
        self.anchors = (np.array(YOLO_ANCHORS).T / self.strides).T
        self.per = YOLO_ANCHOR_PER_SCALE
        self.cap = YOLO_MAX_BBOX_PER_SCALE
        self.cache = self.cache_data(tag)
        self.total = len(self.cache)
        self.cycles = int(np.ceil(self.total / self.batch))
        self.index = 0

    def cache_data(self, tag):
        store = []
        with open(self.src, 'r') as f:
            raw = f.read().splitlines()
            lines = [x.strip() for x in raw if len(x.strip().split()[1:]) != 0]
        np.random.shuffle(lines)
        for line in lines:
            items = line.split()
            img_path, pivot = '', 1
            for idx, val in enumerate(items):
                if not val.replace(',', '').isnumeric():
                    img_path += ('' if img_path == '' else ' ') + val
                else:
                    pivot = idx
                    break
            if not os.path.exists(img_path):
                raise KeyError(f"{img_path} missing")
            image = cv2.imread(img_path) if self.load else ''
            store.append([img_path, items[pivot:], image])
        return store

    def __iter__(self):
        return self

    def __len__(self):
        return self.cycles

    def wipe(self, entry):
        path = entry[0]
        name = path.split('/')[-1]
        with open(self.src, "r+") as f:
            data = f.readlines()
            f.seek(0)
            for line in data:
                if name not in line:
                    f.write(line)
            f.truncate()

    def __next__(self):
        with tf.device('/cpu:0'):
            size = random.choice([self.size])
            out = size // self.strides
            batch_x = np.zeros((self.batch, size, size, 3), dtype=np.float32)

            if self.mini:
                box_m = np.zeros((self.batch, out[0], out[0], self.per, 5 + self.nclass), dtype=np.float32)
                box_l = np.zeros((self.batch, out[1], out[1], self.per, 5 + self.nclass), dtype=np.float32)
            else:
                box_s = np.zeros((self.batch, out[0], out[0], self.per, 5 + self.nclass), dtype=np.float32)
                box_m = np.zeros((self.batch, out[1], out[1], self.per, 5 + self.nclass), dtype=np.float32)
                box_l = np.zeros((self.batch, out[2], out[2], self.per, 5 + self.nclass), dtype=np.float32)
                sb = np.zeros((self.batch, self.cap, 4), dtype=np.float32)
            mb = np.zeros((self.batch, self.cap, 4), dtype=np.float32)
            lb = np.zeros((self.batch, self.cap, 4), dtype=np.float32)

            if self.index < self.cycles:
                ptr = 0
                while ptr < self.batch:
                    i = self.index * self.batch + ptr
                    if i >= self.total: i -= self.total
                    entry = self.cache[i]
                    image, boxes = self.process(entry)
                    try:
                        if self.mini:
                            lmb, llb, mm, ml = self.encode(boxes)
                        else:
                            lsb, lmb, llb, ss, mm, ml = self.encode(boxes)
                    except IndexError:
                        self.wipe(entry)
                        raise Exception("Corrupt annotation found and removed. Restart training.")

                    batch_x[ptr] = image
                    box_m[ptr] = lmb
                    box_l[ptr] = llb
                    mb[ptr] = mm
                    lb[ptr] = ml
                    if not self.mini:
                        box_s[ptr] = lsb
                        sb[ptr] = ss
                    ptr += 1

                self.index += 1
                if not self.mini:
                    pack_s = (box_s, sb)
                pack_m = (box_m, mb)
                pack_l = (box_l, lb)
                return batch_x, (pack_m, pack_l) if self.mini else (pack_s, pack_m, pack_l)
            else:
                self.index = 0
                np.random.shuffle(self.cache)
                raise StopIteration

    def flip(self, img, box):
        if random.random() < 0.5:
            h, w, _ = img.shape
            img = img[:, ::-1, :]
            box[:, [0, 2]] = w - box[:, [2, 0]]
        return img, box

    def crop(self, img, box):
        if random.random() < 0.5:
            h, w, _ = img.shape
            maxbox = np.concatenate([np.min(box[:, 0:2], axis=0), np.max(box[:, 2:4], axis=0)], axis=-1)
            tx1 = max(0, int(maxbox[0] - random.uniform(0, maxbox[0])))
            ty1 = max(0, int(maxbox[1] - random.uniform(0, maxbox[1])))
            tx2 = max(w, int(maxbox[2] + random.uniform(0, w - maxbox[2])))
            ty2 = max(h, int(maxbox[3] + random.uniform(0, h - maxbox[3])))
            img = img[ty1:ty2, tx1:tx2]
            box[:, [0, 2]] -= tx1
            box[:, [1, 3]] -= ty1
        return img, box

    def shift(self, img, box):
        if random.random() < 0.5:
            h, w, _ = img.shape
            maxbox = np.concatenate([np.min(box[:, 0:2], axis=0), np.max(box[:, 2:4], axis=0)], axis=-1)
            tx = random.uniform(-(maxbox[0] - 1), (w - maxbox[2] - 1))
            ty = random.uniform(-(maxbox[1] - 1), (h - maxbox[3] - 1))
            mat = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, mat, (w, h))
            box[:, [0, 2]] += tx
            box[:, [1, 3]] += ty
        return img, box

    def process(self, entry, mAP=False):
        path, data, cached = entry
        img = cached if self.load else cv2.imread(path)
        boxes = np.array([list(map(int, box.split(','))) for box in data])
        if self.augment:
            img, boxes = self.flip(img.copy(), boxes.copy())
            img, boxes = self.crop(img.copy(), boxes.copy())
            img, boxes = self.shift(img.copy(), boxes.copy())
        if mAP: return img, boxes
        return image_preprocess(img.copy(), [self.size, self.size], boxes.copy())

    def encode(self, boxes):
        levels = len(self.strides)
        label = [np.zeros((self.size // self.strides[i], self.size // self.strides[i], self.per, 5 + self.nclass)) for i in range(levels)]
        coords = [np.zeros((self.cap, 4)) for _ in range(levels)]
        tally = np.zeros((levels,))

        for box in boxes:
            bcoor, bind = box[:4], box[4]
            onehot = np.zeros(self.nclass, dtype=np.float32)
            onehot[bind] = 1.0
            spread = np.full(self.nclass, 1.0 / self.nclass)
            label_vec = onehot * 0.99 + spread * 0.01
            ctrwh = np.concatenate([(bcoor[2:] + bcoor[:2]) * 0.5, bcoor[2:] - bcoor[:2]], axis=-1)
            ctrscaled = ctrwh[np.newaxis, :] / self.strides[:, np.newaxis]
            iou = []
            hit = False

            for i in range(levels):
                anchor = np.zeros((self.per, 4))
                anchor[:, 0:2] = np.floor(ctrscaled[i, 0:2]).astype(np.int32) + 0.5
                anchor[:, 2:4] = self.anchors[i]
                iou_val = bbox_iou(ctrscaled[i][np.newaxis, :], anchor)
                iou.append(iou_val)
                mask = iou_val > 0.3

                if np.any(mask):
                    x, y = np.floor(ctrscaled[i, 0:2]).astype(np.int32)
                    label[i][y, x, mask] = 0
                    label[i][y, x, mask, 0:4] = ctrwh
                    label[i][y, x, mask, 4:5] = 1.0
                    label[i][y, x, mask, 5:] = label_vec
                    pos = int(tally[i] % self.cap)
                    coords[i][pos, :4] = ctrwh
                    tally[i] += 1
                    hit = True

            if not hit:
                index = np.argmax(np.concatenate(iou))
                layer = index // self.per
                anchor = index % self.per
                x, y = np.floor(ctrscaled[layer, 0:2]).astype(np.int32)
                label[layer][y, x, anchor] = 0
                label[layer][y, x, anchor, 0:4] = ctrwh
                label[layer][y, x, anchor, 4:5] = 1.0
                label[layer][y, x, anchor, 5:] = label_vec
                pos = int(tally[layer] % self.cap)
                coords[layer][pos, :4] = ctrwh
                tally[layer] += 1

        return (label[1], label[2], coords[1], coords[2]) if self.mini else (label[0], label[1], label[2], coords[0], coords[1], coords[2])
