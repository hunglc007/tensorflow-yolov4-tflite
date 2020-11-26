"""
Script to debug the drawing of training dataset.
"""

from PIL import Image
import colorsys
import cv2
import numpy as np
import os
import random


# Config
# dataset (copy from dataset text file)
DATA_DIR = "./data"
ANNOTATION_PATH = "./data/dataset/csgo.txt"


def get_random_image(data_dir, annotation_path):
    with open(annotation_path, 'r') as f:
        annotations  = f.read()
        annotations = [x for x in annotations.split("\n") if x.strip() != ""]
        random_annotation = random.choice(annotations)
        return f"{data_dir}{random_annotation}"


def swap_dimension(coord):
    """
    swap x, y, x, y to y, x, y, x
    """
    for dimension in coord:
        dimension[0], dimension[1], dimension[2], dimension[3] = dimension[1], dimension[0], dimension[3], dimension[2]
    return coord


def read_from_annotation(annotation_string):
    splits = annotation_string.split(' ')
    image_path = splits[0]
    boxes = splits[1:]
    coord = []

    for box in boxes:
        # skip empty string
        if len(box) == 0:
            continue

        # coord.append([int(x) for x in box.split(",")[:-1]]) # last one is classification
        coord.append([int(x) for x in box.split(",")])

    return image_path, swap_dimension(coord)


def draw_bbox(image_path, coor, num_classes = 2):
    """
    coor must be start_y, start_x, end_y, end_x
    """

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    font_scale = 0.8
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for box in coor:
        bbox_mess = "CT" if box[-1] == 0 else "T"
        t_size = cv2.getTextSize(bbox_mess, 0, font_scale, thickness=bbox_thick // 2)[0]
        bbox_color = colors[0] if box[-1] == 0 else colors[1]

        #         x       y         x      y
        c1, c2 = (box[1], box[0]), (box[3], box[2])
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, bbox_color, int(bbox_thick // 1.5), lineType=cv2.LINE_AA)

    return image


if __name__ == '__main__':
    example_annotation = get_random_image(DATA_DIR, ANNOTATION_PATH)
    print(example_annotation)

    image_path, coord = read_from_annotation(example_annotation)
    image = draw_bbox(image_path, coord)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
