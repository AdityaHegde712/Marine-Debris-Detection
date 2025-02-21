import os
import shutil
from glob import glob
import cv2
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

SPLIT_FOLDER_NAME = "dataset_splits_new"


def get_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Read image and return the image array along with size tuple.

    Args:
        image_path (str): Image path.

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: Pair of image and its dimensions.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image.shape


def get_boxes(label_path: str, size: Tuple[int, int] = (256, 256)) -> List[List[int]]:
    with open(label_path, "r") as f:
        lines = f.readlines()
    lines = [[float(j) for j in i.split()[1:]] + ["0"] for i in lines]
    return lines


def draw_yolo_boxes(image: np.ndarray, bboxes: List[List[float]]) -> np.ndarray:
    """Function to draw YOLO boxes on the image.

    Args:
        image (np.ndarray): Image array.
        bboxes (List[List[int]]): List of lists containing bboxes and class name.

    Returns:
        np.ndarray: Image array with boxes drawn.
    """
    for bbox in bboxes:
        x, y, w, h = bbox[:4]
        x1 = int((x - w / 2) * image.shape[1])
        y1 = int((y - h / 2) * image.shape[0])
        x2 = int((x + w / 2) * image.shape[1])
        y2 = int((y + h / 2) * image.shape[0])
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image


def knitty_gritties():
    dir_prefix = input("Enter the directory prefix: ")
    if dir_prefix.lower() == "stop":
        return
    if dir_prefix:
        dir_prefix = dir_prefix.rstrip("_") + "_"

    save_dir = f"datasets/Planet/{SPLIT_FOLDER_NAME}/train/drawn_{dir_prefix}images"
    shutil.rmtree(save_dir, ignore_errors=True) if os.path.exists(save_dir) else None
    os.makedirs(save_dir, exist_ok=True)

    return save_dir, dir_prefix.lower()


def main():
    save_dir, dir_prefix = knitty_gritties()
    if dir_prefix == "stop":
        return
    images = glob(f"datasets/Planet/{SPLIT_FOLDER_NAME}/train/{dir_prefix}images/*.jpg")
    labels = glob(f"datasets/Planet/{SPLIT_FOLDER_NAME}/train/{dir_prefix}labels/*.txt")

    pbar = tqdm(total=len(images), desc="Drawing boxes on images")
    for image_path, label_path in zip(images, labels):
        image, size = get_image(image_path)
        bboxes = get_boxes(label_path, size)

        drawn_image = draw_yolo_boxes(image, bboxes)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR))
        pbar.update(1)


if __name__ == "__main__":
    main()
