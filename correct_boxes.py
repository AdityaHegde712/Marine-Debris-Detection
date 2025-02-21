import os
from glob import glob
from draw_boxes import get_boxes
from tqdm import tqdm

LABELS_DIR = "datasets/Planet/dataset_splits_new/test/labels_old"
TARGET_DIR = "datasets/Planet/dataset_splits_new/test/labels"
os.makedirs(TARGET_DIR, exist_ok=True)


def main():
    all_labels = glob(os.path.join(LABELS_DIR, "*.txt"))
    for label_path in tqdm(all_labels, desc="Correcting boxes"):
        bboxes = get_boxes(label_path)
        fixed_bboxes = []

        for bbox in bboxes:
            x, y, width, height, class_name = bbox

            top_left_x = min(max(x - width / 2, 0), 1)
            top_left_y = min(max(y - height / 2, 0), 1)
            bottom_right_x = min(max(x + width / 2, 0), 1)
            bottom_right_y = min(max(y + height / 2, 0), 1)

            x = (top_left_x + bottom_right_x) / 2
            y = (top_left_y + bottom_right_y) / 2
            width = bottom_right_x - top_left_x
            height = bottom_right_y - top_left_y

            fixed_bboxes.append([class_name, x, y, width, height])

        with open(os.path.join(TARGET_DIR, os.path.basename(label_path)), "w") as f:
            f.writelines([" ".join([str(j) for j in i]) + "\n" for i in fixed_bboxes])


if __name__ == "__main__":
    main()
