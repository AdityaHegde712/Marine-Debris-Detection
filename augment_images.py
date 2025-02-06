import albumentations as A
import cv2
import numpy as np
from typing import Tuple, List
import os
import shutil
from glob import glob
from tqdm import tqdm

IMAGES_DIR = "datasets/Planet/dataset_splits/train/images"
LABELS_DIR = "datasets/Planet/dataset_splits/train/labels"
AUGMENTED_IMAGES_DIR = "datasets/Planet/dataset_splits/train/augmented_images"
AUGMENTED_LABELS_DIR = "datasets/Planet/dataset_splits/train/augmented_labels"
AUGS_PER_IMAGE = 5
SEGREGATION_IMAGES_DIR = "datasets/Planet/dataset_splits/train/segregated_images"
SEGREGATION_LABELS_DIR = "datasets/Planet/dataset_splits/train/segregated_labels"
ATTEMPT_LIMIT = 150

os.makedirs(SEGREGATION_IMAGES_DIR, exist_ok=True)
os.makedirs(SEGREGATION_LABELS_DIR, exist_ok=True)


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


def get_boxes(label_path: str) -> List:
    """Function to read a yolo txt file, re-arrange the coords and class name, and return the list of lists.

    Args:
        label_path (str): Path to the txt label.
        size (Tuple[int, int]): Tuple of image dimensions (height, width, channels).

    Returns:
        List[List[int]]: List of lists containing bboxes and class name.
    """
    with open(label_path, "r") as f:
        lines = f.readlines()
    lines = [[float(j) for j in i.split()[1:]] + ["0"] for i in lines]
    return lines


def make_writable_boxes(bboxes: List) -> List:
    """Function to restore the bboxes to writable format.

    Args:
        bboxes (List): List of lists containing bboxes and class name.

    Returns:
        List: List of lists containing bboxes and class name.
    """
    return [" ".join([str(int(i[-1]))] + [str(j) for j in i[:-1]]) + "\n" for i in bboxes]


def filter_boxes(bboxes: List) -> bool:
    filtered_boxes = bboxes.copy()
    num_deleted = 0
    for i, box in enumerate(bboxes):
        width, height = box[2], box[3]
        area = width * height

        negative_dims = width <= 0 or height <= 0
        small_area = area < 0.01
        large_area = area > 0.95

        if negative_dims or small_area or large_area:
            del filtered_boxes[i - num_deleted]
            num_deleted += 1

    return filtered_boxes


def round_off(bboxes: List[List[float]]) -> List[List[float]]:
    fixed_boxes = []
    for box in bboxes:
        box[0] = round(box[0], 6) if 0 < box[0] < 1 else 0 if box[0] < 0 else 1
        box[1] = round(box[1], 6) if 0 < box[1] < 1 else 0 if box[1] < 0 else 1
        box[2] = round(box[2], 6) if 0 < box[2] < 1 else 0 if box[2] < 0 else 1
        box[3] = round(box[3], 6) if 0 < box[3] < 1 else 0 if box[3] < 0 else 1
        fixed_boxes.append(box)
    return fixed_boxes


def main():
    if os.path.exists(AUGMENTED_IMAGES_DIR) and os.path.exists(AUGMENTED_LABELS_DIR):
        _ = input("Removing existing augmentations. If you want these results, make a backup and hit enter when you're ready.")
        shutil.rmtree(AUGMENTED_IMAGES_DIR)
        shutil.rmtree(AUGMENTED_LABELS_DIR)
        os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)
        os.makedirs(AUGMENTED_LABELS_DIR, exist_ok=True)

    os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)
    os.makedirs(AUGMENTED_LABELS_DIR, exist_ok=True)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.75),
            A.VerticalFlip(p=0.75),
            A.RandomRotate90(p=0.75),
            A.Transpose(p=0.75),
        ],
        bbox_params=A.BboxParams(format="yolo"),
    )
    print("Initialized transformation object.")

    all_images = sorted(glob(os.path.join(IMAGES_DIR, "*.jpg")))
    all_labels = sorted(glob(os.path.join(LABELS_DIR, "*.txt")))

    print(f"Found {len(all_images)} images and {len(all_labels)} labels.")
    assert len(all_images) == len(all_labels), "Number of images and labels do not match."
    print("Beginning augmentation...")

    pbar = tqdm(total=len(all_images), desc="Augmenting images")
    erroneous_images = set()
    skipped_images = list()
    for image_path, label_path in zip(all_images, all_labels):
        original_image, _ = get_image(image_path)
        original_bboxes = filter_boxes(get_boxes(label_path))

        if not original_bboxes:
            shutil.move(image_path, SEGREGATION_IMAGES_DIR)
            shutil.move(label_path, SEGREGATION_LABELS_DIR)
            continue

        images_list = []
        bboxes_list = []
        compare_list = []
        counter = 0
        attempts = 0
        while counter < AUGS_PER_IMAGE:
            try:
                transformed = transform(image=original_image, bboxes=original_bboxes)
            except ValueError:
                erroneous_images.add(image_path)
            except Exception as e:
                print(f"Encountered new error. \n\n{e}\n\nSkipping image {image_path}.")
                break

            transformed_image = transformed["image"]
            transformed_image = cv2.resize(transformed_image, (original_image.shape[1], original_image.shape[0]))
            transformed_bboxes = round_off(transformed["bboxes"])

            # Check for uniqueness by comparing the transformations
            unique = True
            if compare_list and transformed_bboxes in compare_list:
                unique = False

            if np.array_equal(transformed_image, original_image) and transformed_bboxes == original_bboxes:
                unique = False

            if unique:
                images_list.append(transformed_image)
                compare_list.append(transformed_bboxes)
                restored_boxes = make_writable_boxes(transformed_bboxes)
                bboxes_list.append(restored_boxes)
                counter += 1
            attempts += 1

            if attempts > ATTEMPT_LIMIT:
                skipped_images.append(image_path)
                break

        # if 5 < attempts < 20:
        #     print(f"Took {attempts} attempts to generate {AUGS_PER_IMAGE} unique augmentations for {image_path}.\n")
        for i, (image, bboxes) in enumerate(zip(images_list, bboxes_list)):
            image_name = os.path.join(AUGMENTED_IMAGES_DIR, f"{os.path.basename(image_path).split('.')[0]}_{i:02f}.jpg")
            txt_name = os.path.join(AUGMENTED_LABELS_DIR, f"{os.path.basename(label_path).split('.')[0]}_{i:02f}.txt")
            cv2.imwrite(image_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            with open(txt_name, "w") as f:
                f.writelines(bboxes)

        pbar.update(1)

    pbar.close()
    print("Augmentation complete.")

    erroneous_images = list(erroneous_images)
    skipped_images = list(skipped_images)
    print(f"Encountered {len(erroneous_images)} erroneous images with clipping issues. Here they are:")
    print(*erroneous_images, sep="\n")
    print(f"Skipped {len(skipped_images)} images due to excessive attempts at uniqueness. Blame albumentations. Stupid library. This is why I write this shit myself.")
    print(*skipped_images, sep="\n")

    with open("skipped_images.txt", "w", encoding="utf-8") as f:
        f.writelines([i + "\n" for i in skipped_images])
    with open("error_images.txt", "w", encoding="utf-8") as f:
        f.writelines([i + "\n" for i in erroneous_images])


if __name__ == "__main__":
    main()
