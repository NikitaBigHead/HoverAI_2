import argparse
from pathlib import Path

import cv2
import mediapipe as mp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MARGIN_RATIO = 0.15
RAW_HANDS_GLOB = "raw_hands_*"
DEFAULT_SAVE_DIRNAME = "cropped"


def collect_images(images_dir):
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_input_directories(input_dir):
    input_dir = input_dir.resolve()

    direct_images = collect_images(input_dir)
    if direct_images:
        return [(input_dir, input_dir.name)]

    matched_dirs = sorted(
        path for path in input_dir.iterdir()
        if path.is_dir() and path.name.startswith("raw_hands_")
    )
    labeled_dirs = []
    for raw_dir in matched_dirs:
        for child_dir in sorted(path for path in raw_dir.iterdir() if path.is_dir()):
            if collect_images(child_dir):
                labeled_dirs.append((child_dir, child_dir.name))
    return labeled_dirs


def resolve_output_path(output_dir, image_path, source_dir_name):
    candidate = output_dir / image_path.name
    if not candidate.exists():
        return candidate

    stem = f"{source_dir_name}_{image_path.stem}"
    suffix = image_path.suffix
    candidate = output_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = output_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def resolve_label(input_dir, input_dirs, label):
    if label:
        return label

    if len(input_dirs) == 1:
        return input_dirs[0][1]

    raise RuntimeError(
        "A training label is required for this input. "
        "Use --label <gesture_name> to save images into cropped/<gesture_name>"
    )


def crop_single_hand(image_bgr, hands_detector):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None

    image_h, image_w = image_bgr.shape[:2]
    best_bbox = None
    best_area = -1

    for hand_landmarks in result.multi_hand_landmarks:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        min_x = max(int(min(xs) * image_w), 0)
        max_x = min(int(max(xs) * image_w), image_w - 1)
        min_y = max(int(min(ys) * image_h), 0)
        max_y = min(int(max(ys) * image_h), image_h - 1)

        box_w = max_x - min_x + 1
        box_h = max_y - min_y + 1
        margin_x = int(box_w * MARGIN_RATIO)
        margin_y = int(box_h * MARGIN_RATIO)

        min_x = max(min_x - margin_x, 0)
        max_x = min(max_x + margin_x, image_w - 1)
        min_y = max(min_y - margin_y, 0)
        max_y = min(max_y + margin_y, image_h - 1)

        area = (max_x - min_x + 1) * (max_y - min_y + 1)
        if area > best_area:
            best_area = area
            best_bbox = (min_x, min_y, max_x, max_y)

    if best_bbox is None:
        return None

    min_x, min_y, max_x, max_y = best_bbox
    return image_bgr[min_y:max_y + 1, min_x:max_x + 1]


def process_directory(images_dir, output_dir):
    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    processed = 0
    skipped = 0

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands_detector:
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                skipped += 1
                print(f"Skip unreadable file: {image_path}")
                continue

            cropped_hand = crop_single_hand(image_bgr, hands_detector)
            if cropped_hand is None:
                skipped += 1
                print(f"Skip no-hand image: {image_path}")
                continue

            output_path = resolve_output_path(output_dir, image_path, images_dir.name)
            ok = cv2.imwrite(str(output_path), cropped_hand)
            if not ok:
                skipped += 1
                print(f"Skip failed save: {output_path}")
                continue

            processed += 1
            print(f"Saved: {output_path}")

    print(f"Done. Processed: {processed}, skipped: {skipped}, output: {output_dir}")


def process_input(input_dir, save_root, label=None):
    input_dirs = collect_input_directories(input_dir)
    if not input_dirs:
        raise RuntimeError(
            f"No images found in {input_dir} and no subdirectories matching "
            f"{RAW_HANDS_GLOB} with images were found"
        )

    if label:
        output_dir = save_root.resolve() / label
        print(f"Saving all cropped images to: {output_dir}")
        for images_dir, _detected_label in input_dirs:
            print(f"Processing directory: {images_dir}")
            process_directory(images_dir, output_dir)
        return

    for images_dir, detected_label in input_dirs:
        output_dir = save_root.resolve() / detected_label
        print(f"Saving label '{detected_label}' to: {output_dir}")
        print(f"Processing directory: {images_dir}")
        process_directory(images_dir, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop one hand from each image in a directory or in raw_hands_* subdirectories"
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory with input images or root containing raw_hands_* subdirectories",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(DEFAULT_SAVE_DIRNAME),
        help="Root directory for saving cropped hand images (default: ./cropped)",
    )
    parser.add_argument(
        "--label",
        help="Override gesture label. If omitted, labels are taken from subdirectory names",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_input(args.images_dir, args.save_dir, args.label)
