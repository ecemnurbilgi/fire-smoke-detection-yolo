from pathlib import Path
from collections import Counter

DATASET_PATH = Path(r"C:\Users\DELL\vscode\fire-smoke-yolo-detection\data")

SPLITS = ["train", "validation", "test"]

CLASS_NAMES = {
    0: "smoke",
    1: "fire"
}


def audit_split(split_name):
    image_dir = DATASET_PATH / split_name / "images"
    label_dir = DATASET_PATH / split_name / "labels"

    image_files = list(image_dir.glob("*"))
    label_files = list(label_dir.glob("*.txt"))

    class_counter = Counter()
    empty_labels = 0
    missing_labels = 0
    total_boxes = 0

    for image_path in image_files:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            missing_labels += 1
            continue

        lines = label_path.read_text().strip().splitlines()

        if len(lines) == 0:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                print(f"Invalid label format: {label_path}")
                continue

            class_id = int(parts[0])
            class_counter[class_id] += 1
            total_boxes += 1

    print(f"\n===== {split_name.upper()} SET =====")
    print(f"Images: {len(image_files)}")
    print(f"Label files: {len(label_files)}")
    print(f"Missing labels: {missing_labels}")
    print(f"Empty label files / background images: {empty_labels}")
    print(f"Total bounding boxes: {total_boxes}")

    for class_id, count in class_counter.items():
        class_name = CLASS_NAMES.get(class_id, "unknown")
        print(f"{class_name}: {count}")


def main():
    for split in SPLITS:
        audit_split(split)


if __name__ == "__main__":
    main()