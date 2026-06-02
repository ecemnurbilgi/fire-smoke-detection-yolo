from pathlib import Path
from collections import defaultdict

DATASET_PATH = Path(r"C:\Users\DELL\vscode\fire-smoke-yolo-detection\data")

SPLITS = ["train", "validation", "test"]

CLASS_NAMES = {
    0: "smoke",
    1: "fire"
}


def classify_box_size(area_ratio):
    if area_ratio < 0.01:
        return "small"
    elif area_ratio < 0.10:
        return "medium"
    else:
        return "large"


def analyze_split(split):
    label_dir = DATASET_PATH / split / "labels"

    stats = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0, "total": 0})

    for label_path in label_dir.glob("*.txt"):
        lines = label_path.read_text().strip().splitlines()

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            box_width = float(parts[3])
            box_height = float(parts[4])

            area_ratio = box_width * box_height
            size_category = classify_box_size(area_ratio)

            class_name = CLASS_NAMES.get(class_id, "unknown")

            stats[class_name][size_category] += 1
            stats[class_name]["total"] += 1

    print(f"\n===== {split.upper()} BOX SIZE ANALYSIS =====")

    for class_name, values in stats.items():
        total = values["total"]

        print(f"\nClass: {class_name}")
        print(f"Total boxes: {total}")

        for category in ["small", "medium", "large"]:
            count = values[category]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{category}: {count} ({percentage:.2f}%)")


def main():
    for split in SPLITS:
        analyze_split(split)


if __name__ == "__main__":
    main()