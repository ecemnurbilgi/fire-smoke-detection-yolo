from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = Path(r"C:\Users\DELL\vscode\fire-smoke-yolo-detection\data")

SPLIT = "train"

CLASS_NAMES = {
    0: "smoke",
    1: "fire"
}


def draw_yolo_boxes(image_path, label_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape

    if label_path.exists():
        lines = label_path.read_text().strip().splitlines()

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            label = CLASS_NAMES.get(class_id, "unknown")

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    return image


def main():
    image_dir = DATASET_PATH / SPLIT / "images"
    label_dir = DATASET_PATH / SPLIT / "labels"

    image_files = list(image_dir.glob("*"))

    selected_images = random.sample(image_files, 6)

    plt.figure(figsize=(12, 8))

    for i, image_path in enumerate(selected_images):
        label_path = label_dir / f"{image_path.stem}.txt"
        image = draw_yolo_boxes(image_path, label_path)

        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(image_path.name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()