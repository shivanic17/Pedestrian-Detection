import json
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_ROOT = BASE_DIR / "cityscapes" / "leftImg8bit"
ANNOTATION_ROOT = BASE_DIR / "gtBboxCityPersons" / "gtBboxCityPersons"

# Only labels are created here
LABEL_ROOT = BASE_DIR / "citypersons_yolo" / "labels"

TARGET_LABEL = "pedestrian"


# ============================================================
# CONVERT SPLIT
# ============================================================

def convert_split(split):

    annotation_dir = ANNOTATION_ROOT / split
    output_label_dir = LABEL_ROOT / split

    output_label_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(
        annotation_dir.rglob("*_gtBboxCityPersons.json")
    )

    print(f"\n{'=' * 50}")
    print(f"Processing {split}")
    print(f"{'=' * 50}")
    print(f"Annotation files found: {len(json_files)}")

    total_images = 0
    total_pedestrians = 0

    for json_file in json_files:

        # ----------------------------------------------------
        # Read JSON
        # ----------------------------------------------------

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_width = data["imgWidth"]
        img_height = data["imgHeight"]

        # ----------------------------------------------------
        # Determine city and image name
        # ----------------------------------------------------

        city = json_file.parent.name

        image_name = json_file.name.replace(
            "_gtBboxCityPersons.json",
            "_leftImg8bit.png"
        )

        image_path = IMAGE_ROOT / split / city / image_name

        # Verify image exists
        if not image_path.exists():
            print(f"WARNING: Image not found:")
            print(image_path)
            continue

        # ----------------------------------------------------
        # Create corresponding label directory
        # ----------------------------------------------------

        label_city_dir = output_label_dir / city
        label_city_dir.mkdir(parents=True, exist_ok=True)

        label_file = label_city_dir / (
            image_path.stem + ".txt"
        )

        # ----------------------------------------------------
        # Convert bounding boxes
        # ----------------------------------------------------

        yolo_labels = []

        for obj in data.get("objects", []):

            # We only detect pedestrians
            if obj.get("label") != TARGET_LABEL:
                continue

            bbox = obj.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            x, y, width, height = bbox

            # Skip invalid bounding boxes
            if width <= 0 or height <= 0:
                continue

            # ------------------------------------------------
            # CityPersons:
            #
            # x, y, width, height
            #
            # YOLO:
            #
            # class center_x center_y width height
            # ------------------------------------------------

            center_x = x + width / 2
            center_y = y + height / 2

            # Normalize to 0-1
            center_x /= img_width
            center_y /= img_height
            width /= img_width
            height /= img_height

            # Class 0 = pedestrian
            yolo_labels.append(
                f"0 {center_x:.6f} {center_y:.6f} "
                f"{width:.6f} {height:.6f}"
            )

        # ----------------------------------------------------
        # Write label file
        # ----------------------------------------------------

        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_labels))

        total_images += 1
        total_pedestrians += len(yolo_labels)

    print(f"\nImages processed: {total_images}")
    print(f"Pedestrians converted: {total_pedestrians}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CityPersons → YOLO Label Converter")
    print("Storage-saving version")
    print("=" * 60)

    convert_split("train")
    convert_split("val")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(f"\nLabels created at:")
    print(LABEL_ROOT)