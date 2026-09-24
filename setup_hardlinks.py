import os
import glob
import shutil

PROJECT_ROOT = r"C:\Users\chunchula shivani\Downloads\AAN_CaseStudy"
CITYSCAPES_DIR = os.path.join(PROJECT_ROOT, "cityscapes", "leftImg8bit")
YOLO_IMAGES_DIR = os.path.join(PROJECT_ROOT, "citypersons_yolo", "images")

# Clean up existing junctions/folders first
for split in ["train", "val"]:
    target_dir = os.path.join(YOLO_IMAGES_DIR, split)
    if os.path.exists(target_dir) or os.path.islink(target_dir):
        print(f"Removing old folder/junction: {target_dir}")
        try:
            if os.path.islink(target_dir):
                os.unlink(target_dir)  # Remove junction
            else:
                shutil.rmtree(target_dir)  # Remove directory
        except Exception as e:
            print(f"Notice during cleanup: {e}")

# Create hard links
for split in ["train", "val"]:
    src_split = os.path.join(CITYSCAPES_DIR, split)
    dst_split = os.path.join(YOLO_IMAGES_DIR, split)
    
    png_files = glob.glob(os.path.join(src_split, "*", "*.png"))
    print(f"Found {len(png_files)} images in {split}")

    for src_file in png_files:
        rel_path = os.path.relpath(src_file, src_split)
        dst_file = os.path.join(dst_split, rel_path)

        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        if not os.path.exists(dst_file):
            try:
                os.link(src_file, dst_file)
            except Exception as e:
                print(f"Error linking {src_file}: {e}")

print("Hard link setup complete!")