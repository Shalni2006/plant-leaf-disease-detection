"""
Dataset Inspection and Health Check Script
Scans the dataset directory to inspect:
- Identified classes / folders
- Categorization into 'Healthy' vs 'Diseased'
- Image counts and formats
- Image dimensions and sample readability
"""
import os
from pathlib import Path
from PIL import Image

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

def inspect_dataset(data_path=DATASET_DIR):
    print("=" * 60)
    print("      PLANTVILLAGE DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"Target Directory: {data_path}")
    print("-" * 60)

    if not data_path.exists():
        print(f"Directory not found: {data_path}")
        return

    # Check for direct subfolders or nested subfolder (e.g. PlantVillage/...)
    subdirs = [p for p in data_path.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name.lower() in ["plantvillage", "plant_village", "dataset", "segmented", "color"]:
        print(f"Found root subfolder '{subdirs[0].name}'. Inspecting contents...")
        subdirs = [p for p in subdirs[0].iterdir() if p.is_dir()]

    if not subdirs:
        print("[!] No class subdirectories found in the dataset folder.")
        print("    Please place your class folders inside: " + str(data_path))
        print("=" * 60)
        return

    total_images = 0
    formats = {}
    class_stats = []

    for category_dir in sorted(subdirs, key=lambda x: x.name):
        class_name = category_dir.name
        is_healthy = "healthy" in class_name.lower()
        status = "HEALTHY" if is_healthy else "DISEASED"
        
        # Collect images
        image_files = [f for f in category_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        count = len(image_files)
        total_images += count

        # Check formats and inspect sample image
        sample_size = None
        for img_file in image_files:
            ext = img_file.suffix.lower()
            formats[ext] = formats.get(ext, 0) + 1
            if sample_size is None:
                try:
                    with Image.open(img_file) as img:
                        sample_size = img.size
                except Exception:
                    sample_size = "Error reading"

        class_stats.append({
            "name": class_name,
            "status": status,
            "count": count,
            "sample_size": sample_size
        })

    print(f"Total Classes Identified : {len(class_stats)}")
    print(f"Total Images Found       : {total_images}")
    print(f"Image Formats            : {formats}")
    print("-" * 60)
    print(f"{'Class Name':<42} | {'Condition':<9} | {'Count':<6} | {'Sample Resolution'}")
    print("-" * 75)

    healthy_count = 0
    diseased_count = 0

    for stat in class_stats:
        if stat["status"] == "HEALTHY":
            healthy_count += stat["count"]
        else:
            diseased_count += stat["count"]
        res_str = f"{stat['sample_size'][0]}x{stat['sample_size'][1]}" if isinstance(stat['sample_size'], tuple) else str(stat['sample_size'])
        print(f"{stat['name']:<42} | {stat['status']:<9} | {stat['count']:<6} | {res_str}")

    print("-" * 75)
    print(f"Total Healthy Leaves  : {healthy_count}")
    print(f"Total Diseased Leaves : {diseased_count}")
    print("=" * 75)

if __name__ == "__main__":
    inspect_dataset()
