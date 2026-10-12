"""
Read-Only Dataset Verification Script
Compares source (original download) and destination (project workspace) datasets.
Does NOT modify, move, delete, or rename any files.
"""
from pathlib import Path
from PIL import Image

SRC_DIR = Path(r"C:\Users\shaln\Downloads\archive (1)\PlantVillage\PlantVillage")
DST_DIR = Path(r"E:\Plant Leaf Disease Detection\dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def verify_and_compare():
    print("=" * 70)
    print("        READ-ONLY DATASET COMPARISON & INTEGRITY REPORT")
    print("=" * 70)
    print(f"Source Path      : {SRC_DIR}")
    print(f"Destination Path : {DST_DIR}")
    print("-" * 70)

    # 1. Discover class folders
    src_classes = sorted([d.name for d in SRC_DIR.iterdir() if d.is_dir()])
    dst_classes = sorted([d.name for d in DST_DIR.iterdir() if d.is_dir()])

    all_classes = sorted(list(set(src_classes + dst_classes)))

    print(f"{'Class Name':<42} | {'Source':<7} | {'Dest':<7} | {'Status'}")
    print("-" * 70)

    total_src_images = 0
    total_dst_images = 0
    all_matched = True

    non_image_files_src = []
    non_image_files_dst = []

    corrupt_samples_dst = 0

    for cls in all_classes:
        src_path = SRC_DIR / cls
        dst_path = DST_DIR / cls

        src_imgs = []
        dst_imgs = []

        if src_path.exists():
            for f in src_path.iterdir():
                if f.is_file():
                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        src_imgs.append(f)
                    else:
                        non_image_files_src.append(f)
        
        if dst_path.exists():
            for f in dst_path.iterdir():
                if f.is_file():
                    if f.suffix.lower() in IMAGE_EXTENSIONS:
                        dst_imgs.append(f)
                    else:
                        non_image_files_dst.append(f)

        src_count = len(src_imgs)
        dst_count = len(dst_imgs)
        total_src_images += src_count
        total_dst_images += dst_count

        status = "MATCH" if src_count == dst_count else "MISMATCH"
        if status != "MATCH":
            all_matched = False

        print(f"{cls:<42} | {src_count:<7} | {dst_count:<7} | {status}")

    print("-" * 70)
    print(f"{'TOTAL IMAGES':<42} | {total_src_images:<7} | {total_dst_images:<7} | {'PERFECT' if all_matched else 'DIFF'}")
    print("=" * 70)

    # 2. Non-image files report
    print("\n--- Non-Image Files Investigation ---")
    print(f"Non-image files found in Source      : {len(non_image_files_src)}")
    for f in non_image_files_src:
        print(f"  - Source: {f} (Size: {f.stat().st_size} bytes)")
    print(f"Non-image files found in Destination : {len(non_image_files_dst)}")
    for f in non_image_files_dst:
        print(f"  - Dest  : {f} (Size: {f.stat().st_size} bytes)")

    # 3. Potato___healthy investigation
    print("\n--- Potato___healthy Investigation ---")
    potato_src = SRC_DIR / "Potato___healthy"
    potato_dst = DST_DIR / "Potato___healthy"
    src_potato_count = len(list(potato_src.iterdir())) if potato_src.exists() else 0
    dst_potato_count = len(list(potato_dst.iterdir())) if potato_dst.exists() else 0
    print(f"Actual file count in Source Potato___healthy     : {src_potato_count}")
    print(f"Actual file count in Destination Potato___healthy: {dst_potato_count}")

    # 4. Image opening / corruption check on destination
    print("\n--- Image Integrity Check (Sampling Destination) ---")
    tested_count = 0
    corrupt_count = 0
    for cls in dst_classes:
        folder = DST_DIR / cls
        for i, img_path in enumerate(folder.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                if i < 20:  # test first 20 per folder = 300 sampled images
                    tested_count += 1
                    try:
                        with Image.open(img_path) as im:
                            im.verify()
                    except Exception as e:
                        corrupt_count += 1
                        print(f"  Corrupt image: {img_path} ({e})")
    
    print(f"Images sampled and verified: {tested_count}")
    print(f"Corrupt images found       : {corrupt_count}")
    print("=" * 70)

if __name__ == "__main__":
    verify_and_compare()
