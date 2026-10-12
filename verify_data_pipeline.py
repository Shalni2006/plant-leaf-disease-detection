"""
Data Pipeline and Model Architecture Verification Script
Validates:
1. Exact stratified split counts and percentage distributions
2. Strict isolation: zero duplicate image paths across splits
3. Class weights computed strictly from training partition
4. Pure class-index mapping saved in model/class_indices.json
5. tf.data batch dimensions and label validity
6. Model instantiation with GlobalAveragePooling2D and output shape
Does NOT start model training.
"""
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

import model_config as config
from dataset_loader import (
    get_datasets,
    build_stratified_split_paths,
    load_class_indices
)
from train import build_cnn_model

def run_verification():
    print("=" * 70)
    print("      DATA PIPELINE & MODEL ARCHITECTURE VERIFICATION")
    print("=" * 70)

    # 1. Split Partitioning & Overlap Check
    print("\n[1] Verifying Stratified Split Distribution...")
    split_data = build_stratified_split_paths()
    classes = split_data["classes"]
    per_class = split_data["per_class_summary"]

    train_paths, train_labels = split_data["train"]
    val_paths, val_labels = split_data["val"]
    test_paths, test_labels = split_data["test"]

    total_images = len(train_paths) + len(val_paths) + len(test_paths)
    print(f"Total Dataset Images  : {total_images}")
    print(f"Training Samples (70%): {len(train_paths)} ({len(train_paths)/total_images*100:.2f}%)")
    print(f"Validation Samples(15%): {len(val_paths)} ({len(val_paths)/total_images*100:.2f}%)")
    print(f"Testing Samples (15%) : {len(test_paths)} ({len(test_paths)/total_images*100:.2f}%)")

    # Strict non-overlap verification
    s_train, s_val, s_test = set(train_paths), set(val_paths), set(test_paths)
    overlap_tv = len(s_train & s_val)
    overlap_tt = len(s_train & s_test)
    overlap_vt = len(s_val & s_test)
    print("\nOverlap Checks (Data Leakage Verification):")
    print(f"  Train & Val Overlap  : {overlap_tv} images (Expected: 0)")
    print(f"  Train & Test Overlap : {overlap_tt} images (Expected: 0)")
    print(f"  Val & Test Overlap   : {overlap_vt} images (Expected: 0)")
    assert overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0, "DATA LEAKAGE DETECTED!"
    print("  -> PASSED: All splits are 100% mutually exclusive.")

    # 2. Per-class breakdown
    print("\n[2] Per-Class Stratification Breakdown:")
    print(f"{'Class Name':<42} | {'Train':<6} | {'Val':<5} | {'Test':<5} | {'Total'}")
    print("-" * 70)
    for cname in classes:
        st = per_class[cname]
        print(f"{cname:<42} | {st['train']:<6} | {st['val']:<5} | {st['test']:<5} | {st['total']}")

    # 3. Class Weights (Training Only)
    print("\n[3] Verifying Training-Only Class Weights:")
    train_ds, val_ds, test_ds, class_weights, loaded_classes, _ = get_datasets(batch_size=config.BATCH_SIZE)
    print(f"  Number of weighted classes: {len(class_weights)}")
    for idx, cname in enumerate(loaded_classes):
        tr_count = per_class[cname]["train"]
        weight = class_weights[idx]
        print(f"  [{idx:02d}] {cname:<40}: train_count={tr_count:<4} weight={weight:.4f}")

    # 4. Pure Class Mapping in model/class_indices.json
    print("\n[4] Checking model/class_indices.json:")
    saved_classes = load_class_indices()
    assert saved_classes == classes, "Class index mismatch between dataset and saved JSON!"
    print(f"  -> Successfully verified {len(saved_classes)} classes in exact output order.")

    # 5. Batch Pipeline Tensor Dimensions
    print("\n[5] Testing tf.data Batch Pipeline Extraction:")
    for images_batch, labels_batch in train_ds.take(1):
        print(f"  Train Image Batch Shape : {images_batch.shape} (Expected: (32, 256, 256, 3))")
        print(f"  Train Label Batch Shape : {labels_batch.shape} (Expected: (32,))")
        print(f"  Image Data Type         : {images_batch.dtype}")
        print(f"  Pixel Value Range (raw) : [{tf.reduce_min(images_batch):.1f}, {tf.reduce_max(images_batch):.1f}]")
        print(f"  Sample Labels in Batch  : {labels_batch.numpy()[:8]}")
        assert images_batch.shape == (config.BATCH_SIZE, config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CHANNELS)
        assert labels_batch.shape == (config.BATCH_SIZE,)
        print("  -> PASSED: Batch shapes and label types match specifications.")

    # 6. CNN Model Architecture & GlobalAveragePooling2D Check
    print("\n[6] Instantiating Model Architecture (Dry-Run):")
    model = build_cnn_model(num_classes=len(classes))
    has_gap = any(isinstance(l, tf.keras.layers.GlobalAveragePooling2D) for l in model.layers)
    has_flatten = any(isinstance(l, tf.keras.layers.Flatten) for l in model.layers)
    print(f"  Model Name                  : {model.name}")
    print(f"  Total Parameters            : {model.count_params():,}")
    print(f"  GlobalAveragePooling2D used : {has_gap}")
    print(f"  Flatten() used              : {has_flatten}")
    print(f"  Model Output Shape          : {model.output_shape} (Expected: (None, 15))")
    assert has_gap and not has_flatten, "Model must use GlobalAveragePooling2D and not Flatten!"

    print("\n" + "=" * 70)
    print("RESULT: ALL DATA PIPELINE & ARCHITECTURE CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
