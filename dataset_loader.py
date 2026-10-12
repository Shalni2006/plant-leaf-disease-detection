"""
Dataset Loader & Stratified Data Pipeline for Plant Leaf Disease Detection
- Reproducible stratified split (70% train, 15% val, 15% test) per class
- In-place dataset reading (original images are untouched, no duplicate copies)
- Strict split isolation (no overlap between splits)
- Class weights computed strictly from training samples
- Pure model class-name index mapping saved to model/class_indices.json
"""
import os
import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf

import model_config as config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_class_names(dataset_dir=config.DATASET_DIR):
    """
    Returns the sorted list of class directory names.
    This exact alphabetical order defines the class indices [0..14].
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    if len(classes) != config.NUM_CLASSES:
        raise ValueError(
            f"Expected {config.NUM_CLASSES} class folders, but found {len(classes)} in {dataset_dir}"
        )
    return classes

def save_class_indices(class_names, output_path=config.CLASS_INDICES_PATH):
    """
    Saves pure list of class names in exact model output index order.
    Index i in the array corresponds to model output node i.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[OK] Saved model class indices to: {output_path}")

def load_class_indices(input_path=config.CLASS_INDICES_PATH):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_stratified_split_paths(
    dataset_dir=config.DATASET_DIR,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    seed=config.SEED
):
    """
    Creates reproducible, stratified partitions by index slicing.
    Guarantees zero overlap and leaves original filesystem completely untouched.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    classes = get_class_names(dataset_dir)
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    per_class_summary = {}

    for label_idx, class_name in enumerate(classes):
        class_folder = dataset_dir / class_name
        # Collect all valid image paths sorted for deterministic behavior
        all_imgs = sorted([
            str(p) for p in class_folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        
        total_cls_imgs = len(all_imgs)
        if total_cls_imgs == 0:
            raise ValueError(f"No valid images found in class folder: {class_folder}")

        # Deterministic shuffle per class
        rng = random.Random(seed + label_idx)
        shuffled = all_imgs.copy()
        rng.shuffle(shuffled)

        # Stratified index slicing
        train_end = int(round(train_ratio * total_cls_imgs))
        val_end = train_end + int(round(val_ratio * total_cls_imgs))
        # Ensure at least 1 sample in val and test if possible, and test gets remainder
        val_end = min(val_end, total_cls_imgs - 1) if total_cls_imgs >= 3 else val_end

        cls_train = shuffled[:train_end]
        cls_val = shuffled[train_end:val_end]
        cls_test = shuffled[val_end:]

        train_paths.extend(cls_train)
        train_labels.extend([label_idx] * len(cls_train))

        val_paths.extend(cls_val)
        val_labels.extend([label_idx] * len(cls_val))

        test_paths.extend(cls_test)
        test_labels.extend([label_idx] * len(cls_test))

        per_class_summary[class_name] = {
            "total": total_cls_imgs,
            "train": len(cls_train),
            "val": len(cls_val),
            "test": len(cls_test),
        }

    # Strict integrity verification: zero data leakage check
    set_train = set(train_paths)
    set_val = set(val_paths)
    set_test = set(test_paths)

    assert len(set_train & set_val) == 0, "Data leakage detected: overlap between train and val!"
    assert len(set_train & set_test) == 0, "Data leakage detected: overlap between train and test!"
    assert len(set_val & set_test) == 0, "Data leakage detected: overlap between val and test!"

    split_data = {
        "classes": classes,
        "per_class_summary": per_class_summary,
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    return split_data

def compute_training_class_weights(train_labels, num_classes=config.NUM_CLASSES):
    """
    Computes balanced class weights strictly from the training partition.
    weight[j] = N_train_total / (num_classes * N_train_j)
    """
    train_labels = np.array(train_labels)
    total_train = len(train_labels)
    class_weights = {}

    for cls_idx in range(num_classes):
        cls_count = np.sum(train_labels == cls_idx)
        if cls_count == 0:
            raise ValueError(f"Class {cls_idx} has zero samples in training set!")
        weight = total_train / (num_classes * cls_count)
        class_weights[cls_idx] = float(weight)

    return class_weights

def parse_image_and_label(path, label):
    """
    Decodes an image file into a normalized tensor.
    Reads uint8 JPEG/PNG, decodes to 3 channels, and resizes to target shape.
    Pixel values are kept in [0, 255] float32; the model's Rescaling layer converts to [0, 1].
    """
    raw_img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw_img, channels=config.NUM_CHANNELS)
    img = tf.image.resize(img, config.IMG_SIZE)
    img = tf.cast(img, tf.float32)
    return img, label

def create_tf_dataset(paths, labels, is_training=False, batch_size=config.BATCH_SIZE, seed=config.SEED):
    """
    Builds a high-performance tf.data.Dataset pipeline.
    """
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((paths_ds, labels_ds))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(paths), 2048), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds

def get_datasets(batch_size=config.BATCH_SIZE):
    """
    Main entry point for training and evaluation.
    Returns (train_ds, val_ds, test_ds, class_weights, class_names).
    """
    split_data = build_stratified_split_paths()
    classes = split_data["classes"]
    
    # Save pure class index mapping to model directory
    save_class_indices(classes)

    train_paths, train_labels = split_data["train"]
    val_paths, val_labels = split_data["val"]
    test_paths, test_labels = split_data["test"]

    class_weights = compute_training_class_weights(train_labels, len(classes))

    train_ds = create_tf_dataset(train_paths, train_labels, is_training=True, batch_size=batch_size)
    val_ds = create_tf_dataset(val_paths, val_labels, is_training=False, batch_size=batch_size)
    test_ds = create_tf_dataset(test_paths, test_labels, is_training=False, batch_size=batch_size)

    return train_ds, val_ds, test_ds, class_weights, classes, split_data
