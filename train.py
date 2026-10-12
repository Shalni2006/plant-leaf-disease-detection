"""
Model Definition & Training Script for Plant Leaf Disease Detection
- 4 Convolutional Blocks + BatchNormalization + ReLU + MaxPooling2D
- GlobalAveragePooling2D() + Dense(128) + Dropout(0.4) + Softmax(15)
- In-model Rescaling(1./255) for seamless production inference
- Data augmentation applied ONLY during training
- Class weights applied to mitigate class imbalance
- Test set strictly isolated and untouched during training
"""
import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import model_config as config
from dataset_loader import get_datasets

def build_cnn_model(num_classes=config.NUM_CLASSES):
    """
    Constructs the custom CNN architecture.
    """
    # 1. Input Layer
    inputs = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_CHANNELS), name="input_image")

    # 2. Rescaling Layer (maps [0, 255] -> [0.0, 1.0])
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # 3. Data Augmentation Pipeline (only active during training=True)
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(x)
    x = layers.RandomRotation(0.1, name="aug_rotation")(x)
    x = layers.RandomZoom(0.1, name="aug_zoom")(x)

    # 4. Conv Block 1 (32 filters)
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # 5. Conv Block 2 (64 filters)
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # 6. Conv Block 3 (128 filters)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # 7. Conv Block 4 (128 filters)
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # 8. Classifier Head with GlobalAveragePooling2D
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense128")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="plant_leaf_disease_cnn")
    return model

def compile_model(model, learning_rate=config.LEARNING_RATE):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train(epochs=config.DEFAULT_EPOCHS, batch_size=config.BATCH_SIZE):
    print("=" * 65)
    print("    STARTING CNN MODEL TRAINING PIPELINE")
    print("=" * 65)
    
    # 1. Prepare Datasets (Test set is intentionally held back)
    train_ds, val_ds, test_ds, class_weights, classes, split_data = get_datasets(batch_size=batch_size)
    print(f"Total Classes     : {len(classes)}")
    print(f"Training Batches  : {len(train_ds)}")
    print(f"Validation Batches: {len(val_ds)}")
    print("-" * 65)

    # 2. Build & Compile Model
    model = build_cnn_model(num_classes=len(classes))
    model = compile_model(model)
    model.summary()

    # 3. Callbacks
    class EpochLogger(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            try:
                lr = self.model.optimizer.learning_rate
                if hasattr(lr, "numpy"):
                    lr_val = float(lr.numpy())
                elif hasattr(lr, "__call__"):
                    lr_val = float(lr(self.model.optimizer.iterations).numpy())
                else:
                    lr_val = float(lr)
            except Exception:
                lr_val = config.LEARNING_RATE

            loss = logs.get("loss", 0.0)
            acc = logs.get("accuracy", 0.0)
            val_loss = logs.get("val_loss", 0.0)
            val_acc = logs.get("val_accuracy", 0.0)

            print(f"\n>>> [EPOCH {epoch + 1:02d}/{epochs:02d}] "
                  f"Train Loss: {loss:.4f} | Train Acc: {acc * 100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
                  f"LR: {lr_val:.6f}")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        ),
        EpochLogger()
    ]

    # 4. Train Model (Test set is NOT passed here)
    print("\nTraining beginning (class weights applied)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=1
    )

    # 5. Save History Metrics
    history_dict = {
        "loss": [float(v) for v in history.history.get("loss", [])],
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
    }
    history_path = config.MODEL_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[OK] Training history saved to: {history_path}")
    print(f"[OK] Best model saved to: {config.MODEL_PATH}")

    return model, history_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN for Plant Leaf Disease Detection")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
