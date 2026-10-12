"""
Flask Backend Application for Plant Leaf Disease Detection
- Serves interactive web UI for uploading leaf photos
- Preprocesses images to 256x256 RGB tensors (without double-normalizing)
- Runs deep learning inference via custom CNN model
- Returns disease condition, confidence score, prevention suggestions,
  and an educational disclaimer about experimental accuracy (~49.97%).
"""
import os
import json
import time
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, send_from_directory

import model_config as config

app = Flask(__name__)

# Upload configuration
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB maximum upload
app.config["UPLOAD_FOLDER"] = str(config.UPLOADS_DIR)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Global model and metadata variables
MODEL = None
CLASS_NAMES = []
DISEASE_INFO = {}

def load_system_artifacts():
    """
    Loads model weights, class indices, and disease knowledge base at startup.
    """
    global MODEL, CLASS_NAMES, DISEASE_INFO
    import tensorflow as tf

    print("=" * 60)
    print("Initializing Plant Leaf Disease Detection System...")
    print("=" * 60)

    # 1. Load class indices
    if not config.CLASS_INDICES_PATH.exists():
        raise FileNotFoundError(f"Class indices file not found: {config.CLASS_INDICES_PATH}")
    with open(config.CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)
    print(f"[OK] Loaded {len(CLASS_NAMES)} class indices.")

    # 2. Load disease prevention info
    if not config.DISEASE_INFO_PATH.exists():
        raise FileNotFoundError(f"Disease info file not found: {config.DISEASE_INFO_PATH}")
    with open(config.DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        DISEASE_INFO = json.load(f)
    print(f"[OK] Loaded metadata for {len(DISEASE_INFO)} disease classes.")

    # 3. Load trained Keras model
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {config.MODEL_PATH}")
    print(f"Loading CNN model from: {config.MODEL_PATH}...")
    MODEL = tf.keras.models.load_model(config.MODEL_PATH)
    print(f"[OK] Model loaded successfully. Input shape: {MODEL.input_shape}, Output classes: {MODEL.output_shape[-1]}")
    print("=" * 60)

# Initialize artifacts when module loads
load_system_artifacts()

def is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_uploads(max_age_seconds=600):
    """
    Safely prunes temporary uploaded images older than max_age_seconds (default 10 minutes)
    to prevent disk accumulation on cloud platforms like Render.
    Preserves .gitkeep and never raises errors if a file is being accessed.
    """
    try:
        if not config.UPLOADS_DIR.exists():
            return
        now = time.time()
        for item in config.UPLOADS_DIR.iterdir():
            if item.is_file() and item.name != ".gitkeep":
                try:
                    file_age = now - item.stat().st_mtime
                    if file_age > max_age_seconds:
                        item.unlink()
                except OSError:
                    pass
    except Exception:
        pass


def preprocess_image(image_path):
    """
    Reads and preprocesses an image file for CNN inference:
    - Verifies image integrity
    - Converts color space to 3-channel RGB
    - Resizes to (256, 256) pixels
    - Converts to float32 NumPy array with raw values in [0.0, 255.0]
    NOTE: Pixel values are NOT divided by 255 here, because the CNN model
    contains an in-graph Rescaling(1./255) layer that normalizes values once.
    """
    try:
        # 1. Verify readability
        with Image.open(image_path) as img:
            img.verify()
        
        # 2. Re-open to process (PIL verify closes/invalidates the file handle)
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(config.IMG_SIZE, Image.Resampling.BILINEAR)
            img_array = np.array(img, dtype=np.float32)

        # 3. Add batch dimension: (1, 256, 256, 3)
        img_tensor = np.expand_dims(img_array, axis=0)
        return img_tensor, None
    except Exception as e:
        return None, f"Image processing failed: {str(e)}"

@app.route("/")
def index():
    """
    Renders the homepage dashboard.
    """
    return render_template(
        "index.html",
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES
    )

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image uploads and returns disease prediction with care tips.
    Supports both multipart form data and sample file references.
    """
    if MODEL is None:
        return jsonify({"success": False, "error": "Model is not loaded."}), 500

    saved_filepath = None
    display_image_url = None

    # Check if a built-in sample was requested
    sample_name = request.form.get("sample_image")
    if sample_name:
        sample_path = config.PROJECT_ROOT / "static" / "samples" / secure_filename(sample_name)
        if sample_path.exists():
            saved_filepath = sample_path
            display_image_url = f"/static/samples/{sample_name}"
        else:
            return jsonify({
                "success": False,
                "error": f"Sample image '{sample_name}' not found in static/samples."
            }), 404

    # Otherwise, process the uploaded file
    if not saved_filepath:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part in request."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected."}), 400

        if not is_allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": f"Invalid file type. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS)).upper()}"
            }), 400

        # Prune older temporary uploads before saving the new file
        cleanup_old_uploads()

        # Secure filename and unique ID
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex[:12]}.{ext}"
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = config.UPLOADS_DIR / unique_filename

        try:
            file.save(str(dest_path))
            saved_filepath = dest_path
            display_image_url = f"/static/uploads/{unique_filename}"
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to save upload: {str(e)}"}), 500

    # Preprocess image
    img_tensor, prep_error = preprocess_image(saved_filepath)
    if prep_error:
        # Clean up temporary failed file if it was in uploads
        if "uploads" in str(saved_filepath) and os.path.exists(saved_filepath):
            try:
                os.remove(saved_filepath)
            except Exception:
                pass
        return jsonify({"success": False, "error": f"Invalid or corrupt image: {prep_error}"}), 400

    # Run CNN inference
    try:
        predictions = MODEL.predict(img_tensor, verbose=0)[0]
    except Exception as e:
        return jsonify({"success": False, "error": f"Model inference error: {str(e)}"}), 500

    # Top prediction
    top_idx = int(np.argmax(predictions))
    top_prob = float(predictions[top_idx])
    raw_class_name = CLASS_NAMES[top_idx]

    # Retrieve disease metadata
    info = DISEASE_INFO.get(raw_class_name, {
        "display_name": raw_class_name.replace("_", " ").title(),
        "crop": "Plant",
        "condition": "Unknown",
        "disease": raw_class_name,
        "description": "No description available.",
        "prevention": ["Consult an agricultural specialist for confirmation."]
    })

    # Top 3 alternative predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    top_3 = []
    for idx in top_3_indices:
        c_name = CLASS_NAMES[idx]
        c_info = DISEASE_INFO.get(c_name, {})
        top_3.append({
            "class_name": c_name,
            "display_name": c_info.get("display_name", c_name),
            "confidence": round(float(predictions[idx]) * 100, 2)
        })

    # Prepare response payload
    result = {
        "success": True,
        "image_url": display_image_url,
        "raw_class": raw_class_name,
        "display_name": info.get("display_name"),
        "crop": info.get("crop"),
        "condition": info.get("condition"),
        "disease": info.get("disease"),
        "confidence": round(top_prob * 100, 2),
        "description": info.get("description"),
        "prevention": info.get("prevention", []),
        "top_predictions": top_3,
        "disclaimer": (
            "NOTICE: Predictions are experimental and generated by a mini-project CNN "
            "with ~49.97% test accuracy on the PlantVillage dataset. Results may be inaccurate "
            "and must NOT be used as professional agricultural, diagnostic, or treatment advice."
        )
    }

    # Content negotiation: return JSON for AJAX or JSON clients, HTML for standard browser forms
    wants_json = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest" or
        "application/json" in request.headers.get("Accept", "") or
        request.args.get("format") == "json"
    )
    if wants_json:
        return jsonify(result)
    
    return render_template("index.html", result=result, num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES)

@app.route("/api/info", methods=["GET"])
def api_info():
    """
    Returns system status and supported plant disease classes.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "total_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "test_accuracy": "49.97%",
        "architecture": "Custom CNN with GlobalAveragePooling2D"
    })

# Error Handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "The uploaded file is too large. Maximum file size allowed is 10 MB."
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Requested resource not found."}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error occurred."}), 500

if __name__ == "__main__":
    # Local development server (Gunicorn is used in cloud production)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"\nStarting local Flask development server at http://{host}:{port} ...")
    app.run(host=host, port=port, debug=False)

