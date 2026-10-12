"""
Comprehensive Backend Test Suite for Plant Leaf Disease Detection
Tests app.py with valid, invalid, and edge-case inputs.
Distinguishes between:
1. API Pipeline Integrity (HTTP 200, schema correctness, confidence metrics)
2. Classification Correctness (Ground-truth verification vs. model predictions)
Also tests:
- 10MB max upload size limit (HTTP 413)
- Unsupported file types (HTTP 400)
- Corrupt image handling (HTTP 400)
- Missing file / empty filename (HTTP 400)
- Built-in sample buttons (HTTP 200 and 404 on missing)
- Decoupled disclaimer presence in all responses
"""
import io
import json
from pathlib import Path
from PIL import Image

from app import app
import model_config as config

def run_tests():
    print("=" * 70)
    print("         PLANT LEAF DISEASE DETECTION - BACKEND TEST SUITE")
    print("=" * 70)

    client = app.test_client()
    passed_tests = 0
    total_tests = 0

    def record_pass(test_name, details=""):
        nonlocal passed_tests, total_tests
        total_tests += 1
        passed_tests += 1
        print(f"  [PASS] {test_name}" + (f" -> {details}" if details else ""))

    # ---------------------------------------------------------
    # TEST 1: GET / (Homepage)
    # ---------------------------------------------------------
    print("\n[TEST 1] GET / (Homepage Rendering)...")
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Plant Leaf Disease Detection" in html
    assert "49.97%" in html, "Empirical test accuracy disclaimer must be visible on homepage"
    record_pass("Homepage loaded", "Status 200, UI title and experimental disclaimer present")

    # ---------------------------------------------------------
    # TEST 2: GET /api/info (System Status)
    # ---------------------------------------------------------
    print("\n[TEST 2] GET /api/info (System Metadata)...")
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["total_classes"] == 15
    assert data["test_accuracy"] == "49.97%"
    record_pass("API Info endpoint", f"Model active, 15 classes, reported accuracy: {data['test_accuracy']}")

    # ---------------------------------------------------------
    # TEST 3: POST /predict with missing file payload
    # ---------------------------------------------------------
    print("\n[TEST 3] POST /predict without file or sample parameter...")
    res = client.post("/predict")
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "No file part" in data["error"]
    record_pass("Missing payload rejected", f"HTTP 400: '{data['error']}'")

    # ---------------------------------------------------------
    # TEST 4: POST /predict with empty filename
    # ---------------------------------------------------------
    print("\n[TEST 4] POST /predict with empty filename...")
    data = {"file": (io.BytesIO(b""), "")}
    res = client.post("/predict", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    res_data = res.get_json()
    assert res_data["success"] is False
    assert "No file selected" in res_data["error"]
    record_pass("Empty filename rejected", f"HTTP 400: '{res_data['error']}'")

    # ---------------------------------------------------------
    # TEST 5: POST /predict with unsupported file extensions
    # ---------------------------------------------------------
    print("\n[TEST 5] POST /predict with unsupported file extensions...")
    unsupported = ["sample.txt", "document.pdf", "script.py", "archive.zip"]
    for filename in unsupported:
        payload = {"file": (io.BytesIO(b"fake data"), filename)}
        res = client.post("/predict", data=payload, content_type="multipart/form-data")
        assert res.status_code == 400
        res_data = res.get_json()
        assert res_data["success"] is False
        assert "Invalid file type" in res_data["error"]
    record_pass("Unsupported extensions rejected", f"Tested: {', '.join(unsupported)}")

    # ---------------------------------------------------------
    # TEST 6: POST /predict with corrupt image data
    # ---------------------------------------------------------
    print("\n[TEST 6] POST /predict with corrupt image file...")
    corrupt_bytes = b"\xFF\xD8\xFF\xE0" + b"\x00" * 50  # Invalid JPEG header
    payload = {"file": (io.BytesIO(corrupt_bytes), "corrupt.jpg")}
    res = client.post("/predict", data=payload, content_type="multipart/form-data")
    assert res.status_code == 400
    res_data = res.get_json()
    assert res_data["success"] is False
    assert "Invalid or corrupt image" in res_data["error"]
    record_pass("Corrupt image rejected", f"HTTP 400: '{res_data['error']}'")

    # ---------------------------------------------------------
    # TEST 7: POST /predict exceeding 10MB upload limit
    # ---------------------------------------------------------
    print("\n[TEST 7] POST /predict with oversized payload (>10MB)...")
    oversized_data = b"0" * (11 * 1024 * 1024)  # 11 MB
    payload = {"file": (io.BytesIO(oversized_data), "huge_image.jpg")}
    res = client.post("/predict", data=payload, content_type="multipart/form-data")
    assert res.status_code == 413, f"Expected 413 for oversized file, got {res.status_code}"
    res_data = res.get_json()
    assert "too large" in res_data["error"].lower()
    record_pass("Oversized payload rejected", f"HTTP 413: '{res_data['error']}'")

    # ---------------------------------------------------------
    # TEST 8: POST /predict using sample button parameter
    # ---------------------------------------------------------
    print("\n[TEST 8] POST /predict with valid sample parameter...")
    res = client.post("/predict", data={"sample_image": "sample_tomato_healthy.jpg"}, headers={"Accept": "application/json"})
    assert res.status_code == 200
    res_data = res.get_json()
    assert res_data["success"] is True
    assert res_data["image_url"] == "/static/samples/sample_tomato_healthy.jpg"
    record_pass("Sample button route", f"Loaded sample image with HTTP 200: '{res_data['display_name']}'")

    # ---------------------------------------------------------
    # TEST 9: POST /predict with non-existent sample parameter
    # ---------------------------------------------------------
    print("\n[TEST 9] POST /predict with non-existent sample parameter...")
    res = client.post("/predict", data={"sample_image": "non_existent.jpg"}, headers={"Accept": "application/json"})
    assert res.status_code == 404
    res_data = res.get_json()
    assert res_data["success"] is False
    record_pass("Missing sample rejected", f"HTTP 404: '{res_data['error']}'")

    # ---------------------------------------------------------
    # TEST 10: Evaluating API Integrity vs. Ground-Truth Classification
    # ---------------------------------------------------------
    print("\n[TEST 10] Testing API Pipeline & Evaluating Classification Correctness...")
    eval_cases = [
        {
            "filename": "sample_tomato_healthy.jpg",
            "ground_truth_class": "Tomato_healthy",
            "expected_condition": "Healthy"
        },
        {
            "filename": "sample_pepper_healthy.jpg",
            "ground_truth_class": "Pepper__bell___healthy",
            "expected_condition": "Healthy"
        },
        {
            "filename": "sample_tomato_early_blight.jpg",
            "ground_truth_class": "Tomato_Early_blight",
            "expected_condition": "Diseased"
        }
    ]

    for case in eval_cases:
        sample_path = config.STATIC_DIR / "samples" / case["filename"]
        assert sample_path.exists(), f"Sample file {case['filename']} missing!"
        with open(sample_path, "rb") as f:
            b_data = f.read()

        payload = {"file": (io.BytesIO(b_data), case["filename"])}
        res = client.post("/predict", data=payload, content_type="multipart/form-data", headers={"Accept": "application/json"})
        
        # 1. Pipeline Verification (Must pass for system to function)
        assert res.status_code == 200, f"API failed on {case['filename']}"
        res_data = res.get_json()
        assert res_data["success"] is True
        assert 0.0 <= res_data["confidence"] <= 100.0
        assert len(res_data["prevention"]) > 0
        assert "49.97%" in res_data["disclaimer"]

        # 2. Honest Classification Evaluation (Does prediction match ground truth?)
        predicted_class = res_data["raw_class"]
        is_exact_match = (predicted_class == case["ground_truth_class"])
        is_condition_match = (res_data["condition"] == case["expected_condition"])

        print(f"\n  Image Evaluated: {case['filename']}")
        print(f"    - Ground Truth Class : {case['ground_truth_class']} ({case['expected_condition']})")
        print(f"    - Model Prediction   : {predicted_class} ({res_data['condition']})")
        print(f"    - Confidence Score   : {res_data['confidence']}%")
        print(f"    - API Pipeline Check : PASSED (HTTP 200, valid JSON response contract)")
        
        if is_exact_match:
            print(f"    - Classification     : ACCURATE (Matches ground truth)")
        else:
            print(f"    - Classification     : INACCURATE (Misclassified - Expected with ~49.97% test accuracy)")

        record_pass(f"API integrity for {case['filename']}", f"Pipeline passed. Match: {is_exact_match}")

    # ---------------------------------------------------------
    # TEST 11: Standard Browser Form Submission (HTML Response)
    # ---------------------------------------------------------
    print("\n[TEST 11] Standard Browser Form Submission (HTML Fallback)...")
    sample_path = config.STATIC_DIR / "samples" / "sample_tomato_healthy.jpg"
    with open(sample_path, "rb") as f:
        b_data = f.read()
    payload = {"file": (io.BytesIO(b_data), "browser_upload.jpg")}
    res = client.post("/predict", data=payload, content_type="multipart/form-data")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Plant Leaf Disease Detection" in html
    assert "Diagnostic Results" in html
    record_pass("HTML Form Submission", "Renders results directly into HTML template")

    print("\n" + "=" * 70)
    print(f"RESULT: ALL {total_tests} TESTS PASSED! ({passed_tests}/{total_tests})")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
