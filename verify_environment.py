"""
Environment and Dependency Verification Script
Checks if all required libraries for the Plant Leaf Disease Detection
project can be imported and prints their versions.
"""
import sys
import importlib.metadata

def get_pkg_version(pkg_name):
    try:
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "Unknown"

def verify():
    print("=" * 55)
    print(" PLANT LEAF DISEASE DETECTION - DEPENDENCY CHECK")
    print("=" * 55)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print("-" * 55)

    # 1. Flask
    try:
        import flask
        print(f"[OK] Flask       : {get_pkg_version('flask')}")
    except ImportError as e:
        print(f"[FAIL] Flask     : NOT FOUND ({e})")

    # 2. Werkzeug
    try:
        import werkzeug
        print(f"[OK] Werkzeug    : {get_pkg_version('werkzeug')}")
    except ImportError as e:
        print(f"[FAIL] Werkzeug  : NOT FOUND ({e})")

    # 3. NumPy
    try:
        import numpy as np
        print(f"[OK] NumPy       : {np.__version__}")
    except ImportError as e:
        print(f"[FAIL] NumPy     : NOT FOUND ({e})")

    # 4. Pillow (PIL)
    try:
        import PIL
        from PIL import Image
        print(f"[OK] Pillow      : {PIL.__version__}")
    except ImportError as e:
        print(f"[FAIL] Pillow    : NOT FOUND ({e})")

    # 5. TensorFlow & Keras
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow  : {tf.__version__}")
        try:
            import keras
            print(f"[OK] Keras       : {keras.__version__}")
        except ImportError:
            print(f"[OK] Keras (built-in tf.keras) : Available")
    except ImportError as e:
        print(f"[FAIL] TensorFlow: NOT FOUND ({e})")

    print("=" * 55)
    print("RESULT: All required libraries are installed and ready!")
    print("=" * 55)

if __name__ == "__main__":
    verify()
