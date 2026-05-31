import os
import sys

# Change directory to ensure we run from the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add parent directory (workspace root) to path for configurations and PyTorch models imports
sys.path.append(os.path.dirname(script_dir))

# Add the backend folder path as well to import app correctly
sys.path.append(os.path.join(script_dir, "backend"))

from backend.app import app, DEVICE

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100, help="Port to run Flask server")
    args = parser.parse_args()
    
    print("==================================================")
    print("    BILABEL PREFERENCE CLASSIFIER STUDIO v3")
    print("==================================================")
    print(f" * Compute Hardware Detect : {DEVICE.upper()}")
    print(f" * Local Server Interface   : http://127.0.0.1:{args.port}")
    print("==================================================")
    
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping BiLabel Preference Classifier server...")
