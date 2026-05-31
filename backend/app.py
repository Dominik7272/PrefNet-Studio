import os
import sys
import json
import shutil
import queue
import threading
import traceback
from datetime import datetime
from flask import Flask, jsonify, request, send_file, abort, send_from_directory
from PIL import Image, ImageOps

# Add parent workspace directories to path to import configs/inference/train if needed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PARENT_DIR)
sys.path.append(WORKSPACE_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPModel, CLIPProcessor

try:
    from configs import AnimeClassifier, get_device
except ImportError:
    class AnimeClassifier(nn.Module):
        def __init__(self, input_size):
            super(AnimeClassifier, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)
    
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device()

class DynamicAnimeClassifier(nn.Module):
    def __init__(self, input_size, layer_specs):
        super(DynamicAnimeClassifier, self).__init__()
        layers = []
        current_features = input_size
        
        for spec in layer_specs:
            ltype = spec.get("type")
            if ltype == "Linear":
                out_feats = int(spec.get("out_features", 1))
                bias = bool(spec.get("bias", True))
                layers.append(nn.Linear(current_features, out_feats, bias=bias))
                current_features = out_feats
            elif ltype == "BatchNorm1d":
                num_feats = spec.get("num_features")
                if num_feats is None or num_feats == "":
                    num_feats = current_features
                else:
                    num_feats = int(num_feats)
                eps = float(spec.get("eps", 1e-5))
                momentum = float(spec.get("momentum", 0.1))
                layers.append(nn.BatchNorm1d(num_feats, eps=eps, momentum=momentum))
            elif ltype == "ReLU":
                inplace = bool(spec.get("inplace", False))
                layers.append(nn.ReLU(inplace=inplace))
            elif ltype == "LeakyReLU":
                slope = float(spec.get("negative_slope", 0.01))
                inplace = bool(spec.get("inplace", False))
                layers.append(nn.LeakyReLU(slope, inplace=inplace))
            elif ltype == "Sigmoid":
                layers.append(nn.Sigmoid())
            elif ltype == "Tanh":
                layers.append(nn.Tanh())
            elif ltype == "Dropout":
                p = float(spec.get("p", 0.5))
                inplace = bool(spec.get("inplace", False))
                layers.append(nn.Dropout(p, inplace=inplace))
                
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".avif"}
RECENT_FILE = os.path.join(WORKSPACE_ROOT, "recent_workspaces.json")
CLIP_DEFAULT_ID = "openai/clip-vit-large-patch14"

# Global Clip Cache
class ClipCache:
    def __init__(self):
        self.model_id = None
        self.model = None
        self.processor = None
        self.lock = threading.Lock()

    def get(self, model_id, device, log_func=None):
        with self.lock:
            if self.model is not None and self.model_id == model_id:
                return self.model, self.processor
            if log_func:
                log_func(f"Loading CLIP model: {model_id} ...")
            self.model = CLIPModel.from_pretrained(model_id).to(device)
            self.model.eval()
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model_id = model_id
            if log_func:
                log_func(f"CLIP model loaded on: {device.upper()}")
            return self.model, self.processor

clip_cache = ClipCache()

# Core helper functions
def norm_path(path):
    if not path:
        return ""
    return os.path.abspath(os.path.expanduser(path))

def image_paths(folder, recursive=False):
    folder = norm_path(folder)
    if not os.path.isdir(folder):
        return []
    found = []
    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    found.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                found.append(path)
    return sorted(found, key=lambda p: os.path.basename(p).lower())

def safe_open_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def unique_dest(folder, filename):
    os.makedirs(folder, exist_ok=True)
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(folder, filename)
    index = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem}_{index}{ext}")
        index += 1
    return candidate

def get_file_hash(filepath):
    import hashlib
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception:
        return None

def find_duplicate_in_dataset(src_path, a_dir, b_dir):
    filename = os.path.basename(src_path)
    # Check filename first (fast check)
    if a_dir:
        cand_a = os.path.join(a_dir, filename)
        if os.path.isfile(cand_a):
            return "a", cand_a
    if b_dir:
        cand_b = os.path.join(b_dir, filename)
        if os.path.isfile(cand_b):
            return "b", cand_b
            
    # Check MD5 hash (content-based check)
    src_hash = get_file_hash(src_path)
    if not src_hash:
        return None, None
        
    for label_dir, loc in [(a_dir, "a"), (b_dir, "b")]:
        if not label_dir:
            continue
        try:
            for f in os.listdir(label_dir):
                fpath = os.path.join(label_dir, f)
                if os.path.isfile(fpath):
                    if get_file_hash(fpath) == src_hash:
                        return loc, fpath
        except Exception:
            pass
                    
    return None, None

def safe_folder_name(value, fallback):
    cleaned = []
    for char in (value or "").strip().lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in (" ", "-", "_"):
            cleaned.append("_")
    name = "".join(cleaned).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or fallback

def model_display_name(value):
    name = os.path.basename(value or "").strip()
    if name.lower().endswith(".pth"):
        name = os.path.splitext(name)[0]
    return name

def model_file_name(value):
    name = model_display_name(value)
    if not name:
        return ""
    return f"{name}.pth"

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)

def load_classifier_model(model_path, input_size, device):
    spec_path = model_path.replace(".pth", ".json")
    state = safe_torch_load(model_path, device)
    
    if os.path.exists(spec_path):
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                layer_specs = json.load(f)
            classifier = DynamicAnimeClassifier(input_size, layer_specs).to(device)
            classifier.load_state_dict(state)
            return classifier
        except Exception as e:
            print(f"Error loading custom architecture sidecar, falling back: {e}")
            
    # Fallback to standard AnimeClassifier
    classifier = AnimeClassifier(input_size).to(device)
    classifier.load_state_dict(state)
    return classifier

def predict_score(path, clip_model, processor, classifier, device):
    image = safe_open_image(path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features.pooler_output if hasattr(features, "pooler_output") else features
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return float(classifier(features).item())

# Flask App Initialisation
app = Flask(__name__, static_folder="../frontend/dist", static_url_path="/")

# CORS middleware for development
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Active Workspace state
active_workspace_path = None
active_workspace_data = None
label_history = []  # [(dest_path_or_None, src_path, label_key, copy_move_mode)]

# Thread locks and states
training_status = {"status": "idle", "progress": 0, "log": [], "epoch_stats": []}
training_stop_event = threading.Event()
training_thread = None

predict_status = {"status": "idle", "progress": 0, "log": [], "results": []}
predict_stop_event = threading.Event()
predict_thread = None

batch_status = {"status": "idle", "progress": 0, "log": []}
batch_stop_event = threading.Event()
batch_thread = None

# Workspace config defaults
def get_workspace_defaults(name):
    return {
        "name": name,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "settings": {
            "clip_model_id": CLIP_DEFAULT_ID,
            "active_dataset": "default",
            "datasets": [
                {"name": "default", "label_a": "Liked", "label_b": "Disliked"},
            ],
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 0.001,
            "save_path": "model",
            "load_path": "",
            "threshold": 50,
            "recursive_import": True,
            "copy_move_mode": "copy",
        },
        "scripts": [],
    }

# Helper to get paths for active workspace
def get_datasets_root():
    if not active_workspace_path:
        return None
    p = os.path.join(active_workspace_path, "datasets")
    os.makedirs(p, exist_ok=True)
    return p

def get_models_dir():
    if not active_workspace_path:
        return None
    p = os.path.join(active_workspace_path, "models")
    os.makedirs(p, exist_ok=True)
    return p

def get_workspace_models():
    models_dir = get_models_dir()
    if not models_dir or not os.path.isdir(models_dir):
        return []
    models = []
    for m in os.listdir(models_dir):
        if m.lower().endswith(".pth"):
            p = os.path.join(models_dir, m)
            try:
                stat = os.stat(p)
                size_mb = round(stat.st_size / (1024 * 1024), 2)
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                size_mb = 0.0
                modified = "Unknown"
            models.append({
                "name": model_display_name(m),
                "size_mb": size_mb,
                "modified": modified
            })
    return sorted(models, key=lambda x: x["name"].lower())

def get_dataset_folder(ds_name):
    root = get_datasets_root()
    if not root:
        return None
    return os.path.join(root, safe_folder_name(ds_name, "default"))

def get_label_dirs(ds_data):
    folder = get_dataset_folder(ds_data["name"])
    if not folder:
        return None, None
    return (
        os.path.join(folder, safe_folder_name(ds_data["label_a"], "label_100")),
        os.path.join(folder, safe_folder_name(ds_data["label_b"], "label_0")),
    )

def ensure_dataset_folders(ds_data):
    a_dir, b_dir = get_label_dirs(ds_data)
    if a_dir and b_dir:
        os.makedirs(a_dir, exist_ok=True)
        os.makedirs(b_dir, exist_ok=True)

def get_active_dataset():
    if not active_workspace_data:
        return None
    settings = active_workspace_data.get("settings", {})
    active_name = settings.get("active_dataset", "default")
    for ds in settings.get("datasets", []):
        if ds["name"] == active_name:
            return ds
    # Fallback if active dataset is missing
    if settings.get("datasets"):
        return settings["datasets"][0]
    return {"name": "default", "label_a": "Liked", "label_b": "Disliked"}

def get_workspace_stats():
    if not active_workspace_path:
        return {}
    settings = active_workspace_data.get("settings", {})
    datasets = settings.get("datasets", [])
    stats = {}
    for ds in datasets:
        a_dir, b_dir = get_label_dirs(ds)
        cnt_a = len(image_paths(a_dir)) if a_dir and os.path.isdir(a_dir) else 0
        cnt_b = len(image_paths(b_dir)) if b_dir and os.path.isdir(b_dir) else 0
        stats[ds["name"]] = {
            "count_a": cnt_a,
            "count_b": cnt_b,
            "path": get_dataset_folder(ds["name"])
        }
    return stats

# Recent workspaces list management
def get_recents():
    if not os.path.isfile(RECENT_FILE):
        return []
    try:
        with open(RECENT_FILE, "r", encoding="utf-8") as fh:
            rows = json.load(fh).get("recent", [])
        return [r for r in rows if os.path.isdir(r.get("path", ""))]
    except Exception:
        return []

def touch_recent(name, path):
    path = norm_path(path)
    rows = [r for r in get_recents() if norm_path(r.get("path", "")) != path]
    rows.insert(0, {"name": name, "path": path, "last_opened": datetime.now().strftime("%Y-%m-%d %H:%M")})
    try:
        with open(RECENT_FILE, "w", encoding="utf-8") as fh:
            json.dump({"recent": rows[:10]}, fh, indent=2)
    except Exception as e:
        print(f"Error saving recents: {e}")

# Routes: Workspace store API
@app.route('/api/workspaces/recents', methods=['GET'])
def api_get_recents():
    return jsonify(get_recents())

@app.route('/api/workspaces/open', methods=['POST'])
def api_open_workspace():
    global active_workspace_path, active_workspace_data, label_history
    req = request.json or {}
    path = norm_path(req.get("path"))
    if not path or not os.path.isdir(path):
        return jsonify({"error": "Directory does not exist."}), 400
    
    ws_file = os.path.join(path, "workspace.json")
    if not os.path.isfile(ws_file):
        return jsonify({"error": "No workspace.json found in this directory."}), 404
        
    try:
        with open(ws_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        
        # Ensure default keys
        if "settings" not in data:
            data["settings"] = get_workspace_defaults(data.get("name", "Workspace"))["settings"]
        if "scripts" not in data:
            data["scripts"] = []
            
        active_workspace_path = path
        active_workspace_data = data
        label_history.clear()
        
        # Ensure datasets folder layout exists
        datasets = data["settings"].get("datasets", [])
        for ds in datasets:
            ensure_dataset_folders(ds)
            
        touch_recent(data.get("name", os.path.basename(path)), path)
        
        # Available models
        models_detailed = get_workspace_models()
        models = [m["name"] for m in models_detailed]
        
        return jsonify({
            "active": True,
            "name": data.get("name"),
            "path": path,
            "settings": data["settings"],
            "scripts": data["scripts"],
            "models": models,
            "models_detailed": models_detailed,
            "stats": get_workspace_stats(),
            "device": DEVICE
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load workspace: {str(e)}"}), 500

@app.route('/api/workspaces/create', methods=['POST'])
def api_create_workspace():
    global active_workspace_path, active_workspace_data, label_history
    req = request.json or {}
    path = norm_path(req.get("path"))
    name = req.get("name", "").strip()
    if not path:
        return jsonify({"error": "Path is required."}), 400
    if not name:
        return jsonify({"error": "Name is required."}), 400
        
    try:
        folder_name = safe_folder_name(name, "Workspace")
        ws_path = norm_path(os.path.join(path, folder_name))
        
        os.makedirs(ws_path, exist_ok=True)
        ws_file = os.path.join(ws_path, "workspace.json")
        if os.path.exists(ws_file):
            return jsonify({"error": "Workspace already exists at this path."}), 400
            
        data = get_workspace_defaults(name)
        with open(ws_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            
        active_workspace_path = ws_path
        active_workspace_data = data
        label_history.clear()
        
        # Setup folders
        ensure_dataset_folders(data["settings"]["datasets"][0])
        touch_recent(name, ws_path)
        
        models_detailed = get_workspace_models()
        models = [m["name"] for m in models_detailed]
        
        return jsonify({
            "active": True,
            "name": name,
            "path": ws_path,
            "settings": data["settings"],
            "scripts": data["scripts"],
            "models": models,
            "models_detailed": models_detailed,
            "stats": get_workspace_stats(),
            "device": DEVICE
        })
    except Exception as e:
        return jsonify({"error": f"Failed to create workspace: {str(e)}"}), 500

@app.route('/api/workspace/info', methods=['GET'])
def api_workspace_info():
    if not active_workspace_path:
        return jsonify({"active": False})
    
    models_detailed = get_workspace_models()
    models = [m["name"] for m in models_detailed]
    
    return jsonify({
        "active": True,
        "name": active_workspace_data.get("name"),
        "path": active_workspace_path,
        "settings": active_workspace_data["settings"],
        "scripts": active_workspace_data["scripts"],
        "models": models,
        "models_detailed": models_detailed,
        "stats": get_workspace_stats(),
        "device": DEVICE
    })

@app.route('/api/workspace/save-settings', methods=['POST'])
def api_save_settings():
    if not active_workspace_path:
        return jsonify({"error": "No active workspace loaded."}), 400
    req = request.json or {}
    settings = req.get("settings")
    scripts = req.get("scripts")
    
    if settings:
        active_workspace_data["settings"].update(settings)
    if scripts is not None:
        active_workspace_data["scripts"] = scripts
        
    try:
        ws_file = os.path.join(active_workspace_path, "workspace.json")
        with open(ws_file, "w", encoding="utf-8") as fh:
            json.dump(active_workspace_data, fh, indent=2)
        return jsonify({"success": True, "settings": active_workspace_data["settings"], "stats": get_workspace_stats()})
    except Exception as e:
        return jsonify({"error": f"Failed to save settings: {str(e)}"}), 500

# Routes: Dataset management API
@app.route('/api/workspace/dataset/add', methods=['POST'])
def api_add_dataset():
    if not active_workspace_path:
        return jsonify({"error": "No active workspace loaded."}), 400
    req = request.json or {}
    name = req.get("name", "").strip()
    label_a = req.get("label_a", "Liked").strip()
    label_b = req.get("label_b", "Disliked").strip()
    
    if not name:
        return jsonify({"error": "Dataset name is required."}), 400
        
    settings = active_workspace_data["settings"]
    datasets = settings.setdefault("datasets", [])
    
    # Check duplicate
    if any(ds["name"].lower() == name.lower() for ds in datasets):
        return jsonify({"error": f"Dataset profile '{name}' already exists."}), 400
        
    new_ds = {"name": name, "label_a": label_a, "label_b": label_b}
    datasets.append(new_ds)
    ensure_dataset_folders(new_ds)
    
    # Save workspace
    try:
        ws_file = os.path.join(active_workspace_path, "workspace.json")
        with open(ws_file, "w", encoding="utf-8") as fh:
            json.dump(active_workspace_data, fh, indent=2)
        return jsonify({"success": True, "settings": settings, "stats": get_workspace_stats()})
    except Exception as e:
        return jsonify({"error": f"Failed to add dataset: {str(e)}"}), 500

@app.route('/api/workspace/dataset/delete', methods=['POST'])
def api_delete_dataset():
    if not active_workspace_path:
        return jsonify({"error": "No active workspace loaded."}), 400
    req = request.json or {}
    name = req.get("name", "").strip()
    delete_files = bool(req.get("delete_files", False))
    
    settings = active_workspace_data["settings"]
    datasets = settings.get("datasets", [])
    
    if len(datasets) <= 1:
        return jsonify({"error": "You must keep at least one dataset profile profile active."}), 400
        
    ds_idx = -1
    for idx, ds in enumerate(datasets):
        if ds["name"] == name:
            ds_idx = idx
            break
            
    if ds_idx == -1:
        return jsonify({"error": f"Dataset profile '{name}' not found."}), 404
        
    # Delete folder if requested
    if delete_files:
        ds_folder = get_dataset_folder(name)
        if ds_folder and os.path.isdir(ds_folder):
            try:
                shutil.rmtree(ds_folder)
            except Exception as e:
                return jsonify({"error": f"Failed to delete files: {str(e)}"}), 500
                
    datasets.pop(ds_idx)
    
    # Update active dataset if it was deleted
    if settings.get("active_dataset") == name:
        settings["active_dataset"] = datasets[0]["name"]
        
    try:
        ws_file = os.path.join(active_workspace_path, "workspace.json")
        with open(ws_file, "w", encoding="utf-8") as fh:
            json.dump(active_workspace_data, fh, indent=2)
        return jsonify({"success": True, "settings": settings, "stats": get_workspace_stats()})
    except Exception as e:
        return jsonify({"error": f"Failed to delete dataset: {str(e)}"}), 500

@app.route('/api/workspace/dataset/save', methods=['POST'])
def api_save_dataset_profile():
    if not active_workspace_path:
        return jsonify({"error": "No active workspace loaded."}), 400
    req = request.json or {}
    old_name = req.get("old_name", "").strip()
    new_name = req.get("name", "").strip()
    new_label_a = req.get("label_a", "Liked").strip()
    new_label_b = req.get("label_b", "Disliked").strip()
    
    if not old_name or not new_name:
        return jsonify({"error": "Dataset names are required."}), 400
        
    settings = active_workspace_data["settings"]
    datasets = settings.get("datasets", [])
    
    target_ds = None
    for ds in datasets:
        if ds["name"] == old_name:
            target_ds = ds
            break
            
    if not target_ds:
        return jsonify({"error": f"Dataset profile '{old_name}' not found."}), 404
        
    # Rename folder files if rename occurred
    old_data = {"name": old_name, "label_a": target_ds["label_a"], "label_b": target_ds["label_b"]}
    new_data = {"name": new_name, "label_a": new_label_a, "label_b": new_label_b}
    
    old_a_dir, old_b_dir = get_label_dirs(old_data)
    new_a_dir, new_b_dir = get_label_dirs(new_data)
    
    # Move files to new naming layout
    for o_dir, n_dir in [(old_a_dir, new_a_dir), (old_b_dir, new_b_dir)]:
        if o_dir and n_dir and norm_path(o_dir) != norm_path(n_dir) and os.path.isdir(o_dir):
            os.makedirs(n_dir, exist_ok=True)
            for f in os.listdir(o_dir):
                src = os.path.join(o_dir, f)
                if os.path.isfile(src):
                    try:
                        shutil.move(src, unique_dest(n_dir, f))
                    except Exception as e:
                        print(f"Error migrating files: {e}")
            try:
                os.rmdir(o_dir)
            except OSError:
                pass
                
    old_ds = get_dataset_folder(old_name)
    new_ds = get_dataset_folder(new_name)
    if old_ds and new_ds and norm_path(old_ds) != norm_path(new_ds) and os.path.isdir(old_ds):
        try:
            if not os.listdir(old_ds):
                os.rmdir(old_ds)
        except OSError:
            pass
            
    # Update config
    target_ds["name"] = new_name
    target_ds["label_a"] = new_label_a
    target_ds["label_b"] = new_label_b
    
    if settings.get("active_dataset") == old_name:
        settings["active_dataset"] = new_name
        
    ensure_dataset_folders(target_ds)
    
    try:
        ws_file = os.path.join(active_workspace_path, "workspace.json")
        with open(ws_file, "w", encoding="utf-8") as fh:
            json.dump(active_workspace_data, fh, indent=2)
        return jsonify({"success": True, "settings": settings, "stats": get_workspace_stats()})
    except Exception as e:
        return jsonify({"error": f"Failed to save dataset: {str(e)}"}), 500

@app.route('/api/workspace/dataset/activate', methods=['POST'])
def api_activate_dataset():
    if not active_workspace_data:
        return jsonify({"error": "No workspace loaded."}), 400
    req = request.json or {}
    name = req.get("name", "").strip()
    
    settings = active_workspace_data["settings"]
    if not any(ds["name"] == name for ds in settings.get("datasets", [])):
        return jsonify({"error": f"Dataset profile '{name}' not found."}), 404
        
    settings["active_dataset"] = name
    
    try:
        ws_file = os.path.join(active_workspace_path, "workspace.json")
        with open(ws_file, "w", encoding="utf-8") as fh:
            json.dump(active_workspace_data, fh, indent=2)
        return jsonify({"success": True, "settings": settings, "stats": get_workspace_stats()})
    except Exception as e:
        return jsonify({"error": f"Failed to activate dataset: {str(e)}"}), 500

# Routes: File system browser API
@app.route('/api/fs/browse', methods=['GET'])
def api_fs_browse():
    path_param = request.args.get("path", "").strip()
    
    # If path is empty, return user's home directory + logical drives on Windows
    if not path_param:
        home = norm_path("~")
        drives = []
        if sys.platform == "win32":
            import string
            # We can check logical drives
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append(drive)
        return jsonify({
            "current_path": home,
            "parent_path": os.path.dirname(home),
            "folders": [{"name": f, "path": os.path.join(home, f)} for f in sorted(os.listdir(home)) if os.path.isdir(os.path.join(home, f)) and not f.startswith(".")] if os.path.isdir(home) else [],
            "drives": drives,
            "error": None
        })
        
    target = norm_path(path_param)
    if not os.path.isdir(target):
        return jsonify({"error": f"Directory '{target}' does not exist."}), 404
        
    try:
        folders = []
        for f in sorted(os.listdir(target)):
            full_p = os.path.join(target, f)
            if os.path.isdir(full_p):
                # Hide system folders
                if not f.startswith(".") and f.lower() not in ["$recycle.bin", "system volume information"]:
                    folders.append({"name": f, "path": full_p})
        return jsonify({
            "current_path": target,
            "parent_path": os.path.dirname(target) if os.path.dirname(target) != target else "",
            "folders": folders,
            "drives": [],
            "error": None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to stream image safely
@app.route('/api/fs/image', methods=['GET'])
def api_get_image():
    img_path = request.args.get("path", "").strip()
    if not img_path or not os.path.isfile(img_path):
        # Fallback: check if the image has been moved to active dataset class A or B folder
        if active_workspace_path and active_workspace_data:
            try:
                ds = get_active_dataset()
                a_dir, b_dir = get_label_dirs(ds)
                filename = os.path.basename(img_path)
                cand_a = os.path.join(a_dir, filename) if a_dir else ""
                cand_b = os.path.join(b_dir, filename) if b_dir else ""
                if cand_a and os.path.isfile(cand_a):
                    img_path = cand_a
                elif cand_b and os.path.isfile(cand_b):
                    img_path = cand_b
            except Exception:
                pass
                
    if not img_path or not os.path.isfile(img_path):
        return abort(404)
        
    ext = os.path.splitext(img_path)[1].lower()
    if ext not in IMAGE_EXTS:
        return abort(400, "Unsupported image type")
        
    try:
        return send_file(img_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to get images inside a specific dataset class directory
@app.route('/api/workspace/dataset/gallery', methods=['GET'])
def api_dataset_gallery():
    if not active_workspace_path:
        return jsonify({"error": "No workspace loaded."}), 400
    ds_name = request.args.get("dataset", "").strip()
    label_class = request.args.get("label_class", "a").strip()
    
    settings = active_workspace_data.get("settings", {})
    datasets = settings.get("datasets", [])
    ds_data = None
    for ds in datasets:
        if ds["name"] == ds_name:
            ds_data = ds
            break
    if not ds_data:
        return jsonify({"error": f"Dataset profile '{ds_name}' not found."}), 404
        
    a_dir, b_dir = get_label_dirs(ds_data)
    target_dir = a_dir if label_class == "a" else b_dir
    
    if not target_dir or not os.path.isdir(target_dir):
        return jsonify({"images": []})
        
    paths = image_paths(target_dir, recursive=False)
    return jsonify({"images": paths})

# Route to execute move/delete action on image inside a dataset class directory
@app.route('/api/workspace/dataset/gallery/action', methods=['POST'])
def api_dataset_gallery_action():
    if not active_workspace_path:
        return jsonify({"error": "No workspace loaded."}), 400
    req = request.json or {}
    ds_name = req.get("dataset", "").strip()
    label_class = req.get("label_class", "a").strip()
    img_path = req.get("image_path", "").strip()
    action = req.get("action", "").strip() # "delete" or "move"
    
    if not img_path or not os.path.isfile(img_path):
        return jsonify({"error": f"Image file '{img_path}' not found."}), 404
        
    settings = active_workspace_data.get("settings", {})
    datasets = settings.get("datasets", [])
    ds_data = None
    for ds in datasets:
        if ds["name"] == ds_name:
            ds_data = ds
            break
    if not ds_data:
        return jsonify({"error": f"Dataset profile '{ds_name}' not found."}), 404
        
    a_dir, b_dir = get_label_dirs(ds_data)
    source_dir = a_dir if label_class == "a" else b_dir
    dest_dir = b_dir if label_class == "a" else a_dir
    
    try:
        if action == "delete":
            os.remove(img_path)
        elif action == "move":
            # Check for duplicates in the target folder (either by filename or MD5)
            dup_loc, dup_path = find_duplicate_in_dataset(img_path, dest_dir, None)
            if dup_loc is not None:
                # Duplicate already exists in destination class folder, just delete source
                os.remove(img_path)
            else:
                dest_path = unique_dest(dest_dir, os.path.basename(img_path))
                shutil.move(img_path, dest_path)
        else:
            return jsonify({"error": "Action must be 'delete' or 'move'."}), 400
            
        return jsonify({"success": True, "stats": get_workspace_stats()})
    except Exception as e:
        return jsonify({"error": f"Operation failed: {str(e)}"}), 500

# Routes: Keyboard labeling API
@app.route('/api/workspace/label/queue', methods=['GET'])
def api_label_queue():
    if not active_workspace_path:
        return jsonify({"error": "No workspace loaded"}), 400
    folder_param = request.args.get("folder", "").strip()
    recursive = request.args.get("recursive", "true").lower() == "true"
    
    if not folder_param:
        return jsonify({"error": "Folder path is required."}), 400
        
    folders = [f.strip() for f in folder_param.split(";") if f.strip()]
    if not folders:
        return jsonify({"error": "No folder paths provided."}), 400
        
    all_paths = []
    for f in folders:
        if not os.path.isdir(f):
            return jsonify({"error": f"Directory '{f}' not found."}), 400
        all_paths.extend(image_paths(f, recursive))
        
    # Deduplicate while preserving order
    seen = set()
    deduped_paths = []
    for p in all_paths:
        p_norm = norm_path(p)
        if p_norm not in seen:
            seen.add(p_norm)
            deduped_paths.append(p)
            
    return jsonify({
        "total": len(deduped_paths),
        "paths": deduped_paths
    })

@app.route('/api/workspace/label/action', methods=['POST'])
def api_label_action():
    global label_history
    if not active_workspace_path:
        return jsonify({"error": "No workspace loaded"}), 400
    req = request.json or {}
    src = req.get("src", "").strip()
    action = req.get("action", "").strip() # "a" or "b" or "skip"
    mode = req.get("mode", "copy").strip() # "copy" or "move"
    
    if not src:
        return jsonify({"error": "Source path is required."}), 400
        
    if action not in ["a", "b", "skip"]:
        return jsonify({"error": "Action must be 'a', 'b', or 'skip'."}), 400
        
    try:
        ds = get_active_dataset()
        a_dir, b_dir = get_label_dirs(ds)
        filename = os.path.basename(src)
        
        # 1. Determine where the file is currently sitting
        actual_src = None
        current_loc = None # "unlabelled", "a", "b"
        
        if os.path.isfile(src):
            actual_src = src
            current_loc = "unlabelled"
        else:
            # Check if it was already moved to label_a or label_b directory
            cand_a = os.path.join(a_dir, filename) if a_dir else ""
            cand_b = os.path.join(b_dir, filename) if b_dir else ""
            if cand_a and os.path.isfile(cand_a):
                actual_src = cand_a
                current_loc = "a"
            elif cand_b and os.path.isfile(cand_b):
                actual_src = cand_b
                current_loc = "b"
                
        # If the file cannot be found anywhere, return error
        if not actual_src:
            return jsonify({"error": f"Image file '{filename}' could not be located in source or dataset directories."}), 400
            
        # 2. Perform the labeling/move/copy based on action and current location
        if action == "skip":
            if current_loc == "a" or current_loc == "b":
                if mode == "move":
                    # Move it back to original unlabelled folder
                    shutil.move(actual_src, src)
                else:
                    # In copy mode, just delete the copied file from dataset
                    os.remove(actual_src)
            # Add to backend history for bookkeeping
            label_history.append((None, src, "skip", None))
            return jsonify({"success": True, "history_len": len(label_history), "stats": get_workspace_stats()})
            
        target_dir = a_dir if action == "a" else b_dir
        dest = os.path.join(target_dir, filename)
        
        # Check for duplicates in liked/disliked dataset (by filename or MD5 hash)
        dup_loc, dup_path = find_duplicate_in_dataset(actual_src, a_dir, b_dir)
        
        if dup_loc is not None:
            # Duplicate found in the dataset!
            if dup_loc == action:
                # Already in the target folder, skip copying/moving to prevent duplicate
                dest = dup_path
                if current_loc == "unlabelled" and mode == "move":
                    try:
                        os.remove(actual_src)
                    except Exception:
                        pass
            else:
                # Exists in the opposite folder, move the existing file to the target folder
                dest = unique_dest(target_dir, os.path.basename(dup_path))
                shutil.move(dup_path, dest)
                if current_loc == "unlabelled" and mode == "move":
                    try:
                        os.remove(actual_src)
                    except Exception:
                        pass
        else:
            # No duplicate in the dataset, normal flow
            if current_loc == "unlabelled":
                dest = unique_dest(target_dir, filename)
                if mode == "move":
                    shutil.move(actual_src, dest)
                else:
                    shutil.copy2(actual_src, dest)
            elif current_loc != action: # "a" -> "b" or "b" -> "a"
                # Move the file from one label folder to the other
                dest = unique_dest(target_dir, filename)
                shutil.move(actual_src, dest)
            else:
                # Already in the correct location
                dest = actual_src
            
        label_history.append((dest, src, action, mode))
        return jsonify({"success": True, "history_len": len(label_history), "stats": get_workspace_stats()})
    except Exception as e:
        return jsonify({"error": f"IO operation failed: {str(e)}"}), 500

@app.route('/api/workspace/label/undo', methods=['POST'])
def api_label_undo():
    global label_history
    if not label_history:
        return jsonify({"error": "No labeling actions in history."}), 400
        
    dest, src, action, mode = label_history.pop()
    
    try:
        if dest and os.path.isfile(dest):
            if mode == "move":
                shutil.move(dest, src)
            else:
                os.remove(dest)
        return jsonify({"success": True, "undone": {"src": src, "action": action}, "history_len": len(label_history), "stats": get_workspace_stats()})
    except Exception as e:
        # Put it back on stack if it failed
        label_history.append((dest, src, action, mode))
        return jsonify({"error": f"Undo failed: {str(e)}"}), 500

# Background Training thread runner
def training_runner(settings, save_path, ds_data, a_paths, b_paths):
    global training_status
    try:
        device = settings.get("device", DEVICE)
        batch_size = max(1, int(settings.get("batch_size", 32)))
        epochs = max(1, int(settings.get("epochs", 20)))
        lr = float(settings.get("learning_rate", 0.001))
        clip_id = settings.get("clip_model_id", CLIP_DEFAULT_ID)
        
        label_a_name = ds_data["label_a"]
        label_b_name = ds_data["label_b"]
        
        training_status["status"] = "running"
        training_status["progress"] = 1
        training_status["log"].append("Initializing training runner...")
        
        # Load CLIP
        training_status["log"].append(f"Loading HuggingFace CLIP model: {clip_id} ...")
        clip_model, processor = clip_cache.get(clip_id, device, lambda m: training_status["log"].append(m))
        
        training_status["log"].append(f"CLIP dimensions: {clip_model.config.projection_dim} features")
        training_status["log"].append(f"Dataset stats: {len(a_paths)} Liked ({label_a_name}), {len(b_paths)} Disliked ({label_b_name})")
        
        if training_stop_event.is_set():
            training_status["status"] = "cancelled"
            training_status["log"].append("Training stopped by user.")
            return

        total_batches = ((len(a_paths) + batch_size - 1) // batch_size) + ((len(b_paths) + batch_size - 1) // batch_size)
        
        # Extract Class A
        training_status["log"].append(f"Extracting embeddings for Class A ({label_a_name})...")
        a_embeddings = []
        for start in range(0, len(a_paths), batch_size):
            if training_stop_event.is_set():
                training_status["status"] = "cancelled"
                training_status["log"].append("Training stopped by user.")
                return
            batch = a_paths[start:start+batch_size]
            imgs = []
            for p in batch:
                try:
                    imgs.append(safe_open_image(p))
                except Exception as exc:
                    training_status["log"].append(f"Skipped image {os.path.basename(p)}: {exc}")
            if not imgs:
                continue
            inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = clip_model.get_image_features(**inputs)
                out = out.pooler_output if hasattr(out, "pooler_output") else out
                out = out / out.norm(p=2, dim=-1, keepdim=True)
                a_embeddings.append(out.cpu())
                
            done_b = (start // batch_size) + 1
            pct = int((done_b / total_batches) * 45)
            training_status["progress"] = pct
            
        if not a_embeddings:
            training_status["status"] = "error"
            training_status["log"].append("Error: No valid Class A embeddings generated.")
            return
            
        # Extract Class B
        training_status["log"].append(f"Extracting embeddings for Class B ({label_b_name})...")
        b_embeddings = []
        for start in range(0, len(b_paths), batch_size):
            if training_stop_event.is_set():
                training_status["status"] = "cancelled"
                training_status["log"].append("Training stopped by user.")
                return
            batch = b_paths[start:start+batch_size]
            imgs = []
            for p in batch:
                try:
                    imgs.append(safe_open_image(p))
                except Exception as exc:
                    training_status["log"].append(f"Skipped image {os.path.basename(p)}: {exc}")
            if not imgs:
                continue
            inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = clip_model.get_image_features(**inputs)
                out = out.pooler_output if hasattr(out, "pooler_output") else out
                out = out / out.norm(p=2, dim=-1, keepdim=True)
                b_embeddings.append(out.cpu())
                
            done_b = ((len(a_paths) + batch_size - 1) // batch_size) + (start // batch_size) + 1
            pct = int((done_b / total_batches) * 45)
            training_status["progress"] = pct
            
        if not b_embeddings:
            training_status["status"] = "error"
            training_status["log"].append("Error: No valid Class B embeddings generated.")
            return
            
        a_x = torch.cat(a_embeddings)
        b_x = torch.cat(b_embeddings)
        x = torch.cat([a_x, b_x])
        y = torch.cat([torch.ones(len(a_x), 1), torch.zeros(len(b_x), 1)])
        
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
        custom_layers = settings.get("custom_layers")
        if custom_layers:
            classifier = DynamicAnimeClassifier(x.shape[1], custom_layers).to(device)
        else:
            classifier = AnimeClassifier(x.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(classifier.parameters(), lr=lr)
        
        training_status["log"].append("Training Neural Network weights...")
        for epoch in range(epochs):
            if training_stop_event.is_set():
                training_status["status"] = "cancelled"
                training_status["log"].append("Training stopped by user.")
                return
            classifier.train()
            loss_sum = 0.0
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item())
                correct += int(((outputs > 0.5).float() == labels).sum().item())
                total += int(labels.size(0))
                
            avg_loss = loss_sum / max(1, len(loader))
            acc = 100.0 * correct / max(1, total)
            log_line = f"Epoch {epoch + 1}/{epochs} - loss={avg_loss:.4f}  accuracy={acc:.1f}%"
            training_status["log"].append(log_line)
            training_status["epoch_stats"].append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc})
            training_status["progress"] = 45 + int(((epoch + 1) / epochs) * 53)
            
        # Save model
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        classifier.eval()
        torch.save(classifier.state_dict(), save_path)
        
        # Save sidecar architecture spec if custom_layers is defined
        custom_layers = settings.get("custom_layers")
        if custom_layers:
            spec_path = save_path.replace(".pth", ".json")
            try:
                with open(spec_path, "w", encoding="utf-8") as f:
                    json.dump(custom_layers, f, indent=2)
                training_status["log"].append(f"Model sidecar specification saved to: {os.path.basename(spec_path)}")
            except Exception as e:
                print(f"Error saving model sidecar spec: {e}")
                
        training_status["log"].append(f"Model saved successfully to: {os.path.basename(save_path)}")
        training_status["progress"] = 100
        training_status["status"] = "completed"
        
    except Exception as e:
        traceback.print_exc()
        training_status["status"] = "error"
        training_status["log"].append(f"Exception encountered: {str(e)}")

# Routes: Model Training API
@app.route('/api/workspace/train/start', methods=['POST'])
def api_train_start():
    global training_thread, training_status
    if not active_workspace_path:
        return jsonify({"error": "No workspace loaded."}), 400
        
    if training_status["status"] == "running":
        return jsonify({"error": "Training is already in progress."}), 400
        
    # Get configuration settings
    settings = active_workspace_data["settings"]
    ds = get_active_dataset()
    a_dir, b_dir = get_label_dirs(ds)
    
    a_paths = image_paths(a_dir) if a_dir else []
    b_paths = image_paths(b_dir) if b_dir else []
    
    if not a_paths or not b_paths:
        return jsonify({"error": f"Training requires images in both folders. Please label images for '{ds['label_a']}' and '{ds['label_b']}' first."}), 400
        
    model_name = request.json.get("model_name", "model").strip()
    if not model_name:
        return jsonify({"error": "Model save name is required."}), 400
        
    save_path = os.path.join(get_models_dir(), model_file_name(model_name))
    settings["save_path"] = save_path
    settings["device"] = DEVICE
    
    # Save the selected save name to workspace settings
    settings["load_path"] = model_file_name(model_name)
    try:
        ws_file = os.path.join(active_workspace_path, "workspace.json")
        with open(ws_file, "w", encoding="utf-8") as fh:
            json.dump(active_workspace_data, fh, indent=2)
    except Exception as e:
        print(f"Error saving workspace configuration: {e}")
        
    training_status = {"status": "running", "progress": 0, "log": [], "epoch_stats": []}
    training_stop_event.clear()
    
    training_thread = threading.Thread(
        target=training_runner,
        args=(dict(settings), save_path, ds, a_paths, b_paths),
        daemon=True
    )
    training_thread.start()
    return jsonify({"success": True, "status": training_status})

@app.route('/api/workspace/train/stop', methods=['POST'])
def api_train_stop():
    if training_status["status"] != "running":
        return jsonify({"error": "Training is not running."}), 400
    training_stop_event.set()
    return jsonify({"success": True})

@app.route('/api/workspace/train/status', methods=['GET'])
def api_train_status():
    return jsonify(training_status)

# Background Inference runner
def inference_runner(paths, model_path, settings):
    global predict_status
    try:
        device = settings.get("device", DEVICE)
        clip_id = settings.get("clip_model_id", CLIP_DEFAULT_ID)
        
        predict_status["status"] = "running"
        predict_status["progress"] = 1
        predict_status["log"].append("Initializing prediction classifier weights...")
        
        clip_model, processor = clip_cache.get(clip_id, device, lambda m: predict_status["log"].append(m))
        
        classifier = load_classifier_model(model_path, clip_model.config.projection_dim, device)
        classifier.eval()
        
        total = len(paths)
        predict_status["log"].append(f"Running prediction on {total} items using CLIP model...")
        
        results = []
        for index, path in enumerate(paths, 1):
            if predict_stop_event.is_set():
                predict_status["status"] = "cancelled"
                predict_status["log"].append("Prediction cancelled by user.")
                return
            try:
                score = predict_score(path, clip_model, processor, classifier, device)
                results.append({"path": path, "score": score, "error": None})
            except Exception as e:
                results.append({"path": path, "score": None, "error": str(e)})
            predict_status["results"] = results
            predict_status["progress"] = int(index / max(1, total) * 100)
            
        predict_status["log"].append("Prediction complete.")
        predict_status["status"] = "completed"
    except Exception as e:
        traceback.print_exc()
        predict_status["status"] = "error"
        predict_status["log"].append(f"Prediction failed: {str(e)}")

# Routes: Predictions/Inference API
@app.route('/api/workspace/predict/start', methods=['POST'])
def api_predict_start():
    global predict_thread, predict_status
    if not active_workspace_path:
        return jsonify({"error": "No workspace loaded."}), 400
        
    if predict_status["status"] == "running":
        return jsonify({"error": "Inference is already in progress."}), 400
        
    req = request.json or {}
    paths = req.get("paths", [])
    folder = req.get("folder", "").strip()
    model_name = req.get("model", "").strip()
    
    if folder and os.path.isdir(folder):
        recursive = bool(req.get("recursive", True))
        paths = image_paths(folder, recursive)
        
    if not paths:
        return jsonify({"error": "No files or folders containing image media were specified."}), 400
        
    if not model_name:
        return jsonify({"error": "Active model weights must be selected."}), 400
        
    model_path = os.path.join(get_models_dir(), model_file_name(model_name))
    if not os.path.isfile(model_path):
        return jsonify({"error": f"Model weights '{model_name}' not found."}), 404
        
    settings = dict(active_workspace_data["settings"])
    settings["device"] = DEVICE
    
    predict_status = {"status": "running", "progress": 0, "log": [], "results": []}
    predict_stop_event.clear()
    
    predict_thread = threading.Thread(
        target=inference_runner,
        args=(paths, model_path, settings),
        daemon=True
    )
    predict_thread.start()
    return jsonify({"success": True, "status": predict_status})

@app.route('/api/workspace/predict/stop', methods=['POST'])
def api_predict_stop():
    if predict_status["status"] != "running":
        return jsonify({"error": "Inference is not running."}), 400
    predict_stop_event.set()
    return jsonify({"success": True})

@app.route('/api/workspace/predict/status', methods=['GET'])
def api_predict_status():
    return jsonify(predict_status)

@app.route('/api/workspace/predict/action', methods=['POST'])
def api_predict_action():
    # Allows a manual override / copy or move from bulk prediction grid
    if not active_workspace_path:
        return jsonify({"error": "No workspace loaded."}), 400
    req = request.json or {}
    src = req.get("src", "").strip()
    action = req.get("action", "").strip() # "a" or "b" or "skip"
    mode = req.get("mode", "copy").strip() # "copy" or "move"
    
    if not src or not os.path.isfile(src):
        return jsonify({"error": f"Source file '{src}' does not exist."}), 400
        
    if action not in ["a", "b", "skip"]:
        return jsonify({"error": "Action must be 'a', 'b', or 'skip'."}), 400
        
    try:
        if action == "skip":
            return jsonify({"success": True, "stats": get_workspace_stats()})
            
        ds = get_active_dataset()
        a_dir, b_dir = get_label_dirs(ds)
        target_dir = a_dir if action == "a" else b_dir
        
        # Check for duplicates in liked/disliked dataset (by filename or MD5 hash)
        dup_loc, dup_path = find_duplicate_in_dataset(src, a_dir, b_dir)
        
        if dup_loc is not None:
            # Duplicate found in the dataset!
            if dup_loc == action:
                # Already in the target folder, skip copying/moving to prevent duplicate
                dest = dup_path
                if mode == "move":
                    try:
                        os.remove(src)
                    except Exception:
                        pass
            else:
                # Exists in the opposite folder, move the existing file to the target folder
                dest = unique_dest(target_dir, os.path.basename(dup_path))
                shutil.move(dup_path, dest)
                if mode == "move":
                    try:
                        os.remove(src)
                    except Exception:
                        pass
        else:
            dest = unique_dest(target_dir, os.path.basename(src))
            if mode == "move":
                shutil.move(src, dest)
            else:
                shutil.copy2(src, dest)
            
        return jsonify({"success": True, "stats": get_workspace_stats()})
    except Exception as e:
        return jsonify({"error": f"IO move/copy failed: {str(e)}"}), 500

# Background Batch Jobs runner
def batch_runner(scripts, settings, models_dir):
    global batch_status
    try:
        device = settings.get("device", DEVICE)
        clip_id = settings.get("clip_model_id", CLIP_DEFAULT_ID)
        
        batch_status["status"] = "running"
        batch_status["progress"] = 1
        batch_status["log"].append("Initializing batch worker settings...")
        
        clip_model, processor = clip_cache.get(clip_id, device, lambda m: batch_status["log"].append(m))
        
        # Calculate total images to process
        total_files = 0
        for s in scripts:
            total_files += len(image_paths(s["input"], True))
            
        batch_status["log"].append(f"Total files in queue across {len(scripts)} jobs: {total_files}")
        
        completed = 0
        copy_mode = settings.get("copy_move_mode", "copy")
        
        for s_idx, script in enumerate(scripts, 1):
            if batch_stop_event.is_set():
                batch_status["status"] = "cancelled"
                batch_status["log"].append("Batch runner terminated by user.")
                return
                
            model_name = script["model"]
            model_path = os.path.join(models_dir, model_file_name(model_name))
            if not os.path.isfile(model_path):
                batch_status["log"].append(f"Skipping Job {s_idx}: Weight file '{model_name}' not found.")
                continue
                
            batch_status["log"].append(f"Job {s_idx}/{len(scripts)}: Loading model weights {model_name}...")
            classifier = load_classifier_model(model_path, clip_model.config.projection_dim, device)
            classifier.eval()
            
            label_a_name = script.get("label_a_name") or settings.get("label_a", "Liked")
            label_b_name = script.get("label_b_name") or settings.get("label_b", "Disliked")
            
            label_a_dir = os.path.join(script["output"], safe_folder_name(label_a_name, "class_a"))
            label_b_dir = os.path.join(script["output"], safe_folder_name(label_b_name, "class_b"))
            os.makedirs(label_a_dir, exist_ok=True)
            os.makedirs(label_b_dir, exist_ok=True)
            
            paths = image_paths(script["input"], True)
            threshold = max(0.0, min(1.0, float(script["threshold"]) / 100.0))
            
            batch_status["log"].append(f"Job {s_idx}: Processing {len(paths)} files from '{script['input']}'...")
            
            for path in paths:
                if batch_stop_event.is_set():
                    batch_status["status"] = "cancelled"
                    batch_status["log"].append("Batch runner terminated by user.")
                    return
                try:
                    score = predict_score(path, clip_model, processor, classifier, device)
                    target = label_a_dir if score >= threshold else label_b_dir
                    
                    dest = unique_dest(target, os.path.basename(path))
                    if copy_mode == "move":
                        shutil.move(path, dest)
                    else:
                        shutil.copy2(path, dest)
                except Exception as exc:
                    batch_status["log"].append(f"Skipped file {os.path.basename(path)} due to error: {exc}")
                
                completed += 1
                batch_status["progress"] = int(completed / max(1, total_files) * 100)
                
        batch_status["log"].append("Automated batch jobs execution complete.")
        batch_status["status"] = "completed"
    except Exception as e:
        traceback.print_exc()
        batch_status["status"] = "error"
        batch_status["log"].append(f"Batch runner error: {str(e)}")

# Routes: Batch processing API
@app.route('/api/workspace/batch/start', methods=['POST'])
def api_batch_start():
    global batch_thread, batch_status
    if not active_workspace_path:
        return jsonify({"error": "No workspace loaded."}), 400
        
    if batch_status["status"] == "running":
        return jsonify({"error": "Batch jobs are already running."}), 400
        
    req = request.json or {}
    scripts = req.get("scripts", [])
    
    if not scripts:
        return jsonify({"error": "Please provide at least one batch job script config."}), 400
        
    # Validate directories and weights exist
    for idx, s in enumerate(scripts, 1):
        if not s.get("input") or not os.path.isdir(s["input"]):
            return jsonify({"error": f"Job {idx}: Input images directory does not exist."}), 400
        if not s.get("output"):
            return jsonify({"error": f"Job {idx}: Output save directory must be specified."}), 400
        if not s.get("model"):
            return jsonify({"error": f"Job {idx}: Model weights file must be selected."}), 400
            
    # Save jobs to workspace config
    active_workspace_data["scripts"] = scripts
    try:
        ws_file = os.path.join(active_workspace_path, "workspace.json")
        with open(ws_file, "w", encoding="utf-8") as fh:
            json.dump(active_workspace_data, fh, indent=2)
    except Exception as e:
        print(f"Error saving workspace: {e}")
        
    settings = dict(active_workspace_data["settings"])
    settings["device"] = DEVICE
    
    batch_status = {"status": "running", "progress": 0, "log": []}
    batch_stop_event.clear()
    
    batch_thread = threading.Thread(
        target=batch_runner,
        args=(scripts, settings, get_models_dir()),
        daemon=True
    )
    batch_thread.start()
    return jsonify({"success": True, "status": batch_status})

@app.route('/api/workspace/batch/stop', methods=['POST'])
def api_batch_stop():
    if batch_status["status"] != "running":
        return jsonify({"error": "Batch jobs are not running."}), 400
    batch_stop_event.set()
    return jsonify({"success": True})

@app.route('/api/workspace/batch/status', methods=['GET'])
def api_batch_status():
    return jsonify(batch_status)


# System Update Check API
CURRENT_VERSION = "0.0.0"

@app.route('/api/system/update-check', methods=['GET'])
def api_update_check():
    import urllib.request
    import json
    url = "https://api.github.com/repos/Dominik7272/PrefNet-Studio/releases/latest"
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'PrefNet-App-Update-Checker'}
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))
            tag_name = data.get("tag_name", "")
            if not tag_name:
                return jsonify({
                    "current_version": CURRENT_VERSION,
                    "latest_version": CURRENT_VERSION,
                    "update_available": False
                })
            
            remote_ver = tag_name.lstrip('v')
            local_ver = CURRENT_VERSION.lstrip('v')
            
            try:
                remote_parts = [int(x) for x in remote_ver.split('.')]
                local_parts = [int(x) for x in local_ver.split('.')]
                max_len = max(len(remote_parts), len(local_parts))
                remote_parts += [0] * (max_len - len(remote_parts))
                local_parts += [0] * (max_len - len(local_parts))
                update_available = remote_parts > local_parts
            except ValueError:
                update_available = remote_ver != local_ver
                
            return jsonify({
                "current_version": CURRENT_VERSION,
                "latest_version": tag_name,
                "update_available": update_available
            })
    except Exception as e:
        return jsonify({
            "current_version": CURRENT_VERSION,
            "latest_version": CURRENT_VERSION,
            "update_available": False,
            "error": str(e)
        })


@app.route('/api/system/update', methods=['POST'])
def api_system_update():
    import subprocess
    try:
        # Run git pull from the parent directory of this script (repository root)
        repo_dir = os.path.dirname(SCRIPT_DIR)
        result = subprocess.run(
            ["git", "pull"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "output": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "error": result.stderr or result.stdout or "Unknown git error"
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })




# Serve React app main page and static assets
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100, help="Port to run Flask backend")
    args = parser.parse_args()
    
    print("--------------------------------------------------")
    print(f"BiLabel Preference Classifier Server starting...")
    print(f"Compute Hardware Device Detect: {DEVICE.upper()}")
    print(f"Server is listening on http://localhost:{args.port}")
    print("--------------------------------------------------")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
