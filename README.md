# BiLabel Preference Classifier Studio v3 (Web Version)

A professional, hardware-accelerated, lightweight web application for training binary classification neural networks on CLIP image feature embeddings (e.g. Liked/Disliked preferences).

---

## ⚡ Quick Start

### 1. Requirements

Ensure you have installed the core requirements for the PyTorch / Transformers model loader, along with `flask` for the web server backend:

```bash
pip install -r ../requirements.txt
pip install flask
```

### 2. Run the App

From the workspace folder (where this README is located), simply run:

```bash
python server.py
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🛠️ Features Overview

1. **Dashboard & Workspaces**: Create or load modular workspace configurations which store setting parameters, classification profiles, model paths, batch automated scripts, and recents histories.
2. **Dataset Management**: Setup distinct classification profiles with customized labels (e.g. Liked/Disliked vs A/B classes).
3. **Interactive Keyboard Labeling**: Stream local unlabelled image queues with responsive fit-to-screen controls. Classify items instantly using keyboard keys:
   - <kbd>→</kbd> (Right Arrow) to label as **Class A**
   - <kbd>←</kbd> (Left Arrow) to label as **Class B**
   - <kbd>↑</kbd> (Up Arrow) to **Skip**
   - <kbd>↓</kbd> (Down Arrow) to **Undo last labeling**
4. **Neural Net Training**: Tune linear parameters layer weights. View training logs output streams and live-drawn custom SVG coordinate line charts plotting Loss and Accuracy curves in real-time.
5. **Bulk Predictions / Inference**: Select any trained model weights to categorize folders of images. Filter or sort prediction grids by confidence metrics, and inspect detail cards with manual override capability. Export predictions lists directly as CSV files.
6. **Automated Batch Jobs**: Script backgrounds jobs to watch target inputs folders and sort new files into class subdirectories based on classification confidence thresholds.

---

## 📂 Project Structure

```
classifier_v3/
├── backend/
│   └── app.py          # Flask REST API + Background threads runners
├── frontend/
│   ├── dist/           # Compiled React static resources (served by Flask)
│   ├── src/
│   │   ├── App.jsx     # Main React SPA component
│   │   ├── index.css   # Custom Slate & Indigo Design system styles
│   │   └── main.jsx    # React mounting point
│   ├── package.json
│   └── vite.config.js  # Vite dev server proxies
├── README.md           # Instructions documentation
└── server.py           # Unified application boot launcher
```

---

## 🧪 Development (Optional)

If you wish to make changes to the frontend React source code:

1. Navigate to the `frontend/` folder:
   ```bash
   cd frontend
   ```
2. Run Vite hot-reload dev server:
   ```bash
   npm run dev
   ```
3. Open `http://localhost:5173`. Any API requests to `/api/*` will automatically be proxied to the Flask server running on port `8000`.
4. Rebuild static assets for production deployment:
   ```bash
   npm run build
   ```
