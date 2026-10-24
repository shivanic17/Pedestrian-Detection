# 🚗 Pedestrian Detection & Perception Suite

A pedestrian detection system for autonomous driving scenes, built on **YOLO11 Nano** and trained on the **CityPersons / Cityscapes** dataset. It ships with an interactive **Streamlit dashboard** that runs inference on an uploaded driving frame, colour-codes each pedestrian by a proximity-based risk zone, and reports telemetry and analytics.

---

## ✨ Features

- **Pedestrian detection** using a fine-tuned YOLO11n model (single class: pedestrian).
- **Proximity risk zones** – bounding-box height is used as a heuristic for distance to the vehicle:

  | Zone | Rule (box height) | Approx. distance | Box colour |
  |------|-------------------|------------------|------------|
  | 🔴 Critical | > 250 px | < 10 m | Red |
  | 🟠 Warning | 100 – 250 px | 10 – 25 m | Orange |
  | 🟢 Safe | ≤ 100 px | > 25 m | Green |

- **Collision status banner** – Critical Hazard / Caution Required / Clear Roadway.
- **Live dashboard metrics** – pedestrian count, average confidence, frame resolution, inference latency (ms).
- **Interactive analytics** (Plotly) – confidence per detection and bounding-box size vs. risk zone.
- **Telemetry table + CSV export** for every detection (ID, confidence, zone, width, height, area, coordinates).
- **Adjustable inference settings** – confidence cutoff and NMS IoU threshold from the sidebar.
- **Dataset tooling** – scripts to convert CityPersons annotations to YOLO format and to build the image folders without duplicating files.

---

## 📁 Project Structure

```
Pedestrian-Detection/
├── .streamlit/               # Streamlit configuration
├── runs/detect/              # Training runs / outputs (incl. best.pt checkpoint)
├── weights/                  # Model weights
├── app.py                    # Streamlit dashboard
├── convert_citypersons.py    # CityPersons JSON  ->  YOLO label converter
├── setup_hardlinks.py        # Builds YOLO image folders using hard links
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## 🛠️ Tech Stack

- **Model:** [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO11 Nano (`yolo11n.pt`)
- **UI:** Streamlit
- **Vision / data:** OpenCV, Pillow, NumPy, Pandas
- **Charts:** Plotly

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shivanic17/Pedestrian-Detection.git
cd Pedestrian-Detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

### 5. Use the dashboard

1. In the sidebar, confirm the **Weights Checkpoint** path. The default is:
   ```
   runs/detect/runs/detect/citypersons_fast_full/weights/best.pt
   ```
   Change it if your `best.pt` lives elsewhere (e.g. inside `weights/`).
2. Adjust **Confidence Cutoff** (default `0.25`) and **NMS IoU Threshold** (default `0.45`).
3. Upload a driving-scene image (`.jpg`, `.jpeg`, `.png`) – ideally from CityPersons / Cityscapes.
4. Review the annotated frame, risk matrix, analytics tabs, and export the telemetry CSV.

> Inference runs on **CPU** by default (`device="cpu"` in `app.py`). Change it to `"cuda"` or `0` if you have a GPU.

---

## 📊 Dataset Preparation (optional – only needed for training)

The model is trained on **CityPersons** annotations over **Cityscapes** images. Both datasets must be downloaded separately from their official sites (registration required) and are **not** included in this repo.

Expected layout before running the scripts:

```
<project root>/
├── cityscapes/
│   └── leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
└── gtBboxCityPersons/
    └── gtBboxCityPersons/{train,val}/<city>/*_gtBboxCityPersons.json
```

### Step 1 – Convert annotations to YOLO format

```bash
python convert_citypersons.py
```

This reads each `*_gtBboxCityPersons.json`, keeps only objects labelled `pedestrian`, converts `[x, y, w, h]` boxes to normalised YOLO format (`class cx cy w h`, class `0` = pedestrian), and writes labels to:

```
citypersons_yolo/labels/{train,val}/<city>/*.txt
```

### Step 2 – Link the images (saves disk space)

```bash
python setup_hardlinks.py
```

This creates `citypersons_yolo/images/{train,val}/` using **hard links** to the original Cityscapes images instead of copying them.

> ⚠️ `setup_hardlinks.py` contains a hard-coded `PROJECT_ROOT` (a local Windows path). Update it to your own project directory before running.

### Step 3 – Train (example)

Training code isn't included in this repo; the following is a sample using the Ultralytics CLI. Create a `data.yaml` pointing at the folders above:

```yaml
path: citypersons_yolo
train: images/train
val: images/val
names:
  0: pedestrian
```

```bash
yolo detect train model=yolo11n.pt data=data.yaml epochs=50 imgsz=640
```

The resulting `best.pt` can then be loaded in the dashboard via the sidebar.

---

## ⚙️ How the Risk Logic Works

The app does **not** use depth sensing or camera calibration. Distance is approximated purely from the pixel height of each detected bounding box: taller boxes imply the pedestrian is closer to the camera.

```python
if box_h > 250:   # CRITICAL  (< 10 m)
elif box_h > 100: # WARNING   (10 m – 25 m)
else:             # SAFE      (> 25 m)
```

Thresholds are tuned for Cityscapes-style frames (2048×1024) and will not transfer directly to other cameras or resolutions.

---

## ⚠️ Limitations

- Proximity zones are a **heuristic**, not a measured distance.
- Single-frame inference only – no tracking, velocity, or time-to-collision estimation.
- Not intended for safety-critical or real-world deployment; this is an educational / research project.
- Dataset files and large weights may not be tracked in the repo.

---

## 🔮 Future Work

- Video / webcam stream support
- Multi-object tracking and time-to-collision estimation
- Camera-calibrated distance estimation
- GPU inference toggle in the UI
- Evaluation metrics (mAP, precision, recall) displayed in the dashboard

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [CityPersons Dataset](https://github.com/cvgroup-njust/CityPersons)
- [Streamlit](https://streamlit.io/)

---

## Deployed Link 

https://pedestrian-detection-rpxmmhn74oyc2fmbdjnvve.streamlit.app/

---

## 👤 Author

**shivanic17** – [GitHub](https://github.com/shivanic17)
