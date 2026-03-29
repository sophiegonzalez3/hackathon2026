# LiDAR Object Detection Pipeline

Detect infrastructure objects (Antenna, Cable, Electric Pole, Wind Turbine) in airborne LiDAR point clouds using PointPillars.

## Repository Structure

```
project/
│
├── pointpillars/                  # MODEL PACKAGE (self-contained)
│   ├── __init__.py
│   ├── config.py                  # Model configuration
│   ├── model.py                   # PointPillars + CenterHead
│   ├── dataset.py                 # Dataset & data loading
│   ├── train.py                   # Training script
│   ├── inference.py               # Inference (CLI + API)
│   └── utils.py                   # Decoding, losses, etc.
│
├── pipeline/                      # 🔧 PREPROCESSING & POSTPROCESSING
│   ├── preprocess_frames.py       # H5 → NPZ frames
│   ├── generate_gt_bboxes.py      # Generate ground truth
│   ├── extra_cleaning_bbox.py     # Clean GT outliers
│   ├── ground_removal.py          # Ground removal + sanity check
│   └── postprocess_predictions.py # Filter predictions (after training + inference)
│
├── configs/                       # Configuration files for the Bbox dbscan algo
│   └── run_05_merge.yaml          # GT generation config
│
├── checkpoints/                   # Trained models
│   └── best.pth
│                                   # Data directories
├── trainingData/                  # Raw H5 files (trainingData/)
│   ├── scene_*.h5                      
│── processed/                     # Preprocessed NPZ (with ground)
│   ├── scene_*/
│   │   └── frame_*.npz
│   └── manifest.csv
│── cleaned/                       # Ground-removed NPZ
│   │   ├── scene_*/
│   │   │   └── frame_*.npz
│   │   └── manifest.csv
│── gt_runs/                        # Ground truth bbox CSVs can be compared using ln.compare_side_bboxes or compare_side_bboxes_batch
│       ├── gt_bboxes.csv
│       └── gt_bboxes_clean.csv
│
├── results/                       # Outputs
│   ├── predictions.csv
│   ├── predictions_submission.csv
│   └── predictions_filtered.csv
│

```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FULL PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ PREPROCESSING (pipeline/)                                            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │  1. preprocess_frames.py      H5 → NPZ + manifest.csv               │    │
│  │           ↓                                                          │    │
│  │  2. generate_gt_bboxes.py     NPZ + config → gt_bboxes.csv          │    │
│  │           ↓                                                          │    │
│  │  3. extra_cleaning_bbox.py    Clean GT outliers                     │    │
│  │           ↓                                                          │    │
│  │  4. ground_removal.py         Remove ground + fix manifest/GT       │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ TRAINING (pointpillars/)                                             │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │  5. python -m pointpillars.train                                    │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ INFERENCE (pointpillars/ + pipeline/)                                │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                      │    │
│  │  6. python -m pointpillars.inference    → predictions.csv           │    │
│  │           ↓                                                          │    │
│  │  7. postprocess_predictions.py          → predictions_filtered.csv  │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Training (from scratch)

Upload h5 files to a trainingData folder

```bash
# 1. Preprocess H5 → NPZ
python pipeline/preprocess_frames.py \
    --data-dir trainingData/ \
    --out-dir processed/

# 2. Generate ground truth
python pipeline/generate_gt_bboxes.py \
    --processed-dir processed/ \
    --config configs/run_05_merge.yaml \
    --out gt_runs/gt_bboxes.csv

# 3. Clean GT
python pipeline/extra_cleaning_bbox.py \
    --input gt_runs/gt_bboxes.csv \
    --out gt_runs/gt_bboxes_clean.csv

# 4. Remove ground (+ fix manifest & GT paths)
python pipeline/ground_removal.py \
    --processed-dir processed/ \
    --output-dir cleaned/ \
    --gt-csv gt_runs/gt_bboxes_clean.csv \
    --sanity-check

# 5. Train
python -m pointpillars.train \
    --processed-dir cleaned/ \
    --gt-csv gt_runs/gt_bboxes_clean2.csv \
    --checkpoint-dir checkpoints/

# 6. Inference
python -m pointpillars.inference \
    --checkpoint checkpoints/best.pth \
    --processed-dir cleaned/ \
    --output results/predictions.csv

# 7. Post-process
python pipeline/postprocess_predictions.py \
    results/predictions.csv \
    results/predictions_filtered.csv
```

### Test data (Eval Day)

```bash
# 1. Preprocess test H5
python pipeline/preprocess_frames.py \
    --data-dir Eval/testData/ \
    --out-dir Eval/test_processed/

# 2. Remove ground
python pipeline/ground_removal.py \
    --processed-dir Eval/test_processed/ \
    --output-dir Eval/test_cleaned/ \
    --threshold 2.0 #Use the best/safest after analyse on your training

# 3. Inference
python -m pointpillars.inference \
    --checkpoint checkpoints/best.pth \
    --processed-dir Eval/test_cleaned/ \
    --output submission.csv

# 4. Post-process (optional but recommended)
python pipeline/postprocess_predictions.py \
    submission.csv \
    submission_filtered.csv
```

---

## Script Reference

### 1. `pipeline/ground_removal.py`

Removes ground points using local tile-based height estimation.

```bash
# Full pipeline (analyze + clean + sanity check)
python pipeline/ground_removal.py \
    --processed-dir data/processed/ \
    --output-dir data/cleaned/ \
    --sanity-check

# Analysis only (find best threshold)
python pipeline/ground_removal.py \
    --processed-dir data/processed/ \
    --analyze-only

# Custom threshold
python pipeline/ground_removal.py \
    --processed-dir data/processed/ \
    --output-dir data/cleaned/ \
    --threshold 3.0

# Also fix GT paths
python pipeline/ground_removal.py \
    --processed-dir data/processed/ \
    --output-dir data/cleaned/ \
    --gt-csv data/gt/gt_bboxes_clean.csv

# Visualize single frame (for debugging)
python pipeline/ground_removal.py \
    --visualize data/processed/scene_1/frame_050.npz

# Visualize class survival
python pipeline/ground_removal.py \
    --visualize-class data/processed/scene_1/frame_050.npz \
    --threshold 5.0
```

**Output:**
- `cleaned/` — NPZ files with ground removed
- `cleaned/manifest.csv` — Frame manifest (PointPillars-compatible)
- `gt_bboxes_clean2.csv` — GT with updated paths (if `--gt-csv` provided)

---

### 2. `pointpillars/inference.py`

Run trained model on any preprocessed data.

```bash
# Basic inference
python -m pointpillars.inference \
    --checkpoint checkpoints/best.pth \
    --processed-dir data/cleaned/

# Custom output and threshold
python -m pointpillars.inference \
    --checkpoint checkpoints/best.pth \
    --processed-dir data/cleaned/ \
    --output my_predictions.csv \
    --threshold 0.15

# Quick test (subset of frames)
python -m pointpillars.inference \
    --checkpoint checkpoints/best.pth \
    --processed-dir data/cleaned/ \
    --max-frames 50

# Higher batch size (faster, needs more GPU memory)
python -m pointpillars.inference \
    --checkpoint checkpoints/best.pth \
    --processed-dir data/cleaned/ \
    --batch-size 16
```

**Output:**
- `predictions.csv` — Full predictions with confidence scores
- `predictions_submission.csv` — Competition format (13 columns)

---

### 3. `pipeline/postprocess_predictions.py`

Filter and clean predictions.

```bash
# Basic filtering
python pipeline/postprocess_predictions.py \
    predictions.csv \
    predictions_filtered.csv

# Analysis only (understand your predictions)
python pipeline/postprocess_predictions.py \
    predictions.csv \
    --analyze-only

# Custom thresholds
python pipeline/postprocess_predictions.py \
    predictions.csv \
    predictions_filtered.csv \
    --iou-threshold 0.25 \
    --antenna-thresh 0.5 \
    --cable-thresh 0.3
```

**Filters Applied:**
1. Per-class confidence thresholds
2. Dimension sanity filters
3. 3D NMS (removes overlapping boxes)
4. Per-frame limits

---

## Notebook Usage

### Quick Visualization

```python
from pipeline.ground_removal import visualize_frame, visualize_class_survival
from pathlib import Path

# Multi-threshold comparison
visualize_frame(
    Path('data/processed/scene_1/frame_050.npz'),
    thresholds=[0.5, 3.0, 5.0, 10.0, 15.0]
)

# Per-class survival analysis
visualize_class_survival(
    Path('data/processed/scene_1/frame_050.npz'),
    threshold=5.0
)
```

### Quick Inference

```python
from pointpillars.inference import predict

# One-liner
df = predict('checkpoints/best.pth', 'data/cleaned/')

# With custom settings
df = predict(
    'checkpoints/best.pth',
    'data/test_cleaned/',
    output_path='submission.csv',
    score_threshold=0.15,
    batch_size=8
)
```

### Low-Level Control

```python
from pointpillars.inference import load_model, build_dataloader, run_inference
import pandas as pd

# Load model
model, cfg, device = load_model('checkpoints/best.pth', score_threshold=0.2)

# Build dataloader
loader, num_frames = build_dataloader(cfg, 'data/cleaned/', batch_size=8)

# Run inference
results = run_inference(model, loader, cfg, device)
df = pd.DataFrame(results)

# Analyze
print(df['class_label'].value_counts())
print(df.groupby('class_label')['confidence'].describe())
```

### Post-Processing in Notebook

```python
from pipeline.postprocess_predictions import postprocess

# Apply all filters
df_filtered = postprocess(
    input_csv=Path('predictions.csv'),
    output_csv=Path('predictions_filtered.csv')
)

# Or analysis only
from pipeline.postprocess_predictions import analyze_predictions, print_analysis
import pandas as pd

df = pd.read_csv('predictions.csv')
stats = analyze_predictions(df, {})
print_analysis(stats)
```

---

## Configuration

### Default Post-Processing Thresholds

| Class | Confidence | Max/Frame | Width | Length | Height |
|-------|-----------|-----------|-------|--------|--------|
| Antenna | 0.40 | 10 | 2-25m | 2-25m | 10-80m |
| Cable | 0.28 | 30 | 5-100m | 0.3-15m | 0.3-15m |
| Electric Pole | 0.35 | 8 | 2-40m | 2-25m | 15-80m |
| Wind Turbine | 0.25 | 10 | 15-150m | 3-50m | 20-200m |

### Custom Post-Processing Config

Create `my_config.json`:
```json
{
    "class_thresholds": {
        "Antenna": 0.45,
        "Cable": 0.30,
        "Electric Pole": 0.40,
        "Wind Turbine": 0.30
    },
    "max_per_frame": {
        "Antenna": 8,
        "Cable": 25,
        "Electric Pole": 6,
        "Wind Turbine": 8
    },
    "nms_iou_threshold": 0.25
}
```

Use it:
```bash
python pipeline/postprocess_predictions.py \
    predictions.csv \
    output.csv \
    --config my_config.json
```

---

## Troubleshooting

### "No detections" after inference
- Lower the score threshold: `--threshold 0.1`
- Check if ground removal was too aggressive (use `--visualize`)
- Verify model checkpoint is correct

### Poor class survival after ground removal
- Use `--analyze-only` to find better threshold
- Use `--visualize-class` to see what's being removed
- Try lower threshold (e.g., `--threshold 2.0`)

### Import errors
```bash
# Add packages to path
export PYTHONPATH=$PYTHONPATH:/path/to/project

# Or install them
pip install -e /path/to/pointpillars
```

### Out of memory
- Reduce batch size: `--batch-size 2`
- Reduce workers: `--workers 2`

---

## Files Explained

| File | Purpose |
|------|---------|
| `pointpillars/inference.py` | Run trained model (training & evaluation) |
| `pipeline/ground_removal.py` | Remove ground points, fix manifest |
| `pipeline/postprocess_predictions.py` | Filter predictions (NMS, thresholds) |
| `pipeline/preprocess_frames.py` | Convert H5 → NPZ frames |
| `pipeline/generate_gt_bboxes.py` | Generate ground truth from labels |
| `manifest.csv` | Frame index with ego poses and paths |
| `*_submission.csv` | Competition format (13 columns) |

---

## Evaluation Day Checklist

```
□ 1. Download test H5 files → testData/

□ 2. Preprocess
      python pipeline/preprocess_frames.py --data-dir testData/ --out-dir test_processed/

□ 3. Ground removal
      python pipeline/ground_removal.py --processed-dir test_processed/ --output-dir test_cleaned/

□ 4. Inference
      python -m pointpillars.inference --checkpoint checkpoints/best.pth \
          --processed-dir test_cleaned/ --output submission.csv

□ 5. Post-process
      python pipeline/postprocess_predictions.py submission.csv submission_final.csv

□ 6. Verify submission format (13 columns):
      ego_x, ego_y, ego_z, ego_yaw,
      bbox_center_x, bbox_center_y, bbox_center_z,
      bbox_width, bbox_length, bbox_height, bbox_yaw,
      class_ID, class_label

□ 7. Submit submission_final_submission.csv
```

---

## License

MIT License