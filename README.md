# Negev Acoustic Flood Detector

A low-cost, self-contained **acoustic flash-flood early-warning system** that runs entirely
on an edge single-board computer. It uses **YAMNet-based transfer learning** to classify a
live microphone stream into three states — flowing flood water, rainfall, and the dry ambient
baseline — and is designed for arid *wadi* (ephemeral desert stream) environments where flash
floods arrive with little visual warning.

The turbulent flow of water over a rough channel bed produces a broadband, sustained acoustic
signature that is distinct from both rainfall and a quiet dry channel. Acoustic sensing is
attractive for early warning because it is inexpensive, omnidirectional, robust to darkness and
poor visibility, and can detect an approaching front before it reaches the sensor.

> Final project for course 001-2-5074, The Jacob Blaustein Institutes for Desert Research,
> Ben-Gurion University of the Negev. Submitted to Dr. Elad Levintal, July 2026.
> **This is a proof of concept, not a field-validated instrument** — see [Limitations](#limitations).

---

## How it works

```
                        ┌─────────── training (Colab, GPU, run once) ───────────┐
FreeSound clips ──► YAMNet (frozen) ──► 1024-d embedding ──► Dense head ──► TFLite export
  3 classes            AudioSet-pretrained                    (1024→256→128→3)
                        └───────────────────────────────────────────────────────┘
                                                                     │
                                          two small .tflite files    ▼
                        ┌─────────────── deployment (UNIHIKER, edge) ────────────┐
mic ──► 3 s chunk ──► yamnet_fixed.tflite ──► embedding ──► classifier.tflite ──► prediction
  16 kHz              (input [1,48000] → [1,1024])            (~100 kB)          + live UI + CSV log
                        └────────────────────────────────────────────────────────┘
```

Rather than training a CNN from scratch (which overfits badly on the small, imbalanced datasets
typical of this domain), YAMNet — Google's AudioSet-pretrained convolutional model — is used as a
**frozen feature extractor**. Each 3-second segment (48,000 samples at 16 kHz) becomes a
1024-dimensional embedding, and a compact dense classifier is trained on top. Only two small
TensorFlow Lite files are shipped to the device.

**Classes:** `flood_water` · `rain` · `ambient_dry`

---

## Repository structure

```
negev-acoustic-flood-detector/
├── notebooks/
│   ├── 01_freesound_collector.ipynb    # Download labelled audio from FreeSound.org into 3 class folders
│   ├── 02_yamnet_training.ipynb        # Extract YAMNet embeddings, train the dense head, export TFLite
│   └── 03_bathroom_monitoring_plots.ipynb  # Analysis/figures for the indoor validation run
├── device/
│   └── detect_final_3sec_sampling.py                       # On-device real-time inference + touchscreen UI + CSV logging
├── models/
│   └── README.md                       # How to produce the .tflite files (not checked into git)
├── data/
│   └── logs/
│       └── monitoring_run_log.csv      # 8,790-chunk log from the ~38 h indoor run
├── docs/
│   ├── report/
│   │   └── flood_detection_yamnet_report.docx   # Full project report
│   └── figures/                        # (place confusion matrix, run plots, photos here)
├── requirements.txt                    # Training/analysis environment (Colab/desktop)
├── requirements-device.txt             # Minimal runtime for the UNIHIKER
├── .gitignore
└── LICENSE
```

---

## Hardware

The whole system runs on a single **UNIHIKER** SBC — a Linux board with a quad-core ARM
Cortex-A35, 512 MB RAM, a built-in microphone, and a 2.8" (240×320) touchscreen. It runs
Python 3.7 with TensorFlow 2.7, which is the version constraint that drives the fixed-input-shape
export strategy in notebook 02.

| # | Component | Specification | Qty | Function |
|---|-----------|---------------|-----|----------|
| 1 | UNIHIKER single-board computer | Quad-core ARM Cortex-A35, 512 MB RAM, built-in mic, 240×320 touchscreen | 1 | On-device capture, inference, display |
| 2 | USB-C cable | Standard USB-C, power/data | 1 | Power delivery |
| 3 | Portable power bank | 5 V USB, 2500 mAh | 1 | Untethered power for the run |

---

## Getting started

### 1 · Collect the dataset — `notebooks/01_freesound_collector.ipynb`
Downloads Creative-Commons-0 audio from FreeSound.org into `flood_water/`, `rain/`, and
`ambient_dry/` folders. You need a free FreeSound API key
(https://freesound.org/apiv2/apply/).

**Do not hardcode the key.** The notebook reads it from an environment variable or prompts for it:

```bash
export FREESOUND_API_KEY="your_key_here"
```

### 2 · Train & export — `notebooks/02_yamnet_training.ipynb`
Run in Google Colab with a GPU. It loads YAMNet from TF Hub, extracts embeddings for every clip
(averaged to one 1024-d vector, with light Gaussian-noise augmentation to balance classes),
trains the dense head, and exports:

- `yamnet_fixed.tflite` — YAMNet with a fixed `[1, 48000]` input → `[1, 1024]` output
- `flood_classifier_only.tflite` — the trained dense head (~100 kB)
- `class_labels.json` — class order so the device labels predictions correctly

See [`models/README.md`](models/README.md) for details. These files are **not** committed to git.

### 3 · Deploy — `device/detect.py`
Copy the two `.tflite` files and `class_labels.json` to the UNIHIKER under the path set in the
script (default `/root/flood_detector`), then run it. The script records 3 s chunks with
`sounddevice`, runs inference in a background thread, shows live per-class probability bars on the
touchscreen, and appends a row to `log.csv` every 15 s.

```bash
python3 detect_final_3sec_sampling.py
```

Key runtime constants at the top of `detect_final_3sec_sampling.py`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `SAMPLE_RATE` | 16000 | Required by YAMNet |
| `DURATION` | 3 | Seconds per inference chunk |
| `LOG_INTERVAL` | 15 | Seconds between CSV log rows |
| `THRESHOLD` | 0.75 | Confidence needed to raise the on-screen flood alert |
| `MODEL_DIR` | `/root/flood_detector` | Where the model files live on the device |

---

## Results

On the validation set (n = 203) the classifier reached **95% overall accuracy** — recall 0.88 for
`flood_water`, 0.98 for `rain`, 0.95 for `ambient_dry` — comparable to published YAMNet
transfer-learning baselines. Residual errors concentrate at the `flood_water`/`rain` boundary,
which is expected given the spectral similarity of broadband water noise in both.

The detector was then run indoors in a domestic bathroom over three sessions (~38 h total), using
shower/sink/toilet events as rough acoustic proxies for sustained flow. Across 8,782 logged chunks
it predicted `ambient_dry` 95.9% of the time, giving a non-ambient (false-positive) rate of ~4.1%
under predominantly dry conditions. Sustained events (showers) produced prolonged high-probability
excursions, while brief incidental water noise (handwashing) produced only one- or two-chunk
spikes — consistent with the intended "three-consecutive-detections" debounce rule. The raw log is
in `data/logs/monitoring_run_log.csv`; the figures come from `notebooks/03_...`.

---

## Limitations

This is a **proof of concept**, not a validated flood detector:

- The bathroom run characterizes the false-positive rate under dry conditions; it does **not**
  measure recall against real floods. Indoor water events are only approximate proxies for wadi flow.
- Rigorous validation needs controlled, timestamped water onsets (e.g. an instrumented flume) to
  quantify detection latency and recall against ground truth.
- Field deployment adds unsolved requirements: remote connectivity, weatherproofing and microphone
  protection, and validation against natural wadi background sounds (wind, fauna, vehicles).
- **Power** is the hard constraint: one 2500 mAh bank lasted ~12.6 h. An unattended station would
  need a solar-plus-battery supply and a lower duty cycle. (Turning the screen off didn't help — the
  LCD backlight draws similar power whether the screen is black or lit.)

---

## Authors

Arie Kobylansky · Nitzan Hahamov · Noam Budin
Ben-Gurion University of the Negev — The Jacob Blaustein Institutes for Desert Research.

## References

Key sources are listed in full in the project report (`docs/report/`): Bansal & Garg (2022);
Jesudhas & Ranjan (2024); Liu et al. (2025); Saber et al. (2015); De Sousa et al. (2025).

## Academic Project Notice

This repository contains a final course project developed at
Ben-Gurion University of the Negev (course 001-2-5074, The Jacob
Blaustein Institutes for Desert Research), by Arie Kobylansky,
Nitzan Hahamov, and Noam Budin.

It is shared publicly for demonstration and portfolio purposes.
All rights reserved by the authors. The code and report are not
licensed for reuse, redistribution, or modification without the
authors' permission.

Third-party components retain their own terms: YAMNet is Google's
model (Apache 2.0), and audio clips used in training are sourced
from FreeSound.org under Creative Commons 0.
