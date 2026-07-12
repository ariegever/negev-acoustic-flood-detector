# Models

The trained model artifacts are **not** committed to the repository (they are build outputs, and
`.tflite` / `.keras` files are in `.gitignore`). Regenerate them by running
`notebooks/02_yamnet_training.ipynb` end to end.

## Files produced by the training notebook

| File | Shape / size | Role |
|------|--------------|------|
| `yamnet_fixed.tflite` | input `[1, 48000]` → output `[1, 1024]` | YAMNet wrapped with a **fixed** input shape so it behaves identically across the TF versions used for training vs. the device. Extracts the 1024-d embedding from a 3 s / 16 kHz chunk. |
| `flood_classifier_only.tflite` | ~100 kB | The trained dense head (`1024 → 256 → 128 → 3`). Input is the YAMNet embedding, output is 3 class probabilities. |
| `class_labels.json` | — | Ordered class list (`["flood_water", "rain", "ambient_dry"]`) so the device maps output indices to names. |
| `best_yamnet_classifier.keras` | — | Full Keras checkpoint (for retraining / inspection). |
| `yamnet_confusion_matrix.png`, `yamnet_training_curves.png` | — | Evaluation figures. |

## Why a fixed input shape?

The training environment and the UNIHIKER run different TensorFlow versions. A dynamic-shape
YAMNet export behaved inconsistently between them, so the notebook builds a wrapper with a fixed
`[1, 48000]` input (3 s at 16 kHz) that converts reliably.

## Deploying to the device

Copy these three files to the directory referenced by `MODEL_DIR` in `device/detect.py`
(default `/root/flood_detector`):

```
yamnet_fixed.tflite
flood_classifier_only.tflite
class_labels.json
```
