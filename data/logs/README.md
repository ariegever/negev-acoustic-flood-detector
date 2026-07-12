# Monitoring logs

`monitoring_run_log.csv` is the raw device log from the indoor validation run
(~38 h across three sessions, 8,790 chunk-level predictions). One row per inference chunk.

| Column | Meaning |
|--------|---------|
| `timestamp` | Local time the chunk was logged (`dd/mm/yyyy HH:MM`) |
| `flood_water_%` | Model probability for the flood-water class |
| `rain_%` | Model probability for the rain class |
| `ambient_dry_%` | Model probability for the dry-ambient class |
| `prediction` | Argmax class label for that chunk |

Load and plot it with `notebooks/03_bathroom_monitoring_plots.ipynb`. Note the run was split by
power interruptions; the notebook re-segments it into separate runs at logging gaps > 10 min.
