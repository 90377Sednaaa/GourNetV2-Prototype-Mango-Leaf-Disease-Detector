# Where to put your trained models

This folder is where `app.py` looks for trained weights at startup. Drop in
the `.keras` files saved by the notebooks, using these **exact
filenames**:

```
weights/
├── gournet_model.keras          <- Baseline model (Standard dataset)
├── gournet_model_12k.keras      <- Baseline model (12k dataset; currently shares Standard weights until retrained)
├── gournet_v2_model.keras       <- GourNet v2 (Standard dataset)
└── gournet_v2_model_12k.keras   <- GourNet v2 (12-seed multi-seed / 12k dataset)
```

## Note on the 12k Baseline Weights
`gournet_model_12k.keras` currently shares weights with `gournet_model.keras` (Standard).
When a separate baseline model trained on the ~12k dataset is available, replace
`weights/gournet_model_12k.keras` with that file.

## Behavior without weights

If a file is missing, the app still runs: it builds the architecture with
random (untrained) weights and shows a warning banner in the UI. Add the
`.keras` files to get meaningful results.

## Using a different / retrained model

If you retrain and save under a different filename, either rename it to
match the names above, or edit `MODEL_REGISTRY` in `app.py`
(`weights` dictionary entries) to point at your file.
