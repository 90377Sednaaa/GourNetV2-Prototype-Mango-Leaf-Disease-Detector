# Where to put your trained models

This folder is where `app.py` looks for trained weights at startup. Drop in
the `.keras` files saved by the two notebooks, using these **exact
filenames**:

```
weights/
├── gournet_model.keras       <- from baseline-gournet-12-mseed-result.ipynb
│                                 (cell: model.save("gournet_model.keras"))
└── gournet_v2_model.keras    <- from gournet-v2-12-mseed-with-no-augment.ipynb
                                  (cell: model_v2.save("gournet_v2_model.keras"))
```

Both notebooks already save these files during a normal run (Section 8,
"Save the trained model") — just download them from your Kaggle/Colab
output and copy them here.

## Behavior without weights

If a file is missing, the app still runs: it builds the architecture with
random (untrained) weights and shows a clear warning banner in the UI. This
is only useful for checking the app wiring, image upload, and layout — not
for real predictions. Add the `.keras` files to get meaningful results.

## Using a different / retrained model

If you retrain and save under a different filename, either rename it to
match the names above, or edit `MODEL_REGISTRY` in `app.py`
(`weights_file` entries) to point at your file.
