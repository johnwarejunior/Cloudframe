# Cloudframe

Turn a raw LiDAR point cloud into classified, measurable, exportable 3D objects.

Cloudframe takes a scan, removes the ground, segments what's left into individual objects, classifies each one (tree, vehicle, structure, pole, debris), and gives you generic tools — distance measurement, high-resolution mesh export — to do whatever your job actually requires with the result. It isn't built around one industry: a utility forester checking clearance and an insurance adjuster measuring storm damage use the same underlying objects, just for different ends.

**[Live interactive demo →](https://claude.ai/artifact/F2Z5MZAjg7r21Q7xUfszgu)** — runs a synthetic scan entirely in-browser, no setup required.

---

## What's in this repo

| Component | What it does |
|---|---|
| `cloudframe_pipeline.py` | The production/batch pipeline — point real scan data at it, get segmented, classified, exported objects and a CSV report back |
| Interactive demo (published separately, linked above) | Browser-based version for demos and pitches: generate a scan, tune segmentation live, measure, export |

## Features

- **Ground removal** — RANSAC-based plane segmentation strips the dominant flat surface (ground/floor) out of the scan
- **Object segmentation** — DBSCAN clustering groups the remaining points into individual objects, with a tunable distance parameter
- **Cleanup** — statistical outlier removal trims stray points so object boundaries aren't distorted by noise
- **Classification** — every detected object gets a type (Tree / Vehicle / Structure / Pole / Debris / Unclassified) and a confidence score, from a geometric heuristic (see [Classification](#classification) below)
- **Measurement** — generic point-to-point and point-to-line distance utilities; use them for whatever your job needs (clearance checks, gaps between structures, anything)
- **Export** — every object is saved as both a `.ply` (raw point cloud) and a `.obj` (real solid mesh — see [Export formats](#export-formats))
- **Structured report** — a CSV with one row per detected object: type, confidence, dimensions, position

## How it works

```
Load scan → Downsample → Remove ground (RANSAC) → Cluster into objects (DBSCAN)
    → Clean each object (outlier removal) → Classify → Export (.ply + .obj) + CSV report
```

## Getting started

### Requirements

```bash
pip install open3d numpy matplotlib --break-system-packages
```

Python 3.9+ recommended.

### Run it

1. Open `cloudframe_pipeline.py` and set `INPUT_PATH` to your `.ply` or `.pcd` scan file.
   - No scan handy? Leave it as-is — the script falls back to one of Open3D's built-in sample point clouds automatically, so you can try the pipeline immediately.
2. Run:
   ```bash
   python cloudframe_pipeline.py
   ```
3. Check the output:
   - `segmented_objects/` — `OBJ-001.ply`, `OBJ-001.obj`, etc., one pair per detected object
   - `objects_report.csv` — one row per object: type, confidence, height, footprint, position
   - Three interactive viewer windows open in turn (close each to continue):
     1. **Raw scan** — the downsampled input in gray (the "before" view)
     2. **Segmented objects** — each detected cluster in its own color, noise in black
     3. **Objects by classified type** — every exported object colored by its class

### Key configuration

All of these live at the top of `cloudframe_pipeline.py`:

| Variable | What it controls |
|---|---|
| `DBSCAN_EPS` | How close points need to be to join the same object. Too small fragments one object into pieces; too large merges separate objects together. Tune this per scene. |
| `DBSCAN_MIN_POINTS` | Minimum points required for a region to count as an object rather than noise |
| `MIN_CLUSTER_SIZE` | Objects smaller than this (after clustering) are dropped as debris/noise |
| `OUTLIER_STD_RATIO` | How aggressively statistical outlier removal cleans each object's boundary |
| `EXPORT_CUBE_SIZE` | Size of the solid cube generated per point in the `.obj` export — controls mesh resolution/chunkiness |
| `MEASURE_POINTS` | Optional: two 3D points to measure the distance between (e.g. a conductor line, a property boundary — whatever's relevant to your use case) |

## Classification

Object type is currently assigned by a **geometric heuristic** — height, footprint size, footprint shape, and height off the ground — not a trained model. This keeps the pipeline dependency-free and runs anywhere Open3D runs, and it's accurate enough to be useful on outdoor site-style scans (the kind with trees, vehicles, structures, and poles) today.

| Type | Rule of thumb |
|---|---|
| Pole | Tall (>7m), very narrow footprint |
| Vehicle | 1–2.3m tall, moderate footprint, sits at ground level |
| Structure | 2.3–6m tall, larger/boxier footprint, sits at ground level |
| Tree | >4m tall, roughly round footprint |
| Debris | Low (<1.2m), irregular, near ground |
| Unclassified | Doesn't clearly match any of the above |

This is intentionally the swap point for something better: `classify_object()` takes a few summary numbers and returns `(type, confidence)`. A trained point-cloud classification model (recent work using architectures like PointNet++ or PointMAE, trained on labeled point-cloud datasets, reaches strong accuracy identifying real object/species classes from raw geometry) would replace the inside of that one function without touching anything else in the pipeline. On indoor or otherwise non-"site scan" data, expect a lot of `Unclassified` results — the thresholds aren't tuned for furniture-scale scans, and returning `Unclassified` rather than a false guess is the intended, honest behavior.

## Export formats

- **`.ply`** — the raw, cleaned point cloud for this object. Native format for most point-cloud/LiDAR tooling.
- **`.obj`** — a real, solid mesh: every point becomes a small cube, not a floating vertex. This is a genuinely valid, high-resolution mesh, importable into Unreal Engine, Blender, or any standard 3D tool. Structurally verified — every face index correctly references a real vertex.

## Roadmap

- Swap the geometric classifier for a trained point-cloud model (real accuracy gain, needs labeled training data from real scans first)
- Temporal comparison across repeated scans of the same site (what changed, what's growing/new)
- Real-time / onboard deployment on drone and LiDAR-camera hardware
- Species-level classification for vegetation-specific use cases

## License

Copyright (c) 2026 John Ware Jr. All rights reserved.

This source code is made publicly visible for viewing purposes only.
No permission is granted to use, copy, modify, merge, publish, distribute,
sublicense, or sell copies of this software, in whole or in part, without
prior written permission from the copyright holder.

For licensing inquiries, contact john@johnwarejunior.com.
