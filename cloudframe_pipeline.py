"""
Cloudframe Pipeline - General-Purpose LiDAR Segmentation & Classification
----------------------------------------------------------------------------
Scan -> segment -> classify -> measure/export. This is the general-purpose,
cross-industry version of the pipeline: it doesn't assume vegetation, claims,
or any single use case. It detects and classifies whatever is in the scan
(trees, vehicles, structures, poles, debris, or unclassified), and gives you
generic building blocks (distance measurement, high-resolution mesh export)
to apply to whatever job you're actually doing.

Classification here is a geometric heuristic (height, footprint, base
elevation) — a deliberately simple stand-in so this script has no external
model dependencies. It's accurate enough to be useful today, but it is NOT
the production classifier. Production should swap in a trained point-cloud
model — recent research (e.g. PointNet++, PointMAE) trained on datasets like
FOR-species20K reaches strong accuracy on real point-cloud classification
tasks. See classify_object() below for exactly where that swap happens.

Install dependencies first:
    pip install open3d numpy matplotlib --break-system-packages
"""

import open3d as o3d      # Core library for point cloud I/O, processing, and visualization
import numpy as np        # Used for array/label manipulation and geometric calculations
import matplotlib.pyplot as plt  # Used to generate distinct colors per cluster
import os                 # Used to create the output folder and build file paths
import csv                # Used to write the structured per-object report


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

INPUT_PATH = "sample_scan.ply"
# Path to the input point cloud file (.ply or .pcd). Replace with your own
# scan, or point this at an Open3D sample dataset path (see load_point_cloud()).

OUTPUT_DIR = "segmented_objects"
# Folder where each individual object is saved: a .ply (raw point cloud,
# native to most point-cloud/LiDAR tools) and a .obj (solid mesh, for
# Unreal/Blender/game engines and anything that wants a real 3D mesh).

REPORT_PATH = "objects_report.csv"
# Path to the structured, per-object CSV report written at the end of the run.

PLANE_DISTANCE_THRESHOLD = 0.02
# Maximum distance (in scene units, usually meters) a point can be from the
# candidate plane and still count as part of that plane. Larger values treat
# more points as ground, smaller values are stricter.

PLANE_RANSAC_N = 3
# Number of points RANSAC randomly samples to define a candidate plane.
# 3 is the mathematical minimum needed to define a plane and is standard.

PLANE_NUM_ITERATIONS = 1000
# How many random plane candidates RANSAC tests before picking the best one.

DBSCAN_EPS = 0.05
# Maximum distance between two points for DBSCAN to consider them
# neighbors (part of the same object). Tune per scene: too small fragments
# objects into pieces, too large merges separate objects together.

DBSCAN_MIN_POINTS = 20
# Minimum neighboring points required for DBSCAN to treat a region as a
# valid object rather than noise.

MIN_CLUSTER_SIZE = 30
# After clustering, any cluster with fewer than this many points is dropped
# from the final object list (treated as debris/noise, not a real object).
# Real scans are usually denser than a synthetic demo, so raise this if your
# scan resolution is higher.

VOXEL_SIZE = 0.01
# Voxel grid size (scene units) used to downsample before processing.

OUTLIER_NB_NEIGHBORS = 20
# Neighbors examined per point when removing statistical outliers from each
# detected object, for cleaner object boundaries.

OUTLIER_STD_RATIO = 2.0
# How many standard deviations a point's average neighbor distance must
# exceed before it's removed as a statistical outlier.

EXPORT_CUBE_SIZE = 0.05
# Edge length (scene units) of the small solid cube generated per point for
# the .obj export. This is what turns a "cloud of points" into a real,
# importable solid mesh rather than floating vertices. Larger values look
# chunkier but are more visible on small/sparse objects.

MEASURE_POINTS = None
# Optional: two 3D points, as [(x1, y1, z1), (x2, y2, z2)], to measure the
# distance between (see point_to_point_distance()). This is a generic
# example of the measurement utility — a conductor clearance check, a gap
# between two structures, whatever your job needs. Leave as None to skip.
# Example: MEASURE_POINTS = [(-20, 0, 12), (20, 0, 10.5)]


# ---------------------------------------------------------------------------
# STEP 1: LOAD, CLEAN, AND DOWNSAMPLE
# ---------------------------------------------------------------------------

def load_point_cloud(path):
    """
    Load a point cloud from disk. If the given path doesn't exist, fall back
    to one of Open3D's built-in sample datasets so this script runs even
    without your own scan file.
    """
    if os.path.exists(path):
        # pcd holds the loaded point cloud (points + optional colors/normals)
        pcd = o3d.io.read_point_cloud(path)
    else:
        # sample holds a reference to Open3D's built-in sample point cloud object
        sample = o3d.data.PLYPointCloud()
        # pcd holds the point cloud loaded from that built-in sample's file path
        pcd = o3d.io.read_point_cloud(sample.path)
        print(f"'{path}' not found — using Open3D's built-in sample point cloud instead.")
    return pcd  # returns the loaded Open3D PointCloud object


def downsample_point_cloud(pcd, voxel_size):
    """Reduce point density using a voxel grid filter, to speed up processing."""
    # downsampled holds the reduced-density version of the input point cloud
    downsampled = pcd.voxel_down_sample(voxel_size)
    return downsampled  # returns the downsampled PointCloud object


# ---------------------------------------------------------------------------
# STEP 2: REMOVE THE DOMINANT FLAT SURFACE (GROUND) WITH RANSAC
# ---------------------------------------------------------------------------

def remove_largest_plane(pcd, distance_threshold, ransac_n, num_iterations):
    """
    Detect and strip out the single largest flat surface in the scene
    (typically the ground), so clustering can focus on actual objects.
    """
    # plane_model holds the (a, b, c, d) coefficients of the detected plane: ax+by+cz+d=0
    # inlier_indices holds the list of point indices that belong to that plane
    plane_model, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    # objects_only holds every point NOT belonging to the detected plane
    objects_only = pcd.select_by_index(inlier_indices, invert=True)
    # ground_only holds only the points that make up the detected ground plane
    ground_only = pcd.select_by_index(inlier_indices)
    # returns (non-ground points, ground points, plane coefficients)
    return objects_only, ground_only, plane_model


def ground_height_at(plane_model, x, y):
    """
    Z of the ground plane (ax+by+cz+d=0) directly below (x, y). Falls back to
    0.0 if the plane is (near-)vertical, where "height under a point" is undefined.
    """
    a, b, c, d = plane_model
    if abs(c) < 1e-6:
        return 0.0
    return float(-(a * x + b * y + d) / c)


# ---------------------------------------------------------------------------
# STEP 3: CLUSTER REMAINING POINTS INTO INDIVIDUAL OBJECTS
# ---------------------------------------------------------------------------

def cluster_objects(pcd, eps, min_points):
    """
    Group the remaining (non-ground) points into distinct clusters, where
    each cluster is intended to represent one physical object in the scene.
    """
    # labels holds one integer cluster ID per point; -1 means "noise" (no cluster)
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )
    # cluster_count holds the total number of distinct clusters found (excluding noise)
    cluster_count = int(labels.max()) + 1 if labels.size else 0
    print(f"Found {cluster_count} candidate object clusters before cleanup and classification.")
    return labels, cluster_count  # returns (per-point cluster labels, number of clusters)


# ---------------------------------------------------------------------------
# STEP 4: CLEAN EACH OBJECT (STATISTICAL OUTLIER REMOVAL)
# ---------------------------------------------------------------------------

def clean_cluster(cluster_pcd):
    """
    Remove statistical outliers from a single object's point cloud, so its
    boundary isn't distorted by a few stray, noisy points.
    """
    # cleaned holds the cluster after outliers are dropped
    # _ holds the indices Open3D kept (unused; we already have `cleaned`)
    cleaned, _ = cluster_pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB_NEIGHBORS,
        std_ratio=OUTLIER_STD_RATIO,
    )
    return cleaned  # returns the outlier-cleaned PointCloud for this one object


# ---------------------------------------------------------------------------
# STEP 5: CLASSIFY EACH OBJECT (geometric heuristic — see module docstring)
# ---------------------------------------------------------------------------

def classify_object(height, foot_max, foot_min, base_z):
    """
    Assign a plain-language object type from basic shape measurements.

    This is intentionally simple — height, footprint, how elongated the
    footprint is, and how far above the ground the object's base sits — so the whole pipeline runs with zero model dependencies.
    THIS IS WHERE A TRAINED POINT-CLOUD CLASSIFIER (e.g. a PointNet++ or
    PointMAE model, taking the raw points as input instead of these four
    summary numbers) would replace this function in production. Everything
    downstream (the report, the exports) only cares that this function
    returns a (type, confidence) pair — swapping the implementation doesn't
    require changing anything else in the pipeline.
    """
    foot_avg = (foot_max + foot_min) / 2
    elongation = foot_max / max(foot_min, 0.05)

    if height > 7 and foot_max < 1.2:
        return "Pole", 0.9
    if 1.0 <= height <= 2.3 and 1.2 <= foot_avg <= 3.2 and base_z < 0.3:
        return "Vehicle", 0.85
    if 2.3 <= height <= 6 and foot_avg >= 2.8 and base_z < 0.3:
        return "Structure", 0.85
    if height > 4 and elongation < 1.7:
        return "Tree", 0.9
    if height < 1.2 and base_z < 0.6:
        return "Debris", 0.6
    return "Unclassified", 0.3


# ---------------------------------------------------------------------------
# STEP 6: GENERIC DISTANCE MEASUREMENT (the building block, not a feature)
# ---------------------------------------------------------------------------

def point_to_point_distance(a, b):
    """Straight-line 3D distance between two points. The simplest measurement primitive."""
    return float(np.linalg.norm(np.array(a) - np.array(b)))  # returns the Euclidean distance


def point_to_segment_distance(point, seg_a, seg_b):
    """
    Shortest distance from a point to a line segment (seg_a to seg_b). Useful
    whenever "distance to a linear feature" matters more than "distance to a
    single point" — a conductor, a property line, a pipeline run, etc.
    """
    p = np.array(point)  # p holds the point as a NumPy array for vector math
    a = np.array(seg_a)  # a holds the segment's first endpoint
    b = np.array(seg_b)  # b holds the segment's second endpoint
    ab = b - a  # ab holds the vector from a to b (the segment's direction)
    ab_len_sq = np.dot(ab, ab)  # ab_len_sq holds the squared length of the segment
    if ab_len_sq == 0:
        return float(np.linalg.norm(p - a))  # a and b coincide — just point-to-point
    # t holds how far along the segment (0 to 1) the closest point falls, clamped to the segment
    t = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
    closest = a + t * ab  # closest holds the actual closest point on the segment
    return float(np.linalg.norm(p - closest))  # returns the distance from point to that closest point


# ---------------------------------------------------------------------------
# STEP 7: PER-OBJECT STATS
# ---------------------------------------------------------------------------

def compute_object_stats(obj_id, cluster_pcd, plane_model=None):
    """Compute summary statistics and classification for one detected object."""
    # points holds this object's point coordinates as a NumPy array
    points = np.asarray(cluster_pcd.points)
    # mins/maxs hold the per-axis minimum and maximum coordinates (the bounding box)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    # centroid holds the average position of all points in this object
    centroid = points.mean(axis=0)
    # height holds the object's vertical extent (assumes Z is "up")
    height = float(maxs[2] - mins[2])
    # foot_max/foot_min hold the object's horizontal footprint dimensions
    foot_max = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    foot_min = float(min(maxs[0] - mins[0], maxs[1] - mins[1]))
    # ground_z holds the ground plane's height directly below this object's centroid
    ground_z = ground_height_at(plane_model, centroid[0], centroid[1]) if plane_model is not None else 0.0
    # base_z holds how far above the ground this object's lowest point sits
    base_z = float(mins[2]) - ground_z

    # obj_type/confidence hold the classification result for this object
    obj_type, confidence = classify_object(height, foot_max, foot_min, base_z)

    return {
        "object_id": obj_id,
        "type": obj_type,
        "confidence": round(confidence, 2),
        "point_count": len(points),
        "height_m": round(height, 2),
        "footprint_m": round((foot_max + foot_min) / 2, 2),
        "centroid_x": round(float(centroid[0]), 2),
        "centroid_y": round(float(centroid[1]), 2),
        "centroid_z": round(float(centroid[2]), 2),
    }  # returns a dict of this object's computed statistics


# ---------------------------------------------------------------------------
# STEP 8: EXPORT — PER-OBJECT .PLY (POINTS) + .OBJ (SOLID MESH)
# ---------------------------------------------------------------------------

def write_cube_obj(points, cube_size, out_path, obj_id, obj_type):
    """
    Write a real, valid, high-resolution solid mesh: one small cube per
    point. This is what makes the export an actual 3D object (importable
    into Unreal/Blender/game engines) rather than a file of floating
    vertices — every point becomes solid geometry, not just a coordinate.
    """
    s = cube_size / 2  # s holds the half-edge-length of each point's cube
    # offsets holds the 8 corner positions of a cube centered at the origin
    offsets = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s),
    ]
    # faces holds the 12 triangles (2 per cube face) as local 0-7 vertex indices,
    # wound counter-clockwise seen from outside so normals face outward
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 5, 4), (0, 1, 5), (3, 6, 2), (3, 7, 6),
        (1, 6, 5), (1, 2, 6), (0, 7, 3), (0, 4, 7),
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Cloudframe export — {obj_id} ({obj_type})\n")
        f.write(f"# {len(points)} points, solid cube-per-point mesh\n")
        for p in points:
            for ox, oy, oz in offsets:
                f.write(f"v {p[0] + ox:.4f} {p[1] + oy:.4f} {p[2] + oz:.4f}\n")
        for i in range(len(points)):
            base = i * 8  # base holds this point's starting vertex index (0-based)
            for a, b, c in faces:
                # OBJ face indices are 1-based, hence the +1s below
                f.write(f"f {base + a + 1} {base + b + 1} {base + c + 1}\n")


def export_objects(pcd, labels, cluster_count, output_dir, report_path, plane_model=None):
    """
    For each cluster above the minimum size: clean it, classify it, save it
    as both a .ply (point cloud) and a .obj (solid mesh), and add a row to
    the structured CSV report. Returns the list of accepted objects.
    """
    os.makedirs(output_dir, exist_ok=True)

    # points/colors hold the full non-ground point cloud's coordinates and colors
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    accepted_objects = []  # accepted_objects holds the stats for every cluster that passed all filters
    next_id = 1  # next_id holds a running counter for clean, sequential object IDs

    for cluster_id in range(cluster_count):
        # mask holds a True/False flag per point: True if it belongs to this cluster
        mask = labels == cluster_id
        if int(mask.sum()) < MIN_CLUSTER_SIZE:
            continue  # skip small clusters — likely noise/debris fragments

        # cluster_pcd holds a new PointCloud containing just this cluster's points
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(points[mask])
        if colors is not None:
            cluster_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        # cleaned_pcd holds the cluster after statistical outlier removal
        cleaned_pcd = clean_cluster(cluster_pcd)
        if len(cleaned_pcd.points) < MIN_CLUSTER_SIZE:
            continue  # cleaning dropped it below the size threshold — skip it

        # obj_id holds this object's display identifier, e.g. "OBJ-001"
        obj_id = f"OBJ-{next_id:03d}"
        # stats holds the computed statistics + classification for this object
        stats = compute_object_stats(obj_id, cleaned_pcd, plane_model)
        stats["cluster_id"] = cluster_id  # kept for internal use (visualization), not in the CSV
        accepted_objects.append(stats)

        # ply_path holds the point-cloud export path for this object
        ply_path = os.path.join(output_dir, f"{obj_id}.ply")
        o3d.io.write_point_cloud(ply_path, cleaned_pcd)

        # obj_path holds the solid-mesh export path for this object
        obj_path = os.path.join(output_dir, f"{obj_id}.obj")
        write_cube_obj(np.asarray(cleaned_pcd.points), EXPORT_CUBE_SIZE, obj_path, obj_id, stats["type"])

        print(f"Saved {obj_id}: {stats['type']} ({int(stats['confidence']*100)}% confidence), "
              f"{stats['point_count']} points, {stats['height_m']} m tall — {ply_path}, {obj_path}")

        next_id += 1

    write_report(accepted_objects, report_path)
    return accepted_objects  # returns the list of stats dicts for every exported object


def write_report(objects, report_path):
    """Write the structured, per-object CSV report."""
    # fieldnames holds the CSV column order
    fieldnames = [
        "object_id", "type", "confidence", "point_count", "height_m",
        "footprint_m", "centroid_x", "centroid_y", "centroid_z",
    ]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        # writer holds the CSV writer; extrasaction="ignore" drops internal-only
        # keys (like cluster_id) that aren't meant for the user-facing report
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for obj in objects:
            writer.writerow(obj)
    if not objects:
        print(f"No objects met the minimum size threshold — wrote empty report: {report_path}")
    else:
        print(f"Wrote report: {report_path} ({len(objects)} objects)")


# ---------------------------------------------------------------------------
# STEP 9: VISUALIZE — colored by detected class
# ---------------------------------------------------------------------------

def colorize_gray(pcd):
    """Return a copy of the cloud painted uniform gray — the raw "before" view."""
    gray = o3d.geometry.PointCloud(pcd)  # copy so the caller's cloud keeps its colors
    gray.paint_uniform_color([0.6, 0.6, 0.6])
    return gray  # returns the gray copy


def colorize_clusters(pcd, labels):
    """
    Return a copy of the cloud with a distinct color per DBSCAN cluster (the
    segmentation "after" view). Noise points (label -1) are black.
    """
    cmap = plt.get_cmap("tab20")  # cmap holds 20 visually distinct colors
    labels = np.asarray(labels)
    # colors holds one RGB color per point; tab20 is indexed directly, cycling past 20 clusters
    colors = cmap(np.maximum(labels, 0) % 20)[:, :3] if labels.size else np.zeros((0, 3))
    colors[labels < 0] = 0
    colored = o3d.geometry.PointCloud(pcd)
    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored  # returns the copy, colored by cluster


def colorize_by_class(pcd, labels, objects):
    """Color each point by its object's classified type, for a final visual check."""
    # class_colors holds the RGB color assigned to each object type
    class_colors = {
        "Tree": [0.12, 0.56, 0.39],
        "Vehicle": [0.15, 0.39, 0.66],
        "Structure": [0.73, 0.45, 0.05],
        "Pole": [0.49, 0.31, 0.80],
        "Debris": [0.72, 0.22, 0.23],
        "Unclassified": [0.48, 0.51, 0.57],
    }
    # cluster_to_type maps each object's raw DBSCAN cluster id back to its classified type
    cluster_to_type = {obj["cluster_id"]: obj["type"] for obj in objects}

    points = np.asarray(pcd.points)
    # colors holds one RGB color per point, default dim gray for noise/unclassified points
    colors = np.full((len(points), 3), 0.15)
    for i, label in enumerate(labels):
        obj_type = cluster_to_type.get(label)
        if obj_type in class_colors:
            colors[i] = class_colors[obj_type]

    colored = o3d.geometry.PointCloud(pcd)  # copy so the input cloud isn't modified
    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored  # returns a copy of the point cloud, colored by classified object type


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    """
    Run the full pipeline: load -> downsample -> remove ground -> cluster ->
    clean each object -> classify -> export (.ply + .obj) + CSV report.
    """
    raw_pcd = load_point_cloud(INPUT_PATH)  # raw_pcd holds the point cloud as loaded from disk
    print(f"Loaded point cloud with {len(raw_pcd.points)} points.")

    ds_pcd = downsample_point_cloud(raw_pcd, VOXEL_SIZE)  # ds_pcd holds the downsampled cloud
    print(f"Downsampled to {len(ds_pcd.points)} points.")

    # "Before" view: the raw downsampled scan in gray
    o3d.visualization.draw_geometries([colorize_gray(ds_pcd)], window_name="1 - Raw scan")

    if len(ds_pcd.points) < PLANE_RANSAC_N:
        print("Not enough points to detect a ground plane — nothing to segment.")
        return

    objects_pcd, ground_pcd, plane_model = remove_largest_plane(  # split point sets + plane coefficients
        ds_pcd, PLANE_DISTANCE_THRESHOLD, PLANE_RANSAC_N, PLANE_NUM_ITERATIONS
    )
    print(f"Removed ground: {len(ground_pcd.points)} points. "
          f"{len(objects_pcd.points)} points remain for clustering.")

    cluster_labels, num_clusters = cluster_objects(  # cluster_labels/num_clusters hold the DBSCAN result
        objects_pcd, DBSCAN_EPS, DBSCAN_MIN_POINTS
    )

    # "After" view: each segmented object in its own color
    o3d.visualization.draw_geometries([colorize_clusters(objects_pcd, cluster_labels)],
                                      window_name="2 - Segmented objects (colored by cluster)")

    accepted_objects = export_objects(  # accepted_objects holds the final list of exported object stats
        objects_pcd, cluster_labels, num_clusters, OUTPUT_DIR, REPORT_PATH, plane_model
    )

    if accepted_objects:
        type_counts = {}
        for obj in accepted_objects:
            type_counts[obj["type"]] = type_counts.get(obj["type"], 0) + 1
        summary = ", ".join(f"{count} {t}" for t, count in sorted(type_counts.items()))
        print(f"\nClassification summary: {summary} (of {len(accepted_objects)} objects).")

    if MEASURE_POINTS is not None:
        # distance holds the straight-line distance between the two configured points
        distance = point_to_point_distance(MEASURE_POINTS[0], MEASURE_POINTS[1])
        print(f"\nMEASURE_POINTS distance: {distance:.2f} m")

    # colored_pcd holds the object cloud colored by classified type, for a final visual check
    colored_pcd = colorize_by_class(objects_pcd, cluster_labels, accepted_objects)
    o3d.visualization.draw_geometries([colored_pcd], window_name="3 - Objects by classified type")

    print(f"\nDone. {len(accepted_objects)} objects exported to '{OUTPUT_DIR}/' "
          f"(.ply + .obj each), report written to '{REPORT_PATH}'.")


if __name__ == "__main__":
    main()