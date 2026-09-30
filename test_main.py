"""Tests for the Cloudframe pipeline. Run with:  python -m unittest -v test_main"""

import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import open3d as o3d

import cloudframe_pipeline as main


def make_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return pcd


def box_surface(center, size, step):
    """Points on the surface of an axis-aligned box."""
    cx, cy, cz = center
    sx, sy, sz = size
    xs = np.arange(-sx / 2, sx / 2 + 1e-9, step)
    ys = np.arange(-sy / 2, sy / 2 + 1e-9, step)
    zs = np.arange(0, sz + 1e-9, step)
    pts = []
    for x in xs:
        for y in ys:
            pts += [(x, y, 0), (x, y, sz)]
        for z in zs:
            pts += [(x, -sy / 2, z), (x, sy / 2, z)]
    for y in ys:
        for z in zs:
            pts += [(-sx / 2, y, z), (sx / 2, y, z)]
    return np.array(pts) + np.array([cx, cy, cz])


def synthetic_scene(ground_z=5.0):
    """Flat ground at ground_z with a vehicle, a pole, and a low debris pile on it."""
    rng = np.random.default_rng(0)
    g = np.arange(-15, 15, 0.1)
    gx, gy = np.meshgrid(g, g)
    ground = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, ground_z)]
    vehicle = box_surface((-6, 0, ground_z + 0.2), (4.0, 1.8, 1.5), 0.1)
    pole = box_surface((6, 6, ground_z + 0.2), (0.3, 0.3, 9.0), 0.1)
    debris = box_surface((6, -6, ground_z + 0.2), (1.0, 1.0, 0.6), 0.1)
    pts = np.vstack([ground, vehicle, pole, debris]) + rng.normal(0, 0.002, (1, 3))
    return make_pcd(pts)


class TestLoadAndDownsample(unittest.TestCase):
    def test_load_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scan.ply")
            o3d.io.write_point_cloud(path, make_pcd(np.random.rand(50, 3)))
            self.assertEqual(len(main.load_point_cloud(path).points), 50)

    def test_load_missing_file_falls_back_to_sample(self):
        fake = mock.Mock(path="sample.ply")
        with mock.patch.object(o3d.data, "PLYPointCloud", return_value=fake), \
             mock.patch.object(o3d.io, "read_point_cloud", return_value=make_pcd([[0, 0, 0]])) as rd:
            pcd = main.load_point_cloud("does_not_exist.ply")
        rd.assert_called_once_with("sample.ply")
        self.assertEqual(len(pcd.points), 1)

    def test_downsample_reduces_points(self):
        pcd = make_pcd(np.random.default_rng(1).random((2000, 3)))
        self.assertLess(len(main.downsample_point_cloud(pcd, 0.2).points), 2000)


class TestGroundRemoval(unittest.TestCase):
    def test_removes_plane_and_returns_model(self):
        pcd = synthetic_scene(ground_z=5.0)
        objs, ground, model = main.remove_largest_plane(pcd, 0.02, 3, 1000)
        self.assertEqual(len(objs.points) + len(ground.points), len(pcd.points))
        self.assertTrue(np.all(np.abs(np.asarray(ground.points)[:, 2] - 5.0) < 0.05))
        self.assertAlmostEqual(main.ground_height_at(model, 1.0, 2.0), 5.0, delta=0.02)

    def test_ground_height_at(self):
        self.assertAlmostEqual(main.ground_height_at([0, 0, 1, -3], 7, 7), 3.0)
        self.assertAlmostEqual(main.ground_height_at([0.1, 0, 1, 0], 10, 0), -1.0)
        self.assertEqual(main.ground_height_at([1, 0, 0, 0], 1, 1), 0.0)  # vertical plane


class TestClustering(unittest.TestCase):
    def test_two_separate_blobs(self):
        rng = np.random.default_rng(2)
        pts = np.vstack([rng.normal(0, 0.02, (100, 3)), rng.normal(5, 0.02, (100, 3))])
        labels, count = main.cluster_objects(make_pcd(pts), 0.1, 10)
        self.assertEqual(count, 2)
        self.assertEqual(len(labels), 200)

    def test_empty_cloud_does_not_crash(self):
        labels, count = main.cluster_objects(o3d.geometry.PointCloud(), 0.1, 10)
        self.assertEqual(count, 0)
        self.assertEqual(len(labels), 0)


class TestCleanCluster(unittest.TestCase):
    def test_removes_far_outlier(self):
        rng = np.random.default_rng(3)
        pts = np.vstack([rng.normal(0, 0.01, (200, 3)), [[10, 10, 10]]])
        cleaned = main.clean_cluster(make_pcd(pts))
        self.assertLess(np.asarray(cleaned.points).max(), 1.0)


class TestClassify(unittest.TestCase):
    def test_each_class(self):
        c = main.classify_object
        self.assertEqual(c(9, 0.3, 0.3, 0)[0], "Pole")
        self.assertEqual(c(1.5, 4, 1.8, 0.1)[0], "Vehicle")
        self.assertEqual(c(4, 6, 5, 0)[0], "Structure")
        self.assertEqual(c(10, 3, 2.5, 0)[0], "Tree")
        self.assertEqual(c(0.5, 1, 1, 0.1)[0], "Debris")
        self.assertEqual(c(3, 1, 0.2, 2)[0], "Unclassified")

    def test_returns_type_and_confidence(self):
        t, conf = main.classify_object(9, 0.3, 0.3, 0)
        self.assertIsInstance(t, str)
        self.assertTrue(0 <= conf <= 1)


class TestDistances(unittest.TestCase):
    def test_point_to_point(self):
        self.assertAlmostEqual(main.point_to_point_distance((0, 0, 0), (3, 4, 0)), 5.0)

    def test_point_to_segment(self):
        f = main.point_to_segment_distance
        self.assertAlmostEqual(f((5, 3, 0), (0, 0, 0), (10, 0, 0)), 3.0)   # perpendicular
        self.assertAlmostEqual(f((-3, 4, 0), (0, 0, 0), (10, 0, 0)), 5.0)  # clamps to a
        self.assertAlmostEqual(f((13, 4, 0), (0, 0, 0), (10, 0, 0)), 5.0)  # clamps to b
        self.assertAlmostEqual(f((3, 4, 0), (0, 0, 0), (0, 0, 0)), 5.0)    # degenerate segment


class TestObjectStats(unittest.TestCase):
    def test_stats_and_ground_relative_base(self):
        # A 1.5 m tall vehicle-sized box sitting on ground at z=5.
        pcd = make_pcd(box_surface((0, 0, 5.1), (4.0, 1.8, 1.5), 0.1))
        with_plane = main.compute_object_stats("OBJ-001", pcd, [0, 0, 1, -5])
        self.assertEqual(with_plane["type"], "Vehicle")
        self.assertAlmostEqual(with_plane["height_m"], 1.5, delta=0.01)
        self.assertAlmostEqual(with_plane["footprint_m"], 2.9, delta=0.01)
        self.assertEqual(with_plane["point_count"], len(pcd.points))
        # Without the ground plane the absolute Z (5.1) is treated as base height.
        self.assertNotEqual(main.compute_object_stats("OBJ-001", pcd)["type"], "Vehicle")


class TestExports(unittest.TestCase):
    def test_cube_obj_is_valid_and_outward_facing(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "o.obj")
            main.write_cube_obj(pts, 0.1, path, "OBJ-001", "Pole")
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        verts = np.array([list(map(float, l.split()[1:])) for l in lines if l.startswith("v ")])
        faces = [list(map(int, l.split()[1:])) for l in lines if l.startswith("f ")]
        self.assertEqual(len(verts), 16)
        self.assertEqual(len(faces), 24)
        self.assertTrue(all(1 <= i <= 16 for f in faces for i in f))
        for f in faces:
            a, b, c = verts[f[0] - 1], verts[f[1] - 1], verts[f[2] - 1]
            normal = np.cross(b - a, c - a)
            center = pts[(f[0] - 1) // 8]
            self.assertGreater(np.dot(normal, (a + b + c) / 3 - center), 0, f"inward face {f}")

    def test_write_report_empty_still_writes_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            with open(path, "w") as f:
                f.write("stale\n")
            main.write_report([], path)
            with open(path, encoding="utf-8") as f:
                self.assertTrue(f.read().startswith("object_id,type"))

    def test_export_objects_and_report(self):
        rng = np.random.default_rng(4)
        pts = np.vstack([rng.normal(0, 0.05, (100, 3)), rng.normal(5, 0.05, (100, 3)),
                         rng.normal(-5, 0.05, (5, 3))])  # third cluster too small
        pcd = make_pcd(pts)
        labels = np.array([0] * 100 + [1] * 100 + [2] * 5)
        with tempfile.TemporaryDirectory() as d:
            out, rpt = os.path.join(d, "objs"), os.path.join(d, "r.csv")
            objs = main.export_objects(pcd, labels, 3, out, rpt)
            self.assertEqual([o["object_id"] for o in objs], ["OBJ-001", "OBJ-002"])
            self.assertEqual(sorted(os.listdir(out)),
                             ["OBJ-001.obj", "OBJ-001.ply", "OBJ-002.obj", "OBJ-002.ply"])
            with open(rpt, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertNotIn("cluster_id", rows[0])

    def test_colorize_by_class(self):
        pcd = make_pcd(np.zeros((3, 3)))
        labels = np.array([0, 1, -1])
        objs = [{"cluster_id": 0, "type": "Tree"}, {"cluster_id": 1, "type": "Pole"}]
        colors = np.asarray(main.colorize_by_class(pcd, labels, objs).colors)
        np.testing.assert_allclose(colors[0], [0.12, 0.56, 0.39])
        np.testing.assert_allclose(colors[1], [0.49, 0.31, 0.80])
        np.testing.assert_allclose(colors[2], [0.15, 0.15, 0.15])
        self.assertFalse(pcd.has_colors())  # input cloud is not modified

    def test_colorize_gray(self):
        pcd = make_pcd(np.zeros((4, 3)))
        pcd.paint_uniform_color([1, 0, 0])
        gray = main.colorize_gray(pcd)
        np.testing.assert_allclose(np.asarray(gray.colors), 0.6)
        np.testing.assert_allclose(np.asarray(pcd.colors)[0], [1, 0, 0])

    def test_colorize_clusters(self):
        pcd = make_pcd(np.zeros((4, 3)))
        colors = np.asarray(main.colorize_clusters(pcd, np.array([0, 1, 1, -1])).colors)
        self.assertFalse(np.allclose(colors[0], colors[1]))
        np.testing.assert_allclose(colors[1], colors[2])
        np.testing.assert_allclose(colors[3], 0)  # noise is black
        self.assertFalse(pcd.has_colors())
        empty = main.colorize_clusters(o3d.geometry.PointCloud(), np.array([], dtype=int))
        self.assertEqual(len(empty.points), 0)


class TestMainPipeline(unittest.TestCase):
    def test_end_to_end_on_elevated_ground(self):
        with tempfile.TemporaryDirectory() as d:
            scan = os.path.join(d, "scan.ply")
            o3d.io.write_point_cloud(scan, synthetic_scene(ground_z=5.0))
            cfg = dict(INPUT_PATH=scan, OUTPUT_DIR=os.path.join(d, "out"),
                       REPORT_PATH=os.path.join(d, "r.csv"), VOXEL_SIZE=0.05,
                       PLANE_DISTANCE_THRESHOLD=0.05, DBSCAN_EPS=0.25, DBSCAN_MIN_POINTS=5,
                       MEASURE_POINTS=[(0, 0, 0), (3, 4, 0)])
            with mock.patch.multiple(main, **cfg), \
                 mock.patch.object(o3d.visualization, "draw_geometries") as draw:
                main.main()
            titles = [c.kwargs["window_name"] for c in draw.call_args_list]
            self.assertEqual(titles, ["1 - Raw scan", "2 - Segmented objects (colored by cluster)",
                                      "3 - Objects by classified type"])
            raw_colors = np.asarray(draw.call_args_list[0].args[0][0].colors)
            np.testing.assert_allclose(raw_colors, 0.6)  # "before" view is all gray
            seg_colors = np.asarray(draw.call_args_list[1].args[0][0].colors)
            self.assertGreaterEqual(len(np.unique(seg_colors, axis=0)), 3)  # one color per object
            with open(cfg["REPORT_PATH"], encoding="utf-8") as f:
                types = sorted(r["type"] for r in csv.DictReader(f))
        self.assertEqual(types, ["Debris", "Pole", "Vehicle"])

    def test_too_few_points_exits_cleanly(self):
        with mock.patch.object(main, "load_point_cloud", return_value=make_pcd([[0, 0, 0]])), \
             mock.patch.object(o3d.visualization, "draw_geometries") as draw:
            main.main()
        draw.assert_called_once()  # only the raw "before" view; no segmentation views


if __name__ == "__main__":
    unittest.main()
