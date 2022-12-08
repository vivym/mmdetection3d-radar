from typing import List, Tuple
from pathlib import Path
import pickle
from unicodedata import category

import torch
import open3d as o3d
import numpy as np
from tqdm import tqdm

from tools.data_converter.create_gt_database import create_groundtruth_database
from tools.data_converter.kitti_data_utils import add_difficulty_to_annos
from mmdet3d.core.bbox.structures import (
    LiDARInstance3DBoxes, Box3DMode
)

LABEL_SUFFIXES = {
    "ouster": "",
    "oculii": "_oculii",
    "rs": "_rs",
}

CATEGORIES = {
    "car", "cyclist", "pedestrian", "cone"
}


def load_raw_data(raw_data_root: Path, sensor_type: str):
    label_suffix = LABEL_SUFFIXES[sensor_type]

    data_paths = []
    idx = 0

    for prefix in ["part_1", "part_2"]:
        pcd_dir = raw_data_root / f"{prefix}_pcd"
        label_dir = raw_data_root / f"{prefix}_label"

        for scene_type_dir in pcd_dir.glob("*"):
            if not scene_type_dir.is_dir():
                continue

            scene_type = scene_type_dir.name

            for pcd_path in (scene_type_dir / sensor_type).glob("*.pcd"):
                scene_id = pcd_path.stem
                file_name = f"{scene_id}{label_suffix}.txt"

                label_path = label_dir / scene_type_dir.name / "3d" / file_name
                if not label_path.exists():
                    label_path = label_dir / scene_type_dir.name / file_name
                if not label_path.exists():
                    continue

                scene_id = f"{prefix}_{scene_type}_{scene_id}"
                data_paths.append((idx, scene_type, pcd_path, label_path))
                idx += 1

    return data_paths


def split_train_val(data_paths: List[Tuple[str, str, Path, Path]]):
    train_paths = filter(lambda x: x[1] != "park_2", data_paths)
    val_paths = filter(lambda x: x[1] == "park_2", data_paths)

    return list(train_paths), list(val_paths)


def parse_kitti_line(line):
    items = line.split(" ")
    name = items[0].capitalize()
    h, w, l = map(float, items[8:11])
    x, y, z, yaw = map(float, items[11:15])

    if np.isnan([h, w, l, x, y, z, yaw]).any():
        print("Skip line:", line)
        return None

    return {
        "location": [x, y, z],
        "dimension": [l, w, h],
        "yaw": yaw,
        "name": name,
        "bbox": [0., 0., 100., 100.],
    }


def save_data_infos(
    data_root: Path,
    split: str,
    data_paths: List[Tuple[str, str, Path, Path]],
):
    all_dimensions = {k: [] for k in CATEGORIES}

    data_infos = []
    for idx, scene_type, pcd_path, label_path in tqdm(data_paths):
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)

        assert not np.any(np.isnan(points))

        if colors.shape[0] > 0:
            points = np.concatenate([points, colors[:, 0:1]], axis=1)
        else:
            points = np.concatenate([points, points[:, 2:3]], axis=1)

        assert points.shape[0] > 0
        points.tofile(data_root / "training" / "velodyne" / f"{idx:06d}.bin")
        points.tofile(data_root / "training" / "velodyne_reduced" / f"{idx:06d}.bin")

        with open(label_path) as f:
            lines = map(lambda x: x.strip(), f.readlines())
            lines = filter(
                lambda x: len(x) > 0 and x.split(" ")[0] in CATEGORIES, lines
            )
            annos = map(parse_kitti_line, lines)
            annos = filter(lambda x: x is not None, annos)
            annos = list(annos)

        rt_mat = np.asarray([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
        Trv2c = np.eye(4)
        Trv2c[:3, :3] = rt_mat
        P2 = np.asarray([
            [0.0001, 0.0000, 0.0000, 1.],
            [0.0000, 0.0001, 0.0000, 1.],
            [0.0000, 0.0000, 0.0001, 1.],
            [0.0000, 0.0000, 0.0000, 1.],
        ])

        locations = np.asarray([anno["location"] for anno in annos])
        dimensions = np.asarray([anno["dimension"] for anno in annos])
        yaws = np.asarray([anno["yaw"] for anno in annos])
        name = np.asarray([anno["name"] for anno in annos])
        bbox = np.asarray([anno["bbox"] for anno in annos])

        xy = locations[:, :2]
        dist = (xy ** 2).sum(-1) ** 0.5
        mask = dist < 12

        locations = locations[mask]
        dimensions = dimensions[mask]
        yaws = yaws[mask]
        name = name[mask]
        bbox = bbox[mask]

        for anno in annos:
            category, dimension = anno["name"], anno["dimension"]
            all_dimensions[category.lower()].append(dimension)

        locations[:, 2] -= dimensions[:, 2] / 2.

        bboxes_3d = np.concatenate([locations, dimensions, yaws[:, np.newaxis]], axis=1)
        bboxes_3d_cam = LiDARInstance3DBoxes(bboxes_3d).convert_to(
            Box3DMode.CAM, rt_mat=Trv2c
        ).tensor.numpy()
        location = bboxes_3d_cam[:, 0:3]
        dimensions = bboxes_3d_cam[:, 3:6]
        rotation_y = bboxes_3d_cam[:, 6]

        assert not np.any(np.isnan(location)), (
            location[13], annos[13],
        )

        data_info = {
            "image": {
                "image_idx": idx,
                "image_path": "",
                "image_shape": (10000, 10000),
            },
            "calib": {
                "R0_rect": np.eye(4),
                "Tr_velo_to_cam": Trv2c,
                "P2": P2,
            },
            "annos": {
                "location": location,
                "dimensions": dimensions,
                "rotation_y": rotation_y,
                "name": name,
                "bbox": bbox,
                "occluded": np.zeros(location.shape[0]),
                "truncated": np.zeros(location.shape[0]),
                "alpha": np.ones(location.shape[0]) * -10,
            },
        }
        add_difficulty_to_annos(data_info)
        data_infos.append(data_info)

    for category, dimensions in all_dimensions.items():
        print(category, "mean dimensions:", np.mean(dimensions, axis=0))

    with open(data_root / f"kitti_infos_{split}.pkl", "wb") as f:
        pickle.dump(data_infos, f)


def main():
    sensor_type = "rs"

    if sensor_type == "ouster":
        data_root = Path("data/kitti2")
    else:
        data_root = Path("data/kitti2") / sensor_type
    velodyne_path1 = data_root / "training" / "velodyne"
    if not velodyne_path1.exists():
        velodyne_path1.mkdir(parents=True)
    velodyne_path2 = data_root / "training" / "velodyne_reduced"
    if not velodyne_path2.exists():
        velodyne_path2.mkdir(parents=True)

    data_paths = load_raw_data(Path("data") / "kitti2_raw_data", sensor_type)
    train_paths, val_paths = split_train_val(data_paths)

    save_data_infos(data_root, "val", val_paths)


if __name__ == "__main__":
    main()
