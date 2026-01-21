# -- coding: utf-8 --

"""
Datasets/Colmap.py: Provides a dataset class for scenes preprocessed by Colmap.
Code partially copied and adapted from:
https://github.com/colmap/colmap/blob/dev/scripts/python
and
https://github.com/graphdeco-inria/gaussian-splatting/
"""

import os
from pathlib import Path

import numpy as np
import collections
import struct
import torch
from plyfile import PlyData, PlyElement

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import CameraProperties, RadialTangentialDistortion
from Datasets.Base import BaseDataset
from Datasets.utils import CameraCoordinateSystemsTransformations, WorldCoordinateSystemTransformations, applyImageScaleFactor, getNearFarFromPointCloud, loadImagesParallel, \
                           BasicPointCloud, loadOpticalFlowParallel, loadImage, transformPosesPCA
from Logging import Logger


# Utility functions for reading COLMAP files.

def quaternion_to_R(q):
    return np.eye(3) + 2 * np.array((
        (-q[2] * q[2] - q[3] * q[3], q[1] * q[2] -
         q[3] * q[0], q[1] * q[3] + q[2] * q[0]),
        (q[1] * q[2] + q[3] * q[0], -q[1] * q[1] -
         q[3] * q[3], q[2] * q[3] - q[1] * q[0]),
        (q[1] * q[3] - q[2] * q[0],
         q[2] * q[3] + q[1] * q[0],
         -q[1] * q[1] - q[2] * q[2])))


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                if xyzs is None:
                    xyzs = xyz[None, ...]
                    rgbs = rgb[None, ...]
                    errors = error[None, ...]
                else:
                    xyzs = np.append(xyzs, xyz[None, ...], axis=0)
                    rgbs = np.append(rgbs, rgb[None, ...], axis=0)
                    errors = np.append(errors, error[None, ...], axis=0)
    return xyzs, rgbs, errors


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            _ = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return BasicPointCloud(
        positions=torch.from_numpy(positions).float().cpu(),
        colors=torch.from_numpy(colors).float().cpu(),
    )


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


@Framework.Configurable.configure(
    PATH='dataset/colmap/myscene',
    BACKGROUND_COLOR=[0.0, 0.0, 0.0],
    TEST_STEP=0,
    APPLY_PCA=False,
)
class CustomDataset(BaseDataset):
    """Dataset class for Colmap-calibrated scenes."""

    def __init__(self, path: str) -> None:
        super().__init__(
            path,
            PerspectiveCamera(0.01, 1.0),  # will be updated according to sfm pointcloud
            CameraCoordinateSystemsTransformations.LEFT_HAND,
            WorldCoordinateSystemTransformations.XnZY
        )
        self.camera.near_plane, self.camera.far_plane = getNearFarFromPointCloud(self.camera, self.point_cloud, self.data['train'], 0.1)

    def load(self) -> dict[str, list[CameraProperties] | None]:
        """Loads the dataset into a dict containing lists of CameraProperties for training, evaluation, and testing."""
        dataset: dict[str, list[CameraProperties]] = {subset: [] for subset in self.subsets}
        load_segmentation = Path(self.dataset_path / 'sfm_masks').exists()
        load_flow = Path(self.dataset_path / 'flow').exists()
        load_disp = Path(self.dataset_path / 'monoc_depth').exists()
        # load colmap data
        try:
            cameras_extrinsic_file = self.dataset_path / 'sparse' / '0' / 'images.bin'
            cameras_intrinsic_file = self.dataset_path / 'sparse' / '0' / 'cameras.bin'
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except Exception:
            cameras_extrinsic_file = self.dataset_path / 'sparse' / '0' / 'images.txt'
            cameras_intrinsic_file = self.dataset_path / 'sparse' / '0' / 'cameras.txt'
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        # create camera properties
        data: list[CameraProperties] = []
        for cam_data in cam_intrinsics.values():
            images = [data for data in cam_extrinsics.values() if data.camera_id == cam_data.id]
            images = sorted(images, key=lambda data: data.name)
            # load images
            image_filenames = [str(self.dataset_path / 'images' / image.name) for image in images]
            rgbs, alphas = loadImagesParallel(image_filenames, self.IMAGE_SCALE_FACTOR, num_threads=12, desc=f'camera {cam_data.id}')
            forward_flows, backward_flows = [None] * len(rgbs), [None] * len(rgbs)
            if load_flow:
                # load optical flow
                forward_flows, backward_flows = loadOpticalFlowParallel(self.dataset_path / 'flow', [image.name for image in images],
                                                                        num_threads=4, image_scale_factor=self.IMAGE_SCALE_FACTOR)
            for idx, (image, rgb, alpha) in enumerate(zip(images, rgbs, alphas)):
                # extract c2w matrix
                rotation_matrix = torch.from_numpy(quaternion_to_R(image.qvec)).float()
                translation_vector = torch.from_numpy(image.tvec).float()
                w2c = torch.eye(4, device=torch.device('cpu'))
                w2c[:3, :3] = rotation_matrix
                w2c[:3, 3] = translation_vector
                c2w = torch.linalg.inv(w2c)
                # intrinsics
                focal_x = cam_data.params[0]
                focal_y = cam_data.params[1]
                principal_offset_x = (cam_data.params[2] - cam_data.width / 2)
                principal_offset_y = (cam_data.params[3] - cam_data.height / 2)
                if self.IMAGE_SCALE_FACTOR is not None:
                    scale_factor_intrinsics_x = rgb.shape[2] / cam_data.width
                    scale_factor_intrinsics_y = rgb.shape[1] / cam_data.height
                    focal_x *= scale_factor_intrinsics_x
                    focal_y *= scale_factor_intrinsics_y
                    principal_offset_x *= scale_factor_intrinsics_x
                    principal_offset_y *= scale_factor_intrinsics_y
                # distortion parameters
                distortion_parameters: RadialTangentialDistortion | None = None
                match cam_data.model:
                    case 'SIMPLE_PINHOLE':
                        distortion_parameters = None
                    case 'PINHOLE':
                        distortion_parameters = None
                    case 'SIMPLE_RADIAL':
                        distortion_parameters = RadialTangentialDistortion(
                            k1=cam_data.params[3]
                        )
                    case 'RADIAL':
                        distortion_parameters = RadialTangentialDistortion(
                            k1=cam_data.params[3],
                            k2=cam_data.params[4],
                        )
                    case 'OPENCV':
                        distortion_parameters = RadialTangentialDistortion(
                            k1=cam_data.params[4],
                            k2=cam_data.params[5],
                            p1=cam_data.params[6],
                            p2=cam_data.params[7],
                        )
                    case _:
                        raise Framework.DatasetError(f'Unknown camera model "{cam_data.model}"')
                disp = segmentation = None
                if load_disp:
                    disp = torch.from_numpy(np.load(self.dataset_path / 'monoc_depth' / f'{Path(image_filenames[idx]).name}.npy'))
                    if self.IMAGE_SCALE_FACTOR is not None:
                        disp = applyImageScaleFactor(disp, self.IMAGE_SCALE_FACTOR, 'nearest')
                if load_segmentation:
                    segmentation, _ = loadImage(self.dataset_path / 'sfm_masks' / f'{Path(image_filenames[idx]).name}.png', scale_factor=self.IMAGE_SCALE_FACTOR)
                    segmentation = 1.0 - segmentation
                # create camera properties and subsets
                data.append(CameraProperties(
                    width=rgb.shape[2],
                    height=rgb.shape[1],
                    rgb=rgb,
                    _misc=disp,
                    segmentation=segmentation,
                    alpha=alpha,
                    c2w=c2w,
                    focal_x=focal_x,
                    focal_y=focal_y,
                    principal_offset_x=principal_offset_x,
                    principal_offset_y=principal_offset_y,
                    distortion_parameters=distortion_parameters,
                    forward_flow=forward_flows[idx],
                    backward_flow=backward_flows[idx],
                    timestamp=idx / (len(images) - 1),
                ))

        # load point cloud
        ply_path = self.dataset_path / 'sparse' / '0' / 'points3D.ply'
        bin_path = self.dataset_path / 'sparse' / '0' / 'points3D.bin'
        txt_path = self.dataset_path / 'sparse' / '0' / 'points3D.txt'
        if not os.path.exists(ply_path):
            Logger.logInfo('Found new scene. Converting sparse SfM points to .ply format.')
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except Exception:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            self.point_cloud = fetchPly(str(ply_path))
        except Exception:
            raise Framework.DatasetError(f'Could not load point cloud from "{ply_path}"')

        if self.APPLY_PCA:
            # rotate/scale poses to align ground with xy plane and fit to [-1, 1]^3 cube
            c2ws = torch.stack([camera.c2w for camera in data])
            c2ws, transformation = transformPosesPCA(c2ws)
            for camera_properties, c2w in zip(data, c2ws):
                camera_properties.c2w = c2w
            self.point_cloud.transform(transformation)
            self.world_coordinate_system = None

        # extract bounding box
        self.point_cloud.filterOutliers(filter_percentage=0.95)
        self._bounding_box = self.point_cloud.getBoundingBox(tolerance_factor=0.05)

        # perform test split
        if self.TEST_STEP > 0:
            for i in range(len(data)):
                if i % self.TEST_STEP == 0:
                    dataset['test'].append(data[i])
                else:
                    dataset['train'].append(data[i])
        else:
            dataset['train'] = data
        return dataset
