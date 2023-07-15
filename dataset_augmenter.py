import open3d as o3d
import torch
import numpy as np

import objaverse
import multiprocessing

import os
import glob
import shutil

from tqdm import tqdm


class DatasetAugmenter:
    def __init__(self, path2d: str, path3d: str):
        self.path2d = path2d
        self.path3d = path3d

        self.model_triangle_meshes = {}
        self.model_positions = {}

        self.scene_name = None
        self.scene_point_cloud = None
        self.scene_labels = None

    def load_model(self, uid, label, rot_axis_angle=None, height=None):
        if label in self.model_triangle_meshes:
            print(f'Label {label} already present!')
            return

        processes = multiprocessing.cpu_count()
        objects = objaverse.load_objects(uids=[uid], download_processes=processes)

        self.model_triangle_meshes[label] = o3d.io.read_triangle_mesh(objects[uid], True)

        if rot_axis_angle is not None:
            self.rotate_model(label, rot_axis_angle)
        if height is not None:
            self.scale_model(label, height)
        self.__center_model(label)

        self.model_positions[label] = np.array([0., 0., 0.])

    def visualize_model(self, label):
        o3d.visualization.draw_plotly([self.model_triangle_meshes[label]])

    def rotate_model(self, label, rot_axis_angle):
        rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis_angle)
        self.model_triangle_meshes[label].rotate(rot_matrix)
        self.__center_model(label)

    def scale_model(self, label, height):
        model_bounding_box = self.model_triangle_meshes[label].get_axis_aligned_bounding_box()
        scale_factor = height / model_bounding_box.max_bound[2]
        self.model_triangle_meshes[label].scale(scale_factor, np.zeros(3))
        self.__center_model(label)

    def __center_model(self, label):
        model_bounding_box = self.model_triangle_meshes[label].get_axis_aligned_bounding_box()
        bb_center = model_bounding_box.get_center()
        translation = np.array([-bb_center[0], -bb_center[1], -model_bounding_box.min_bound[2]])

        self.model_triangle_meshes[label].translate(translation)
        model_bounding_box.translate(translation)

    def load_scene(self, scene_name):
        self.scene_name = scene_name
        scene_files = glob.glob(os.path.join(self.path3d, f'**/{scene_name}*'), recursive=True)  # can be train/val
        assert len(scene_files) == 1

        scene_path = scene_files[0]
        scene_array, scene_colors, self.scene_labels = torch.load(scene_path)

        self.scene_point_cloud = o3d.geometry.PointCloud()
        self.scene_point_cloud.points = o3d.utility.Vector3dVector(scene_array)
        self.scene_point_cloud.colors = o3d.utility.Vector3dVector(scene_colors)

    def visualize_scene(self):
        o3d.visualization.draw_plotly([self.scene_point_cloud])

    def place_model_in_scene(self, xy_position, label):
        model_bounding_box = self.model_triangle_meshes[label].get_axis_aligned_bounding_box()
        bb_min = model_bounding_box.get_min_bound()
        bb_max = model_bounding_box.get_max_bound()

        scene_points = np.asarray(self.scene_point_cloud.points)
        lower_bound_mask = (xy_position + bb_min[:2] < scene_points[:, :2]).all(axis=1)
        upper_bound_mask = (xy_position + bb_max[:2] > scene_points[:, :2]).all(axis=1)
        bound_mask = lower_bound_mask & upper_bound_mask

        assert bound_mask.sum() > 0

        max_height = scene_points[bound_mask][:, 2].max()
        self.model_positions[label] = np.concatenate([xy_position, np.array([max_height])])

    def visualize_placement(self):
        for label, mesh in self.model_triangle_meshes.items():
            mesh.translate(self.model_positions[label])
        o3d.visualization.draw_plotly([self.scene_point_cloud] +
                                      [mesh for label, mesh in self.model_triangle_meshes.items()])
        for label, mesh in self.model_triangle_meshes.items():
            mesh.translate(-self.model_positions[label])

    def export(self, output_path):
        assert os.path.isdir(output_path)

        output_path2d = os.path.join(output_path, '2d')
        os.makedirs(output_path2d, exist_ok=True)
        output_path3d = os.path.join(output_path, '3d', 'train')
        os.makedirs(output_path3d, exist_ok=True)

        augmented_scene_array = np.asarray(self.scene_point_cloud.points)
        augmented_scene_colors = np.asarray(self.scene_point_cloud.colors)
        augmented_scene_labels = self.scene_labels

        # 3d export
        for label, mesh in self.model_triangle_meshes.items():
            mesh.translate(self.model_positions[label])
            model_point_cloud = mesh.sample_points_uniformly(1000)
            mesh.translate(-self.model_positions[label])

            model_point_cloud = model_point_cloud.voxel_down_sample(voxel_size=0.05)

            if np.asarray(model_point_cloud.colors).size == 0:
                model_point_cloud.paint_uniform_color(np.array([0, 0, 0]))

            augmented_scene_array = np.concatenate([augmented_scene_array,
                                                    np.asarray(model_point_cloud.points)])
            augmented_scene_colors = np.concatenate([augmented_scene_colors,
                                                     np.asarray(model_point_cloud.colors)])
            augmented_scene_labels = np.concatenate([augmented_scene_labels,
                                                     np.repeat(label, len(model_point_cloud.points))])

        torch.save((augmented_scene_array, augmented_scene_colors, augmented_scene_labels),
                   os.path.join(output_path3d, f'{self.scene_name}_vh.pth'))

        # 2d export
        intrinsic_path = os.path.join(self.path2d, 'intrinsics.txt')
        aug_intrinsic_path = os.path.join(output_path2d, 'intrinsics.txt')
        shutil.copyfile(intrinsic_path, aug_intrinsic_path)

        intrinsic_matrix = np.loadtxt(intrinsic_path)
        image_ids = [_id.split(".")[0] for _id in os.listdir(os.path.join(self.path2d, self.scene_name, 'color'))]

        aug_scene_folder = os.path.join(output_path2d, self.scene_name)
        aug_color_path = os.path.join(aug_scene_folder, "color")
        aug_depth_path = os.path.join(aug_scene_folder, "depth")
        aug_pose_path = os.path.join(aug_scene_folder, "pose")

        if not os.path.exists(aug_scene_folder):
            os.makedirs(aug_scene_folder)
        if not os.path.exists(aug_color_path):
            os.makedirs(aug_color_path)
        if not os.path.exists(aug_depth_path):
            os.makedirs(aug_depth_path)
        if not os.path.exists(aug_pose_path):
            os.makedirs(aug_pose_path)

        original_pose_path = os.path.join(self.path2d, self.scene_name, "pose")
        ori_p_files = os.listdir(original_pose_path)
        tgt_p_files = os.listdir(aug_pose_path)
        if not tgt_p_files == ori_p_files:
            for file in ori_p_files:
                shutil.copyfile(os.path.join(original_pose_path, file), os.path.join(aug_pose_path, file))

        depth_path = os.path.join(self.path2d, self.scene_name, 'depth', f'{image_ids[0]}.png')
        depth = np.asarray(o3d.io.read_image(depth_path))
        img_height, img_width = depth.shape

        render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
        render.scene.set_background([1, 1, 1, 0])

        mesh_path = os.path.join(output_path3d, 'mesh.obj')

        for label, mesh in self.model_triangle_meshes.items():
            mesh.translate(self.model_positions[label])
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            mesh.translate(-self.model_positions[label])

            # Workaround to preserve textures
            _object = o3d.io.read_triangle_model(mesh_path)
            render.scene.add_model(f'object_{label}', _object)

        vfov = 2 * np.arctan(img_height / (2 * intrinsic_matrix[1, 1]))

        for image_id in tqdm(image_ids):
            extrinsic_matrix = np.loadtxt(os.path.join(self.path2d, self.scene_name, 'pose', f'{image_id}.txt'))

            img_path = os.path.join(self.path2d, self.scene_name, 'color', f'{image_id}.jpg')
            depth_path = os.path.join(self.path2d, self.scene_name, 'depth', f'{image_id}.png')

            scene_img = np.asarray(o3d.io.read_image(img_path))
            scene_depth = np.asarray(o3d.io.read_image(depth_path))

            eye = extrinsic_matrix[:3, 3]
            lookat = eye + (extrinsic_matrix[:3, :3] @ np.array([0, 0, 1]))
            up = (extrinsic_matrix[:3, :3] @ np.array([0, -1, 0]))
            render.setup_camera(vfov * 180 / np.pi, lookat, eye, up)

            img_model = np.asarray(render.render_to_image())
            depth_model = np.asarray(render.render_to_depth_image(z_in_view_space=True)) * 1000

            aug_img = np.asarray(scene_img).copy()
            aug_depth = np.asarray(scene_depth).copy()
            aug_depth_interp = aug_depth.copy()

            mask = (aug_depth == 0)
            aug_depth_interp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), aug_depth[~mask])
            model_mask = aug_depth_interp > depth_model

            aug_depth[model_mask] = depth_model[model_mask]
            aug_img[model_mask] = img_model[model_mask]

            o3d.io.write_image(os.path.join(aug_color_path, f'{image_id}.jpg'), o3d.geometry.Image(aug_img))
            o3d.io.write_image(os.path.join(aug_depth_path, f'{image_id}.png'), o3d.geometry.Image(aug_depth))

            tqdm.write(f'{image_id} is written', end="\r")
