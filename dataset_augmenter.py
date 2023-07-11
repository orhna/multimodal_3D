import open3d as o3d
import torch
import numpy as np

import objaverse
import multiprocessing

import os
import glob


class DatasetAugmenter:
    def __init__(self, path2d: str, path3d: str):
        self.path2d = path2d
        self.path3d = path3d

        self.model_triangle_mesh = None
        self.model_position = None

        self.scene_name = None
        self.scene_point_cloud = None
        self.scene_labels = None

    def load_model(self, uid, rot_axis_angle=None, height=None):
        processes = multiprocessing.cpu_count()
        objects = objaverse.load_objects(uids=[uid], download_processes=processes)

        self.model_triangle_mesh = o3d.io.read_triangle_mesh(objects[uid], True)

        if rot_axis_angle is not None:
            self.rotate_model(rot_axis_angle)
        if height is not None:
            self.scale_model(height)
        self.__center_model()

        self.model_position = np.array([0., 0., 0.])

    def visualize_model(self):
        o3d.visualization.draw_plotly([self.model_triangle_mesh])

    def rotate_model(self, rot_axis_angle):
        rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis_angle)
        self.model_triangle_mesh.rotate(rot_matrix)
        self.__center_model()

    def scale_model(self, height):
        model_bounding_box = self.model_triangle_mesh.get_axis_aligned_bounding_box()
        scale_factor = height / model_bounding_box.max_bound[2]
        self.model_triangle_mesh.scale(scale_factor, np.zeros(3))
        self.__center_model()

    def __center_model(self):
        model_bounding_box = self.model_triangle_mesh.get_axis_aligned_bounding_box()
        bb_center = model_bounding_box.get_center()
        translation = np.array([-bb_center[0], -bb_center[1], -model_bounding_box.min_bound[2]])

        self.model_triangle_mesh.translate(translation)
        model_bounding_box.translate(translation)

    def load_scene(self, scene_name):
        self.scene_name = scene_name
        scene_files = glob.glob(os.path.join(self.path3d, f'**/{scene_name}*'), recursive=True)
        assert len(scene_files) == 1

        scene_path = scene_files[0]
        scene_array, scene_colors, self.scene_labels = torch.load(scene_path)

        self.scene_point_cloud = o3d.geometry.PointCloud()
        self.scene_point_cloud.points = o3d.utility.Vector3dVector(scene_array)
        self.scene_point_cloud.colors = o3d.utility.Vector3dVector(scene_colors)

    def visualize_scene(self):
        o3d.visualization.draw_plotly([self.scene_point_cloud])

    def place_model_in_scene(self, xy_position):
        model_bounding_box = self.model_triangle_mesh.get_axis_aligned_bounding_box()
        bb_min = model_bounding_box.get_min_bound()
        bb_max = model_bounding_box.get_max_bound()

        scene_points = np.asarray(self.scene_point_cloud.points)
        lower_bound_mask = (xy_position + bb_min[:2] < scene_points[:, :2]).all(axis=1)
        upper_bound_mask = (xy_position + bb_max[:2] > scene_points[:, :2]).all(axis=1)
        bound_mask = lower_bound_mask & upper_bound_mask

        assert bound_mask.sum() > 0

        max_height = scene_points[bound_mask][:, 2].max()
        self.model_position = np.concatenate([xy_position, np.array([max_height])])

    def visualize_placement(self):
        self.model_triangle_mesh.translate(self.model_position)
        o3d.visualization.draw_plotly([self.scene_point_cloud, self.model_triangle_mesh])
        self.model_triangle_mesh.translate(-self.model_position)

    def export(self, output_path, label_model=-1):
        assert os.path.isdir(output_path)

        output_path2d = os.path.join(output_path, '2d')
        output_path3d = os.path.join(output_path, '3d')

        os.mkdir(output_path2d)
        os.mkdir(output_path3d)

        # 3d export
        model_point_cloud = self.model_triangle_mesh.sample_points_uniformly(1000)
        model_point_cloud = model_point_cloud.voxel_down_sample(voxel_size=0.05)

        if np.asarray(model_point_cloud.colors).size == 0:
            model_point_cloud.paint_uniform_color(np.array([0, 0, 0]))

        augmented_scene_array = np.concatenate([np.asarray(self.scene_point_cloud.points),
                                                np.asarray(model_point_cloud.points)])
        augmented_scene_colors = np.concatenate([np.asarray(self.scene_point_cloud.colors),
                                                 np.asarray(model_point_cloud.colors)])
        augmented_scene_labels = np.concatenate([self.scene_labels,
                                                 np.repeat(label_model, len(model_point_cloud.points))])

        torch.save((augmented_scene_array, augmented_scene_colors, augmented_scene_labels),
                   os.path.join(output_path3d, self.scene_name + 'aug.pth'))

        # 2d export


