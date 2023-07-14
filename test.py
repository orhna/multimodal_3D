import numpy as np
from dataset_augmenter import DatasetAugmenter

augmenter = DatasetAugmenter('/mnt/3dcv/projects/openscene_test/data/scannet/scannet_2d',
                             '/mnt/3dcv/projects/openscene_test/data/scannet/scannet_3d')

augmenter.load_model('fed35fa514924ff480a9d7b761f977b0', 20,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=0.4)

augmenter.load_model('ff8cb5a736494a84929199e111bfefd3', 21,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=0.3)

augmenter.load_scene('scene0000_00')
augmenter.place_model_in_scene(np.array([4., 4.]), 20)
augmenter.place_model_in_scene(np.array([2.3, 3.]), 21)

augmenter.export('/home/aleks/3dcv/openseg_aug')
