import numpy as np
from dataset_augmenter import DatasetAugmenter

augmenter = DatasetAugmenter('/mnt/3dcv/projects/openscene_test/data/scannet/scannet_2d',
                             '/mnt/3dcv/projects/openscene_test/data/scannet/scannet_3d')

# STOOL
augmenter.load_model('fa2ec7896f044e09957bcec0b9d77405', 20,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=0.6)
# ARMCHAIR
augmenter.load_model('dba206fede0844c6a9b33d6fc72ce65f', 21,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=0.8)
augmenter.rotate_model(21, np.array([0, 0, np.pi]))
# ROCKING CHAIR
augmenter.load_model('bcd3f4e05ec44902af3b5668b2b831ca', 22,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=0.9)
# EGG CHAIR
augmenter.load_model('4363b93c1ea74caa8768bbcea426bb81', 23,
                     rot_axis_angle=np.array([np.pi/2, 0, 0]),
                     height=1.)
augmenter.rotate_model(23, np.array([0, 0, -np.pi/2]))

augmenter.load_scene('scene0000_00')
augmenter.place_model_in_scene(np.array([4.6, 3.6]), 20)
augmenter.place_model_in_scene(np.array([3.15, 3.15]), 21)
augmenter.place_model_in_scene(np.array([1.6, 3.5]), 22)
augmenter.place_model_in_scene(np.array([6., 4.5]), 23)

augmenter.export('/home/aleks/3dcv/openseg_aug')
